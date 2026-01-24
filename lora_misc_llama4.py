# Workaround for bitsandbytes Windows compatibility issues
class DummyBnbModule:
    def __getattr__(self, name):
        return None

try:
    import bitsandbytes as bnb
    if not hasattr(bnb, 'nn'):
        bnb.nn = DummyBnbModule()
except:
    pass

import evaluate
import numpy as np
import os
import pandas as pd
import sys
import torch
from datasets import Dataset, DatasetDict
from dataprocessing import preprocess_llm_data
from functools import partial
from misc import *
from pynvml import *
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers.trainer_pt_utils import get_parameter_names
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Llama4TextModel, set_seed, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class Llama4ForRegression(torch.nn.Module):
    """Wrapper to add regression head to Llama4TextModel"""
    def __init__(self, base_model, num_labels=1):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.config.num_labels = num_labels
        self.config.problem_type = "regression"
        

        
        # Get hidden size from text model config
        hidden_size = base_model.config.hidden_size
        print(f"Creating regression head with hidden_size={hidden_size}")
        
        # Simple regression head - will be moved to device with the model
        self.regression_head = torch.nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get outputs from base text model
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
        )
        
        # Pool hidden states using mean pooling with attention mask
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        if attention_mask is not None:
            masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
            pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Get regression predictions
        logits = self.regression_head(pooled)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())
        
        # Return proper SequenceClassifierOutput for Trainer compatibility
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Delegate to base model for PEFT compatibility"""
        if hasattr(self.base_model, 'prepare_inputs_for_generation'):
            return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
        return {}
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for VRAM savings"""
        self.base_model.gradient_checkpointing_enable()
    
    def enable_input_require_grads(self):
        """Enable gradients for input embeddings"""
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        self.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    def to(self, *args, **kwargs):
        """Override to() to properly move both base model and regression head"""
        self.base_model = self.base_model.to(*args, **kwargs)
        self.regression_head = self.regression_head.to(*args, **kwargs)
        return super().to(*args, **kwargs)


def load_model(model_name, config, hftoken):
    """Load Llama 4 text model for regression tasks"""
    # Prefer BF16 on A100/Hopper to reduce VRAM
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    from_pretrained_kwargs = {}
    if dtype is not None:
        from_pretrained_kwargs["dtype"] = dtype
    if hftoken:
        from_pretrained_kwargs["token"] = hftoken
    
    # REMOVED: device_map="auto" causes meta tensor issues
    # Instead use low_cpu_mem_usage for memory efficiency
    from_pretrained_kwargs["low_cpu_mem_usage"] = True

    print(f"Loading {model_name} with dtype={dtype}...")
    print("This may take 10-20 minutes for 17B model...")
    
    # Load base Llama4 TEXT model (not the multimodal one)
    base_model = Llama4TextModel.from_pretrained(
        model_name,
        **from_pretrained_kwargs,
    )
    
    # Move to GPU explicitly after loading
    if torch.cuda.is_available():
        base_model = base_model.to("cuda:0")
    
    print(f"✓ Base model loaded on device: {base_model.device}")
    print(f"Model memory footprint: {base_model.get_memory_footprint() / 1e9:.2f} GB")
    
    # Wrap with regression head
    model = Llama4ForRegression(base_model, num_labels=config.num_labels)
    
    # Move regression head to same device as base model
    if torch.cuda.is_available():
        model.regression_head = model.regression_head.to("cuda:0")
    
    print("="*50)
    print(f"✓ Model loaded on device: {next(model.parameters()).device}")
    print(f"Regression head device: {next(model.regression_head.parameters()).device}")
    print("="*50)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hftoken)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def get_max_length(model):
    """Get maximum sequence length from model config"""
    # Access base model config
    conf = model.base_model.config if hasattr(model, 'base_model') else model.config
    max_length = None
    
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(conf, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    
    return max_length


def find_all_linear_names(model):
    """Find all linear layer names for LoRA"""
    # Access the base model
    base = model.base_model if hasattr(model, 'base_model') else model
    
    cls = torch.nn.Linear
    lora_module_names = set()
    
    for name, module in base.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # Remove output heads and router (router returns tuple, incompatible with LoRA)
    exclude_names = {'lm_head', 'regression_head', 'router'}
    lora_module_names = lora_module_names - exclude_names
    
    print(f"LoRA target modules: {lora_module_names}")
    return list(lora_module_names)


def create_peft_config(r, lora_alpha, lora_dropout, modules):
    """Create LoRA config for regression task"""
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=modules,
        bias="none",
        task_type="CAUSAL_LM",  # Use CAUSAL_LM for Llama4, not SEQ_CLS
    )
    return config


def compute_metrics(eval_preds):
    """Compute MSE for regression"""
    predictions, references = eval_preds
    predictions = predictions.flatten()
    references = references.flatten()
    
    mse_metric = evaluate.load("mse")
    loss = mse_metric.compute(predictions=predictions, references=references)
    
    return loss
