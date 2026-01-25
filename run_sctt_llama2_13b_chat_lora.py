import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn

# Workaround for bitsandbytes Windows compatibility issues
import sys
class DummyBnbModule:
    def __getattr__(self, name):
        return None
        
try:
    import bitsandbytes as bnb
    if not hasattr(bnb, 'nn'):
        bnb.nn = DummyBnbModule()
except:
    pass

from dotenv import load_dotenv
import evaluate
import numpy as np
import pandas as pd
import sys
import torch
torch.cuda.empty_cache()
from datasets import Dataset, DatasetDict
from dataprocessing import preprocess_llm_data
from functools import partial
from lora_misc_llama2_13b import *
from pynvml import *
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from transformers.trainer_pt_utils import get_parameter_names
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, DataCollatorForLanguageModeling

#  SETTINGS
torch.cuda.empty_cache() # Clear GPU memory
load_dotenv()  # Load environment variables from .env file
np.random.seed(42) # sets a randomization seed for reproducibility
model_name = 'meta-llama/Llama-2-13b-chat-hf'
hftoken = os.getenv('HF_TOKEN')
config = AutoConfig.from_pretrained(model_name, token=hftoken)
config.num_labels = 1
config.problem_type = "regression"
# model_names = ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-7b-chat-hf']
epochs = 10
val_pct = 0.10 # proportion of total dataset allocated to validation
test_pct = 0.20  # proportion of the dataset to devote to held-out test set
val_train_pct = (1.0/(1.0-test_pct))*val_pct # we have to get the val set from training subset, so pct needs to be modified
max_length_divisor = 2
prefix = "A creative "
connector1 = " for "
connector2 = " is " # we'll use prefix/conn to construct inputs to the model
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")  # setting for whether to use gpu or cpu


print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA not available. Check drivers and PyTorch version.")



scaler = MinMaxScaler()
expname = "LORA_{}_epochs".format(epochs)
trainer_args = TrainingArguments(
  evaluation_strategy = "steps",
  eval_steps = 1000,
  save_strategy="steps",
  save_steps = 1000,
  learning_rate = 5e-5,
  per_device_train_batch_size=4,  # Reduced to avoid OOM
  per_device_eval_batch_size=4,
  gradient_accumulation_steps=1,  # Effective batch size = 4
  warmup_steps = 1000,
  bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
  fp16 = not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
  num_train_epochs = epochs,
  load_best_model_at_end = True,
  output_dir='./sctt_results_{}_{}'.format(expname,model_name.split('/')[1]),
  logging_steps=10,  # Log more frequently
  dataloader_num_workers=0,  # Windows compatibility
  ddp_find_unused_parameters=False,  # Important for models with device_map
)
r = 4
lora_alpha = 32
lora_dropout = 0.1

#  LOAD & PREPARE DATA
d = pd.read_csv('all_sctt_jrt.csv')
gen = pd.read_csv('sctt_item-generalization_jrt.csv')

#  INITIALIZE MODEL & TOKENIZER (PRE-TRAINED)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, use_auth_token=hftoken)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hftoken)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Add missing method for PEFT compatibility
if not hasattr(model, 'prepare_inputs_for_generation'):
    model.prepare_inputs_for_generation = lambda *args, **kwargs: {}

# Save VRAM during training
model.gradient_checkpointing_enable()
if getattr(model, "config", None) is not None:
  model.config.use_cache = False
max_length = int(get_max_length(model)/max_length_divisor)

#  GENERATE PEFT CONFIG & PEFT MODEL
modules = find_all_linear_names(model)
peftconfig = create_peft_config(r, lora_alpha, lora_dropout, modules)
model = get_peft_model(model, peftconfig)
print(model.print_trainable_parameters())

#  PREPROCESS DATA
def tokenize_function(examples): #  define wrapper tokenizer function (for batch training)
  # Filter out None values and convert to strings
  texts = [str(text) if text is not None else "" for text in examples['text']]
  return tokenizer(texts, padding = 'max_length', truncation = True, max_length = 512)
d = preprocess_llm_data(d, gen, val_pct, val_train_pct, test_pct, prefix, connector1, connector2)
# Filter out rows with None text values before tokenizing
d = d.filter(lambda example: example['text'] is not None and example['text'] != '')
tokenized_datasets = d.map(tokenize_function, batched = True) # applies wrapper to our dataset

#  TRAIN MODEL
# Mark model as already on device to prevent Trainer from trying to move it
model.is_parallelizable = True
model.model_parallel = True

trainer = Trainer(
  model=model,
  args=trainer_args,
  train_dataset=tokenized_datasets["train"],
  eval_dataset=tokenized_datasets["validation"],
  compute_metrics=compute_metrics,
  tokenizer=tokenizer,
)

# Check for existing checkpoints and resume if available
import glob
checkpoint_dirs = glob.glob('./sctt_results_{}_{}/checkpoint-*'.format(expname,model_name.split('/')[1]))
if checkpoint_dirs:
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getctime)
    print(f"\n✓ Resuming training from checkpoint: {latest_checkpoint}\n")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("\n✓ Starting training from scratch\n")
    trainer.train()

# VALIDATION
val_prediction = trainer.predict(tokenized_datasets['validation'])
val_output_df = pd.DataFrame({'preds': val_prediction.predictions.flatten(), 'ratings': val_prediction.label_ids})
val_output_df.to_csv('validation_output_sctt_results_{}_{}.csv'.format(expname,model_name.split('/')[1]), index = False)
# print('\n\n\n\n\n\nVALIDATION MSE:', val_prediction.metrics['eval_mse'])
# print('VALIDATION CORR:', val_prediction.metrics['eval_corr'])

# TEST
test_prediction = trainer.predict(tokenized_datasets['test'])
test_output_df = pd.DataFrame({'preds': test_prediction.predictions.flatten(), 'ratings': test_prediction.label_ids})
test_output_df.to_csv('test_output_sctt_results_{}_{}.csv'.format(expname,model_name.split('/')[1], index = False))
# print('\n\n\n\n\n\nTEST MSE:', test_prediction.metrics['test_mse'])
# print('TEST CORR:', test_prediction.metrics['test_corr'])

#  HELDOUT TEST
heldout_prediction = trainer.predict(tokenized_datasets['heldout'])
heldoutprompt_output_df = pd.DataFrame({'preds': heldout_prediction.predictions.flatten(), 'ratings': heldout_prediction.label_ids})
heldoutprompt_output_df.to_csv('heldoutprompt_output_sctt_results_{}_{}.csv'.format(expname,model_name.split('/')[1], index = False))
# print('\n\n\n\n\n\nHOLDOUT MSE:', heldout_prediction.metrics['heldout_prompt_mse'])
# print('HOLDOUT CORR:', heldout_prediction.metrics['heldout_prompt_corr'])
