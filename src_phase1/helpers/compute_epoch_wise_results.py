import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # set GPU import quant import torch import torch.nn as nn
from dotenv import load_dotenv
import evaluate
import numpy as np
import pandas as pd
import sys
# Add workspace root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
torch.cuda.empty_cache()
from datasets import Dataset, DatasetDict
from src_phase1.helpers.dataprocessing import preprocess_llm_data
from functools import partial
from lora_misc_llama2_13b import *

# Load environment variables
load_dotenv()
from pynvml import *
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaForSequenceClassification, TrainingArguments, Trainer, LlamaConfig
from transformers.trainer_pt_utils import get_parameter_names
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, DataCollatorForLanguageModeling

#  SETTINGS
np.random.seed(42) # sets a randomization seed for reproducibility
model_name = 'meta-llama/Llama-2-13b-chat-hf'
hftoken = os.getenv('HF_TOKEN')
checkpoints_dirs = ['sctt_results_LORA_10_epochs_Llama-2-13b-chat-hf']
config = LlamaConfig.from_pretrained(model_name, token=hftoken)
config.num_labels = 1
config.problem_type = "regression"
# model_names = ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-7b-chat-hf']
epochs = 10
val_pct = 0.10 # proportion of total dataset allocated to validation
test_pct = 0.20  # proportion of the dataset to devote to held-out test set
val_train_pct = (1.0/(1.0-test_pct))*val_pct # we have to get the val set from training subset, so pct needs to be modified
prefix = "A creative "
connector1 = " for "
connector2 = " is " # we'll use prefix/conn to construct inputs to the model
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")  # setting for whether to use gpu or cpu
scaler = MinMaxScaler()
expname = "LORA_{}_epochs".format(epochs)
test_args = TrainingArguments(
  do_train = False,
  do_predict = True,
  per_device_eval_batch_size=4,
  bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
  fp16 = False,
  output_dir='./sctt_results_{}_{}'.format(expname,model_name.split('/')[1]),
)

#  LOAD & PREPARE DATA
d = pd.read_csv('all_sctt_jrt.csv')
gen = pd.read_csv('sctt_item-generalization_jrt.csv')

#  PREPROCESS DATA
d = preprocess_llm_data(d, gen, val_pct, val_train_pct, test_pct, prefix, connector1, connector2)

# STORAGE LIST
train_datadict = []
val_datadict = []

# LOOP THRU MODELS (foundation/chat)
for model_type in checkpoints_dirs:
  # Get only checkpoint directories and sort them numerically
  all_items = os.listdir(model_type)
  checkpoints = sorted([c for c in all_items if c.startswith('checkpoint-')], 
                       key=lambda x: int(x.split('-')[1]))
  
  print(f"\nFound {len(checkpoints)} checkpoints: {checkpoints}\n")
  
  # LOOP THRU CHECKPOINTS WTIHIN MODEL TYPE
  for ind, checkpoint in enumerate(checkpoints):
    print(f"Processing {checkpoint}...")
    peft_model_id = '{}/{}'.format(model_type, checkpoint)
    
    # Extract step number from checkpoint name
    steps = int(checkpoint.split('-')[1])
    
    config = PeftConfig.from_pretrained(peft_model_id)
    
    # Load base model with proper configuration
    base_config = LlamaConfig.from_pretrained(config.base_model_name_or_path, token=hftoken)
    base_config.num_labels = 1
    base_config.problem_type = "regression"
    
    inference_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        config=base_config,
        token=hftoken
    )
    
    # Add missing method for PEFT compatibility
    if not hasattr(inference_model, 'prepare_inputs_for_generation'):
        inference_model.prepare_inputs_for_generation = lambda *args, **kwargs: {}
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, token=hftoken)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set pad_token_id in model config
    inference_model.config.pad_token_id = tokenizer.pad_token_id
    
    def tokenize_function(examples): #  define wrapper tokenizer function (for batch training)
      return tokenizer(examples['text'], padding = 'max_length', truncation = True, max_length = 512)
    tokenized_datasets = d.map(tokenize_function, batched = True) # applies wrapper to our dataset
    
    model = PeftModel.from_pretrained(inference_model, peft_model_id)

    trainer = Trainer(
      model=model,
      args=test_args,
      compute_metrics=compute_metrics,
    )
    
    print(f"  Checkpoint steps: {steps}")
    
    # EVALUATE ON BOTH TRAIN AND VALIDATION
    for split_name in ['train', 'validation']:
      eval_dataset = tokenized_datasets[split_name]
      print(f"  Evaluating on {split_name} set...")
      
      prediction = trainer.predict(eval_dataset)
      predictions_clean = prediction.predictions.flatten()
      labels_clean = prediction.label_ids.flatten()
      
      # Quick diagnostic
      print(f"    Predictions range: [{predictions_clean.min():.3f}, {predictions_clean.max():.3f}]")
      print(f"    Labels range: [{labels_clean.min():.3f}, {labels_clean.max():.3f}]")
      
      num_preds = len(predictions_clean)
      target_dict = train_datadict if split_name == 'train' else val_datadict
      
      for i in range(num_preds):
        target_dict.append({
          'peft_model_id': peft_model_id, 
          'steps': steps, 
          'predictions': predictions_clean[i], 
          'ratings': labels_clean[i]
        })
    
    del model
    del tokenizer
    del trainer
    del tokenized_datasets
    torch.cuda.empty_cache()
    print(f"  Completed {checkpoint}\n")

print(f"\nProcessed {len(train_datadict)} train predictions and {len(val_datadict)} validation predictions across {len(checkpoints)} checkpoints")

# Save train results
train_df = pd.DataFrame.from_dict(train_datadict)
train_df.to_csv('epoch_wise_LORA_results_train.csv', index=False)
print("Train results saved to: epoch_wise_LORA_results_train.csv")

# Save validation results
val_df = pd.DataFrame.from_dict(val_datadict)
val_df.to_csv('epoch_wise_LORA_results_validation.csv', index=False)
print("Validation results saved to: epoch_wise_LORA_results_validation.csv\n")
