import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # set GPU import quant import torch import torch.nn as nn

# bitsandbytes not needed - using full precision training
import sys

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
from lora_misc import *
from pynvml import *
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, LlamaForSequenceClassification, TrainingArguments, Trainer, LlamaConfig
from transformers.trainer_pt_utils import get_parameter_names
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig,DataCollatorForLanguageModeling, Trainer, TrainingArguments

#  SETTINGS
load_dotenv()  # Load environment variables from .env file
np.random.seed(42) # sets a randomization seed for reproducibility
model_name = os.getenv('MODEL_NAME', 'meta-llama/Llama-2-7b-hf')
hftoken = os.getenv('HF_TOKEN')
config = LlamaConfig(model_name, problem_type = "regression")
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
scaler = MinMaxScaler()
expname = "LORA_{}_epochs".format(epochs)
trainer_args = TrainingArguments(
  eval_strategy = "steps",
  eval_steps = 1000,
  save_strategy="steps",
  save_steps = 1000,
  learning_rate = 5e-5,
  per_device_train_batch_size=1,  # Reduced to avoid OOM
  per_device_eval_batch_size=1,
  gradient_accumulation_steps=4,  # Effective batch size = 4
  warmup_steps = 1000,
  fp16 = True,
  num_train_epochs = epochs,
  load_best_model_at_end = True,
  output_dir='./sctt_results_{}_{}'.format(expname,model_name.split('/')[1]),
  logging_steps=10,  # Log more frequently
  dataloader_num_workers=0,  # Windows compatibility
)
r = 4
lora_alpha = 32
lora_dropout = 0.1

#  LOAD & PREPARE DATA
d = pd.read_csv('all_sctt_jrt.csv')
gen = pd.read_csv('sctt_item-generalization_jrt.csv')

#  INITIALIZE MODEL & TOKENIZER (PRE-TRAINED)
model, tokenizer = load_model(model_name, config, hftoken)
# Explicitly move model to GPU
model = model.to(device)
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
