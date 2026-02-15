import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # set GPU

# bitsandbytes not needed - using full precision
import sys

from dotenv import load_dotenv
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from dataprocessing import preprocess_llm_data
from lora_misc import load_model, get_max_length, compute_metrics
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoConfig
from peft import AutoPeftModelForSequenceClassification
from transformers import Trainer

# SETTINGS
load_dotenv()  # Load environment variables from .env file
np.random.seed(42) # sets a randomization seed for reproducibility
model_name = os.getenv('MODEL_NAME', 'meta-llama/Llama-2-7b-chat-hf')
hftoken = os.getenv('HF_TOKEN')
config = AutoConfig.from_pretrained(model_name, token=hftoken)
config.num_labels = 1
config.problem_type = "regression"
epochs = 10
val_pct = 0.10 # proportion of total dataset allocated to validation
test_pct = 0.20  # proportion of the dataset to devote to held-out test set
val_train_pct = (1.0/(1.0-test_pct))*val_pct # we have to get the val set from training subset, so pct needs to be modified
max_length_divisor = 2
prefix = "A creative "
connector1 = " for "
connector2 = " is " # we'll use prefix/conn to construct inputs to the model
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")  # setting for whether to use gpu or cpu
print('Using device:', device)
scaler = MinMaxScaler()
expname = "LORA_{}_epochs".format(epochs)
# Use trained model adapter
model_path = './models/Llama-2-7b_sctt_regression'
print(f'Loading model from: {model_path}')

# LOAD DATA
d = pd.read_csv('data/raw/all_sctt_jrt.csv')
gen = pd.read_csv('data/raw/sctt_item-generalization_jrt.csv')

from transformers import LlamaForSequenceClassification
from peft import PeftModel


# LOAD MODEL & TOKENIZER
base_model = LlamaForSequenceClassification.from_pretrained(model_name, config=config, token=hftoken, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model = PeftModel.from_pretrained(base_model, model_path)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hftoken)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model = model.to(device)
model = model.merge_and_unload()  # <-- here, collapses LoRA + wrappers into base weights

# ---- DIAGNOSTIC: check if score head loaded correctly ----
# print("\n=== SCORE HEAD DIAGNOSTIC ===")
# print(model.base_model.model.score)
# print("---")
# print("Score weights sample:", model.base_model.model.score.modules_to_save.default.weight.data.flatten()[:10])
# # Also run the safetensors check:
# from safetensors import safe_open
# f = safe_open('./Llama-2-7b_sctt_regression/adapter_model.safetensors', framework='pt')
# score_keys = [k for k in f.keys() if 'score' in k]
# print("Score keys in saved adapter:", score_keys)
# print("=== END DIAGNOSTIC ===\n")
# -----------------------------------------------------------

max_length = int(get_max_length(model)/max_length_divisor)

# PREPROCESS DATA
def tokenize_function(examples): #  define wrapper tokenizer function (for batch training)
  # Filter out None values and convert to strings
  texts = [str(text) if text is not None else "" for text in examples['text']]
  return tokenizer(texts, padding = 'max_length', truncation = True, max_length = 512)
d = preprocess_llm_data(d, gen, val_pct, val_train_pct, test_pct, prefix, connector1, connector2)
# Filter out rows with None text values before tokenizing
d = d.filter(lambda example: example['text'] is not None and example['text'] != '')
tokenized_datasets = d.map(tokenize_function, batched = True) # applies wrapper to our dataset

# INFERENCE
trainer = Trainer(
  model=model,
  compute_metrics=compute_metrics,
)

# Create output directory
os.makedirs('results/training', exist_ok=True)

print('\nRunning inference on training set...')
train_prediction = trainer.predict(tokenized_datasets['train'])
train_output_df = pd.DataFrame({'preds': train_prediction.predictions.flatten(), 'ratings': train_prediction.label_ids})
train_output_path = 'results/training/train_output_inference_{}_{}.csv'.format(expname, model_name.split('/')[1])
train_output_df.to_csv(train_output_path, index = False)
print(f'Training metrics: {train_prediction.metrics}')
print(f'Saved to: {train_output_path}')

# print('\nRunning inference on validation set...')
# val_prediction = trainer.predict(tokenized_datasets['validation'])
# val_output_df = pd.DataFrame({'preds': val_prediction.predictions.flatten(), 'ratings': val_prediction.label_ids})
# val_output_path = 'results/training/validation_output_inference_{}_{}.csv'.format(expname, model_name.split('/')[1])
# val_output_df.to_csv(val_output_path, index = False)
# print(f'Validation metrics: {val_prediction.metrics}')
# print(f'Saved to: {val_output_path}')

# print('\nRunning inference on test set...')
# test_prediction = trainer.predict(tokenized_datasets['test'])
# test_output_df = pd.DataFrame({'preds': test_prediction.predictions.flatten(), 'ratings': test_prediction.label_ids})
# test_output_path = 'results/training/test_output_inference_{}_{}.csv'.format(expname, model_name.split('/')[1])
# test_output_df.to_csv(test_output_path, index = False)
# print(f'Test metrics: {test_prediction.metrics}')
# print(f'Saved to: {test_output_path}')

# print('\nRunning inference on heldout set...')
# heldout_prediction = trainer.predict(tokenized_datasets['heldout'])
# heldout_output_df = pd.DataFrame({'preds': heldout_prediction.predictions.flatten(), 'ratings': heldout_prediction.label_ids})
# heldout_output_path = 'results/training/heldout_output_inference_{}_{}.csv'.format(expname, model_name.split('/')[1])
# heldout_output_df.to_csv(heldout_output_path, index = False)
# print(f'Heldout metrics: {heldout_prediction.metrics}')
# print(f'Saved to: {heldout_output_path}')

print('\n' + '='*80)
print('INFERENCE COMPLETE')
print('='*80)

print(f"Preds - min: {train_prediction.predictions.min():.4f}, max: {train_prediction.predictions.max():.4f}, mean: {train_prediction.predictions.mean():.4f}, std: {train_prediction.predictions.std():.4f}")
print(f"Labels - min: {train_prediction.label_ids.min():.4f}, max: {train_prediction.label_ids.max():.4f}, mean: {train_prediction.label_ids.mean():.4f}, std: {train_prediction.label_ids.std():.4f}")