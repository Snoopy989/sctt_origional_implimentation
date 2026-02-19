import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
# Add workspace root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
torch.cuda.empty_cache()
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import AutoPeftModelForSequenceClassification
from datasets import Dataset
from src_phase1.helpers.dataprocessing import preprocess_llm_data
from lora_misc_llama2_13b import compute_metrics

# Settings
model_name = 'meta-llama/Llama-2-13b-chat-hf'
checkpoint_path = './sctt_results_LORA_10_epochs_Llama-2-13b-chat-hf/checkpoint-4000'
epochs = 10
val_pct = 0.10
test_pct = 0.20
val_train_pct = (1.0/(1.0-test_pct))*val_pct
prefix = "A creative "
connector1 = " for "
connector2 = " is "

# Load data
print("Loading data...")
d = pd.read_csv('all_sctt_jrt.csv')
gen = pd.read_csv('sctt_item-generalization_jrt.csv')
d = preprocess_llm_data(d, gen, val_pct, val_train_pct, test_pct, prefix, connector1, connector2)

# Load model using AutoPeftModelForSequenceClassification  
print(f"\nLoading model from {checkpoint_path} using AutoPeftModelForSequenceClassification...")
model = AutoPeftModelForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = d.map(tokenize_function, batched=True)

# Create trainer
test_args = TrainingArguments(
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=4,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    output_dir='./debug_output',
)

trainer = Trainer(
    model=model,
    args=test_args,
    compute_metrics=compute_metrics,
)

# Predict on validation set
print("\nPredicting on validation set...")
prediction = trainer.predict(tokenized_datasets['validation'])
predictions = prediction.predictions.flatten()
labels = prediction.label_ids.flatten()

# Compute correlation
r, p = pearsonr(predictions, labels)
print(f"\nResults:")
print(f"  Pearson r: {r:.4f}")
print(f"  p-value: {p:.4e}")
print(f"  Samples: {len(predictions)}")
print(f"  Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
print(f"  Labels range: [{labels.min():.4f}, {labels.max():.4f}]")
print(f"  First 5 predictions: {predictions[:5]}")

# Compare with validation_output file
val_out = pd.read_csv('validation_output_sctt_results_LORA_10_epochs_Llama-2-13b-chat-hf.csv')
r_val_out, _ = pearsonr(val_out['preds'], val_out['ratings'])
print(f"\nValidation_output file:")
print(f"  Pearson r: {r_val_out:.4f}")
print(f"  Predictions range: [{val_out['preds'].min():.4f}, {val_out['preds'].max():.4f}]")
print(f"  First 5 predictions: {val_out['preds'].head().values}")

# Check if predictions match
max_diff = np.abs(predictions - val_out['preds'].values).max()
print(f"\nMax difference between predictions: {max_diff:.6f}")
if max_diff < 0.001:
    print("✓ Predictions MATCH validation_output file!")
else:
    print("✗ Predictions DO NOT match validation_output file")
