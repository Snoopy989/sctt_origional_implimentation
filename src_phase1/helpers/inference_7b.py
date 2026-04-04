"""
Inference Script — Llama-2-7b-chat-hf (SCTT Regression)
=========================================================
Runs regression inference using the fine-tuned 7b LoRA adapter on all data
splits (train, validation, test, heldout) and writes predictions to CSV.

Architecture:
    Base model : AutoModelForSequenceClassification (Llama-2-7b-chat-hf)
    Adapter    : SEQ_CLS LoRA — includes trained score.weight (regression head)

Configuration (set in .env at project root):
    MODEL_NAME      = meta-llama/Llama-2-7b-chat-hf
    BASE_MODEL_PATH = ./downloaded_models/meta-llama--Llama-2-7b-chat-hf
    ADAPTER_PATH    = ./downloaded_adapters/PhillipGre--Llama-2-7b_sctt_regression
    ADAPTER_NAME    = Llama-2-7b_sctt_regression
    EPOCHS          = 10
    HF_TOKEN        = <your token>

Output:
    results/training/<ADAPTER_NAME>/<split>_output_inference_LORA_<n>_epochs_<name>.csv

Usage (from project root, external terminal recommended):
    python src_phase1/helpers/inference_7b.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dotenv import load_dotenv
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from src_phase1.helpers.dataprocessing import preprocess_llm_data
from src_phase1.helpers.lora_misc import load_model, get_max_length, compute_metrics
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig, TaskType
from transformers import Trainer

# SETTINGS
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / '.env')
np.random.seed(42)
model_name = os.getenv('MODEL_NAME', 'meta-llama/Llama-2-7b-chat-hf')
base_model_path = Path(os.getenv('BASE_MODEL_PATH', './downloaded_models/meta-llama--Llama-2-7b-chat-hf'))
adapter_path_raw = os.getenv('ADAPTER_PATH', './downloaded_adapters/PhillipGre--Llama-2-7b_sctt_regression')
adapter_path = Path(adapter_path_raw)
adapter_name = os.getenv('ADAPTER_NAME', adapter_path_raw.split('/')[-1])

if not base_model_path.is_absolute():
    base_model_path = (project_root / base_model_path).resolve()
if not adapter_path.is_absolute():
    _local = (project_root / adapter_path).resolve()
    adapter_path_for_peft = str(_local) if _local.exists() else adapter_path_raw
else:
    adapter_path_for_peft = str(adapter_path)

hftoken = os.getenv('HF_TOKEN')

# LOAD CONFIG
if os.path.exists(base_model_path):
    config = AutoConfig.from_pretrained(str(base_model_path))
    print(f'Using local base model config: {base_model_path}')
else:
    config = AutoConfig.from_pretrained(model_name, token=hftoken)
    print(f'Using HuggingFace model config: {model_name}')
config.num_labels = 1
config.problem_type = "regression"

epochs = int(os.getenv('EPOCHS', '10'))
val_pct = 0.10
test_pct = 0.20
val_train_pct = (1.0 / (1.0 - test_pct)) * val_pct
max_length_divisor = 2
prefix = "A creative "
connector1 = " for "
connector2 = " is "
if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA-capable GPU detected. Inference on a 7b model requires a GPU. "
        "Check that your drivers are installed and CUDA_VISIBLE_DEVICES is set correctly."
    )
device = torch.device("cuda:0")
gpu_name = torch.cuda.get_device_name(0)
gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
scaler = MinMaxScaler()
expname = "LORA_{}_epochs".format(epochs)

print(f'Using device:  {device} ({gpu_name}, {gpu_mem_gb:.1f} GB)')
print(f'Adapter:       {adapter_path_for_peft}')
print(f'Adapter name:  {adapter_name}')
print(f'Base model:    {base_model_path}')

# LOAD DATA
d = pd.read_csv(project_root / 'data/raw/all_sctt_jrt.csv')
gen = pd.read_csv(project_root / 'data/raw/sctt_item-generalization_jrt.csv')

# LOAD MODEL AND TOKENIZER
print("Loading base model + PEFT adapter for regression...")
if os.path.exists(base_model_path):
    print(f"Loading base model from local path: {base_model_path}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        str(base_model_path),
        config=config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
else:
    print(f"Loading base model from HuggingFace: {model_name}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=hftoken,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hftoken)

# Load PEFT adapter — 7b adapter uses SEQ_CLS with score.weight saved, override is safe
print(f"Loading adapter from: {adapter_path_for_peft}")
peft_config = PeftConfig.from_pretrained(adapter_path_for_peft, token=hftoken)
print(f"Adapter task_type in config: {peft_config.task_type}  (overriding to SEQ_CLS)")
peft_config.task_type = TaskType.SEQ_CLS
model = PeftModel.from_pretrained(base_model, adapter_path_for_peft, config=peft_config, token=hftoken)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model = model.to(device)
print(f"Model loaded successfully. Device: {device}")

max_length = int(get_max_length(model) / max_length_divisor)

# PREPROCESS DATA
def tokenize_function(examples):
    texts = [str(text) if text is not None else "" for text in examples['text']]
    return tokenizer(texts, padding='max_length', truncation=True, max_length=512)

d = preprocess_llm_data(d, gen, val_pct, val_train_pct, test_pct, prefix, connector1, connector2)
d = d.filter(lambda example: example['text'] is not None and example['text'] != '')
tokenized_datasets = d.map(tokenize_function, batched=True)

# INFERENCE
trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)

results_dir = project_root / 'results' / 'training' / adapter_name
results_dir.mkdir(parents=True, exist_ok=True)

_processed = {
    'train':      project_root / 'data/processed/training_items_sctt.csv',
    'validation': project_root / 'data/processed/validation_items_sctt.csv',
    'test':       project_root / 'data/processed/test_items_sctt.csv',
    'heldout':    project_root / 'data/processed/heldoutprompt_items_sctt.csv',
}

def _save_split(split_name, prediction):
    meta_df = pd.read_csv(_processed[split_name], index_col=0)
    meta_df = meta_df[meta_df['text'].notna() & (meta_df['text'].str.strip() != '')].reset_index(drop=True)
    assert len(meta_df) == len(prediction.predictions), (
        f"{split_name}: metadata rows ({len(meta_df)}) != predictions ({len(prediction.predictions)})"
    )
    out_df = meta_df.rename(columns={'label': 'ratings'})
    out_df['preds'] = prediction.predictions.flatten()
    out_path = str(results_dir / '{}_output_inference_{}_{}.csv'.format(
        split_name, expname, adapter_name))
    out_df.to_csv(out_path, index=False)
    print(f'  Metrics: {prediction.metrics}')
    print(f'  Saved {len(out_df)} samples to {out_path}')
    return out_path

print('\nRunning inference on training set...')
train_prediction = trainer.predict(tokenized_datasets['train'])
train_output_path = _save_split('train', train_prediction)

# print('\nRunning inference on validation set...')
# val_prediction = trainer.predict(tokenized_datasets['validation'])
# val_output_path = _save_split('validation', val_prediction)

# print('\nRunning inference on test set...')
# test_prediction = trainer.predict(tokenized_datasets['test'])
# test_output_path = _save_split('test', test_prediction)

# print('\nRunning inference on heldout set...')
# heldout_prediction = trainer.predict(tokenized_datasets['heldout'])
# heldout_output_path = _save_split('heldout', heldout_prediction)

print('\n' + '=' * 80)
print('INFERENCE COMPLETE')
print('=' * 80)
