import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # set GPU

# bitsandbytes not needed - using full precision
import sys
from pathlib import Path
# Add workspace root to Python path for imports
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
from peft import PeftModel, PeftConfig
from transformers import Trainer

# SETTINGS
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / '.env')  # Load environment variables from .env file
np.random.seed(42) # sets a randomization seed for reproducibility
model_name = os.getenv('MODEL_NAME', 'meta-llama/Llama-2-7b-chat-hf')
base_model_path = Path(os.getenv('BASE_MODEL_PATH', './downloaded_models/meta-llama--Llama-2-7b-chat-hf'))
adapter_path = Path(os.getenv('ADAPTER_PATH', './Llama-2-7b_sctt_regression_oldgood'))

if not base_model_path.is_absolute():
    base_model_path = (project_root / base_model_path).resolve()
if not adapter_path.is_absolute():
    adapter_path = (project_root / adapter_path).resolve()

hftoken = os.getenv('HF_TOKEN')

# Load config from local model if available, otherwise from HF
if os.path.exists(base_model_path):
    config = AutoConfig.from_pretrained(str(base_model_path))
    print(f'Using local base model config: {base_model_path}')
else:
    config = AutoConfig.from_pretrained(model_name, token=hftoken)
    print(f'Using HuggingFace model config: {model_name}')
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
print(f'Adapter path: {adapter_path}')
print(f'Base model path: {base_model_path}')

if not adapter_path.exists():
    raise FileNotFoundError(f'Adapter path not found: {adapter_path}')

# LOAD DATA
d = pd.read_csv(project_root / 'data/raw/all_sctt_jrt.csv')
gen = pd.read_csv(project_root / 'data/raw/sctt_item-generalization_jrt.csv')

# LOAD MODEL & TOKENIZER
print("Loading base model + PEFT adapter for regression...")

# Load base model from local path if available, otherwise from HF
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

# Load PEFT adapter
print(f"Loading adapter from: {adapter_path}")
model = PeftModel.from_pretrained(base_model, str(adapter_path))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model = model.to(device)
print(f"Model loaded successfully. Device: {device}")

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
results_dir = project_root / 'results/training'
results_dir.mkdir(parents=True, exist_ok=True)

print('\nRunning inference on training set...')
train_prediction = trainer.predict(tokenized_datasets['train'])
train_output_df = pd.DataFrame({'preds': train_prediction.predictions.flatten(), 'ratings': train_prediction.label_ids})
train_output_path = str(results_dir / 'train_output_inference_{}_{}.csv'.format(expname, model_name.split('/')[1]))
train_output_df.to_csv(train_output_path, index = False)
print(f'Training metrics: {train_prediction.metrics}')
print(f'Saved to: {train_output_path}')

print('\nRunning inference on validation set...')
val_prediction = trainer.predict(tokenized_datasets['validation'])
val_output_df = pd.DataFrame({'preds': val_prediction.predictions.flatten(), 'ratings': val_prediction.label_ids})
val_output_path = str(results_dir / 'validation_output_inference_{}_{}.csv'.format(expname, model_name.split('/')[1]))
val_output_df.to_csv(val_output_path, index = False)
print(f'Validation metrics: {val_prediction.metrics}')
print(f'Saved to: {val_output_path}')

print('\nRunning inference on test set...')
test_prediction = trainer.predict(tokenized_datasets['test'])
test_output_df = pd.DataFrame({'preds': test_prediction.predictions.flatten(), 'ratings': test_prediction.label_ids})
test_output_path = str(results_dir / 'test_output_inference_{}_{}.csv'.format(expname, model_name.split('/')[1]))
test_output_df.to_csv(test_output_path, index = False)
print(f'Test metrics: {test_prediction.metrics}')
print(f'Saved to: {test_output_path}')

print('\nRunning inference on heldout set...')
heldout_prediction = trainer.predict(tokenized_datasets['heldout'])
heldout_output_df = pd.DataFrame({'preds': heldout_prediction.predictions.flatten(), 'ratings': heldout_prediction.label_ids})
heldout_output_path = str(results_dir / 'heldout_output_inference_{}_{}.csv'.format(expname, model_name.split('/')[1]))
heldout_output_df.to_csv(heldout_output_path, index = False)
print(f'Heldout metrics: {heldout_prediction.metrics}')
print(f'Saved to: {heldout_output_path}')

print('\n' + '='*80)
print('INFERENCE COMPLETE')
print('='*80)

print(f"Preds - min: {train_prediction.predictions.min():.4f}, max: {train_prediction.predictions.max():.4f}, mean: {train_prediction.predictions.mean():.4f}, std: {train_prediction.predictions.std():.4f}")
print(f"Labels - min: {train_prediction.label_ids.min():.4f}, max: {train_prediction.label_ids.max():.4f}, mean: {train_prediction.label_ids.mean():.4f}, std: {train_prediction.label_ids.std():.4f}")