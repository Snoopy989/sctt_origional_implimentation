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
output_dir = './sctt_results_{}_{}'.format(expname, model_name.split('/')[1])

# Find the latest checkpoint
import glob
checkpoint_dirs = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
if checkpoint_dirs:
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getctime)
    model_path = latest_checkpoint
else:
    raise ValueError("No checkpoints found in {}".format(output_dir))

# LOAD DATA
d = pd.read_csv('all_sctt_jrt.csv')
gen = pd.read_csv('sctt_item-generalization_jrt.csv')

from transformers import LlamaForSequenceClassification
from peft import PeftModel

# LOAD MODEL & TOKENIZER
base_model = LlamaForSequenceClassification.from_pretrained(model_name, config=config, token=hftoken, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model = PeftModel.from_pretrained(base_model, model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model = model.to(device)
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
  tokenizer=tokenizer,
)

# VALIDATION
val_prediction = trainer.predict(tokenized_datasets['validation'])
val_output_df = pd.DataFrame({'preds': val_prediction.predictions.flatten(), 'ratings': val_prediction.label_ids})
val_output_df.to_csv('validation_output_inference_{}_{}.csv'.format(expname, model_name.split('/')[1]), index = False)

# TEST
test_prediction = trainer.predict(tokenized_datasets['test'])
test_output_df = pd.DataFrame({'preds': test_prediction.predictions.flatten(), 'ratings': test_prediction.label_ids})
test_output_df.to_csv('test_output_inference_{}_{}.csv'.format(expname, model_name.split('/')[1]), index = False)

# HELDOUT
heldout_prediction = trainer.predict(tokenized_datasets['heldout'])
heldout_output_df = pd.DataFrame({'preds': heldout_prediction.predictions.flatten(), 'ratings': heldout_prediction.label_ids})
heldout_output_df.to_csv('heldout_output_inference_{}_{}.csv'.format(expname, model_name.split('/')[1]), index = False)