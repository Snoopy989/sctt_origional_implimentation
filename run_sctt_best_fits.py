import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # set GPU import quant import torch import torch.nn as nn
import bitsandbytes as bnb
import evaluate
import numpy as np
import pandas as pd
import sys
import torch
torch.cuda.empty_cache()
from datasets import Dataset, load_metric, DatasetDict
from dataprocessing import preprocess_llm_data
from functools import partial
from lora_misc import *
from pynvml import *
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, LlamaForSequenceClassification, TrainingArguments, Trainer, LlamaConfig
from transformers.trainer_pt_utils import get_parameter_names
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig,DataCollatorForLanguageModeling, Trainer, TrainingArguments

#  SETTINGS
np.random.seed(42) # sets a randomization seed for reproducibility
model_names = ['meta-llama/Llama-2-7b-hf','meta-llama/Llama-2-7b-chat-hf']
checkpoints = ['sctt_results_LORA_10_epochs_Llama-2-7b-hf/checkpoint-29000',
                    'sctt_results_LORA_10_epochs_Llama-2-7b-chat-hf/checkpoint-20000']
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
expname = "RUN4_LORA_{}_epochs".format(epochs)

#  LOAD & PREPARE DATA
d = pd.read_csv('all_sctt_jrt.csv')
gen = pd.read_csv('sctt_item-generalization_jrt.csv')

#  PREPROCESS DATA
d = preprocess_llm_data(d, gen, val_pct, val_train_pct, test_pct, prefix, connector1, connector2)

# STORAGE LIST
datadict = []

# LOOP THRU MODELS (foundation/chat)
for ind, peft_model_id in enumerate(checkpoints):
  model_name = model_names[ind]
  config = PeftConfig.from_pretrained(peft_model_id)
  inference_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels = 1)
  tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
  tokenizer.pad_token = tokenizer.eos_token
  def tokenize_function(examples): #  define wrapper tokenizer function (for batch training)
    return tokenizer(examples['text'], padding = 'max_length', truncation = True)
  tokenized_datasets = d.map(tokenize_function, batched = True) # applies wrapper to our dataset

  test_args = TrainingArguments(
    do_train = False,
    do_predict = True,
    per_device_eval_batch_size=4,
    fp16 = True,
    output_dir='./sctt_results_{}_{}'.format(expname,model_name.split('/')[1]),
  )

  model = PeftModel.from_pretrained(inference_model, peft_model_id)

  trainer = Trainer(
    model=model,
    args=test_args,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
  )

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

  ### CLEAR MEMORY
  del model
  del tokenizer
  del trainer
  del tokenized_datasets
  torch.cuda.empty_cache()
