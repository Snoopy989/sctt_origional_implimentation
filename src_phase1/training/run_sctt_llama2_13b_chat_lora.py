import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn

# bitsandbytes not needed - using full precision training
import sys
# Add workspace root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dotenv import load_dotenv
import evaluate
import logging
import numpy as np
import pandas as pd
import sys
import torch
torch.cuda.empty_cache()
from datasets import Dataset, DatasetDict
from src_phase1.helpers.dataprocessing import preprocess_llm_data
from functools import partial
from lora_misc_llama2_13b import *
from pynvml import *
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from transformers.trainer_pt_utils import get_parameter_names
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, DataCollatorForLanguageModeling

#  SETTINGS
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("CUDA not available. Check drivers and PyTorch version.")



scaler = MinMaxScaler()
expname = "LORA_{}_epochs".format(epochs)
trainer_args = TrainingArguments(
  eval_strategy = "steps",
  eval_steps = 1000,
  save_strategy="steps",
  save_steps = 1000,
  learning_rate = 5e-5,
  per_device_train_batch_size=10,  # Optimized for A100 40GB
  per_device_eval_batch_size=10,
  gradient_accumulation_steps=2,  # Effective batch size = 20
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
logger.info("Loading data files...")
d = pd.read_csv('all_sctt_jrt.csv')
gen = pd.read_csv('sctt_item-generalization_jrt.csv')
logger.info(f"Loaded {len(d)} records from all_sctt_jrt.csv and {len(gen)} records from sctt_item-generalization_jrt.csv")

#  INITIALIZE MODEL & TOKENIZER (PRE-TRAINED)
logger.info(f"Loading model and tokenizer: {model_name}")
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, token=hftoken)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hftoken)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Set pad_token_id in model config
model.config.pad_token_id = tokenizer.pad_token_id

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
logger.info(model.print_trainable_parameters())

#  PREPROCESS DATA
logger.info("Preprocessing data...")
def tokenize_function(examples): #  define wrapper tokenizer function (for batch training)
  # Filter out None values and convert to strings
  texts = [str(text) if text is not None else "" for text in examples['text']]
  return tokenizer(texts, padding = 'max_length', truncation = True, max_length = 512)
d = preprocess_llm_data(d, gen, val_pct, val_train_pct, test_pct, prefix, connector1, connector2)
# Filter out rows with None text values before tokenizing
d = d.filter(lambda example: example['text'] is not None and example['text'] != '')
tokenized_datasets = d.map(tokenize_function, batched = True) # applies wrapper to our dataset
logger.info(f"Tokenization complete. Train: {len(tokenized_datasets['train'])}, Val: {len(tokenized_datasets['validation'])}, Test: {len(tokenized_datasets['test'])}")

#  TRAIN MODEL
logger.info("Initializing trainer...")
# Mark model as already on device to prevent Trainer from trying to move it
model.is_parallelizable = True
model.model_parallel = True

trainer = Trainer(
  model=model,
  args=trainer_args,
  train_dataset=tokenized_datasets["train"],
  eval_dataset=tokenized_datasets["validation"],
  compute_metrics=compute_metrics,
)

# Check for existing checkpoints and resume if available
import glob
checkpoint_dirs = glob.glob('./sctt_results_{}_{}/checkpoint-*'.format(expname,model_name.split('/')[1]))
if checkpoint_dirs:
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getctime)
    logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    logger.info("Starting training from scratch")
    trainer.train()

logger.info("\n" + "="*70)
logger.info("TRAINING COMPLETE - Saving final model and generating predictions")
logger.info("="*70)

# SAVE MERGED MODEL (LoRA + classification head)
logger.info("Merging LoRA adapters and saving full model...")
final_model_path = './final_model_sctt_results_{}_{}'.format(expname,model_name.split('/')[1])

# Merge LoRA weights into base model
merged_model = model.merge_and_unload()
merged_model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
logger.info(f"✓ Full merged model saved to: {final_model_path}")

# VALIDATION
logger.info("\nRunning validation predictions...")
val_prediction = trainer.predict(tokenized_datasets['validation'])
val_output_df = pd.DataFrame({'preds': val_prediction.predictions.flatten(), 'ratings': val_prediction.label_ids})
val_output_df.to_csv('validation_output_sctt_results_{}_{}.csv'.format(expname,model_name.split('/')[1]), index=False)
logger.info(f"✓ Validation predictions saved ({len(val_output_df)} samples)")
logger.info(f"  Metrics: {val_prediction.metrics}")

# TEST
logger.info("\nRunning test predictions...")
test_prediction = trainer.predict(tokenized_datasets['test'])
test_output_df = pd.DataFrame({'preds': test_prediction.predictions.flatten(), 'ratings': test_prediction.label_ids})
test_output_df.to_csv('test_output_sctt_results_{}_{}.csv'.format(expname,model_name.split('/')[1]), index=False)
logger.info(f"✓ Test predictions saved ({len(test_output_df)} samples)")
logger.info(f"  Metrics: {test_prediction.metrics}")

#  HELDOUT TEST
logger.info("\nRunning heldout predictions...")
heldout_prediction = trainer.predict(tokenized_datasets['heldout'])
heldoutprompt_output_df = pd.DataFrame({'preds': heldout_prediction.predictions.flatten(), 'ratings': heldout_prediction.label_ids})
heldoutprompt_output_df.to_csv('heldoutprompt_output_sctt_results_{}_{}.csv'.format(expname,model_name.split('/')[1]), index=False)
logger.info(f"✓ Heldout predictions saved ({len(heldoutprompt_output_df)} samples)")
logger.info(f"  Metrics: {heldout_prediction.metrics}")

logger.info("\n" + "="*70)
logger.info("ALL TASKS COMPLETE!")
logger.info("="*70)
logger.info(f"Final model: {final_model_path}")
logger.info("Validation predictions: validation_output_sctt_results_{}_{}.csv".format(expname,model_name.split('/')[1]))
logger.info("Test predictions: test_output_sctt_results_{}_{}.csv".format(expname,model_name.split('/')[1]))
logger.info("Heldout predictions: heldoutprompt_output_sctt_results_{}_{}.csv".format(expname,model_name.split('/')[1]))