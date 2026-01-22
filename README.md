# SCTT Llama-2 LoRA Fine-tuning

Fine-tuning Llama-2 models with LoRA for Scientific Creative Thinking Task (SCTT) evaluation.

## Credits

**Original Code:** [OSF Project](https://osf.io/439zs/overview?view_only=4cbda208526948a99afba0050a2c043f)  
**Updated by:** Phillip Gregory (2026) - Modernized for current libraries and Windows compatibility

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA support (recommended)
- ~20GB disk space for model downloads
- HuggingFace account with Llama-2 access

## Setup

### 1. Get Llama-2 Access
1. Visit [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
2. Request access and accept Meta's license
3. Create a HuggingFace token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 2. Install Dependencies

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### 3. Configure Environment

Update `.env` with your HuggingFace token:
```env
HF_TOKEN=hf_your_token_here
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
```

## Running

### Train Model
```powershell
python run_sctt_llama2chat_lora.py
```

### Troubleshoot CUDA
```powershell
python cuda_troubleshoot.py
```

## What It Does

- **Fine-tunes** Llama-2-7b-chat with LoRA adapters
- **Trains on** SCTT dataset (scientific creativity evaluation)
- **Saves checkpoints** every 1000 steps
- **Outputs** predictions and validation results as CSV

## Training Configuration

- **Epochs:** 10 (configurable in script)
- **Batch size:** 1 (with gradient accumulation)
- **Learning rate:** 5e-5
- **LoRA rank:** 4
- **Precision:** FP16

## Files

- `run_sctt_llama2chat_lora.py` - Main training script (Llama-2-chat)
- `run_sctt_llama2_lora.py` - Base Llama-2 training
- `dataprocessing.py` - Data preprocessing utilities
- `lora_misc.py` - LoRA helper functions
- `cuda_troubleshoot.py` - GPU diagnostics
- `.env` - Environment configuration
- `requirements.txt` - Python dependencies

## Outputs

- `sctt_results_LORA_*_epochs_*/` - Model checkpoints
- `validation_output_sctt_results_*.csv` - Validation predictions
- `test_items_sctt.csv`, `train_items_sctt.csv`, etc. - Split datasets

## Notes

- Training on CPU is **not recommended** (extremely slow)
- First run will download ~13GB model from HuggingFace
- Checkpointing allows resuming interrupted training
- Windows users: Ensure CUDA toolkit is installed for GPU support
