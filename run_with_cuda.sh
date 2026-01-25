#!/bin/bash
# Script to run training with proper CUDA environment
export LD_LIBRARY_PATH=/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
# Skip bitsandbytes for now since it doesn't have ARM64 + CUDA 12.8 support
python3 run_sctt_llama2_13b_chat_lora.py "$@"
