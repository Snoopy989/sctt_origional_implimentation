import os
import torch
from transformers import Llama4TextModel, AutoConfig
from dotenv import load_dotenv

load_dotenv()

# Test if the model exists and can be loaded
model_name = 'meta-llama/Llama-4-Scout-17B-16E-Instruct'
hftoken = os.getenv('HF_TOKEN')

print(f"Attempting to load config for: {model_name}")
try:
    config = AutoConfig.from_pretrained(model_name, token=hftoken)
    print(f"✓ Config loaded successfully")
    print(f"Model type: {config.model_type}")
except Exception as e:
    print(f"✗ Failed to load config: {e}")
    print("\nThis model may not exist or you may not have access to it.")
    print("Check: https://huggingface.co/" + model_name)
    exit(1)

print("\n" + "="*50)
print("Now attempting to load Llama4TextModel...")
print("This may take 10-20 minutes for a 17B model")
print("="*50 + "\n")

try:
    # Use same settings as your training script
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = Llama4TextModel.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
        token=hftoken,
    )
    
    print(f"\n✓ Model loaded successfully!")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Dtype: {next(model.parameters()).dtype}")
    print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
    
except Exception as e:
    print(f"\n✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
