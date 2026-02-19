"""Quick script to check what modules are actually saved in the adapter"""
import torch
from safetensors import safe_open

adapter_path = "./models/Llama-2-7b_sctt_regression/adapter_model.safetensors"

print("Modules in saved adapter:")
print("-" * 60)

with safe_open(adapter_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    
    # Get unique module prefixes
    module_names = set()
    for key in keys:
        # Extract the module name (first part before the dot)
        module = key.split('.')[0]
        module_names.add(module)
    
    print(f"\nUnique top-level modules: {sorted(module_names)}")
    
    # Show all keys
    print(f"\nAll keys ({len(keys)} total):")
    for key in sorted(keys):
        print(f"  - {key}")
