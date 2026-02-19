"""
CUDA Troubleshooting Script
Run this script to diagnose CUDA and GPU issues with PyTorch
"""

import sys
import os

print("=" * 70)
print("CUDA TROUBLESHOOTING DIAGNOSTIC")
print("=" * 70)

# Check Python version
print(f"\n1. Python Version: {sys.version}")

# Check PyTorch installation
try:
    import torch
    print(f"2. PyTorch Version: {torch.__version__}")
    print(f"   PyTorch CUDA Compiled Version: {torch.version.cuda if torch.version.cuda else 'CPU-only'}")
except ImportError:
    print("2. ❌ PyTorch NOT installed!")
    print("   Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# Check CUDA availability
print(f"\n3. CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   ✓ CUDA is working!")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"      Name: {torch.cuda.get_device_name(i)}")
        print(f"      Capability: {torch.cuda.get_device_capability(i)}")
        print(f"      Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
    # Test a simple CUDA operation
    try:
        x = torch.rand(5, 3).cuda()
        print(f"\n   ✓ Test tensor created on GPU successfully")
        print(f"   Current Device: {torch.cuda.current_device()}")
        print(f"   Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    except Exception as e:
        print(f"\n   ❌ Failed to create tensor on GPU: {e}")
else:
    print("   ❌ CUDA is NOT available!")
    print("\n   Possible issues:")
    print("   1. PyTorch installed without CUDA support (CPU-only version)")
    print("   2. No NVIDIA GPU detected")
    print("   3. NVIDIA drivers not installed or outdated")
    print("   4. CUDA toolkit version mismatch")
    
    print("\n   Solutions:")
    print("   • Reinstall PyTorch with CUDA:")
    print("     pip uninstall torch torchvision torchaudio -y")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("   • Update NVIDIA drivers: https://www.nvidia.com/download/index.aspx")
    print("   • Check if GPU is enabled in Device Manager")

# Check other related packages
print("\n4. Related Packages:")
packages = ['transformers', 'datasets', 'peft', 'accelerate', 'bitsandbytes']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✓ {pkg}: {version}")
    except ImportError:
        print(f"   ❌ {pkg}: NOT installed")

# Check environment variables
print("\n5. Environment Variables:")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"   CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")

# Check cuDNN
print("\n6. cuDNN:")
print(f"   Available: {torch.backends.cudnn.is_available()}")
if torch.backends.cudnn.is_available():
    print(f"   Version: {torch.backends.cudnn.version()}")
    print(f"   Enabled: {torch.backends.cudnn.enabled}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

# Final recommendation
if torch.cuda.is_available():
    print("\n✓ Your system is ready for GPU-accelerated deep learning!")
else:
    print("\n⚠ GPU acceleration is NOT available. Training will be very slow.")
    print("   Please follow the solutions above to enable CUDA.")
