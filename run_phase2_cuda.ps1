# PowerShell script to run Phase 2 semantic entropy evaluation with CUDA
# Usage: .\run_phase2_cuda.ps1 [additional arguments]
# Example: .\run_phase2_cuda.ps1 --num_generations 10 --max_samples 50

# Set CUDA device (change if you have multiple GPUs)
$env:CUDA_VISIBLE_DEVICES = "0"

# Run the Phase 2 script with CUDA enabled using the virtual environment
.\.venv\Scripts\python.exe src_phase2\phase2_semantic_entropy.py --device cuda $args
