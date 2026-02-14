# Convergence Analysis Runner
# Tests semantic entropy with different numbers of generations (n)

$env:CUDA_VISIBLE_DEVICES = "0"

# Activate virtual environment
& .\.venv\Scripts\python.exe src_phase2\quick_test.py @args
