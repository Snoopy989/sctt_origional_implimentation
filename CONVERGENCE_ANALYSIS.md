# Convergence Analysis - Usage Guide

## Purpose
Test how semantic entropy metrics converge with different numbers of stochastic generations (n).

This helps determine:
- Is weak AUROC (~0.57) due to insufficient generations?
- Or is the signal genuinely weak?
- What is the optimal n for stable uncertainty estimation?

## Setup

First, install required visualization packages:

```powershell
.\.venv\Scripts\pip install matplotlib seaborn
```

## Usage

### Basic Run (Default: 250 samples, n=10,20,30,50)
```powershell
.\run_convergence_analysis.ps1
```

### Custom Sample Size
```powershell
.\run_convergence_analysis.ps1 --num_samples 200
```

### Custom n Values
```powershell
.\run_convergence_analysis.ps1 --n_values 10 15 20 25 30
```

### With LoRA Adapter
```powershell
.\run_convergence_analysis.ps1 --lora_adapter ./Llama-2-7b_sctt_regression
```

### Full Example
```powershell
.\run_convergence_analysis.ps1 --num_samples 300 --n_values 10 20 30 50 100
```

### Custom Output Directory (Optional)
```powershell
# Save to a different location
.\run_convergence_analysis.ps1 --output_dir ./custom_results
```

By default, results are saved to `phase2_results/` in the project root.

## Output

The script generates:

1. **Convergence Plot** (`convergence_analysis_TIMESTAMP.png`):
   - 6 subplots showing metric evolution across n values
   - Pearson/Spearman correlations
   - AUROC trend
   - Entropy stability
   - Clustering behavior

2. **Summary Table** (`convergence_summary_TIMESTAMP.csv`):
   - CSV with all metrics for each n value

3. **JSON Analysis** (`convergence_analysis_TIMESTAMP.json`):
   - Full results with interpretation
   - Recommended optimal n value

## Interpretation

### If AUROC improves n=10→20→30 then plateaus:
✓ **Increase n to 30-50** for better calibration  
→ Weak signal was due to insufficient sampling

### If AUROC stays flat across all n:
✓ **n=10 is sufficient**  
→ Current signal is real (not noise from insufficient generations)  
→ Need to investigate other sources of weak calibration

### If AUROC varies without clear trend:
? **Need more samples** to determine optimal n  
→ Run with --num_samples 500

## Example Output Location

All results are saved to the project root's `phase2_results/` directory:

```
phase2_results/
├── convergence_analysis_20260214_143052.png     # Visualization
├── convergence_summary_20260214_143052.csv      # Metrics table
└── convergence_analysis_20260214_143052.json    # Full analysis
```

(Same directory used by the main phase2_semantic_entropy.py script)

## Parameters

- `--num_samples`: Sample size (default: 250, range: 200-300 recommended)
- `--n_values`: List of n to test (default: 10 20 30 50)
- `--base_model`: Base model path (default: from .env MODEL_NAME)
- `--lora_adapter`: LoRA adapter path (optional)
- `--test_data`: Path to test CSV (default: ../data/test_items_sctt.csv)
- `--output_dir`: Output directory (default: ./phase2_results)
- `--device`: cuda or cpu (default: cuda)

## Typical Runtime

- **n=10**: ~2 min per 100 samples
- **n=20**: ~4 min per 100 samples  
- **n=30**: ~6 min per 100 samples
- **n=50**: ~10 min per 100 samples

For 250 samples testing [10, 20, 30, 50]:
- **Total runtime**: ~55 minutes on RTX 4080

## Next Steps After Analysis

1. **If metrics improve**: Rerun full evaluation with higher n
   ```powershell
   .\run_phase2_cuda.ps1 --sampling_fraction 0.2 --num_generations 30
   ```

2. **If metrics stay flat**: Current n=10 is fine, investigate other factors:
   - Model calibration during training
   - DBSCAN hyperparameters (eps, min_samples)
   - Alternative clustering approaches
