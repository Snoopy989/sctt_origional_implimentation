# Phase 2: Semantic Entropy Calibration - Complete Package

## 📋 Overview

This package implements **Phase 2** of your thesis: **Calibration and Uncertainty Quantification via Semantic Entropy** for LLM-based creativity assessment.

**Key Question**: Does the model know when it is uncertain about different meanings?

**Method**: Semantic Entropy (Kuhn et al., 2023) applied to both pairwise ranking and direct regression models trained on SCTT data.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv_phase2
source venv_phase2/bin/activate

# Install dependencies
pip install -r requirements_phase2.txt
```

### 2. Quick Test (Verify Installation)

```bash
# Test semantic entropy calculation
python quick_test.py --test_type entropy

# Test with your data
python quick_test.py --test_type both --data_path /mnt/project/all_sctt_jrt.csv
```

### 3. Run Full Pipeline

**Option A: Automated Pipeline (Recommended)**
```bash
# Edit run_phase2_pipeline.sh to set your paths
# Then run:
./run_phase2_pipeline.sh
```

**Option B: Manual Step-by-Step**
```bash
# Step 1: Evaluate regression model
python phase2_semantic_entropy.py \
    --data_path /mnt/project/all_sctt_jrt.csv \
    --base_model meta-llama/Llama-2-7b-chat-hf \
    --lora_adapter /path/to/regression_adapter \
    --model_type regression \
    --output_dir ./results/regression

# Step 2: Evaluate pairwise model
python phase2_semantic_entropy.py \
    --data_path /mnt/project/all_sctt_jrt.csv \
    --base_model meta-llama/Llama-2-7b-chat-hf \
    --lora_adapter /path/to/pairwise_adapter \
    --model_type pairwise \
    --output_dir ./results/pairwise

# Step 3: Generate visualizations
python phase2_visualizations.py \
    --results_csv ./results/regression/results_*.csv \
    --metrics_json ./results/regression/metrics_*.json \
    --output_dir ./viz/regression

# Step 4: Compare paradigms
python compare_paradigms.py \
    --regression_results ./results/regression/results_*.csv \
    --regression_metrics ./results/regression/metrics_*.json \
    --pairwise_results ./results/pairwise/results_*.csv \
    --pairwise_metrics ./results/pairwise/metrics_*.json \
    --output_dir ./comparison
```

## 📁 File Structure

```
phase2_scripts/
├── README.md                        # This file
├── README_phase2.md                 # Detailed documentation
├── requirements_phase2.txt          # Dependencies
├── phase2_semantic_entropy.py       # Main evaluation script
├── phase2_visualizations.py         # Visualization script
├── compare_paradigms.py             # Paradigm comparison
├── quick_test.py                    # Testing utilities
└── run_phase2_pipeline.sh           # Automated pipeline
```

## 🎯 What Each Script Does

### 1. `phase2_semantic_entropy.py` - Core Evaluation
**Purpose**: Compute semantic entropy and evaluate calibration

**Key Features**:
- Generates M=10 sequences per test instance (T=0.5)
- Clusters using DeBERTa NLI for bidirectional entailment
- Computes semantic entropy over clusters
- Evaluates Pearson/Spearman correlations and AUROC

**Output**: 
- `results_{model}_{timestamp}.csv` - Per-sample predictions & entropy
- `metrics_{model}_{timestamp}.json` - Aggregated metrics

### 2. `phase2_visualizations.py` - Analysis & Plots
**Purpose**: Create visualizations for calibration assessment

**Generated Plots**:
- Entropy vs Error scatter plot with correlation
- Entropy distribution by error quartiles
- Semantic cluster analysis
- ROC curve for error prediction
- Calibration reliability diagram

**Output**: 5 plots + summary report

### 3. `compare_paradigms.py` - Statistical Comparison
**Purpose**: Compare pairwise vs regression paradigms

**Analysis**:
- T-tests and Mann-Whitney U tests
- Fisher's z-transformation for correlation comparison
- Side-by-side visualizations
- Winner determination

**Output**: Comparison report with statistical significance

### 4. `quick_test.py` - Sanity Checks
**Purpose**: Quick verification without full model

**Tests**:
- NLI entailment checking
- Semantic clustering algorithm
- Data loading and validation


## ⚙️ Configuration Tips

### Memory Optimization
```bash
# Reduce batch size
--batch_size 1

# Reduce generations (minimum 5)
--num_generations 5

# Test on subset first
--max_samples 100
```

### Speed Optimization
```bash
# Use smaller NLI model (edit in script)
nli_model_name = "microsoft/deberta-base-mnli"  # Instead of large

# Parallel processing (if multiple GPUs)
CUDA_VISIBLE_DEVICES=0,1 python ...
```

### Quick Testing
```bash
# Test on 100 samples only
--max_samples 100

# Use 5 generations instead of 10
--num_generations 5
```





## Citation

Key paper for this method:
```bibtex
@article{kuhn2023semantic,
  title={Semantic entropy probes: Robust and cheap hallucination detection in LLMs},
  author={Kuhn, Lorenz and Gal, Yarin and Farquhar, Sebastian},
  journal={arXiv preprint arXiv:2406.15927},
  year={2023}
}
```