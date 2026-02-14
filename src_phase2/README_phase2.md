# Phase 2: Calibration and Uncertainty Quantification via Semantic Entropy

This directory contains the implementation of Phase 2 of the thesis research, which evaluates semantic entropy as an uncertainty quantification method for LLM-based creativity assessment.

## Overview

**Research Question**: Does the model know when it is uncertain about different meanings?

**Method**: Semantic Entropy (Kuhn et al., 2023)
- **Generation**: M = 10 stochastic forward passes with temperature T = 0.5
- **Semantic Clustering**: Bidirectional entailment via DeBERTa-large NLI classifier
- **Entropy Computation**: Entropy over semantic equivalence classes

**Evaluation Metrics**:
- Pearson correlation between semantic entropy and absolute prediction error
- Spearman correlation between semantic entropy and absolute prediction error
- AUROC for predicting response correctness from entropy magnitude

## Files

### Core Scripts

1. **`phase2_semantic_entropy.py`**
   - Main evaluation script
   - Implements semantic entropy calculation
   - Evaluates model predictions with uncertainty quantification
   - Computes calibration metrics

2. **`phase2_visualizations.py`**
   - Creates visualizations for calibration analysis
   - Generates plots: entropy vs error, ROC curves, reliability diagrams
   - Produces summary reports

3. **`compare_paradigms.py`**
   - Compares pairwise ranking vs direct regression paradigms
   - Statistical significance testing
   - Side-by-side visualization

### Support Files

4. **`requirements_phase2.txt`**
   - Python dependencies

## Installation

```bash
# Create virtual environment
python -m venv venv_phase2
source venv_phase2/bin/activate  # On Windows: venv_phase2\Scripts\activate

# Install dependencies
pip install -r requirements_phase2.txt

# Install HuggingFace CLI (if needed for model access)
pip install huggingface_hub
huggingface-cli login
```

## Usage

### Step 1: Run Semantic Entropy Evaluation

**For Regression Model:**
```bash
python phase2_semantic_entropy.py \
    --data_path /path/to/all_sctt_jrt.csv \
    --base_model meta-llama/Llama-2-7b-chat-hf \
    --lora_adapter /path/to/regression_adapter \
    --model_type regression \
    --output_dir ./results/regression \
    --num_generations 10 \
    --temperature 0.5 \
    --max_samples 100  # Optional: for quick testing
```

**For Pairwise Ranking Model:**
```bash
python phase2_semantic_entropy.py \
    --data_path /path/to/all_sctt_jrt.csv \
    --base_model meta-llama/Llama-2-7b-chat-hf \
    --lora_adapter /path/to/pairwise_adapter \
    --model_type pairwise \
    --output_dir ./results/pairwise \
    --num_generations 10 \
    --temperature 0.5
```

**Output Files:**
- `results_{model_type}_{timestamp}.csv`: Detailed results for each sample
- `metrics_{model_type}_{timestamp}.json`: Aggregated calibration metrics
- `config_{model_type}_{timestamp}.json`: Experiment configuration

### Step 2: Generate Visualizations

```bash
python phase2_visualizations.py \
    --results_csv ./results/regression/results_regression_*.csv \
    --metrics_json ./results/regression/metrics_regression_*.json \
    --output_dir ./visualizations/regression
```

**Generated Plots:**
- `entropy_vs_error.png`: Scatter plot showing calibration relationship
- `entropy_distribution.png`: Distribution of entropy values
- `clusters_analysis.png`: Relationship between semantic clusters and error
- `roc_curve.png`: ROC curve for error prediction
- `calibration_reliability.png`: Reliability diagram
- `summary_report.txt`: Comprehensive text report

### Step 3: Compare Paradigms

```bash
python compare_paradigms.py \
    --regression_results ./results/regression/results_regression_*.csv \
    --regression_metrics ./results/regression/metrics_regression_*.json \
    --pairwise_results ./results/pairwise/results_pairwise_*.csv \
    --pairwise_metrics ./results/pairwise/metrics_pairwise_*.json \
    --output_dir ./comparison
```

## Command Line Arguments

### phase2_semantic_entropy.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_path` | str | Required | Path to SCTT data CSV |
| `--base_model` | str | `meta-llama/Llama-2-7b-chat-hf` | Base model path |
| `--lora_adapter` | str | `None` | Path to LoRA adapter |
| `--model_type` | str | `regression` | Training paradigm: `regression` or `pairwise` |
| `--output_dir` | str | `./phase2_results` | Output directory |
| `--num_generations` | int | `10` | Number of stochastic generations (M) |
| `--temperature` | float | `0.5` | Sampling temperature (T) |
| `--batch_size` | int | `1` | Batch size for evaluation |
| `--max_samples` | int | `None` | Max samples to evaluate (for testing) |
| `--device` | str | `cuda` | Device for computation |

## Expected Results

### Well-Calibrated Model
- **Pearson correlation > 0.3**: Strong positive correlation between entropy and error
- **AUROC > 0.7**: Good discrimination for identifying incorrect predictions
- **Interpretation**: High semantic entropy reliably indicates high prediction error

### Poorly-Calibrated Model
- **Pearson correlation < 0.1**: Weak correlation
- **AUROC < 0.6**: Poor discrimination
- **Interpretation**: Semantic entropy doesn't predict errors well

## Interpreting the Results

### Semantic Entropy
- **Low entropy (< 1.0)**: Model generates semantically similar responses → high confidence
- **High entropy (> 2.0)**: Model generates diverse semantic meanings → high uncertainty

### Number of Semantic Clusters
- **Few clusters (1-3)**: Responses convey similar meanings (paraphrases)
- **Many clusters (5+)**: Responses convey distinct meanings (high uncertainty)

### Calibration Quality
1. **Perfect Calibration**: Linear relationship between entropy and error (r ≈ 1.0)
2. **Good Calibration**: Strong positive correlation (r > 0.3, AUROC > 0.7)
3. **Poor Calibration**: Weak/negative correlation (r < 0.1, AUROC < 0.6)

## Computational Requirements

### Memory
- **7B Model**: ~14 GB GPU memory (FP16)
- **13B Model**: ~26 GB GPU memory (FP16)
- With 10 generations: Add ~20% overhead

### Time Estimates (NVIDIA A100)
- **7B Model**: ~5-10 seconds per sample (10 generations + clustering)
- **Full dataset (17k samples)**: ~24-48 hours
- **Subset (1000 samples)**: ~2-3 hours

### Optimization Tips
1. Use `--max_samples` for quick prototyping
2. Reduce `--num_generations` for faster testing (minimum: 5)
3. Increase `--batch_size` if memory permits
4. Use gradient checkpointing for larger models

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 1

# Use 8-bit quantization
# Add to script: load_in_8bit=True

# Reduce number of generations
--num_generations 5
```

### Slow NLI Clustering
- The DeBERTa NLI model performs O(n²) comparisons
- For 10 generations: 45 pairwise comparisons per sample
- This is the bottleneck - consider caching results

### Model Loading Issues
```bash
# Ensure HuggingFace authentication
huggingface-cli login

# Download model first
python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-2-7b-chat-hf')"
```

## Citation

If you use this code, please cite:

```bibtex
@article{kuhn2023semantic,
  title={Semantic entropy probes: Robust and cheap hallucination detection in LLMs},
  author={Kuhn, Lorenz and Gal, Yarin and Farquhar, Sebastian},
  journal={arXiv preprint arXiv:2406.15927},
  year={2023}
}
```

## Thesis Context

This is Phase 2 of a larger thesis project:
- **Phase 1**: Elo-based pairwise ranking for creativity assessment
- **Phase 2**: Semantic entropy for uncertainty quantification (this phase)

The goal is to determine which training paradigm (pairwise vs regression) produces better-calibrated uncertainty estimates for hybrid human-AI workflows where the model flags low-confidence predictions for human review.

## Contact

For questions or issues, contact [your email] or open an issue in the repository.

## License

[Your chosen license]
