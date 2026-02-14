#!/bin/bash
# Phase 2: Example Usage Pipeline
# =================================
# 
# This script demonstrates the complete Phase 2 workflow:
# 1. Run semantic entropy evaluation on both paradigms
# 2. Generate visualizations for each
# 3. Compare paradigms statistically
# 4. Create comprehensive analysis

set -e  # Exit on error

echo "========================================================================"
echo "PHASE 2: SEMANTIC ENTROPY CALIBRATION - EXAMPLE PIPELINE"
echo "========================================================================"

# Configuration
DATA_PATH="/mnt/project/all_sctt_jrt.csv"
BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"
OUTPUT_ROOT="./phase2_outputs"
MAX_SAMPLES=100  # Set to null for full dataset

# Create output directories
mkdir -p $OUTPUT_ROOT/regression
mkdir -p $OUTPUT_ROOT/pairwise
mkdir -p $OUTPUT_ROOT/visualizations
mkdir -p $OUTPUT_ROOT/comparison

echo ""
echo "Configuration:"
echo "  Data: $DATA_PATH"
echo "  Base Model: $BASE_MODEL"
echo "  Output: $OUTPUT_ROOT"
echo "  Max Samples: $MAX_SAMPLES (use for quick testing)"
echo ""

# ============================================================================
# STEP 1: Evaluate Regression Model
# ============================================================================
echo "========================================================================"
echo "STEP 1: Evaluating REGRESSION model with semantic entropy"
echo "========================================================================"

python phase2_semantic_entropy.py \
    --data_path $DATA_PATH \
    --base_model $BASE_MODEL \
    --lora_adapter ./models/regression_lora \
    --model_type regression \
    --output_dir $OUTPUT_ROOT/regression \
    --num_generations 10 \
    --temperature 0.5 \
    --batch_size 1 \
    --max_samples $MAX_SAMPLES

echo ""
echo "✓ Regression evaluation complete!"
echo ""

# ============================================================================
# STEP 2: Evaluate Pairwise Model
# ============================================================================
echo "========================================================================"
echo "STEP 2: Evaluating PAIRWISE model with semantic entropy"
echo "========================================================================"

python phase2_semantic_entropy.py \
    --data_path $DATA_PATH \
    --base_model $BASE_MODEL \
    --lora_adapter ./models/pairwise_lora \
    --model_type pairwise \
    --output_dir $OUTPUT_ROOT/pairwise \
    --num_generations 10 \
    --temperature 0.5 \
    --batch_size 1 \
    --max_samples $MAX_SAMPLES

echo ""
echo "✓ Pairwise evaluation complete!"
echo ""

# ============================================================================
# STEP 3: Generate Visualizations for Regression
# ============================================================================
echo "========================================================================"
echo "STEP 3: Generating visualizations for REGRESSION"
echo "========================================================================"

# Find the most recent results files
REG_RESULTS=$(ls -t $OUTPUT_ROOT/regression/results_regression_*.csv | head -1)
REG_METRICS=$(ls -t $OUTPUT_ROOT/regression/metrics_regression_*.json | head -1)

python phase2_visualizations.py \
    --results_csv $REG_RESULTS \
    --metrics_json $REG_METRICS \
    --output_dir $OUTPUT_ROOT/visualizations/regression

echo ""
echo "✓ Regression visualizations complete!"
echo ""

# ============================================================================
# STEP 4: Generate Visualizations for Pairwise
# ============================================================================
echo "========================================================================"
echo "STEP 4: Generating visualizations for PAIRWISE"
echo "========================================================================"

PAIR_RESULTS=$(ls -t $OUTPUT_ROOT/pairwise/results_pairwise_*.csv | head -1)
PAIR_METRICS=$(ls -t $OUTPUT_ROOT/pairwise/metrics_pairwise_*.json | head -1)

python phase2_visualizations.py \
    --results_csv $PAIR_RESULTS \
    --metrics_json $PAIR_METRICS \
    --output_dir $OUTPUT_ROOT/visualizations/pairwise

echo ""
echo "✓ Pairwise visualizations complete!"
echo ""

# ============================================================================
# STEP 5: Compare Paradigms
# ============================================================================
echo "========================================================================"
echo "STEP 5: Comparing REGRESSION vs PAIRWISE paradigms"
echo "========================================================================"

python compare_paradigms.py \
    --regression_results $REG_RESULTS \
    --regression_metrics $REG_METRICS \
    --pairwise_results $PAIR_RESULTS \
    --pairwise_metrics $PAIR_METRICS \
    --output_dir $OUTPUT_ROOT/comparison

echo ""
echo "✓ Paradigm comparison complete!"
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo "========================================================================"
echo "PHASE 2 PIPELINE COMPLETE!"
echo "========================================================================"
echo ""
echo "Results are organized in $OUTPUT_ROOT:"
echo ""
echo "  regression/"
echo "    └── Regression model results and metrics"
echo ""
echo "  pairwise/"
echo "    └── Pairwise model results and metrics"
echo ""
echo "  visualizations/"
echo "    ├── regression/"
echo "    │   └── Plots and reports for regression model"
echo "    └── pairwise/"
echo "        └── Plots and reports for pairwise model"
echo ""
echo "  comparison/"
echo "    └── Side-by-side comparison and statistical tests"
echo ""
echo "Key files to review:"
echo "  1. $OUTPUT_ROOT/comparison/comparison_report.txt"
echo "  2. $OUTPUT_ROOT/visualizations/regression/summary_report.txt"
echo "  3. $OUTPUT_ROOT/visualizations/pairwise/summary_report.txt"
echo ""
echo "Next steps:"
echo "  - Review comparison_report.txt for winner determination"
echo "  - Check calibration plots for visual assessment"
echo "  - Use results for thesis Phase 2 write-up"
echo ""
echo "========================================================================"
