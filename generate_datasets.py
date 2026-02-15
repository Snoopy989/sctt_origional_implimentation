"""
Standalone script to generate train/validation/test/heldout datasets
without running full training pipeline.
"""
import pandas as pd
from dataprocessing import preprocess_llm_data

# Settings (matching training script)
val_pct = 0.10          # 10% validation
test_pct = 0.20         # 20% test
val_train_pct = (1.0/(1.0-test_pct))*val_pct  # Adjusted validation percentage

# Prompt formatting
prefix = "A creative "
connector1 = " for "
connector2 = " is "

print("="*80)
print("GENERATING TRAIN/VAL/TEST DATASETS")
print("="*80)
print(f"Split: {70}% train, {val_pct*100:.0f}% validation, {test_pct*100:.0f}% test")
print(f"Seed: 42 (reproducible)")
print("="*80)

# Load source data
print("\nLoading source data...")
d = pd.read_csv('data/raw/all_sctt_jrt.csv')
gen = pd.read_csv('data/raw/sctt_item-generalization_jrt.csv')
print(f"Loaded {len(d)} main samples")
print(f"Loaded {len(gen)} heldout samples")

# Generate splits
print("\nGenerating splits...")
dataset = preprocess_llm_data(d, gen, val_pct, val_train_pct, test_pct, prefix, connector1, connector2)

print("\n" + "="*80)
print("DATASETS GENERATED SUCCESSFULLY")
print("="*80)
print("\nGenerated files in data/processed/:")
print("  - training_items_sctt.csv")
print("  - validation_items_sctt.csv")
print("  - test_items_sctt.csv")
print("  - heldoutprompt_items_sctt.csv")
print("\nDebugging files in data/debug/:")
print("  - sctt_jrt_cleaned.csv")
print("  - d_dupes.csv")
print("  - d_dupes_avg.csv")
print("="*80)
