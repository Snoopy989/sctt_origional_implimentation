import re
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULT_DIRS = [
    PROJECT_ROOT / "results" / "training",
    PROJECT_ROOT / "results" / "training_cluster_good",
]


def _find_col(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _epoch_from_filename(path: Path):
    match = re.search(r"LORA_(\d+)_epochs", path.name)
    if match:
        return int(match.group(1))
    return None


def evaluate_inference_file(csv_path: Path, split: str):
    df = pd.read_csv(csv_path)

    pred_col = _find_col(df, ["preds", "predictions"])
    rating_col = _find_col(df, ["ratings", "label", "labels"])

    if pred_col is None or rating_col is None:
        return None

    preds = df[pred_col].astype(float).values
    ratings = df[rating_col].astype(float).values

    if len(preds) < 2:
        return None

    corr, p_value = pearsonr(preds, ratings)
    epoch = _epoch_from_filename(csv_path)

    return {
        "split": split,
        "epoch": epoch,
        "pearson_r": corr,
        "p_value": p_value,
        "n_samples": len(preds),
        "source_file": str(csv_path.relative_to(PROJECT_ROOT)),
        "source_dir": str(csv_path.parent.relative_to(PROJECT_ROOT)),
    }


def collect_results_for_split(split: str):
    rows = []
    pattern = f"{split}_output_inference_LORA_*_epochs_*.csv"

    for result_dir in RESULT_DIRS:
        if not result_dir.exists():
            continue
        for csv_path in sorted(result_dir.glob(pattern)):
            row = evaluate_inference_file(csv_path, split)
            if row:
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False).reset_index(drop=True)


def main():
    split_tables = {}

    for split in ["train", "validation", "test", "heldout"]:
        df = collect_results_for_split(split)
        if df.empty:
            continue

        split_tables[split] = df

        print("\n" + "=" * 90)
        print(f"EPOCH PERFORMANCE ({split.upper()} - Pearson r)")
        print("=" * 90)
        print(df.to_string(index=False))
        print("=" * 90)

        best = df.iloc[0]
        print(f"\nBEST ({split}): epoch={int(best['epoch']) if pd.notna(best['epoch']) else 'unknown'}")
        print(f"  Pearson r: {best['pearson_r']:.4f}")
        print(f"  p-value:   {best['p_value']:.6f}")
        print(f"  Samples:   {int(best['n_samples'])}")

        output_file = PROJECT_ROOT / f"epoch_performance_{split}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file.relative_to(PROJECT_ROOT)}")

    if "validation" in split_tables and not split_tables["validation"].empty:
        best_val = split_tables["validation"].iloc[0]
        print("\n" + "#" * 90)
        print("RECOMMENDED EPOCH (based on VALIDATION Pearson r)")
        print("#" * 90)
        print(f"epoch:      {int(best_val['epoch']) if pd.notna(best_val['epoch']) else 'unknown'}")
        print(f"pearson_r:  {best_val['pearson_r']:.4f}")
        print(f"source:     {best_val['source_file']}")
        print("#" * 90)
    elif not split_tables:
        print("\nNo matching inference CSV files found in current structure.")
        print("Expected under: results/training and/or results/training_cluster_good")


if __name__ == "__main__":
    main()
