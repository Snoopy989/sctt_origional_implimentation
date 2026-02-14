"""
Phase 2: Visualization and Analysis
====================================

Creates visualizations for semantic entropy calibration results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import argparse
import os

sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_entropy_vs_error(results_df: pd.DataFrame, output_path: str):
    """
    Plot semantic entropy vs absolute prediction error.
    
    Well-calibrated uncertainty should show positive correlation:
    higher entropy → higher error.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(
        results_df['semantic_entropy'],
        results_df['error'],
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add regression line
    z = np.polyfit(results_df['semantic_entropy'], results_df['error'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(
        results_df['semantic_entropy'].min(),
        results_df['semantic_entropy'].max(),
        100
    )
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label='Linear Fit')
    
    # Calculate correlations
    pearson_r, pearson_p = stats.pearsonr(
        results_df['semantic_entropy'],
        results_df['error']
    )
    spearman_r, spearman_p = stats.spearmanr(
        results_df['semantic_entropy'],
        results_df['error']
    )
    
    # Add correlation info to plot
    textstr = f'Pearson r = {pearson_r:.3f} (p = {pearson_p:.3e})\n'
    textstr += f'Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.3e})'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(
        0.05, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=props
    )
    
    ax.set_xlabel('Semantic Entropy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Prediction Error', fontsize=12, fontweight='bold')
    ax.set_title(
        'Calibration: Semantic Entropy vs Prediction Error',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_entropy_distribution(results_df: pd.DataFrame, output_path: str):
    """Plot distribution of semantic entropy values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(
        results_df['semantic_entropy'],
        bins=30,
        edgecolor='black',
        alpha=0.7
    )
    axes[0].axvline(
        results_df['semantic_entropy'].mean(),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean = {results_df["semantic_entropy"].mean():.3f}'
    )
    axes[0].set_xlabel('Semantic Entropy', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Semantic Entropy', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by error quartiles
    results_df['error_quartile'] = pd.qcut(
        results_df['error'],
        q=4,
        labels=['Q1 (Low Error)', 'Q2', 'Q3', 'Q4 (High Error)']
    )
    
    results_df.boxplot(
        column='semantic_entropy',
        by='error_quartile',
        ax=axes[1]
    )
    axes[1].set_xlabel('Error Quartile', fontsize=12)
    axes[1].set_ylabel('Semantic Entropy', fontsize=12)
    axes[1].set_title(
        'Semantic Entropy by Error Quartile',
        fontsize=13,
        fontweight='bold'
    )
    plt.suptitle('')  # Remove auto title
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_clusters_analysis(results_df: pd.DataFrame, output_path: str):
    """Analyze relationship between number of clusters and error."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Clusters vs error
    axes[0].scatter(
        results_df['num_clusters'],
        results_df['error'],
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    
    correlation, pval = stats.spearmanr(
        results_df['num_clusters'],
        results_df['error']
    )
    
    axes[0].set_xlabel('Number of Semantic Clusters', fontsize=12)
    axes[0].set_ylabel('Absolute Prediction Error', fontsize=12)
    axes[0].set_title(
        f'Clusters vs Error (ρ = {correlation:.3f}, p = {pval:.3e})',
        fontsize=13,
        fontweight='bold'
    )
    axes[0].grid(True, alpha=0.3)
    
    # Distribution of clusters
    cluster_counts = results_df['num_clusters'].value_counts().sort_index()
    axes[1].bar(
        cluster_counts.index,
        cluster_counts.values,
        edgecolor='black',
        alpha=0.7
    )
    axes[1].set_xlabel('Number of Semantic Clusters', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(
        'Distribution of Cluster Counts',
        fontsize=13,
        fontweight='bold'
    )
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_roc_curve(results_df: pd.DataFrame, output_path: str):
    """
    Plot ROC curve for predicting incorrect responses using entropy.
    """
    from sklearn.metrics import roc_curve, auc
    
    # Define correctness
    error_threshold = results_df['error'].median()
    y_true = (results_df['error'] > error_threshold).astype(int)
    y_score = results_df['semantic_entropy']
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.3f})'
    )
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(
        'ROC Curve: Predicting Incorrect Responses from Entropy',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_calibration_reliability(results_df: pd.DataFrame, output_path: str):
    """
    Create reliability diagram for calibration assessment.
    Bins predictions by entropy and shows actual error in each bin.
    """
    # Create entropy bins
    n_bins = 10
    results_df['entropy_bin'] = pd.qcut(
        results_df['semantic_entropy'],
        q=n_bins,
        duplicates='drop'
    )
    
    # Calculate mean entropy and error per bin
    bin_stats = results_df.groupby('entropy_bin').agg({
        'semantic_entropy': 'mean',
        'error': 'mean',
        'response_id': 'count'
    }).reset_index()
    
    bin_stats.columns = ['entropy_bin', 'mean_entropy', 'mean_error', 'count']
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar plot with error bars
    ax.bar(
        range(len(bin_stats)),
        bin_stats['mean_error'],
        alpha=0.7,
        edgecolor='black',
        label='Mean Error per Entropy Bin'
    )
    
    # Add line for entropy values (scaled)
    ax2 = ax.twinx()
    ax2.plot(
        range(len(bin_stats)),
        bin_stats['mean_entropy'],
        color='red',
        marker='o',
        linewidth=2,
        markersize=8,
        label='Mean Entropy per Bin'
    )
    
    ax.set_xlabel('Entropy Bin (Low → High)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold', color='blue')
    ax2.set_ylabel('Mean Semantic Entropy', fontsize=12, fontweight='bold', color='red')
    
    ax.set_title(
        'Calibration Reliability: Error vs Entropy Bins',
        fontsize=14,
        fontweight='bold'
    )
    
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add sample counts as text
    for i, row in bin_stats.iterrows():
        ax.text(
            i, row['mean_error'] + 0.01,
            f'n={int(row["count"])}',
            ha='center',
            fontsize=8
        )
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_report(
    results_df: pd.DataFrame,
    metrics: dict,
    output_path: str
):
    """Create a text summary report."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 2: SEMANTIC ENTROPY CALIBRATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("Dataset Statistics:\n")
        f.write("-"*40 + "\n")
        f.write(f"  Total Samples: {len(results_df)}\n")
        f.write(f"  Mean JRT Score: {results_df['jrt_true'].mean():.4f}\n")
        f.write(f"  Std JRT Score: {results_df['jrt_true'].std():.4f}\n")
        f.write("\n")
        
        f.write("Semantic Entropy Statistics:\n")
        f.write("-"*40 + "\n")
        f.write(f"  Mean Entropy: {metrics['mean_entropy']:.4f}\n")
        f.write(f"  Std Entropy: {metrics['std_entropy']:.4f}\n")
        f.write(f"  Min Entropy: {results_df['semantic_entropy'].min():.4f}\n")
        f.write(f"  Max Entropy: {results_df['semantic_entropy'].max():.4f}\n")
        f.write(f"  Mean Clusters: {metrics['mean_clusters']:.2f}\n")
        f.write("\n")
        
        f.write("Calibration Metrics:\n")
        f.write("-"*40 + "\n")
        f.write(f"  Pearson Correlation: {metrics['pearson_correlation']:.4f}\n")
        f.write(f"    p-value: {metrics['pearson_pvalue']:.4e}\n")
        f.write(f"  Spearman Correlation: {metrics['spearman_correlation']:.4f}\n")
        f.write(f"    p-value: {metrics['spearman_pvalue']:.4e}\n")
        f.write(f"  AUROC: {metrics['auroc']:.4f}\n")
        f.write("\n")
        
        f.write("Prediction Performance:\n")
        f.write("-"*40 + "\n")
        f.write(f"  Mean Absolute Error: {metrics['mean_error']:.4f}\n")
        f.write(f"  RMSE: {np.sqrt((results_df['error']**2).mean()):.4f}\n")
        
        # Correlation between predictions and ground truth
        pred_corr = stats.pearsonr(
            results_df['jrt_true'],
            results_df['jrt_pred']
        )[0]
        f.write(f"  Prediction Correlation: {pred_corr:.4f}\n")
        f.write("\n")
        
        f.write("Interpretation:\n")
        f.write("-"*40 + "\n")
        
        if metrics['pearson_correlation'] > 0.3:
            f.write("✓ Strong positive correlation between entropy and error\n")
            f.write("  → Good calibration: high uncertainty predicts high error\n")
        elif metrics['pearson_correlation'] > 0.1:
            f.write("~ Moderate correlation between entropy and error\n")
            f.write("  → Some calibration present but could be improved\n")
        else:
            f.write("✗ Weak correlation between entropy and error\n")
            f.write("  → Poor calibration: uncertainty doesn't predict errors well\n")
        
        f.write("\n")
        
        if metrics['auroc'] > 0.7:
            f.write("✓ Good discrimination: entropy effectively identifies errors\n")
        elif metrics['auroc'] > 0.6:
            f.write("~ Moderate discrimination: entropy somewhat identifies errors\n")
        else:
            f.write("✗ Poor discrimination: entropy doesn't identify errors\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Phase 2 calibration results"
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        required=True,
        help="Path to results CSV from phase2_semantic_entropy.py"
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        required=True,
        help="Path to metrics JSON from phase2_semantic_entropy.py"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./phase2_visualizations",
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading results...")
    results_df = pd.read_csv(args.results_csv)
    
    with open(args.metrics_json, 'r') as f:
        metrics = json.load(f)
    
    print(f"Loaded {len(results_df)} results")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_entropy_vs_error(
        results_df,
        os.path.join(args.output_dir, "entropy_vs_error.png")
    )
    
    plot_entropy_distribution(
        results_df,
        os.path.join(args.output_dir, "entropy_distribution.png")
    )
    
    plot_clusters_analysis(
        results_df,
        os.path.join(args.output_dir, "clusters_analysis.png")
    )
    
    plot_roc_curve(
        results_df,
        os.path.join(args.output_dir, "roc_curve.png")
    )
    
    plot_calibration_reliability(
        results_df,
        os.path.join(args.output_dir, "calibration_reliability.png")
    )
    
    # Create summary report
    create_summary_report(
        results_df,
        metrics,
        os.path.join(args.output_dir, "summary_report.txt")
    )
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print(f"All outputs saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
