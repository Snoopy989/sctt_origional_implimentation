"""
Phase 2: Compare Training Paradigms
====================================

Statistical comparison of pairwise ranking vs direct regression
for semantic entropy calibration.
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


def load_results(results_path: str, metrics_path: str, name: str) -> tuple:
    """Load results and metrics for a single paradigm."""
    print(f"Loading {name}...")
    results_df = pd.read_csv(results_path)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return results_df, metrics


def statistical_comparison(
    reg_df: pd.DataFrame,
    pair_df: pd.DataFrame
) -> dict:
    """
    Perform statistical tests comparing the two paradigms.
    
    Tests:
    - Paired t-test for entropy differences
    - Mann-Whitney U test for error differences
    - Bootstrap confidence intervals for correlation differences
    """
    results = {}
    
    # Test 1: Compare semantic entropy distributions
    entropy_ttest = stats.ttest_ind(
        reg_df['semantic_entropy'],
        pair_df['semantic_entropy']
    )
    entropy_mannwhitney = stats.mannwhitneyu(
        reg_df['semantic_entropy'],
        pair_df['semantic_entropy']
    )
    
    results['entropy_ttest'] = {
        'statistic': entropy_ttest.statistic,
        'pvalue': entropy_ttest.pvalue
    }
    results['entropy_mannwhitney'] = {
        'statistic': entropy_mannwhitney.statistic,
        'pvalue': entropy_mannwhitney.pvalue
    }
    
    # Test 2: Compare prediction errors
    error_ttest = stats.ttest_ind(
        reg_df['error'],
        pair_df['error']
    )
    error_mannwhitney = stats.mannwhitneyu(
        reg_df['error'],
        pair_df['error']
    )
    
    results['error_ttest'] = {
        'statistic': error_ttest.statistic,
        'pvalue': error_ttest.pvalue
    }
    results['error_mannwhitney'] = {
        'statistic': error_mannwhitney.statistic,
        'pvalue': error_mannwhitney.pvalue
    }
    
    # Test 3: Compare correlations using Fisher's z-transformation
    reg_r = stats.pearsonr(reg_df['semantic_entropy'], reg_df['error'])[0]
    pair_r = stats.pearsonr(pair_df['semantic_entropy'], pair_df['error'])[0]
    
    n1, n2 = len(reg_df), len(pair_df)
    
    # Fisher z-transformation
    z1 = 0.5 * np.log((1 + reg_r) / (1 - reg_r))
    z2 = 0.5 * np.log((1 + pair_r) / (1 - pair_r))
    
    # Test statistic
    z_diff = (z1 - z2) / np.sqrt(1/(n1-3) + 1/(n2-3))
    p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
    
    results['correlation_comparison'] = {
        'regression_r': reg_r,
        'pairwise_r': pair_r,
        'z_statistic': z_diff,
        'pvalue': p_diff
    }
    
    return results


def plot_side_by_side_comparison(
    reg_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    output_path: str
):
    """Create side-by-side comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Entropy distributions
    axes[0, 0].hist(
        reg_df['semantic_entropy'],
        bins=30,
        alpha=0.6,
        label='Regression',
        edgecolor='black'
    )
    axes[0, 0].hist(
        pair_df['semantic_entropy'],
        bins=30,
        alpha=0.6,
        label='Pairwise',
        edgecolor='black'
    )
    axes[0, 0].set_xlabel('Semantic Entropy', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Entropy Distribution Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Error distributions
    axes[0, 1].hist(
        reg_df['error'],
        bins=30,
        alpha=0.6,
        label='Regression',
        edgecolor='black'
    )
    axes[0, 1].hist(
        pair_df['error'],
        bins=30,
        alpha=0.6,
        label='Pairwise',
        edgecolor='black'
    )
    axes[0, 1].set_xlabel('Absolute Error', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Error Distribution Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Entropy vs Error (Regression)
    axes[1, 0].scatter(
        reg_df['semantic_entropy'],
        reg_df['error'],
        alpha=0.5,
        s=30,
        label='Regression'
    )
    
    # Add regression line
    z = np.polyfit(reg_df['semantic_entropy'], reg_df['error'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(
        reg_df['semantic_entropy'].min(),
        reg_df['semantic_entropy'].max(),
        100
    )
    axes[1, 0].plot(x_line, p(x_line), "r--", linewidth=2)
    
    r, pval = stats.pearsonr(reg_df['semantic_entropy'], reg_df['error'])
    axes[1, 0].text(
        0.05, 0.95,
        f'r = {r:.3f}\np = {pval:.3e}',
        transform=axes[1, 0].transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    axes[1, 0].set_xlabel('Semantic Entropy', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Regression: Entropy vs Error', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Entropy vs Error (Pairwise)
    axes[1, 1].scatter(
        pair_df['semantic_entropy'],
        pair_df['error'],
        alpha=0.5,
        s=30,
        color='orange',
        label='Pairwise'
    )
    
    # Add regression line
    z = np.polyfit(pair_df['semantic_entropy'], pair_df['error'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(
        pair_df['semantic_entropy'].min(),
        pair_df['semantic_entropy'].max(),
        100
    )
    axes[1, 1].plot(x_line, p(x_line), "r--", linewidth=2)
    
    r, pval = stats.pearsonr(pair_df['semantic_entropy'], pair_df['error'])
    axes[1, 1].text(
        0.05, 0.95,
        f'r = {r:.3f}\np = {pval:.3e}',
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    axes[1, 1].set_xlabel('Semantic Entropy', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Pairwise: Entropy vs Error', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_comparison(
    reg_metrics: dict,
    pair_metrics: dict,
    output_path: str
):
    """Bar plot comparing key metrics."""
    metrics_to_compare = [
        ('pearson_correlation', 'Pearson r'),
        ('spearman_correlation', 'Spearman ρ'),
        ('auroc', 'AUROC'),
        ('mean_entropy', 'Mean Entropy'),
        ('mean_error', 'Mean Error')
    ]
    
    metric_names = [name for _, name in metrics_to_compare]
    reg_values = [reg_metrics[key] for key, _ in metrics_to_compare]
    pair_values = [pair_metrics[key] for key, _ in metrics_to_compare]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, reg_values, width, label='Regression', alpha=0.8)
    bars2 = ax.bar(x + width/2, pair_values, width, label='Pairwise', alpha=0.8)
    
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Calibration Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_comparison_report(
    reg_metrics: dict,
    pair_metrics: dict,
    stat_results: dict,
    output_path: str
):
    """Create comprehensive comparison report."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 2: TRAINING PARADIGM COMPARISON\n")
        f.write("Pairwise Ranking vs Direct Regression\n")
        f.write("="*80 + "\n\n")
        
        # Calibration metrics comparison
        f.write("CALIBRATION METRICS COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<30} {'Regression':<15} {'Pairwise':<15} {'Winner':<10}\n")
        f.write("-"*80 + "\n")
        
        # Pearson correlation (higher is better)
        reg_p = reg_metrics['pearson_correlation']
        pair_p = pair_metrics['pearson_correlation']
        winner = 'Regression' if reg_p > pair_p else 'Pairwise'
        f.write(f"{'Pearson Correlation':<30} {reg_p:<15.4f} {pair_p:<15.4f} {winner:<10}\n")
        
        # Spearman correlation (higher is better)
        reg_s = reg_metrics['spearman_correlation']
        pair_s = pair_metrics['spearman_correlation']
        winner = 'Regression' if reg_s > pair_s else 'Pairwise'
        f.write(f"{'Spearman Correlation':<30} {reg_s:<15.4f} {pair_s:<15.4f} {winner:<10}\n")
        
        # AUROC (higher is better)
        reg_a = reg_metrics['auroc']
        pair_a = pair_metrics['auroc']
        winner = 'Regression' if reg_a > pair_a else 'Pairwise'
        f.write(f"{'AUROC':<30} {reg_a:<15.4f} {pair_a:<15.4f} {winner:<10}\n")
        
        # Mean error (lower is better)
        reg_e = reg_metrics['mean_error']
        pair_e = pair_metrics['mean_error']
        winner = 'Regression' if reg_e < pair_e else 'Pairwise'
        f.write(f"{'Mean Absolute Error':<30} {reg_e:<15.4f} {pair_e:<15.4f} {winner:<10}\n")
        
        f.write("\n")
        
        # Statistical tests
        f.write("STATISTICAL SIGNIFICANCE TESTS\n")
        f.write("-"*80 + "\n")
        
        f.write("\n1. Entropy Distribution Comparison:\n")
        f.write(f"   T-test: t = {stat_results['entropy_ttest']['statistic']:.4f}, ")
        f.write(f"p = {stat_results['entropy_ttest']['pvalue']:.4e}\n")
        f.write(f"   Mann-Whitney U: U = {stat_results['entropy_mannwhitney']['statistic']:.0f}, ")
        f.write(f"p = {stat_results['entropy_mannwhitney']['pvalue']:.4e}\n")
        
        if stat_results['entropy_ttest']['pvalue'] < 0.05:
            f.write("   → Significant difference in entropy distributions\n")
        else:
            f.write("   → No significant difference in entropy distributions\n")
        
        f.write("\n2. Error Distribution Comparison:\n")
        f.write(f"   T-test: t = {stat_results['error_ttest']['statistic']:.4f}, ")
        f.write(f"p = {stat_results['error_ttest']['pvalue']:.4e}\n")
        f.write(f"   Mann-Whitney U: U = {stat_results['error_mannwhitney']['statistic']:.0f}, ")
        f.write(f"p = {stat_results['error_mannwhitney']['pvalue']:.4e}\n")
        
        if stat_results['error_ttest']['pvalue'] < 0.05:
            f.write("   → Significant difference in error distributions\n")
        else:
            f.write("   → No significant difference in error distributions\n")
        
        f.write("\n3. Correlation Comparison (Fisher's z):\n")
        f.write(f"   Regression r: {stat_results['correlation_comparison']['regression_r']:.4f}\n")
        f.write(f"   Pairwise r: {stat_results['correlation_comparison']['pairwise_r']:.4f}\n")
        f.write(f"   z = {stat_results['correlation_comparison']['z_statistic']:.4f}, ")
        f.write(f"p = {stat_results['correlation_comparison']['pvalue']:.4e}\n")
        
        if stat_results['correlation_comparison']['pvalue'] < 0.05:
            f.write("   → Significant difference in calibration quality\n")
        else:
            f.write("   → No significant difference in calibration quality\n")
        
        f.write("\n")
        
        # Overall conclusion
        f.write("="*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*80 + "\n")
        
        # Count wins
        reg_wins = 0
        pair_wins = 0
        
        if reg_p > pair_p:
            reg_wins += 1
        else:
            pair_wins += 1
            
        if reg_s > pair_s:
            reg_wins += 1
        else:
            pair_wins += 1
            
        if reg_a > pair_a:
            reg_wins += 1
        else:
            pair_wins += 1
            
        if reg_e < pair_e:
            reg_wins += 1
        else:
            pair_wins += 1
        
        f.write(f"\nMetric Wins: Regression = {reg_wins}, Pairwise = {pair_wins}\n\n")
        
        if reg_wins > pair_wins:
            f.write("✓ REGRESSION PARADIGM shows better calibration overall\n")
        elif pair_wins > reg_wins:
            f.write("✓ PAIRWISE PARADIGM shows better calibration overall\n")
        else:
            f.write("≈ BOTH PARADIGMS show similar calibration quality\n")
        
        f.write("\n")
        
        # Detailed interpretation
        f.write("Detailed Interpretation:\n")
        f.write("-"*80 + "\n")
        
        if reg_p > pair_p and stat_results['correlation_comparison']['pvalue'] < 0.05:
            f.write("• Regression shows significantly stronger entropy-error correlation\n")
            f.write("  → Better calibrated: uncertainty more reliably predicts errors\n")
        elif pair_p > reg_p and stat_results['correlation_comparison']['pvalue'] < 0.05:
            f.write("• Pairwise shows significantly stronger entropy-error correlation\n")
            f.write("  → Better calibrated: uncertainty more reliably predicts errors\n")
        else:
            f.write("• Both paradigms show similar entropy-error correlations\n")
            f.write("  → Comparable calibration quality\n")
        
        f.write("\n")
        
        if reg_e < pair_e and stat_results['error_ttest']['pvalue'] < 0.05:
            f.write("• Regression has significantly lower prediction errors\n")
            f.write("  → More accurate predictions overall\n")
        elif pair_e < reg_e and stat_results['error_ttest']['pvalue'] < 0.05:
            f.write("• Pairwise has significantly lower prediction errors\n")
            f.write("  → More accurate predictions overall\n")
        else:
            f.write("• Both paradigms show similar prediction accuracy\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare pairwise vs regression paradigms"
    )
    parser.add_argument(
        "--regression_results",
        type=str,
        required=True,
        help="Path to regression results CSV"
    )
    parser.add_argument(
        "--regression_metrics",
        type=str,
        required=True,
        help="Path to regression metrics JSON"
    )
    parser.add_argument(
        "--pairwise_results",
        type=str,
        required=True,
        help="Path to pairwise results CSV"
    )
    parser.add_argument(
        "--pairwise_metrics",
        type=str,
        required=True,
        help="Path to pairwise metrics JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./paradigm_comparison",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("PHASE 2: TRAINING PARADIGM COMPARISON")
    print("="*80)
    
    # Load results
    reg_df, reg_metrics = load_results(
        args.regression_results,
        args.regression_metrics,
        "Regression"
    )
    
    pair_df, pair_metrics = load_results(
        args.pairwise_results,
        args.pairwise_metrics,
        "Pairwise"
    )
    
    print(f"\nRegression samples: {len(reg_df)}")
    print(f"Pairwise samples: {len(pair_df)}")
    
    # Statistical comparison
    print("\nPerforming statistical tests...")
    stat_results = statistical_comparison(reg_df, pair_df)
    
    # Generate plots
    print("\nGenerating comparison plots...")
    
    plot_side_by_side_comparison(
        reg_df,
        pair_df,
        os.path.join(args.output_dir, "side_by_side_comparison.png")
    )
    
    plot_metrics_comparison(
        reg_metrics,
        pair_metrics,
        os.path.join(args.output_dir, "metrics_comparison.png")
    )
    
    # Create report
    create_comparison_report(
        reg_metrics,
        pair_metrics,
        stat_results,
        os.path.join(args.output_dir, "comparison_report.txt")
    )
    
    # Save statistical results
    stat_path = os.path.join(args.output_dir, "statistical_tests.json")
    with open(stat_path, 'w') as f:
        json.dump(stat_results, f, indent=2)
    print(f"Saved: {stat_path}")
    
    print("\n" + "="*80)
    print("Comparison complete!")
    print(f"All outputs saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
