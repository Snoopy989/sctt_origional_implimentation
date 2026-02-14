"""
Phase 2: Convergence Analysis Script
====================================

Analyzes how semantic entropy metrics converge with different numbers 
of stochastic generations (n). Tests n=10, 20, 30, 50 to determine 
optimal number of generations for stable uncertainty estimation.

This helps answer: Is weak AUROC due to insufficient generations, 
or is the signal genuinely weak?
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed
)
from peft import PeftModel
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Optional
import warnings
from tqdm import tqdm
import json
from datetime import datetime
from dotenv import load_dotenv
import re

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

set_seed(42)


class SCTTDataset(Dataset):
    """Dataset for SCTT responses with creativity ratings."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 2048):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        response_id = row.get('response_id', row.get('ID', idx))
        return {
            'item': row['item'],
            'prompt': row['prompt'],
            'response': row['response'],
            'label': row['label'],
            'jrt': row.get('jrt', row['label']),
            'response_id': response_id,
            'ID': row.get('ID', idx)
        }


class SemanticEntropyCalculator:
    """Semantic entropy calculator with configurable num_generations."""
    
    def __init__(
        self,
        model,
        tokenizer,
        num_generations: int = 10,
        temperature: float = 0.5,
        max_new_tokens: int = 128,
        eps: float = 0.5,
        min_samples: int = 1,
        device: str = "cuda",
        max_length: int = 2048
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_generations = num_generations
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.eps = eps
        self.min_samples = min_samples
        self.device = device
        self.max_length = max_length
    
    def generate_sequences(self, prompt: str) -> List[Dict[str, any]]:
        """Generate M sequences using multinomial sampling."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        sequences = []
        
        with torch.no_grad():
            for _ in range(self.num_generations):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.95,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                
                generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                generated_text = self.tokenizer.decode(
                    generated_ids, 
                    skip_special_tokens=True
                ).strip()
                
                scores = torch.stack(outputs.scores, dim=1)[0]
                probs = torch.softmax(scores, dim=-1)
                token_log_probs = torch.log(
                    probs[torch.arange(len(generated_ids)), generated_ids]
                )
                avg_log_prob = token_log_probs.mean().item()
                
                try:
                    numerical_score = float(generated_text.strip())
                except:
                    numbers = re.findall(r'-?\d+\.?\d*', generated_text)
                    numerical_score = float(numbers[0]) if numbers else 0.0
                
                sequences.append({
                    'text': generated_text,
                    'score': numerical_score,
                    'log_prob': avg_log_prob,
                    'tokens': generated_ids.cpu().tolist()
                })
        
        return sequences
    
    def cluster_sequences(self, sequences: List[Dict[str, any]]) -> List[List[int]]:
        """Cluster sequences using DBSCAN."""
        scores = np.array([seq['score'] for seq in sequences]).reshape(-1, 1)
        
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(scores)
        
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            if cluster_indices:
                clusters.append(cluster_indices)
        
        if not clusters:
            clusters = [[i] for i in range(len(sequences))]
        
        return clusters
    
    def compute_semantic_entropy(
        self, 
        sequences: List[Dict[str, any]], 
        clusters: List[List[int]]
    ) -> float:
        """Compute semantic entropy over clusters."""
        if len(clusters) == 1:
            return 0.0
        
        cluster_probs = []
        total_prob = 0.0
        
        for cluster in clusters:
            cluster_log_prob = np.log(sum(
                np.exp(sequences[idx]['log_prob']) 
                for idx in cluster
            ))
            cluster_probs.append(cluster_log_prob)
            total_prob += np.exp(cluster_log_prob)
        
        cluster_probs = [np.exp(lp) / total_prob for lp in cluster_probs]
        
        se = -np.sum([
            p * np.log(p + 1e-10) 
            for p in cluster_probs if p > 0
        ])
        
        return se
    
    def calculate_entropy(self, prompt: str) -> Dict[str, any]:
        """Full semantic entropy calculation pipeline."""
        sequences = self.generate_sequences(prompt)
        clusters = self.cluster_sequences(sequences)
        semantic_entropy = self.compute_semantic_entropy(sequences, clusters)
        
        return {
            'semantic_entropy': semantic_entropy,
            'num_sequences': len(sequences),
            'num_clusters': len(clusters),
            'sequences': sequences,
            'clusters': clusters
        }


def predict_creativity(
    model, 
    tokenizer, 
    entropy_calculator: SemanticEntropyCalculator,
    prompt: str, 
    response: str,
    device: str = "cuda",
    max_length: int = 2048
) -> Dict[str, any]:
    """Get model prediction with uncertainty."""
    input_text = f"A creative research question for {prompt} is {response.lower()}"
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            num_return_sequences=1
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    try:
        prediction = float(generated_text)
    except:
        numbers = re.findall(r'-?\d+\.?\d*', generated_text)
        prediction = float(numbers[0]) if numbers else 0.0
    
    entropy_result = entropy_calculator.calculate_entropy(prompt=input_text)
    
    return {
        'prediction': prediction,
        'semantic_entropy': entropy_result['semantic_entropy'],
        'num_clusters': entropy_result['num_clusters'],
        'entropy_metadata': entropy_result
    }


def compute_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """Compute calibration metrics."""
    error_threshold = results_df['error'].median()
    results_df['is_correct'] = (results_df['error'] <= error_threshold).astype(int)
    
    pearson_corr, pearson_pval = pearsonr(
        results_df['semantic_entropy'],
        results_df['error']
    )
    
    spearman_corr, spearman_pval = spearmanr(
        results_df['semantic_entropy'],
        results_df['error']
    )
    
    auroc = roc_auc_score(
        1 - results_df['is_correct'],
        results_df['semantic_entropy']
    )
    
    return {
        'pearson_correlation': pearson_corr,
        'pearson_pvalue': pearson_pval,
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_pval,
        'auroc': auroc,
        'mean_entropy': results_df['semantic_entropy'].mean(),
        'std_entropy': results_df['semantic_entropy'].std(),
        'mean_error': results_df['error'].mean(),
        'mean_clusters': results_df['num_clusters'].mean()
    }


def get_max_length(model):
    """Get maximum sequence length from model config."""
    conf = model.base_model.config if hasattr(model, 'base_model') else model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(conf, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def load_model_and_tokenizer(
    base_model_path: str,
    lora_adapter_path: Optional[str] = None,
    device: str = "cuda"
) -> Tuple:
    """Load model and tokenizer."""
    print(f"Loading model from: {base_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dtype = torch.float32 if device == "cpu" else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map={"": device} if device == "cpu" else "auto"
    )
    
    if lora_adapter_path:
        print(f"Loading LoRA adapter from: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model = model.merge_and_unload()
    
    model.eval()
    return model, tokenizer


def run_convergence_analysis(
    model,
    tokenizer,
    df: pd.DataFrame,
    n_values: List[int] = [10, 20, 30, 50],
    device: str = "cuda",
    max_length: int = 2048,
    output_dir: str = "./phase2_results"
):
    """Run convergence analysis across different n values."""
    
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS: Testing different numbers of generations")
    print("="*80)
    print(f"Dataset size: {len(df)} samples")
    print(f"Testing n values: {n_values}")
    print("="*80 + "\n")
    
    all_results = {}
    
    for n in n_values:
        print(f"\n{'='*80}")
        print(f"Running with n={n} generations...")
        print(f"{'='*80}")
        
        # Initialize entropy calculator with current n
        entropy_calculator = SemanticEntropyCalculator(
            model=model,
            tokenizer=tokenizer,
            num_generations=n,
            temperature=0.5,
            eps=0.5,
            min_samples=1,
            device=device,
            max_length=max_length
        )
        
        results = []
        
        with tqdm(total=len(df), desc=f"n={n}") as pbar:
            for idx, row in df.iterrows():
                prompt = row['prompt']
                response = row['response']
                label_true = row['label']
                
                result = predict_creativity(
                    model, tokenizer, entropy_calculator,
                    prompt, response, device, max_length
                )
                
                results.append({
                    'prompt': prompt,
                    'response': response,
                    'label_true': label_true,
                    'label_pred': result['prediction'],
                    'semantic_entropy': result['semantic_entropy'],
                    'num_clusters': result['num_clusters'],
                    'error': abs(label_true - result['prediction'])
                })
                
                pbar.update(1)
        
        results_df = pd.DataFrame(results)
        metrics = compute_metrics(results_df)
        
        all_results[n] = {
            'metrics': metrics,
            'results_df': results_df
        }
        
        print(f"\nResults for n={n}:")
        print(f"  Pearson correlation: {metrics['pearson_correlation']:.4f} (p={metrics['pearson_pvalue']:.4e})")
        print(f"  Spearman correlation: {metrics['spearman_correlation']:.4f} (p={metrics['spearman_pvalue']:.4e})")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  Mean entropy: {metrics['mean_entropy']:.4f} (±{metrics['std_entropy']:.4f})")
        print(f"  Mean clusters: {metrics['mean_clusters']:.2f}")
    
    return all_results


def plot_convergence(all_results: Dict, output_dir: str = "./phase2_results"):
    """Plot convergence analysis results."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    n_values = sorted(all_results.keys())
    
    # Extract metrics
    pearson_corr = [all_results[n]['metrics']['pearson_correlation'] for n in n_values]
    spearman_corr = [all_results[n]['metrics']['spearman_correlation'] for n in n_values]
    auroc = [all_results[n]['metrics']['auroc'] for n in n_values]
    mean_entropy = [all_results[n]['metrics']['mean_entropy'] for n in n_values]
    std_entropy = [all_results[n]['metrics']['std_entropy'] for n in n_values]
    mean_clusters = [all_results[n]['metrics']['mean_clusters'] for n in n_values]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Convergence Analysis: Effect of Number of Generations (n)', fontsize=16, fontweight='bold')
    
    # Plot 1: Pearson Correlation
    axes[0, 0].plot(n_values, pearson_corr, 'o-', linewidth=2, markersize=8, color='steelblue')
    axes[0, 0].set_xlabel('Number of Generations (n)', fontweight='bold')
    axes[0, 0].set_ylabel('Pearson Correlation', fontweight='bold')
    axes[0, 0].set_title('Entropy vs Error Correlation (Pearson)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: Spearman Correlation
    axes[0, 1].plot(n_values, spearman_corr, 'o-', linewidth=2, markersize=8, color='darkgreen')
    axes[0, 1].set_xlabel('Number of Generations (n)', fontweight='bold')
    axes[0, 1].set_ylabel('Spearman Correlation', fontweight='bold')
    axes[0, 1].set_title('Entropy vs Error Correlation (Spearman)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 3: AUROC
    axes[0, 2].plot(n_values, auroc, 'o-', linewidth=2, markersize=8, color='darkred')
    axes[0, 2].set_xlabel('Number of Generations (n)', fontweight='bold')
    axes[0, 2].set_ylabel('AUROC', fontweight='bold')
    axes[0, 2].set_title('AUROC (Predicting Incorrectness)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    axes[0, 2].legend()
    
    # Plot 4: Mean Entropy
    axes[1, 0].errorbar(n_values, mean_entropy, yerr=std_entropy, fmt='o-', linewidth=2, 
                        markersize=8, color='purple', capsize=5, capthick=2)
    axes[1, 0].set_xlabel('Number of Generations (n)', fontweight='bold')
    axes[1, 0].set_ylabel('Mean Semantic Entropy', fontweight='bold')
    axes[1, 0].set_title('Entropy Stability (Mean ± Std)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Std Entropy
    axes[1, 1].plot(n_values, std_entropy, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1, 1].set_xlabel('Number of Generations (n)', fontweight='bold')
    axes[1, 1].set_ylabel('Std Dev of Entropy', fontweight='bold')
    axes[1, 1].set_title('Entropy Variability')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Mean Clusters
    axes[1, 2].plot(n_values, mean_clusters, 'o-', linewidth=2, markersize=8, color='teal')
    axes[1, 2].set_xlabel('Number of Generations (n)', fontweight='bold')
    axes[1, 2].set_ylabel('Mean Number of Clusters', fontweight='bold')
    axes[1, 2].set_title('Clustering Behavior')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"convergence_analysis_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Also save a summary table
    summary_df = pd.DataFrame({
        'n': n_values,
        'pearson_r': pearson_corr,
        'spearman_r': spearman_corr,
        'auroc': auroc,
        'mean_entropy': mean_entropy,
        'std_entropy': std_entropy,
        'mean_clusters': mean_clusters
    })
    
    summary_path = os.path.join(output_dir, f"convergence_summary_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to: {summary_path}")
    
    return summary_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convergence Analysis: Test different numbers of generations"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=os.getenv('MODEL_NAME'),
        help="Base model path"
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="../data/test_items_sctt.csv",
        help="Path to test dataset CSV"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=250,
        help="Number of samples to use (default: 250)"
    )
    parser.add_argument(
        "--n_values",
        type=int,
        nargs='+',
        default=[10, 20, 30, 50],
        help="List of n values to test (default: 10 20 30 50)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./phase2_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (cpu or cuda)"
    )
    
    args = parser.parse_args()
    
    # Validate
    if args.base_model is None:
        raise ValueError("MODEL_NAME not set! Set in .env or use --base_model")
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if not os.path.isabs(args.test_data):
        test_path1 = os.path.join(script_dir, args.test_data)
        test_path2 = os.path.join(project_root, args.test_data.replace('../', ''))
        args.test_data = test_path1 if os.path.exists(test_path1) else test_path2
    
    # Resolve output_dir path relative to project root
    if not os.path.isabs(args.output_dir):
        # If relative path like "./phase2_results", resolve from project root
        output_path_from_root = os.path.join(project_root, args.output_dir.replace('./', '').replace('.\\', ''))
        args.output_dir = output_path_from_root
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)
    print(f"Base Model: {args.base_model}")
    if args.lora_adapter:
        print(f"LoRA Adapter: {args.lora_adapter}")
    print(f"Test Data: {args.test_data}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Number of Samples: {args.num_samples}")
    print(f"n values to test: {args.n_values}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(args.test_data)
    
    # Random sample
    if len(df) > args.num_samples:
        df = df.sample(n=args.num_samples, random_state=42)
        print(f"Randomly sampled {args.num_samples} samples")
    
    print(f"Final dataset size: {len(df)} samples")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(
        args.base_model,
        args.lora_adapter,
        args.device
    )
    
    max_length = int(get_max_length(model) / 2)
    print(f"Using max_length: {max_length}")
    
    # Run convergence analysis
    all_results = run_convergence_analysis(
        model,
        tokenizer,
        df,
        n_values=args.n_values,
        device=args.device,
        max_length=max_length,
        output_dir=args.output_dir
    )
    
    # Plot results
    print("\n" + "="*80)
    print("Generating convergence plots...")
    print("="*80)
    summary_df = plot_convergence(all_results, args.output_dir)
    
    # Print interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    n_values = sorted(all_results.keys())
    auroc_values = [all_results[n]['metrics']['auroc'] for n in n_values]
    
    # Check for improvement
    auroc_improvement = auroc_values[-1] - auroc_values[0]
    
    print(f"\nAUROC progression: {' → '.join([f'{a:.3f}' for a in auroc_values])}")
    print(f"Total improvement: {auroc_improvement:+.3f}")
    
    if auroc_improvement > 0.05:
        print("\n✓ METRICS IMPROVE with more generations")
        print("  → Increase n to 30-50 for better calibration")
        print("  → The weak signal may be due to insufficient sampling")
    elif abs(auroc_improvement) < 0.02:
        print("\n→ METRICS STABLE across all n values")
        print("  → n=10 is sufficient")
        print("  → The current signal is real (not noise from insufficient generations)")
    else:
        print("\n? METRICS VARY but no clear trend")
        print("  → May need more samples to determine optimal n")
    
    # Save final summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_summary = {
        'analysis_date': timestamp,
        'num_samples': len(df),
        'n_values_tested': args.n_values,
        'auroc_by_n': {n: all_results[n]['metrics']['auroc'] for n in n_values},
        'pearson_by_n': {n: all_results[n]['metrics']['pearson_correlation'] for n in n_values},
        'interpretation': {
            'auroc_improvement': auroc_improvement,
            'recommended_n': n_values[auroc_values.index(max(auroc_values))]
        }
    }
    
    summary_json_path = os.path.join(args.output_dir, f"convergence_analysis_{timestamp}.json")
    with open(summary_json_path, 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"\nFull analysis saved to: {summary_json_path}")
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
