"""
Phase 2: Calibration and Uncertainty Quantification via Semantic Entropy
=========================================================================

This script implements semantic entropy-based uncertainty quantification
following Kuhn et al. (2023) for both pairwise ranking and direct regression
models trained on SCTT data.

Author: [Your Name]
Date: February 2026
"""

import os
import re
import numpy as np
import pandas as pd
import torch
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
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional
import warnings
from tqdm import tqdm
import json
from collections import defaultdict
import argparse
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Set random seed for reproducibility
RANDOM_SEED = 42
set_seed(RANDOM_SEED)


class SCTTDataset(Dataset):
    """Dataset for SCTT responses with creativity ratings."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):  # Match training max_length
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Use ID if response_id doesn't exist
        response_id = row.get('response_id', row.get('ID', idx))
        return {
            'item': row['item'],
            'prompt': row['prompt'],
            'response': row['response'],
            'label': row['label'],  # Ground truth creativity score
            'jrt': row.get('jrt', row['label']),  # Keep jrt for analysis
            'response_id': response_id,
            'ID': row.get('ID', idx)
        }


class SemanticEntropyCalculator:
    """
    Implements semantic entropy calculation following Kuhn et al. (2023).
    Adapted for numerical regression outputs using DBSCAN clustering.
    
    Steps:
    1. Generation: M stochastic forward passes with temperature T
    2. Numerical Clustering: DBSCAN clustering on predicted scores
    3. Entropy Computation: Entropy over numerical clusters
    """
    
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
        max_length: int = 512
    ):
        """
        Initialize semantic entropy calculator.
        
        Args:
            model: Fine-tuned language model
            tokenizer: Model tokenizer
            num_generations: Number of stochastic generations (M)
            temperature: Sampling temperature (T)
            max_new_tokens: Maximum tokens to generate
            eps: DBSCAN epsilon (maximum distance for clustering)
            min_samples: DBSCAN minimum samples per cluster
            device: Device for computation
            max_length: Maximum sequence length for tokenization
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_generations = num_generations
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.eps = eps
        self.min_samples = min_samples
        self.device = device
        self.max_length = max_length
    
    def generate_sequences(
        self, 
        prompt: str, 
        instruction: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Generate M sequences using multinomial sampling.
        
        Args:
            prompt: Input prompt (full text in training format)
            instruction: Optional instruction for the model (not used, kept for compatibility)
            
        Returns:
            List of dictionaries with 'text' and 'log_prob' keys
        """
        # Use the prompt as-is (already in training format)
        full_prompt = prompt
        
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length  # Use dynamic max_length
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
                
                # Decode generated sequence
                generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                generated_text = self.tokenizer.decode(
                    generated_ids, 
                    skip_special_tokens=True
                ).strip()
                
                # Calculate average log probability
                scores = torch.stack(outputs.scores, dim=1)[0]  # [seq_len, vocab_size]
                probs = torch.softmax(scores, dim=-1)
                token_log_probs = torch.log(
                    probs[torch.arange(len(generated_ids)), generated_ids]
                )
                avg_log_prob = token_log_probs.mean().item()
                
                # Extract numerical score from text
                try:
                    numerical_score = float(generated_text.strip())
                except:
                    # If can't parse, try extracting first number
                    numbers = re.findall(r'-?\d+\.?\d*', generated_text)
                    numerical_score = float(numbers[0]) if numbers else 0.0
                
                sequences.append({
                    'text': generated_text,
                    'score': numerical_score,
                    'log_prob': avg_log_prob,
                    'tokens': generated_ids.cpu().tolist()
                })
        
        return sequences
    
    def cluster_sequences(
        self, 
        sequences: List[Dict[str, any]]
    ) -> List[List[int]]:
        """
        Cluster sequences based on numerical scores using DBSCAN.
        
        Args:
            sequences: List of generated sequences with 'score' field
            
        Returns:
            List of clusters (each cluster is a list of indices)
        """
        # Extract numerical scores
        scores = np.array([seq['score'] for seq in sequences]).reshape(-1, 1)
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(scores)
        
        # Convert labels to clusters (list of lists of indices)
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            if cluster_indices:  # Only add non-empty clusters
                clusters.append(cluster_indices)
        
        # If no clusters formed (all noise), treat each point as its own cluster
        if not clusters:
            clusters = [[i] for i in range(len(sequences))]
        
        return clusters
    
    def compute_semantic_entropy(
        self, 
        sequences: List[Dict[str, any]], 
        clusters: List[List[int]]
    ) -> float:
        """
        Compute semantic entropy over clusters.
        
        SE(x) ≈ -|C|^-1 * Σ log p(C_i | x)
        
        Args:
            sequences: Generated sequences with log probabilities
            clusters: Semantic clusters
            
        Returns:
            Semantic entropy value
        """
        if len(clusters) == 1:
            return 0.0
        
        # Calculate cluster probabilities
        cluster_probs = []
        total_prob = 0.0
        
        for cluster in clusters:
            # Sum probabilities of all sequences in cluster
            cluster_log_prob = np.log(sum(
                np.exp(sequences[idx]['log_prob']) 
                for idx in cluster
            ))
            cluster_probs.append(cluster_log_prob)
            total_prob += np.exp(cluster_log_prob)
        
        # Normalize and compute entropy
        cluster_probs = [np.exp(lp) / total_prob for lp in cluster_probs]
        
        # Semantic entropy
        se = -np.sum([
            p * np.log(p + 1e-10) 
            for p in cluster_probs if p > 0
        ])
        
        return se
    
    def calculate_entropy(
        self, 
        prompt: str, 
        instruction: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Full semantic entropy calculation pipeline.
        
        Args:
            prompt: Input prompt
            instruction: Optional instruction
            
        Returns:
            Dictionary with entropy and metadata
        """
        # Step 1: Generate sequences
        sequences = self.generate_sequences(prompt, instruction)
        
        # Step 2: Cluster sequences
        clusters = self.cluster_sequences(sequences)
        
        # Step 3: Compute semantic entropy
        semantic_entropy = self.compute_semantic_entropy(sequences, clusters)
        
        return {
            'semantic_entropy': semantic_entropy,
            'num_sequences': len(sequences),
            'num_clusters': len(clusters),
            'sequences': sequences,
            'clusters': clusters
        }


class ModelEvaluator:
    """Evaluates model predictions with uncertainty quantification."""
    
    def __init__(
        self,
        model,
        tokenizer,
        entropy_calculator: SemanticEntropyCalculator,
        device: str = "cuda",
        max_length: int = 512
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.entropy_calculator = entropy_calculator
        self.device = device
        self.max_length = max_length
        self.model.eval()
    
    def predict_creativity(
        self, 
        prompt: str, 
        response: str
    ) -> Dict[str, any]:
        """
        Get model prediction for creativity using regression.
        
        Args:
            prompt: SCTT prompt
            response: Response to evaluate
            
        Returns:
            Dictionary with prediction and uncertainty
        """
        # Match training format: "A creative research question for {prompt} is {response.lower()}"
        input_text = f"A creative research question for {prompt} is {response.lower()}"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length  # Use dynamic max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                num_return_sequences=1
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        try:
            prediction = float(generated_text)
        except:
            # Try to extract number from text
            numbers = re.findall(r'-?\d+\.?\d*', generated_text)
            prediction = float(numbers[0]) if numbers else 0.0
        
        # Calculate semantic entropy using the same training format
        entropy_result = self.entropy_calculator.calculate_entropy(
            prompt=input_text
        )
        
        return {
            'prediction': prediction,
            'semantic_entropy': entropy_result['semantic_entropy'],
            'num_clusters': entropy_result['num_clusters'],
            'entropy_metadata': entropy_result
        }
    
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        save_sequences: bool = False
    ) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """
        Evaluate entire dataset with uncertainty quantification.
        
        Args:
            dataloader: DataLoader for test data
            save_sequences: Whether to save generated sequences for re-clustering
            
        Returns:
            Tuple of (results DataFrame, sequences dict if save_sequences=True)
        """
        results = []
        sequences_data = {} if save_sequences else None
        
        # Calculate total number of items for progress bar
        total_items = sum(len(batch['prompt']) for batch in dataloader)
        
        # Reset dataloader iterator after counting
        with tqdm(total=total_items, desc="Evaluating") as pbar:
            for batch in dataloader:
                for i in range(len(batch['prompt'])):
                    prompt = batch['prompt'][i]
                    response = batch['response'][i]
                    label_true = batch['label'][i].item()  # Ground truth
                    jrt_value = batch['jrt'][i].item()  # Keep for reference
                    response_id = batch['response_id'][i]  # Now always a string from the list
                    
                    # Get prediction and uncertainty
                    result = self.predict_creativity(
                        prompt, response
                    )
                    
                    results.append({
                        'response_id': response_id,
                        'prompt': prompt,
                        'response': response,
                        'label_true': label_true,
                        'label_pred': result['prediction'],
                        'jrt': jrt_value,  # Keep for analysis
                        'semantic_entropy': result['semantic_entropy'],
                        'num_clusters': result['num_clusters'],
                        'error': abs(label_true - result['prediction'])
                    })
                    
                    # Save sequences for re-clustering analysis
                    if save_sequences:
                        sequences_data[str(response_id)] = {
                            'scores': [seq['score'] for seq in result['entropy_metadata']['sequences']],
                            'log_probs': [seq['log_prob'] for seq in result['entropy_metadata']['sequences']],
                            'texts': [seq['text'] for seq in result['entropy_metadata']['sequences']],
                            'clusters': result['entropy_metadata']['clusters']
                        }
                    
                    # Update progress bar after each item
                    pbar.update(1)
        
        return pd.DataFrame(results), sequences_data


def compute_calibration_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute calibration metrics following thesis specifications.
    
    Metrics:
    - Pearson correlation between entropy and absolute error
    - Spearman correlation between entropy and absolute error  
    - AUROC for predicting incorrect responses
    
    Args:
        results_df: DataFrame with predictions and uncertainties
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Define correctness threshold (can be adjusted)
    error_threshold = results_df['error'].median()
    results_df['is_correct'] = (results_df['error'] <= error_threshold).astype(int)
    
    # Pearson correlation: entropy vs absolute error
    pearson_corr, pearson_pval = pearsonr(
        results_df['semantic_entropy'],
        results_df['error']
    )
    
    # Spearman correlation: entropy vs absolute error
    spearman_corr, spearman_pval = spearmanr(
        results_df['semantic_entropy'],
        results_df['error']
    )
    
    # AUROC: predicting correctness from entropy
    # Higher entropy should predict incorrectness
    auroc = roc_auc_score(
        1 - results_df['is_correct'],  # Flip so high entropy = incorrect
        results_df['semantic_entropy']
    )
    
    metrics = {
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
    
    return metrics


def load_model_and_tokenizer(
    base_model_path: str,
    lora_adapter_path: Optional[str] = None,
    device: str = "cuda"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load fine-tuned model with optional LoRA adapter.
    
    Args:
        base_model_path: Path to base model
        lora_adapter_path: Path to LoRA adapter (if used)
        device: Device for model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from: {base_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use float32 for CPU, float16 for CUDA
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


def get_max_length(model):
    """Get maximum sequence length from model config"""
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


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Semantic Entropy Calibration"
    )
    parser.add_argument(
        "--use_combined_data",
        action="store_true",
        help="Use combined test and validation data"
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
        "--dbscan_eps",
        type=float,
        default=0.5,
        help="DBSCAN epsilon parameter for numerical clustering"
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=1,
        help="DBSCAN minimum samples per cluster"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="../data/test_items_sctt.csv",
        help="Path to test dataset CSV"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="../data/validation_items_sctt.csv",
        help="Path to validation dataset CSV"
    )
    parser.add_argument(
        "--sampling_fraction",
        type=float,
        default=0.2,
        help="Fraction of data to sample (stratified by jrt and prompt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./phase2_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=10,
        help="Number of stochastic generations (M)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature (T)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (increase for faster processing on GPU)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (cpu or cuda)"
    )
    parser.add_argument(
        "--save_sequences",
        action="store_true",
        help="Save generated sequences for re-clustering experiments"
    )
    
    args = parser.parse_args()
    
    # Validate required environment variables
    if args.base_model is None:
        raise ValueError(
            "MODEL_NAME not set! Please set MODEL_NAME in your .env file or provide --base_model argument."
        )
    
    # Resolve data paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from src_phase2
    
    # Convert relative paths to absolute paths based on project structure
    if not os.path.isabs(args.test_data):
        # Try relative to script first (../data/...), then relative to project root (data/...)
        test_path1 = os.path.join(script_dir, args.test_data)
        test_path2 = os.path.join(project_root, args.test_data.replace('../', ''))
        args.test_data = test_path1 if os.path.exists(test_path1) else test_path2
    
    if not os.path.isabs(args.val_data):
        val_path1 = os.path.join(script_dir, args.val_data)
        val_path2 = os.path.join(project_root, args.val_data.replace('../', ''))
        args.val_data = val_path1 if os.path.exists(val_path1) else val_path2
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*80)
    print("Phase 2: Calibration and Uncertainty Quantification (Regression Model)")
    print("="*80)
    print(f"Base Model: {args.base_model}")
    if args.lora_adapter:
        print(f"LoRA Adapter: {args.lora_adapter}")
    print(f"Test Data: {args.test_data}")
    print(f"Val Data: {args.val_data}")
    print(f"Sampling Fraction: {args.sampling_fraction * 100:.1f}%")
    print(f"Num Generations (M): {args.num_generations}")
    print(f"Temperature (T): {args.temperature}")
    print(f"DBSCAN eps: {args.dbscan_eps}")
    print(f"DBSCAN min_samples: {args.dbscan_min_samples}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    if args.use_combined_data:
        # Combine test and validation datasets
        df_test = pd.read_csv(args.test_data)
        df_val = pd.read_csv(args.val_data)
        df = pd.concat([df_test, df_val], ignore_index=True)
        print(f"Combined test ({len(df_test)}) and validation ({len(df_val)}) datasets")
    else:
        # Use only test data
        df = pd.read_csv(args.test_data)
        print(f"Using test dataset only")
    
    print(f"Total samples before sampling: {len(df)}")
    
    # Stratified sampling by jrt bins and prompt
    if args.sampling_fraction < 1.0:
        # Create jrt bins for stratification
        df['jrt_bin'] = pd.qcut(df['jrt'], q=5, labels=False, duplicates='drop')
        df['strata'] = df['prompt'].astype(str) + '_' + df['jrt_bin'].astype(str)
        
        # Stratified split
        try:
            df_sampled, _ = train_test_split(
                df,
                train_size=args.sampling_fraction,
                stratify=df['strata'],
                random_state=RANDOM_SEED
            )
            df = df_sampled.drop(columns=['jrt_bin', 'strata'])
            print(f"Stratified sampling complete: {len(df)} samples selected")
        except ValueError as e:
            # If stratification fails (e.g., too few samples in some strata), use simple random sampling
            print(f"Warning: Stratified sampling failed ({e}), using random sampling")
            df = df.sample(frac=args.sampling_fraction, random_state=RANDOM_SEED)
            df = df.drop(columns=['jrt_bin', 'strata'], errors='ignore')
    
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=RANDOM_SEED)
        print(f"Limited to {args.max_samples} samples")
    
    print(f"Final dataset size: {len(df)} samples")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(
        args.base_model,
        args.lora_adapter,
        args.device
    )
    
    # Get max_length dynamically from model (matching training script)
    max_length_divisor = 2
    max_length = int(get_max_length(model) / max_length_divisor)
    print(f"Using max_length: {max_length} (model max / {max_length_divisor})")
    
    # Initialize semantic entropy calculator
    print("\nInitializing semantic entropy calculator with DBSCAN clustering...")
    entropy_calculator = SemanticEntropyCalculator(
        model=model,
        tokenizer=tokenizer,
        num_generations=args.num_generations,
        temperature=args.temperature,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
        device=args.device,
        max_length=max_length  # Pass max_length to calculator
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model=model,
        tokenizer=tokenizer,
        entropy_calculator=entropy_calculator,
        device=args.device,
        max_length=max_length  # Pass max_length to evaluator
    )
    
    # Create dataset and dataloader
    dataset = SCTTDataset(df, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: {
            'prompt': [item['prompt'] for item in x],
            'response': [item['response'] for item in x],
            'label': torch.tensor([item['label'] for item in x]),  # Ground truth
            'jrt': torch.tensor([item['jrt'] for item in x]),  # Keep for analysis
            'response_id': [item['response_id'] for item in x]  # Keep as list, not tensor
        }
    )
    
    # Evaluate
    print("\nRunning evaluation with semantic entropy...")
    results_df, sequences_data = evaluator.evaluate_dataset(
        dataloader, 
        save_sequences=args.save_sequences
    )
    
    # Compute metrics
    print("\nComputing calibration metrics...")
    metrics = compute_calibration_metrics(results_df)
    
    # Print results
    print("\n" + "="*80)
    print("CALIBRATION RESULTS")
    print("="*80)
    print(f"Pearson Correlation (entropy vs error): {metrics['pearson_correlation']:.4f}")
    print(f"  p-value: {metrics['pearson_pvalue']:.4e}")
    print(f"Spearman Correlation (entropy vs error): {metrics['spearman_correlation']:.4f}")
    print(f"  p-value: {metrics['spearman_pvalue']:.4e}")
    print(f"AUROC (predicting incorrectness): {metrics['auroc']:.4f}")
    print(f"\nMean Semantic Entropy: {metrics['mean_entropy']:.4f} (±{metrics['std_entropy']:.4f})")
    print(f"Mean Absolute Error: {metrics['mean_error']:.4f}")
    print(f"Mean Number of Clusters: {metrics['mean_clusters']:.2f}")
    print("="*80)
    
    # Save results
    results_path = os.path.join(
        args.output_dir, 
        f"results_regression_{timestamp}.csv"
    )
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    metrics_path = os.path.join(
        args.output_dir,
        f"metrics_regression_{timestamp}.json"
    )
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Save configuration
    config = vars(args)
    config['timestamp'] = timestamp
    config['num_samples'] = len(df)
    config_path = os.path.join(
        args.output_dir,
        f"config_regression_{timestamp}.json"
    )
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    # Save sequences if requested
    if args.save_sequences and sequences_data:
        sequences_path = os.path.join(
            args.output_dir,
            f"sequences_regression_{timestamp}.json"
        )
        with open(sequences_path, 'w') as f:
            json.dump(sequences_data, f, indent=2)
        print(f"Sequences saved to: {sequences_path}")
        print(f"  ({len(sequences_data)} samples × {args.num_generations} generations each)")


if __name__ == "__main__":
    main()
