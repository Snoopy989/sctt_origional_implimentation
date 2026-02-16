"""
Phase 2: Convergence Analysis Script (Refactored)
Analyzes semantic entropy convergence across different numbers of stochastic generations.
Tests n=10-200 to determine optimal generations for stable uncertainty estimation.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForSequenceClassification,
    AutoConfig,
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
load_dotenv()
set_seed(42)


def configure_logger(output_dir: str) -> logging.Logger:
    """Configure logging to file and console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    log_file = os.path.join(output_dir, 'convergence_analysis.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


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
        max_new_tokens: int = 10,
        eps: float = 0.5,
        min_samples: int = 1,
        device: str = "cuda",
        max_length: int = 2048,
        generation_batch_size: int = 20
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
        self.generation_batch_size = generation_batch_size  # Batch size for generation to avoid OOM
    
    def generate_sequences(self, prompt: str) -> List[Dict[str, any]]:
        """Generate M sequences using multinomial sampling with batching to avoid OOM."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        sequences = []
        
        # Generate in batches to avoid OOM for large num_generations
        num_batches = (self.num_generations + self.generation_batch_size - 1) // self.generation_batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.generation_batch_size
                batch_end = min(batch_start + self.generation_batch_size, self.num_generations)
                batch_size = batch_end - batch_start
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.95,
                    num_return_sequences=batch_size,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                
                input_length = inputs.input_ids.shape[1]
                
                for i in range(batch_size):
                    generated_ids = outputs.sequences[i][input_length:]
                    generated_text = self.tokenizer.decode(
                        generated_ids, 
                        skip_special_tokens=True
                    ).strip()
                    
                    scores = torch.stack(outputs.scores, dim=1)
                    probs = torch.softmax(scores[i], dim=-1)
                    
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
                
                # Clear GPU cache between batches to avoid memory fragmentation
                if self.device == "cuda" and batch_idx < num_batches - 1:
                    torch.cuda.empty_cache()
        
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


class ModelPredictor:
    """Handles model predictions with uncertainty quantification."""
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_length: int = 2048,
        uncertainty_threshold: float = 0.5
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.uncertainty_threshold = uncertainty_threshold
    
    def predict(
        self,
        entropy_calculator: SemanticEntropyCalculator,
        prompt: str,
        response: str,
        formatted_text: str = None
    ) -> Dict[str, any]:
        """Get model prediction with uncertainty."""
        # Use pre-formatted text from CSV if available (matches training format exactly)
        if formatted_text:
            input_text = formatted_text
        else:
            input_text = f"A creative research question for {prompt} is {response.lower()}"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # FIXED: Direct regression prediction using logits, not text generation
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = outputs.logits.squeeze().item()
        
        # Calculate semantic entropy for uncertainty estimation
        entropy_result = entropy_calculator.calculate_entropy(prompt=input_text)
        semantic_entropy = entropy_result['semantic_entropy']
        
        margin = 2 * semantic_entropy
        confidence_interval = {
            'lower': prediction - margin,
            'upper': prediction + margin,
            'width': 2 * margin
        }
        
        needs_review = semantic_entropy >= self.uncertainty_threshold
        if confidence_interval['width'] > 3.0:
            needs_review = True
        
        return {
            'prediction': prediction,
            'confidence_interval': confidence_interval,
            'semantic_entropy': semantic_entropy,
            'num_clusters': entropy_result['num_clusters'],
            'needs_review': needs_review,
            'entropy_metadata': entropy_result
        }


class MetricsCalculator:
    """Computes evaluation metrics."""
    
    @staticmethod
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


class ModelLoader:
    """Handles model and tokenizer loading."""
    
    @staticmethod
    def get_max_length(model) -> int:
        """Get maximum sequence length from model config."""
        conf = model.base_model.config if hasattr(model, 'base_model') else model.config
        max_length = None
        for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
            max_length = getattr(conf, length_setting, None)
            if max_length:
                break
        return max_length if max_length else 1024
    
    @staticmethod
    def load_generative_model(
        base_model_path: str,
        device: str = "cuda"
    ) -> Tuple:
        """Load generative model for semantic entropy calculation."""
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dtype = torch.float32 if device == "cpu" else torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map={"": device} if device == "cpu" else "auto",
            use_cache=True,
            low_cpu_mem_usage=True
        )
        
        model.eval()
        return model, tokenizer
    
    @staticmethod
    def load_model_and_tokenizer(
        base_model_path: str,
        lora_adapter_path: Optional[str] = None,
        device: str = "cuda"
    ) -> Tuple:
        """Load model and tokenizer with GPU optimizations."""
        # Setup config for regression (CRITICAL for fine-tuned model)
        config = AutoConfig.from_pretrained(base_model_path)
        config.num_labels = 1
        config.problem_type = "regression"
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dtype = torch.float32 if device == "cpu" else torch.float16
        
        # FIXED: Use LlamaForSequenceClassification for regression, not CausalLM
        model = LlamaForSequenceClassification.from_pretrained(
            base_model_path,
            config=config,
            torch_dtype=dtype,
            device_map={"": device} if device == "cpu" else "auto",
            low_cpu_mem_usage=True
        )
        
        if lora_adapter_path:
            model = PeftModel.from_pretrained(model, lora_adapter_path)
            model = model.merge_and_unload()
        
        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        return model, tokenizer


class ConvergenceAnalyzer:
    """Main class for convergence analysis."""
    
    def __init__(
        self,
        model,
        tokenizer,
        output_dir: str,
        logger: logging.Logger,
        generative_model=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generative_model = generative_model if generative_model is not None else model
        self.output_dir = output_dir
        self.logger = logger
        self.max_length = int(ModelLoader.get_max_length(model) / 2)
    
    def run_analysis(
        self,
        df: pd.DataFrame,
        n_values: List[int],
        device: str = "cuda",
        uncertainty_threshold: float = 0.5
    ) -> Dict:
        """Run convergence analysis across different n values."""
        self.logger.info(f"Starting convergence analysis with n_values: {n_values}")
        self.logger.info(f"Dataset size: {len(df)} samples")
        
        all_results = {}
        predictor = ModelPredictor(
            self.model,
            self.tokenizer,
            device,
            self.max_length,
            uncertainty_threshold
        )
        
        for n in n_values:
            self.logger.info(f"Processing n={n}")
            
            # Use generative model for entropy calculation
            entropy_calculator = SemanticEntropyCalculator(
                model=self.generative_model,
                tokenizer=self.tokenizer,
                num_generations=n,
                temperature=0.5,
                eps=0.5,
                min_samples=1,
                device=device,
                max_length=self.max_length
            )
            
            results = []
            flagged_for_review = []
            
            with tqdm(total=len(df), desc=f"n={n}", leave=False) as pbar:
                for idx, row in df.iterrows():
                    prompt = row['prompt']
                    response = row['response']
                    label_true = row['label']
                    formatted_text = row.get('text', None)  # Use pre-formatted text from CSV
                    
                    result = predictor.predict(entropy_calculator, prompt, response, formatted_text)
                    
                    result_record = {
                        'sample_id': row.get('ID', idx),
                        'item': row.get('item', ''),
                        'prompt': prompt,
                        'response': response,
                        'label_true': label_true,
                        'label_pred': result['prediction'],
                        'ci_lower': result['confidence_interval']['lower'],
                        'ci_upper': result['confidence_interval']['upper'],
                        'ci_width': result['confidence_interval']['width'],
                        'semantic_entropy': result['semantic_entropy'],
                        'num_clusters': result['num_clusters'],
                        'error': abs(label_true - result['prediction']),
                        'needs_review': result['needs_review'],
                        'entropy_metadata': result['entropy_metadata']  # Save full metadata
                    }
                    
                    results.append(result_record)
                    if result['needs_review']:
                        flagged_for_review.append(result_record)
                    
                    pbar.update(1)
            
            results_df = pd.DataFrame(results)
            metrics = MetricsCalculator.compute_metrics(results_df)
            
            review_stats = {
                'total_samples': len(results_df),
                'flagged_count': len(flagged_for_review),
                'flagged_percentage': 100 * len(flagged_for_review) / len(results_df) if len(results_df) > 0 else 0,
                'avg_error_flagged': np.mean([r['error'] for r in flagged_for_review]) if flagged_for_review else 0,
                'avg_error_not_flagged': np.mean([r['error'] for r in results if not r['needs_review']]) if results else 0,
                'avg_entropy_flagged': np.mean([r['semantic_entropy'] for r in flagged_for_review]) if flagged_for_review else 0,
                'avg_entropy_not_flagged': np.mean([r['semantic_entropy'] for r in results if not r['needs_review']]) if results else 0
            }
            
            all_results[n] = {
                'metrics': metrics,
                'results_df': results_df,
                'review_stats': review_stats,
                'flagged_samples': flagged_for_review,
                'raw_results': results  # Keep raw results with metadata
            }
            
            self._save_results(n, results_df, flagged_for_review)
            self._save_comprehensive_results(n, results)  # Save all sequence-level data
            self._log_metrics(n, metrics, review_stats)
        
        # Save combined dataset across all n values
        self._save_combined_comprehensive_results(all_results)
        
        self.logger.info("Convergence analysis completed")
        return all_results
    
    def _save_results(self, n: int, results_df: pd.DataFrame, flagged_for_review: List[Dict]):
        """Save results to CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary results (without metadata)
        summary_df = results_df.drop(columns=['entropy_metadata'], errors='ignore')
        detailed_path = os.path.join(self.output_dir, f'summary_results_n{n}_{timestamp}.csv')
        summary_df.to_csv(detailed_path, index=False)
        self.logger.debug(f"Saved summary results to: {detailed_path}")
        
        if flagged_for_review:
            flagged_df = pd.DataFrame(flagged_for_review)
            flagged_df = flagged_df.drop(columns=['entropy_metadata'], errors='ignore')
            flagged_path = os.path.join(self.output_dir, f'flagged_for_review_n{n}_{timestamp}.csv')
            flagged_df.to_csv(flagged_path, index=False)
            self.logger.debug(f"Saved flagged samples to: {flagged_path}")
    
    def _save_comprehensive_results(self, n: int, results: List[Dict]):
        """Save comprehensive results in long format for advanced analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Long format: one row per generated sequence
        sequence_data = []
        
        for result in results:
            sample_id = result['sample_id']
            item = result.get('item', '')
            prompt = result['prompt']
            response = result['response']
            label_true = result['label_true']
            label_pred = result['label_pred']
            semantic_entropy = result['semantic_entropy']
            num_clusters = result['num_clusters']
            error = result['error']
            needs_review = result['needs_review']
            
            metadata = result.get('entropy_metadata', {})
            sequences = metadata.get('sequences', [])
            clusters = metadata.get('clusters', [])
            
            # Determine cluster membership for each sequence
            sequence_to_cluster = {}
            for cluster_id, cluster_indices in enumerate(clusters):
                for seq_idx in cluster_indices:
                    sequence_to_cluster[seq_idx] = cluster_id
            
            # Add greedy prediction as sequence index -1
            sequence_data.append({
                'sample_id': sample_id,
                'item': item,
                'prompt': prompt,
                'response': response,
                'label_true': label_true,
                'n_generations': n,
                'sequence_idx': -1,
                'sequence_text': response,  # Original response
                'sequence_score': label_pred,  # Greedy prediction
                'sequence_log_prob': None,
                'cluster_id': None,
                'is_greedy': True,
                'semantic_entropy': semantic_entropy,
                'num_clusters': num_clusters,
                'error': error,
                'needs_review': needs_review
            })
            
            # Add all sampled sequences
            for seq_idx, seq in enumerate(sequences):
                sequence_data.append({
                    'sample_id': sample_id,
                    'item': item,
                    'prompt': prompt,
                    'response': response,
                    'label_true': label_true,
                    'n_generations': n,
                    'sequence_idx': seq_idx,
                    'sequence_text': seq.get('text', ''),
                    'sequence_score': seq.get('score', None),
                    'sequence_log_prob': seq.get('log_prob', None),
                    'cluster_id': sequence_to_cluster.get(seq_idx, -1),
                    'is_greedy': False,
                    'semantic_entropy': semantic_entropy,
                    'num_clusters': num_clusters,
                    'error': error,
                    'needs_review': needs_review
                })
        
        # Save as both CSV and Parquet for flexibility
        df_sequences = pd.DataFrame(sequence_data)
        
        csv_path = os.path.join(self.output_dir, f'sequences_n{n}_{timestamp}.csv')
        df_sequences.to_csv(csv_path, index=False)
        self.logger.info(f"Saved sequence-level data to: {csv_path}")
        
        try:
            parquet_path = os.path.join(self.output_dir, f'sequences_n{n}_{timestamp}.parquet')
            df_sequences.to_parquet(parquet_path, index=False)
            self.logger.info(f"Saved sequence-level data to: {parquet_path}")
        except Exception as e:
            self.logger.warning(f"Could not save parquet (install pyarrow): {e}")
    
    def _save_combined_comprehensive_results(self, all_results: Dict):
        """Save combined dataset across all n values for comparative analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        combined_data = []
        
        for n, results_dict in all_results.items():
            raw_results = results_dict.get('raw_results', [])
            
            for result in raw_results:
                sample_id = result['sample_id']
                item = result.get('item', '')
                prompt = result['prompt']
                response = result['response']
                label_true = result['label_true']
                label_pred = result['label_pred']
                semantic_entropy = result['semantic_entropy']
                num_clusters = result['num_clusters']
                error = result['error']
                needs_review = result['needs_review']
                ci_lower = result['ci_lower']
                ci_upper = result['ci_upper']
                ci_width = result['ci_width']
                
                metadata = result.get('entropy_metadata', {})
                sequences = metadata.get('sequences', [])
                clusters = metadata.get('clusters', [])
                
                # Aggregate statistics over sequences
                if sequences:
                    sequence_scores = [seq.get('score', 0) for seq in sequences]
                    sequence_log_probs = [seq.get('log_prob', 0) for seq in sequences if seq.get('log_prob') is not None]
                    
                    combined_data.append({
                        'sample_id': sample_id,
                        'item': item,
                        'prompt': prompt,
                        'response': response,
                        'label_true': label_true,
                        'label_pred': label_pred,
                        'n_generations': n,
                        'semantic_entropy': semantic_entropy,
                        'num_clusters': num_clusters,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'ci_width': ci_width,
                        'error': error,
                        'needs_review': needs_review,
                        'num_sequences': len(sequences),
                        'mean_sampled_score': np.mean(sequence_scores) if sequence_scores else None,
                        'std_sampled_score': np.std(sequence_scores) if sequence_scores else None,
                        'min_sampled_score': np.min(sequence_scores) if sequence_scores else None,
                        'max_sampled_score': np.max(sequence_scores) if sequence_scores else None,
                        'mean_log_prob': np.mean(sequence_log_probs) if sequence_log_probs else None,
                        'std_log_prob': np.std(sequence_log_probs) if sequence_log_probs else None
                    })
        
        df_combined = pd.DataFrame(combined_data)
        
        csv_path = os.path.join(self.output_dir, f'combined_analysis_{timestamp}.csv')
        df_combined.to_csv(csv_path, index=False)
        self.logger.info(f"Saved combined analysis dataset to: {csv_path}")
        
        try:
            parquet_path = os.path.join(self.output_dir, f'combined_analysis_{timestamp}.parquet')
            df_combined.to_parquet(parquet_path, index=False)
            self.logger.info(f"Saved combined analysis dataset to: {parquet_path}")
        except Exception as e:
            self.logger.warning(f"Could not save parquet (install pyarrow): {e}")
    
    def _log_metrics(self, n: int, metrics: Dict, review_stats: Dict):
        """Log computed metrics."""
        self.logger.info(f"n={n}: AUROC={metrics['auroc']:.4f}, "
                        f"Pearson r={metrics['pearson_correlation']:.4f}, "
                        f"Spearman r={metrics['spearman_correlation']:.4f}")
        self.logger.debug(f"n={n}: Mean entropy={metrics['mean_entropy']:.4f}, "
                         f"Std entropy={metrics['std_entropy']:.4f}, "
                         f"Mean clusters={metrics['mean_clusters']:.2f}")
        self.logger.debug(f"n={n}: Flagged={review_stats['flagged_count']}/{review_stats['total_samples']} "
                         f"({review_stats['flagged_percentage']:.1f}%)")


class ResultsVisualizer:
    """Handles visualization and reporting."""
    
    def __init__(self, output_dir: str, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger
    
    def plot_convergence(self, all_results: Dict) -> pd.DataFrame:
        """Plot convergence analysis results."""
        self.logger.info("Generating convergence plots")
        
        n_values = sorted(all_results.keys())
        
        pearson_corr = [all_results[n]['metrics']['pearson_correlation'] for n in n_values]
        spearman_corr = [all_results[n]['metrics']['spearman_correlation'] for n in n_values]
        auroc = [all_results[n]['metrics']['auroc'] for n in n_values]
        mean_entropy = [all_results[n]['metrics']['mean_entropy'] for n in n_values]
        std_entropy = [all_results[n]['metrics']['std_entropy'] for n in n_values]
        mean_clusters = [all_results[n]['metrics']['mean_clusters'] for n in n_values]
        
        pearson_changes = [abs(pearson_corr[i] - pearson_corr[i-1]) for i in range(1, len(pearson_corr))]
        spearman_changes = [abs(spearman_corr[i] - spearman_corr[i-1]) for i in range(1, len(spearman_corr))]
        auroc_changes = [abs(auroc[i] - auroc[i-1]) for i in range(1, len(auroc))]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle('Convergence Analysis: Sample Size Optimization', 
                     fontsize=16, fontweight='bold')
        
        axes[0, 0].plot(n_values, pearson_corr, 'o-', linewidth=2.5, markersize=8, color='steelblue')
        axes[0, 0].axvspan(10, 50, alpha=0.1, color='red')
        axes[0, 0].axvspan(50, 150, alpha=0.1, color='green')
        axes[0, 0].axvspan(150, 200, alpha=0.1, color='yellow')
        axes[0, 0].set_xlabel('Number of Generations (n)', fontweight='bold')
        axes[0, 0].set_ylabel('Pearson Correlation', fontweight='bold')
        axes[0, 0].set_title('Entropy vs Error Correlation (Pearson)')
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        axes[0, 1].plot(n_values, spearman_corr, 'o-', linewidth=2.5, markersize=8, color='darkgreen')
        axes[0, 1].axvspan(10, 50, alpha=0.1, color='red')
        axes[0, 1].axvspan(50, 150, alpha=0.1, color='green')
        axes[0, 1].axvspan(150, 200, alpha=0.1, color='yellow')
        axes[0, 1].set_xlabel('Number of Generations (n)', fontweight='bold')
        axes[0, 1].set_ylabel('Spearman Correlation', fontweight='bold')
        axes[0, 1].set_title('Entropy vs Error Correlation (Spearman)')
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        axes[0, 2].plot(n_values, auroc, 'o-', linewidth=2.5, markersize=8, color='darkred')
        axes[0, 2].axvspan(10, 50, alpha=0.1, color='red')
        axes[0, 2].axvspan(50, 150, alpha=0.1, color='green')
        axes[0, 2].axvspan(150, 200, alpha=0.1, color='yellow')
        axes[0, 2].set_xlabel('Number of Generations (n)', fontweight='bold')
        axes[0, 2].set_ylabel('AUROC', fontweight='bold')
        axes[0, 2].set_title('AUROC (Predicting Incorrectness)')
        axes[0, 2].grid(True, alpha=0.3, linestyle='--')
        axes[0, 2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        
        axes[1, 0].errorbar(n_values, mean_entropy, yerr=std_entropy, fmt='o-', linewidth=2.5, 
                           markersize=8, color='purple', capsize=5, capthick=2)
        axes[1, 0].axvspan(10, 50, alpha=0.1, color='red')
        axes[1, 0].axvspan(50, 150, alpha=0.1, color='green')
        axes[1, 0].axvspan(150, 200, alpha=0.1, color='yellow')
        axes[1, 0].set_xlabel('Number of Generations (n)', fontweight='bold')
        axes[1, 0].set_ylabel('Mean Semantic Entropy', fontweight='bold')
        axes[1, 0].set_title('Entropy Stability (Mean +/- Std Dev)')
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        
        n_values_changes = n_values[1:]
        axes[1, 1].plot(n_values_changes, pearson_changes, 'o-', label='Pearson change', linewidth=2, markersize=7, alpha=0.7)
        axes[1, 1].plot(n_values_changes, spearman_changes, 's-', label='Spearman change', linewidth=2, markersize=7, alpha=0.7)
        axes[1, 1].plot(n_values_changes, auroc_changes, '^-', label='AUROC change', linewidth=2, markersize=7, alpha=0.7)
        axes[1, 1].axvline(x=50, color='green', linestyle='--', linewidth=2, alpha=0.6)
        axes[1, 1].axvline(x=150, color='orange', linestyle='--', linewidth=2, alpha=0.6)
        axes[1, 1].set_xlabel('Number of Generations (n)', fontweight='bold')
        axes[1, 1].set_ylabel('Absolute Change in Metric', fontweight='bold')
        axes[1, 1].set_title('Metric Stability: Change Between Consecutive n')
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        axes[1, 1].legend(loc='best', fontsize=9)
        
        axes[1, 2].plot(n_values, mean_clusters, 'o-', linewidth=2.5, markersize=8, color='teal')
        axes[1, 2].axvspan(10, 50, alpha=0.1, color='red')
        axes[1, 2].axvspan(50, 150, alpha=0.1, color='green')
        axes[1, 2].axvspan(150, 200, alpha=0.1, color='yellow')
        axes[1, 2].set_xlabel('Number of Generations (n)', fontweight='bold')
        axes[1, 2].set_ylabel('Mean Number of Clusters', fontweight='bold')
        axes[1, 2].set_title('Clustering Behavior Convergence')
        axes[1, 2].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.output_dir, f"convergence_analysis_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Convergence plot saved: {plot_path}")
        
        summary_df = pd.DataFrame({
            'n': n_values,
            'pearson_r': pearson_corr,
            'spearman_r': spearman_corr,
            'auroc': auroc,
            'mean_entropy': mean_entropy,
            'std_entropy': std_entropy,
            'mean_clusters': mean_clusters
        })
        
        summary_path = os.path.join(self.output_dir, f"convergence_summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"Convergence summary saved: {summary_path}")
        
        return summary_df
    
    def generate_report(self, all_results: Dict, num_samples: int, n_values: List[int]):
        """Generate analysis report."""
        self.logger.info("Generating analysis report")
        
        n_values_sorted = sorted(all_results.keys())
        auroc_values = [all_results[n]['metrics']['auroc'] for n in n_values_sorted]
        pearson_values = [all_results[n]['metrics']['pearson_correlation'] for n in n_values_sorted]
        
        auroc_improvement = auroc_values[-1] - auroc_values[0]
        pearson_improvement = pearson_values[-1] - pearson_values[0]
        
        auroc_changes = [abs(auroc_values[i] - auroc_values[i-1]) for i in range(1, len(auroc_values))]
        recommended_n = 100
        for i, change in enumerate(auroc_changes):
            avg_change_threshold = np.mean(auroc_changes[-3:]) if len(auroc_changes) >= 3 else np.mean(auroc_changes)
            if change < avg_change_threshold * 0.5:
                recommended_n = n_values_sorted[i]
                break
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'num_samples': num_samples,
            'n_values_tested': n_values,
            'metrics_by_n': {
                str(n): {
                    'auroc': float(all_results[n]['metrics']['auroc']),
                    'pearson': float(all_results[n]['metrics']['pearson_correlation']),
                    'spearman': float(all_results[n]['metrics']['spearman_correlation']),
                    'mean_entropy': float(all_results[n]['metrics']['mean_entropy']),
                    'std_entropy': float(all_results[n]['metrics']['std_entropy'])
                }
                for n in n_values_sorted
            },
            'improvement': {
                'auroc': float(auroc_improvement),
                'pearson': float(pearson_improvement)
            },
            'recommended_n': int(recommended_n),
            'interpretation': self._interpret_results(auroc_improvement, pearson_improvement)
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"analysis_report_{timestamp}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Analysis report saved: {report_path}")
        self._log_report_summary(report, n_values_sorted, auroc_values, pearson_values)
        
        return report
    
    def _interpret_results(self, auroc_improvement: float, pearson_improvement: float) -> str:
        """Interpret results."""
        if auroc_improvement > 0.05:
            return "Metrics improve with more generations. Extend n to 50-150 for better calibration."
        elif abs(auroc_improvement) < 0.02:
            return "Metrics stable across n values. Original n=10 is sufficient."
        else:
            return "Metrics vary but no clear trend. Use n=100 based on elbow analysis."
    
    def _log_report_summary(self, report: Dict, n_values: List[int], auroc: List[float], pearson: List[float]):
        """Log report summary."""
        self.logger.info("=" * 80)
        self.logger.info("CONVERGENCE ANALYSIS REPORT")
        self.logger.info("=" * 80)
        self.logger.info(f"Samples analyzed: {report['num_samples']}")
        self.logger.info(f"n values tested: {report['n_values_tested']}")
        self.logger.info(f"Recommended n: {report['recommended_n']}")
        self.logger.info(f"AUROC improvement: {report['improvement']['auroc']:+.4f}")
        self.logger.info(f"Pearson improvement: {report['improvement']['pearson']:+.4f}")
        self.logger.info(f"Interpretation: {report['interpretation']}")
        self.logger.info("=" * 80)


class DataProcessor:
    """Handles data loading and preprocessing."""
    
    @staticmethod
    def load_and_prepare_data(
        csv_path: str,
        num_samples: int,
        logger: logging.Logger
    ) -> pd.DataFrame:
        """Load and prepare data."""
        logger.info(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples")
        
        if len(df) > num_samples:
            df['label_bin'] = pd.qcut(df['label'], q=3, labels=['low', 'medium', 'high'], duplicates='drop')
            
            df = df.groupby('label_bin', group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), num_samples // 3), random_state=42)
            ).reset_index(drop=True)
            
            if len(df) < num_samples:
                remaining = num_samples - len(df)
                df_full = pd.read_csv(csv_path)
                df_remaining = df_full[~df_full.index.isin(df.index)].sample(
                    n=min(remaining, len(df_full) - len(df)),
                    random_state=42
                )
                df = pd.concat([df, df_remaining]).reset_index(drop=True)
            
            df = df.drop(columns=['label_bin'], errors='ignore')
            logger.info(f"Stratified sampling: {num_samples} samples")
            logger.info(f"Label range: [{df['label'].min():.2f}, {df['label'].max():.2f}]")
            logger.info(f"Label mean: {df['label'].mean():.2f} (+/- {df['label'].std():.2f})")
        
        return df.reset_index(drop=True)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convergence Analysis: Determine optimal number of generations (n=10-200)"
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
        default="PhillipGre/Llama-2-7b_sctt_regression",
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
        default=500,
        help="Number of samples to use (default: 500)"
    )
    parser.add_argument(
        "--n_values",
        type=int,
        nargs='+',
        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200],
        help="List of n values to test"
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
    
    if args.base_model is None:
        raise ValueError("MODEL_NAME not set. Set in .env or use --base_model")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if not os.path.isabs(args.test_data):
        test_path1 = os.path.join(script_dir, args.test_data)
        test_path2 = os.path.join(project_root, args.test_data.replace('../', ''))
        args.test_data = test_path1 if os.path.exists(test_path1) else test_path2
    
    if not os.path.isabs(args.output_dir):
        output_path_from_root = os.path.join(project_root, args.output_dir.replace('./', '').replace('.\\', ''))
        args.output_dir = output_path_from_root
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger = configure_logger(args.output_dir)
    
    logger.info("Starting convergence analysis")
    logger.info(f"Base model: {args.base_model}")
    if args.lora_adapter:
        logger.info(f"LoRA adapter: {args.lora_adapter}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"n values: {args.n_values}")
    
    logger.info("Loading model and tokenizer")
    model, tokenizer = ModelLoader.load_model_and_tokenizer(
        args.base_model,
        args.lora_adapter,
        args.device
    )
    
    logger.info("Loading generative model for semantic entropy calculation")
    generative_model, _ = ModelLoader.load_generative_model(
        args.base_model,
        args.device
    )
    
    df = DataProcessor.load_and_prepare_data(args.test_data, args.num_samples, logger)
    
    analyzer = ConvergenceAnalyzer(model, tokenizer, args.output_dir, logger, generative_model)
    all_results = analyzer.run_analysis(df, args.n_values, args.device)
    
    visualizer = ResultsVisualizer(args.output_dir, logger)
    visualizer.plot_convergence(all_results)
    visualizer.generate_report(all_results, len(df), args.n_values)
    
    logger.info("Convergence analysis complete")


if __name__ == "__main__":
    main()