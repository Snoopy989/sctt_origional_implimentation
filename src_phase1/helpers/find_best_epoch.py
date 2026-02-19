import pandas as pd
from scipy.stats import pearsonr
import re

# Process both train and validation results
for split in ['train', 'validation']:
    filename = f'epoch_wise_LORA_results_{split}.csv'
    
    try:
        # Load epoch-wise results
        df = pd.read_csv(filename)
        
        # Extract checkpoint number from peft_model_id
        df['checkpoint'] = df['peft_model_id'].str.extract(r'checkpoint-(\d+)').astype(int)
        
        # Group by checkpoint and calculate Pearson correlation
        results = []
        for checkpoint in sorted(df['checkpoint'].unique()):
            checkpoint_data = df[df['checkpoint'] == checkpoint]
            preds = checkpoint_data['predictions'].values
            ratings = checkpoint_data['ratings'].values
            
            if len(preds) > 1:
                corr, p_value = pearsonr(preds, ratings)
                results.append({
                    'checkpoint': checkpoint,
                    'pearson_r': corr,
                    'p_value': p_value,
                    'n_samples': len(preds)
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results).sort_values('pearson_r', ascending=False)
        
        print("\n" + "="*70)
        print(f"CHECKPOINT PERFORMANCE ({split.upper()} Set - Pearson r)")
        print("="*70)
        print(results_df.to_string(index=False))
        print("="*70)
        
        # Find best checkpoint
        best = results_df.iloc[0]
        print(f"\nBEST CHECKPOINT: checkpoint-{int(best['checkpoint'])}")
        print(f"  Pearson r:  {best['pearson_r']:.4f}")
        print(f"  p-value:    {best['p_value']:.6f}")
        print(f"  Samples:    {int(best['n_samples'])}")
        print()
        
        # Save detailed results
        output_file = f'checkpoint_performance_{split}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"Detailed results saved to: {output_file}\n")
        
    except FileNotFoundError:
        print(f"\nFile not found: {filename}")
        print("Please run compute_epoch_wise_results.py first.\n")
