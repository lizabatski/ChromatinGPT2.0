#!/usr/bin/env python3
"""
Utility functions for per-histone model training and evaluation
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import json
import os
import pandas as pd

HISTONE_NAMES = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']

def load_model_results(results_dir: str) -> Dict:
    """Load results for all histone models"""
    results = {}
    
    for i, histone_name in enumerate(HISTONE_NAMES):
        # Load test results
        test_file = os.path.join(results_dir, f'{histone_name}_test_results.json')
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                results[histone_name] = json.load(f)
        
        # Load training history
        history_file = os.path.join(results_dir, f'{histone_name}_training_history.json')
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
                if histone_name in results:
                    results[histone_name]['training_history'] = history
                else:
                    results[histone_name] = {'training_history': history}
    
    return results

def calculate_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              bin_names: Optional[List[str]] = None) -> Dict:
    """Calculate detailed metrics including per-bin analysis"""
    metrics = {}
    
    # Overall metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    metrics['overall'] = {
        'mse': mean_squared_error(y_true_flat, y_pred_flat),
        'mae': mean_absolute_error(y_true_flat, y_pred_flat),
        'r2': r2_score(y_true_flat, y_pred_flat),
        'rmse': np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
        'pearson_r': np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
    }
    
    # Per-bin metrics if data has multiple dimensions
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        metrics['per_bin'] = {}
        
        for bin_idx in range(y_true.shape[1]):
            bin_true = y_true[:, bin_idx]
            bin_pred = y_pred[:, bin_idx]
            
            bin_name = f'bin_{bin_idx}' if bin_names is None else bin_names[bin_idx]
            
            metrics['per_bin'][bin_name] = {
                'mse': mean_squared_error(bin_true, bin_pred),
                'mae': mean_absolute_error(bin_true, bin_pred),
                'r2': r2_score(bin_true, bin_pred),
                'pearson_r': np.corrcoef(bin_true, bin_pred)[0, 1]
            }
    
    # Statistical summaries
    metrics['prediction_stats'] = {
        'mean': float(np.mean(y_pred_flat)),
        'std': float(np.std(y_pred_flat)),
        'min': float(np.min(y_pred_flat)),
        'max': float(np.max(y_pred_flat)),
        'median': float(np.median(y_pred_flat))
    }
    
    metrics['target_stats'] = {
        'mean': float(np.mean(y_true_flat)),
        'std': float(np.std(y_true_flat)),
        'min': float(np.min(y_true_flat)),
        'max': float(np.max(y_true_flat)),
        'median': float(np.median(y_true_flat))
    }
    
    return metrics

def plot_training_curves(results_dir: str, save_path: Optional[str] = None):
    """Plot training curves for all histone models"""
    results = load_model_results(results_dir)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, histone_name in enumerate(HISTONE_NAMES):
        ax = axes[i]
        
        if histone_name in results and 'training_history' in results[histone_name]:
            history = results[histone_name]['training_history']
            
            epochs = range(1, len(history['train_losses']) + 1)
            
            ax.plot(epochs, history['train_losses'], 'b-', label='Train Loss', alpha=0.7)
            ax.plot(epochs, history['val_losses'], 'r-', label='Val Loss', alpha=0.7)
            
            # Mark best epoch
            best_epoch = np.argmin(history['val_losses']) + 1
            best_val_loss = min(history['val_losses'])
            ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
            ax.scatter([best_epoch], [best_val_loss], color='g', s=50, zorder=5)
            
            ax.set_title(f'{histone_name} Training Curves')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data for {histone_name}', 
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'{histone_name} - No Data')
    
    # Remove empty subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()

def plot_performance_comparison(results_dir: str, save_path: Optional[str] = None):
    """Plot performance comparison across all histone models"""
    results = load_model_results(results_dir)
    
    # Extract metrics
    histone_names = []
    r2_scores = []
    mse_scores = []
    mae_scores = []
    
    for histone_name in HISTONE_NAMES:
        if histone_name in results and 'test_metrics' in results[histone_name]:
            histone_names.append(histone_name)
            metrics = results[histone_name]['test_metrics']
            r2_scores.append(metrics['r2'])
            mse_scores.append(metrics['mse'])
            mae_scores.append(metrics['mae'])
    
    if not histone_names:
        print("No test results found!")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # R2 scores
    bars1 = axes[0].bar(histone_names, r2_scores, color='skyblue', alpha=0.7)
    axes[0].set_title('R² Scores by Histone Mark')
    axes[0].set_ylabel('R² Score')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars1, r2_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
    
    # MSE scores
    bars2 = axes[1].bar(histone_names, mse_scores, color='lightcoral', alpha=0.7)
    axes[1].set_title('MSE by Histone Mark')
    axes[1].set_ylabel('Mean Squared Error')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    for bar, score in zip(bars2, mse_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_scores)*0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    
    # MAE scores
    bars3 = axes[2].bar(histone_names, mae_scores, color='lightgreen', alpha=0.7)
    axes[2].set_title('MAE by Histone Mark')
    axes[2].set_ylabel('Mean Absolute Error')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    for bar, score in zip(bars3, mae_scores):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_scores)*0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison saved to {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print(f"Average R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"Average MSE: {np.mean(mse_scores):.6f} ± {np.std(mse_scores):.6f}")
    print(f"Average MAE: {np.mean(mae_scores):.6f} ± {np.std(mae_scores):.6f}")

def create_results_summary_table(results_dir: str, save_path: Optional[str] = None) -> pd.DataFrame:
    """Create a summary table of results for all histone models"""
    results = load_model_results(results_dir)
    
    summary_data = []
    
    for histone_name in HISTONE_NAMES:
        if histone_name in results:
            row = {'Histone': histone_name}
            
            # Test metrics
            if 'test_metrics' in results[histone_name]:
                metrics = results[histone_name]['test_metrics']
                row['R²'] = metrics.get('r2', np.nan)
                row['MSE'] = metrics.get('mse', np.nan)
                row['MAE'] = metrics.get('mae', np.nan)
            
            # Training info
            if 'training_history' in results[histone_name]:
                history = results[histone_name]['training_history']
                row['Best_Val_R²'] = history.get('best_val_r2', np.nan)
                row['Best_Val_Loss'] = history.get('best_val_loss', np.nan)
                row['Epochs_Trained'] = len(history.get('train_losses', []))
            
            summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Results summary saved to {save_path}")
    
    return df

def predict_with_model(model_path: str, dna_data: np.ndarray, dnase_data: np.ndarray, 
                      histone_data: np.ndarray, target_histone_idx: int, 
                      use_gpu: bool = True) -> np.ndarray:
    """Make predictions using a trained per-histone model"""
    from per_histone import PerHistoneTrainer, PerHistoneModel
    
    # Load model config from the model file
    checkpoint = torch.load(model_path, map_location='cpu')
    target_histone_idx = checkpoint['target_histone_idx']
    
    # Initialize trainer (this will need model config - you might want to save this too)
    trainer = PerHistoneTrainer(
        target_histone_idx=target_histone_idx,
        use_gpu=use_gpu
        # Note: You'll need to save model_config in the checkpoint to fully restore
    )
    
    trainer.load_model(model_path)
    
    # Prepare data
    other_indices = [i for i in range(7) if i != target_histone_idx]
    other_histones = histone_data[:, other_indices, :]
    
    # Make predictions
    predictions = []
    batch_size = 32
    
    for i in range(0, len(dna_data), batch_size):
        end_idx = min(i + batch_size, len(dna_data))
        
        batch_dna = dna_data[i:end_idx]
        batch_dnase = dnase_data[i:end_idx]
        batch_histones = other_histones[i:end_idx]
        
        with torch.no_grad():
            dna_input = torch.tensor(batch_dna, dtype=torch.float32)
            dnase_input = torch.tensor(batch_dnase, dtype=torch.float32).unsqueeze(1)
            histone_input = torch.tensor(batch_histones, dtype=torch.float32)
            
            if use_gpu and torch.cuda.is_available():
                dna_input = dna_input.cuda()
                dnase_input = dnase_input.cuda()
                histone_input = histone_input.cuda()
            
            batch_pred = trainer.model(dna_input, dnase_input, histone_input)
            predictions.append(batch_pred.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)

def plot_prediction_vs_true(y_true: np.ndarray, y_pred: np.ndarray, 
                           histone_name: str, save_path: Optional[str] = None):
    """Create scatter plot of predictions vs true values"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Overall scatter plot
    axes[0].scatter(y_true.flatten(), y_pred.flatten(), alpha=0.6, s=1)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(f'{histone_name} - Overall Predictions vs True')
    
    # Calculate R²
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    axes[0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Residuals plot
    residuals = y_pred.flatten() - y_true.flatten()
    axes[1].scatter(y_true.flatten(), residuals, alpha=0.6, s=1)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('True Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals vs True Values')
    
    # Histogram of residuals
    axes[2].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Residuals')
    
    # Mean prediction per bin
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        bin_means_true = np.mean(y_true, axis=0)
        bin_means_pred = np.mean(y_pred, axis=0)
        bin_indices = np.arange(len(bin_means_true))
        
        axes[3].plot(bin_indices, bin_means_true, 'o-', label='True', alpha=0.7)
        axes[3].plot(bin_indices, bin_means_pred, 'o-', label='Predicted', alpha=0.7)
        axes[3].set_xlabel('Bin Index')
        axes[3].set_ylabel('Mean Signal')
        axes[3].set_title('Mean Signal per Bin')
        axes[3].legend()
    else:
        axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to {save_path}")
    
    plt.show()

def batch_train_all_histones(data_file: str, output_dir: str, config: Dict):
    """Train models for all 7 histone marks"""
    import subprocess
    import sys
    
    os.makedirs(output_dir, exist_ok=True)
    
    script_path = os.path.join(os.path.dirname(__file__), 'per_histone_train.py')
    
    for histone_idx in range(7):
        histone_name = HISTONE_NAMES[histone_idx]
        print(f"\n{'='*50}")
        print(f"Training model for {histone_name} (index {histone_idx})")
        print(f"{'='*50}")
        
        # Build command
        cmd = [
            sys.executable, script_path,
            '--data_file', data_file,
            '--target_histone', str(histone_idx),
            '--output_dir', output_dir,
        ]
        
        # Add config parameters
        for key, value in config.items():
            cmd.extend([f'--{key}', str(value)])
        
        # Run training
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"Error training model for {histone_name}")
            continue
        
        print(f"Successfully trained model for {histone_name}")
    
    print(f"\nAll models trained! Results saved to {output_dir}")
    
    # Generate summary plots and tables
    plot_training_curves(output_dir, os.path.join(output_dir, 'training_curves.png'))
    plot_performance_comparison(output_dir, os.path.join(output_dir, 'performance_comparison.png'))
    create_results_summary_table(output_dir, os.path.join(output_dir, 'results_summary.csv'))

def prepare_per_histone_data(all_histone_data: np.ndarray, target_histone_idx: int):
    """
    Prepare data for per-histone training from all histone data
    
    Args:
        all_histone_data: (batch, 7, seq_len) - All 7 histone signals
        target_histone_idx: Index of target histone (0-6)
    
    Returns:
        tuple: (other_histone_data, target_histone_data)
    """
    
    # Extract target histone
    target_histone_data = all_histone_data[:, target_histone_idx, :]
    
    # Extract other 6 histones (exclude target)
    other_indices = [i for i in range(7) if i != target_histone_idx]
    other_histone_data = all_histone_data[:, other_indices, :]
    
    return other_histone_data, target_histone_data

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Utility functions for per-histone models')
    parser.add_argument('--action', required=True, choices=['plot', 'summary', 'batch_train'],
                       help='Action to perform')
    parser.add_argument('--results_dir', help='Directory with model results')
    parser.add_argument('--data_file', help='Data file for batch training')
    parser.add_argument('--output_dir', help='Output directory')
    
    args = parser.parse_args()
    
    if args.action == 'plot' and args.results_dir:
        plot_training_curves(args.results_dir)
        plot_performance_comparison(args.results_dir)
    elif args.action == 'summary' and args.results_dir:
        df = create_results_summary_table(args.results_dir)
        print(df)
    elif args.action == 'batch_train' and args.data_file and args.output_dir:
        # Default config
        config = {
            'channels': 768,
            'num_transformer_layers': 4,
            'num_heads': 8,
            'dropout': 0.4,
            'batch_size': 32,
            'num_epochs': 50,
            'learning_rate': 0.001
        }
        batch_train_all_histones(args.data_file, args.output_dir, config)