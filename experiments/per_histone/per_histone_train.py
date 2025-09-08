#!/usr/bin/env python3
"""
Training script for per-histone models

Usage:
    python per_histone_train.py \
        --data_file data/histone_data.npz \
        --target_histone 0 \
        --output_dir models/per_histone/
"""

import argparse
import numpy as np
import os
import json
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directories to path
sys.path.append(os.path.dirname(__file__))

from per_histone import PerHistoneTrainer

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate regression metrics"""
    metrics = {}
    
    # Flatten for overall metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    metrics['mse'] = mean_squared_error(y_true_flat, y_pred_flat)
    metrics['mae'] = mean_absolute_error(y_true_flat, y_pred_flat)
    metrics['r2'] = r2_score(y_true_flat, y_pred_flat)
    
    # Per-bin metrics
    if len(y_true.shape) > 1:
        for bin_idx in range(min(10, y_true.shape[1])):  # Only first 10 bins to avoid too much output
            bin_true = y_true[:, bin_idx]
            bin_pred = y_pred[:, bin_idx]
            metrics[f'bin_{bin_idx}_r2'] = r2_score(bin_true, bin_pred)
    
    return metrics

def evaluate(trainer, dataloader, device):
    """Evaluate model on validation/test set"""
    trainer.model.eval()
    
    all_losses = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            seq_batch, dns_batch, histone_input_batch, histone_target_batch = batch
            
            # Move to numpy for trainer interface
            seq_batch = seq_batch.numpy()
            dns_batch = dns_batch.numpy()
            histone_input_batch = histone_input_batch.numpy()
            histone_target_batch = histone_target_batch.numpy()
            
            loss, predictions = trainer.eval_on_batch(seq_batch, dns_batch, histone_input_batch, histone_target_batch)
            
            all_losses.append(loss)
            all_predictions.append(predictions)
            
            # Extract target histone for this batch
            target_idx = trainer.target_histone_idx
            targets = histone_target_batch[:, target_idx, :]
            all_targets.append(targets)
    
    # Concatenate all results
    avg_loss = np.mean(all_losses)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_predictions)
    
    return avg_loss, metrics

class HistoneDataset(Dataset):
    """Dataset for loading histone data"""
    def __init__(self, dna_data, dnase_data, histone_inputs, histone_targets):
        self.dna_data = torch.tensor(dna_data, dtype=torch.float32)
        self.dnase_data = torch.tensor(dnase_data, dtype=torch.float32)
        self.histone_inputs = torch.tensor(histone_inputs, dtype=torch.float32)
        self.histone_targets = torch.tensor(histone_targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.dna_data)
    
    def __getitem__(self, idx):
        return (self.dna_data[idx], self.dnase_data[idx], 
                self.histone_inputs[idx], self.histone_targets[idx])

def load_and_split_data(npz_file: str, test_size: float = 0.2, val_size: float = 0.1):
    """Load data and create train/val/test splits"""
    
    print(f"Loading data from {npz_file}")
    data = np.load(npz_file, allow_pickle=True)
    
    dna_data = data['dna']
    dnase_data = data['dnase'] 
    histone_inputs = data['histone_inputs']  # Full resolution inputs
    histone_targets = data['targets']        # Binned targets
    
    print(f"Data shapes:")
    print(f"  DNA: {dna_data.shape}")
    print(f"  DNase: {dnase_data.shape}")
    print(f"  Histone inputs: {histone_inputs.shape}")
    print(f"  Histone targets: {histone_targets.shape}")
    
    # Check for NaNs and handle them
    dna_nans = np.isnan(dna_data).sum()
    dnase_nans = np.isnan(dnase_data).sum()
    histone_input_nans = np.isnan(histone_inputs).sum()
    histone_target_nans = np.isnan(histone_targets).sum()
    
    print(f"NaN counts: DNA={dna_nans}, DNase={dnase_nans}, Histone_inputs={histone_input_nans}, Histone_targets={histone_target_nans}")
    
    if histone_target_nans > 0 or dnase_nans > 0 or histone_input_nans > 0:
        print("Warning: Found NaNs in data. Replacing with zeros.")
        dnase_data = np.nan_to_num(dnase_data, nan=0.0)
        histone_inputs = np.nan_to_num(histone_inputs, nan=0.0)
        histone_targets = np.nan_to_num(histone_targets, nan=0.0)
    
    # Create splits
    indices = np.arange(len(dna_data))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size/(1-test_size), random_state=42)
    
    print(f"Data splits:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Validation: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")
    
    # Split the data - now returning 4 arrays per split
    train_data = (dna_data[train_idx], dnase_data[train_idx], 
                  histone_inputs[train_idx], histone_targets[train_idx])
    val_data = (dna_data[val_idx], dnase_data[val_idx], 
                histone_inputs[val_idx], histone_targets[val_idx])
    test_data = (dna_data[test_idx], dnase_data[test_idx], 
                 histone_inputs[test_idx], histone_targets[test_idx])
    
    return train_data, val_data, test_data

def main():
    parser = argparse.ArgumentParser(description='Train per-histone model')
    
    # Data arguments
    parser.add_argument('--data_file', required=True, help='Path to NPZ data file')
    parser.add_argument('--target_histone', type=int, required=True, choices=range(7),
                       help='Target histone index (0-6)')
    parser.add_argument('--output_dir', required=True, help='Output directory for models')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set fraction')
    
    # Model arguments
    parser.add_argument('--channels', type=int, default=768, help='Model channels')
    parser.add_argument('--num_transformer_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--output_bins', type=int, default=1024, help='Number of output bins')
    parser.add_argument('--pooling_type', default='attention', choices=['attention', 'max'], help='Pooling type')
    parser.add_argument('--num_conv_blocks', type=int, default=2, help='Number of conv blocks per pathway')
    parser.add_argument('--fusion_type', default='concat', choices=['concat', 'hierarchical', 'mil'],
                       help='Fusion strategy')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU usage')
    
    args = parser.parse_args()
    
    # Histone names for reference
    histone_names = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']
    target_name = histone_names[args.target_histone]
    
    print(f"Training model for {target_name} (index {args.target_histone})")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and split data
    train_data, val_data, test_data = load_and_split_data(
        args.data_file, args.test_size, args.val_size
    )
    
    # Create datasets and dataloaders
    train_dataset = HistoneDataset(*train_data)
    val_dataset = HistoneDataset(*val_data)
    test_dataset = HistoneDataset(*test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model configuration
    model_config = {
        'channels': args.channels,
        'num_transformer_layers': args.num_transformer_layers,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'output_bins': args.output_bins,
        'pooling_type': args.pooling_type,
        'num_conv_blocks': args.num_conv_blocks,
        'fusion_type': args.fusion_type
    }
    
    # Initialize trainer
    trainer = PerHistoneTrainer(
        target_histone_idx=args.target_histone,
        histone_names=histone_names,
        use_gpu=not args.no_gpu,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        **model_config
    )
    
    # Save config
    config_path = os.path.join(args.output_dir, f'{target_name}_config.json')
    config_to_save = {
        'target_histone': args.target_histone,
        'target_name': target_name,
        'model_config': model_config,
        'training_config': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'num_epochs': args.num_epochs
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    print(f"Configuration saved to {config_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Training loop
    print(f"\nStarting training for {target_name}...")
    
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_r2s = []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Training
        trainer.model.train()
        epoch_train_losses = []
        
        for batch in tqdm(train_loader, desc="Training"):
            seq_batch, dns_batch, histone_input_batch, histone_target_batch = batch
            
            # Convert to numpy for trainer interface
            seq_batch = seq_batch.numpy()
            dns_batch = dns_batch.numpy()
            histone_input_batch = histone_input_batch.numpy()
            histone_target_batch = histone_target_batch.numpy()
            
            loss = trainer.train_on_batch(seq_batch, dns_batch, histone_input_batch, histone_target_batch)
            epoch_train_losses.append(loss)
        
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        val_loss, val_metrics = evaluate(trainer, val_loader, device)
        val_losses.append(val_loss)
        val_r2s.append(val_metrics['r2'])
        
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Val R2: {val_metrics['r2']:.4f}")
        print(f"Val MAE: {val_metrics['mae']:.6f}")
        
        # Save best model based on R2 score
        if val_metrics['r2'] > best_val_r2:
            best_val_loss = val_loss
            best_val_r2 = val_metrics['r2']
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(args.output_dir, f'{target_name}_best_model.pth')
            trainer.save_model(best_model_path)
            
            print(f"New best model saved! R2: {best_val_r2:.4f}")
            
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.early_stopping_patience}")
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, f'{target_name}_final_model.pth')
    trainer.save_model(final_model_path)
    
    # Save training history
    history = {
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'val_r2s': [float(x) for x in val_r2s],
        'best_val_loss': float(best_val_loss),
        'best_val_r2': float(best_val_r2)
    }
    
    history_path = os.path.join(args.output_dir, f'{target_name}_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Test evaluation
    print(f"\nEvaluating on test set...")
    
    # Load best model for testing
    trainer.load_model(best_model_path)
    test_loss, test_metrics = evaluate(trainer, test_loader, device)
    
    print(f"Test Results for {target_name}:")
    print(f"  Loss: {test_loss:.6f}")
    print(f"  R2: {test_metrics['r2']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.6f}")
    print(f"  MSE: {test_metrics['mse']:.6f}")
    
    # Save test results
    test_results = {
        'target_histone': target_name,
        'test_loss': float(test_loss),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()}
    }
    
    test_results_path = os.path.join(args.output_dir, f'{target_name}_test_results.json')
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTraining completed for {target_name}!")
    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_model_path}")
    print(f"Training history: {history_path}")
    print(f"Test results: {test_results_path}")

if __name__ == "__main__":
    main()