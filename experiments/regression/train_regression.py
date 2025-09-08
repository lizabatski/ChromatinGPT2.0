import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Optional
import json
from tqdm import tqdm


from dhica_regression import DeepHistoneEnformer

# Define target names
TARGETS = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']

class PreprocessedDataset(Dataset):
    """Dataset for loading preprocessed numpy arrays"""
    
    def __init__(self, npz_file: str, normalize: bool = True):
        """
        Args:
            npz_file: Path to .npz file with preprocessed data
            normalize: Whether to normalize DNase and target values
        """
        data = np.load(npz_file)
        self.dna = data['dna'].astype(np.float32)
        self.dnase = data['dnase'].astype(np.float32)
        self.targets = data['targets'].astype(np.float32)
        
        # Add channel dimension to DNase if needed
        if len(self.dnase.shape) == 2:
            self.dnase = np.expand_dims(self.dnase, axis=1)
        
        # Store normalization parameters for potential reuse
        self.normalization_params = {}
        
        # Normalize DNase signals
        if normalize:
            # Log transform and normalize DNase
            self.dnase = np.log1p(self.dnase)
            dnase_mean = np.mean(self.dnase)
            dnase_std = np.std(self.dnase)
            self.normalization_params['dnase_mean'] = dnase_mean
            self.normalization_params['dnase_std'] = dnase_std
            
            if dnase_std > 0:
                self.dnase = (self.dnase - dnase_mean) / dnase_std
            
            # Log transform and normalize targets
            self.targets = np.log1p(self.targets)
            self.target_means = np.mean(self.targets, axis=0)
            self.target_stds = np.std(self.targets, axis=0)
            self.normalization_params['target_means'] = self.target_means
            self.normalization_params['target_stds'] = self.target_stds
            
            for i in range(self.targets.shape[1]):
                if (self.target_stds[i] > 0).any():
                    self.targets[:, i] = (self.targets[:, i] - self.target_means[i]) / self.target_stds[i]
        
        print(f"Loaded dataset with {len(self)} samples")
        print(f"DNA shape: {self.dna.shape}")
        print(f"DNase shape: {self.dnase.shape}")
        print(f"Targets shape: {self.targets.shape}")
    
    def __len__(self):
        return len(self.dna)
    
    def __getitem__(self, idx):
        return self.dna[idx], self.dnase[idx], self.targets[idx]

def split_dataset(dataset: Dataset, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        dataset: The dataset to split
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set  
        test_ratio: Proportion for test set
        seed: Random seed for reproducible splits
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set random seed for reproducible splits
    generator = torch.Generator().manual_seed(seed)
    
    # Calculate sizes
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size  # Ensure all samples are used
    
    print(f"Splitting dataset of {total_size} samples:")
    print(f"  Training: {train_size} samples ({train_size/total_size:.1%})")
    print(f"  Validation: {val_size} samples ({val_size/total_size:.1%})")
    print(f"  Test: {test_size} samples ({test_size/total_size:.1%})")
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    return train_dataset, val_dataset, test_dataset

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics for 3D target arrays"""
    metrics = {}
    
    # Overall metrics (flatten everything)
    metrics['mse'] = mean_squared_error(y_true.flatten(), y_pred.flatten())
    metrics['mae'] = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    metrics['r2'] = r2_score(y_true.flatten(), y_pred.flatten())
    
    # Per-target metrics (flatten across bins for each target)
    for i in range(min(y_true.shape[1], len(TARGETS))):
        target_name = TARGETS[i]
        # Flatten samples and bins for this target: (samples, bins) -> (samples*bins,)
        true_flat = y_true[:, i, :].flatten()
        pred_flat = y_pred[:, i, :].flatten()
        
        metrics[f'{target_name}_mse'] = mean_squared_error(true_flat, pred_flat)
        metrics[f'{target_name}_mae'] = mean_absolute_error(true_flat, pred_flat)
        metrics[f'{target_name}_r2'] = r2_score(true_flat, pred_flat)
    
    return metrics

def train_epoch(model, train_loader, device: str) -> float:
    """Train for one epoch"""
    model.forward_fn.train()
    train_losses = []
    
    for dna_batch, dnase_batch, target_batch in tqdm(train_loader, desc="Training"):
        # Move to device
        if device == 'cuda':
            dna_batch = dna_batch.cuda()
            dnase_batch = dnase_batch.cuda()
            target_batch = target_batch.cuda()
        
        # Convert to numpy (if your model expects numpy inputs)
        dna_np = dna_batch.cpu().numpy()
        dnase_np = dnase_batch.cpu().numpy()
        target_np = target_batch.cpu().numpy()
        
        # Train on batch
        loss = model.train_on_batch(dna_np, dnase_np, target_np)
        train_losses.append(loss)
    
    return np.mean(train_losses)

def evaluate(model, val_loader, device: str) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on validation set"""
    model.forward_fn.eval()
    val_losses = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for dna_batch, dnase_batch, target_batch in tqdm(val_loader, desc="Evaluating"):
            # Move to device
            if device == 'cuda':
                dna_batch = dna_batch.cuda()
                dnase_batch = dnase_batch.cuda()
                target_batch = target_batch.cuda()
            
            # Convert to numpy
            dna_np = dna_batch.cpu().numpy()
            dnase_np = dnase_batch.cpu().numpy()
            target_np = target_batch.cpu().numpy()
            
            # Evaluate batch
            loss, pred = model.eval_on_batch(dna_np, dnase_np, target_np)
            
            val_losses.append(loss)
            all_preds.append(pred)
            all_targets.append(target_np)
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_preds)
    avg_loss = np.mean(val_losses)
    
    return avg_loss, metrics

def print_metrics(metrics: Dict[str, float], phase: str = "Validation"):
    """Print metrics in a formatted way"""
    print(f"\n{phase} Metrics:")
    print(f"  Overall MSE: {metrics['mse']:.4f}")
    print(f"  Overall MAE: {metrics['mae']:.4f}")
    print(f"  Overall R²:  {metrics['r2']:.4f}")
    
    print("\n  Per-target metrics:")
    print(f"  {'Target':12s} {'MSE':>8s} {'MAE':>8s} {'R²':>8s}")
    print("  " + "-" * 40)
    
    for target_name in TARGETS:
        if f'{target_name}_mse' in metrics:
            mse = metrics[f'{target_name}_mse']
            mae = metrics[f'{target_name}_mae']
            r2 = metrics[f'{target_name}_r2']
            print(f"  {target_name:12s} {mse:8.4f} {mae:8.4f} {r2:8.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train histone modification prediction model')
    
    # Data arguments
    parser.add_argument('--data', required=True, help='Path to the main .npz data file')
    parser.add_argument('--train_ratio', type=float, default=0.7, 
                       help='Proportion of data for training (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15, 
                       help='Proportion of data for validation (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15, 
                       help='Proportion of data for testing (default: 0.15)')
    
    # Model arguments
    parser.add_argument('--channels', type=int, default=768, help='Number of channels')
    parser.add_argument('--num_transformer_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--fusion_type', choices=['concat', 'cross_attention', 'gated', 'mil'], 
                       default='concat', help='Fusion strategy')
    parser.add_argument('--pooling_type', choices=['attention', 'max'], default='attention',
                       help='Pooling type')
    parser.add_argument('--num_conv_blocks', type=int, default=2, help='Number of conv blocks')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    
    # Other arguments
    parser.add_argument('--output_dir', default='./models', help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--save_splits', action='store_true', 
                       help='Save the split indices for reproducibility')
    
    args = parser.parse_args()
    
    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total_ratio}")
    
    print("Starting training script...")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading and splitting data...")
    
    # Load full dataset
    full_dataset = PreprocessedDataset(args.data, normalize=False)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset, 
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio, 
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Save split information if requested
    if args.save_splits:
        split_info = {
            'total_samples': len(full_dataset),
            'train_indices': train_dataset.indices,
            'val_indices': val_dataset.indices,
            'test_indices': test_dataset.indices,
            'split_ratios': {
                'train': args.train_ratio,
                'val': args.val_ratio,
                'test': args.test_ratio
            },
            'seed': args.seed
        }
        with open(os.path.join(args.output_dir, 'split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"Saved split information to {os.path.join(args.output_dir, 'split_info.json')}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )
    
    # Initialize model
    model = DeepHistoneEnformer(
        use_gpu=(device == 'cuda'),
        learning_rate=args.learning_rate,
        channels=args.channels,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_targets=len(TARGETS),
        pooling_type=args.pooling_type,
        num_conv_blocks=args.num_conv_blocks,
        fusion_type=args.fusion_type
    )
    
    print(f"\nModel initialized with fusion type: {args.fusion_type}")
    print(f"Total parameters: {sum(p.numel() for p in model.forward_fn.parameters()):,}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, device)
        print(f"Training loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_metrics = evaluate(model, val_loader, device)
        print(f"Validation loss: {val_loss:.4f}")
        print_metrics(val_metrics, "Validation")
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_metrics'].append(val_metrics)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            model_path = os.path.join(args.output_dir, 'best_model.pth')
            model.save_model(model_path)
            print(f"Saved best model to {model_path}")
            
            # Save metrics
            with open(os.path.join(args.output_dir, 'best_metrics.json'), 'w') as f:
                json.dump(val_metrics, f, indent=2)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Learning rate decay
        if (epoch + 1) % 20 == 0:
            model.updateLR(0.5)
            print(f"Learning rate reduced by half")
    
    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Test on test set
    print("\n" + "="*50)
    print("Testing on test set...")
    
    # Load best model for testing
    model.load_model(os.path.join(args.output_dir, 'best_model.pth'))
    
    # Evaluate on test set
    test_loss, test_metrics = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")
    print_metrics(test_metrics, "Test")
    
    # Save test metrics
    with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Save final summary
    summary = {
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'model_config': {
            'channels': args.channels,
            'num_transformer_layers': args.num_transformer_layers,
            'num_heads': args.num_heads,
            'dropout': args.dropout,
            'fusion_type': args.fusion_type,
            'pooling_type': args.pooling_type,
            'num_conv_blocks': args.num_conv_blocks
        },
        'training_config': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs_trained': len(training_history['train_loss']),
            'early_stopping_patience': args.early_stopping_patience
        }
    }
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining complete! Results saved to {args.output_dir}")
    print(f"Final test R²: {test_metrics['r2']:.4f}")

if __name__ == "__main__":
    main()