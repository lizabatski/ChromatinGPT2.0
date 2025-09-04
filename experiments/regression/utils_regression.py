import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple
import pyBigWig
import pyfaidx
from torch.utils.data import Dataset

# Define target names for regression
targets = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']

def one_hot_encode_dna(sequence: str) -> np.ndarray:
    """One-hot encode DNA sequence"""
    mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    seq_upper = sequence.upper()
    
    # Create one-hot encoding
    one_hot = np.zeros((4, len(seq_upper)))
    for i, base in enumerate(seq_upper):
        if base in mapping:
            one_hot[mapping[base], i] = 1
    
    return one_hot

def extract_sequence_from_fasta(fasta_file: str, chrom: str, start: int, end: int) -> str:
    """Extract DNA sequence from FASTA file"""
    fasta = pyfaidx.Fasta(fasta_file)
    sequence = fasta[chrom][start:end].seq
    return sequence

def extract_signal_from_bigwig(bigwig_file: str, chrom: str, start: int, end: int, 
                              seq_length: int) -> np.ndarray:
    """Extract signal from BigWig file"""
    bw = pyBigWig.open(bigwig_file)
    
    # Get values for the region
    try:
        values = bw.values(chrom, start, end)
        if values is None:
            values = [0.0] * (end - start)
    except:
        values = [0.0] * (end - start)
    
    bw.close()
    
    # Convert None values to 0
    values = [v if v is not None else 0.0 for v in values]
    
    # Resize to seq_length if needed
    values = np.array(values)
    if len(values) != seq_length:
        if len(values) < seq_length:
            # Pad with zeros
            pad_size = seq_length - len(values)
            values = np.pad(values, (0, pad_size), 'constant', constant_values=0)
        else:
            # Truncate
            values = values[:seq_length]
    
    return values

class BigWigDataset(Dataset):
    """Dataset for loading data from BigWig files"""
    
    def __init__(self, regions: List[Tuple], target_files: List[str], dnase_file: str, 
                 fasta_file: str, seq_length: int = 1024):
        self.regions = regions
        self.target_files = target_files
        self.dnase_file = dnase_file
        self.fasta_file = fasta_file
        self.seq_length = seq_length
        
        # Initialize FASTA reader
        self.fasta = pyfaidx.Fasta(fasta_file)
    
    def __len__(self):
        return len(self.regions)
    
    def __getitem__(self, idx):
        chrom, start, end = self.regions[idx]
        
        # Adjust region to seq_length
        center = (start + end) // 2
        region_start = center - self.seq_length // 2
        region_end = center + self.seq_length // 2
        
        # Extract DNA sequence
        try:
            sequence = self.fasta[chrom][region_start:region_end].seq
            dna_features = one_hot_encode_dna(sequence)
        except:
            # Handle chromosomes not in FASTA
            dna_features = np.zeros((4, self.seq_length))
        
        # Extract DNase signal
        dnase_signal = extract_signal_from_bigwig(
            self.dnase_file, chrom, region_start, region_end, self.seq_length
        )
        
        # Extract target signals (histone modifications)
        target_signals = []
        for target_file in self.target_files:
            signal = extract_signal_from_bigwig(
                target_file, chrom, region_start, region_end, self.seq_length
            )
            # For regression, we can use mean signal across the region
            target_signals.append(np.mean(signal))
        
        target_signals = np.array(target_signals)
        
        # Convert to tensors
        dna_tensor = torch.tensor(dna_features, dtype=torch.float32)
        dnase_tensor = torch.tensor(dnase_signal, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        target_tensor = torch.tensor(target_signals, dtype=torch.float32)
        
        return dna_tensor, dnase_tensor, target_tensor

def loadRegionsFromBigWig(regions: List[Tuple], target_files: List[str], dnase_file: str, 
                         fasta_file: str, seq_length: int = 1024) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load DNA, DNase, and target data for given regions from BigWig files"""
    
    dna_regions = []
    dnase_regions = []
    target_regions = []
    
    fasta = pyfaidx.Fasta(fasta_file)
    
    for chrom, start, end in regions:
        # Adjust region to seq_length
        center = (start + end) // 2
        region_start = center - seq_length // 2
        region_end = center + seq_length // 2
        
        # Extract DNA sequence
        try:
            sequence = fasta[chrom][region_start:region_end].seq
            dna_features = one_hot_encode_dna(sequence)
        except:
            dna_features = np.zeros((4, seq_length))
        
        # Extract DNase signal
        dnase_signal = extract_signal_from_bigwig(
            dnase_file, chrom, region_start, region_end, seq_length
        )
        
        # Extract target signals
        target_signals = []
        for target_file in target_files:
            signal = extract_signal_from_bigwig(
                target_file, chrom, region_start, region_end, seq_length
            )
            target_signals.append(np.mean(signal))  # Use mean for regression
        
        dna_regions.append(dna_features)
        dnase_regions.append(dnase_signal)
        target_regions.append(target_signals)
    
    return (np.array(dna_regions), 
            np.array(dnase_regions), 
            np.array(target_regions))

def model_train(model, train_loader, device: str) -> float:
    """Training function using DataLoader"""
    model.forward_fn.train()
    train_losses = []
    
    for batch_idx, (dna_batch, dnase_batch, target_batch) in enumerate(train_loader):
        if device == 'cuda':
            dna_batch = dna_batch.cuda()
            dnase_batch = dnase_batch.cuda()
            target_batch = target_batch.cuda()
        
        # Convert to numpy for the model's expected input format
        dna_np = dna_batch.cpu().numpy()
        dnase_np = dnase_batch.cpu().numpy()
        target_np = target_batch.cpu().numpy()
        
        loss = model.train_on_batch(dna_np, dnase_np, target_np)
        train_losses.append(loss)
    
    return np.mean(train_losses)

def model_eval(model, eval_loader, device: str) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluation function using DataLoader"""
    model.forward_fn.eval()
    losses = []
    all_preds = []
    all_targets = []
    
    for batch_idx, (dna_batch, dnase_batch, target_batch) in enumerate(eval_loader):
        if device == 'cuda':
            dna_batch = dna_batch.cuda()
            dnase_batch = dnase_batch.cuda()
            target_batch = target_batch.cuda()
        
        # Convert to numpy for the model's expected input format
        dna_np = dna_batch.cpu().numpy()
        dnase_np = dnase_batch.cpu().numpy()
        target_np = target_batch.cpu().numpy()
        
        loss, pred = model.eval_on_batch(dna_np, dnase_np, target_np)
        
        losses.append(loss)
        all_preds.append(pred)
        all_targets.append(target_np)
    
    avg_loss = np.mean(losses)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return avg_loss, all_targets, all_preds

def model_predict(model, test_loader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Prediction function using DataLoader"""
    model.forward_fn.eval()
    all_preds = []
    all_targets = []
    
    for batch_idx, (dna_batch, dnase_batch, target_batch) in enumerate(test_loader):
        if device == 'cuda':
            dna_batch = dna_batch.cuda()
            dnase_batch = dnase_batch.cuda()
        
        # Convert to numpy for the model's expected input format
        dna_np = dna_batch.cpu().numpy()
        dnase_np = dnase_batch.cpu().numpy()
        target_np = target_batch.cpu().numpy()
        
        pred = model.test_on_batch(dna_np, dnase_np)
        
        all_preds.append(pred)
        all_targets.append(target_np)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return all_targets, all_preds

def calculate_regression_metrics(targets_true: np.ndarray, targets_pred: np.ndarray, 
                               target_idx: int) -> Tuple[float, float, float]:
    """Calculate MSE, MAE, and R² for a single target"""
    try:
        y_true = targets_true[:, target_idx]
        y_pred = targets_pred[:, target_idx]
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return mse, mae, r2
    except Exception as e:
        print(f"ERROR in calculate_regression_metrics for target {target_idx}: {e}")
        return 0.0, 0.0, 0.0

def regression_metrics(targets_true: np.ndarray, targets_pred: np.ndarray, 
                      phase: str = 'Test', loss: float = None) -> Dict[str, float]:
    """Calculate regression metrics for all targets"""
    
    print(f'\n--- {phase} Results (REGRESSION) ---')
    if loss is not None:
        print(f'Loss: {loss:.4f}')
    
    per_target_mse = []
    per_target_mae = []
    per_target_r2 = []
    
    print(f'{"Target":15s} {"MSE":>8s} {"MAE":>8s} {"R²":>8s}')
    print('-' * 50)
    
    for i, target in enumerate(targets):
        if i < targets_true.shape[1] and i < targets_pred.shape[1]:
            mse, mae, r2 = calculate_regression_metrics(targets_true, targets_pred, i)
            per_target_mse.append(mse)
            per_target_mae.append(mae)
            per_target_r2.append(r2)
            print(f'{target:15s} {mse:8.4f} {mae:8.4f} {r2:8.4f}')
        else:
            per_target_mse.append(0.0)
            per_target_mae.append(0.0)
            per_target_r2.append(0.0)
            print(f'{target:15s} {"0.0000":>8s} {"0.0000":>8s} {"0.0000":>8s} (missing)')
    
    # Calculate overall metrics
    overall_mse = mean_squared_error(targets_true.flatten(), targets_pred.flatten())
    overall_mae = mean_absolute_error(targets_true.flatten(), targets_pred.flatten())
    overall_r2 = r2_score(targets_true.flatten(), targets_pred.flatten())
    
    mean_mse = np.mean(per_target_mse)
    mean_mae = np.mean(per_target_mae)
    mean_r2 = np.mean(per_target_r2)
    
    print('-' * 50)
    print(f'{"Overall":15s} {overall_mse:8.4f} {overall_mae:8.4f} {overall_r2:8.4f}')
    print(f'{"Mean":15s} {mean_mse:8.4f} {mean_mae:8.4f} {mean_r2:8.4f}')
    print('-' * 50)
    
    return {
        'mse': overall_mse,
        'mae': overall_mae,
        'r2': overall_r2,
        'mean_mse': mean_mse,
        'mean_mae': mean_mae,
        'mean_r2': mean_r2,
        'per_target_mse': per_target_mse,
        'per_target_mae': per_target_mae,
        'per_target_r2': per_target_r2
    }