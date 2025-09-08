#!/usr/bin/env python3
"""
Extract and preprocess genomic data for per-histone models

This script extracts:
- DNA sequences at base-pair resolution (one-hot encoded)
- DNase accessibility at base-pair resolution
- Histone ChIP-seq signals at base-pair resolution (inputs)
- Histone ChIP-seq signals binned to 128 bins (targets)

Usage:
    python extract_per_histone.py \
        --bed_file regions.bed \
        --fasta_file genome.fa \
        --dnase_file dnase.bw \
        --histone_files h3k4me1.bw h3k4me3.bw h3k27me3.bw h3k36me3.bw h3k9me3.bw h3k9ac.bw h3k27ac.bw \
        --output_prefix per_histone_data
"""

import numpy as np
import pyBigWig
import pyfaidx
import argparse
import os
from typing import List, Tuple, Optional
from tqdm import tqdm

# Histone modification names in order
HISTONE_NAMES = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']

def one_hot_encode_dna(sequence: str) -> np.ndarray:
    """One-hot encode DNA sequence"""
    mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    seq_upper = sequence.upper()
    
    one_hot = np.zeros((4, len(seq_upper)))
    for i, base in enumerate(seq_upper):
        if base in mapping:
            one_hot[mapping[base], i] = 1
        # Handle N's and other ambiguous bases - leave as zeros
    
    return one_hot

def extract_signal_from_bigwig(bigwig_file: str, chrom: str, start: int, end: int, 
                              expected_length: int) -> Optional[np.ndarray]:
    """Extract signal from BigWig file, return None if NaNs found"""
    bw = pyBigWig.open(bigwig_file)
    
    try:
        values = bw.values(chrom, start, end)
        if values is None:
            bw.close()
            return None  # No data available
    except Exception:
        bw.close()
        return None  # Error reading data
    
    bw.close()
    
    # Convert None values to 0, but keep as list for NaN checking
    values = [v if v is not None else 0.0 for v in values]
    values = np.array(values, dtype=np.float32)
    
    # Check for NaNs/infs BEFORE any conversion
    if np.isnan(values).any() or np.isinf(values).any():
        return None  # Signal has NaNs/infs, reject this region
    
    # Ensure correct length
    if len(values) != expected_length:
        if len(values) < expected_length:
            pad_size = expected_length - len(values)
            values = np.pad(values, (0, pad_size), 'constant', constant_values=0.0)
        else:
            values = values[:expected_length]
    
    return values

def bin_signal(signal: np.ndarray, num_bins: int) -> np.ndarray:
    """Bin signal into specified number of bins by averaging"""
    seq_len = len(signal)
    bin_size = seq_len // num_bins
    
    binned = []
    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size
        
        # Handle last bin (might be slightly larger due to rounding)
        if i == num_bins - 1:
            end_idx = seq_len
        
        bin_data = signal[start_idx:end_idx]
        
        if len(bin_data) == 0:
            bin_mean = 0.0
        else:
            bin_mean = np.mean(bin_data)
        
        # Check for NaN/inf in binned result
        if np.isnan(bin_mean) or np.isinf(bin_mean):
            return None  # Return None to indicate this region should be skipped
        
        binned.append(bin_mean)
    
    return np.array(binned, dtype=np.float32)

def load_regions_from_bed(bed_file: str) -> List[Tuple[str, int, int]]:
    """Load genomic regions from BED file"""
    regions = []
    with open(bed_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                regions.append((chrom, start, end))
    return regions

def load_chromosome_sizes(chrom_sizes_file: str) -> dict:
    """Load chromosome sizes from file"""
    chrom_sizes = {}
    with open(chrom_sizes_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                chrom = parts[0]
                size = int(parts[1])
                chrom_sizes[chrom] = size
    return chrom_sizes

def create_genome_tiles(chrom_sizes: dict, chromosomes: List[str], 
                       seq_length: int, step_size: int) -> List[Tuple[str, int, int]]:
    """Create genome-wide tiles"""
    regions = []
    
    for chrom in chromosomes:
        if chrom not in chrom_sizes:
            print(f"Warning: {chrom} not found in chromosome sizes")
            continue
            
        chrom_size = chrom_sizes[chrom]
        
        # Create tiles across this chromosome
        pos = 0
        while pos + seq_length <= chrom_size:
            regions.append((chrom, pos, pos + seq_length))
            pos += step_size
        
        # Handle the last window if there's remaining sequence
        if pos < chrom_size and chrom_size - pos > seq_length // 2:
            start = max(0, chrom_size - seq_length)
            regions.append((chrom, start, start + seq_length))
    
    return regions

def process_regions(regions: List[Tuple[str, int, int]], 
                   histone_files: List[str], 
                   dnase_file: str,
                   fasta_file: str, 
                   seq_length: int = 131072,
                   num_target_bins: int = 128,
                   center_regions: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process genomic regions to extract features and targets for per-histone models
    
    Returns:
        dna_data: (n_samples, 4, seq_length) - one-hot DNA
        dnase_data: (n_samples, seq_length) - DNase signal at base resolution
        histone_data: (n_samples, 7, seq_length) - all histone signals at base resolution
        histone_targets: (n_samples, 7, num_target_bins) - binned histone targets
        coords: (n_samples, 3) - coordinates for each region
    """
    
    dna_data = []
    dnase_data = []
    histone_data = []
    histone_targets = []
    coords = []
    
    print(f"Processing {len(regions)} regions...")
    fasta = pyfaidx.Fasta(fasta_file)
    
    valid_regions = 0
    skipped_regions = 0
    
    for idx, (chrom, start, end) in enumerate(tqdm(regions, desc="Processing regions")):
        
        if center_regions:
            # Center the region and adjust to seq_length
            center = (start + end) // 2
            region_start = center - seq_length // 2
            region_end = center + seq_length // 2
        else:
            # Use regions as-is (for genome tiling)
            region_start = start
            region_end = end
        
        # Ensure we don't go negative
        if region_start < 0:
            region_start = 0
            region_end = seq_length
        
        # Check if we have the chromosome in the fasta
        if chrom not in fasta:
            print(f"Warning: {chrom} not found in FASTA file, skipping")
            skipped_regions += 1
            continue
            
        # Check bounds against chromosome size
        chrom_len = len(fasta[chrom])
        if region_end > chrom_len:
            if chrom_len < seq_length:
                print(f"Warning: {chrom} is shorter than seq_length, skipping")
                skipped_regions += 1
                continue
            region_end = chrom_len
            region_start = max(0, region_end - seq_length)
        
        # Final length check
        if region_end - region_start != seq_length:
            print(f"Warning: Region {chrom}:{region_start}-{region_end} is not correct length, skipping")
            skipped_regions += 1
            continue
        
        # Extract DNA sequence
        try:
            sequence = fasta[chrom][region_start:region_end].seq
            if len(sequence) != seq_length:
                # Pad or truncate as needed
                if len(sequence) < seq_length:
                    sequence = sequence + 'N' * (seq_length - len(sequence))
                else:
                    sequence = sequence[:seq_length]
            dna_features = one_hot_encode_dna(sequence)
        except Exception as e:
            print(f"Warning: Could not extract sequence for {chrom}:{region_start}-{region_end}: {e}")
            skipped_regions += 1
            continue
        
        # Extract DNase signal at base resolution
        dnase_signal = extract_signal_from_bigwig(
            dnase_file, chrom, region_start, region_end, seq_length
        )
        if dnase_signal is None:
            print(f"Warning: NaN/missing DNase data for {chrom}:{region_start}-{region_end}, skipping")
            skipped_regions += 1
            continue
        
        # Extract histone signals at base resolution
        region_histone_data = []
        region_histone_targets = []
        skip_region = False
        
        for j, histone_file in enumerate(histone_files):
            # Extract full-resolution signal
            histone_signal = extract_signal_from_bigwig(
                histone_file, chrom, region_start, region_end, seq_length
            )
            
            if histone_signal is None:
                print(f"Warning: NaN/missing data in histone {j} for {chrom}:{region_start}-{region_end}, skipping")
                skip_region = True
                break
            
            # Bin the signal for targets
            binned_signal = bin_signal(histone_signal, num_target_bins)
            
            if binned_signal is None:
                print(f"Warning: NaN/inf in binning histone {j} for {chrom}:{region_start}-{region_end}, skipping")
                skip_region = True
                break
            
            region_histone_data.append(histone_signal)
            region_histone_targets.append(binned_signal)
        
        if skip_region:
            skipped_regions += 1
            continue
        
        # Store valid region
        dna_data.append(dna_features)
        dnase_data.append(dnase_signal)
        histone_data.append(np.array(region_histone_data))
        histone_targets.append(np.array(region_histone_targets))
        coords.append([chrom, region_start, region_end])
        valid_regions += 1
    
    print(f"\nProcessing summary:")
    print(f"Total regions: {len(regions)}")
    print(f"Valid regions: {valid_regions}")
    print(f"Skipped regions: {skipped_regions}")
    
    if valid_regions == 0:
        raise ValueError("No valid regions found!")
    
    # Convert to arrays
    dna_array = np.array(dna_data, dtype=np.float32)
    dnase_array = np.array(dnase_data, dtype=np.float32)
    histone_array = np.array(histone_data, dtype=np.float32)
    histone_targets_array = np.array(histone_targets, dtype=np.float32)
    coords_array = np.array(coords, dtype=object)
    
    # Final verification - should be zero NaNs
    print(f"\nFinal array NaN counts:")
    print(f"DNA: {np.isnan(dna_array).sum()} NaNs")
    print(f"DNase: {np.isnan(dnase_array).sum()} NaNs")
    print(f"Histone inputs: {np.isnan(histone_array).sum()} NaNs")
    print(f"Histone targets: {np.isnan(histone_targets_array).sum()} NaNs")
    
    print(f"\nFinal data shapes:")
    print(f"DNA: {dna_array.shape}")
    print(f"DNase: {dnase_array.shape}")
    print(f"Histone inputs: {histone_array.shape}")
    print(f"Histone targets: {histone_targets_array.shape}")
    
    return dna_array, dnase_array, histone_array, histone_targets_array, coords_array

def main():
    parser = argparse.ArgumentParser(description='Extract genomic data for per-histone models')
    
    # Input mode: either BED file OR genome-wide tiling
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--bed_file', help='BED file with genomic regions')
    input_group.add_argument('--genome_wide', action='store_true', 
                           help='Create genome-wide tiles (requires --chrom_sizes and --chromosomes)')
    
    # Required files
    parser.add_argument('--fasta_file', required=True, help='Reference genome FASTA file')
    parser.add_argument('--dnase_file', required=True, help='DNase BigWig file')
    parser.add_argument('--histone_files', nargs='+', required=True, 
                       help='BigWig files for histone marks in order: H3K4me1, H3K4me3, H3K27me3, H3K36me3, H3K9me3, H3K9ac, H3K27ac')
    
    # For genome-wide mode
    parser.add_argument('--chrom_sizes', help='Chromosome sizes file (required for --genome_wide)')
    parser.add_argument('--chromosomes', nargs='+', 
                       help='Chromosomes to process (default: chr1-chr22,chrX,chrY)',
                       default=['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY'])
    
    # Output
    parser.add_argument('--output_prefix', required=True, 
                       help='Prefix for output files')
    
    # Parameters
    parser.add_argument('--seq_length', type=int, default=131072, 
                       help='Fixed sequence length (default: 131072 = 128kb)')
    parser.add_argument('--num_target_bins', type=int, default=1024,
                       help='Number of bins for target histone signals (default: 1024)')
    parser.add_argument('--step_size', type=int, 
                       help='Step size for genome tiling (default: seq_length//2)')
    parser.add_argument('--no_center', action='store_true',
                       help='Don\'t center regions (useful for genome tiling)')
    
    args = parser.parse_args()
    
    # Set default step size
    if args.step_size is None:
        args.step_size = args.seq_length // 2  # 50% overlap by default
    
    # Validate genome-wide requirements
    if args.genome_wide and not args.chrom_sizes:
        parser.error("--genome_wide requires --chrom_sizes")
    
    # Validate number of histone files
    if len(args.histone_files) != len(HISTONE_NAMES):
        print(f"Warning: Expected {len(HISTONE_NAMES)} histone files, got {len(args.histone_files)}")
        print(f"Expected order: {', '.join(HISTONE_NAMES)}")
    
    # Load regions
    if args.bed_file:
        print(f"Loading regions from {args.bed_file}")
        regions = load_regions_from_bed(args.bed_file)
        center_regions = not args.no_center
    else:
        print(f"Creating genome-wide tiles...")
        chrom_sizes = load_chromosome_sizes(args.chrom_sizes)
        regions = create_genome_tiles(chrom_sizes, args.chromosomes, 
                                    args.seq_length, args.step_size)
        center_regions = False  # Don't center for genome tiling
        print(f"Created {len(regions)} genome-wide tiles")
    
    print(f"Processing {len(regions)} regions with seq_length={args.seq_length}")
    
    # Process data
    dna_data, dnase_data, histone_data, histone_targets, coords = process_regions(
        regions=regions,
        histone_files=args.histone_files,
        dnase_file=args.dnase_file,
        fasta_file=args.fasta_file,
        seq_length=args.seq_length,
        num_target_bins=args.num_target_bins,
        center_regions=center_regions
    )
    
    # Save preprocessed data
    output_file = f"{args.output_prefix}.npz"
    np.savez_compressed(
        output_file,
        dna=dna_data,
        dnase=dnase_data,
        histone_inputs=histone_data,    # Full resolution histone inputs
        targets=histone_targets,        # Binned histone targets
        coords=coords,
        histone_names=HISTONE_NAMES,
        seq_length=args.seq_length,
        num_target_bins=args.num_target_bins
    )
    
    print(f"\nData saved to {output_file}")
    print(f"DNA shape: {dna_data.shape}")
    print(f"DNase shape: {dnase_data.shape}")
    print(f"Histone inputs shape: {histone_data.shape}")
    print(f"Histone targets shape: {histone_targets.shape}")
    
    # Print summary statistics
    print("\nData summary statistics:")
    print(f"DNA: mean={np.mean(dna_data):.3f}, std={np.std(dna_data):.3f}")
    print(f"DNase: mean={np.mean(dnase_data):.3f}, std={np.std(dnase_data):.3f}")
    
    print("\nHistone input statistics (base resolution):")
    for i, histone_name in enumerate(HISTONE_NAMES[:len(args.histone_files)]):
        hist_data = histone_data[:, i, :]
        print(f"{histone_name:12s}: mean={np.mean(hist_data):.3f}, "
              f"std={np.std(hist_data):.3f}, "
              f"max={np.max(hist_data):.3f}")
    
    print("\nHistone target statistics (binned):")
    for i, histone_name in enumerate(HISTONE_NAMES[:len(args.histone_files)]):
        target_data = histone_targets[:, i, :]
        print(f"{histone_name:12s}: mean={np.mean(target_data):.3f}, "
              f"std={np.std(target_data):.3f}, "
              f"bins={target_data.shape[1]}")

if __name__ == "__main__":
    main()