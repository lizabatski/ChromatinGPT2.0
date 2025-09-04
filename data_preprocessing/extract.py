import numpy as np
import pyBigWig
import pyfaidx
import argparse
import os
from typing import List, Tuple, Optional
from tqdm import tqdm

TARGETS = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']

def one_hot_encode_dna(sequence: str) -> np.ndarray:
    """One-hot encode DNA sequence"""
    mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    seq_upper = sequence.upper()
    
    one_hot = np.zeros((4, len(seq_upper)))
    for i, base in enumerate(seq_upper):
        if base in mapping:
            one_hot[mapping[base], i] = 1
        else:
            # Handle N's and other ambiguous bases - leave as zeros
            pass
    
    return one_hot

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
    
    # Handle None values
    values = [v if v is not None else 0.0 for v in values]
    
    # Ensure correct length
    values = np.array(values)
    if len(values) != seq_length:
        if len(values) < seq_length:
            # pad with zeros
            pad_size = seq_length - len(values)
            values = np.pad(values, (0, pad_size), 'constant', constant_values=0)
        else:
            # truncate
            values = values[:seq_length]
    
    return values

def load_chromosome_sizes(chrom_sizes_file: str) -> dict:
    """Load chromosome sizes from file"""
    chrom_sizes = {}
    with open(chrom_sizes_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                chrom = parts[0]
                size = int(parts[1])
                chrom_sizes[chrom] = size
    return chrom_sizes

def create_genome_tiles(chrom_sizes: dict, chromosomes: List[str], 
                       seq_length: int, step_size: int) -> List[Tuple[str, int, int]]:
    """Create genome-wide tiles like Enformer"""
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
        if pos < chrom_size and chrom_size - pos > seq_length // 2:  # Only if substantial sequence remains
            start = max(0, chrom_size - seq_length)
            regions.append((chrom, start, start + seq_length))
    
    return regions

def load_regions_from_bed(bed_file: str) -> List[Tuple[str, int, int]]:
    """Load genomic regions from BED file"""
    regions = []
    with open(bed_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                regions.append((chrom, start, end))
    return regions

def process_regions(regions: List[Tuple[str, int, int]], 
                   target_files: List[str], 
                   dnase_file: str,
                   fasta_file: str, 
                   seq_length: int = 131072,  # Enformer uses 128kb windows
                   aggregation: str = 'full',
                   bin_size: int = 128,
                   center_regions: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process genomic regions to extract features and targets
    """
    
    dna_data = []
    dnase_data = []
    target_data = []
    coords = []
    
    print(f"Processing {len(regions)} regions...")
    fasta = pyfaidx.Fasta(fasta_file)
    
    for chrom, start, end in tqdm(regions):
        if center_regions:
            # Center the region and adjust to seq_length
            center = (start + end) // 2
            region_start = center - seq_length // 2
            region_end = center + seq_length // 2
        else:
            # Use regions as-is (for genome tiling)
            region_start = start
            region_end = end
        
        # Ensure we don't go negative or exceed chromosome bounds
        if region_start < 0:
            region_start = 0
            region_end = seq_length
        
        # Check if we have the chromosome in the fasta
        if chrom not in fasta:
            print(f"Warning: {chrom} not found in FASTA file")
            continue
            
        # Check bounds against chromosome size
        chrom_len = len(fasta[chrom])
        if region_end > chrom_len:
            if chrom_len < seq_length:
                print(f"Warning: {chrom} is shorter than seq_length, skipping")
                continue
            region_end = chrom_len
            region_start = max(0, region_end - seq_length)
        
        coords.append((chrom, region_start, region_end))
        
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
            
            if aggregation == 'mean':
                # Use mean signal across the region (single value per target)
                target_signals.append(np.mean(signal))
            elif aggregation == 'max':
                # Use max signal across the region
                target_signals.append(np.max(signal))
            elif aggregation == 'full':
                # Bin the signal like Enformer (default behavior)
                num_bins = seq_length // bin_size
                binned = np.array([
                    np.mean(signal[i * bin_size:(i + 1) * bin_size])
                    for i in range(num_bins)
                ])
                target_signals.append(binned)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        dna_data.append(dna_features)
        dnase_data.append(dnase_signal)
        target_data.append(target_signals)
    
    return np.array(dna_data), np.array(dnase_data), np.array(target_data), np.array(coords, dtype=object)

def main():
    parser = argparse.ArgumentParser(description='Extract and preprocess genomic data (Enformer-style)')
    
    # Input mode: either BED file OR genome-wide tiling
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--bed_file', help='BED file with genomic regions')
    input_group.add_argument('--genome_wide', action='store_true', 
                           help='Create genome-wide tiles (requires --chrom_sizes and --chromosomes)')
    
    # Required files
    parser.add_argument('--fasta_file', required=True, help='Reference genome FASTA file')
    parser.add_argument('--dnase_file', required=True, help='DNase BigWig file')
    parser.add_argument('--target_files', nargs='+', required=True, 
                       help='BigWig files for histone marks (in order: H3K4me1, H3K4me3, etc.)')
    
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
                       help='Fixed sequence length (default: 131072 = 128kb like Enformer)')
    parser.add_argument('--step_size', type=int, 
                       help='Step size for genome tiling (default: seq_length//2)')
    parser.add_argument('--aggregation', choices=['mean', 'max', 'full'], default='full',
                       help='How to aggregate target signals (default: full)')
    parser.add_argument('--bin_size', type=int, default=128,
                       help='Bin size for full-resolution aggregation (default: 128)')
    parser.add_argument('--no_center', action='store_true',
                       help='Don\'t center regions (useful for genome tiling)')
    
    args = parser.parse_args()
    
    # Set default step size
    if args.step_size is None:
        args.step_size = args.seq_length // 2  # 50% overlap by default
    
    # Validate genome-wide requirements
    if args.genome_wide and not args.chrom_sizes:
        parser.error("--genome_wide requires --chrom_sizes")
    
    # Validate number of target files
    if len(args.target_files) != len(TARGETS):
        print(f"Warning: Expected {len(TARGETS)} target files, got {len(args.target_files)}")
        print(f"Expected order: {', '.join(TARGETS)}")
    
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
    dna_data, dnase_data, target_data, coords = process_regions(
        regions=regions,
        target_files=args.target_files,
        dnase_file=args.dnase_file,
        fasta_file=args.fasta_file,
        seq_length=args.seq_length,
        aggregation=args.aggregation,
        bin_size=args.bin_size,
        center_regions=center_regions
    )
    
    # Save preprocessed data
    output_file = f"{args.output_prefix}.npz"
    np.savez_compressed(
        output_file,
        dna=dna_data,
        dnase=dnase_data,
        targets=target_data,
        coords=coords,
        target_names=TARGETS,
        seq_length=args.seq_length,
        bin_size=args.bin_size,
        aggregation=args.aggregation
    )
    
    print(f"\nData saved to {output_file}")
    print(f"DNA shape: {dna_data.shape}")
    print(f"DNase shape: {dnase_data.shape}")
    print(f"Targets shape: {target_data.shape}")
    
    # Print summary statistics
    print("\nTarget summary statistics:")
    for i, target_name in enumerate(TARGETS[:len(args.target_files)]):
        if args.aggregation in ['mean', 'max']:
            target_values = target_data[:, i]
            print(f"{target_name:12s}: mean={np.mean(target_values):.3f}, "
                  f"std={np.std(target_values):.3f}, "
                  f"min={np.min(target_values):.3f}, "
                  f"max={np.max(target_values):.3f}")
        elif args.aggregation == 'full':
            # Show stats for the binned signals
            target_signals = target_data[:, i, :]  # All regions, this target, all bins
            print(f"{target_name:12s}: mean={np.mean(target_signals):.3f}, "
                  f"std={np.std(target_signals):.3f}, "
                  f"bins={target_signals.shape[1]}")

if __name__ == "__main__":
    main()