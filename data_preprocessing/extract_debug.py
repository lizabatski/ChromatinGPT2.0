#!/usr/bin/env python3
"""
Debug version of extract.py with extensive logging to identify issues
"""

import numpy as np
import pyBigWig
import pyfaidx
import argparse
import os
from typing import List, Tuple, Optional
from tqdm import tqdm

TARGETS = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H3K36me3', 'H3K9me3', 'H3K9ac', 'H3K27ac']

def one_hot_encode_dna(sequence: str, debug: bool = False) -> np.ndarray:
    """One-hot encode DNA sequence with debug info"""
    mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    seq_upper = sequence.upper()
    
    if debug:
        print(f"    DNA sequence length: {len(seq_upper)}")
        print(f"    First 50 bases: {seq_upper[:50]}")
        
        # count bases
        base_counts = {base: seq_upper.count(base) for base in 'ATGCN'}
        print(f"    Base counts: {base_counts}")
        
        # check for all N's
        n_count = seq_upper.count('N')
        if n_count == len(seq_upper):
            print("    *** WARNING: Sequence is all N's! ***")
        elif n_count > len(seq_upper) * 0.5:
            print(f"    *** WARNING: Sequence is {n_count/len(seq_upper)*100:.1f}% N's ***")
    
    one_hot = np.zeros((4, len(seq_upper)))
    valid_bases = 0
    
    for i, base in enumerate(seq_upper):
        if base in mapping:
            one_hot[mapping[base], i] = 1
            valid_bases += 1
    
    if debug:
        print(f"    Valid bases encoded: {valid_bases}/{len(seq_upper)} ({valid_bases/len(seq_upper)*100:.1f}%)")
        print(f"    One-hot shape: {one_hot.shape}")
        print(f"    One-hot sum: {one_hot.sum()} (should equal valid bases)")
    
    return one_hot

def debug_bigwig_file(bigwig_file: str, chrom: str, start: int, end: int):
    """Debug BigWig file access"""
    print(f"    Debugging BigWig: {os.path.basename(bigwig_file)}")
    print(f"    File exists: {os.path.exists(bigwig_file)}")
    
    try:
        bw = pyBigWig.open(bigwig_file)
        print(f"    BigWig opened successfully")
        
        # Check if chromosome exists
        chroms = bw.chroms()
        if chrom in chroms:
            print(f"    Chromosome {chrom} found, length: {chroms[chrom]}")
        else:
            print(f"    *** ERROR: Chromosome {chrom} not found in BigWig ***")
            print(f"    Available chromosomes: {list(chroms.keys())[:10]}...")
            bw.close()
            return False
        
        # Check coordinate bounds
        chrom_length = chroms[chrom]
        if start < 0 or end > chrom_length:
            print(f"    *** ERROR: Coordinates {start}-{end} out of bounds for {chrom} (length: {chrom_length}) ***")
            bw.close()
            return False
        
        print(f"    Coordinates {start}-{end} are valid")
        bw.close()
        return True
        
    except Exception as e:
        print(f"    *** ERROR opening BigWig: {e} ***")
        return False

def extract_signal_from_bigwig(bigwig_file: str, chrom: str, start: int, end: int, 
                              expected_length: int, debug: bool = False) -> Optional[np.ndarray]:
    
    if debug:
        print(f"    Extracting from {os.path.basename(bigwig_file)}")
        print(f"    Region: {chrom}:{start}-{end} (length: {end-start})")
        print(f"    Expected length: {expected_length}")
        
        # Debug file access first
        if not debug_bigwig_file(bigwig_file, chrom, start, end):
            return None
    
    try:
        bw = pyBigWig.open(bigwig_file)
        values = bw.values(chrom, start, end)
        bw.close()
        
        if values is None:
            if debug:
                print(f"    *** BigWig returned None for {chrom}:{start}-{end} ***")
            return None
        
        if debug:
            print(f"    Raw values retrieved: {len(values)} values")
            print(f"    FULL RAW VALUES (first 50):")
            print(f"    {values[:50]}")
            print(f"    FULL RAW VALUES (last 50):")
            print(f"    {values[-50:]}")
            print(f"    Sample values: {values[:10] if len(values) >= 10 else values}")
            
            # Check for any non-zero values
            non_zero_indices = [i for i, v in enumerate(values) if v != 0.0]
            print(f"    Non-zero value count: {len(non_zero_indices)}")
            if len(non_zero_indices) > 0:
                print(f"    First 10 non-zero indices: {non_zero_indices[:10]}")
                print(f"    Non-zero values: {[values[i] for i in non_zero_indices[:10]]}")
            else:
                print(f"    *** ALL VALUES ARE ZERO! ***")
            
        # Convert None values to 0
        original_nones = sum(1 for v in values if v is None)
        values = [v if v is not None else 0.0 for v in values]
        
        if debug and original_nones > 0:
            print(f"    Converted {original_nones} None values to 0.0")
        
        values = np.array(values, dtype=np.float32)
        
        # Check for NaNs/infs
        nan_count = np.isnan(values).sum()
        inf_count = np.isinf(values).sum()
        
        if debug:
            print(f"    NaN count: {nan_count}")
            print(f"    Inf count: {inf_count}")
            print(f"    Value range: {np.nanmin(values):.6f} to {np.nanmax(values):.6f}")
            print(f"    Mean value: {np.nanmean(values):.6f}")
            
        if nan_count > 0 or inf_count > 0:
            if debug:
                print(f"    *** Rejecting region due to {nan_count} NaNs and {inf_count} Infs ***")
            return None
        
        # Adjust length if needed
        if len(values) != expected_length:
            if debug:
                print(f"    Length adjustment needed: {len(values)} -> {expected_length}")
                
            if len(values) < expected_length:
                pad_size = expected_length - len(values)
                values = np.pad(values, (0, pad_size), 'constant', constant_values=0.0)
                if debug:
                    print(f"    Padded with {pad_size} zeros")
            else:
                values = values[:expected_length]
                if debug:
                    print(f"    Truncated to {expected_length}")
        
        if debug:
            print(f"    Final array: length={len(values)}, mean={np.mean(values):.6f}")
            
        return values
        
    except Exception as e:
        if debug:
            print(f"    *** Exception extracting from BigWig: {e} ***")
        return None

def process_regions(regions: List[Tuple[str, int, int]], 
                   target_files: List[str], 
                   dnase_file: str,
                   fasta_file: str, 
                   seq_length: int = 2048,
                   aggregation: str = 'full',
                   bin_size: int = 128,
                   center_regions: bool = True,
                   debug_samples: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process genomic regions with extensive debugging"""
    
    dna_data = []
    dnase_data = []
    target_data = []
    coords = []
    
    print(f"Processing {len(regions)} regions...")
    print(f"Target files: {[os.path.basename(f) for f in target_files]}")
    print(f"DNase file: {os.path.basename(dnase_file)}")
    print(f"FASTA file: {os.path.basename(fasta_file)}")
    print(f"Seq length: {seq_length}, Bin size: {bin_size}, Aggregation: {aggregation}")
    
    # Test FASTA file access
    print(f"\nTesting FASTA file access...")
    try:
        fasta = pyfaidx.Fasta(fasta_file)
        print(f"FASTA file opened successfully")
        print(f"Available chromosomes: {list(fasta.keys())[:10]}...")
    except Exception as e:
        print(f"*** ERROR: Cannot open FASTA file: {e} ***")
        return None, None, None, None
    
    # Test BigWig file access
    print(f"\nTesting BigWig file access...")
    all_files = [dnase_file] + target_files
    for i, bw_file in enumerate(all_files):
        file_type = "DNase" if i == 0 else f"Target_{i-1}"
        print(f"{file_type}: {os.path.basename(bw_file)}")
        try:
            bw = pyBigWig.open(bw_file)
            chroms = bw.chroms()
            print(f"  Opened successfully, {len(chroms)} chromosomes")
            bw.close()
        except Exception as e:
            print(f"  *** ERROR: {e} ***")
            return None, None, None, None
    
    valid_regions = 0
    skipped_regions = 0
    
    for idx, (chrom, start, end) in enumerate(tqdm(regions)):
        debug = idx < debug_samples
        
        if debug:
            print(f"\n{'='*60}")
            print(f"PROCESSING REGION {idx}: {chrom}:{start}-{end}")
            print(f"{'='*60}")
        
        if center_regions:
            center = (start + end) // 2
            region_start = center - seq_length // 2
            region_end = center + seq_length // 2
            if debug:
                print(f"  Centering: {chrom}:{start}-{end} -> {chrom}:{region_start}-{region_end}")
        else:
            region_start = start
            region_end = end
            if debug:
                print(f"  Using as-is: {chrom}:{region_start}-{region_end}")
        
        # Boundary checks
        if region_start < 0:
            if debug:
                print(f"  Adjusting negative start: {region_start} -> 0")
            region_start = 0
            region_end = seq_length
        
        if chrom not in fasta:
            if debug:
                print(f"  *** ERROR: Chromosome {chrom} not in FASTA ***")
                print(f"  Available: {list(fasta.keys())[:10]}...")
            skipped_regions += 1
            continue
            
        chrom_len = len(fasta[chrom])
        if debug:
            print(f"  Chromosome length: {chrom_len}")
            
        if region_end > chrom_len:
            if chrom_len < seq_length:
                if debug:
                    print(f"  *** ERROR: Chromosome too short ({chrom_len} < {seq_length}) ***")
                skipped_regions += 1
                continue
            region_end = chrom_len
            region_start = max(0, region_end - seq_length)
            if debug:
                print(f"  Adjusted for chromosome end: {chrom}:{region_start}-{region_end}")
        
        if region_end - region_start != seq_length:
            if debug:
                print(f"  *** ERROR: Final length mismatch: {region_end - region_start} != {seq_length} ***")
            skipped_regions += 1
            continue
        
        if debug:
            print(f"  Final coordinates: {chrom}:{region_start}-{region_end}")
        
        # Extract DNA sequence
        if debug:
            print(f"\n  EXTRACTING DNA SEQUENCE...")
            
        try:
            sequence = fasta[chrom][region_start:region_end].seq
            if debug:
                print(f"  Raw sequence length: {len(sequence)}")
                
            if len(sequence) != seq_length:
                if debug:
                    print(f"  *** Length mismatch: {len(sequence)} != {seq_length} ***")
                if len(sequence) < seq_length:
                    pad_length = seq_length - len(sequence)
                    sequence = sequence + 'N' * pad_length
                    if debug:
                        print(f"  Padded with {pad_length} N's")
                else:
                    sequence = sequence[:seq_length]
                    if debug:
                        print(f"  Truncated to {seq_length}")
                        
            dna_features = one_hot_encode_dna(sequence, debug=debug)

            n_fraction = sequence.upper().count('N') / len(sequence)
            
            if n_fraction > 0.35:
                if debug:
                    print(f"  *** SKIPPING: {n_fraction*100:.1f}% N content exceeds 35% threshold ***")
                skipped_regions += 1
                continue

            dna_features = one_hot_encode_dna(sequence, debug=debug)
            
            if debug:
                print(f"  DNA encoding successful: {dna_features.shape}")
                print(f"  DNA has signal: {dna_features.sum() > 0}")
                
        except Exception as e:
            if debug:
                print(f"  *** ERROR extracting DNA: {e} ***")
            skipped_regions += 1
            continue
        
        # Extract DNase signal
        if debug:
            print(f"\n  EXTRACTING DNASE SIGNAL...")
            
        dnase_signal = extract_signal_from_bigwig(
            dnase_file, chrom, region_start, region_end, seq_length, debug=debug
        )
        
        if dnase_signal is None:
            if debug:
                print(f"  *** DNase extraction failed ***")
            skipped_regions += 1
            continue
            
        if debug:
            print(f"  DNase extraction successful: {dnase_signal.shape}")
            print(f"  DNase has signal: {dnase_signal.sum() > 0}")
        
        # Extract target signals
        if debug:
            print(f"\n  EXTRACTING TARGET SIGNALS...")
            
        target_signals = []
        skip_region = False
        
        for j, target_file in enumerate(target_files):
            if debug:
                print(f"\n    TARGET {j}: {TARGETS[j] if j < len(TARGETS) else 'Unknown'}")
                
            signal = extract_signal_from_bigwig(
                target_file, chrom, region_start, region_end, seq_length, debug=debug
            )
            
            if signal is None:
                if debug:
                    print(f"    *** Target {j} extraction failed ***")
                skip_region = True
                break
            
            # Process based on aggregation method
            if aggregation == 'mean':
                result = np.mean(signal)
                if debug:
                    print(f"    Mean aggregation: {result}")
                target_signals.append(result)
                
            elif aggregation == 'max':
                result = np.max(signal)
                if debug:
                    print(f"    Max aggregation: {result}")
                target_signals.append(result)
                
            elif aggregation == 'full':
                num_bins = seq_length // bin_size
                if debug:
                    print(f"    Binning into {num_bins} bins of size {bin_size}")
                    
                binned = []
                for i in range(num_bins):
                    start_idx = i * bin_size
                    end_idx = (i + 1) * bin_size
                    bin_data = signal[start_idx:end_idx]
                    
                    bin_mean = np.mean(bin_data) if len(bin_data) > 0 else 0.0
                    binned.append(bin_mean)
                    
                    if debug and i < 3:  # Show first 3 bins
                        print(f"      Bin {i}: {bin_mean:.6f}")
                
                binned_array = np.array(binned)
                target_signals.append(binned_array)
                
                if debug:
                    print(f"    Binned result: {binned_array.shape}, mean={np.mean(binned_array):.6f}")
                    print(f"    Binned has signal: {binned_array.sum() > 0}")
        
        if skip_region:
            skipped_regions += 1
            continue
        
        # Store valid region
        dna_data.append(dna_features)
        dnase_data.append(dnase_signal)
        target_data.append(target_signals)
        coords.append((chrom, region_start, region_end))
        valid_regions += 1
        
        if debug:
            print(f"\n  REGION {idx} SUCCESSFULLY PROCESSED")
            print(f"  DNA sum: {dna_features.sum()}")
            print(f"  DNase sum: {dnase_signal.sum()}")
            print(f"  Target signals: {len(target_signals)}")
    
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total regions attempted: {len(regions)}")
    print(f"Valid regions processed: {valid_regions}")
    print(f"Skipped regions: {skipped_regions}")
    
    if valid_regions == 0:
        print("*** ERROR: No valid regions found! ***")
        return None, None, None, None
    
    # Convert to arrays
    print(f"\nConverting to numpy arrays...")
    dna_array = np.array(dna_data, dtype=np.float32)
    dnase_array = np.array(dnase_data, dtype=np.float32)
    target_array = np.array(target_data, dtype=np.float32)
    coords_array = np.array(coords, dtype=object)
    
    print(f"Final shapes:")
    print(f"  DNA: {dna_array.shape}")
    print(f"  DNase: {dnase_array.shape}")
    print(f"  Targets: {target_array.shape}")
    print(f"  Coords: {coords_array.shape}")
    
    # Data quality checks
    print(f"\nData quality checks:")
    print(f"  DNA - Sum: {dna_array.sum()}, NaNs: {np.isnan(dna_array).sum()}")
    print(f"  DNase - Sum: {dnase_array.sum():.3f}, NaNs: {np.isnan(dnase_array).sum()}")
    print(f"  Targets - Sum: {target_array.sum():.3f}, NaNs: {np.isnan(target_array).sum()}")
    
    # Check if data is all zeros
    if dna_array.sum() == 0:
        print("*** WARNING: DNA data is all zeros! ***")
    if dnase_array.sum() == 0:
        print("*** WARNING: DNase data is all zeros! ***")
    if target_array.sum() == 0:
        print("*** WARNING: Target data is all zeros! ***")
    
    return dna_array, dnase_array, target_array, coords_array

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
    """Create genome-wide tiles"""
    regions = []
    
    for chrom in chromosomes:
        if chrom not in chrom_sizes:
            print(f"Warning: {chrom} not found in chromosome sizes")
            continue
            
        chrom_size = chrom_sizes[chrom]
        print(f"Creating tiles for {chrom} (size: {chrom_size})")
        
        pos = 0
        tile_count = 0
        while pos + seq_length <= chrom_size:
            regions.append((chrom, pos, pos + seq_length))
            pos += step_size
            tile_count += 1
        
        print(f"  Created {tile_count} tiles for {chrom}")
        
        # Handle the last window if there's remaining sequence
        if pos < chrom_size and chrom_size - pos > seq_length // 2:
            start = max(0, chrom_size - seq_length)
            regions.append((chrom, start, start + seq_length))
            print(f"  Added final tile: {chrom}:{start}-{start + seq_length}")
    
    return regions

def load_regions_from_bed(bed_file: str) -> List[Tuple[str, int, int]]:
    """Load genomic regions from BED file"""
    regions = []
    print(f"Loading regions from {bed_file}")
    
    with open(bed_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                regions.append((chrom, start, end))
            else:
                print(f"  Warning: Skipping invalid line {line_num}: {line.strip()}")
    
    print(f"Loaded {len(regions)} regions from BED file")
    return regions

def main():
    parser = argparse.ArgumentParser(description='Extract genomic data with extensive debugging')
    
    # Input mode
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--bed_file', help='BED file with genomic regions')
    input_group.add_argument('--genome_wide', action='store_true', 
                           help='Create genome-wide tiles')
    
    # Required files
    parser.add_argument('--fasta_file', required=True, help='Reference genome FASTA file')
    parser.add_argument('--dnase_file', required=True, help='DNase BigWig file')
    parser.add_argument('--target_files', nargs='+', required=True, 
                       help='BigWig files for histone marks')
    
    # For genome-wide mode
    parser.add_argument('--chrom_sizes', help='Chromosome sizes file')
    parser.add_argument('--chromosomes', nargs='+', 
                       help='Chromosomes to process',
                       default=['chr22'])  # Default to chr22 for testing
    
    # Output
    parser.add_argument('--output_prefix', required=True, help='Output prefix')
    
    # Parameters
    parser.add_argument('--seq_length', type=int, default=2048, 
                       help='Sequence length (default: 2048)')
    parser.add_argument('--step_size', type=int, help='Step size')
    parser.add_argument('--aggregation', choices=['mean', 'max', 'full'], default='full')
    parser.add_argument('--bin_size', type=int, default=128, help='Bin size')
    parser.add_argument('--no_center', action='store_true')
    parser.add_argument('--debug_samples', type=int, default=5, 
                       help='Number of samples to debug in detail')
    
    args = parser.parse_args()
    
    # Set default step size
    if args.step_size is None:
        args.step_size = args.seq_length  # No overlap by default for debugging
    
    print(f"{'='*80}")
    print(f"GENOMIC DATA EXTRACTION - DEBUG MODE")
    print(f"{'='*80}")
    print(f"Parameters:")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  Step size: {args.step_size}")
    print(f"  Bin size: {args.bin_size}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Debug samples: {args.debug_samples}")
    
    # Validate files exist
    print(f"\nValidating input files...")
    for file_type, file_path in [
        ("FASTA", args.fasta_file),
        ("DNase", args.dnase_file)
    ] + [(f"Target_{i}", f) for i, f in enumerate(args.target_files)]:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  {file_type}: {file_path} (size: {size:,} bytes)")
        else:
            print(f"  *** ERROR: {file_type} file not found: {file_path} ***")
            return
    
    # Load regions
    if args.bed_file:
        regions = load_regions_from_bed(args.bed_file)
        center_regions = not args.no_center
    else:
        if not args.chrom_sizes:
            print("*** ERROR: --genome_wide requires --chrom_sizes ***")
            return
        chrom_sizes = load_chromosome_sizes(args.chrom_sizes)
        regions = create_genome_tiles(chrom_sizes, args.chromosomes, 
                                    args.seq_length, args.step_size)
        center_regions = False
    
    if not regions:
        print("*** ERROR: No regions to process ***")
        return
    
    print(f"\nProcessing {len(regions)} regions...")
    
    # Process data
    dna_data, dnase_data, target_data, coords = process_regions(
        regions=regions,
        target_files=args.target_files,
        dnase_file=args.dnase_file,
        fasta_file=args.fasta_file,
        seq_length=args.seq_length,
        aggregation=args.aggregation,
        bin_size=args.bin_size,
        center_regions=center_regions,
        debug_samples=args.debug_samples
    )
    
    if dna_data is None:
        print("*** EXTRACTION FAILED ***")
        return
    
    # Save data
    output_file = f"{args.output_prefix}.npz"
    print(f"\nSaving data to {output_file}...")
    
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
    
    print(f"Data saved successfully!")
    print(f"Final shapes:")
    print(f"  DNA: {dna_data.shape}")
    print(f"  DNase: {dnase_data.shape}")  
    print(f"  Targets: {target_data.shape}")
    
    # Summary statistics
    print(f"\nSummary statistics:")
    for i, target_name in enumerate(TARGETS[:len(args.target_files)]):
        if args.aggregation == 'full':
            target_signals = target_data[:, i, :]
            print(f"  {target_name}: mean={np.mean(target_signals):.6f}, "
                  f"std={np.std(target_signals):.6f}")
        else:
            target_values = target_data[:, i]
            print(f"  {target_name}: mean={np.mean(target_values):.6f}, "
                  f"std={np.std(target_values):.6f}")

if __name__ == "__main__":
    main()