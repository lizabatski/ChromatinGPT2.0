#!/usr/bin/env python3
"""
Recreate BigWig files from extracted NPZ data for validation
"""
import numpy as np
import pyBigWig
import argparse
import os
from typing import List, Tuple

def recreate_bigwig_from_npz(npz_file: str, 
                            output_dir: str, 
                            chrom_sizes_file: str,
                            target_names: List[str] = None,
                            include_dnase: bool = True) -> None:
    """
    Recreate BigWig files from preprocessed NPZ data
    
    Args:
        npz_file: Path to NPZ file with extracted data
        output_dir: Directory to save recreated BigWig files
        chrom_sizes_file: Chromosome sizes file
        target_names: Names for target tracks
        include_dnase: Whether to also recreate DNase track
    """
    
    # Load data
    print(f"Loading data from {npz_file}")
    data = np.load(npz_file, allow_pickle=True)
    
    dna = data['dna']
    dnase = data['dnase'] 
    targets = data['targets']
    coords = data['coords']
    seq_length = int(data['seq_length'])
    
    # Get metadata
    aggregation = str(data.get('aggregation', 'unknown'))
    bin_size = int(data.get('bin_size', 128))
    
    print(f"Data shapes: DNA={dna.shape}, DNase={dnase.shape}, Targets={targets.shape}")
    print(f"Aggregation: {aggregation}, Bin size: {bin_size}")
    print(f"Sequence length: {seq_length}")
    
    # Default target names
    if target_names is None:
        if 'target_names' in data:
            target_names = data['target_names'].tolist()
        else:
            target_names = [f'Target_{i}' for i in range(targets.shape[1])]
    
    # Load chromosome sizes
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
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Recreate DNase track
    if include_dnase:
        print("Recreating DNase track...")
        dnase_file = os.path.join(output_dir, "recreated_dnase.bw")
        recreate_single_track(dnase, coords, chrom_sizes, dnase_file, 
                             seq_length, bin_size, aggregation, "DNase")
    
    # Recreate target tracks
    for target_idx in range(targets.shape[1]):
        target_name = target_names[target_idx] if target_idx < len(target_names) else f'Target_{target_idx}'
        print(f"Recreating {target_name} track...")
        
        target_file = os.path.join(output_dir, f"recreated_{target_name}.bw")
        target_data = targets[:, target_idx]
        
        recreate_single_track(target_data, coords, chrom_sizes, target_file,
                             seq_length, bin_size, aggregation, target_name)
    
    print(f"\nRecreated BigWig files saved to: {output_dir}")
    print(f"Load these alongside your original BigWigs in a genome browser to compare")

def recreate_single_track(track_data: np.ndarray, 
                         coords: np.ndarray,
                         chrom_sizes: dict,
                         output_file: str,
                         seq_length: int,
                         bin_size: int, 
                         aggregation: str,
                         track_name: str) -> None:
    """
    Recreate a single BigWig track from extracted data
    """
    
    # Create BigWig file
    bw = pyBigWig.open(output_file, "w")
    
    # Add header with chromosome sizes
    chrom_list = [(chrom, size) for chrom, size in chrom_sizes.items()]
    bw.addHeader(chrom_list)
    
    print(f"  Processing {len(coords)} regions for {track_name}...")
    
    # Process each region
    for region_idx, (chrom, start, end) in enumerate(coords):
        region_data = track_data[region_idx]
        
        if aggregation in ['mean', 'max']:
            # Single value per region - create constant signal across region
            if not np.isnan(region_data) and region_data != 0:
                bw.addEntries([chrom], [start], [end], [float(region_data)])
        
        elif aggregation == 'full':
            # Binned data - reconstruct full resolution
            if region_data.ndim == 0:
                # Handle case where it's accidentally a scalar
                if not np.isnan(region_data) and region_data != 0:
                    bw.addEntries([chrom], [start], [end], [float(region_data)])
            else:
                # Handle binned data
                num_bins = len(region_data)
                
                chroms = []
                starts = []
                ends = []
                values = []
                
                for bin_idx, bin_value in enumerate(region_data):
                    if np.isnan(bin_value) or bin_value == 0:
                        continue  # Skip zero/nan values to keep file smaller
                    
                    bin_start = start + (bin_idx * bin_size)
                    bin_end = min(start + ((bin_idx + 1) * bin_size), end)
                    
                    # Make sure we don't exceed chromosome boundaries
                    if chrom in chrom_sizes:
                        bin_end = min(bin_end, chrom_sizes[chrom])
                    
                    if bin_start < bin_end:  # Valid interval
                        chroms.append(chrom)
                        starts.append(bin_start)
                        ends.append(bin_end)
                        values.append(float(bin_value))
                
                # Add all intervals for this region at once
                if chroms:
                    bw.addEntries(chroms, starts, ends, values)
        
        else:
            print(f"  Warning: Unknown aggregation method: {aggregation}")
    
    bw.close()
    print(f"  Saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Recreate BigWig files from NPZ data for validation')
    
    parser.add_argument('--npz_file', required=True, help='Input NPZ file with extracted data')
    parser.add_argument('--output_dir', required=True, help='Output directory for recreated BigWig files')
    parser.add_argument('--chrom_sizes', required=True, help='Chromosome sizes file')
    parser.add_argument('--target_names', nargs='*', 
                       help='Names for target tracks (default: use names from NPZ or Target_N)')
    parser.add_argument('--no_dnase', action='store_true', help='Skip recreating DNase track')
    
    args = parser.parse_args()
    
    recreate_bigwig_from_npz(
        npz_file=args.npz_file,
        output_dir=args.output_dir, 
        chrom_sizes_file=args.chrom_sizes,
        target_names=args.target_names,
        include_dnase=not args.no_dnase
    )
    
    print("\nValidation steps:")
    print("1. Load original BigWig files in genome browser (IGV/UCSC)")
    print("2. Load recreated BigWig files from output directory")
    print("3. Navigate to specific regions and compare tracks visually")
    print("4. Check for:")
    print("   - Signal levels match")
    print("   - Peak positions align") 
    print("   - No coordinate shifts")
    print("   - Boundary handling looks correct")

if __name__ == "__main__":
    main()