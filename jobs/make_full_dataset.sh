#!/bin/bash
#SBATCH --job-name=create_full_epigenome
#SBATCH --output=logs/create_full_epigenome_%j.out
#SBATCH --error=logs/create_full_epigenome_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

# Create directories
mkdir -p data_preprocessing/bed_files
mkdir -p data_preprocessing/datasets
mkdir -p logs

# Parameters
WINDOW_SIZE=2048
STRIDE=2048  # No overlap for initial dataset

echo "============================================"
echo "Creating FULL RESOLUTION epigenome dataset"
echo "Window size: ${WINDOW_SIZE}bp"
echo "Output: Full signal arrays (not averaged)"
echo "============================================"

# Step 1: Create BED file for chr1-22
echo "Creating BED file for chr1-22..."
python3 << EOF
import pyBigWig

bw = pyBigWig.open("data/bigwig/E005-DNase.fc.signal.bigwig")
chroms = bw.chroms()

total_regions = 0
with open("data_preprocessing/bed_files/chr1_22_full.bed", "w") as f:
    for i in range(1, 23):
        chrom = f"chr{i}"
        if chrom in chroms:
            chrom_len = chroms[chrom]
            count = 0
            # Tile genome with no overlap
            for start in range(0, chrom_len - ${WINDOW_SIZE} + 1, ${STRIDE}):
                end = start + ${WINDOW_SIZE}
                f.write(f"{chrom}\t{start}\t{end}\n")
                count += 1
                total_regions += 1
            print(f"{chrom}: {count} regions ({chrom_len:,} bp)")

print(f"\nTotal regions: {total_regions:,}")
bw.close()
EOF

# Step 2: Extract FULL RESOLUTION data (aggregation='full')
echo ""
echo "Extracting full resolution epigenome data..."
echo "WARNING: This will create a LARGE file with shape (n_regions, 7, ${WINDOW_SIZE})"
echo ""

python3 extract.py \
    --bed_file data_preprocessing/bed_files/chr1_22_full.bed \
    --fasta_file data/genome/hg19.fa \
    --dnase_file data/bigwig/E005-DNase.fc.signal.bigwig \
    --target_files \
        data/bigwig/E005-H3K4me1.fc.signal.bigwig \
        data/bigwig/E005-H3K4me3.fc.signal.bigwig \
        data/bigwig/E005-H3K27me3.fc.signal.bigwig \
        data/bigwig/E005-H3K36me3.fc.signal.bigwig \
        data/bigwig/E005-H3K9me3.fc.signal.bigwig \
        data/bigwig/E005-H3K9ac.fc.signal.bigwig \
        data/bigwig/E005-H3K27ac.fc.signal.bigwig \
    --output_prefix data_preprocessing/datasets/chr1_22_full_resolution \
    --seq_length ${WINDOW_SIZE} \
    --aggregation full

# Step 3: Create chr22 test set with full resolution
echo ""
echo "Creating chr22 test set with full resolution..."
python3 << EOF
import pyBigWig

bw = pyBigWig.open("data/bigwig/E005-DNase.fc.signal.bigwig")
chroms = bw.chroms()

with open("data_preprocessing/bed_files/chr22_full.bed", "w") as f:
    chrom = "chr22"
    if chrom in chroms:
        chrom_len = chroms[chrom]
        count = 0
        for start in range(0, chrom_len - ${WINDOW_SIZE} + 1, ${STRIDE}):
            end = start + ${WINDOW_SIZE}
            f.write(f"{chrom}\t{start}\t{end}\n")
            count += 1
        print(f"chr22 test set: {count} regions")

bw.close()
EOF

python3 extract.py \
    --bed_file data_preprocessing/bed_files/chr22_full.bed \
    --fasta_file data/genome/hg19.fa \
    --dnase_file data/bigwig/E005-DNase.fc.signal.bigwig \
    --target_files \
        data/bigwig/E005-H3K4me1.fc.signal.bigwig \
        data/bigwig/E005-H3K4me3.fc.signal.bigwig \
        data/bigwig/E005-H3K27me3.fc.signal.bigwig \
        data/bigwig/E005-H3K36me3.fc.signal.bigwig \
        data/bigwig/E005-H3K9me3.fc.signal.bigwig \
        data/bigwig/E005-H3K9ac.fc.signal.bigwig \
        data/bigwig/E005-H3K27ac.fc.signal.bigwig \
    --output_prefix data_preprocessing/datasets/chr22_test_full_resolution \
    --seq_length ${WINDOW_SIZE} \
    --aggregation full

# Step 4: Verify the data structure
echo ""
echo "============================================"
echo "Verifying data structure..."
echo "============================================"

python3 << EOF
import numpy as np

# Load and check main dataset
print("Main dataset (chr1-22):")
data = np.load("data_preprocessing/datasets/chr1_22_full_resolution.npz")
print(f"  DNA shape: {data['dna'].shape}")
print(f"  DNase shape: {data['dnase'].shape}")
print(f"  Targets shape: {data['targets'].shape}")
print(f"  Expected: (n_regions, 7, {${WINDOW_SIZE}})")
print()

# Check test dataset
print("Test dataset (chr22):")
test_data = np.load("data_preprocessing/datasets/chr22_test_full_resolution.npz")
print(f"  DNA shape: {test_data['dna'].shape}")
print(f"  DNase shape: {test_data['dnase'].shape}")
print(f"  Targets shape: {test_data['targets'].shape}")
print()

# Memory estimate
main_size_gb = (data['dna'].nbytes + data['dnase'].nbytes + data['targets'].nbytes) / (1024**3)
test_size_gb = (test_data['dna'].nbytes + test_data['dnase'].nbytes + test_data['targets'].nbytes) / (1024**3)

print(f"File sizes:")
print(f"  Main dataset: ~{main_size_gb:.1f} GB")
print(f"  Test dataset: ~{test_size_gb:.1f} GB")
print()

# Show example of full resolution data
print("Example target signals (first region, first 20 positions):")
for i, name in enumerate(data['target_names'][:3]):  # Show first 3 targets
    signal = data['targets'][0, i, :20]
    print(f"  {name}: {signal}")
EOF

echo ""
echo "============================================"
echo "Dataset creation complete!"
echo "============================================"
echo ""
echo "Created files:"
echo "  - data_preprocessing/datasets/chr1_22_full_resolution.npz"
echo "  - data_preprocessing/datasets/chr22_test_full_resolution.npz"
echo ""
echo "Data structure:"
echo "  DNA: (n_regions, 4, ${WINDOW_SIZE}) - one-hot encoded"
echo "  DNase: (n_regions, ${WINDOW_SIZE}) - per-position signal"
echo "  Targets: (n_regions, 7, ${WINDOW_SIZE}) - per-position for each histone"
echo ""
echo "IMPORTANT: These are LARGE files with full signal resolution!"
echo "You'll need to modify your model to handle (7, ${WINDOW_SIZE}) targets"
echo "instead of 7 single values."