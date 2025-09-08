#!/bin/bash
#SBATCH --job-name=make_chr22_per_histone
#SBATCH --output=logs/make_chr22_per_histone_%j.out
#SBATCH --error=logs/make_chr22_per_histone_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-majewski
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca

cd ~/ChromatinGPT2.0

source myenv/bin/activate

python data_preprocessing/extract_per_histone.py \
  --genome_wide \
  --chrom_sizes data/genome/hg19.chrom.sizes.txt \
  --chromosomes chr22 \
  --fasta_file data/genome/hg19.fa \
  --dnase_file data/bigwig/E005-DNase.fc.signal.bigwig \
  --histone_files \
    data/bigwig/E005-H3K4me1.fc.signal.bigwig \
    data/bigwig/E005-H3K4me3.fc.signal.bigwig \
    data/bigwig/E005-H3K27me3.fc.signal.bigwig \
    data/bigwig/E005-H3K36me3.fc.signal.bigwig \
    data/bigwig/E005-H3K9me3.fc.signal.bigwig \
    data/bigwig/E005-H3K9ac.fc.signal.bigwig \
    data/bigwig/E005-H3K27ac.fc.signal.bigwig \
  --output_prefix jobs/per_histone_chr22_128bp \
  --seq_length 2048 \
  --num_target_bins 16 \
  --step_size 2048

echo "chr22 per-histone dataset complete: jobs/per_histone_chr22_128bp.npz"