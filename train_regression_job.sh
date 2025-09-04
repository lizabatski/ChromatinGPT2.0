#!/bin/bash
#SBATCH --job-name=train_regression
#SBATCH --output=logs/train_regression_%j.out
#SBATCH --error=logs/train_regression_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1              
#SBATCH --account=def-majewski    
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=elizabeth.kourbatski@mail.mcgill.ca


module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2


source ~/ChromatinGPT2.0/myenv/bin/activate

python experiments/regression/train_regression.py \
    --train_data data/enformer_style_chr22p_128bp.npz \
    --output_dir logs/train_regression_run \
    --epochs 5 \
    --batch_size 16 \
    --early_stopping_patience 2 \
    --val_split 0.2 \
    --test_split 0.2 \
    --seed 42
