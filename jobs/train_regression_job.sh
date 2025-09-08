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

# load modules
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2

# activate virtual environment
source ~/ChromatinGPT2.0/myenv/bin/activate

# create logs directory if it doesn't exist
mkdir -p logs

# navigate to the correct directory 
cd ~/ChromatinGPT2.0

# run training
python experiments/regression/train_regression.py \
    --data data/E005_chr22.npz \
    --output_dir logs/train_regression_run \
    --epochs 5 \
    --batch_size 8 \
    --early_stopping_patience 2 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --test_ratio 0.2 \
    --save_splits \
    --num_workers 4 \
    --seed 42 \
    --channels 256 \
    --num_transformer_layers 2