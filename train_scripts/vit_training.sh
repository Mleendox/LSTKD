#!/bin/bash
# SBATCH --job-name=vit_training
# SBATCH --partition=bigbatch
# SBATCH -o /train_scripts/slurm .% N .% j. out
# SBATCH -e /train_scripts/slurm .% N .% j. err
python /train_scripts/vit_training.py --model_id google/vit-base-patch16-224-in21k \
--dataset_id cifar100 \
--batch_size 16 \
--epochs 10 \
--learning_rate 0.0000050
