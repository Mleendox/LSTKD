#!/bin/bash
# SBATCH --job-name=studentX
# SBATCH --partition=bigbatch
# SBATCH -o /train_scripts/slurm .% N .% j. out
# SBATCH -e /train_scripts/slurm .% N .% j. err
python /train_scripts/student_training.py --architecture ResNet152 \
--dataset_id cifar100 \
--epochs 35 \
--batch_size 128
--learning_rate 0.0001
