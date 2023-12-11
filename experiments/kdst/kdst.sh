#!/bin/bash
# SBATCH --job-name=kdst
# SBATCH --partition=bigbatch
# SBATCH -o /experiments/slurm .% N .% j. out
# SBATCH -e /experiments/slurm .% N .% j. err
python /experiments/kdst/kdst.py --vit_model_id google/vit-base-patch16-224-in21k \
--student_name ResNet18 \
--dataset_id cifar100 \
--batch_size 16 \
--alpha 0.50 \
--beta 2.0 \
--gamma 2.0 \
--tau 4.0 \
--epochs 20 \
--learning_rate 0.000075
