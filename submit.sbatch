#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=salt_teacher
#SBATCH --output=/scratch/sd6701/logs/teacher_%j.log

source /scratch/sd6701/miniconda3/etc/profile.d/conda.sh
conda activate salt

export LD_LIBRARY_PATH=/scratch/sd6701/miniconda3/envs/salt/lib:$LD_LIBRARY_PATH


python train_teacher.py --config configs/teacher_kinetics700_5class_10pct_vitl_random.yaml