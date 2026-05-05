#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=salt_teacher
#SBATCH --output=/scratch/sd6701/salt/p-salt/random_mask/logs/teacher_%j.log
#SBATCH --error=/scratch/sd6701/salt/p-salt/random_mask/logs/teacher_%j.err

source /scratch/sd6701/miniconda3/etc/profile.d/conda.sh
conda activate salt

export LD_LIBRARY_PATH=/scratch/sd6701/miniconda3/envs/salt/lib:$LD_LIBRARY_PATH
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=300

mkdir -p /scratch/sd6701/salt/p-salt/logs

export VJEPA_DEBUG_MASK_COUNTS=1

python3 app/main.py --config configs/teacher_kinetics400_local_extracted.yaml
