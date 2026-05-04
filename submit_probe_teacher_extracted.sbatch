#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=salt_probe_extracted
#SBATCH --output=/scratch/sd6701/salt/p-salt/logs/probe_teacher_%j.log
#SBATCH --error=/scratch/sd6701/salt/p-salt/logs/probe_teacher_%j.err

set -euo pipefail

WORKDIR=/scratch/sd6701/salt/p-salt
TRAIN_MANIFEST=/scratch/sd6701/salt/p-salt/data/kinetics400_subset_2/train_256_manifest.txt
VAL_MANIFEST=/scratch/sd6701/salt/p-salt/data/kinetics400_subset_2/val_256_manifest.txt
TEACHER_CHECKPOINT=/scratch/sd6701/salt/p-salt/results/checkpoints/ba0e9395/best/checkpoint.pth
PROBE_CHECKPOINT_DIR=/scratch/sd6701/salt/p-salt/results/probe_checkpoints/teacher_ba0e9395

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

source /scratch/sd6701/miniconda3/etc/profile.d/conda.sh
conda activate salt

export LD_LIBRARY_PATH=/scratch/sd6701/miniconda3/envs/salt/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Uncomment to enable wandb without editing the yaml.
# export WANDB_API_KEY=...
# export WANDB_MODE=online
# export WANDB_PROJECT=rethinking-jepa

mkdir -p "${WORKDIR}/logs"
mkdir -p "${PROBE_CHECKPOINT_DIR}"

if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  echo "expected training manifest not found: ${TRAIN_MANIFEST}" >&2
  exit 1
fi

if [[ ! -f "${VAL_MANIFEST}" ]]; then
  echo "expected validation manifest not found: ${VAL_MANIFEST}" >&2
  exit 1
fi

if [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
  echo "expected teacher checkpoint not found: ${TEACHER_CHECKPOINT}" >&2
  exit 1
fi

cd "${WORKDIR}"

python3 app/main.py --config configs/probe_teacher_kinetics400_local_extracted.yaml
