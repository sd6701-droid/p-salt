#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=salt_teacher_extracted
#SBATCH --output=/scratch/sd6701/salt/p-salt/random_mask/logs/teacher_extracted_%j.log
#SBATCH --error=/scratch/sd6701/salt/p-salt/random_mask/logs/teacher_extracted_%j.err

set -euo pipefail

WORKDIR=/scratch/sd6701/salt/p-salt
TRAIN_ROOT=/scratch/sd6701/salt/p-salt/data/kinetics400_subset_2/videos/kinetics_256/train_256

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

source /scratch/sd6701/miniconda3/etc/profile.d/conda.sh
conda activate salt

export LD_LIBRARY_PATH=/scratch/sd6701/miniconda3/envs/salt/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export=online
export WANDB__SERVICE_WAIT=300

mkdir -p "${WORKDIR}/logs"

if [[ ! -d "${TRAIN_ROOT}" ]]; then
  echo "expected extracted train root not found: ${TRAIN_ROOT}" >&2
  echo "run Stage 0 extraction first or submit via submit_extract_then_teacher.sh" >&2
  exit 1
fi

cd "${WORKDIR}"

python3 app/main.py --config configs/teacher_kinetics400_local_extracted.yaml
