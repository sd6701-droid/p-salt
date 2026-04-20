#!/bin/bash
#SBATCH --job-name=salt_extract_k400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --output=/scratch/sd6701/salt/p-salt/logs/extract_%j.log
#SBATCH --error=/scratch/sd6701/salt/p-salt/logs/extract_%j.err

set -euo pipefail

WORKDIR=/scratch/sd6701/salt/p-salt

source /scratch/sd6701/miniconda3/etc/profile.d/conda.sh
conda activate salt

export LD_LIBRARY_PATH=/scratch/sd6701/miniconda3/envs/salt/lib:${LD_LIBRARY_PATH:-}

mkdir -p "${WORKDIR}/logs"
mkdir -p "${WORKDIR}/data/kinetics400_subset"

cd "${WORKDIR}"

python3 app/main.py --config configs/extract_kinetics400_local_sqf.yaml
