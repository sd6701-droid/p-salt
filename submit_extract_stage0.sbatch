#!/bin/bash
#SBATCH --job-name=salt_extract_k400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --output=/scratch/sd6701/salt/p-salt/logs/extract_%j.log
#SBATCH --error=/scratch/sd6701/salt/p-salt/logs/extract_%j.err

set -euo pipefail

WORKDIR=/scratch/sd6701/salt/p-salt

source /scratch/sd6701/miniconda3/etc/profile.d/conda.sh
conda activate salt

export LD_LIBRARY_PATH=/scratch/sd6701/miniconda3/envs/salt/lib:${LD_LIBRARY_PATH:-}

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  module load squashfs-tools 2>/dev/null || module load squashfs 2>/dev/null || true
fi

mkdir -p "${WORKDIR}/logs"
mkdir -p "${WORKDIR}/data/kinetics400_subset"

cd "${WORKDIR}"

echo "unsquashfs path: $(command -v unsquashfs || echo not-found)"
echo "sqfscat path: $(command -v sqfscat || echo not-found)"

python3 app/main.py --config configs/extract_kinetics400_local_sqf.yaml
