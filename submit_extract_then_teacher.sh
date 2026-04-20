#!/bin/bash

set -euo pipefail

stage0_job_id="$(sbatch --parsable submit_extract_stage0.sbatch)"
echo "Submitted Stage 0 extraction job: ${stage0_job_id}"

stage1_job_id="$(sbatch --parsable --dependency=afterok:${stage0_job_id} submit_teacher_stage1_extracted.sbatch)"
echo "Submitted Stage 1 teacher job: ${stage1_job_id}"
echo "Stage 1 will start only after Stage 0 completes successfully."
