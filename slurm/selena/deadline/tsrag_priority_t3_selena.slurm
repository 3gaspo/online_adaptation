#!/bin/bash
#SBATCH --job-name=s_dl_tsrag
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/%u/codes/online_adaptation/logs_selena/%x_%j.out
#SBATCH --error=/scratch/users/%u/codes/online_adaptation/logs_selena/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=an
#SBATCH --qos=an_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000
#SBATCH --exclusive
#SBATCH --wckey=P12CU:DATASCIENCE

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATASETS_OVERRIDE=null
RANGES_OVERRIDE=null
EXPERIMENT_MODE=full
EXPERIMENT_FAMILY=deadline_tsrag_comparison
DEADLINE_PART=priority
DEADLINE_FINALIZE=false
QUERY_STRIDE="${QUERY_STRIDE:-127}"
STAGES="${STAGES:-extract,adapt,tables}"
source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"
EXPERIMENT_LAUNCH_ID="selena_${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}"
export PROJECT_ROOT DATASETS_OVERRIDE RANGES_OVERRIDE EXPERIMENT_MODE
export EXPERIMENT_FAMILY DEADLINE_PART DEADLINE_FINALIZE QUERY_STRIDE STAGES
export OUTPUTS_ROOT LOGS_ROOT EXPERIMENT_LAUNCH_ID
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/src/slurm/run_family.sh"
