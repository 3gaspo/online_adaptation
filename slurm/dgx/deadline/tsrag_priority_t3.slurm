#!/bin/bash
#SBATCH --job-name=dl_tsrag_p1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATASETS_OVERRIDE=null
RANGES_OVERRIDE=null
EXPERIMENT_MODE=full
EXPERIMENT_FAMILY=deadline_tsrag_comparison
DEADLINE_PART=priority
DEADLINE_FINALIZE=false
QUERY_STRIDE="${QUERY_STRIDE:-127}"
STAGES="${STAGES:-extract,adapt,tables}"
export PROJECT_ROOT DATASETS_OVERRIDE RANGES_OVERRIDE EXPERIMENT_MODE
export EXPERIMENT_FAMILY DEADLINE_PART DEADLINE_FINALIZE QUERY_STRIDE STAGES
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/src/slurm/run_family.sh"
