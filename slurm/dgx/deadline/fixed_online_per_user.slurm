#!/bin/bash
#SBATCH --job-name=dl_online_user
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
DATASETS_OVERRIDE='[Electricity,Solar]'
RANGES_OVERRIDE='[short]'
EXPERIMENT_MODE=small
EXPERIMENT_FAMILY=deadline_fixed_protocol
DEADLINE_PART=online_per_user
DEADLINE_FINALIZE=false
QUERY_STRIDE="${QUERY_STRIDE:-127}"
STAGES="${STAGES:-extract,adapt}"
export PROJECT_ROOT DATASETS_OVERRIDE RANGES_OVERRIDE EXPERIMENT_MODE
export EXPERIMENT_FAMILY DEADLINE_PART DEADLINE_FINALIZE QUERY_STRIDE STAGES
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/src/slurm/run_family.sh"
