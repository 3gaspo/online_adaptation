#!/bin/bash
#SBATCH --job-name=s_abl_backbones
#SBATCH --gres=gpu:1
#SBATCH --output=logs_selena/%x_%j.out
#SBATCH --error=logs_selena/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=an
#SBATCH --qos=an_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --wckey=P12CU:DATASCIENCE

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
EXPERIMENT_FAMILY=backbone_ablation
OUTPUTS_ROOT="$PROJECT_ROOT/outputs_selena"
LOGS_ROOT="$PROJECT_ROOT/logs_selena"
EXPERIMENT_LAUNCH_ID="selena_${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}"
# TiRex-2 remains adapter-supported but is commented out of the task profile.
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY OUTPUTS_ROOT LOGS_ROOT EXPERIMENT_LAUNCH_ID
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/src/slurm/run_family.sh"
