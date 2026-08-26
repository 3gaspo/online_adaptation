#!/bin/bash
set -euo pipefail

case "${EXPERIMENT_MODE:-test}" in
  test|small|full) ;;
  *) echo "EXPERIMENT_MODE must be test, small, or full" >&2; exit 2 ;;
esac

case "${EXPERIMENT_FAMILY:?}" in
  main|online_gates|n_store_ablation|n_fit_ablation|fit_stride_ablation|alpha_ablation|k_ablation|l_ablation|h_ablation|feature_design_ablation|formulation_ablation|fixed_protocol_ablation|sota_backbone_ablation|general_scope_ablation|homogeneous_ablation|backbone_ablation) ;;
  *) echo "unknown EXPERIMENT_FAMILY=$EXPERIMENT_FAMILY" >&2; exit 2 ;;
esac

if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "activate the project environment or create $PROJECT_ROOT/.venv before submission" >&2
  exit 1
fi

resolve_shared_root() {
  local explicit="$1" name="$2" candidate
  local entries=()
  if [ -n "$explicit" ]; then
    realpath "$explicit"
    return
  fi
  # Keep the archived Adaptation lookup order: local, immediate parent, then
  # the nested-workspace shared parent.
  for candidate in "$PROJECT_ROOT/$name" "$PROJECT_ROOT/../$name" "$PROJECT_ROOT/../../../$name"; do
    if [ -d "$candidate" ]; then
      shopt -s nullglob
      entries=("$candidate"/*)
      shopt -u nullglob
      if [ "${#entries[@]}" -gt 0 ]; then
        realpath "$candidate"
        return
      fi
    fi
  done
  echo "cannot locate $name; set ${name^^}_ROOT" >&2
  return 1
}

DATA_ROOT="$(resolve_shared_root "${DATA_ROOT:-}" datasets)"
WEIGHTS_ROOT="$(resolve_shared_root "${WEIGHTS_ROOT:-}" weights)"
STAGES="${STAGES:-extract,adapt,tables}"
if [ "${EXPERIMENT_MODE:-test}" = test ]; then
  PROFILE_N_STORE="${N_STORE:-30000}"
  PROFILE_N_FIT="${N_FIT:-100}"
  PROFILE_PURPOSE="${PURPOSE:-smoke}"
  PROFILE_QUERY_STRIDE="${QUERY_STRIDE:-257}"
  PROFILE_CATBOOST_ITERATIONS="${CATBOOST_ITERATIONS:-2}"
else
  PROFILE_N_STORE="${N_STORE:-30000}"
  PROFILE_N_FIT="${N_FIT:-100}"
  PROFILE_PURPOSE="${PURPOSE:-publication}"
  PROFILE_QUERY_STRIDE="${QUERY_STRIDE:-127}"
  PROFILE_CATBOOST_ITERATIONS="${CATBOOST_ITERATIONS:-300}"
fi
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/outputs"

export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error

COMMON_ARGS=(
  "family=$EXPERIMENT_FAMILY"
  "mode=${EXPERIMENT_MODE:-test}"
  "data_root=$DATA_ROOT"
  "weights_root=$WEIGHTS_ROOT"
  "outputs_root=$PROJECT_ROOT/outputs"
  "device=${DEVICE:-cuda}"
  "seed=${SEED:-1}"
  "purpose=$PROFILE_PURPOSE"
  "n_store=$PROFILE_N_STORE"
  "n_fit=$PROFILE_N_FIT"
  "fitting_scope=${FITTING_SCOPE:-same_user}"
  "alpha=${ALPHA:-0.01}"
  "max_k=${MAX_K:-20}"
  "candidate_k_grid=${CANDIDATE_K_GRID:-[1,5,10,15]}"
  "used_k=${USED_K:-null}"
  "tune_alpha=${TUNE_ALPHA:-true}"
  "ridge_validation_ratio=${RIDGE_VALIDATION_RATIO:-0.2}"
  "ridge_alpha_grid=${RIDGE_ALPHA_GRID:-[0.1,0.01,0.001]}"
  "tsrag_k=${TSRAG_K:-5}"
  "query_stride=$PROFILE_QUERY_STRIDE"
  "store_stride=${STORE_STRIDE:-0}"
  "fit_stride=${FIT_STRIDE:-0}"
  "align_period=${ALIGN_PERIOD:-true}"
  "period=${PERIOD:-0}"
  "retrieval_covariate_mode=${RETRIEVAL_COVARIATE_MODE:-null}"
  "catboost_iterations=$PROFILE_CATBOOST_ITERATIONS"
  "conflict_policy=${RUN_CONFLICT_POLICY:-overwrite_exact}"
  "skip_completed=${SKIP_COMPLETE:-true}"
)

IFS=',' read -r -a REQUESTED_STAGES <<< "$STAGES"
for stage in "${REQUESTED_STAGES[@]}"; do
  case "$stage" in
    extract) module=src.scripts.extract ;;
    adapt) module=src.scripts.adapt ;;
    tables) module=src.scripts.tables ;;
    *) echo "STAGES contains unknown stage: $stage" >&2; exit 2 ;;
  esac
  srun --ntasks=1 python -m "$module" "${COMMON_ARGS[@]}"
done
