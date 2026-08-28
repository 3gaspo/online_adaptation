#!/bin/bash

NNI_FILE="${NNI_FILE:-$HOME/codes/.secrets/nni}"
if [ ! -f "$NNI_FILE" ]; then
  echo "missing Selena NNI file: $NNI_FILE" >&2
  exit 1
fi

SELENA_NNI="$(sed -n '1p' "$NNI_FILE" | tr -d '[:space:]')"
selena_nni="${SELENA_NNI,,}"
if [[ ! "$selena_nni" =~ ^[a-z][a-z0-9_-]*$ ]]; then
  echo "the Selena NNI file must contain one valid account name" >&2
  exit 1
fi

PROJECT_NAME="$(basename "$PROJECT_ROOT")"
SELENA_SCRATCH_PROJECT_ROOT="/scratch/users/$selena_nni/codes/$PROJECT_NAME"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-$SELENA_SCRATCH_PROJECT_ROOT/outputs_selena}"
LOGS_ROOT="${LOGS_ROOT:-$SELENA_SCRATCH_PROJECT_ROOT/logs_selena}"
mkdir -p "$OUTPUTS_ROOT" "$LOGS_ROOT"
export OUTPUTS_ROOT LOGS_ROOT SELENA_SCRATCH_PROJECT_ROOT
