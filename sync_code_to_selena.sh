#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
NNI_FILE="$HOME/codes/.secrets/nni"

if [ ! -f "$NNI_FILE" ]; then
    echo "ERROR: missing $NNI_FILE" >&2
    exit 1
fi

NNI="$(sed -n '1p' "$NNI_FILE" | tr -d '[:space:]')"
nni="${NNI,,}"
if [[ ! "$nni" =~ ^[a-z][a-z0-9_-]*$ ]]; then
    echo "ERROR: $NNI_FILE must contain one valid NNI" >&2
    exit 1
fi

SOURCE_ROOT="$PROJECT_ROOT/"
DESTINATION="$nni@selena.hpc.edf.fr:~/codes/$PROJECT_NAME/"
SELENA_HOST="$nni@selena.hpc.edf.fr"
SCRATCH_PROJECT_ROOT="/scratch/users/$nni/codes/$PROJECT_NAME"

if [ ! -f "$SOURCE_ROOT/README.md" ] || [ ! -f "$SOURCE_ROOT/sync_code_to_selena.sh" ]; then
    echo "ERROR: project root not found at $SOURCE_ROOT" >&2
    exit 1
fi

echo "Synchronizing $PROJECT_NAME code from DGX to Selena..."
rsync -rlptz --delete --partial --info=progress2 \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='.secrets/' \
    --exclude='pyproject.toml' \
    --exclude='uv.lock' \
    --exclude='datasets/' \
    --exclude='weights/' \
    --exclude='outputs/' \
    --exclude='logs/' \
    --include='outputs_selena/' \
    --include='outputs_selena/.gitkeep' \
    --exclude='outputs_selena/***' \
    --include='logs_selena/' \
    --include='logs_selena/.gitkeep' \
    --exclude='logs_selena/***' \
    "$SOURCE_ROOT" \
    "$DESTINATION"

ssh "$SELENA_HOST" \
    "mkdir -p '$SCRATCH_PROJECT_ROOT/outputs_selena' '$SCRATCH_PROJECT_ROOT/logs_selena'"

echo "SUCCESS: Selena's $PROJECT_NAME code matches DGX."
echo "Selena results root: $SCRATCH_PROJECT_ROOT"
echo "Preserved on Selena: .venv, .secrets, pyproject.toml, uv.lock, datasets, weights, outputs, and logs."
