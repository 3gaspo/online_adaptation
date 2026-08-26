#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
SECRET_FILE="$HOME/codes/.secrets/proxy.credentials"

if [ ! -f "$SECRET_FILE" ]; then
    echo "ERROR: missing $SECRET_FILE" >&2
    exit 1
fi

NNI="$(sed -n '1p' "$SECRET_FILE" | tr -d '[:space:]')"
nni="${NNI,,}"
if [[ ! "$nni" =~ ^[a-z][a-z0-9_-]*$ ]]; then
    echo "ERROR: the first line of $SECRET_FILE must contain the NNI" >&2
    exit 1
fi

SOURCE_ROOT="$nni@selena.hpc.edf.fr:~/codes/$PROJECT_NAME"
DESTINATION_ROOT="$PROJECT_ROOT"

mkdir -p "$DESTINATION_ROOT/outputs_selena" "$DESTINATION_ROOT/logs_selena"

echo "Pulling $PROJECT_NAME Selena outputs to DGX..."
rsync -rlptz --partial --info=progress2 \
    "$SOURCE_ROOT/outputs_selena/" \
    "$DESTINATION_ROOT/outputs_selena/"

echo "Pulling $PROJECT_NAME Selena logs to DGX..."
rsync -rlptz --partial --info=progress2 \
    "$SOURCE_ROOT/logs_selena/" \
    "$DESTINATION_ROOT/logs_selena/"

echo "SUCCESS: outputs_selena and logs_selena were pulled from Selena to DGX."
