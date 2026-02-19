#!/bin/bash
#
# Merge LoRA adapter weights into the base model.
#
# Usage:
#   ./scripts/merge.sh
#   ./scripts/merge.sh --adapter-path ./outputs/checkpoint-500 --output-path ./outputs/merged-500
#   ./scripts/merge.sh --push-to-hub --hub-repo-id your-name/your-model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

ADAPTER_PATH="${PROJECT_DIR}/outputs/final"
OUTPUT_PATH="${PROJECT_DIR}/outputs/merged"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "Warning: Default adapter path not found: $ADAPTER_PATH"
    echo "Checking for alternative paths..."

    LATEST_CHECKPOINT=$(ls -d "${PROJECT_DIR}/outputs/checkpoint-"* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        ADAPTER_PATH="$LATEST_CHECKPOINT"
        echo "Found latest checkpoint: $ADAPTER_PATH"
    else
        echo "Error: No adapter or checkpoint found in ./outputs/"
        echo "Run training first: ./scripts/train.sh"
        exit 1
    fi
fi

echo "============================================================"
echo "  LoRA Weight Merging"
echo "============================================================"
echo "  Adapter:  $ADAPTER_PATH"
echo "  Output:   $OUTPUT_PATH"
echo "============================================================"

python -m src.merge_model \
    --adapter-path "$ADAPTER_PATH" \
    --output-path "$OUTPUT_PATH" \
    "$@"

echo ""
echo "Merged model saved to: $OUTPUT_PATH"
echo ""
echo "To run inference:"
echo "  python -m src.inference --model-path $OUTPUT_PATH --interactive"
