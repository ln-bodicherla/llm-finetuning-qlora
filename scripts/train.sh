#!/bin/bash
#
# Launch QLoRA fine-tuning with the default configuration.
#
# Usage:
#   ./scripts/train.sh
#   ./scripts/train.sh --base-model mistralai/Mistral-7B-v0.1
#   ./scripts/train.sh --resume ./outputs/checkpoint-500

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

CONFIG="${PROJECT_DIR}/configs/training_config.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "  QLoRA Fine-tuning"
echo "============================================================"
echo "  Config:      $CONFIG"
echo "  Working dir: $PROJECT_DIR"
echo "  Python:      $(python --version)"
echo "  PyTorch:     $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA:        $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'not found')"
echo "============================================================"

python -m src.trainer --config "$CONFIG" "$@"
