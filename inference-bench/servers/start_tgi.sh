#!/usr/bin/env bash
# Text Generation Inference (optional, E5) on :8002, OpenAI-compatible at /v1/chat/completions.
set -euo pipefail
MODEL="${1:?model id}"
exec docker run --rm --gpus all --shm-size 1g -p 8002:80 \
     -v "$HOME/.cache/huggingface:/data" \
     ghcr.io/huggingface/text-generation-inference:latest \
     --model-id "$MODEL" --max-input-tokens 3840 --max-total-tokens 4096
