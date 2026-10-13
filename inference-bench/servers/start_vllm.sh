#!/usr/bin/env bash
# vLLM OpenAI-compatible server on :8000 (Linux GPU VM).
#   servers/start_vllm.sh Qwen/Qwen2.5-7B-Instruct [extra vllm flags...]
# Flags change between vLLM versions: check `vllm serve --help` and keep the exact command
# (bench.run copies server_command into results/<run_id>/command.txt).
set -euo pipefail
MODEL="${1:?model id}"; shift
exec vllm serve "$MODEL" --served-model-name qwen2.5-7b --max-model-len 4096 \
     --gpu-memory-utilization 0.90 --port 8000 "$@"
