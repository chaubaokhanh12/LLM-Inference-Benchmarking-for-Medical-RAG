#!/usr/bin/env bash
# Ollama on :11434. Defaults serve few requests in parallel with a short context, which would
# make it look worse than it is: raise both before benchmarking.
set -euo pipefail
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-32}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-4096}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
export OLLAMA_MAX_QUEUE="${OLLAMA_MAX_QUEUE:-512}"
exec ollama serve
