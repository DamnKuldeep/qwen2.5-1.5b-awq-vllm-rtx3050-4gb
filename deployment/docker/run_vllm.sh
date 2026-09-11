#!/usr/bin/env bash
# run_vllm.sh — launch the tuned vLLM server (Linux / WSL / bash)
#
# Line-for-line mirror of run_vllm.ps1, which is the version actually used on
# this Windows machine. This file exists because Stage 7 (Docker Compose) and
# Stage 12 (Kubernetes) both target Linux, and the flag set below is what gets
# translated into those manifests. Keep the two files in sync — if you change a
# flag in one, change it in the other.
#
# See run_vllm.ps1 and DECISIONS.md for the reasoning behind each flag.
#
# Usage:   ./deployment/docker/run_vllm.sh
set -euo pipefail

VLLM_IMAGE="vllm/vllm-openai:v0.11.0"
MODEL="Qwen/Qwen2.5-3B-Instruct-AWQ"
REVISION="3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd"   # pinned at the end of Stage 1; see run_vllm.ps1
CONTAINER_NAME="vllm-server"

# See run_vllm.ps1 for the full reasoning. Short version: vLLM checks this
# fraction against FREE memory at startup, and Windows/WDDM reserves ~0.79 GiB
# of the 4 GiB card, capping this at ~0.80. Note that a real Linux host would
# not have the WDDM reservation and could run this higher.
GPU_MEM_UTIL="0.78"

docker run --rm -it \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v hf-cache:/root/.cache/huggingface \
  -v vllm-cache:/root/.cache/vllm \
  "$VLLM_IMAGE" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --tokenizer-revision "$REVISION" \
  --quantization awq_marlin \
  --dtype float16 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-prefix-caching \
  --swap-space 2 \
  --host 0.0.0.0 \
  --port 8000
