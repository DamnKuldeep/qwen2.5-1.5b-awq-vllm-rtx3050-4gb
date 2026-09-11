# run_vllm.ps1 — launch the tuned vLLM server (Windows / PowerShell)
#
# Stage 1 of PROJECT_PLAN.md. Every flag below is a deliberate choice sized
# against this machine's 4096 MiB RTX 3050 — see DECISIONS.md and the header
# comments here for the reasoning. Do not "clean up" a flag without reading why
# it is here first.
#
# Usage:   .\deployment\docker\run_vllm.ps1
# Stop:    Ctrl+C, or `docker stop vllm-server` from another terminal.

# --- Pinned versions (Boundary 2: pinned versions, reasoning written down) ---
# A floating :latest tag means a rebuild months from now silently swaps the
# engine underneath the benchmark numbers we are about to record, and the
# before/after tables in Stage 11 stop meaning anything. Pin it.
$VLLM_IMAGE = "vllm/vllm-openai:v0.11.0"

$MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"

# Model revision pinning (Boundary 2, second half). A HF repo name alone is a
# moving target — the owner can push re-quantized weights to it at any time, and
# every benchmark number recorded before that push silently stops being
# comparable to every number recorded after. This hash was read out of the
# downloaded snapshot at the end of Stage 1:
#   docker run --rm -v hf-cache:/cache --entrypoint ls vllm/vllm-openai:v0.11.0 `
#     /cache/hub/models--Qwen--Qwen2.5-3B-Instruct-AWQ/snapshots
$REVISION = "3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd"

$CONTAINER_NAME = "vllm-server"

# --- The memory dial (Stage 1, revised after first launch failure) ---
# vLLM compares this fraction against FREE memory at startup, not total. On this
# machine only ~3.21 of 4.0 GiB is free once a CUDA context exists, because
# Windows keeps the GPU in WDDM mode and reserves VRAM for the desktop
# compositor. That puts the hard ceiling at ~0.80; 0.78 leaves slack for the
# free figure drifting as the desktop redraws.
# This is the first dial to revisit in Stage 11 — raising it is the cheapest
# available source of extra KV cache.
$GPU_MEM_UTIL = "0.78"

docker run --rm -it `
  --name $CONTAINER_NAME `
  --gpus all `
  --ipc=host `
  -p 8000:8000 `
  -v hf-cache:/root/.cache/huggingface `
  -v vllm-cache:/root/.cache/vllm `
  $VLLM_IMAGE `
  --model $MODEL `
  --revision $REVISION `
  --tokenizer-revision $REVISION `
  --quantization awq_marlin `
  --dtype float16 `
  --max-model-len 4096 `
  --max-num-seqs 8 `
  --max-num-batched-tokens 2048 `
  --gpu-memory-utilization $GPU_MEM_UTIL `
  --enable-prefix-caching `
  --swap-space 2 `
  --host 0.0.0.0 `
  --port 8000
