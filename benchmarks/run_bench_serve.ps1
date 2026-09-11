# run_bench_serve.ps1 — online benchmark against the running vLLM server.
#
# Stage 2 baseline, reused by Stage 9 (concurrency ramp) and Stage 11
# (before/after optimization tables). Parameterized so that every run differs
# only in the values passed here — re-typing a long flag list by hand is how
# benchmark comparisons quietly become invalid.
#
# Requires the server from run_vllm.ps1 running in another terminal.
#
# Usage:
#   .\benchmarks\run_bench_serve.ps1                      # concurrency 1 (baseline)
#   .\benchmarks\run_bench_serve.ps1 -Concurrency 8       # batched
#   .\benchmarks\run_bench_serve.ps1 -Concurrency 4 -NumPrompts 64

param(
    # Requests in flight at once. 1 gives TTFT/ITL with no queueing mixed in.
    [int]$Concurrency = 1,

    # Total requests sent. Must be comfortably larger than $Concurrency so the
    # measurement covers steady state, not just ramp-up and drain. At
    # concurrency 8, 32 prompts is only four waves and the average is dominated
    # by ramp-up/drain — use 128 or more for concurrent runs.
    [int]$NumPrompts = 32,

    # Synthetic prompt/response sizes. Fixed rather than sampled from a real
    # dataset so run-to-run variance doesn't mix with the effect being measured.
    [int]$InputLen = 512,
    [int]$OutputLen = 128,

    # Pinned so the same prompts are generated on every run.
    [int]$Seed = 42,

    # Which model to request. Empty means "ask the engine what it is serving",
    # which is what you want after a model swap - a hardcoded name 404s.
    [string]$Model = ""
)

if (-not $Model) {
    $Model = "Qwen/Qwen2.5-3B-Instruct-AWQ"
    try {
        $id = (Invoke-RestMethod -Uri "http://localhost:8000/v1/models" -UseBasicParsing).data[0].id
        if ($id) { $Model = $id }
    } catch { }
}

# The benchmark client runs inside the vLLM image. The HF cache volume is
# mounted only so the tokenizer (needed to count tokens) is read from disk
# instead of re-downloaded on every run.
#
# --gpus all is required even though this client does no GPU work. Without it
# Docker does not map the driver libraries into the container, so libcuda.so.1
# is missing, vLLM resolves the platform as "UnspecifiedPlatform", and the
# `vllm` entrypoint dies while BUILDING ITS ARGUMENT PARSER — it constructs
# every subparser including `serve`, whose parser instantiates VllmConfig
# defaults, which require a resolvable device. The failure happens at import
# time, before any benchmark code runs.
#
# This does not meaningfully compete with the server for VRAM: no model is
# loaded and no KV cache is allocated here, only device detection.
#
# host.docker.internal is Docker Desktop's name for the Windows host, which is
# where the server's port 8000 is published. If it fails to resolve, add
#   --add-host=host.docker.internal:host-gateway
# to the docker run flags below.

Write-Host "`n=== bench serve | concurrency=$Concurrency prompts=$NumPrompts in=$InputLen out=$OutputLen seed=$Seed ===`n"

docker run --rm `
    --gpus all `
    -v hf-cache:/root/.cache/huggingface `
    --entrypoint vllm `
    vllm/vllm-openai:v0.11.0 `
    bench serve `
    --backend openai-chat `
    --base-url http://host.docker.internal:8000 `
    --endpoint /v1/chat/completions `
    --model $Model `
    --dataset-name random `
    --random-input-len $InputLen `
    --random-output-len $OutputLen `
    --num-prompts $NumPrompts `
    --max-concurrency $Concurrency `
    --seed $Seed `
    --ignore-eos `
    --temperature 0 `
    --percentile-metrics "ttft,tpot,itl,e2el"

# --ignore-eos and --temperature 0 are both required for the numbers to be
# comparable across runs, and the first benchmark attempt proved it.
#
# Without --ignore-eos, the model stops at an EOS token, so --random-output-len
# is a ceiling rather than a quantity. The random dataset feeds it gibberish
# prompts, which it answers briefly: two runs asking for 32 x 128 = 4096 tokens
# produced 2,683 and 2,823 tokens respectively. With --ignore-eos, every request
# generates exactly $OutputLen tokens, so every run does identical work.
#
# Without --temperature 0, the server applies Qwen's HF generation config
# defaults (temperature 0.7, top_p 0.8, top_k 20 — logged as a warning at
# startup), so sampling is stochastic and EOS lands in a different place each
# time. That is why the same --seed produced different token counts above.
#
# Note this overrides sampling only for benchmarking. The server keeps Qwen's
# recommended defaults, which are the right choice for the chat UI in Stage 6.
