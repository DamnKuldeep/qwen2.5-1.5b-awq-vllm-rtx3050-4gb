# smoke_test.ps1 — one real generation against the running vLLM server.
#
# Stage 1 closing check. Not a benchmark (that's Stage 2) and not a contract
# test (that's Stage 5) — this only answers "does the model actually produce
# coherent tokens", so that if Stage 2's numbers look strange we already know
# the engine itself is sound.
#
# Uses Invoke-RestMethod rather than curl: in PowerShell, `curl` is an alias for
# Invoke-WebRequest and silently mangles curl's own flag syntax. Building the
# body as a hashtable and letting ConvertTo-Json handle escaping also avoids the
# nested-quote problems of inlining JSON on a Windows command line.
#
# Requires the server from run_vllm.ps1 to be running in another terminal.
#
# Usage:   .\deployment\docker\smoke_test.ps1

# Ask vLLM which model it is serving rather than hardcoding one. Hits vLLM
# directly (port 8000), which needs no auth. A hardcoded name 404s the moment
# the engine serves something else.
$Model = "Qwen/Qwen2.5-3B-Instruct-AWQ"
try {
    $id = (Invoke-RestMethod -Uri "http://localhost:8000/v1/models" -UseBasicParsing).data[0].id
    if ($id) { $Model = $id }
} catch { }
Write-Host "Serving model: $Model" -ForegroundColor DarkGray

$body = @{
    model    = $Model
    messages = @(
        @{ role = "user"; content = "In two sentences, explain what a KV cache does during LLM inference." }
    )
    # Sampling is pinned explicitly rather than inherited. The server logs a
    # warning that Qwen's HF generation config overrides vLLM's defaults with
    # creative settings (temperature 0.7, top_p 0.8, top_k 20). Deterministic
    # output is what we want for a repeatable check.
    temperature = 0
    max_tokens  = 120
} | ConvertTo-Json -Depth 5

$response = Invoke-RestMethod `
    -Uri "http://localhost:8000/v1/chat/completions" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body

Write-Host "`n--- Response ---`n"
Write-Host $response.choices[0].message.content
Write-Host "`n--- Usage ---`n"
$response.usage | Format-List
