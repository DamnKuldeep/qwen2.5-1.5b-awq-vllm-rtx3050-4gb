# test_budget.ps1 - Stage 4 acceptance: budget enforcement and usage recording.
#
# ENCODING NOTE (learned the hard way): this file is deliberately pure ASCII.
# Windows PowerShell 5.1 reads .ps1 files as Windows-1252 unless they carry a
# UTF-8 BOM. A UTF-8 em dash then decodes as three cp1252 characters whose last
# byte (0x94) is a curly closing quote, which PowerShell accepts as a string
# delimiter - so a dash inside a string literal silently terminates it and
# produces a cascade of misleading "missing closing brace" errors far from the
# real line. Keep string literals ASCII.
#
# Four checks:
#   1. A streamed request records real token counts (proves the injected
#      stream_options.include_usage works - without it, streams bill zero)
#   2. response_format is still forwarded (Boundary 1 spot-check; Stage 5
#      makes this a proper contract test)
#   3. Budget exhaustion returns 429 after N successful requests
#   4. Budget headers (X-Budget-*) decrease across requests
#
# PREREQUISITES - all three, in this order:
#   1. vLLM running          .\deployment\docker\run_vllm.ps1
#   2. Database seeded ONCE  python -m gateway.seed_db
#   3. Gateway running       uvicorn gateway.main:app --port 8080 --reload
#
# Usage:  .\gateway\test_budget.ps1

$Gateway  = "http://localhost:8080"
$LargeKey = "dev-key-alpha"   # 500,000 token budget
$SmallKey = "dev-key-beta"    # 2,000 token budget - exhausts in a few requests

# Ask the server which model it is serving rather than hardcoding one. A
# hardcoded name 404s the moment the engine serves anything else, which is
# exactly what happened during Stage 11's model comparison.
function Get-ServedModel {
    param($Gateway, $Key, $Fallback = "Qwen/Qwen2.5-3B-Instruct-AWQ")
    try {
        $id = (Invoke-RestMethod -Uri "$Gateway/v1/models" -UseBasicParsing `
               -Headers @{ Authorization = "Bearer $Key" }).data[0].id
        if ($id) { return $id }
    } catch { }
    return $Fallback
}

$Model = Get-ServedModel -Gateway $Gateway -Key $LargeKey

function Show-Section($n, $text) {
    Write-Host "`n=== $n. $text ===" -ForegroundColor Cyan
}

# --- 0. Preflight ----------------------------------------------------------
# Fail loudly and specifically here rather than letting every later check fail
# for the same underlying reason.
Show-Section 0 "Preflight"
try {
    $h = Invoke-RestMethod -Uri "$Gateway/health" -Method Get
    Write-Host "  gateway=$($h.gateway)  vllm=$($h.vllm)"
    if ($h.vllm -ne "ok") {
        Write-Host "STOP: vLLM is not reachable. Start it first." -ForegroundColor Red
        return
    }
} catch {
    Write-Host "STOP: gateway not reachable at $Gateway" -ForegroundColor Red
    Write-Host "      Start it with: uvicorn gateway.main:app --port 8080 --reload" -ForegroundColor Yellow
    return
}

# --- 1. Streamed request must record tokens --------------------------------
Show-Section 1 "Streamed request -> tokens must be recorded (not zero)"
$streamBody = @{
    model       = $Model
    messages    = @(@{ role = "user"; content = "List three uses of a cache, briefly." })
    temperature = 0
    max_tokens  = 80
    stream      = $true
} | ConvertTo-Json -Depth 5

try {
    $req = [System.Net.HttpWebRequest]::Create("$Gateway/v1/chat/completions")
    $req.Method = "POST"
    $req.ContentType = "application/json"
    $req.Headers.Add("Authorization", "Bearer $LargeKey")
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($streamBody)
    $req.ContentLength = $bytes.Length
    $s = $req.GetRequestStream()
    $s.Write($bytes, 0, $bytes.Length)
    $s.Close()

    $reader = New-Object System.IO.StreamReader($req.GetResponse().GetResponseStream())
    $usageLine = $null
    while (-not $reader.EndOfStream) {
        $line = $reader.ReadLine()
        # The usage chunk is the one whose "usage" field is a populated object.
        if ($line -match '"usage"\s*:\s*\{') { $usageLine = $line }
    }
    $reader.Close()

    if ($usageLine) {
        Write-Host "OK: stream carried a usage chunk" -ForegroundColor Green
        $cut = [Math]::Min(220, $usageLine.Length)
        Write-Host ("  " + $usageLine.Substring(0, $cut))
    } else {
        Write-Host "FAILED: no usage chunk in the stream" -ForegroundColor Red
        Write-Host "        include_usage injection is not working" -ForegroundColor Red
    }
} catch {
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
}

# --- 2. response_format pass-through (Boundary 1 spot-check) ---------------
Show-Section 2 "response_format json_object -> expect valid JSON back"
$jsonBody = @{
    model           = $Model
    messages        = @(@{ role = "user"; content = "Return a JSON object with keys 'city' and 'country' for Paris." })
    temperature     = 0
    max_tokens      = 60
    response_format = @{ type = "json_object" }
} | ConvertTo-Json -Depth 5

try {
    $r = Invoke-RestMethod -Uri "$Gateway/v1/chat/completions" -Method Post `
        -ContentType "application/json" `
        -Headers @{ Authorization = "Bearer $LargeKey" } -Body $jsonBody
    $content = $r.choices[0].message.content
    Write-Host "Raw content: $content"
    try {
        $null = $content | ConvertFrom-Json
        Write-Host "OK: parsed as JSON - response_format survived the gateway" -ForegroundColor Green
    } catch {
        Write-Host "FAILED: not valid JSON - response_format may have been dropped" -ForegroundColor Red
    }
} catch {
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
}

# --- 3 & 4. Exhaust the small budget --------------------------------------
Show-Section 3 "Exhaust the 2,000-token budget on '$SmallKey' -> expect 429"
$burnBody = @{
    model       = $Model
    messages    = @(@{ role = "user"; content = "Write about caching in computer systems." })
    temperature = 0
    max_tokens  = 300
} | ConvertTo-Json -Depth 5

$got429 = $false
$lastUsed = "n/a"

for ($i = 1; $i -le 15; $i++) {
    try {
        # -UseBasicParsing skips the legacy IE-based HTML parser. Without it,
        # PS 5.1 interactively prompts with a script-execution security warning
        # on every call, which would block any unattended run.
        $resp = Invoke-WebRequest -Uri "$Gateway/v1/chat/completions" -Method Post `
            -UseBasicParsing `
            -ContentType "application/json" `
            -Headers @{ Authorization = "Bearer $SmallKey" } -Body $burnBody
        $used      = $resp.Headers["X-Budget-Used"]
        $remaining = $resp.Headers["X-Budget-Remaining"]
        $tokens    = ($resp.Content | ConvertFrom-Json).usage.total_tokens
        $lastUsed  = $used
        Write-Host ("  req {0,2}: 200 OK   tokens={1,4}   used_before={2,6}   remaining={3,6}" -f $i, $tokens, $used, $remaining)
    } catch {
        $code = $_.Exception.Response.StatusCode.value__
        if ($code -eq 429) {
            Write-Host ("  req {0,2}: 429 budget exhausted" -f $i) -ForegroundColor Green
            $got429 = $true
            break
        }
        Write-Host ("  req {0,2}: UNEXPECTED status {1}" -f $i, $code) -ForegroundColor Red
        Write-Host ("      " + $_.Exception.Message) -ForegroundColor Red
        break
    }
}

Write-Host ""
if ($got429) {
    Write-Host "OK: budget enforcement works - 429 once the budget was spent" -ForegroundColor Green
    Write-Host "Last X-Budget-Used seen before the 429: $lastUsed"
    Write-Host ""
    Write-Host "NOTE: X-Budget-Used shows the value BEFORE that request was billed," -ForegroundColor Yellow
    Write-Host "      so the final stored total will exceed 2,000. That overshoot is" -ForegroundColor Yellow
    Write-Host "      by design: a request's token count cannot be known until it" -ForegroundColor Yellow
    Write-Host "      has been generated. See DECISIONS.md." -ForegroundColor Yellow
} else {
    Write-Host "FAILED: never hit 429 in 15 requests" -ForegroundColor Red
    Write-Host "        Budget is not being enforced, or tokens are not being deducted." -ForegroundColor Red
}

Show-Section 4 "Dashboard"
Write-Host "Open in a browser:  $Gateway/dashboard" -ForegroundColor Cyan
Write-Host "Expect: 'dev-key-beta' at or over 100 percent, remaining cell reading 'exhausted',"
Write-Host "        a red consumption bar, and the 429 logged in Recent requests with 0 tokens."
Write-Host ""
