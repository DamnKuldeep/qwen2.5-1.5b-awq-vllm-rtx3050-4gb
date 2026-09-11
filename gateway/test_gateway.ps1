# test_gateway.ps1 — exercise the Stage 3 gateway by hand.
#
# Five checks, in the order they should be diagnosed if something fails:
#   1. Gateway is up and can see vLLM behind it
#   2. A valid API key gets a completion
#   3. An invalid API key gets 401
#   4. A missing Authorization header gets 401
#   5. Streaming passes through incrementally rather than being buffered
#
# Requires: the vLLM server (run_vllm.ps1) AND the gateway (uvicorn) running.
#
# Usage:  .\gateway\test_gateway.ps1

$Gateway = "http://localhost:8080"
$ValidKey = "dev-key-alpha"

# Ask the server which model it is serving rather than hardcoding one.
# A hardcoded name breaks with a 404 the moment the engine serves anything
# else - which happened during Stage 11's model comparison, to this script,
# to test_budget.ps1, and to the chat UI. The gateway proxies /v1/models
# precisely so nothing downstream has to guess.
function Get-ServedModel {
    param($Gateway, $Key, $Fallback = "Qwen/Qwen2.5-3B-Instruct-AWQ")
    try {
        $id = (Invoke-RestMethod -Uri "$Gateway/v1/models" -UseBasicParsing `
               -Headers @{ Authorization = "Bearer $Key" }).data[0].id
        if ($id) { return $id }
    } catch { }
    return $Fallback
}

$Model = Get-ServedModel -Gateway $Gateway -Key $ValidKey
Write-Host "Serving model: $Model" -ForegroundColor DarkGray

function Show-Section($n, $text) {
    Write-Host "`n=== $n. $text ===" -ForegroundColor Cyan
}

# --- 1. Health -------------------------------------------------------------
Show-Section 1 "Health check (unauthenticated)"
try {
    $health = Invoke-RestMethod -Uri "$Gateway/health" -Method Get
    $health | Format-List
} catch {
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
}

# --- 2. Valid key ----------------------------------------------------------
Show-Section 2 "Valid API key -> expect a completion"
$body = @{
    model       = $Model
    messages    = @(@{ role = "user"; content = "Say 'gateway works' and nothing else." })
    temperature = 0
    max_tokens  = 20
} | ConvertTo-Json -Depth 5

try {
    $resp = Invoke-RestMethod -Uri "$Gateway/v1/chat/completions" -Method Post `
        -ContentType "application/json" `
        -Headers @{ Authorization = "Bearer $ValidKey" } `
        -Body $body
    Write-Host "Response: $($resp.choices[0].message.content)" -ForegroundColor Green
    Write-Host "Tokens:   prompt=$($resp.usage.prompt_tokens) completion=$($resp.usage.completion_tokens)"
} catch {
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
}

# --- 3. Invalid key --------------------------------------------------------
# Invoke-RestMethod throws on any non-2xx, so the 401 we WANT arrives as an
# exception. Catching it and reading the status code is the check, not an error.
Show-Section 3 "Invalid API key -> expect 401"
try {
    Invoke-RestMethod -Uri "$Gateway/v1/chat/completions" -Method Post `
        -ContentType "application/json" `
        -Headers @{ Authorization = "Bearer totally-wrong-key" } `
        -Body $body | Out-Null
    Write-Host "FAILED: request succeeded but should have been rejected" -ForegroundColor Red
} catch {
    $code = $_.Exception.Response.StatusCode.value__
    if ($code -eq 401) {
        Write-Host "OK: got 401 as expected" -ForegroundColor Green
    } else {
        Write-Host "UNEXPECTED: got $code, wanted 401" -ForegroundColor Red
    }
}

# --- 4. Missing header -----------------------------------------------------
Show-Section 4 "No Authorization header -> expect 401"
try {
    Invoke-RestMethod -Uri "$Gateway/v1/chat/completions" -Method Post `
        -ContentType "application/json" -Body $body | Out-Null
    Write-Host "FAILED: request succeeded but should have been rejected" -ForegroundColor Red
} catch {
    $code = $_.Exception.Response.StatusCode.value__
    if ($code -eq 401) {
        Write-Host "OK: got 401 as expected" -ForegroundColor Green
    } else {
        Write-Host "UNEXPECTED: got $code, wanted 401" -ForegroundColor Red
    }
}

# --- 5. Streaming ----------------------------------------------------------
# The important property is not that the text is correct but that chunks arrive
# OVER TIME. If the gateway buffered the response, every chunk would land at
# once at the end and the elapsed-time column would show a single jump.
Show-Section 5 "Streaming -> expect chunks arriving incrementally"
$streamBody = @{
    model       = $Model
    messages    = @(@{ role = "user"; content = "Count slowly from 1 to 15, one number per line." })
    temperature = 0
    max_tokens  = 100
    stream      = $true
} | ConvertTo-Json -Depth 5

try {
    $req = [System.Net.HttpWebRequest]::Create("$Gateway/v1/chat/completions")
    $req.Method = "POST"
    $req.ContentType = "application/json"
    $req.Headers.Add("Authorization", "Bearer $ValidKey")
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($streamBody)
    $req.ContentLength = $bytes.Length
    $reqStream = $req.GetRequestStream()
    $reqStream.Write($bytes, 0, $bytes.Length)
    $reqStream.Close()

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $reader = New-Object System.IO.StreamReader($req.GetResponse().GetResponseStream())
    $chunks = 0
    while (-not $reader.EndOfStream) {
        $line = $reader.ReadLine()
        if ($line -match '^data: ' -and $line -notmatch '\[DONE\]') {
            $chunks++
            if ($chunks -le 5 -or $chunks % 20 -eq 0) {
                Write-Host ("  chunk {0,3}  at {1,6} ms" -f $chunks, $sw.ElapsedMilliseconds)
            }
        }
    }
    $reader.Close()
    Write-Host "Total chunks: $chunks over $($sw.ElapsedMilliseconds) ms" -ForegroundColor Green
    Write-Host "If the ms values increase across chunks, streaming is NOT buffered." -ForegroundColor Yellow
} catch {
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
