# Ablation of one gateway setting, judged from the user's side.
#
# Every arm runs the same seeded chat workloads and is scored on SLO
# ATTAINMENT: the share of messages whose first token arrived within 1.5 s of
# their FIRST send, refusals and retries included. The first value is repeated
# last as a drift control.
#
#   # queue timeout (the arms in docs/RESULTS.md, tagged qt0.6 / qt1.0 / qt2.0)
#   powershell -File benchmarks/ablate_gateway.ps1 -Setting GATEWAY_QUEUE_TIMEOUT_S `
#       -Values 0.6,1.0,2.0 -Tag qt -Workloads chat20,chat40,conv8000
#
#   # admission limit, at the chosen timeout
#   powershell -File benchmarks/ablate_gateway.ps1 -Setting GATEWAY_MAX_INFLIGHT `
#       -Values 6,8,10 -Tag inflight -Fixed "GATEWAY_QUEUE_TIMEOUT_S=1.0" -Workloads chat20,chat40
#
# WHY the queue timeout needed re-measuring: 0.6 s was derived when a refusal
# looked free (worst admitted TTFT ~ timeout + engine TTFT). With clients that
# retry after Retry-After, every refusal costs its user >= 2 s, so a longer
# wait can beat an immediate 503.
#
# WHY the admission limit did: 6 was measured with ramp.py, where EVERY request
# is a cold ~512-token prefill. Real chat hits the prefix cache 70-80% of the
# time, so each request needs far less prefill, and the engine may hold more of
# them inside the TTFT SLO.

param(
    [Parameter(Mandatory)] [string]$Setting,
    [Parameter(Mandatory)] [string[]]$Values,
    [Parameter(Mandatory)] [string]$Tag,
    [string[]]$Fixed = @(),
    [string[]]$Workloads = @("chat20", "chat40"),
    [int]$Duration = 120,
    [string]$OutDir = "load_testing/results/ablation"
)

$ErrorActionPreference = "Continue"

# Keep Windows awake for the length of this script. A laptop that sleeps
# mid-run produces numbers that straddle the sleep: one ablation measured a
# 408 ms "idle" baseline and a 202 ms "under load" figure across an 8-hour
# suspend. Released automatically when this process exits. Closing the lid can
# still override it, depending on the power plan.
Add-Type -Namespace Win32 -Name Power -MemberDefinition `
    '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED (0x80000001; a hex literal parses as a negative Int32 in PS 5.1)
# `powershell -File` passes "6,8,10" as ONE string, not an array. The first run
# of this script set GATEWAY_MAX_INFLIGHT to the literal "6,8,10" and silently
# fell back to the default. Split explicitly, whichever way it was called.
$Values = ($Values -join ",") -split "," | Where-Object { $_ }
$Workloads = ($Workloads -join ",") -split "," | Where-Object { $_ }
$Fixed = ($Fixed -join ",") -split "," | Where-Object { $_ }
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$compose = @("compose", "-f", "deployment/docker/docker-compose.yml",
             "-f", "deployment/docker/docker-compose.1_5b-awq.yml")

foreach ($kv in $Fixed) {
    $k, $v = $kv -split "=", 2
    Set-Item "Env:$k" $v
}

function Gpu($label) {
    $l = (nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader)
    Write-Host ("[gpu] {0,-26} {1}" -f $label, $l) -ForegroundColor DarkCyan
}

function Apply($value) {
    Set-Item "Env:$Setting" $value
    & docker @compose up -d --no-deps --force-recreate gateway | Out-Null
    do {
        Start-Sleep -Seconds 2
        try { $ok = (Invoke-WebRequest -UseBasicParsing http://localhost:8080/ready -TimeoutSec 3).StatusCode -eq 200 }
        catch { $ok = $false }
    } while (-not $ok)
}

function Run($workload, $name) {
    $a = @("load_testing/chat_sim.py", "--duration", $Duration, "--turns", 6, "--think-mean", 12,
           "--pattern", "steady", "--seed", 42, "--out", "$OutDir/$name.json", "--label", $name)
    switch ($workload) {
        "chat20"   { $a += @("--users", 20) }
        "chat40"   { $a += @("--users", 40) }
        "conv8000" { $a += @("--users", 20, "--conversation-mix", "8000") }
    }
    & $py @a
}

Apply $Values[0]
Write-Host "heat soak (discarded)" -ForegroundColor Cyan
& $py load_testing/chat_sim.py --users 30 --duration 95 --turns 8 --think-mean 4 --max-tokens 192 --seed 7 | Out-Null

foreach ($v in $Values) {
    Write-Host "`n=== $Setting = $v ===" -ForegroundColor Cyan
    Apply $v
    Gpu "start $Tag$v"
    foreach ($w in $Workloads) {
        $name = "$Tag${v}_$($w -replace 'chat(\d+)', 'chat_${1}users' -replace 'conv(\d+)', 'conv_${1}tok')"
        Run $w $name
    }
    Gpu "end $Tag$v"
}

Write-Host "`n=== control: $Setting = $($Values[0]) ===" -ForegroundColor Cyan
Apply $Values[0]
$w = $Workloads[0]
Run $w "$Tag$($Values[0])_$($w -replace 'chat(\d+)', 'chat_${1}users' -replace 'conv(\d+)', 'conv_${1}tok')_control"

Remove-Item "Env:$Setting" -ErrorAction SilentlyContinue
foreach ($kv in $Fixed) { Remove-Item "Env:$(($kv -split '=', 2)[0])" -ErrorAction SilentlyContinue }
& docker @compose up -d --no-deps --force-recreate gateway | Out-Null
Write-Host "`nAblation of $Setting written to $OutDir" -ForegroundColor Green
