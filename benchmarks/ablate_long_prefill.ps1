# Ablation: --long-prefill-token-threshold against head-of-line blocking.
#
# Failure-matrix case 3 measured a single ~20k-token prompt pushing concurrent
# short requests from ~80 ms to ~10,000 ms of TTFT: vLLM 0.11's V1 scheduler
# keeps one prefill in flight and gives it the whole 2,048-token step budget.
# The threshold caps how many tokens one long prefill may take per step, which
# leaves the rest of the step for new arrivals.
#
# A lower threshold helps case 3 more, but it also splits EVERY prompt longer
# than the threshold into more steps - so it has to be measured on ordinary
# traffic too, not only on the worst case it was chosen for. Each threshold
# gets the same four workloads, and the first threshold is repeated last as a
# thermal-drift control.
#
#   powershell -File benchmarks/ablate_long_prefill.ps1

param(
    [int[]]$Thresholds = @(0, 512, 1024, 0),
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
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$compose = @("compose", "-f", "deployment/docker/docker-compose.yml",
             "-f", "deployment/docker/docker-compose.1_5b-awq.yml")

function Gpu($label) {
    $l = (nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader)
    Write-Host ("[gpu] {0,-26} {1}" -f $label, $l) -ForegroundColor DarkCyan
}

function Restart-Engine($t) {
    $env:VLLM_LONG_PREFILL_THRESHOLD = "$t"
    & docker @compose up -d --force-recreate vllm | Out-Null
    Start-Sleep -Seconds 15
    do {
        Start-Sleep -Seconds 5
        $h = (docker inspect --format "{{.State.Health.Status}}" vllm)
    } while ($h -ne "healthy")
    # The first requests after a restart pay CUDA-graph and allocator warm-up
    # (measured: 4,252 ms for a "Say hello"). Absorb it before measuring.
    & $py load_testing/chat_sim.py --users 10 --duration 30 --turns 4 --think-mean 2 --max-tokens 64 --seed 3 | Out-Null
}

$seen = @{}
foreach ($t in $Thresholds) {
    $tag = if ($seen.ContainsKey($t)) { "t${t}_control" } else { "t$t" }
    $seen[$t] = $true
    Write-Host "`n=== long-prefill-token-threshold = $t ($tag) ===" -ForegroundColor Cyan
    Restart-Engine $t
    if ($tag -eq "t$($Thresholds[0])") {
        # Heat soak before the first measured set only; the card then stays hot.
        & $py load_testing/chat_sim.py --users 30 --duration 95 --turns 8 --think-mean 4 --max-tokens 192 --seed 7 | Out-Null
    }
    Gpu "start $tag"

    & $py load_testing/failure_matrix.py --case long_prompt --out "$OutDir/${tag}_case3.json"
    & $py load_testing/chat_sim.py --users 20 --duration $Duration --turns 6 --think-mean 12 --pattern steady `
        --seed 42 --out "$OutDir/${tag}_capacity_20users.json" --label "${tag}_capacity_20users"
    foreach ($len in 2000, 8000) {
        & $py load_testing/chat_sim.py --users 20 --duration $Duration --turns 6 --think-mean 12 --pattern steady `
            --conversation-mix "$len" --seed 42 --out "$OutDir/${tag}_conv_${len}tok.json" --label "${tag}_conv_${len}tok"
    }
    Gpu "end $tag"
}
Remove-Item Env:VLLM_LONG_PREFILL_THRESHOLD -ErrorAction SilentlyContinue
# Leave the engine on the SHIPPED default, not on whichever arm ran last.
& docker @compose up -d --force-recreate vllm | Out-Null
Write-Host "`nAblation written to $OutDir (engine restored to the shipped threshold)" -ForegroundColor Green
