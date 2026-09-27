# Produces every measured artefact in the repository, in one run.
#
#   .\benchmarks\run_evidence_suite.ps1
#
# ~60 minutes. Order is deliberate and the rules are the measurement protocol's
# (benchmarks/optimization_results.md), not preferences:
#
#   * heat soak FIRST and discarded - two runs of an identical command differed
#     by 42% purely from the temperature the card started at
#   * read-only traffic shapes before failure injection before anything that
#     kills a container
#   * Grafana captured DURING load, not after, or the panels are flat
#   * fixed seeds throughout
#
# Requires the capture profile for the Grafana PNGs:
#   docker compose --profile observability --profile capture up -d

param(
    [switch]$SkipSoak,
    [switch]$NoCapture,
    [int]$Duration = 120,
    [string]$OutDir = "load_testing/results"
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

function Gpu($label) {
    $l = (nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,pstate --format=csv,noheader)
    Write-Host ("[gpu] {0,-22} {1}" -f $label, $l) -ForegroundColor DarkCyan
}
function Step($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Sim($simArgs, $name) {
    Step $name
    & $py load_testing/chat_sim.py @simArgs --out "$OutDir/$name.json" --label $name --seed 42
}

Gpu "start"

if (-not $SkipSoak) {
    Step "heat soak (discarded - the card must be at steady state)"
    & $py load_testing/chat_sim.py --users 30 --duration 95 --turns 8 --think-mean 4 --max-tokens 192 --seed 7 | Out-Null
    Gpu "after soak"
}

$sessionStart = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - 30

# --- 1. Capacity: the curve the whole capacity model rests on ---------------
foreach ($u in 10, 20, 25, 30, 40, 60) {
    Sim @("--users",$u,"--duration",$Duration,"--turns",6,"--think-mean",12,"--pattern","steady") "evidence_capacity_${u}users"
    Gpu "after ${u} users"
}
# Control: repeat the first level last. If it does not reproduce, thermal drift
# dominated the sweep and none of it is comparable.
Sim @("--users",10,"--duration",$Duration,"--turns",6,"--think-mean",12,"--pattern","steady") "evidence_capacity_10users_control"

# --- 2. Traffic shapes a real service actually meets ------------------------
Sim @("--users",80,"--duration",$Duration,"--turns",6,"--think-mean",12,"--pattern","burst")   "evidence_shape_burst"
Sim @("--users",60,"--duration",$Duration,"--turns",6,"--think-mean",12,"--pattern","herd")    "evidence_shape_herd"
Sim @("--users",60,"--duration",$Duration,"--turns",6,"--think-mean",12,"--pattern","ramp")    "evidence_shape_ramp"
Sim @("--users",40,"--duration",$Duration,"--turns",6,"--think-mean",12,"--pattern","diurnal") "evidence_shape_diurnal"
Gpu "after shapes"

# --- 3. Conversation length: the other half of the duty cycle --------------
foreach ($len in 300, 2000, 8000) {
    Sim @("--users",20,"--duration",$Duration,"--turns",6,"--think-mean",12,"--pattern","steady",
          "--conversation-mix","$len") "evidence_conv_${len}tok"
}

# --- 4. Fairness. The attacker runs in ITS OWN PROCESS ---------------------
# Sharing an event loop between attacker and victim measured the simulator's
# own scheduling delay as server latency - a ~10x error in the direction that
# makes a working fairness mechanism look broken.
Step "evidence_adversarial (attacker isolated in a second process)"
$abuser = Start-Process -FilePath $py -ArgumentList `
    "load_testing\chat_sim.py","--abusive-only","--abusive-concurrency","50",
    "--duration","$($Duration+5)","--max-tokens","192","--out","$OutDir/evidence_abuser.json" `
    -PassThru -NoNewWindow
Start-Sleep -Seconds 3
& $py load_testing/chat_sim.py --users 10 --duration $Duration --turns 6 --think-mean 12 `
    --pattern adversarial --abusive-external --seed 42 `
    --out "$OutDir/evidence_adversarial.json" --label evidence_adversarial
try { $abuser | Wait-Process -Timeout 90 } catch { $abuser | Stop-Process -Force }

# --- 5. Capture the whole session as the dashboards and Prometheus saw it --
# An ABSOLUTE window from the first measured run to now. A relative "last 20
# minutes" capture was tried first and missed the capacity sweep entirely.
if (-not $NoCapture) {
    Step "capture dashboards and session chart"
    Start-Sleep -Seconds 20           # let the last scrape land
    $sessionEnd = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    & $py benchmarks/capture_dashboards.py --label session --start $sessionStart --end $sessionEnd
    & $py benchmarks/plot_prometheus.py --start $sessionStart --end $sessionEnd --out docs/img/session
    Gpu "after capture"
}

# --- 6. Failure injection, last because it breaks things -------------------
Step "failure matrix"
& $py load_testing/failure_matrix.py --case all --out "$OutDir/evidence_failure_matrix.json"
Step "engine crash under load"
& $py load_testing/kill_test.py --case engine --kill-mode enginecore --out "$OutDir/evidence_kill_engine.json"
Step "gateway restart under load"
& $py load_testing/kill_test.py --case gateway --out "$OutDir/evidence_kill_gateway.json"

Gpu "end"
Write-Host "`nAll evidence written to $OutDir and docs/img/" -ForegroundColor Green
