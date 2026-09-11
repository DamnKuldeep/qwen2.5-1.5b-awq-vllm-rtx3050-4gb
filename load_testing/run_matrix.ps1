# Runs the v2 measurement matrix in one command, writing JSON per scenario.
#
#   .\load_testing\run_matrix.ps1                  # everything
#   .\load_testing\run_matrix.ps1 -Only capacity   # just the capacity sweep
#   .\load_testing\run_matrix.ps1 -Only scenarios  # just the traffic shapes
#
# WHY ONE COMMAND
# ---------------
# A matrix that takes six commands and a paragraph of instructions stops being
# run, and then the scenarios drift apart until "the burst test" means whatever
# the last person typed. Everything here is seeded and version-controlled, so a
# result from six weeks ago is comparable to one from today.
#
# THE RULES BELOW ARE NOT STYLE, THEY ARE SCAR TISSUE
# ----------------------------------------------------
#  * A heat soak runs FIRST and is discarded. Finalization Phase 2 measured two
#    runs of an identical command differing by 42% purely because one started
#    from a cooler card.
#  * The soak and the measurements are CHAINED with no pause. Phase 3 lost a
#    measurement to a 66-second gap during which the card shed 13 C - verifying
#    the thermal state is what destroyed it.
#  * The lowest capacity level is REPEATED at the end as a control. Session
#    drift biases later runs downward, and the sweep runs ascending, so drift
#    would exaggerate exactly the knee we are looking for. If the control
#    matches its original, drift did not dominate; if it does not, the sweep is
#    not trustworthy and says so.
#  * Fixed seed, so arrivals, think times and prompt content are reproducible.

param(
    [string]$Only = "all",
    [string]$OutDir = "load_testing/results",
    [int]$Duration = 120,
    [int]$Turns = 6,
    [double]$ThinkMean = 12.0,
    [int]$Seed = 42,
    # Prefix for result filenames, so an A/B of two gateway configurations does
    # not overwrite itself. Used for the queue-timeout ablation.
    [string]$Tag = "",
    # Skip the heat soak when the card is already at steady state from a run
    # that just finished - chaining is the point, and a soak between arms would
    # insert exactly the gap Phase 3 lost a measurement to.
    [switch]$NoSoak
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Show-Gpu($label) {
    $line = (nvidia-smi --query-gpu=temperature.gpu,clocks.sm,pstate --format=csv,noheader)
    Write-Host "[gpu] $label : $line" -ForegroundColor DarkCyan
}

function Invoke-Sim($simArgs, $name) {
    $full = if ($Tag) { "$Tag$name" } else { $name }
    Write-Host "`n=== $full ===" -ForegroundColor Cyan
    & $py load_testing/chat_sim.py @simArgs --out "$OutDir/$full.json" --label $full --seed $Seed
}

Show-Gpu "before soak"

# --- heat soak, discarded --------------------------------------------------
# Long enough to cross from SwPowerCap into sustained SwThermalSlowdown, which
# the telemetry showed takes about 85 seconds of load.
if (-not $NoSoak) {
    Write-Host "`n=== heat soak (discarded) ===" -ForegroundColor DarkGray
    & $py load_testing/chat_sim.py --users 30 --duration 95 --turns 8 --think-mean 4 `
        --max-tokens 192 --seed 7 | Out-Null
    Show-Gpu "after soak"
}

if ($Only -eq "all" -or $Only -eq "capacity") {
    # --- capacity sweep: the headline curve --------------------------------
    # p95 TTFT and prefix cache hit rate against simulated user count. The
    # prediction on the record is a KNEE rather than a slope, because cache
    # eviction is a threshold effect.
    foreach ($u in 10, 20, 30, 40, 60) {
        Invoke-Sim @("--users", $u, "--duration", $Duration, "--turns", $Turns,
                     "--think-mean", $ThinkMean, "--pattern", "steady") "capacity_${u}users"
        Show-Gpu "after ${u} users"
    }
    # Control: repeat the first level. If this does not reproduce
    # capacity_10users, thermal drift dominated and the sweep is not valid.
    Invoke-Sim @("--users", 10, "--duration", $Duration, "--turns", $Turns,
                 "--think-mean", $ThinkMean, "--pattern", "steady") "capacity_10users_control"
    Show-Gpu "after control"
}

if ($Only -eq "degradation") {
    # --- the overload region only ------------------------------------------
    # Used for the queue-timeout A/B. p95 TTFT under overload is bounded by
    # roughly (queue timeout + engine TTFT), so the timeout is the knob that
    # decides WHERE the flat line sits - and at 2.0 s it sits above the 1.5 s
    # SLO, which makes the degradation graceful but still out of spec.
    foreach ($u in 20, 30, 40, 60) {
        Invoke-Sim @("--users", $u, "--duration", $Duration, "--turns", $Turns,
                     "--think-mean", $ThinkMean, "--pattern", "steady") "capacity_${u}users"
        Show-Gpu "after ${u} users"
    }
}

if ($Only -eq "all" -or $Only -eq "cache") {
    # --- conversation-length ablation: isolating the PREFIX CACHE knee ------
    #
    # The user-count sweep above cannot find the cache knee, and the first run
    # showed why: hit rate went UP from 83% to 89% between 10 and 20 users,
    # because more users means more reuse of the shared system prompt, while
    # admission control caps concurrency long before the working set outgrows
    # the pool. The user sweep therefore measures the ADMISSION knee.
    #
    # This sweep holds users fixed at a level admission can serve, and grows
    # the conversations instead. Working set is users x conversation length, so
    # against a 69,760-token pool:
    #
    #     20 users x  ~1,500 tok  =   30,000   fits comfortably
    #     20 users x  ~3,300 tok  =   66,000   right at the edge
    #     20 users x  ~6,300 tok  =  126,000   ~1.8x over
    #     20 users x ~10,300 tok  =  206,000   ~3x over
    #
    # Prediction on the record: hit rate holds while the working set fits, then
    # falls sharply once it does not - a knee, because LRU eviction is a
    # threshold effect and the hash chain makes a partial hit unlikely.
    foreach ($len in 300, 1000, 2000, 4000, 8000) {
        Invoke-Sim @("--users", 20, "--duration", $Duration, "--turns", $Turns,
                     "--think-mean", $ThinkMean, "--pattern", "steady",
                     "--conversation-mix", "$len") "cache_conv${len}tok"
        Show-Gpu "after conv=${len}"
    }
}

if ($Only -eq "all" -or $Only -eq "scenarios") {
    # --- traffic shapes ----------------------------------------------------
    # Ordered least to most destructive, per the protocol.
    Invoke-Sim @("--users", 80, "--duration", $Duration, "--turns", $Turns,
                 "--think-mean", $ThinkMean, "--pattern", "burst") "shape_burst_80"
    Invoke-Sim @("--users", 60, "--duration", $Duration, "--turns", $Turns,
                 "--think-mean", $ThinkMean, "--pattern", "herd") "shape_herd_60"
    Invoke-Sim @("--users", 60, "--duration", $Duration, "--turns", $Turns,
                 "--think-mean", $ThinkMean, "--pattern", "ramp") "shape_ramp_60"
    Invoke-Sim @("--users", 10, "--duration", $Duration, "--turns", $Turns,
                 "--think-mean", $ThinkMean, "--pattern", "adversarial",
                 "--abusive-concurrency", 50) "shape_adversarial_10plus50"
    Show-Gpu "after scenarios"
}

Write-Host "`nResults in $OutDir" -ForegroundColor Green
Get-ChildItem $OutDir -Filter *.json | ForEach-Object { Write-Host "  $($_.Name)" }
