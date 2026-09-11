# watch_gpu.ps1 — sample GPU clocks, power and throttle reasons once per second.
#
# Diagnostic for Stage 2. Run this in a THIRD terminal while a benchmark is
# running in the second, to see what state the GPU is actually in under load.
#
# The question it answers: is decode slow because of the model/engine, or
# because the hardware is being held below its rated clocks?
#
# What each column means for this project:
#   clocks.sm    - core clock. Drives PREFILL, which is compute-bound.
#   clocks.mem   - memory clock. Drives DECODE, which is bandwidth-bound: every
#                  generated token streams all ~1.95 GiB of weights from VRAM.
#                  If this is low, ITL is bad no matter what the engine does.
#   power.draw   - against the 35 W cap on this card.
#   temperature  - thermal throttling usually begins in the 80s C.
#   pstate       - P0 is maximum performance; higher numbers are lower states.
#   throttle     - the decisive column. The driver names its own reason here:
#                  SwPowerCap, HwSlowdown, SwThermalSlowdown, HwThermalSlowdown.
#
# Usage:  .\benchmarks\watch_gpu.ps1
# Stop:   Ctrl+C

Write-Host "Sampling GPU state every second. Ctrl+C to stop.`n"

nvidia-smi `
    --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,pstate,clocks_throttle_reasons.active `
    --format=csv `
    -l 1
