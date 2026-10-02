#!/usr/bin/env bash
# A/B one env knob end to end. Usage: ab.sh LABEL "ENV=val ..." [bench args...]
# Writes /tmp/ab_LABEL.raw on the remote box; prints tok/s + acceptance + first 60 chars of output.
set -u
label="$1"; shift
envs="$1"; shift
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
env $envs .venv/bin/python tools/strix_halo/bench_mtp.py "$@" > /tmp/ab_${label}.raw 2>&1
printf "%-28s " "$label"
grep -oE "[0-9]+\.[0-9]+ tok/s|acceptance=[0-9.]+%" /tmp/ab_${label}.raw | tr "\n" " "
grep -oE "^output: .{0,70}" /tmp/ab_${label}.raw | head -1
grep -E "JOB ERROR|Traceback|Error" /tmp/ab_${label}.raw | head -2
sleep 6
