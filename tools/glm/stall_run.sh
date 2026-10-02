#!/bin/bash
# stall1 harness (local, env already sourced, cwd = worktree):
#   tools/glm/stall_run.sh <model> <outdir> <nreq> <seed>
# stall_probe + 10 Hz sampler (clocks, gpu_metrics throttle residency, vmstat) + bpftrace KFD tracer.
M=$1; O=$2; N=${3:-100}; SEED=${4:-7}
mkdir -p "$O"
sudo -n bpftrace tools/glm/stall_kfd.bt 2>&1 | while IFS= read -r l; do echo "$(date +%s.%N | cut -c1-14) $l"; done > "$O/kfd.log" &
python -u tools/glm/stall_probe.py "$M" "$O/probe.json" "$N" "$SEED" > "$O/probe.log" 2>&1 &
PP=$!
python tools/glm/stall_sampler.py $PP "$O/samp.jsonl" 10 &
tail -f "$O/probe.log" --pid=$PP | grep --line-buffered -E "^\[|^load|Traceback|Error|DONE"
wait $PP; RC=$?
sudo -n pkill -INT -f "[b]pftrace tools/glm/stall_kfd.bt"; wait
echo "stall_run rc=$RC"
