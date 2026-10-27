#!/bin/bash
# Waits (polling every 30 s, give up after 8 h) for the GPU rule to allow a DS4.1 job, then runs the window once.
REPO=$(cd "$(dirname "$0")/../.." && pwd)
for i in $(seq 1 960); do
    "$REPO/tools/ds41kern/gpu_window.sh" "${1:-$REPO/scratch/gpu}" && exit 0
    rc=$?; [ "$rc" -ne 3 ] && [ "$rc" -ne 4 ] && exit $rc
    sleep 30
done
echo "gave up"; exit 5
