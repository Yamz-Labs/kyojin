#!/usr/bin/env bash
# fp32-out-via-fp16 for prefill: speed A/B + PPL gate. Log: /tmp/f32.log
set -u
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
for v in 0 1; do
  echo "=== EXL3_HIP_F32OUT_VIA_F16=$v : prefill 8192 ==="
  EXL3_HIP_F32OUT_VIA_F16=$v P=8192 .venv/bin/python tools/strix_halo/prefill_prof.py 2>&1 | grep -E "^prefill|^profiled"
  sleep 5
done
echo "=== PPL with the change on (gate: 4.225935) ==="
EXL3_HIP_F32OUT_VIA_F16=1 .venv/bin/python eval/ppl.py -m ~/models/Qwen3.8-Flash-Next-EXL3 -r 20 -l 1024 2>&1 | tail -3
echo "=== decode unaffected? (rows < 32 keeps the old path) ==="
for v in 0 1; do
  EXL3_HIP_F32OUT_VIA_F16=$v .venv/bin/python tools/strix_halo/bench_mtp.py -n 512 -ndt 3 -dds -g -dc 0.6 2>&1 | grep -iE "tok/s|accept" | tail -2
done
echo DONEF32
