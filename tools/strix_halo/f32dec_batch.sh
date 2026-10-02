#!/usr/bin/env bash
# Does fp32-via-fp16 help DECODE (m=1..4 GDN projections)? Log: /tmp/f32dec.log
set -u
cd ~/kyojin && source tools/strix_halo/env.sh >/dev/null 2>&1
export EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2
A=tools/strix_halo/ab.sh
echo "=== via-f16 OFF ==="
bash $A f32off "EXL3_HIP_F32OUT_VIA_F16=0" -n 512 -ndt 3 -dds -g -dc 0.6
echo "=== via-f16 ON min_rows=32 (current default) ==="
bash $A f32d32 "EXL3_HIP_F32OUT_VIA_F16=1 EXL3_HIP_F32OUT_VIA_F16_MIN_ROWS=32" -n 512 -ndt 3 -dds -g -dc 0.6
echo "=== via-f16 ON min_rows=1 (decode too) ==="
bash $A f32d1 "EXL3_HIP_F32OUT_VIA_F16=1 EXL3_HIP_F32OUT_VIA_F16_MIN_ROWS=1" -n 512 -ndt 3 -dds -g -dc 0.6
echo DONEF32DEC
