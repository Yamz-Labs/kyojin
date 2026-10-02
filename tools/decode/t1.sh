#!/bin/bash
cd $HOME/kyojin
for e in exl3_dec_dev exl3_dec_p0 exl3_dec_p1; do
  echo "##### $e"; tools/decode/t.sh tools/decode/test_dec.py --ext $e --bench
done
