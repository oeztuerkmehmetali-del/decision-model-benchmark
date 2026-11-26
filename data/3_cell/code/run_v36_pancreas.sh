#!/bin/bash
# s36 pancreas: v3.1 re-analysis on the CORRECTED pool (STATUS section 36).
# R=3 fresh paid builds x {jevonly, +cooc} on the 372-gene pancreas_pool_fixed.json.
# Same design as s27 (run_v31_pancreas.sh); ONLY the pool + cooc file change.
set -e
cd "$(dirname "$0")/.."
export X7_MODEL=jev-1.13.0 X7_OUTDIR="$PWD/jev/runs"
POOL=input/pool/pancreas_pool_fixed.json
COOC=input/cooc/pancreas_cooc_fixed.json
for r in 1 2 3; do
  python3 code/jev_catalog.py --tissue pancreas --pool $POOL --cooc-mode none --suffix _v36_jevonly_r$r
  python3 code/jev_catalog.py --tissue pancreas --pool $POOL --cooc $COOC --suffix _v36_cooc_r$r
done
echo V36_PANCREAS_DONE