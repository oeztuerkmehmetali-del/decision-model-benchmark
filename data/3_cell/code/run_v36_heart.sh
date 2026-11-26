#!/bin/bash
# v3.1-corrected-pool heart Jev builds (STATUS s36): R=3 fresh repeats x {jevonly, +cooc}
# on paid jev-1.13.0 against input/pool/heart_pool_fixed.json (648 genes).
# usage: run_v36_heart.sh jevonly|cooc
# (cooc arm needs input/cooc/heart_cooc_fixed.json complete; jev_catalog fails closed)
set -eu
cd "$(dirname "$0")/.."
export X7_MODEL=jev-1.13.0 X7_OUTDIR="$PWD/jev/runs"
POOL=input/pool/heart_pool_fixed.json
for r in 1 2 3; do
  if [ "$1" = jevonly ]; then
    python3 code/jev_catalog.py --tissue heart --pool $POOL --cooc-mode none --suffix _v36_jevonly_r$r
  else
    python3 code/jev_catalog.py --tissue heart --pool $POOL --cooc input/cooc/heart_cooc_fixed.json --suffix _v36_cooc_r$r
  fi
done
echo V36_HEART_${1}_DONE
