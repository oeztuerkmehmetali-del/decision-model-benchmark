#!/bin/bash
# x7 per-cell catalogue arms (v36 corrected pool) on vllm-sr/d3-ultra, the second decision model used in every
# other merged test (consistency). Same commands as d3/run_d3.sh (which used plain vllm-sr/d3; not reported).
# Outputs ONLY under d3u/runs; Jev catalogues in jev/runs are not touched. Resume-safe (cache per build).
# usage (from the case folder): bash code/d3u_run.sh
set -eu
cd "$(dirname "$0")/.."
: "${VLLM_SR_API_KEY:?set VLLM_SR_API_KEY in the environment}"
export TYPESAFE_BASE_URL=https://decision.vllm-sr.ai JEV_KEY_ENV=VLLM_SR_API_KEY JEV_TOKEN_CAP=1000000000
export X3_MODEL=vllm-sr/d3-ultra X7_MODEL=vllm-sr/d3-ultra X7_WORKERS="${X7_WORKERS:-4}"
export X7_OUTDIR="$PWD/d3u/runs"
mkdir -p "$X7_OUTDIR"
for t in pancreas heart; do
  POOL=input/pool/${t}_pool_fixed.json
  for r in 1 2 3; do
    python3 code/jev_catalog.py --tissue $t --pool $POOL --cooc-mode none --suffix _v36_jevonly_r$r
    python3 code/jev_catalog.py --tissue $t --pool $POOL --cooc input/cooc/${t}_cooc_fixed.json --suffix _v36_cooc_r$r
  done
done
echo D3U_X7_DONE
