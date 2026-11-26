#!/usr/bin/env bash
# d3-ultra arm under the x3 protocol: dev fan-out -> per-review tau frozen on dev (checksummed) -> test fan-out.
# Same prompts (questions), split (input/split.union.json), recall>=0.95 rule as Jev 1.13.
# Note: 1,792 test records were queried (cached, never scored) by an earlier test-first attempt on 2026-11-24.
# The cache also holds 1,434 keys matching no current record (origin unknown; never read by scoring).
# Needs VLLM_SR_API_KEY (fresh calls) or the cache d3u/d3u_cache.jsonl (all hits, no key needed).
# Run from paper/data/1_screening/code: bash d3u_run.sh > d3u_run.log 2>&1   (then: python3 d3u_arm.py)
# Study-relative paths below (runs/d3u/..) are mapped to d3u/.. by layout.py.
set -u
cd "$(dirname "$0")"  # code/
export X3_MODEL=vllm-sr/d3-ultra TYPESAFE_BASE_URL=https://decision.vllm-sr.ai JEV_KEY_ENV=VLLM_SR_API_KEY \
       X3_CACHE=runs/d3u/d3u_cache.jsonl X3_RUNTAG=d3u JEV_TOKEN_CAP=1000000000 X3_WORKERS=${X3_WORKERS:-6}
fan() {  # retry on 429 / transient errors; cache makes reruns resume
  for t in $(seq 1 60); do
    python3 fanout.py "$1" && return 0
    echo "[$(date -Is)] fanout $1 failed (try $t); backoff 90s"; sleep 90  # omp-allow:H6 429 backoff inside detached job
  done; return 1
}
fan dev || exit 1
X3_DEV=runs/d3u/d3u_dev.json X3_SPLIT=data/processed/split.union.json X3_TAU=runs/d3u/dev_tau.d3u.json python3 sweep.py || exit 1
echo "[$(date -Is)] TAU FROZEN sha256 $(sha256sum ../d3u/dev_tau.d3u.json)"
fan test || exit 1
echo "[$(date -Is)] DONE"
