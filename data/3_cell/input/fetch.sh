#!/bin/bash
# Place/verify the files listed in EXTERNAL.tsv (none is needed for the figure or for eval_v31.py; only
# heart.h5ad is needed, for donor_id, by code/donor_bootstrap_v38.py).
# usage (from input/): [X7_HEART_H5AD=/path/to/heart.h5ad] bash fetch.sh
set -u
cd "$(dirname "$0")"
chk() {  # path sha256
  [ "$2" = unknown ] && { echo "present, sha256 unknown (not checkable): $1"; return 0; }
  got=$(sha256sum "$1" | cut -d' ' -f1)
  [ "$got" = "$2" ] && echo "OK $1" || { echo "SHA MISMATCH $1: $got != $2" >&2; return 1; }
}
rc=0
mkdir -p external/raw
if [ ! -e external/heart.h5ad ] && [ -n "${X7_HEART_H5AD:-}" ]; then cp "$X7_HEART_H5AD" external/heart.h5ad; fi
tail -n +2 EXTERNAL.tsv | while IFS=$'\t' read -r path src sha bytes how; do
  rel=${path#input/}
  if [ -e "$rel" ]; then chk "$rel" "$sha" || rc=1; else echo "MISSING $rel -> $how"; fi
done
