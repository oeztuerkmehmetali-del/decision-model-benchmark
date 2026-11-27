#!/bin/sh
# Downloads the two files listed in EXTERNAL.tsv (> 50 MB, attached to the data-v1 release) and checks sha256.  Run from paper/data/4_massspec/input/.
# Needed ONLY to recompute evidence.tsv / the Sage search; the figure and every offline step use the files already in this folder.
set -e
cd "$(dirname "$0")"
R=https://github.com/oeztuerkmehmetali-del/decision-model-benchmark/releases/download/data-v1
for f in b1931.mzXML b1931.ms2.mzML; do [ -f "$f" ] || curl -fL -o "$f" "$R/$f"; done
tail -n +2 EXTERNAL.tsv | while IFS="$(printf '\t')" read -r p u s b h; do
  [ "$(sha256sum "$(basename "$p")" | cut -d' ' -f1)" = "$s" ] && echo "ok $p" || { echo "SHA256 MISMATCH $p" >&2; exit 1; }
done
