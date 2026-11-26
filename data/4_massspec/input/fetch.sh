#!/bin/sh
# Places the two files listed in EXTERNAL.tsv (not shipped: > 50 MB) and checks sha256.  Run from paper/data/4_massspec/input/.
# Needed ONLY to recompute evidence.tsv / the Sage search; the figure and every offline step use the files already in this folder.
set -e
cd "$(dirname "$0")"
if [ ! -f b1931.mzXML ]; then
  echo "Download PXD001468 run b1931_293T_proteinID_11A from https://www.ebi.ac.uk/pride/archive/projects/PXD001468 as input/b1931.mzXML" >&2
  echo "(the exact PRIDE file name / conversion to mzXML was not recorded, see ../README.md gaps)" >&2; exit 1
fi
[ -f b1931.ms2.mzML ] || python3 ../code/mzxml2mzml.py
tail -n +2 EXTERNAL.tsv | while IFS="$(printf '\t')" read -r p u s b h; do
  [ "$(sha256sum "$(basename "$p")" | cut -d' ' -f1)" = "$s" ] && echo "ok $p" || { echo "SHA256 MISMATCH $p" >&2; exit 1; }
done
