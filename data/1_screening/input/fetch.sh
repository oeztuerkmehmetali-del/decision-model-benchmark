#!/usr/bin/env bash
# No file of this case exceeds 50 MB, so nothing is downloaded: input/EXTERNAL.tsv has no data rows.
# The SYNERGY title/abstract CSVs in input/synergy/ were exported from `synergy-dataset==2.2`
# (https://github.com/asreview/synergy-dataset) by code/export_csv.py; to re-export:
#   pip install synergy-dataset==2.2 && python3 ../code/export_csv.py
# Verify the shipped split/question inputs:
cd "$(dirname "$0")"
echo "expected 9422be8d07745a0dc0a555deed96bb16b0b1d9effb5e3381106bc7fa2ef54007"; sha256sum split.union.json
