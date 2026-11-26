#!/usr/bin/env bash
# Rebuild compare/metrics.json and compare/macros.tex of the drug-trial case from the files in this folder. No API calls.
#   bash code/reproduce.sh            offline rebuild from the stored model answers in jev/ and d3u/
#   bash code/reproduce.sh --verify   also re-check input checksums against code/MANIFEST.json
# Re-querying the models (costs money, answers may drift) is a separate, explicit step:
#   python3 code/run.py --model jev|d3u --task detection|explanation [--limit N --tag _repK]
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--verify" ]]; then python3 code/manifest.py --check; fi
python3 code/analyze.py
