#!/usr/bin/env python3
"""Slice the CellMarker comparators (M0 majority, M1 plain list, M1n size-normalised post hoc, M2 z-score) out of the
Jev eval files into baseline/<tissue>_cellmarker_scorers.json. The comparators do not depend on any model; the same
four entries sit in compare/d3u_vs_baseline/ and are asserted equal here.
Run from the case folder: python3 code/extract_baseline.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # case folder
KEYS = ("M0_majority", "M1_cellmarker_uniform", "M1n_size_normalized_posthoc", "M2_zscore_cellmarker")
for t in ("pancreas", "heart"):
    j = json.loads((HERE / f"compare/jev_vs_baseline/{t}_eval_v36.json").read_text())
    d = json.loads((HERE / f"compare/d3u_vs_baseline/{t}_eval_v36.json").read_text())
    for k in KEYS:  # vs_M1 / vs_M1n are paired against each other, not model-dependent
        assert j["methods"][k] == d["methods"][k], (t, k)
    out = {"tissue": t, "n_test": j["n_test"], "n_genes": j["n_genes"], "types": j["types"],
           "methods": {k: j["methods"][k] for k in KEYS}}
    (HERE / f"baseline/{t}_cellmarker_scorers.json").write_text(json.dumps(out, indent=1))
    print(t, {k: round(out["methods"][k]["acc"], 4) for k in KEYS})
