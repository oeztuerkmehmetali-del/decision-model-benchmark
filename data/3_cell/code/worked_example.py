#!/usr/bin/env python3
"""Worked example: rebuild the exact request for one gene (pancreas, +literature arm) and pull the stored answers.

Run from the case folder: python3 code/worked_example.py [GENE]
Writes input/worked_example.json (state + question sent to both models), jev/worked_example_answer.json and
d3u/worked_example_answer.json (the cached raw responses found under sha256(model, state, questions)).
The state is built exactly as in jev_catalog.py (top-5 co-mention partners, banded by cosine)."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # case folder
sys.path.insert(0, str(HERE.parent / "shared")); sys.path.insert(0, str(HERE / "code"))
import jevlib, questions as Q

t, gene = "pancreas", (sys.argv[1] if len(sys.argv) > 1 else "INS")
options = json.loads((HERE / "input/split_pancreas/pancreas_options.json").read_text())["options"]
cooc = json.loads((HERE / "input/cooc/pancreas_cooc_fixed.json").read_text())
pool = json.loads((HERE / "input/pool/pancreas_pool_fixed.json").read_text())["pool"]
genes = sorted({g for v in pool.values() for g in v})
assert gene in genes, gene
partners = []
for pair, v in cooc["cosine"].items():
    x, y = pair.split("||")
    if gene in (x, y):
        partners.append((y if x == gene else x, v))
partners.sort(key=lambda kv: -kv[1])
state = {"tissue": t, "gene": gene,
         "relations": [{"gene": p, "cooc_band": "high" if v > 0.3 else "med" if v > 0.1 else "low"} for p, v in partners[:5]]}
q = {"celltype": Q.build_questions(options)["celltype"]}
(HERE / "input/worked_example.json").write_text(json.dumps({"state": state, "questions": q}, indent=1))
for folder, model in (("jev", "jev-1.13.0"), ("d3u", "vllm-sr/d3-ultra")):
    cache = HERE / folder / "runs" / f"{t}_catalog_v36_cooc_r1.jsonl"
    hit = jevlib._load(str(cache)).get(jevlib._key(state, q, model))
    assert hit is not None, f"{folder}: no cached answer for {gene} (state mismatch)"
    (HERE / folder / "worked_example_answer.json").write_text(json.dumps({"model": model, "cache": cache.name, "response": hit}, indent=1))
    ans = hit["answers"]["celltype"]
    print(folder, model, "->", ans["choice"], ans.get("confidence"))
