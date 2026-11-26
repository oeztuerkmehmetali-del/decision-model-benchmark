#!/usr/bin/env python3
"""v3 catalog: gene x celltype P matrix via Jev, tissue-param, resume-safe.

Per gene (pool from CellMarker lit pool): state = gene + literature cooc
partners (from jev_cooc.py output) -> ONE choice request over the tissue's
types + none-of-these. Incremental JSONL cache + incremental catalog.json
write (resume after kill). Scorer asserts _tissue match (FM3/FM4).
Usage: python3 jev_catalog.py --tissue pancreas
Writes: jev/runs/<tissue>_catalog<suffix>.json + .jsonl
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # case folder (this file is in code/)
sys.path.insert(0, str(HERE.parent / "shared"))  # jevlib.py (paper/data/shared)

GENE_INSTRUCTIONS = (
    "You judge which cell type a gene is most specifically associated with, "
    "given the gene name and its literature co-occurrence partners (genes it "
    "is frequently mentioned together with in PubMed). Use the partner "
    "evidence plus your knowledge of the gene's function. Answer with the "
    "single best-matching candidate cell type and a confidence. If the "
    "evidence does not support any candidate, answer 'none of these'."
)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tissue", default="pancreas")
    ap.add_argument("--pool", default=None, help="default input/pool/<tissue>_pool.json")
    ap.add_argument("--cooc", default=None, help="default input/cooc/<tissue>_cooc.json")
    ap.add_argument("--cooc-mode", choices=["partners", "none"], default="partners",
                    help="none = Jev-only arm (bare gene state, circularity control)")
    ap.add_argument("--suffix", default="", help="output filename suffix, e.g. _jevonly")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pool_f = Path(a.pool) if a.pool else HERE/"input/pool"/f"{a.tissue}_pool.json"
    cooc_f = Path(a.cooc) if a.cooc else HERE/"input/cooc"/f"{a.tissue}_cooc.json"
    if a.cooc_mode == "partners":
        if not cooc_f.exists():
            sys.exit(f"cooc missing: {cooc_f} - run jev_cooc.py first (fail closed)")
        cooc = json.loads(cooc_f.read_text())
        if not cooc.get("complete", False):
            sys.exit(f"cooc incomplete (dead_letter={len(cooc.get('dead_letter', []))}) - fail closed")
    else:
        cooc = {"cosine": {}}
    poolj = json.loads(pool_f.read_text())
    genes = sorted({g for glist in poolj["pool"].values() for g in glist})
    options = json.loads((HERE/"input"/f"split_{a.tissue}"/f"{a.tissue}_options.json").read_text())["options"]
    import questions as Q
    q = {"celltype": Q.build_questions(options)["celltype"]}
    # partners per gene from cosine (top 5)
    partners: dict[str, list[tuple[str, float]]] = {g: [] for g in genes}
    for pair, v in cooc.get("cosine", {}).items():
        x, y = pair.split("||")
        if x in partners and y in partners:
            partners[x].append((y, v)); partners[y].append((x, v))
    for g in partners:
        partners[g].sort(key=lambda kv: -kv[1])
    import jevlib
    outdir = Path(os.environ["X7_OUTDIR"]) if os.environ.get("X7_OUTDIR") else HERE/"jev/runs"  # d3-ultra runs set X7_OUTDIR=d3u/runs
    outdir.mkdir(parents=True, exist_ok=True)
    cache = str(outdir/f"{a.tissue}_catalog{a.suffix}.jsonl")
    cat_f = outdir/f"{a.tissue}_catalog{a.suffix}.json"
    # resume: load existing catalog
    cat = json.loads(cat_f.read_text()) if cat_f.exists() else {}
    cat.pop("_tissue", None)  # resume must not carry the metadata key into the gene dict
    states = []
    for g in genes:
        if g in cat:
            continue
        rels = [{"gene": p, "cooc_band": ("high" if v > 0.3 else "med" if v > 0.1 else "low")}
                for p, v in partners[g][:5]]
        states.append({"tissue": a.tissue, "gene": g, "relations": rels})
    print(f"tissue={a.tissue} genes={len(genes)} todo={len(states)} cached_cat={len(cat)}")
    if a.dry_run:
        print(json.dumps(states[0] if states else cat, default=str)[:300]); return
    if states:
        resps = jevlib.ask_many([(st, q) for st in states], cache=cache,
                                workers=int(os.environ.get("X7_WORKERS", "2")),
                                model=os.environ.get("X7_MODEL", jevlib.MODEL))
        for st, r in zip(states, resps):
            answ = r.get("answers", r).get("celltype", {})
            cat[st["gene"]] = {"pred": answ.get("choice"),
                               "conf": float(answ.get("confidence", 0) or 0),
                               "probs": answ.get("probabilities", {})}
            cat_f.write_text(json.dumps({"_tissue": a.tissue, **cat}, indent=1))  # incremental
    cat_f.write_text(json.dumps({"_tissue": a.tissue, **cat}, indent=1))
    from collections import Counter
    print(f"catalog: {len(cat)} genes")
    print("top-1 votes:", Counter(v["pred"] for v in cat.values()).most_common(8))
    print("wrote", cat_f)

if __name__ == "__main__":
    main()
