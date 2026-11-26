#!/usr/bin/env python3
"""v3 cooc: PubMed pair-count co-occurrence within type marker lists.

Pre-registered (STATUS §18): pairs only WITHIN each CellMarker type list
(Track-X framing: fixed pool per type). Partner context for the Jev catalog
state; gene-gene co-mention is literature signal, not GT leakage.
Incremental JSONL cache (resume-safe), 429 backoff, dead_letter, fail-closed
complete flag. Usage: python3 jev_cooc.py --tissue pancreas
Writes: input/cooc/<tissue>_cooc.json (+ <tissue>_cooc_cache.jsonl)
"""
from __future__ import annotations
import argparse, json, math, time, urllib.parse, urllib.request
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # case folder (this file is in code/)
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

def esearch_count(term: str, attempts: int = 4) -> int:
    """PubMed esearch count with 429 backoff; -1 only after retries exhausted."""
    p = urllib.parse.urlencode({"db": "pubmed", "term": term, "retmode": "json", "retmax": "0"})
    url = EUTILS + "?" + p
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return int(json.load(r)["esearchresult"]["count"])
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and i < attempts - 1:
                time.sleep(2 ** (i + 1))
                continue
            return -1
        except Exception:
            if i < attempts - 1:
                time.sleep(2 ** (i + 1))
                continue
            return -1
    return -1

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tissue", default="pancreas")
    ap.add_argument("--pool", default=None, help="default input/pool/<tissue>_pool.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cadence", type=float, default=0.35)
    a = ap.parse_args()
    out = Path(a.out) if a.out else HERE/"input/cooc"/f"{a.tissue}_cooc.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_f = out.with_suffix(".cache.jsonl")
    pool_f = Path(a.pool) if a.pool else HERE/"input/pool"/f"{a.tissue}_pool.json"
    poolj = json.loads(pool_f.read_text())
    # provenance guard: pool must belong to the requested tissue
    assert f"tissue={a.tissue.capitalize()}" in poolj["source"], (
        f"pool {pool_f} source {poolj['source']!r} does not match --tissue {a.tissue}")
    pool = poolj["pool"]
    # pairs within type lists, deduped across types
    pair_types: dict[tuple[str, str], list[str]] = {}
    for t, genes in pool.items():
        for x, y in combinations(sorted(genes), 2):
            pair_types.setdefault((x, y), []).append(t)
    singles = sorted({g for genes in pool.values() for g in genes})
    # incremental cache
    done: dict[str, int] = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    if r["n"] >= 0:  # dead-letter (-1) rows are retried on rerun
                        done[r["q"]] = r["n"]
                except json.JSONDecodeError:
                    pass
    queries = {f'"{g}"[tw]': None for g in singles}
    queries.update({f'("{x}"[tw]) AND ("{y}"[tw])': (x, y) for (x, y) in pair_types})
    todo = [q for q in queries if q not in done]
    print(f"singles: {len(singles)} pairs: {len(pair_types)} cached: {len(done)} todo: {len(todo)}")
    dead_letter = []
    t0 = time.time()
    with cache_f.open("a") as cf:
        for i, q in enumerate(todo):
            n = esearch_count(q)
            if n < 0:
                raise RuntimeError(f"PubMed esearch failed for {q!r} after retries; "
                                   f"aborting (fail-closed)")
            done[q] = n
            cf.write(json.dumps({"q": q, "n": n}) + "\n")
            cf.flush()
            time.sleep(a.cadence)
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{len(todo)} wall={time.time()-t0:.0f}s dead={len(dead_letter)}", flush=True)
    single_counts = {}
    for g in singles:
        v = done.get(f'"{g}"[tw]', -1)
        if v > 0: single_counts[g] = v
    cooc, cos = {}, {}
    for (x, y), _ in pair_types.items():
        q = f'("{x}"[tw]) AND ("{y}"[tw])'
        n = done.get(q, -1)
        if n > 0 and single_counts.get(x, 0) > 0 and single_counts.get(y, 0) > 0:
            cooc[f"{x}||{y}"] = n
            cos[f"{x}||{y}"] = round(n / math.sqrt(single_counts[x] * single_counts[y]), 4)
    complete = not dead_letter
    res = {"backend": "pubmed", "tissue": a.tissue, "pool_source": poolj["source"],
           "singles_n": len(singles), "pairs_n": len(pair_types),
           "nonzero_pairs": len(cooc),
           "sparsity": 1 - len(cooc) / max(len(pair_types), 1),
           "single_counts": single_counts, "cooc_counts": cooc, "cosine": cos,
           "dead_letter": dead_letter, "complete": complete,
           "wall_s": round(time.time() - t0, 1)}
    out.write_text(json.dumps(res, indent=1))
    print(f"nonzero={len(cooc)} sparsity={res['sparsity']:.3f} complete={complete} "
          f"dead={len(dead_letter)} wall={res['wall_s']}s")
    print("wrote", out)
    return 0 if complete else 2

if __name__ == "__main__":
    import sys
    sys.exit(main())
