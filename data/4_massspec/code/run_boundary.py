"""x2 Route C stage 1: Jev noul scores for the FDR-boundary PSM band of one dataset.

python3 code/run_boundary.py --pin input/sage_nopep.pin \
    --mok baseline/tB.mokapot.psms.txt --mok-decoy baseline/tB.mokapot.decoy.psms.txt \
    --out jev/tB_boundary.tsv [--qlo 0.005 --qhi 0.05 --batch 16 --limit N]   (paths relative to the case folder)
Band = PSMs (targets AND decoys, label-blind) whose mokapot score lies between the score cutoffs
at target q=qhi and q=qlo. Jev sees search-engine features only (psm_serializer.py).
"""
import argparse, csv, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "shared"))  # shared model client: paper/data/shared/jevlib.py
import jevlib  # noqa: E402
import psm_serializer as ps  # noqa: E402


def read_pin(path):
    """PIN rows may carry extra tab-separated protein columns -> split on header width only."""
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        rows = [dict(zip(hdr, line.rstrip("\n").split("\t", len(hdr) - 1))) for line in f]
    return pd.DataFrame(rows)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--pin", required=True); a.add_argument("--mok", required=True)
    a.add_argument("--mok-decoy", required=True); a.add_argument("--out", required=True)
    a.add_argument("--qlo", type=float, default=0.005); a.add_argument("--qhi", type=float, default=0.05)
    a.add_argument("--batch", type=int, default=16); a.add_argument("--limit", type=int, default=0)
    a.add_argument("--variant", default="base"); a.add_argument("--evidence", default=None); a.add_argument("--cache", default=None)
    a.add_argument("--all", action="store_true", help="brute force: every PSM (targets+decoys), global buckets"); a.add_argument("--qtype", default="noul", choices=("noul", "choice"))
    a.add_argument("--seed", type=int, default=0, help="replicate k>0: shuffle PSM order so batch contexts differ (ensembling)")
    a.add_argument("--fewshot", type=int, default=0, help="K labeled correct + K incorrect dev(even-scan) examples in state; evaluates odd-scan PSMs only")
    a.add_argument("--pairs", type=int, default=0, help="pairwise mode: each odd-scan PSM vs K random odd-scan opponents; score = win rate")
    a = a.parse_args()
    a.cache = a.cache or os.path.join(HERE, "..", "jev", f"boundary_jev_cache_{a.variant}.jsonl")

    pin = read_pin(a.pin); pin["SpecId"] = pin["SpecId"].astype(str)
    if a.evidence:
        pin = pin.merge(pd.read_csv(a.evidence, sep="\t").rename(columns={"psm_id": "SpecId"}).astype({"SpecId": str}), on="SpecId", how="left")
    pin = pin.rename(columns={"Xcorr": "XCorr"}).drop_duplicates("SpecId").set_index("SpecId")
    t = pd.read_csv(a.mok, sep="\t"); t["is_decoy"] = 0
    d = pd.read_csv(a.mok_decoy, sep="\t"); d["is_decoy"] = 1
    m = pd.concat([t, d], ignore_index=True)
    m["SpecId"] = m["SpecId"].astype(str)
    m = m.rename(columns={"mokapot score": "score", "mokapot q-value": "q"})[["SpecId", "ScanNr", "score", "q", "is_decoy"]]
    s_hi = t["mokapot score"][t["mokapot q-value"] <= a.qlo].min()
    s_lo = t["mokapot score"][t["mokapot q-value"] <= a.qhi].min()
    band = m[(m.score >= s_lo) & (m.score < s_hi)].reset_index(drop=True)
    if a.all:
        band = m.reset_index(drop=True)
    if a.limit:
        band = band.sample(n=min(a.limit, len(band)), random_state=0).reset_index(drop=True)
    if a.fewshot or a.pairs:
        band = band[band.ScanNr.astype(int) % 2 == 1].reset_index(drop=True)
    if a.seed:
        band = band.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    print(f"band score [{s_lo:.3f},{s_hi:.3f}): {len(band)} PSMs "
          f"({int((band.is_decoy == 0).sum())} T / {int(band.is_decoy.sum())} D)", file=sys.stderr)

    allpsm = pin.loc[m.SpecId]
    ref = {k: sorted(allpsm[k].astype(float)) for k in ("XCorr", "TailorScore") if k in allpsm}
    feats = pin.loc[band.SpecId]
    ref["band"] = {k: sorted(feats[k].astype(float)) for k in [k for k, _ in ps.NUM_FEATURES] + [k for k, *_ in ps.SAGE_FEATURES] + [k for k, *_ in ps.EVIDENCE_FEATURES] if k in feats}

    def truth(sid):  # label for few-shot examples ONLY (dev half): decoy or ARATH-only target = incorrect
        pr = pin.loc[sid, "Proteins"] if "Proteins" in pin else ""
        return "no" if (m_dec[sid] or ("_ARATH" in pr and "_HUMAN" not in pr)) else "yes"
    state = ps.STATE
    if a.fewshot:
        m_dec = dict(zip(m.SpecId, m.is_decoy))
        dev = m[(m.ScanNr.astype(int) % 2 == 0) & (m.score >= s_lo) & (m.score < s_hi)].sample(frac=1, random_state=0)
        pick = []
        for lab in ("yes", "no"):
            pick += [s for s in dev.SpecId if truth(s) == lab][:a.fewshot]
        ex = [{"psm": ps.describe(pin.loc[s], ref, a.variant), "correct": truth(s)} for s in pick]
        state = {**ps.STATE, "labeled_examples": [ex[i] for i in pd.Series(range(len(ex))).sample(frac=1, random_state=0)]}
    if a.pairs:
        import numpy as np
        rng = np.random.default_rng(a.seed); n = len(band); pq, pid = [], []
        for i in range(n):
            for j in rng.choice([x for x in range(n) if x != i], a.pairs, replace=False) if n < 5000 else rng.integers(0, n, a.pairs):
                if j == i:
                    continue
                pq.append((i, int(j)) if rng.random() < 0.5 else (int(j), i))  # randomised A/B position
        items, ids = [], []
        for b in range(0, len(pq), 8):
            qs = {f"pair_{i}_{j}": {"type": "choice", "criteria": {"A": None, "B": None},
                  "instructions": "Which PSM is more likely a correct identification? PSM A: " + ps.describe(feats.iloc[i], ref, a.variant)
                  + " | PSM B: " + ps.describe(feats.iloc[j], ref, a.variant) + ps.SAGE_HINT} for i, j in pq[b:b + 8]}
            items.append((ps.STATE, qs)); ids.append(pq[b:b + 8])
        resps = jevlib.ask_many(items, cache=a.cache, workers=4)
        w, c = np.zeros(n), np.zeros(n)
        for pairs_, rr in zip(ids, resps):
            for i, j in pairs_:
                pa = rr["answers"][f"pair_{i}_{j}"]["probabilities"]["A"]
                w[i] += pa; c[i] += 1; w[j] += 1 - pa; c[j] += 1
        band["jev_p"] = w / np.maximum(c, 1)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); band.to_csv(a.out, sep="\t", index=False)
        print(f"pairs={len(pq)} requests={jevlib.USAGE['requests']} tokens={jevlib.USAGE['input_tokens']} cost=${jevlib.cost_usd():.4f} -> {a.out}")
        return
    items, ids = [], []
    for b in range(0, len(band), a.batch):
        chunk = [(i, feats.iloc[i]) for i in range(b, min(b + a.batch, len(band)))]
        items.append((state, ps.build_questions(chunk, ref, variant=a.variant, qtype=a.qtype)))
        ids.append([i for i, _ in chunk])
    resps = jevlib.ask_many(items, cache=a.cache, workers=4)

    p, yes = {}, {}
    for idx, r in zip(ids, resps):
        for i in idx:
            ans = r["answers"][f"psm_{i}"]
            p[i] = ans["noul"] if a.qtype == "noul" else ans["probabilities"]["yes"]
            if a.qtype == "choice":
                yes[i] = int(ans["choice"] == "yes")
    band["jev_p"] = [p[i] for i in range(len(band))]
    if yes:
        band["jev_yes"] = [yes[i] for i in range(len(band))]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    band.to_csv(a.out, sep="\t", index=False)
    print(f"requests={jevlib.USAGE['requests']} cache_hits={jevlib.USAGE['cache_hits']} "
          f"tokens={jevlib.USAGE['input_tokens']} cost=${jevlib.cost_usd():.4f} -> {a.out}")


if __name__ == "__main__":
    main()
