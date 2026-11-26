#!/usr/bin/env python3
"""v3.1 independent evaluator (STATUS section 23). Does NOT import jev_score.

Same cells/genes for every method:
  M0 majority | M1 CellMarker-uniform (no Jev) | M2 z-score marker counting
  with CellMarker labels (no Jev) | M3 Jev-only catalog | M4 Jev+cooc catalog
Scorer: W = row-normalized log1p(expr); S = W @ P (23 type columns only);
argmax with lexicographic tie-break (first type in sorted order wins).
Metrics: forced micro acc + Wilson CI, macro acc (mean per-class recall over
test classes), selective acc at 75/50% confidence coverage, exact McNemar vs
M1 (Holm over all Jev catalogs evaluated), per-class recall table.
Usage: python3 eval_v31.py --tissue pancreas --catalogs <glob-suffix list>
Writes compare/jev_vs_baseline/<tissue>_eval<tag>.json
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent  # case folder (this file is in code/)


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def argmax_lex(S, types):
    # types already sorted -> np.argmax returns first max = lexicographic tie-break
    assert types == sorted(types)
    return np.array(types, dtype=object)[S.argmax(1)]


def metrics(pred, gt, conf):
    correct = pred == gt
    n = len(gt)
    classes = sorted(set(gt))
    recall = {c: float(correct[gt == c].mean()) for c in classes}
    sel = {}
    order = np.argsort(-conf, kind="stable")
    for cov in (0.75, 0.5):
        m = order[: int(round(cov * n))]
        sel[f"cov{int(cov * 100)}"] = float(correct[m].mean())
    return {"acc": float(correct.mean()), "wilson": wilson(int(correct.sum()), n),
            "macro_acc": float(np.mean(list(recall.values()))),
            "selective": sel, "per_class_recall": recall,
            "n_classes_zero": sum(v == 0 for v in recall.values())}, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tissue", required=True)
    ap.add_argument("--catalogs", nargs="+", required=True,
                    help="catalog suffixes, e.g. _v31_jevonly_r1 _v31_cooc_r1")
    ap.add_argument("--tag", default="", help="output suffix: results/jev/<tissue>_eval<tag>.json")
    ap.add_argument("--pool-suffix", default="",
                    help="Arm D: input/pool/<tissue>_pool<suffix>.json; catalogs are row-subset to it")
    ap.add_argument("--npz-suffix", default="",
                    help="data/<t>_fullcells_lit<suffix>.npz (default '')")
    ap.add_argument("--catdir", default="jev/runs", help="catalog dir (d3-ultra: d3u/runs)")
    ap.add_argument("--outdir", default="compare/jev_vs_baseline", help="eval output dir (d3-ultra: compare/d3u_vs_baseline)")
    a = ap.parse_args()
    t = a.tissue
    z = np.load(HERE / "input/processed" / f"{t}_fullcells_lit{a.npz_suffix}.npz", allow_pickle=True)
    X, genes = z["X"], [str(g) for g in z["genes"]]
    gt_all, is_test = z["cell_type"].astype(str), z["is_test"].astype(bool)
    split = json.loads((HERE / "input" / f"split_{t}" / f"{t}_split.json").read_text())
    assert set(np.where(is_test)[0].tolist()) == set(split["test_idx"]), "npz/split test mismatch"
    types = sorted(json.loads((HERE / "input" / f"split_{t}" / f"{t}_options.json").read_text())["options"])
    pool = json.loads((HERE / "input/pool" / f"{t}_pool{a.pool_suffix}.json").read_text())["pool"]
    cats = {}
    for s in a.catalogs:
        c = json.loads((HERE / a.catdir / f"{t}_catalog{s}.json").read_text())
        assert c.pop("_tissue", t) == t
        cats[s] = c
    # common gene set: in npz AND in pool; every catalog must cover all of them
    # (a partial catalog would silently shrink the gene set for every method)
    g_use = [g for g in genes if any(g in v for v in pool.values())]
    for s, c in cats.items():
        miss = [g for g in g_use if g not in c]
        assert not miss, f"catalog {s} missing {len(miss)} pool genes (partial build?)"
    gi = [genes.index(g) for g in g_use]
    test = np.where(is_test)[0]
    gt = gt_all[test]
    L = np.log1p(X[np.ix_(test, gi)].astype(np.float64))
    tot = L.sum(1, keepdims=True); tot[tot == 0] = 1
    W = L / tot
    out = {"tissue": t, "n_test": int(len(test)), "n_genes": len(g_use), "types": types,
           "methods": {}}

    # M0 majority (majority of TRAIN labels -> no test leakage)
    train_lab = gt_all[~is_test]
    vals, cnts = np.unique(train_lab, return_counts=True)
    maj = vals[np.argmax(cnts)]
    m0, corr0 = metrics(np.array([maj] * len(gt), dtype=object), gt, np.zeros(len(gt)))
    out["methods"]["M0_majority"] = m0

    # M1 CellMarker-uniform: gene -> 1/k over the k pool types listing it
    P1 = np.zeros((len(g_use), len(types)))
    for i, g in enumerate(g_use):
        ts = [ty for ty, v in pool.items() if g in v and ty in types]
        for ty in ts:
            P1[i, types.index(ty)] = 1 / len(ts)
    S1 = W @ P1
    pred1 = argmax_lex(S1, types)
    conf1 = S1.max(1) / (S1.sum(1) + 1e-12)
    m1, corr1 = metrics(pred1, gt, conf1)
    out["methods"]["M1_cellmarker_uniform"] = m1

    # POST-HOC (not pre-registered): M1n = M1 with each type's column divided by its
    # pool size in the scored gene set. Added because on heart the un-normalized M1
    # collapses onto the type with the largest pool (pDC, 319 genes).
    size = (P1 > 0).sum(0).astype(float); size[size == 0] = 1
    S1n = W @ (P1 / size)
    m1n, corr1n = metrics(argmax_lex(S1n, types), gt, S1n.max(1) / (S1n.sum(1) + 1e-12))
    out["methods"]["M1n_size_normalized_posthoc"] = m1n

    # M2 z-score marker counting with CellMarker labels (z over TEST cells only,
    # transductive, no labels; v3 G7 z-scored over all cells -> 0.7811 vs 0.7791)
    Xr = X[np.ix_(test, gi)].astype(np.float64)
    Zs = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-9)
    S2 = Zs @ P1
    pred2 = argmax_lex(S2, types)
    m2, corr2 = metrics(pred2, gt, S2.max(1))
    out["methods"]["M2_zscore_cellmarker"] = m2

    # Baseline-vs-reference exact McNemar (descriptive; NOT in the Holm family of
    # the six Jev repeats, so the p-values are unadjusted).
    def pair(corr, ref, ref_name):
        b = int((ref & ~corr).sum()); c = int((~ref & corr).sum())
        n, k = b + c, min(b, c)  # exact big-int tail: float p underflows to 0.0 below ~1e-308
        lg = math.log10(2 * sum(math.comb(n, i) for i in range(k + 1))) - n * math.log10(2) if n else 0.0
        return {"delta_acc": float(corr.mean() - ref.mean()), f"b_{ref_name}_only": b,
                "c_method_only": c, "mcnemar_p_unadjusted": mcnemar_exact(b, c),
                "mcnemar_log10_p_unadjusted": min(0.0, lg)}
    for k, cr in (("M0_majority", corr0), ("M1n_size_normalized_posthoc", corr1n),
                  ("M2_zscore_cellmarker", corr2)):
        out["methods"][k]["vs_M1"] = pair(cr, corr1, "M1")
    for k, cr in (("M0_majority", corr0), ("M1_cellmarker_uniform", corr1),
                  ("M2_zscore_cellmarker", corr2)):
        out["methods"][k]["vs_M1n_posthoc"] = pair(cr, corr1n, "M1n")

    # Jev catalogs
    raw_p, keys = [], []
    for s, c in cats.items():
        P = np.zeros((len(g_use), len(types)))
        for i, g in enumerate(g_use):
            for ty, p in c[g]["probs"].items():
                if ty in types:
                    P[i, types.index(ty)] = p
        S = W @ P
        pred = argmax_lex(S, types)
        conf = S.max(1) / (S.sum(1) + 1e-12)
        m, corr = metrics(pred, gt, conf)
        b = int((corr1 & ~corr).sum()); cc = int((~corr1 & corr).sum())
        m["vs_M1"] = {"delta_acc": m["acc"] - m1["acc"], "b_M1_only": b, "c_jev_only": cc,
                      "mcnemar_p": mcnemar_exact(b, cc)}
        bn = int((corr1n & ~corr).sum()); cn = int((~corr1n & corr).sum())
        m["vs_M1n_posthoc"] = {"delta_acc": m["acc"] - m1n["acc"], "b_M1n_only": bn,
                               "c_jev_only": cn, "mcnemar_p": mcnemar_exact(bn, cn)}
        # POST-HOC sensitivity (not pre-registered): renormalize each gene row
        # over the 23 types after dropping the none-of-these mass
        Pn = P / np.where(P.sum(1, keepdims=True) > 0, P.sum(1, keepdims=True), 1)
        Sn = W @ Pn
        mr, corr_r = metrics(argmax_lex(Sn, types), gt, Sn.max(1) / (Sn.sum(1) + 1e-12))
        m["sensitivity_renorm_posthoc"] = {"acc": mr["acc"], "macro_acc": mr["macro_acc"],
                                           "delta_acc_vs_M1": mr["acc"] - m1["acc"]}
        out["methods"][f"jev{s}"] = m
        raw_p.append(m["vs_M1"]["mcnemar_p"]); keys.append(f"jev{s}")
    for k, p in zip(keys, holm(raw_p)):
        out["methods"][k]["vs_M1"]["mcnemar_p_holm"] = p
    for k, p in zip(keys, holm([out["methods"][k]["vs_M1n_posthoc"]["mcnemar_p"] for k in keys])):
        out["methods"][k]["vs_M1n_posthoc"]["mcnemar_p_holm"] = p
    od = HERE / a.outdir; od.mkdir(parents=True, exist_ok=True)
    f = od / f"{t}_eval{a.tag}.json"
    f.write_text(json.dumps(out, indent=1))
    print(f"{'method':28}{'acc':>8}{'macro':>8}{'cov75':>8}{'zero_cls':>9}{'dM1':>8}{'p_holm':>10}")
    for k, m in out["methods"].items():
        v = m.get("vs_M1", {})
        print(f"{k:28}{m['acc']:8.4f}{m['macro_acc']:8.4f}{m['selective']['cov75']:8.4f}"
              f"{m['n_classes_zero']:9d}{v.get('delta_acc', float('nan')):8.4f}"
              f"{v.get('mcnemar_p_holm', float('nan')):10.2e}")
    print("wrote", f, "| genes", len(g_use), "| test", len(test))


if __name__ == "__main__":
    main()
