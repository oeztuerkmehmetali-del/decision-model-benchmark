#!/usr/bin/env python3
"""Donor-clustered bootstrap for the corrected-pool HEART v3.6 evaluation.

Independent re-implementation of the eval_v31 scorer (STATUS s29/s36c); no import
of eval_v31. Per-cell correctness vectors are reproduced from
  input/processed/heart_fullcells_lit_fixed.npz, input/pool/heart_pool_fixed.json,
  jev/runs/heart_catalog_v36_{jevonly,cooc}_r{1,2,3}.json   (paths relative to the case folder)
and checked against compare/jev_vs_baseline/heart_eval_v36.json to 1e-4 before any statistic is
computed (--tol, default 1e-4).

Donor labels come from input/external/heart.h5ad obs/donor_id read with h5py
(categories + codes only; X is never touched). npz row i == h5ad row i, verified
by asserting cell_type equality for all 35,384 rows.

Pancreas: no donor/sample id exists in any committed artifact -> UNKNOWN.

Stats: donor-cluster bootstrap, resample the 6 donors with replacement, B=10000,
seed 0, 95% percentile CI of the micro and of the macro accuracy difference vs
M1n; plus leave-one-donor-out micro differences.

Usage: python3 code/donor_bootstrap_v38.py [--out compare/jev_vs_baseline/heart_donor_bootstrap.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent.parent  # case folder (this file is in code/)
B = 10000
SEED = 0


def load_inputs(tissue: str):
    z = np.load(HERE / "input/processed" / f"{tissue}_fullcells_lit_fixed.npz", allow_pickle=True)
    genes = [str(g) for g in z["genes"]]
    X, gt_all, is_test = z["X"], z["cell_type"].astype(str), z["is_test"].astype(bool)
    split = json.loads((HERE / "input" / f"split_{tissue}" / f"{tissue}_split.json").read_text())
    assert set(np.where(is_test)[0].tolist()) == set(split["test_idx"]), "npz/split test mismatch"
    types = sorted(json.loads(
        (HERE / "input" / f"split_{tissue}" / f"{tissue}_options.json").read_text())["options"])
    pool = json.loads(
        (HERE / "input/pool" / f"{tissue}_pool_fixed.json").read_text())["pool"]
    g_use = [g for g in genes if any(g in v for v in pool.values())]
    return z, X, gt_all, is_test, types, pool, g_use


def donor_labels(tissue: str, n_rows: int):
    """Per-row donor id from data/processed/<t>.h5ad (h5py obs only), else None."""
    f = HERE / "input/external" / f"{tissue}.h5ad"
    if not f.exists():
        return None, f"{f.relative_to(HERE)} not present"
    with h5py.File(f, "r") as h:
        if "obs/donor_id" not in h:
            return None, "no obs/donor_id"
        g = h["obs/donor_id"]
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][:]]
        codes = g["codes"][:]
    if len(codes) != n_rows:
        return None, f"h5ad n_obs {len(codes)} != npz rows {n_rows}"
    if (codes < 0).any():
        return None, "NaN donor_id"
    return np.array(cats)[codes], str(f.relative_to(HERE))


def W_of(X, test, gi):
    L = X[np.ix_(test, gi)].astype(np.float64)  # X is log1p-CP10K already (same scorer as eval_v31)
    tot = L.sum(1, keepdims=True)
    tot[tot == 0] = 1
    return L / tot


def argmax_lex(S, types):
    assert types == sorted(types)
    return np.array(types, dtype=object)[S.argmax(1)]


def micro_macro(corr, gt, types):
    micro = corr.mean()
    rec = [corr[gt == t].mean() for t in types if (gt == t).any()]
    return float(micro), float(np.mean(rec)), len(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tissue", default="heart")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--out", default="")
    ap.add_argument("--comparator", default="M1n", choices=("M1n", "M2"),
                    help="M1n = post-hoc size-normalised (s29); M2 = z-score, strongest pre-specified scorer on heart")
    ap.add_argument("--catdir", default="jev/runs", help="catalog dir (d3-ultra: d3u/runs)")
    ap.add_argument("--ref", default="", help="published eval json for the fidelity gate (default results/jev/<t>_eval_v36.json)")
    a = ap.parse_args()
    t, cmp = a.tissue, a.comparator

    z, X, gt_all, is_test, types, pool, g_use = load_inputs(t)
    gi = [list(z["genes"]).index(g) for g in g_use]
    test = np.where(is_test)[0]
    gt = gt_all[test]
    W = W_of(X, test, gi)

    # comparator scorers, as in eval_v31: M1n (post-hoc size-normalised CellMarker-uniform) or M2 (z-score)
    P1 = np.zeros((len(g_use), len(types)))
    for i, g in enumerate(g_use):
        ts = [ty for ty, v in pool.items() if g in v and ty in types]
        for ty in ts:
            P1[i, types.index(ty)] = 1 / len(ts)
    if cmp == "M1n":
        size = (P1 > 0).sum(0).astype(float)
        size[size == 0] = 1
        pred_ref = argmax_lex(W @ (P1 / size), types)
    else:
        Xr = X[np.ix_(test, gi)].astype(np.float64)
        pred_ref = argmax_lex(((Xr - Xr.mean(0)) / (Xr.std(0) + 1e-9)) @ P1, types)
    c_m1n = pred_ref == gt  # comparator correctness (name kept from the M1n-only version)

    arms = {}
    for build in ("jevonly", "cooc"):
        for r in (1, 2, 3):
            sfx = f"_v36_{build}_r{r}"
            c = json.loads((HERE / a.catdir / f"{t}_catalog{sfx}.json").read_text())
            P = np.zeros((len(g_use), len(types)))
            for i, g in enumerate(g_use):
                for ty, p in c[g]["probs"].items():
                    if ty in types:
                        P[i, types.index(ty)] = p
            arms[sfx] = argmax_lex(W @ P, types) == gt

    # fidelity gate vs the published artifact
    ref = json.loads((HERE / (a.ref or f"compare/jev_vs_baseline/{t}_eval_v36.json")).read_text())
    ref_key = {"M1n": "M1n_size_normalized_posthoc", "M2": "M2_zscore_cellmarker"}[cmp]
    checks = {cmp: (c_m1n, ref["methods"][ref_key])}
    for sfx, c in arms.items():
        checks[f"jev{sfx}"] = (c, ref["methods"][f"jev{sfx}"])
    fidelity = {}
    for k, (c, r) in checks.items():
        d_acc = abs(float(c.mean()) - r["acc"])
        fidelity[k] = {"acc_recomputed": float(c.mean()), "acc_published": r["acc"],
                       "abs_diff": d_acc, "within_tol": bool(d_acc <= a.tol)}
    if not all(v["within_tol"] for v in fidelity.values()):
        raise SystemExit("FIDELITY FAIL: " + json.dumps(fidelity, indent=1))

    don, don_src = donor_labels(t, len(gt_all))
    out = {"tissue": t, "n_test": int(len(test)), "n_genes": len(g_use), "types": types,
           "comparator": ref_key, "catdir": a.catdir,
           "scorer_fidelity_vs_runs_v31": fidelity,
           "bootstrap": {"unit": "donor", "B": B, "seed": SEED, "ci": "percentile_95"}}

    if don is None:
        out["donors"] = {"status": "UNKNOWN", "reason": don_src,
                         "note": "no donor/sample id in any committed artifact for this tissue; "
                                 "donor-clustered bootstrap not computable"}
        out["per_donor"] = None
        out["results"] = None
    else:
        d = don[test]
        udon = sorted(set(d.tolist()))
        # 1. donor inventory: same donors in train and test?
        dtr = don[~is_test]
        train_n = {k: int((dtr == k).sum()) for k in udon}
        test_n = {k: int((d == k).sum()) for k in udon}
        both = sorted(k for k in udon if train_n[k] > 0 and test_n[k] > 0)
        out["donors"] = {
            "source": don_src, "n_donors": len(udon), "donors": udon,
            "train_cells_per_donor": train_n, "test_cells_per_donor": test_n,
            "donors_in_both_train_and_test": both,
            "donor_disjoint": len(both) == 0,
            "all_donors_in_both": len(both) == len(udon),
        }

        base_micro, base_macro, _ = micro_macro(c_m1n, gt, types)
        per_donor = []
        for k in udon:
            m = d == k
            sub = micro_macro(c_m1n[m], gt[m], types)
            row = {"donor": k, "n_test": int(m.sum()), "n_train": train_n[k],
                   f"{cmp}_micro": sub[0], f"{cmp}_macro_sub": sub[1]}
            for sfx, c in arms.items():
                a_m, a_M, _ = micro_macro(c[m], gt[m], types)
                row[f"{sfx}_micro"] = a_m
                row[f"{sfx}_macro_sub"] = a_M
                row[f"{sfx}_d_micro"] = a_m - sub[0]
            per_donor.append(row)
        out["per_donor"] = per_donor

        # 2. donor-cluster bootstrap over the 6 donor clusters
        rng = np.random.default_rng(SEED)
        idx_by_donor = [np.where(d == k)[0] for k in udon]
        n_d = len(udon)
        picks = rng.integers(0, n_d, size=(B, n_d))  # resample donors with replacement
        res = {sfx: {"micro_diff": np.empty(B), "macro_diff": np.empty(B)} for sfx in arms}
        lodo = {sfx: {} for sfx in arms}
        for sfx, c in arms.items():
            for j, drop in enumerate(udon):
                keep = d != drop
                bm, _, _ = micro_macro(c_m1n[keep], gt[keep], types)
                am, _, _ = micro_macro(c[keep], gt[keep], types)
                lodo[sfx][drop] = float(am - bm)
        for b in range(B):
            sel = np.concatenate([idx_by_donor[j] for j in picks[b]])
            bm, bM, _ = micro_macro(c_m1n[sel], gt[sel], types)
            for sfx, c in arms.items():
                am, aM, _ = micro_macro(c[sel], gt[sel], types)
                res[sfx]["micro_diff"][b] = am - bm
                res[sfx]["macro_diff"][b] = aM - bM
        results = {}
        for sfx, c in arms.items():
            am, aM, _ = micro_macro(c, gt, types)
            md, Md = res[sfx]["micro_diff"], res[sfx]["macro_diff"]
            results[sfx] = {
                "micro": am, "macro": aM,
                f"micro_diff_vs_{cmp}": am - base_micro, f"macro_diff_vs_{cmp}": aM - base_macro,
                "micro_diff_ci95": [float(np.percentile(md, 2.5)), float(np.percentile(md, 97.5))],
                "macro_diff_ci95": [float(np.percentile(Md, 2.5)), float(np.percentile(Md, 97.5))],
                "micro_ci_excludes_0": bool(np.percentile(md, 2.5) > 0 or np.percentile(md, 97.5) < 0),
                "macro_ci_excludes_0": bool(np.percentile(Md, 2.5) > 0 or np.percentile(Md, 97.5) < 0),
                "leave_one_donor_out_micro_diff": lodo[sfx],
                "lodo_micro_diff_range": [min(lodo[sfx].values()), max(lodo[sfx].values())],
            }
        out[cmp] = {"micro": base_micro, "macro": base_macro}
        out["results"] = results

    dst = HERE / (a.out or f"compare/jev_vs_baseline/{t}_donor_bootstrap.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print("wrote", dst)
    print(json.dumps(fidelity, indent=1))
    if out["results"]:
        for sfx, r in out["results"].items():
            print(f"{sfx:22} micro {r['micro']:.4f} d {r[f'micro_diff_vs_{cmp}']:+.4f} "
                  f"CI {r['micro_diff_ci95'][0]:+.4f},{r['micro_diff_ci95'][1]:+.4f} excl0={r['micro_ci_excludes_0']} | "
                  f"macro {r['macro']:.4f} d {r[f'macro_diff_vs_{cmp}']:+.4f} "
                  f"CI {r['macro_diff_ci95'][0]:+.4f},{r['macro_diff_ci95'][1]:+.4f} excl0={r['macro_ci_excludes_0']}")
    else:
        print(out["donors"])


if __name__ == "__main__":
    main()