#!/usr/bin/env python3
"""X7 stage D: frozen stratified split + CL branch narrowing.

Reads data/processed/<tissue>.h5ad (with leiden + wilcoxon), writes
data/split/<tissue>_split.json (frozen cell ids, seed 0) and
data/cl_branch/<tissue>_options.json (Jev option list, <=255).

Usage (helix): ../venv/bin/python split_cl.py --tissue pancreas
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "input" / "external"
SPLIT = ROOT / "input"
CL = ROOT / "input"
SEED = 0
TEST_FRAC = 0.2
MAX_OPTIONS = 255


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tissue", required=True)
    ap.add_argument("--test-frac", type=float, default=TEST_FRAC)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    import numpy as np
    import scanpy as sc

    rng = np.random.default_rng(args.seed)
    ad = sc.read_h5ad(PROC / f"{args.tissue}.h5ad")
    labels = ad.obs["cell_type"].astype(str).values
    n = ad.n_obs
    # stratified split: per-class shuffle, hold out test_frac
    test_mask = np.zeros(n, dtype=bool)
    for cls in sorted(set(labels)):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * args.test_frac)))
        test_mask[idx[:k]] = True
    out = {
        "tissue": args.tissue, "seed": args.seed,
        "n_total": int(n), "n_test": int(test_mask.sum()),
        "n_train": int(n - test_mask.sum()),
        "test_idx": test_mask.nonzero()[0].tolist(),
        "classes": sorted(set(labels)),
    }
    # 5-fold stratified CV over TRAIN cells (test stays frozen held-out).
    # Fold 0 = dev (threshold freezing); folds 1-4 = CV eval; pooled = all.
    train_idx = (~test_mask).nonzero()[0]
    fold_of = np.full(n, -1, dtype=int)
    for cls in sorted(set(labels)):
        idx = train_idx[labels[train_idx] == cls]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold_of[i] = j % 5
    out["folds"] = [[int(i) for i in train_idx[(fold_of[train_idx] == k)]]
                    for k in range(5)]
    import hashlib
    digest = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    out["sha256"] = digest
    sd = SPLIT / f"split_{args.tissue}"; sd.mkdir(parents=True, exist_ok=True)
    sp_path = sd / f"{args.tissue}_split.json"
    sp_path.write_text(json.dumps(out, sort_keys=True))
    (sd / f"{args.tissue}_split.sha256").write_text(f"{digest}  {sp_path.name}\n")
    print(f"split {args.tissue}: {n} cells, {out['n_test']} test, {len(out['classes'])} classes")

    # CL branch: full GT vocabulary (23 classes << 255 cap, no narrowing needed)
    opts = sorted(set(labels))
    assert len(opts) <= MAX_OPTIONS, f"{len(opts)} > {MAX_OPTIONS}"
    (sd / f"{args.tissue}_options.json").write_text(
        json.dumps({"tissue": args.tissue, "options": opts}, indent=1))
    print(f"options {args.tissue}: {len(opts)} (cap {MAX_OPTIONS}, narrowing: none needed)")


if __name__ == "__main__":
    main()
