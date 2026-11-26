#!/usr/bin/env python3
"""X7 stage C: QC + normalize + Leiden clustering per tissue.

Runs on helix with x7/venv (py3.11 + scanpy). Reads data/raw/<tissue>.h5ad,
writes data/processed/<tissue>.h5ad with lognorm counts in layers["lognorm"] and
leiden clusters in obs["leiden"]. Local mirror: helix_qc.py -> helix scripts/qc_cluster.py

  ../venv/bin/python qc_cluster.py --tissue pancreas
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "input" / "external" / "raw"
PROC = ROOT / "input" / "external"
LOGS = ROOT / "logs"


def setup_logging(stage: str) -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOGS / f"{stage}.log")],
    )
    return logging.getLogger(stage)


def symbols_as_var_names(adata, log: logging.Logger) -> None:
    """Census var_names are Ensembl ids; switch to gene symbols (keep ids)."""
    if "feature_name" not in adata.var.columns:
        log.warning("no feature_name column; keeping var_names as-is")
        return
    adata.var["gene_ids"] = adata.var_names.astype(str)
    adata.var_names = adata.var["feature_name"].astype(str).values
    adata.var_names_make_unique()


def run_leiden(adata, resolution: float, seed: int, log: logging.Logger) -> None:
    import scanpy as sc

    try:  # leidenalg runs to convergence; igraph n_iterations=2 over-fragments
        sc.tl.leiden(adata, resolution=resolution, key_added="leiden", random_state=seed,
                     flavor="leidenalg")
    except (TypeError, ImportError) as exc:
        log.warning("leidenalg unavailable (%s); falling back to igraph", exc)
        sc.tl.leiden(adata, resolution=resolution, key_added="leiden", random_state=seed,
                     flavor="igraph", directed=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tissue", required=True)
    ap.add_argument("--min-genes", type=int, default=200, help="min genes per cell")
    ap.add_argument("--min-cells", type=int, default=3, help="min cells per gene")
    ap.add_argument("--max-pct-mito", type=float, default=20.0)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--n-pcs", type=int, default=50)
    ap.add_argument("--resolution", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import scanpy as sc

    log = setup_logging(f"qc_{args.tissue}")
    log.info("x7 qc | tissue=%s | python=%s", args.tissue, sys.executable)

    raw = RAW / f"{args.tissue}.h5ad"
    if not raw.exists():
        raise SystemExit(f"missing {raw}; run fetch.py first")
    out = PROC / f"{args.tissue}.h5ad"
    PROC.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(raw)
    symbols_as_var_names(adata, log)
    adata.layers["counts"] = adata.X.copy()
    n0 = adata.n_obs

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    adata = adata[adata.obs["pct_counts_mt"] <= args.max_pct_mito].copy()
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    log.info("cells %d -> %d (%.1f%% kept); genes -> %d",
             n0, adata.n_obs, 100 * adata.n_obs / n0, adata.n_vars)
    if adata.n_obs < 100:
        raise SystemExit(f"only {adata.n_obs} cells left after QC for {args.tissue}")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()

    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg, flavor="seurat")
    log.info("hvg: %d", int(adata.var["highly_variable"].sum()))
    hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(hvg, max_value=10)
    sc.tl.pca(hvg, n_comps=args.n_pcs, svd_solver="arpack", random_state=args.seed)
    adata.obsm["X_pca"] = hvg.obsm["X_pca"]
    adata.uns["pca"] = hvg.uns["pca"]
    if "donor_id" in adata.obs.columns:
        try:
            import scanpy.external as sce
            sce.pp.harmony_integrate(hvg, key="donor_id", basis="X_pca", adjusted_basis="X_pca_harmony")
            adata.obsm["X_pca_harmony"] = hvg.obsm["X_pca_harmony"]
            log.info("harmony: donor-corrected X_pca_harmony")
        except Exception as e:
            adata.obsm["X_pca_harmony"] = hvg.obsm["X_pca"]
            log.warning("harmony failed (%s); using raw PCA", e)
    else:
        adata.obsm["X_pca_harmony"] = hvg.obsm["X_pca"]
        log.info("harmony: no donor_id, using raw PCA")
    del hvg
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=args.n_pcs, random_state=args.seed,
                    use_rep="X_pca_harmony")
    run_leiden(adata, args.resolution, args.seed, log)

    n_clusters = adata.obs["leiden"].nunique()
    log.info("leiden res=%.2f -> %d clusters; sizes=%s", args.resolution, n_clusters,
             adata.obs["leiden"].value_counts().sort_index().tolist())
    if n_clusters < 2:
        raise SystemExit(f"leiden produced {n_clusters} cluster(s) for {args.tissue}")

    adata.uns["x7_qc"] = {
        "tissue": args.tissue,
        "n_cells_in": int(n0),
        "n_cells_out": int(adata.n_obs),
        "n_genes_out": int(adata.n_vars),
        "params": vars(args),
        "n_clusters": int(n_clusters),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = out.with_suffix(".h5ad.tmp")
    adata.write_h5ad(tmp, compression="gzip")
    tmp.replace(out)
    log.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    log.info("qc done")


if __name__ == "__main__":
    main()
