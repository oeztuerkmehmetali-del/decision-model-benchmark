#!/usr/bin/env python3
"""X7 stage B: fetch Tabula Sapiens Pancreas + Heart from CELLxGENE Census.

Runs on helix with x7/venv (py3.11 + cellxgene-census). Writes data/raw/<tissue>.h5ad.
Local mirror: helix_fetch.py -> helix scripts/fetch.py

  ../venv/bin/python fetch.py [--tissue pancreas] [--force]

Env overrides: X7_CENSUS_VERSION (default "2025-11-08", pinned).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "input" / "external" / "raw"
LOGS = ROOT / "logs"

CENSUS_VERSION = os.environ.get("X7_CENSUS_VERSION", "2025-11-08")  # resolved from "stable" on 2026-10-05 (helix log); pinned for reproducibility
TISSUES = ("pancreas", "heart")
OBS_COLS = [
    "cell_type",
    "cell_type_ontology_term_id",
    "tissue",
    "tissue_general",
    "dataset_id",
    "donor_id",
    "assay",
    "is_primary_data",
]
VAR_COLS = ["feature_name"]


def setup_logging(stage: str) -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOGS / f"{stage}.log")],
    )
    return logging.getLogger(stage)


def tabula_sapiens_datasets(census, log: logging.Logger) -> list[str]:
    """Dataset ids whose collection/title is Tabula Sapiens."""
    ds = census["census_info"]["datasets"].read().concat().to_pandas()
    hay = ds["collection_name"].fillna("") + " | " + ds["dataset_title"].fillna("")
    hit = ds[hay.str.contains("Tabula Sapiens", case=False, na=False)]
    if hit.empty:
        raise SystemExit("no Tabula Sapiens dataset found in census %s" % CENSUS_VERSION)
    ids = sorted(hit["dataset_id"].unique())
    tis = sorted(hit["tissue_general"].unique())[:20] if "tissue_general" in hit.columns else "n/a"
    log.info("Tabula Sapiens: %d datasets, tissue_general=%s", len(ids), tis)
    return ids


def fetch_tissue(census, ids: list[str], tissue: str, log: logging.Logger, force: bool) -> None:
    import cellxgene_census

    out = RAW / f"{tissue}.h5ad"
    if out.exists() and not force:
        log.info("skip %s (exists, %.1f MB); --force to refetch", out, out.stat().st_size / 1e6)
        return

    id_list = "[" + ",".join(f'"{i}"' for i in ids) + "]"
    obs_filter = f'dataset_id in {id_list} and tissue_general == "{tissue}" and is_primary_data == True'
    log.info("querying %s | %s", tissue, obs_filter)

    kwargs = dict(organism="Homo sapiens", obs_value_filter=obs_filter, X_name="raw")
    try:  # census >= 1.15 uses column_names; older uses obs_column_names
        adata = cellxgene_census.get_anndata(
            census, column_names={"obs": OBS_COLS, "var": VAR_COLS}, **kwargs
        )
    except TypeError:
        adata = cellxgene_census.get_anndata(
            census, obs_column_names=OBS_COLS, var_column_names=VAR_COLS, **kwargs
        )

    log.info("%s: %d cells x %d genes", tissue, adata.n_obs, adata.n_vars)
    if adata.n_obs == 0:
        raise SystemExit(f"empty fetch for {tissue} (filter matched no cells)")

    adata.uns["x7"] = {
        "source": "cellxgene-census",
        "census_version": CENSUS_VERSION,
        "tissue": tissue,
        "dataset_ids": ids,
        "obs_filter": obs_filter,
        "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".h5ad.tmp")
    adata.write_h5ad(tmp, compression="gzip")
    tmp.replace(out)
    log.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tissue", action="append", choices=TISSUES, help="repeatable; default both")
    ap.add_argument("--force", action="store_true", help="refetch even if output exists")
    args = ap.parse_args()

    log = setup_logging("fetch")
    log.info("x7 fetch | census=%s | python=%s", CENSUS_VERSION, sys.executable)

    import cellxgene_census

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        ids = tabula_sapiens_datasets(census, log)
        for tissue in args.tissue or TISSUES:
            fetch_tissue(census, ids, tissue, log, args.force)

    log.info("fetch done")


if __name__ == "__main__":
    main()
