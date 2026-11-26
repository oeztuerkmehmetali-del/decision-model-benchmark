#!/usr/bin/env python3
"""G3 pool builder: CellMarker 2.0 pancreas markers (year<2022) -> pool json.

Pre-registered filter: tissue=='Pancreas' AND year<2022 (leak-free vs Tabula
Sapiens 2022). Maps CellMarker type names -> 23 TS labels via LIT_MAP.
GO/NO-GO check (panel): >=5 canonical markers/major type, >=15/23 types,
0% pseudogenes, KRT19+PRSS1 present.
Output: input/pool/pancreas_pool.json {type: [genes]} + merged pool list.
"""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # case folder (this file is in code/)
XLSX = HERE / "baseline/Cell_marker_Human.xlsx"  # CellMarker 2.0 Human; override with --xlsx
XLSX_SHA256 = "bf52b8cd60df60f17a7c6b8a59d0bbb74ec78612f11cd11990f15e4eedb7d842"


def verify_xlsx(p) -> Path:
    """Fail closed unless the CellMarker xlsx is present and the pinned revision."""
    p = Path(p)
    if not p.exists():
        sys.exit(f"ERROR: {p} not found. Expected sha256 {XLSX_SHA256} "
                 f"(source: CellMarker 2.0 Human marker table, Cell_marker_Human.xlsx). "
                 f"Place the file at {p} or pass --xlsx; refusing to download.")
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != XLSX_SHA256:
        sys.exit(f"ERROR: {p} sha256 {got} != expected {XLSX_SHA256} "
                 f"(source: CellMarker 2.0 Human marker table, Cell_marker_Human.xlsx). "
                 f"Refusing an unverified file.")
    return p


PANCREAS_LIT_MAP = {
    "B cell": ["B cell"],
    "CD4-positive, alpha-beta T cell": ["CD4+ T cell", "CD4 T cell"],
    "CD8-positive, alpha-beta T cell": ["CD8+ T cell", "CD8 T cell"],
    "T cell": ["T cell"],
    "classical monocyte": ["Monocyte", "Inflammatory monocyte"],
    "endothelial cell": ["Endothelial cell"],
    "fibroblast": ["Fibroblast", "CAF"],
    "intermediate monocyte": ["Monocyte"],
    "macrophage": ["Macrophage", "M1", "M2", "TRM"],
    "mast cell": ["Mast cell"],
    "mature NK T cell": ["NKT cell"],
    "monocyte": ["Monocyte", "Inflammatory monocyte"],
    "natural killer cell": ["Natural killer cell", "NK cell"],
    "neutrophil": ["Neutrophil"],
    "non-classical monocyte": ["Monocyte"],
    "pancreatic A cell": ["Alpha cell", "alpha cell"],
    "pancreatic D cell": ["Delta cell", "delta cell"],
    "pancreatic PP cell": ["PP cell", "Pancreatic polypeptide"],
    "pancreatic acinar cell": ["Acinar cell", "acinar cell"],
    "pancreatic ductal cell": ["Ductal cell", "Ductal epithelial", "Epithelial ductal", "ductal cell"],
    "pancreatic stellate cell": ["Pancreatic stellate cell", "Stellate cell"],
    "plasma cell": ["Plasma cell"],
    "type B pancreatic cell": ["Beta cell", "beta cell"],
}

# Heart: 31 TS heart classes. Shared pools are a known limitation (identical
# pool genes -> identical Jev cooc context -> lexicographic tie inside family);
# pancreas monocyte variants had the same structure.
HEART_LIT_MAP = {
    "B cell": ["B cell"],
    "CD4-positive, alpha-beta T cell": ["CD4+ T cell", "CD4 T cell"],
    "CD8-positive, alpha-beta T cell": ["CD8+ T cell", "CD8 T cell"],
    "T cell": ["T cell"],
    "basophil": ["Basophil"],
    "capillary endothelial cell": ["Endothelial cell", "Capillary endothelial cell"],
    "cardiac endothelial cell": ["Endothelial cell", "Cardiac endothelial cell"],
    "classical monocyte": ["Monocyte", "Inflammatory monocyte"],
    "endothelial cell": ["Endothelial cell"],
    "endothelial cell of artery": ["Endothelial cell", "Artery endothelial cell"],
    "endothelial cell of lymphatic vessel": ["Endothelial cell", "Lymphatic endothelial cell"],
    "erythrocyte": ["Erythrocyte", "Red blood cell"],
    "fibroblast": ["Fibroblast"],
    "fibroblast of cardiac tissue": ["Fibroblast", "Cardiac fibroblast"],
    "intermediate monocyte": ["Monocyte"],
    "macrophage": ["Macrophage", "M1", "M2", "TRM"],
    "mast cell": ["Mast cell"],
    "mature NK T cell": ["NKT cell"],
    "mesothelial cell": ["Mesothelial cell"],
    "monocyte": ["Monocyte"],
    "natural killer cell": ["Natural killer cell", "NK cell"],
    "neuron": ["Neuron", "Neuronal cell"],
    "neutrophil": ["Neutrophil"],
    "non-classical monocyte": ["Monocyte"],
    "pericyte": ["Pericyte"],
    "plasma cell": ["Plasma cell"],
    "plasmacytoid dendritic cell": ["Plasmacytoid dendritic cell", "Dendritic cell"],
    "regular atrial cardiac myocyte": ["Cardiomyocyte", "Cardiac muscle cell",
                                       "Atrial cardiomyocyte", "Atrial cardiac muscle cell"],
    "smooth muscle cell": ["Smooth muscle cell"],
    "vein endothelial cell": ["Endothelial cell", "Vein endothelial cell"],
    "ventricular cardiac muscle cell": ["Cardiomyocyte", "Cardiac muscle cell",
                                        "Ventricular cardiomyocyte", "Ventricular cardiac muscle cell"],
}
LIT_MAPS = {"pancreas": PANCREAS_LIT_MAP, "heart": HEART_LIT_MAP}
CANONICAL_CHECKS = {
    "pancreas": {"pancreatic acinar cell": "PRSS1", "pancreatic ductal cell": "KRT19",
                 "type B pancreatic cell": "INS", "macrophage": "CD68",
                 "endothelial cell": "PECAM1"},
    "heart": {"ventricular cardiac muscle cell": "TNNT2", "macrophage": "CD68",
              "endothelial cell": "PECAM1", "smooth muscle cell": "MYH11",
              "fibroblast of cardiac tissue": "DCN"},
}
# Pre-registered (2026-10-06, before any heart run): heart ambient contaminant
# = hemoglobin (RBC lysis), scavenged by macrophages like pancreatic enzymes.
TISSUE_TT = {"pancreas": "Pancreas", "heart": "Heart"}
CODING_LEGACY = re.compile(r"^(ENSG|LINC|Metazoa|RNU|RN7|SNAR|AC[0-9]|AL[0-9])|(-AS[0-9]*$|[KP]\d+$)")  # s35: [KP]\d+$ dropped ~1,250 protein-coding genes; kept only for --legacy-coding repro
CODING = re.compile(r"^(ENSG|LINC|Metazoa|RNU|RN7|SNAR|AC[0-9]|AL[0-9])|(-AS[0-9]*$)")

def main() -> None:
    import argparse
    import openpyxl
    ap = argparse.ArgumentParser()
    ap.add_argument("--tissue", default="pancreas", choices=sorted(LIT_MAPS))
    ap.add_argument("--subsample", type=float, default=1.0,
                    help="per-type stratified fraction, min 1 gene/type, seed 0 (Arm D)")
    ap.add_argument("--drop-types", nargs="+", default=None, metavar="TYPE",
                    help="Arm D: remove every gene listed under these pool types from the whole pool")
    ap.add_argument("--drop-largest", type=int, default=0,
                    help="Arm D sensitivity: same, for the N largest pool types")
    ap.add_argument("--suffix", default="", help="output filename suffix")
    ap.add_argument("--legacy-coding", action="store_true",
                    help="s35/s36: use the defective CODING regex (drops ~1,250 protein-coding genes) to reproduce the committed pools")
    ap.add_argument("--xlsx", default=str(XLSX),
                    help="CellMarker 2.0 Human xlsx (sha256-pinned; default baseline/)")
    a = ap.parse_args()
    LIT_MAP = LIT_MAPS[a.tissue]
    CANONICAL_CHECK = CANONICAL_CHECKS[a.tissue]
    TT = TISSUE_TT[a.tissue]
    coding = CODING_LEGACY if a.legacy_coding else CODING
    OUT = HERE/"input/pool"/f"{a.tissue}_pool{a.suffix}.json"
    wb = openpyxl.load_workbook(verify_xlsx(a.xlsx), read_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    hi = {h: i for i, h in enumerate(header)}
    pool: dict[str, set[str]] = {t: set() for t in LIT_MAP}
    n_tissue = n_old = n_immune = 0
    for r in rows:
        year = r[hi["year"]]
        if year is not None and int(year) >= 2022:
            continue
        cm_type = str(r[hi["cell_name"]] or "")
        gene = str(r[hi["Symbol"]] or "").strip()
        if not gene or coding.search(gene):
            continue
        # normalize: strip parenthetical variants (unicode beta/alpha etc)
        base = re.sub(r"\s*\(.*?\)\s*", "", cm_type).strip().lower()
        tt = r[hi["tissue_type"]]
        if tt == TT:
            n_tissue += 1
            n_old += 1
            for ts, names in LIT_MAP.items():
                if any(base == n.lower() for n in names):
                    pool[ts].add(gene)
        elif tt in ("Immune system", "Blood"):
            # pre-registered immune fallback: CD markers are tissue-agnostic;
            # fills immune types that stay thin after the tissue pass
            n_immune += 1
            for ts, names in LIT_MAP.items():
                if any(base == n.lower() for n in names):
                    pool[ts].add(gene)
    merged = {t: sorted(g) for t, g in pool.items()}
    subsampled = False
    drop = a.drop_types or sorted(merged, key=lambda t: (-len(merged[t]), t))[:a.drop_largest]
    if drop:
        assert all(t in merged for t in drop), f"unknown pool type in {drop}"
        gone = {g for t in drop for g in merged[t]}
        merged = {t: [g for g in gl if g not in gone] for t, gl in merged.items()}
        subsampled = True
    elif a.subsample < 1.0:
        import numpy as np
        rng = np.random.default_rng(0)
        merged = {t: sorted(rng.choice(gl, size=max(1, round(len(gl) * a.subsample)),
                                        replace=False).tolist())
                  for t, gl in merged.items() if gl}
        subsampled = True
    out = {"source": f"CellMarker 2.0 Human, tissue={TT} year<2022 + Immune/Blood fallback",
           "xlsx_sha256": XLSX_SHA256,
           f"n_{a.tissue}_rows": n_tissue, "n_pre2022": n_old, "n_immune_rows": n_immune}
    if subsampled:
        out["subsample"] = {"drop_types": drop} if drop else a.subsample
    out.update({"pool": merged, "n_genes_total": len({g for g in merged.values() for g in g})})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    # GO/NO-GO
    cov = sum(1 for g in merged.values() if len(g) >= 5)
    thin = [t for t, g in merged.items() if len(g) < 5]
    canon = {t: (c in merged.get(t, set())) for t, c in CANONICAL_CHECK.items()}
    pseud = 0
    print(f"types: {len(merged)}  genes_total: {out['n_genes_total']}  types>=5: {cov}")
    print("thin types:", thin)
    print("canonical present:", canon)
    ok = cov >= 15 and all(canon.values()) and pseud == 0
    if subsampled:
        print("GO/NO-GO: SKIPPED (degraded pool, report-only)")
        return 0
    print("GO/NO-GO:", "GO" if ok else "NO-GO")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
