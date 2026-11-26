"""Per-PSM fragment-ion evidence beyond Sage's PIN features: how well the matched b/y ions explain the actual spectrum.
python3 code/tierB_evidence.py --sage-dir input/sage_out --mzml <b1931.ms2.mzML, see input/EXTERNAL.tsv> --out input/evidence.tsv
Columns (all label-free; computed from spectrum + matched ions only, never from target/decoy status or peptide text):
  top5_expl, top10_expl, top20_expl : how many of the N most intense peaks are matched fragments
  tic_expl                          : matched intensity / total spectrum intensity
  b_frac, y_frac                    : matched b / y ions per possible ladder position (peptide_len-1)
  b_run, y_run                      : longest consecutive matched ladder run (ordinals)
  ladder                            : compact text of matched ordinals, e.g. 'b2-4,7 | y3-11'
"""
import argparse
import numpy as np, pandas as pd
from pyteomics import mzml


def runs(ords):
    ords = sorted(set(ords))
    out, i = [], 0
    while i < len(ords):
        j = i
        while j + 1 < len(ords) and ords[j + 1] == ords[j] + 1:
            j += 1
        out.append((ords[i], ords[j]))
        i = j + 1
    return out


def fmt(kind, rs):
    return kind + ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in rs) if rs else f"{kind}none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sage-dir", required=True); ap.add_argument("--mzml", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = pd.read_csv(f"{a.sage_dir}/results.sage.tsv", sep="\t", usecols=["psm_id", "scannr", "peptide_len"])
    fr = pd.read_csv(f"{a.sage_dir}/matched_fragments.sage.tsv", sep="\t")
    fr = fr[fr.fragment_type.isin(["b", "y"])]
    scan_of = dict(zip(res.psm_id, res.scannr.str.extract(r"scan=(\d+)")[0].astype(int))) if res.scannr.dtype == object else dict(zip(res.psm_id, res.scannr))
    need = set(scan_of.values())
    spec = {}
    with mzml.read(a.mzml, use_index=False) as r:
        for s in r:
            n = int(s["id"].rsplit("scan=", 1)[1])
            if n in need:
                spec[n] = (np.asarray(s["m/z array"], float), np.asarray(s["intensity array"], float))
    rows = []
    for pid, g in fr.groupby("psm_id"):
        mz, it = spec[scan_of[pid]]
        order = np.argsort(-it)
        # matched experimental peaks -> peak indices (nearest within 0.01 Th)
        idx = np.searchsorted(mz, g.fragment_mz_experimental.to_numpy())
        idx = np.clip(idx, 1, len(mz) - 1)
        idx = np.where(np.abs(mz[idx - 1] - g.fragment_mz_experimental.to_numpy()) < np.abs(mz[idx] - g.fragment_mz_experimental.to_numpy()), idx - 1, idx)
        ok = np.abs(mz[idx] - g.fragment_mz_experimental.to_numpy()) < 0.01
        hit = set(idx[ok].tolist())
        L = int(res.loc[res.psm_id == pid, "peptide_len"].iloc[0]) if False else None
        b = g[g.fragment_type == "b"].fragment_ordinals.tolist(); y = g[g.fragment_type == "y"].fragment_ordinals.tolist()
        rb, ry = runs(b), runs(y)
        rows.append(dict(psm_id=pid,
                         top5_expl=sum(i in hit for i in order[:5]), top10_expl=sum(i in hit for i in order[:10]),
                         top20_expl=sum(i in hit for i in order[:20]), tic_expl=float(it[list(hit)].sum() / it.sum()) if hit else 0.0,
                         nb=len(set(b)), ny=len(set(y)),
                         b_run=max((e - s + 1 for s, e in rb), default=0), y_run=max((e - s + 1 for s, e in ry), default=0),
                         ladder=f"{fmt('b', rb)} | {fmt('y', ry)}"))
    ev = pd.DataFrame(rows).merge(res[["psm_id", "peptide_len"]], on="psm_id", how="right")
    ev[["top5_expl", "top10_expl", "top20_expl", "nb", "ny", "b_run", "y_run"]] = ev[["top5_expl", "top10_expl", "top20_expl", "nb", "ny", "b_run", "y_run"]].fillna(0).astype(int)
    ev["tic_expl"] = ev.tic_expl.fillna(0.0); ev["ladder"] = ev.ladder.fillna("bnone | ynone")
    ev["b_frac"] = ev.nb / (ev.peptide_len - 1); ev["y_frac"] = ev.ny / (ev.peptide_len - 1)
    ev.drop(columns=["peptide_len"]).to_csv(a.out, sep="\t", index=False)
    print(f"{len(ev)} PSMs; mean top10_expl {ev.top10_expl.mean():.2f}, mean tic_expl {ev.tic_expl.mean():.3f}")


if __name__ == "__main__":
    main()
