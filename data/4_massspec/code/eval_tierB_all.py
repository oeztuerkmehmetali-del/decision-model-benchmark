"""x2 Tier B brute force: Jev yes/no verdict on EVERY PSM vs mokapot accept (q<=fdr), entrapment+decoy truth.
python3 code/eval_tierB_all.py --pin input/sage_nopep.pin --mok baseline/tB.mokapot.psms.txt --mok-decoy baseline/tB.mokapot.decoy.psms.txt \
   --jev jev/tB_all_choice.tsv [--mok-jev jev/tBjev.mokapot.psms.txt --mok-jev-decoy ...]
Known-false PSMs = decoys + ARATH-only targets. Rules compared (yes/no decisions):
  mokapot q<=1% | Jev choice==yes | Jev p ranked, cut at the same N as mokapot (matched N) | mokapot rerun with jev_p as a feature.
Per rule: N accepted, entrapment hits, decoy hits, FDP_hat(entrapment, combined est.), FDP_hat(TDC), Wilson CI of the entrapment rate.
"""
import argparse
import numpy as np, pandas as pd


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def auroc(s, y):
    r = pd.Series(s).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def report(name, acc, d, r):
    """acc: boolean mask over frame d (is_decoy, is_ent)."""
    s = d[acc]; nt = int((s.is_decoy == 0).sum()); nd = int(s.is_decoy.sum()); ne = int(s.is_ent.sum())
    lo, hi = wilson(ne, nt)
    fe = ne * (1 + 1 / r) / max(nt, 1); ft = (nd + 1) / max(nt, 1)
    print(f"{name:34s} targets {nt:6d}  entrap {ne:4d} (rate {ne/max(nt,1):.4f} Wilson[{lo:.4f},{hi:.4f}])  decoys {nd:5d}  FDP_hat entrap {fe:.4f}  TDC {ft:.4f}")
    return nt, ne


def load(a):
    prot = pd.read_csv(a.pin, sep="\t", usecols=["SpecId", "Proteins"]).astype({"SpecId": str}).set_index("SpecId").Proteins
    j = pd.read_csv(a.jev, sep="\t").astype({"SpecId": str}).set_index("SpecId")
    p = prot.reindex(j.index)
    j["is_ent"] = ((p.str.contains("_ARATH")) & ~p.str.contains("_HUMAN") & (j.is_decoy == 0)).astype(int)
    return j


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--pin", required=True); a.add_argument("--mok", required=True); a.add_argument("--mok-decoy", required=True)
    a.add_argument("--jev", required=True); a.add_argument("--mok-jev"); a.add_argument("--mok-jev-decoy")
    a.add_argument("--fdr", type=float, default=0.01); a.add_argument("--ratio", type=float, default=16345 / 20416)
    a.add_argument("--boot", type=int, default=300)
    a = a.parse_args()
    d = load(a); r = a.ratio
    d["false"] = ((d.is_decoy == 1) | (d.is_ent == 1)).astype(int)
    print(f"PSMs {len(d)}  decoys {int(d.is_decoy.sum())}  entrapment targets {int(d.is_ent.sum())}  Jev yes {int(d.jev_yes.sum())} ({d.jev_yes.mean():.1%})")
    y = 1 - d.false.to_numpy(); rng = np.random.default_rng(0)
    for nm, col in (("Jev p", "jev_p"), ("mokapot score", "score")):
        v = d[col].to_numpy(); au = auroc(v, y); bs = []
        for _ in range(a.boot):
            i = rng.integers(0, len(d), len(d)); bs.append(auroc(v[i], y[i]))
        print(f"AUROC (human target vs decoy/ARATH) all PSMs: {nm} {au:.3f} CI[{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]")
    tg = d[d.is_decoy == 0]; yt = (1 - tg.is_ent).to_numpy()
    rt = np.random.default_rng(1); bj, bm = [], []
    for _ in range(a.boot):
        i = rt.integers(0, len(tg), len(tg)); bj.append(auroc(tg.jev_p.to_numpy()[i], yt[i])); bm.append(auroc(tg.score.to_numpy()[i], yt[i]))
    print(f"HEADLINE AUROC targets only (human vs ARATH-only): Jev {auroc(tg.jev_p.to_numpy(), yt):.3f} CI[{np.percentile(bj,2.5):.3f},{np.percentile(bj,97.5):.3f}] mokapot {auroc(tg.score.to_numpy(), yt):.3f} CI[{np.percentile(bm,2.5):.3f},{np.percentile(bm,97.5):.3f}]; the line above includes decoys (decoy detection, mokapot trained on them)")
    mok_acc = ((d.q <= a.fdr) & (d.is_decoy == 0)).to_numpy()  # decoys never counted as accepted targets; TDC decoy count below
    # mokapot q<=fdr over targets and decoys (decoys passing the score cut count for TDC)
    cut = d.score[(d.q <= a.fdr) & (d.is_decoy == 0)].min()
    acc_m = (d.score >= cut).to_numpy()
    nm_, _ = report("mokapot q<=1%", acc_m, d, r)
    report("Jev choice == yes", d.jev_yes.to_numpy() == 1, d, r)
    # exact N: Jev p descending, ties broken by mokapot score (conservative: favours Jev)
    ij = np.zeros(len(d), bool); ij[np.lexsort((-d.score.to_numpy(), -d.jev_p.to_numpy()))[:int(acc_m.sum())]] = True
    report("Jev p, matched N (exact #PSMs, ties by mokapot score)", ij, d, r)
    both = acc_m & (d.jev_yes.to_numpy() == 1)
    report("mokapot AND Jev yes", both, d, r)
    only_m = acc_m & (d.jev_yes.to_numpy() == 0)
    report("mokapot accept, Jev says no", only_m, d, r)
    only_j = (~acc_m) & (d.jev_yes.to_numpy() == 1)
    report("Jev yes, mokapot rejects", only_j, d, r)
    # paired bootstrap, matched-N: entrapment hits Jev - mokapot
    g = []
    for _ in range(a.boot):
        i = rng.integers(0, len(d), len(d)); g.append(int(d.is_ent.to_numpy()[i][ij[i]].sum()) - int(d.is_ent.to_numpy()[i][acc_m[i]].sum()))
    print(f"matched-N entrapment hits, Jev - mokapot: mean {np.mean(g):+.1f} CI[{np.percentile(g,2.5):+.0f},{np.percentile(g,97.5):+.0f}]  (positive = Jev worse)")
    if a.mok_jev:
        t = pd.read_csv(a.mok_jev, sep="\t").astype({"SpecId": str}); dd = pd.read_csv(a.mok_jev_decoy, sep="\t").astype({"SpecId": str})
        t["is_decoy"] = 0; dd["is_decoy"] = 1; m2 = pd.concat([t, dd]).set_index("SpecId")
        m2["is_ent"] = d.reindex(m2.index).is_ent.to_numpy()
        cut2 = m2["mokapot score"][(m2["mokapot q-value"] <= a.fdr) & (m2.is_decoy == 0)].min()
        acc2 = (m2["mokapot score"] >= cut2).to_numpy()
        report("mokapot WITH jev_p/jev_yes features", acc2, m2.assign(is_decoy=m2.is_decoy), r)


if __name__ == "__main__":
    main()
