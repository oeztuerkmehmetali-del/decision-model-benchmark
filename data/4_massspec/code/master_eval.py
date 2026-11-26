"""One table: every scorer on the same Tier B target PSMs; Arabidopsis-only target = known wrong.
Run from anywhere:  python code/master_eval.py | tee compare/master_result.txt   (paths resolve from this file; case folder layout)
Per scorer: AUROC (human vs Arabidopsis-only) and targets accepted at entrapment FDP <= 1%
(FDP_ent = N_ent(1+1/r)/N_T on the ranked prefix; tie groups kept whole, so no tie-break favours anyone),
with a PSM bootstrap paired against mokapot. All scorer inputs are species-blind, which the entrapment truth needs.
d3 rows appear once compare/d3_all.tsv (make_d3_all.py: d3-ultra asked Jev's exact requests) and d3u/tBd3.mokapot.psms.txt exist.
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_tierB_all import auroc  # noqa: E402

IN, JEV, D3U, BASE, CMP = (os.path.join(ROOT, d) for d in ("input", "jev", "d3u", "baseline", "compare"))  # ROOT = paper/data/4_massspec
R, THR, BOOT = 16345 / 20416, 0.01, 300  # r = paper's entrapment/target protein ratio


def accepted(s, ent):
    """targets in the largest ranked prefix (ending at a tie-group boundary) with FDP_ent <= THR"""
    o = np.argsort(-s, kind="stable"); ss = s[o]; ne = np.cumsum(ent[o]); nt = np.arange(1, len(s) + 1)
    ok = np.r_[ss[1:] != ss[:-1], True] & (ne * (1 + 1 / R) / nt <= THR)
    return int(nt[ok].max()) if ok.any() else 0


def main():
    d = pd.read_csv(f"{JEV}/tB_all_choice.tsv", sep="\t").astype({"SpecId": str})
    pin = pd.read_csv(f"{IN}/sage_ev.pin", sep="\t", usecols=["SpecId", "Proteins", "ln(hyperscore)"]).astype({"SpecId": str}).drop_duplicates("SpecId").set_index("SpecId")
    t = d[d.is_decoy == 0].set_index("SpecId"); pr = pin.Proteins.reindex(t.index)
    ent = (pr.str.contains("_ARATH") & ~pr.str.contains("_HUMAN")).to_numpy().astype(int)
    mok = lambda path: pd.read_csv(path, sep="\t").astype({"SpecId": str}).set_index("SpecId")["mokapot score"].reindex(t.index)
    rows = [("engine score alone (ln hyperscore), no model", pin["ln(hyperscore)"].reindex(t.index)),
            ("Jev 1.13 zero-shot (published run)", t.jev_p),
            ("trained reader on Jev's inputs (mokapot, 24 feat.)", mok(f"{BASE}/tB_fm_ev.mokapot.psms.txt")),
            ("mokapot (standard, 30 feat.)", t.score),
            ("mokapot + Jev 1.13 as extra feature", mok(f"{JEV}/tBjev.mokapot.psms.txt"))]
    f3 = os.path.join(CMP, "d3_all.tsv"); d3 = None
    if os.path.exists(f3):
        d3 = pd.read_csv(f3, sep="\t").astype({"SpecId": str}).set_index("SpecId").reindex(t.index)
        rows.insert(2, ("d3-ultra zero-shot (same requests as Jev)", d3.d3_p))
        # hard-case vote, both rules fixed before any band-level result of this d3 run was looked at: inside the paper's
        # borderline band (mokapot q 0.5-5%, one contiguous stretch of mokapot's ranking, contains the 1% cut) re-rank by
        # w*rank(d3) + (1-w)*rank(mokapot); outside the band keep mokapot's order
        inb = t.index.isin(pd.read_csv(f"{JEV}/tB_boundary_sage_rel_ev.tsv", sep="\t", usecols=["SpecId"]).SpecId.astype(str))
        lo, hi = t.score[inb].min(), t.score[inb].max(); assert t.score.between(lo, hi).sum() == inb.sum(), "band is not contiguous"
        for name, w in (("mokapot, hard cases re-ranked: equal vote with d3", .5), ("mokapot, hard cases re-ranked: d3 decides", 1)):
            r = w * d3.d3_p[inb].rank() + (1 - w) * t.score[inb].rank(); v = t.score.copy()
            v[inb] = (lo + (hi - lo) * (r - r.min()) / (r.max() - r.min())).to_numpy(); rows.append((name, v))
    if os.path.exists(os.path.join(D3U, "tBd3.mokapot.psms.txt")):
        rows.append(("mokapot + d3 as extra feature", mok(os.path.join(D3U, "tBd3.mokapot.psms.txt"))))
    S = {k: v.to_numpy(float) for k, v in rows}
    assert all(not np.isnan(v).any() for v in S.values()), "a scorer is missing target PSMs"
    y = 1 - ent; n = len(t); rng = np.random.default_rng(0); ref = "mokapot (standard, 30 feat.)"
    q1 = (t.q <= 0.01).to_numpy()
    print(f"targets {n} (Arabidopsis-only, known wrong: {int(ent.sum())}); r = {R:.4f}")
    print(f"yardstick: mokapot's own q<=1% set = {int(q1.sum())} targets, measured FDP_ent {ent[q1].sum() * (1 + 1 / R) / q1.sum():.2%}\n")
    bs = {k: [] for k in S}
    for _ in range(BOOT):
        i = rng.integers(0, n, n)
        for k, v in S.items():
            bs[k].append((auroc(v[i], y[i]), accepted(v[i], ent[i])))
    print(f"{'scorer':52s} {'AUROC [95% CI]':22s} {'accepted @1% FDP [95% CI]':30s} vs mokapot [95% CI]")
    for k, v in S.items():
        b = np.array(bs[k]); rel = b[:, 1] / np.array(bs[ref])[:, 1] - 1; na = accepted(v, ent)
        print(f"{k:52s} {auroc(v, y):.3f} [{np.percentile(b[:, 0], 2.5):.3f},{np.percentile(b[:, 0], 97.5):.3f}]  "
              f"{na:6d} [{np.percentile(b[:, 1], 2.5):.0f},{np.percentile(b[:, 1], 97.5):.0f}]  "
              f"{na / accepted(S[ref], ent) - 1:+.1%} [{np.percentile(rel, 2.5):+.1%},{np.percentile(rel, 97.5):+.1%}]")
    if d3 is not None:
        hs = S["engine score alone (ln hyperscore), no model"]  # scrambled PSM assignment would give ~0 here
        rho = pd.Series(S["d3-ultra zero-shot (same requests as Jev)"]).corr(pd.Series(hs), method="spearman")
        print(f"\nmapping self-check: rank correlation of d3 with the engine score = {rho:.2f}"); assert rho > 0.2, "d3 answers look mis-assigned"
        yb = y[inb]; df = []  # the band is cut on mokapot's score, which lowers mokapot's own AUROC inside it
        A = {"mokapot": S[ref][inb], "d3-ultra": S["d3-ultra zero-shot (same requests as Jev)"][inb],
             "equal vote": S["mokapot, hard cases re-ranked: equal vote with d3"][inb], "Jev whole-run answers": S["Jev 1.13 zero-shot (published run)"][inb]}
        for _ in range(BOOT):
            i = rng.integers(0, inb.sum(), inb.sum()); df.append([auroc(v[i], yb[i]) for v in A.values()])
        df = np.array(df); df = df - df[:, :1]
        print(f"hard cases only ({int(inb.sum())} band targets, {int((1 - yb).sum())} known wrong): AUROC " + "; ".join(
            f"{k} {auroc(v, yb):.3f}" + (f" (vs mokapot {auroc(v, yb) - auroc(A['mokapot'], yb):+.3f} [{np.percentile(df[:, j], 2.5):+.3f},{np.percentile(df[:, j], 97.5):+.3f}])" if j else "")
            for j, (k, v) in enumerate(A.items())))


if __name__ == "__main__":
    main()
