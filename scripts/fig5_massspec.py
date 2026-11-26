"""Fig 5, Results 'Mass spectrometry' (x2): Jev 1.13 and d3-ultra as match re-scorers vs mokapot 0.10.0 on one Sage search.
In:  data/4_massspec/compare/{master_result.txt,d3_all.tsv,small_data.tsv}, data/4_massspec/input/sage_ev.pin, data/4_massspec/jev/tB_boundary_sage_rel_ev.tsv; code from data/4_massspec/code (master_eval.py)
Out: figs/fig5_massspec.{pdf,png}; numbers.json["massspec"]
Stored results only. Run from merge_jev/paper/scripts/: python3 fig5_massspec.py"""
import re, sys
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from style import J, OUT, save, dump, JEV, OTHER, REF, BASE, TRAINED, WIDTH, plt
sys.path.insert(0, str(J / "4_massspec/code"))
from master_eval import R, IN, CMP, accepted, auroc, JEV as JEV_DIR  # JEV in style = colour

P = J / "4_massspec/compare/"
NAME = {"eng": "engine score", "jev": "Jev", "d3": "d3-ultra", "mok": "mokapot", "trn": "trained reader"}
COL = {"eng": BASE, "jev": JEV, "d3": OTHER, "mok": REF, "trn": TRAINED}

# ---- stored table (master_result.txt): every scorer, AUROC, accepted, % vs mokapot, CI ----
M = {}
rx = re.compile(r"(.+?)\s+(\d\.\d+) \[(\d\.\d+),(\d\.\d+)\]\s+(\d+) \[(\d+),(\d+)\]\s+([+-][\d.]+)% \[([+-][\d.]+)%,([+-][\d.]+)%\]")
for ln in open(P / "master_result.txt"):
    m = rx.match(ln)
    if m: M[m[1].strip()] = dict(auc=float(m[2]), auc_ci=(float(m[3]), float(m[4])), n=int(m[5]), n_ci=(int(m[6]), int(m[7])),
                                 pct=float(m[8]), ci=(float(m[9]), float(m[10])))
assert len(M) == 9
ROW = {"eng": "engine score alone (ln hyperscore), no model", "jev": "Jev 1.13 zero-shot (published run)", "d3": "d3 zero-shot, full context, labels masked",
       "mok": "mokapot (standard, 30 feat.)", "trn": "trained reader on Jev's inputs (mokapot, 24 feat.)"}
assert M[ROW["jev"]]["pct"] == -30.1 and M[ROW["jev"]]["ci"] == (-33.8, -21.7)
assert M[ROW["mok"]]["n"] == 27047 and M[ROW["trn"]]["n"] == 27071

# ---- per-match scores (same construction as make_figs_v2) ----
d = pd.read_csv(P / "d3_all.tsv", sep="\t").astype({"SpecId": str}).set_index("SpecId")
pin = pd.read_csv(f"{IN}/sage_ev.pin", sep="\t", usecols=["SpecId", "Proteins", "ln(hyperscore)"]).astype({"SpecId": str}).drop_duplicates("SpecId").set_index("SpecId")
t = d[d.is_decoy == 0]; pr = pin.Proteins.reindex(t.index)
ent = (pr.str.contains("_ARATH") & ~pr.str.contains("_HUMAN")).to_numpy().astype(int); y = 1 - ent
assert len(t) == 37061
S = {"eng": pin["ln(hyperscore)"].reindex(t.index).to_numpy(float), "jev": t.jev_p.to_numpy(float), "d3": t.d3_p.to_numpy(float), "mok": t.score.to_numpy(float)}
ends = lambda ss: np.r_[ss[1:] != ss[:-1], True]  # tie-group ends, as in master_eval.accepted

# ---- borderline band (2,799 matches) ----
band = pd.read_csv(f"{JEV_DIR}/tB_boundary_sage_rel_ev.tsv", sep="\t").astype({"SpecId": str}).set_index("SpecId")
inb = t.index.isin(band.index); yb = y[inb]
B = {"mok": S["mok"][inb], "jev": band.jev_p.reindex(t.index[inb]).to_numpy(float), "d3": S["d3"][inb]}  # Jev: its dedicated borderline run
au = {k: auroc(v, yb) for k, v in B.items()}
assert (int(inb.sum()), int((1 - yb).sum())) == (2799, 543) and [f"{au[k]:.3f}" for k in ("mok", "jev", "d3")] == ["0.653", "0.564", "0.551"], au

# ---- subset sizes ----
r = pd.read_csv(P / "small_data.tsv", sep="\t"); K = {"mok": "mokapot", "d3": "d3", "jev": "jev", "eng": "engine"}
st500 = r.loc[r["size"] == 500, "status"]
assert len(st500) == 20 and (st500 == "failed").all(), st500.value_counts()  # paper: mokapot trained on 0 of 20 subsets of 500


def size_panel(a, val, ylabel):
    for k in ("mok", "d3", "eng", "jev"):
        g = r[r["size"] >= 1000] if k == "mok" else r  # at 500 no mokapot model could be trained (20 of 20 failed)
        q = g.groupby("size").apply(lambda x: pd.Series(np.percentile(val(x, K[k]), [25, 50, 75])), include_groups=False)
        a.fill_between(q.index, q[0], q[2], color=COL[k], alpha=.15, lw=0)
        a.plot(q.index, q[1], "o-", color=COL[k], ms=3, lw=1.5 if k == "mok" else 1.1, label=NAME[k])
    sz = sorted(r["size"].unique())
    a.set_xscale("log"); a.set_xticks(sz, [f"{s / 1000:g}k" for s in sz], fontsize=6); a.minorticks_off()
    a.set(xlabel="matches in the run (subsets)", ylabel=ylabel)


acc_share = lambda g, k: 100 * g["acc_" + k] / g.targets
top = r[r["size"] == 20000]
assert abs(acc_share(top, "mokapot").median() - 72.8) < .06 and len(top) == 20  # small_data_result.txt, 20,000 row

# ===== main figure =====
f, ax = plt.subplots(1, 3, figsize=(WIDTH, 2.8), gridspec_kw={"width_ratios": [1.3, .8, 1.3]})
a = ax[0]
for k in ("mok", "d3", "eng", "jev"):
    s = S[k]; o = np.argsort(-s, kind="stable"); b = ends(s[o]); nt = np.arange(1, len(s) + 1)
    q = np.minimum.accumulate((np.cumsum(ent[o]) * (1 + 1 / R) / nt)[b][::-1])[::-1] * 100; n = nt[b]
    na = accepted(s, ent); assert na == M[ROW[k]]["n"] == n[q <= 1].max(), k
    a.plot(q, n / 1000, color=COL[k], lw=1.6 if k == "mok" else 1.1, label=NAME[k], zorder=3)
    a.plot(1, na / 1000, "o", color=COL[k], ms=4, mec="w", zorder=4)
a.axvline(1, color=BASE, ls=":", lw=.8); a.set(xlim=(0, 5), ylim=(0, 34), xlabel="allowed error (%)", ylabel="accepted matches ($\\times10^3$)")
a.set_title("a  Accepted matches vs allowed error")

a = ax[1]
rows = [("eng", "engine score"), ("jev", "Jev"), ("d3", "d3-ultra")]
for i, (k, lab) in enumerate(rows):
    e = M[ROW[k]]; v = e["pct"]
    a.errorbar(v, i, xerr=[[v - e["ci"][0]], [e["ci"][1] - v]], fmt="o", color=COL[k], capsize=2.5, ms=4.5, lw=1.2)
    a.text(e["ci"][1] + 1.2, i, f"{lab}\n{e['n']:,}", va="center", fontsize=6.3, color=COL[k])
a.axvline(0, color=REF, lw=1)
a.set_yticks([]); a.set_ylim(len(rows) - .5, -.5); a.set_xlim(-37, 9); a.grid(axis="y", visible=False); a.spines["left"].set_visible(False)
a.set(xlabel="change vs mokapot (%)", xticks=[-30, -15, 0])
a.set_title("b  At 1% error")

a = ax[2]
size_panel(a, acc_share, "accepted at 1% error (%)")
a.set_ylim(15, 80); a.set_title("c  Smaller runs")
h, l = ax[0].get_legend_handles_labels()
f.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, 1.03))
save(f, "fig5_massspec")
dump("massspec", {
    "targets": 37061, "known_wrong": int(ent.sum()),
    **{k: {"accepted": M[ROW[k]]["n"], "pct_vs_mokapot": M[ROW[k]]["pct"], "ci95": list(M[ROW[k]]["ci"]), "auroc": M[ROW[k]]["auc"]} for k in ROW},
    "borderline": {"n": 2799, "known_wrong": 543, "auroc": {k: round(float(v), 3) for k, v in au.items()}},
    "share_pct_median_at_20000": {k: float(acc_share(top, K[k]).median()) for k in K},
    "subset_mokapot_untrainable_at_500": bool((st500 == "failed").all())})
print("ok", {k: M[v]["n"] for k, v in ROW.items()}, au)
