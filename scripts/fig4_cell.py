"""Fig 4, Results 'Cell types' (x7): Jev 1.13 and d3-ultra vs CellMarker 2.0 scorers on pancreas and heart.
In:  data/3_cell/{compare/jev_vs_baseline/{pancreas,heart}_eval_v36.json, compare/jev_vs_baseline/heart_donor_bootstrap{,_M2}.json,
     compare/d3u_vs_baseline/{pancreas,heart}_eval_v36.json, compare/d3u_vs_baseline/heart_donor_bootstrap_M2.json,
     input/processed/pancreas_fullcells_lit_fixed.npz} (rebuild chain: data/3_cell/README.md)
Out: figs/fig4_cell.{pdf,png}; numbers.json["cell"]
Stored results only (no model calls). Run from merge_jev/paper/: python3 fig4_cell.py"""
import json
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from style import J, OUT, save, dump, JEV, OTHER, REF, BASE, POSTHOC, EXTRA, TRAINED, WIDTH, plt

X7 = J / "3_cell"
load = lambda p: json.load(open(X7 / p))
TISS = ("pancreas", "heart")
ev = {t: load(f"compare/jev_vs_baseline/{t}_eval_v36.json") for t in TISS}
M = {t: ev[t]["methods"] for t in TISS}
NTEST = {t: ev[t]["n_test"] for t in TISS}
assert NTEST == {"pancreas": 2440, "heart": 7075}
pre = lambda m, p: [v for k, v in m.items() if k.startswith(p)]
mean_f = lambda m, p, f: float(np.mean([v[f] for v in pre(m, p)]))
sd_f = lambda m, p, f: float(np.std([v[f] for v in pre(m, p)], ddof=1))

# scorer key, legend label, colour, hatch, alpha
SC = [("M1_cellmarker_uniform", "CellMarker 2.0 list (M1)", REF, None, 1),
      ("M2_zscore_cellmarker", "CellMarker 2.0 z-score (M2)", REF, None, .45),
      ("M1n_size_normalized_posthoc", "CellMarker 2.0 size-normalised, post hoc (M1n)", POSTHOC, "///", 1),
      ("jev_v36_jevonly", "Jev 1.13 alone (M3)", JEV, None, 1),
      ("jev_v36_cooc", "Jev 1.13 + literature co-mention (M4)", EXTRA, None, 1)]
# d3-ultra arms (post hoc second model, x7 Section 3.9): its three builds are identical, so build 1 stands for all
D3 = {t: load(f"compare/d3u_vs_baseline/{t}_eval_v36.json")["methods"] for t in TISS}
D3U = [("jevonly", "d3-ultra alone", OTHER, None, 1), ("cooc", "d3-ultra + literature", OTHER, "...", .5)]
for t in TISS:
    for k in ("M0_majority", "M1_cellmarker_uniform", "M2_zscore_cellmarker", "M1n_size_normalized_posthoc"):
        assert D3[t][k]["acc"] == M[t][k]["acc"]  # same cells and database scorers as the Jev run
    for a_ in ("jevonly", "cooc"):
        assert len({D3[t][f"jev_v36_{a_}_r{r}"]["acc"] for r in (1, 2, 3)}) == 1
D3ACC = {t: {a_: D3[t][f"jev_v36_{a_}_r1"]["acc"] for a_, *_ in D3U} for t in TISS}
D3MAC = {t: {a_: D3[t][f"jev_v36_{a_}_r1"]["macro_acc"] for a_, *_ in D3U} for t in TISS}
assert [round(100 * D3ACC["pancreas"][a_], 1) for a_ in ("jevonly", "cooc")] == [80.0, 76.7]
assert [round(100 * D3ACC["heart"][a_], 1) for a_ in ("jevonly", "cooc")] == [39.3, 28.4]


def stat(t, key, f):  # builds averaged for Jev arms (three builds), single value for database scorers
    return mean_f(M[t], key, f) if key.startswith("jev") else M[t][key][f]


ACC = {t: {k: stat(t, k, "acc") for k, *_ in SC} for t in TISS}
MAC = {t: {k: stat(t, k, "macro_acc") for k, *_ in SC} for t in TISS}
MAJ = {t: M[t]["M0_majority"]["acc"] for t in TISS}
# ---- asserts against paper_readable_v2.tex (abstract / Results 3.x) ----
assert round(100 * ACC["pancreas"]["M1_cellmarker_uniform"], 1) == 83.3
assert round(100 * ACC["pancreas"]["jev_v36_cooc"], 1) == 84.6
assert round(100 * ACC["heart"]["jev_v36_jevonly"], 1) == 44.0
assert round(100 * ACC["heart"]["M1n_size_normalized_posthoc"], 1) == 23.4
assert round(ACC["heart"]["M1_cellmarker_uniform"], 4) == 0.0027  # "uniform database scorer gets 0.0027 of heart cells"
assert round(MAJ["heart"], 3) == 0.232  # "most common type (0.232)"
assert round(100 * ACC["heart"]["M2_zscore_cellmarker"], 1) == 19.8

# heart comparator = M2 (z-score): predates the results but outside x7's written rule (rule comparator M1n, post hoc)

donor = load("compare/jev_vs_baseline/heart_donor_bootstrap_M2.json")["results"]
donor_d3f = load("compare/d3u_vs_baseline/heart_donor_bootstrap_M2.json"); donor_d3 = donor_d3f["results"]
donor_m1n = load("compare/jev_vs_baseline/heart_donor_bootstrap.json")["results"]
assert [round(x, 3) for x in donor_m1n["_v36_jevonly_r1"]["micro_diff_ci95"]] == [-0.016, 0.347]
assert [round(100 * x, 1) for x in donor["_v36_jevonly_r1"]["micro_diff_ci95"]] == [11.1, 36.3]
assert [round(100 * x, 1) for x in donor_d3["_v36_jevonly_r1"]["micro_diff_ci95"]] == [5.9, 33.0]


def wald(t, tag, vk, src=M):  # cell-level paired Wald 95% CI in pp from McNemar b/c counts (as in x7 make_figs.fig_paired)
    v = src[t][tag][vk]; n = NTEST[t]
    b = next(v[k] for k in v if k.startswith("b_")); c = next(v[k] for k in v if k.startswith("c_"))
    se = np.sqrt(max(b + c - (c - b) ** 2 / n, 0)) / n
    d = 100 * v["delta_acc"]
    return d, d - 196 * se, d + 196 * se, v


# =============== MAIN FIGURE ===============
fig = plt.figure(figsize=(WIDTH, 3.3))
g_ = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=.62, left=.06, right=.99, top=.9, bottom=.34)
a, b = (fig.add_subplot(g_[i]) for i in range(2))

# ---- a: accuracy, one CellMarker baseline per tissue (pancreas plain list, heart z-score) ----
BASEK = {"pancreas": "M1_cellmarker_uniform", "heart": "M2_zscore_cellmarker"}
w = .15
BARS = [("CellMarker 2.0 baseline", REF, None, 1, lambda t, f: (ACC if f == "a" else MAC)[t][BASEK[t]])] + \
       [(lab.replace(" (M3)", "").replace(" (M4)", "").replace("literature co-mention", "literature"), col, hat, al,
         lambda t, f, k=key: (ACC if f == "a" else MAC)[t][k]) for key, lab, col, hat, al in SC[3:]] + \
       [(lab, col, hat, al, lambda t, f, k=a_: (D3ACC if f == "a" else D3MAC)[t][k]) for a_, lab, col, hat, al in D3U]
for i, (lab, col, hat, al, get) in enumerate(BARS):
    xs = np.arange(2) + (i - 2) * w
    vals = [100 * get(t, "a") for t in TISS]; mvals = [100 * get(t, "m") for t in TISS]
    a.bar(xs, vals, w, color=col, alpha=al, hatch=hat, ec="k" if hat else "white", lw=.3, label=lab)
    a.scatter(xs, mvals, marker="D", s=9, fc="white", ec="k", lw=.5, zorder=4)
    for x, v, mv in zip(xs, vals, mvals):
        a.text(x, max(v, mv) + 4, f"{v:.1f}", ha="center", va="bottom", rotation=90, fontsize=5)
for g, t in enumerate(TISS):
    a.plot([g - .45, g + .45], [100 * MAJ[t]] * 2, ls=":", color=BASE, lw=1)
a.set_xticks(range(2), [f"pancreas\n{NTEST['pancreas']:,} cells", f"heart\n{NTEST['heart']:,} cells"])
a.grid(axis="x", visible=False)
a.set(ylim=(0, 108), ylabel="test cells named correctly (%)", yticks=range(0, 101, 20))
h, l = a.get_legend_handles_labels()
h += [Line2D([], [], marker="D", ms=3.2, mfc="white", mec="k", mew=.6, ls="none"), Line2D([], [], ls=":", color=BASE, lw=1)]
l += ["macro accuracy", "most common type"]
a.legend(h, l, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.26), fontsize=5.2, handlelength=1.2, handletextpad=.4, labelspacing=.25, columnspacing=.8)
a.set_title("a  Accuracy by tissue")

# ---- b: paired differences with intervals ----
rows = []  # (label, colour, point, lo, hi, macro point)
ARMS = ((M, donor, "jev_v36_jevonly_r1", "_v36_jevonly_r1", JEV, "Jev 1.13 alone"), (M, donor, "jev_v36_cooc_r1", "_v36_cooc_r1", EXTRA, "Jev 1.13 + literature"),
        (D3, donor_d3, "jev_v36_jevonly_r1", "_v36_jevonly_r1", OTHER, "d3-ultra alone"), (D3, donor_d3, "jev_v36_cooc_r1", "_v36_cooc_r1", OTHER, "d3-ultra + literature"))
for src, _, k, _, col, lab in ARMS:
    dd, lo, hi, v = wald("pancreas", k, "vs_M1", src)
    rows.append((f"pancreas, {lab}", col, dd, lo, hi, 100 * (src["pancreas"][k]["macro_acc"] - M["pancreas"]["M1_cellmarker_uniform"]["macro_acc"])))
for _, dn, _, dk, col, lab in ARMS:
    r = dn[dk]
    rows.append((f"heart, {lab}", col, 100 * r["micro_diff_vs_M2"], 100 * r["micro_diff_ci95"][0], 100 * r["micro_diff_ci95"][1], 100 * r["macro_diff_vs_M2"]))
ypos = [0, 1, 2, 3, 5, 6, 7, 8]
for y, (lab, col, p, lo, hi, mp) in zip(ypos, rows):
    b.errorbar(p, y, xerr=[[p - lo], [hi - p]], fmt="o", color=col, capsize=2.5, ms=4.2, lw=1.2, zorder=3, mfc=col if "literature" not in lab or col != OTHER else "white")
    b.scatter(mp, y, marker="D", s=13, fc="white", ec=col, lw=.9, zorder=4)
    b.text(40, y, f"{p:+.1f}  [{lo + 0.0:+.1f}, {hi + 0.0:+.1f}]".replace("-0.0", "0.0"), va="center", fontsize=5.5)
b.axvline(0, color=REF, lw=.9)
short_b = lambda s: s.split(", ", 1)[1].replace(" 1.13", "").replace(" alone", "").replace("literature", "lit.")  # tissue is named by the italic row headers
b.set_yticks(ypos, [short_b(r[0]) for r in rows], fontsize=5.6); b.set_ylim(8.6, -1.2); b.grid(axis="y", visible=False)
b.set_xlim(-20, 74)
b.set_xlabel("difference in accuracy (points)")
b.legend([Line2D([], [], marker="o", color="k", ms=4, lw=1.2), Line2D([], [], marker="D", ms=3.5, mfc="white", mec="k", ls="none")],
         ["micro, 95% interval", "macro, point only"], loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, fontsize=5.6)
b.text(72, -.85, "pancreas − CellMarker baseline; cell-level interval", fontsize=5.2, color=BASE, va="center", ha="right", style="italic")
b.text(72, 4.15, "heart − CellMarker baseline; donor-resampled", fontsize=5.2, color=BASE, va="center", ha="right", style="italic")
b.set_title("b  Difference from the baseline")

# ---- per-class recall, pancreas (data only; panel dropped) ----
z = np.load(X7 / "input/processed/pancreas_fullcells_lit_fixed.npz", allow_pickle=True)
gt = z["cell_type"].astype(str)[z["is_test"].astype(bool)]
types = ev["pancreas"]["types"]
sup = {t: int((gt == t).sum()) for t in types}
assert sum(sup.values()) == 2440 and len(types) == 23
order = sorted(types, key=lambda t: -sup[t])
short = lambda t: (t.replace("type B pancreatic cell", "pancreatic beta cell").replace("pancreatic ", "panc. ")
                   .replace("CD8-positive, alpha-beta T cell", "CD8 T cell").replace("CD4-positive, alpha-beta T cell", "CD4 T cell")
                   .replace("natural killer cell", "NK cell").replace("endothelial cell", "endothelial"))
pc = M["pancreas"]


def recall(t, key):
    if key.startswith("jev"):
        return float(np.mean([v["per_class_recall"][t] for v in pre(pc, key)]))
    return pc[key]["per_class_recall"][t]


D3P = D3["pancreas"]
PCR = {k: np.array([recall(t, k) for t in order]) for k in ("M1_cellmarker_uniform", "jev_v36_jevonly", "jev_v36_cooc")}
PCR["d3_jevonly"] = np.array([D3P["jev_v36_jevonly_r1"]["per_class_recall"][t] for t in order])
PCR["d3_cooc"] = np.array([D3P["jev_v36_cooc_r1"]["per_class_recall"][t] for t in order])

dump("cell", {
    "tool_versions": {"Jev": "1.13 (jev-1.13.0)", "database": "CellMarker 2.0", "second_model": "d3-ultra"},
    "acc_pct": {t: {lab: round(100 * ACC[t][k], 2) for k, lab, *_ in SC} | {lab: round(100 * D3ACC[t][k], 2) for k, lab, *_ in D3U} for t in TISS},
    "macro_pct": {t: {lab: round(100 * MAC[t][k], 2) for k, lab, *_ in SC} | {lab: round(100 * D3MAC[t][k], 2) for k, lab, *_ in D3U} for t in TISS},
    "majority_pct": {t: round(100 * MAJ[t], 2) for t in TISS},
    "pancreas_wald_vs_M1_r1_pp": {r[0]: [round(r[2], 2), round(r[3], 2), round(r[4], 2)] for r in rows[:4]},
    "heart_donor_ci_vs_M2_r1_pp": {r[0]: [round(r[2], 2), round(r[3], 2), round(r[4], 2), round(r[5], 2)] for r in rows[4:]},
    "heart_macro_ci_vs_M2_pp": {f"{a} r{r}": [round(100 * x, 1) for x in donor[f"_v36_{a}_r{r}"]["macro_diff_ci95"]] for a in ("jevonly", "cooc") for r in (1, 2, 3)},
    "heart_donor_ci_vs_M1n_posthoc_r1_pp": {a: [round(100 * donor_m1n[f"_v36_{a}_r1"]["micro_diff_vs_M1n"], 2)] + [round(100 * x, 2) for x in donor_m1n[f"_v36_{a}_r1"]["micro_diff_ci95"]] for a in ("jevonly", "cooc")}})
save(fig, "fig4_cell")
print("ok")