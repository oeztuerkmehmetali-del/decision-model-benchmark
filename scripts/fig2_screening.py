"""Fig 2, Results 'Screening' (x3): Jev 1.13 and d3-ultra vs ASReview LAB 3.0.8 on 39 SYNERGY reviews, work saved at 95% recall.
In:  data/1_screening/{compare/results_v3_fixes.json, d3u/results_d3u.json, d3u/d3u_dev.json, d3u/d3u_test.json, d3u/dev_tau.d3u.json}, code data/1_screening/code/v3_fixes.py (+ layout.py, sweep.py, ablation.py, calib_sim.py; v3_fixes.load() also reads jev/, compare/, baseline/ and input/ files, listed in data/1_screening/SOURCES.tsv)
Out: figs/fig2_screening.{pdf,png}; numbers.json["screening"]
Stored results only. Run from merge_jev/paper/: python3 fig2_screening.py"""
import json, sys
import numpy as np
from matplotlib.patches import Patch
from style import J, save, dump, JEV, OTHER, REF, BASE, WIDTH, plt

X3 = J / "1_screening"
V = json.load(open(X3 / "compare/results_v3_fixes.json"))
arms = V["arms_per_review"]
names = list(arms)
col = lambda k: np.array([arms[n][k] for n in names], float)
je, ni, ul = col("judge"), col("n_inc_test"), col("asr_from_full")  # ASReview given the full review (equal footing)
d = je - ul
PG = V["paired_gaps_and_mde"]["synergy_judge_minus_asr_from_full"]
CI = PG["median_ci95_B10000_seed0"]
# d3-ultra arm (x3/d3u_arm.py; same split, prompts and threshold rule; added after the study closed)
U3 = json.load(open(X3 / "d3u/results_d3u.json"))
assert all(c["pass"] for c in U3["checks"]) and U3["model"] == "vllm-sr/d3-ultra"
UP = U3["paired"]

# d3-ultra per review, rebuilt by the x3 code from the cached answers: must equal the stored values
sys.path.insert(0, str(X3 / "code"))
import v3_fixes as X
from v3_fixes import jevlib, sweep
XD = X.load()
RD = X.build(dict(XD, jd=json.load(open(X3 / "d3u/d3u_dev.json")), jt=json.load(open(X3 / "d3u/d3u_test.json")),
                  dev_tau=json.load(open(X3 / "d3u/dev_tau.d3u.json"))["reviews"]))
ud3 = np.array([jevlib.wss_at_recall(RD[n]["p"], RD[n]["y"]) for n in names])
assert np.allclose(ud3, [U3["per_review"][n]["oracle"] for n in names], atol=1e-12)
assert all(sweep([(a["include"]["noul"], l) for _, a, l in RD[n]["dev"]])[0] == RD[n]["tau"] for n in names)


def boot_ci(x, B=10000, seed=0):
    r = np.random.default_rng(seed); n = len(x)
    return np.percentile([np.median(x[r.integers(0, n, n)]) for _ in range(B)], [2.5, 97.5])


# headline assertions against x3 manuscript_v3 / v3_numbers.md
assert len(je) == 39 and ((d > 0).sum(), (d < 0).sum()) == (20, 18)
assert abs(np.median(d) - PG["median"]) < 1e-9 and round(np.median(d), 3) == 0.004
assert np.allclose(boot_ci(d), CI, atol=.01) and (round(CI[0], 3), round(CI[1], 3)) == (-0.066, 0.059)
assert round(V["oracle_wss_median"], 3) == 0.743 and round(np.median(je), 3) == 0.743
assert round(np.median(ud3), 3) == round(U3["d3u"]["median_oracle_wss"], 3)

# ============================ figure ============================
f, ax = plt.subplots(1, 2, figsize=(WIDTH, 3.1))
a = ax[0]
a.plot([0, 1], [0, 1], ls="--", lw=.8, color=BASE)
for m, c, lab in ((d > 0, JEV, "Jev 1.13 ahead (20)"), (d <= 0, REF, "ASReview ahead or tied (19)")):
    a.scatter(ul[m], je[m], s=8 + 2.2 * ni[m], c=c, alpha=.7, ec="white", lw=.4, zorder=3)
    a.scatter([], [], s=14, c=c, label=lab)
a.scatter(ul, ud3, s=8 + 2.2 * ni, facecolors="none", edgecolors=OTHER, lw=.7, zorder=4)
a.scatter([], [], s=14, facecolors="none", edgecolors=OTHER, label="d3-ultra (open)")
a.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel="ASReview LAB 3.0.8, given the full review", ylabel="model (Jev 1.13, d3-ultra)",
      title="a  Work saved at 95% recall, 39 reviews")
a.set_aspect("equal"); a.legend(loc="lower right", fontsize=6, borderaxespad=.3)
a.text(.0, 1.0, "point size = relevant studies", fontsize=6, va="top", color=BASE)

a = ax[1]
o = np.argsort(d); x = np.arange(39)
a.bar(x, d[o], color=np.where(d[o] > 0, JEV, REF), width=.8)
a.axhline(0, color=REF, lw=.8)
a.axhline(np.median(d), color=BASE, ls="--", lw=.8)
dd3 = ud3 - ul
a.scatter(x, dd3[o], s=9, facecolors="white", edgecolors=OTHER, lw=.8, zorder=4)
a.text(.98, .38, f"Jev median {np.median(d):+.3f}, 95% CI {CI[0]:+.3f} to {CI[1]:+.3f}\nd3-ultra median {np.median(dd3):+.3f}",
       fontsize=6.0, va="top", ha="left")
hb = [Patch(color=JEV, label="Jev 1.13 better"), Patch(color=REF, label="ASReview LAB 3.0.8 better"),
      plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=OTHER, ms=3.5, label="d3-ultra − ASReview, same reviews")]
a.set(xlabel="reviews, sorted by Jev difference", ylabel="model − ASReview (work saved)", xticks=[], title="b  Paired difference per review")
a.legend(handles=hb, loc="lower right", fontsize=5.6, borderaxespad=.2, handlelength=1.2)
save(f, "fig2_screening")

dump("screening", {"reviews": f"{len(names)}", "test_records": f"{int(col('n_test').sum()):,}", "test_eligible": f"{int(ni.sum())}",
                   "jev_median": f"{np.median(je):.2f}", "asreview_median": f"{np.median(ul):.2f}",
                   "jev_minus_asreview": f"{np.median(d):+.3f}", "ci_lo": f"{CI[0]:+.3f}", "ci_hi": f"{CI[1]:+.3f}",
                   "d3u_median": f"{U3['d3u']['median_oracle_wss']:.2f}"})
print("ok")
