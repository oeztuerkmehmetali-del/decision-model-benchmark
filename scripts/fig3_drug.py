"""Fig 3, Results 'Drug-trial claims' (x8_drugcon): detection F1 of Jev 1.13 and d3-ultra beside the eleven published models.
In:  data/2_drug/{compare/metrics.json, compare/macros.tex, jev/detection__jev.tsv, baseline/paper_reference.csv (Lan et al. 2026, Table 3)} (inputs, answers and producing code: data/2_drug/README.md)
Out: figs/fig3_drug.{pdf,png}; numbers.json["drug"]
Stored results only. Run from merge_jev/paper/: python3 fig3_drug.py"""
import json, re
import pandas as pd
from style import J, save, dump, JEV, OTHER, REF, WIDTH, plt

X8 = J / "2_drug"
M = json.load(open(X8 / "compare/metrics.json"))
mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(X8 / "compare/macros.tex").read()))
ref = pd.read_csv(X8 / "baseline/paper_reference.csv", comment="#")
ref["det"] = ref[["cf_f1", "dc_f1"]].max(axis=1)  # best of the two published detection prompts
gold = pd.read_csv(X8 / "jev/detection__jev.tsv", sep="\t").gold_label
LAB = {"jev": "Jev 1.13", "d3u": "d3-ultra"}; COL = {"jev": JEV, "d3u": OTHER}
F1 = {m: M[f"detection_{m}"]["f1"] for m in LAB}

# headline assertions against the x8 paper macros
for m, t in (("jev", "Jev"), ("d3u", "Dthree")):
    assert f"{F1[m]['est']:.3f}" == mac[f"detF{t}"] and f"{F1[m]['ci'][0]:.3f}--{F1[m]['ci'][1]:.3f}" == mac[f"detFCI{t}"]
assert len(gold) == 1731 and int((gold == "Contradiction exists").sum()) == 542 == int(mac["dataDetectionPos"])
assert all(round(r.det, 3) == float(mac["refDet" + k]) for k, (_, r) in zip(
    "Ministral GemmaFour LlamaEight GemmaTwelve GemmaTwentySeven Qwen LlamaSeventy Haiku Sonnet GptFiveFive Nano".split(), ref.iterrows()))
assert len(ref) == 11 and ref.det.max() == 0.793  # Claude-Sonnet-4.5

# "detectably better" = lower end of the paired 95% interval above zero at the reported 3 decimals
# (d3-ultra vs GPT-4.1-nano: lower end +0.0002, reported +0.000, counted as not detectable)
better = {m: [r["model"] for r in M["vs_paper"] if round(r[f"{m}_det_diff"]["ci"][0], 3) > 0] for m in LAB}
weakest6 = list(ref.sort_values("det").model[:6])
assert sorted(better["jev"]) == sorted(weakest6) and sorted(better["d3u"]) == sorted(weakest6 + ["Qwen3-32B"])

# ================= figure =================
f, a = plt.subplots(figsize=(WIDTH * .55, 2.9))
pts = [(r.model, float(r.det), None, REF) for _, r in ref.iterrows()] + [(LAB[m], F1[m]["est"], F1[m]["ci"], COL[m]) for m in LAB]
pts.sort(key=lambda t: t[1])
for i, (nm, v, c, col) in enumerate(pts):
    if c:
        a.errorbar(v, i, xerr=[[v - c[0]], [c[1] - v]], fmt="o", ms=4.5, color=col, capsize=2, lw=1.2)
    else:
        a.plot(v, i, "o", ms=4, color=REF, mfc="white", label="published (Lan et al.)" if i == 0 else None)
a.set_yticks(range(len(pts)), [p[0] for p in pts], fontsize=6.3); a.grid(axis="y", visible=False)
for t_ in a.get_yticklabels():
    t_.set_fontweight("bold" if t_.get_text() in LAB.values() else "normal")
a.set(xlabel="detection F$_1$ (ours with 95% CI)")
for m in LAB:
    a.plot([], [], "o", color=COL[m], label=LAB[m])
a.set_xlim(.55, .89); a.legend(loc="lower right", fontsize=6.2, handletextpad=.2)
save(f, "fig3_drug")

dump("drug", {"pairs": "1,731", "contradictory": "542", "clusters": mac["dataDetectionClusters"],
              "published_models": "eleven", "best_published_f1": f"{ref.det.max():.3f}",
              "jev_f1": f"{F1['jev']['est']:.3f}", "d3u_f1": f"{F1['d3u']['est']:.3f}",
              "jev_detectably_better_than": sorted(better["jev"]), "d3u_detectably_better_than": sorted(better["d3u"])})
print("ok")
