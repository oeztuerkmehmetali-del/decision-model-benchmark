"""x8 DrugCon-Bench: full offline analysis (no API calls). Rebuilds every number the paper uses for the drug-trial case.

Usage (from paper/data/2_drug/):  python3 code/analyze.py
Inputs : {jev,d3u}/{detection,explanation}__<arm>.tsv, {jev,d3u}/*_limit*_rep1.tsv (fresh repeat), {jev,d3u}/log_*.txt,
         compare/rep1_started_utc.txt, baseline/paper_reference.csv (Lan et al. 2026, Table 3).
Outputs: compare/metrics.json, compare/macros.tex (the original standalone figures fig_*.pdf were dropped from this copy).
Uncertainty: cluster bootstrap over connected components of the PMID graph (pairs that share an abstract are
dependent), B=2000, seed 0, same replicates for both models (paired). Paper values are fixed constants: their
own sampling error is not published, so differences to them carry only our model's uncertainty.
"""
import ast, json, os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # case folder (paper/data/2_drug)
sys.path[:0] = [os.path.join(HERE, "..", "shared"), os.path.join(HERE, "code")]  # jevlib; drugcon (authors' scorer)
import jevlib  # noqa: E402
from drugcon.evaluation import POSITIVE, CANONICAL_FACTORS, binary_prf, multilabel_metrics, normalize_factors  # noqa: E402

RUNS, RES = HERE, os.path.join(HERE, "compare")  # per-model answers sit in <HERE>/jev and <HERE>/d3u
os.makedirs(RES, exist_ok=True)
MODELS = {"jev": "Jev 1.13", "d3u": "d3-ultra"}
COL = {"jev": "#0072B2", "d3u": "#D55E00"}  # Okabe-Ito
B, SEED, ALPHA = 2000, 0, 0.05
OUT = {}  # everything that lands in metrics.json


def tsv(name):
    return pd.read_csv(os.path.join(RUNS, name.split("__")[1][:3], name), sep="\t")  # arm dir = jev|d3u


def ci(v):
    v = np.sort(np.asarray(v, float))
    return [float(v[int(ALPHA / 2 * len(v))]), float(v[int((1 - ALPHA / 2) * len(v)) - 1])]


# ---------- dependence structure: pairs sharing a PMID belong to one cluster ----------
def clusterer(pmid_pairs):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pmid_pairs:
        parent[find(a)] = find(b)
    return find


def boot_sets(cluster_ids):
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(cluster_ids == c) for c in np.unique(cluster_ids)]
    return [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(B)]


def prf(y, yhat):
    tp, fp, fn = int((y & yhat).sum()), int((~y & yhat).sum()), int((y & ~yhat).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ---------- detection ----------
det = {m: tsv(f"detection__{m}.tsv") for m in MODELS}
assert (det["jev"].pair_id.values == det["d3u"].pair_id.values).all()
find = clusterer(zip(det["jev"].pmid_1, det["jev"].pmid_2))
cl_det = np.array([find(a) for a in det["jev"].pmid_1])
y = (det["jev"].gold_label == POSITIVE).values
idx_det = boot_sets(cl_det)
OUT["data"] = {"detection_n": len(y), "detection_pos": int(y.sum()), "detection_neg": int((~y).sum()),
               "detection_pmids": int(len(set(det["jev"].pmid_1) | set(det["jev"].pmid_2))),
               "detection_clusters": int(len(np.unique(cl_det))),
               "largest_cluster_pairs": int(pd.Series(cl_det).value_counts().max())}


def exp_cum(conf, v):
    """Expected cumulative sum of v over pairs ranked by conf (descending), ties in random order.
    Jev rounds to 0.01 and 25% of its answers sit at 0/1, so any fixed tie-break would move AURC and coverage."""
    o = np.argsort(-conf, kind="stable")
    c, vv = conf[o], np.asarray(v, float)[o]
    _, start, cnt = np.unique(-c, return_index=True, return_counts=True)
    return np.cumsum(np.repeat(np.add.reduceat(vv, start) / cnt, cnt))


def aurc_tie(conf, ok):
    k = np.arange(1, len(ok) + 1)
    return float(((k - exp_cum(conf, ok)) / k).mean())


assert np.allclose(exp_cum(np.array([1.0, 1.0, 0.5]), [1, 0, 1]), [0.5, 1.0, 2.0])  # tied pair counts 0.5 each


def det_stats(m, ix):
    d = det[m]
    p, yy, pr = d.p_contra.values[ix], y[ix], (d.pred.values[ix] == POSITIVE)
    conf, ok = np.maximum(p, 1 - p), pr == yy
    P, Rc, F = prf(yy, pr)
    return {"precision": P, "recall": Rc, "f1": F, "accuracy": ok.mean(),
            "auroc": jevlib.auroc(p.tolist(), yy.astype(int).tolist()),
            "aurc": aurc_tie(conf, ok),
            "ece": jevlib.ece_binary(p.tolist(), yy.astype(int).tolist()),
            "brier": float(((p - yy) ** 2).mean())}


full = np.arange(len(y))
dstat = {m: det_stats(m, full) for m in MODELS}
dboot = {m: [det_stats(m, ix) for ix in idx_det] for m in MODELS}
for m in MODELS:  # parity with the authors' scorer
    a = binary_prf(det[m].gold_label, det[m].pred)
    assert abs(a["f1"] - dstat[m]["f1"]) < 1e-9 and abs(a["precision"] - dstat[m]["precision"]) < 1e-9
    OUT[f"detection_{m}"] = {k: {"est": float(v), "ci": ci([b[k] for b in dboot[m]])} for k, v in dstat[m].items()}
    OUT[f"detection_{m}"]["counts"] = {k: int(a[k]) for k in ("tp", "fp", "fn", "tn")}
    OUT[f"detection_{m}"]["pair_bootstrap_f1_ci_upstream"] = list(map(float, __import__("drugcon.evaluation", fromlist=["x"]).bootstrap_f1_ci(det[m].gold_label, det[m].pred)))
OUT["detection_diff_jev_minus_d3u"] = {k: {"est": dstat["jev"][k] - dstat["d3u"][k],
                                           "ci": ci([a[k] - b[k] for a, b in zip(dboot["jev"], dboot["d3u"])])}
                                       for k in ("f1", "precision", "recall", "accuracy", "auroc", "aurc", "ece")}
cj, cd = det["jev"].pred.values == det["jev"].gold_label.values, det["d3u"].pred.values == det["d3u"].gold_label.values
pj, pd_ = det["jev"].pred.values == POSITIVE, det["d3u"].pred.values == POSITIVE
po, pe = (pj == pd_).mean(), pj.mean() * pd_.mean() + (1 - pj.mean()) * (1 - pd_.mean())
OUT["detection_agreement"] = {"label_agreement": float(po), "cohen_kappa": float((po - pe) / (1 - pe)),
                              "jev_right_d3u_wrong": int((cj & ~cd).sum()), "jev_wrong_d3u_right": int((~cj & cd).sum()),
                              "mcnemar_exact_p_naive": jevlib.mcnemar_exact(int((cj & ~cd).sum()), int((~cj & cd).sum())),
                              "both_wrong": int((~cj & ~cd).sum())}
OUT["jev_p_contra_distinct_values"] = int(det["jev"].p_contra.nunique())
OUT["jev_p_contra_share_exact_0_or_1"] = float(det["jev"].p_contra.isin([0.0, 1.0]).mean())

# paper reference comparison (our estimate - fixed published value; CI from our bootstrap only)
ref = pd.read_csv(os.path.join(HERE, "baseline", "paper_reference.csv"), comment="#")
ref["det_best_f1"] = ref[["cf_f1", "dc_f1"]].max(axis=1)
OUT["paper_reference"] = ref.to_dict(orient="records")

# ---------- selective prediction (matched coverage) ----------
COVS = (0.25, 0.41, 0.5, 0.53, 0.75, 0.9, 1.0)


def cov_at(m, ix, c):
    p, ok = det[m].p_contra.values[ix], (det[m].pred.values == det[m].gold_label.values)[ix]
    conf, k = np.maximum(p, 1 - p), max(1, int(round(c * len(ix))))
    return exp_cum(conf, ok)[k - 1] / k, exp_cum(conf, y[ix])[k - 1] / y[ix].sum()


cov = {}
for m in MODELS:
    p, ok = det[m].p_contra.values, (det[m].pred.values == det[m].gold_label.values)
    conf = np.maximum(p, 1 - p)
    rows = []
    for c in COVS:
        acc, pos = cov_at(m, full, c)
        rows.append({"coverage": c, "n": int(round(c * len(y))), "accuracy": float(acc), "share_of_positives_kept": float(pos),
                     "ci": ci([cov_at(m, ix, c)[0] for ix in idx_det])})
    cov[m] = rows
    k = np.arange(1, len(ok) + 1)
    cov[m + "_curve"] = (k - exp_cum(conf, ok)) / k
OUT["matched_coverage"] = {m: cov[m] for m in MODELS}
OUT["matched_coverage_note"] = ("post hoc: 0.41 and 0.53 = coverages kept at conf>=0.99 by Jev and d3-ultra in the first report; "
                                "ties in confidence averaged (expected value under random order); CI = cluster bootstrap")

# ---------- explanation ----------
exp = {m: tsv(f"explanation__{m}.tsv") for m in MODELS}
assert (exp["jev"].id.values == exp["d3u"].id.values).all()
G = np.array([[f in normalize_factors(g) for f in CANONICAL_FACTORS] for g in exp["jev"].human_labeled_contextual_factors])
cl_exp = np.array([find(a) for a in exp["jev"].pmid_1])
idx_exp = boot_sets(cl_exp)
OUT["data"].update({"explanation_n": len(G), "explanation_clusters": int(len(np.unique(cl_exp))),
                    "gold_factors_per_pair": float(G.sum(1).mean()), "gold_label_count": int(G.sum())})


def P(m):
    return exp[m][[f"p_{f}" for f in CANONICAL_FACTORS]].values


def micro(Gs, Hs):
    tp, fp, fn = (Gs & Hs).sum(), (~Gs & Hs).sum(), (Gs & ~Hs).sum()
    p, r = tp / (tp + fp), tp / (tp + fn)
    return p, r, 2 * p * r / (p + r)


ebo = {}
for m in MODELS:
    H = P(m) >= 0.5
    t = multilabel_metrics(exp[m].human_labeled_contextual_factors, exp[m].pred).set_index("factor")
    mp, mr, mf = micro(G, H)
    assert abs(mf - t.loc["MICRO", "f1"]) < 1e-9  # parity with authors' scorer
    ebo[m] = [micro(G[ix], H[ix]) for ix in idx_exp]
    per = []
    for j, f in enumerate(CANONICAL_FACTORS):
        yy = G[:, j]
        per.append({"factor": f, "support": int(yy.sum()), "precision": float(t.loc[f, "precision"]), "recall": float(t.loc[f, "recall"]),
                    "f1": float(t.loc[f, "f1"]), "auroc": (jevlib.auroc(P(m)[:, j].tolist(), yy.astype(int).tolist()) if 0 < yy.sum() < len(yy) else None)})
    OUT[f"explanation_{m}"] = {"micro_precision": {"est": float(mp), "ci": ci([b[0] for b in ebo[m]])},
                               "micro_recall": {"est": float(mr), "ci": ci([b[1] for b in ebo[m]])},
                               "micro_f1": {"est": float(mf), "ci": ci([b[2] for b in ebo[m]])},
                               "macro_f1": float(t.loc["MACRO", "f1"]), "per_factor": per,
                               "pred_factors_per_pair": float(H.sum(1).mean()), "empty_predictions": int((H.sum(1) == 0).sum())}
OUT["explanation_diff_jev_minus_d3u"] = {"micro_f1": {"est": OUT["explanation_jev"]["micro_f1"]["est"] - OUT["explanation_d3u"]["micro_f1"]["est"],
                                                      "ci": ci([a[2] - b[2] for a, b in zip(ebo["jev"], ebo["d3u"])])}}

# differences to the paper's backbones
cmp = []
for _, r in ref.iterrows():
    row = {"model": r.model, "det_best_f1": r.det_best_f1, "expl_micro_f1": r.expl_micro_f1}
    for m in MODELS:
        row[f"{m}_det_diff"] = {"est": dstat[m]["f1"] - r.det_best_f1, "ci": ci([b["f1"] - r.det_best_f1 for b in dboot[m]])}
        row[f"{m}_expl_diff"] = {"est": OUT[f"explanation_{m}"]["micro_f1"]["est"] - r.expl_micro_f1,
                                 "ci": ci([b[2] - r.expl_micro_f1 for b in ebo[m]])}
    cmp.append(row)
OUT["vs_paper"] = cmp

# ---------- post hoc threshold sweeps (labelled exploratory) ----------
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
OUT["sweep_detection_f1"] = {m: [prf(y, det[m].p_contra.values >= t)[2] for t in GRID] for m in MODELS}
OUT["sweep_explanation_micro_f1"] = {m: [float(micro(G, P(m) >= t)[2]) for t in GRID] for m in MODELS}
OUT["sweep_grid"] = GRID.tolist()

# ---------- repeatability: fresh API answers on a seeded subset ----------
rep = {}
for m in MODELS:
    a = det[m].merge(tsv(f"detection__{m}_limit200_rep1.tsv"), on="pair_id", suffixes=("", "_r"))
    e0, e1 = exp[m].set_index("id"), tsv(f"explanation__{m}_limit50_rep1.tsv").set_index("id")
    e0 = e0.loc[e1.index]
    P0, P1 = (x[[f"p_{f}" for f in CANONICAL_FACTORS]].values for x in (e0, e1))
    Gs = G[[list(exp[m].id).index(i) for i in e1.index]]
    ya = (a.gold_label == POSITIVE).values
    rep[m] = {"det_n": len(a), "det_label_agreement": float((a.pred == a.pred_r).mean()),
              "det_abs_dp_median": float((a.p_contra - a.p_contra_r).abs().median()),
              "det_abs_dp_max": float((a.p_contra - a.p_contra_r).abs().max()),
              "det_f1_run1": prf(ya, (a.pred == POSITIVE).values)[2], "det_f1_run2": prf(ya, (a.pred_r == POSITIVE).values)[2],
              "exp_n": len(e1), "exp_decision_agreement": float(((P0 >= .5) == (P1 >= .5)).mean()),
              "exp_abs_dp_median": float(np.median(np.abs(P0 - P1))),
              "exp_micro_f1_run1": float(micro(Gs, P0 >= .5)[2]), "exp_micro_f1_run2": float(micro(Gs, P1 >= .5)[2])}
OUT["repeatability"] = rep
with open(os.path.join(RES, "rep1_started_utc.txt")) as fh:
    OUT["repeatability_started_utc"] = fh.read().strip()

# ---------- usage from run logs ----------
use = {}
for fn in sorted(f for arm in ("jev", "d3u") for f in os.listdir(os.path.join(RUNS, arm))):
    mt = re.match(r"log_(detection|explanation)__(jev|d3u)(_rep1)?\.txt$", fn)
    if mt:
        with open(os.path.join(RUNS, mt.group(2), fn)) as fh:
            line = [ln for ln in fh if ln.startswith("USAGE")][-1]
        use[fn[4:-4]] = ast.literal_eval(line.split(" ", 1)[1].split(" tokens->")[0])
OUT["usage"] = use
OUT["price_per_mtok_usd_jev"] = jevlib.PRICE_PER_MTOK

with open(os.path.join(RES, "metrics.json"), "w") as fh:
    json.dump(OUT, fh, indent=1, default=float)

# ---------- LaTeX: macros + tables ----------
f3 = "{:.3f}".format
mac = []


def M(name, val):
    mac.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def cis(d):
    return f"{f3(d['ci'][0])}--{f3(d['ci'][1])}"


COVNAME = {0.25: "TwentyFive", 0.41: "FortyOne", 0.5: "Fifty", 0.53: "FiftyThree", 0.75: "SeventyFive", 0.9: "Ninety", 1.0: "Full"}
REFTAG = {"Ministral-3B": "Ministral", "Gemma-3-4B": "GemmaFour", "Llama-3.1-8B": "LlamaEight", "Gemma-3-12B": "GemmaTwelve",
          "Gemma-3-27B": "GemmaTwentySeven", "Qwen3-32B": "Qwen", "Llama-3.3-70B": "LlamaSeventy", "Claude-Haiku-4.5": "Haiku",
          "Claude-Sonnet-4.5": "Sonnet", "GPT-5.5": "GptFiveFive", "GPT-4.1-nano": "Nano"}
SHORT = {"dosage or exposure duration": "dosage/duration", "route or mode of administration": "route",
         "combined drug effects": "combined drugs", "evolving scientific evidence": "evolving evidence",
         "known controversy or self-qualified claims": "known controversy", "ambiguous referent": "ambig. referent"}


for m, tag in (("jev", "Jev"), ("d3u", "Dthree")):
    D, E = OUT[f"detection_{m}"], OUT[f"explanation_{m}"]
    for k, nm in (("f1", "F"), ("precision", "P"), ("recall", "R"), ("accuracy", "Acc"), ("auroc", "Auroc"), ("aurc", "Aurc"), ("ece", "Ece"), ("brier", "Brier")):
        M(f"det{nm}{tag}", f3(D[k]["est"])); M(f"det{nm}CI{tag}", cis(D[k]))
    M(f"detPairCI{tag}", f"{f3(D['pair_bootstrap_f1_ci_upstream'][0])}--{f3(D['pair_bootstrap_f1_ci_upstream'][1])}")
    for k, nm in (("micro_f1", "F"), ("micro_precision", "P"), ("micro_recall", "R")):
        M(f"exp{nm}{tag}", f3(E[k]["est"])); M(f"exp{nm}CI{tag}", cis(E[k]))
    M(f"expMacro{tag}", f3(E["macro_f1"])); M(f"expPerPair{tag}", f"{E['pred_factors_per_pair']:.2f}"); M(f"expEmpty{tag}", E["empty_predictions"])
    for c in cov[m]:
        nm = COVNAME[c["coverage"]]
        M(f"cov{nm}{tag}", f"{100*c['accuracy']:.1f}")
        M(f"cov{nm}CI{tag}", f"{100*c['ci'][0]:.1f}--{100*c['ci'][1]:.1f}")
        M(f"cov{nm}Pos{tag}", f"{100*c['share_of_positives_kept']:.0f}")
    R_ = rep[m]
    M(f"repAgree{tag}", f"{100*R_['det_label_agreement']:.1f}"); M(f"repExpAgree{tag}", f"{100*R_['exp_decision_agreement']:.1f}")
    M(f"repDp{tag}", f"{R_['det_abs_dp_median']:.3f}"); M(f"repFOne{tag}", f3(R_["det_f1_run1"])); M(f"repFTwo{tag}", f3(R_["det_f1_run2"]))
    M(f"repEFOne{tag}", f3(R_["exp_micro_f1_run1"])); M(f"repEFTwo{tag}", f3(R_["exp_micro_f1_run2"]))
    M(f"sweepDetMax{tag}", f3(max(OUT["sweep_detection_f1"][m]))); M(f"sweepExpMax{tag}", f3(max(OUT["sweep_explanation_micro_f1"][m])))
    M(f"sweepDetArg{tag}", f"{GRID[int(np.argmax(OUT['sweep_detection_f1'][m]))]:.2f}")
    M(f"sweepExpArg{tag}", f"{GRID[int(np.argmax(OUT['sweep_explanation_micro_f1'][m]))]:.2f}")
    M(f"repDpMax{tag}", f"{R_['det_abs_dp_max']:.2f}")
for k, nm in (("f1", "F"), ("auroc", "Auroc"), ("aurc", "Aurc"), ("accuracy", "Acc"), ("ece", "Ece"), ("precision", "P"), ("recall", "R")):
    d = OUT["detection_diff_jev_minus_d3u"][k]
    M(f"diffDet{nm}", f"{d['est']:+.3f}"); M(f"diffDet{nm}CI", f"{d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}")
d = OUT["explanation_diff_jev_minus_d3u"]["micro_f1"]; M("diffExpF", f"{d['est']:+.3f}"); M("diffExpFCI", f"{d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}")
for r in cmp:
    tag = REFTAG[r["model"]]
    for m, mt in (("jev", "Jev"), ("d3u", "Dthree")):
        for k, nm in (("det", "Det"), ("expl", "Exp")):
            dd = r[f"{m}_{k}_diff"]
            M(f"vs{nm}{mt}{tag}", f"{dd['est']:+.3f}"); M(f"vs{nm}CI{mt}{tag}", f"{dd['ci'][0]:+.3f} to {dd['ci'][1]:+.3f}")
A = OUT["detection_agreement"]
M("kappaLabels", f"{A['cohen_kappa']:.2f}"); M("labelAgree", f"{100*A['label_agreement']:.1f}"); M("bothWrong", A["both_wrong"])
M("jevRightOnly", A["jev_right_d3u_wrong"]); M("dthreeRightOnly", A["jev_wrong_d3u_right"])
for k, v in OUT["data"].items():
    M("data" + "".join(w.capitalize() for w in k.split("_")), f"{v:.2f}" if isinstance(v, float) else v)
M("jevDistinctP", OUT["jev_p_contra_distinct_values"]); M("jevShareExtreme", f"{100*OUT['jev_p_contra_share_exact_0_or_1']:.1f}")
tok = {k: v["input_tokens"] for k, v in use.items()}
M("tokJev", f"{(tok['detection__jev'] + tok['explanation__jev'])/1e6:.2f}"); M("tokDthree", f"{(tok['detection__d3u'] + tok['explanation__d3u'])/1e6:.2f}")
M("costJev", f"{(tok['detection__jev'] + tok['explanation__jev'])/1e6*jevlib.PRICE_PER_MTOK:.3f}")
M("retriesDthree", use["detection__d3u"]["retries"] + use["explanation__d3u"]["retries"])
M("bootB", B)
for _, r in ref.iterrows():  # published values, so the text never retypes them
    M(f"refDet{REFTAG[r.model]}", f3(r.det_best_f1)); M(f"refExp{REFTAG[r.model]}", f3(r.expl_micro_f1))
M("covFortyOneReason", "Jev's own confidence $\\ge$ 0.99 level")
fj = {x["factor"]: x["f1"] for x in OUT["explanation_jev"]["per_factor"]}
txt3 = [fj[f] for f in ("population", "dosage or exposure duration", "outcome measures")]
M("factorTextMin", f"{min(txt3):.2f}"); M("factorTextMax", f"{max(txt3):.2f}")
M("factorKnownJev", f"{fj['known controversy or self-qualified claims']:.2f}")
M("factorEvolvingJev", f"{fj['evolving scientific evidence']:.2f}")
M("factorRouteJev", f"{fj['route or mode of administration']:.2f}"); M("factorDesignJev", f"{fj['study design']:.2f}")
# benchmark repository (upstream README.md table; data/README.md Construction steps 4-5)
M("refDetNanoReadme", "0.739"); M("goldNanoSeeded", "494"); M("goldExhaustive", "48")
fd = {x["factor"]: x["f1"] for x in OUT["explanation_d3u"]["per_factor"]}
M("ambigJev", f"{fj['ambiguous referent']:.2f}"); M("ambigDthree", f"{fd['ambiguous referent']:.2f}")
# macro-F1 without "species" (2 gold pairs): that one factor alone decides the ten-factor order
M("expMacroNoSpeciesJev", f3(np.mean([v for k, v in fj.items() if k != "species"])))
M("expMacroNoSpeciesDthree", f3(np.mean([v for k, v in fd.items() if k != "species"])))
M("speciesJev", f"{fj['species']:.2f}"); M("speciesDthree", f"{fd['species']:.2f}")
with open(os.path.join(RES, "macros.tex"), "w") as fh:
    fh.write("% generated by analyze.py - do not edit\n" + "\n".join(mac) + "\n")

print(json.dumps({k: OUT[k] for k in ("data", "detection_diff_jev_minus_d3u", "explanation_diff_jev_minus_d3u", "repeatability")}, indent=1, default=float))
print("wrote", RES)
