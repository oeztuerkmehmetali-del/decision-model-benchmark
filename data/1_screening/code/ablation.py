"""Clause-combination ablation (zero API spend): overall include noul vs AND (min clause noul) vs OR (max clause noul).

noul = Jev probability the answer is YES (record satisfies clause / should be included), so
AND = "pass every clause" = min_i, OR = "pass any clause" = max_i. Same dev-tau rule and
test metrics as score_final.py. Output: results_ablation39.json (+ summary on stdout).
"""
import json, os, statistics as st, sys

from layout import P  # noqa: E402  (study-relative paths -> case folders; adds ../shared to sys.path)
import jevlib  # noqa: E402
from sweep import sweep, jev_conf  # noqa: E402

RULES = {
    "overall": lambda a: a["include"]["noul"],
    "and_min": lambda a: min(v["noul"] for k, v in a.items() if k.startswith("clause_")),
    "or_max": lambda a: max(v["noul"] for k, v in a.items() if k.startswith("clause_")),
}


def pairs(d, rev, rule):
    return [(RULES[rule](r["answers"]), r["label"]) for r in d[rev].values()]


def score(test, tau):
    y = [l for _, l in test]
    ps = [p for p, _ in test]
    pred = [int(p >= tau) for p in ps]
    conf = [jev_conf(p) for p in ps]
    corr = [a == b for a, b in zip(pred, y)]
    return {"rec": round(sum(a and b for a, b in zip(pred, y)) / sum(y), 3),
            "wss": round(jevlib.wss_at_recall(ps, y), 3),
            "aurc": round(jevlib.aurc(conf, corr), 3),
            "auto": round(jevlib.auto_frac_at_loss(conf, corr, y), 3)}


def main():
    dev = json.load(open(P("runs/jev/jev_dev.json")))
    test = json.load(open(P("runs/jev/jev_test.json")))
    rows = []
    for rev in sorted(test):
        row = {"review": rev}
        for rule in RULES:
            tau = sweep(pairs(dev, rev, rule))[0]
            row[rule] = {"tau": tau, **score(pairs(test, rev, rule), tau)}
        rows.append(row)
    json.dump(rows, open(P("results/jev/results_ablation39.json"), "w"), indent=1)
    base = [r["overall"]["wss"] for r in rows]
    for rule in ("and_min", "or_max"):
        w = [r[rule]["wss"] for r in rows]
        p = jevlib.paired_bootstrap_p(st.median, base, w)
        print(f"{rule}: med wss {st.median(w):.3f} vs overall {st.median(base):.3f} | "
              f"rule wins {sum(a > b for a, b in zip(w, base))}/{len(rows)} | paired p {p:.4f} | "
              f"med aurc {st.median(r[rule]['aurc'] for r in rows):.3f} vs "
              f"{st.median(r['overall']['aurc'] for r in rows):.3f} | "
              f"rec<0.95 {sum(r[rule]['rec'] < 0.95 for r in rows)} vs "
              f"{sum(r['overall']['rec'] < 0.95 for r in rows)}")
    assert st.median(base) == 0.743, "overall arm must reproduce results_test.union39.json median 0.743"


if __name__ == "__main__":
    main()
