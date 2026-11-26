#!/usr/bin/env python3
"""d3-ultra screening arm: same records, prompts, split, recall>=0.95 tau rule and WSS formulas as the Jev arm.

Reuses v3_fixes (load, build, arms, f7, frozen, frozen_paired) unchanged. Gate: the Jev arm is recomputed first
and every published Jev check must PASS, and the generic per-arm function below must reproduce the published
Jev frozen numbers, before any d3-ultra number is written.

Inputs (cached, no API calls): runs/d3u/d3u_dev.json, runs/d3u/d3u_test.json, runs/d3u/dev_tau.d3u.json.
Writes results/results_d3u.json and results/d3u_numbers.md. Run from x3/: python3 scripts/d3u_arm.py
Disclosure: 1,792 test records were queried (cached, never scored) before the dev tau was frozen; see d3u_run.sh.
"""
import json, os, statistics as st, sys

import layout                             # noqa: E402  (study-relative paths -> case folders; adds ../shared to sys.path)
import v3_fixes as V                      # noqa: E402
from v3_fixes import jevlib, TARGET       # noqa: E402

MODEL = "vllm-sr/d3-ultra"
PRE_TAU_TEST_QUERIES = 1792  # leading test-split block of runs/d3u/d3u_cache.jsonl, recounted via jevlib._key


def arm(R, revs, A):
    """Per-arm WSS: oracle stop (jevlib.wss_at_recall) and frozen dev tau (achieved and nominal conventions)."""
    f7r = V.f7(R, revs)
    rec = {x["review"]: x["recall"] for x in f7r["rows"]}
    excl = {r: sum(1 for p in R[r]["p"] if p < R[r]["tau"]) / R[r]["n_test"] for r in revs}
    return {"f7": f7r,
            "oracle": {r: jevlib.wss_at_recall(R[r]["p"], R[r]["y"]) for r in revs},
            "achieved": {r: excl[r] - (1 - rec[r]) for r in revs},
            "nominal": {r: excl[r] - (1 - TARGET) for r in revs},
            "reads_frozen": {r: sum(1 for p in R[r]["p"] if p >= R[r]["tau"]) for r in revs},
            "caught": sum(x["n_caught"] for x in f7r["rows"]),
            "eligible": sum(x["n_eligible"] for x in f7r["rows"])}


def pair(a, b, revs):
    d = [round(a[r] - b[r], 12) for r in revs]
    lo, hi = jevlib.bootstrap_ci(st.median, d, n=10000, seed=0)
    return {"median": st.median(d), "ci95_B10000_seed0": [lo, hi],
            "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d), "ties": sum(x == 0 for x in d)}


def main():
    D = V.load()
    revs = D["revs"]
    Rj = V.build(D)
    A = V.arms(D, Rj)
    f7j = V.f7(Rj, revs)
    V.frozen(Rj, revs, A, f7j)
    V.frozen_paired(Rj, revs, A, f7j)
    J = arm(Rj, revs, A)
    asr_full = {r: A[r]["asr_from_full"] for r in revs}
    asr_test = {r: A[r]["asr_test"] for r in revs}
    V.check("generic arm: Jev frozen achieved - ASReview full-pool (published -0.057)",
            pair(J["achieved"], asr_full, revs)["median"], -0.057)
    V.check("generic arm: Jev oracle - ASReview test-only (published +0.156)",
            pair(J["oracle"], asr_test, revs)["median"], 0.156)
    bad = [c["check"] for c in V.CHECKS if not c["pass"]]
    assert not bad, f"Jev reproduction gate failed: {bad}"

    tau = json.load(open(V.path("runs", "d3u", "dev_tau.d3u.json")))
    assert tau["model"] == MODEL, tau["model"]
    D2 = dict(D, jd=json.load(open(V.path("runs", "d3u", "d3u_dev.json"))),
              jt=json.load(open(V.path("runs", "d3u", "d3u_test.json"))), dev_tau=tau["reviews"])
    Rd = V.build(D2)
    none = sum(p is None for r in revs for p in Rd[r]["p"])
    assert none == 0, f"{none} test records without an include probability"
    U = arm(Rd, revs, A)

    def summ(X):
        return {"median_oracle_wss": st.median(X["oracle"].values()),
                "median_frozen_wss_achieved": st.median(X["achieved"].values()),
                "median_frozen_wss_nominal": st.median(X["nominal"].values()),
                "reviews_reaching_095": X["f7"]["n_reach_095"], "reviews_below_095": X["f7"]["n_below_095"],
                "relevant_caught": X["caught"], "relevant_total": X["eligible"],
                "reads_frozen_total": sum(X["reads_frozen"].values())}

    out = {"model": MODEL, "tau_model": tau["model"], "pre_tau_test_queries_disclosed": PRE_TAU_TEST_QUERIES,
           "n_reviews": len(revs), "n_test": sum(Rd[r]["n_test"] for r in revs),
           "jev": summ(J), "d3u": summ(U),
           "median_asr_test_only": st.median(asr_test.values()), "median_asr_full_pool": st.median(asr_full.values()),
           "paired": {
               "d3u_oracle_minus_asr_test_only": pair(U["oracle"], asr_test, revs),
               "d3u_oracle_minus_asr_full_pool": pair(U["oracle"], asr_full, revs),
               "d3u_frozen_achieved_minus_asr_full_pool": pair(U["achieved"], asr_full, revs),
               "d3u_frozen_nominal_minus_asr_full_pool": pair(U["nominal"], asr_full, revs),
               "d3u_frozen_achieved_minus_asr_test_only": pair(U["achieved"], asr_test, revs),
               "d3u_oracle_minus_jev_oracle": pair(U["oracle"], J["oracle"], revs),
               "d3u_frozen_achieved_minus_jev_frozen_achieved": pair(U["achieved"], J["achieved"], revs)},
           "per_review": {r: {"tau": Rd[r]["tau"], "oracle": U["oracle"][r], "frozen_achieved": U["achieved"][r],
                              "recall": next(x["recall"] for x in U["f7"]["rows"] if x["review"] == r)}
                          for r in revs},
           "checks": V.CHECKS}
    json.dump(out, open(V.path("results", "results_d3u.json"), "w"), indent=1)

    L = ["# d3-ultra screening arm (generated by d3u_arm.py; do not edit)", "",
         f"Model {MODEL}; tau frozen on dev (recall >= 0.95 rule, sweep.py); {out['n_reviews']} reviews, "
         f"{out['n_test']} test records. {PRE_TAU_TEST_QUERIES} test records were queried before the tau was "
         "frozen (cached, not scored then).", "",
         "| metric | Jev 1.13 | d3-ultra |", "|---|---|---|"]
    for k in summ(J):
        L.append(f"| {k} | {out['jev'][k]:.3f} | {out['d3u'][k]:.3f} |" if isinstance(out["jev"][k], float)
                 else f"| {k} | {out['jev'][k]} | {out['d3u'][k]} |")
    L += ["", "| paired median (B=10000, seed 0) | median | 95% CI | wins/losses/ties |", "|---|---|---|---|"]
    for k, v in out["paired"].items():
        lo, hi = v["ci95_B10000_seed0"]
        L.append(f"| {k} | {v['median']:+.3f} | {lo:+.3f} to {hi:+.3f} | {v['wins']}/{v['losses']}/{v['ties']} |")
    open(V.path("results", "d3u_numbers.md"), "w").write("\n".join(L) + "\n")
    print("ok", json.dumps({k: out[k] for k in ("jev", "d3u")}, indent=1))
    for k, v in out["paired"].items():
        print(k, round(v["median"], 3), [round(x, 3) for x in v["ci95_B10000_seed0"]], v["wins"], v["losses"])


if __name__ == "__main__":
    main()
