#!/usr/bin/env python3
"""X3 v3 numbers: F7 Clopper-Pearson recall, F8 calibration pass rates, F9 absolute reads per arm,
F10 tie sensitivity and 27/29/50 reconciliation, F12 AURC under two confidences, F13 clause ablation
on split test ids, F19 hierarchical recall bootstrap, and the MDE for paired gaps.

Cached inputs only: no API calls, no ASReview re-runs. Reused, not edited: pilots/jevlib.py
(wss_at_recall, aurc, paired_bootstrap_p, bootstrap_ci, hier_boot_ci), sweep.py (sweep, jev_conf),
ablation.py (RULES, score), calib_sim.py (clopper_pearson).

Writes results/results_v3_fixes.json and results/v3_numbers.md. Run from x3/: python3 scripts/v3_fixes.py
Every stored-value reproduction prints PASS/FAIL; FAIL lines are discrepancies to report, not to hide.
"""
import json, math, os, re, statistics as st, sys

import numpy as np

from layout import P, load_questions      # noqa: E402  (study-relative paths -> case folders; adds ../shared to sys.path)
import jevlib                             # noqa: E402
import ablation                           # noqa: E402
from sweep import sweep, jev_conf         # noqa: E402
from calib_sim import clopper_pearson     # noqa: E402

TARGET = 0.95
TOL = 0.0005                               # stored results are rounded to 3 dp
Z = 1.959963984540054 + 0.8416212335729143  # two-sided alpha 0.05, power 0.80
OUT_JSON = P("results/results_v3_fixes.json")
OUT_MD = P("results/v3_numbers.md")
CHECKS = []


def path(*parts):  # study-relative path parts -> case-folder path (see layout.py)
    return P("/".join(parts))


def check(name, got, want, tol=TOL):
    ok = abs(got - want) <= tol
    CHECKS.append({"check": name, "computed": float(got), "published": float(want), "tol": tol, "pass": ok})
    print(f"{'PASS' if ok else 'FAIL'} {name}: computed {got:.6g} vs stored/published {want:.6g}")


def load():
    stored = {r["review"]: r for r in json.load(open(path("results", "jev", "results_test.union39.json")))}
    revs = sorted(stored)
    assert len(revs) == 39, len(revs)
    return {
        "revs": revs,
        "stored": stored,
        "split": json.load(open(path("data", "processed", "split.union.json")))["reviews"],
        "jt": json.load(open(path("runs", "jev", "jev_test.json"))),
        "jd": json.load(open(path("runs", "jev", "jev_dev.json"))),
        "dev_tau": json.load(open(path("runs", "jev", "dev_tau.union.json")))["reviews"],
        "calib": json.load(open(path("results", "jev", "results_calibsim39.json"))),
        "gaps": json.load(open(path("results", "jev", "results_gaps39.json"))),
        "chan": json.load(open(path("results", "results_chan_compare.json")))["per_review"],
        "net": json.load(open(path("results", "results_net_effort.json"))),
        "abl_old": {r["review"]: r for r in json.load(open(path("results", "jev", "results_ablation39.json")))},
        "asr": {r: json.load(open(path("runs", "asreview", f"{r.lower()}_elas_u4.json"))) for r in revs},
    }


def partition(D, src, rev, part):
    """One split partition of one review: [(record_id, answers, label)] in cache order.
    Asserts every split id is present in the cache (no silent drops)."""
    ids = D["split"][rev]["ids"][part]
    want = set(ids)
    recs = [(rid, r["answers"], r["label"]) for rid, r in src[rev].items() if rid in want]
    assert len(recs) == len(ids) == D["split"][rev]["n"][part], (rev, part, len(recs), len(ids))
    return recs


def build(D):
    R = {}
    for rev in D["revs"]:
        sp = D["split"][rev]
        assert sum(sp["n"].values()) == sp["n_total"], rev
        dev = partition(D, D["jd"], rev, "dev")
        test = partition(D, D["jt"], rev, "test")
        R[rev] = {"dev": dev, "test": test,
                  "ids": [rid for rid, _, _ in test],
                  "y": [l for _, _, l in test],
                  "p": [a["include"]["noul"] for _, a, _ in test],
                  "tau": D["dev_tau"][rev]["tau"],
                  "n_dev": len(dev), "n_test": len(test),
                  "n_holdout": sp["n"]["holdout"], "n_total": sp["n_total"]}
    return R


def reads_to_level(labels, n_inc):
    """Records read in the given order until found/n_inc >= TARGET (jevlib.wss_at_recall stop rule)."""
    found = 0
    for k, y in enumerate(labels, 1):
        found += y
        if found / n_inc >= TARGET:
            return k
    raise RuntimeError("target not reached")


def wss_from_reads(n, k):
    return (n - k) / n - (1 - TARGET)


def reads_from_wss(n, wss):
    k = n * (TARGET - wss)
    assert abs(k - round(k)) < 1e-6, (n, wss)
    return int(round(k))


def ci_sorted(vals):
    """jevlib convention: vals[int(.025 n)], vals[int(.975 n) - 1]."""
    n = len(vals)
    return [float(vals[int(0.025 * n)]), float(vals[int(0.975 * n) - 1])]



# ----------------------------------------------------------------------------- arms (F9 + WSS base)
def arms(D, R):
    """Per review: oracle-stop WSS and reads for the judge and the three ASReview arms."""
    out = {}
    for rev in D["revs"]:
        r, a = R[rev], D["asr"][rev]
        lab = dict(zip(r["ids"], r["y"]))
        n_t, n_inc = r["n_test"], sum(r["y"])
        judge = jevlib.wss_at_recall(r["p"], r["y"])
        at_labels = [lab[i] for i in a["test_order"]]
        assert len(at_labels) == n_t and sum(at_labels) == n_inc, rev
        k_at = reads_to_level(at_labels, n_inc)
        n_full = len(a["full"]["order"])
        assert n_full == r["n_total"], rev
        full_labels = a["full"]["order_labels"]
        k_full = reads_to_level(full_labels, sum(full_labels))
        found = k_sub = pos = 0
        for pos, rid in enumerate(a["full"]["order"], 1):
            if rid in lab:
                k_sub += 1
                found += lab[rid]
                if found / n_inc >= TARGET:
                    break
        out[rev] = {"n_dev": r["n_dev"], "n_test": n_t, "n_full": n_full, "n_inc_test": n_inc,
                    "judge": judge, "k_judge_test": reads_from_wss(n_t, judge),
                    "k_frozen_test": sum(1 for p in r["p"] if p >= r["tau"]),
                    "asr_test": wss_from_reads(n_t, k_at), "k_asr_test": k_at,
                    "asr_full_pool": wss_from_reads(n_full, k_full), "k_asr_full_pool": k_full,
                    "asr_from_full": wss_from_reads(n_t, k_sub), "k_asr_from_full_test": k_sub,
                    "pos_full_reads_for_test": pos,
                    "ref_asr_test": a["test"]["wss95"], "ref_asr_from_full": a["test_from_full"]["wss95"],
                    "ref_asr_full": a["full"]["wss95"], "ref_stored_asr_u4": D["stored"][rev]["asr_u4"],
                    "ref_stored_wss": D["stored"][rev]["wss95"]}
    check("arms: max |ASReview test-only WSS - stored test.wss95|",
          max(abs(v["asr_test"] - v["ref_asr_test"]) for v in out.values()), 0.0, 1e-9)
    check("arms: max |ASReview from-full test WSS - stored test_from_full.wss95|",
          max(abs(v["asr_from_full"] - v["ref_asr_from_full"]) for v in out.values()), 0.0, 1e-9)
    check("arms: max |ASReview full-pool WSS - stored full.wss95|",
          max(abs(v["asr_full_pool"] - v["ref_asr_full"]) for v in out.values()), 0.0, 1e-9)
    check("arms: max |judge WSS - stored results_test wss95| (3-dp rounding)",
          max(abs(v["judge"] - v["ref_stored_wss"]) for v in out.values()), 0.0, TOL)
    check("arms: max |ASReview test-only WSS - stored asr_u4| (3-dp rounding)",
          max(abs(v["asr_test"] - v["ref_stored_asr_u4"]) for v in out.values()), 0.0, TOL)
    return out


# ----------------------------------------------------------------------------- F7
def f7(R, revs):
    rows = []
    for rev in revs:
        r = R[rev]
        n1 = sum(r["y"])
        tp = sum(1 for p, y in zip(r["p"], r["y"]) if y and p >= r["tau"])
        lo, hi = clopper_pearson(tp, n1)
        rows.append({"review": rev, "tau": r["tau"], "n_eligible": n1, "n_caught": tp,
                     "recall": tp / n1, "cp95": [lo, hi]})
    below = [x for x in rows if x["recall"] < TARGET]
    return {
        "rows": rows,
        "n_reach_095": len(rows) - len(below),
        "n_below_095": len(below),
        "n_below_with_cp_upper_below_095": sum(x["cp95"][1] < TARGET for x in below),
        "n_reach_with_cp_lower_below_095": sum(x["recall"] >= TARGET and x["cp95"][0] < TARGET for x in rows),
        "median_cp_halfwidth": st.median((x["cp95"][1] - x["cp95"][0]) / 2 for x in rows),
        "worst_review": min(rows, key=lambda x: x["recall"])["review"],
        "worst_recall": min(x["recall"] for x in rows),
    }


# ----------------------------------------------------------------------------- F8
def f8(D):
    """SYNERGY calibration sim (results_calibsim39.json, rule naive_r0.95 = the frozen max-spec rule).
    Stored pass rates are 4-dp proportions over EVALUATED draws; counts are reconstructed and checked.
    unconditional = passes / ALL draws (no-eligible and unattainable draws count as failures)."""
    c = D["calib"]
    B = c["meta"]["B"]
    res = c["results"]["naive_r0.95"]
    out = {"B_draws_per_review": B, "n_reviews": c["sanity_check"]["n_reviews"],
           "sanity_dev_tau_reproduced": f'{c["sanity_check"]["n_match"]}/{c["sanity_check"]["n_reviews"]}'}
    for k in c["meta"]["k_values"]:
        a = res[str(k)]["all"]
        n_eval, n_noinc, n_unatt = a["n_evaluated"], a["n_no_include_draw"], a["n_threshold_unattainable"]
        n_total = n_eval + n_noinc + n_unatt
        assert n_total == a["n_reviews"] * B, k
        n_pass = round(a["recall_p_ge_095"] * n_eval)
        assert abs(n_pass / n_eval - a["recall_p_ge_095"]) < 5e-5, k
        assert abs((n_noinc + n_unatt) / n_total - a["share_unattainable"]) < 1e-4, k
        out[str(k)] = {"n_draws_total": n_total, "n_draws_evaluated": n_eval,
                       "n_draws_no_eligible": n_noinc, "n_draws_unattainable": n_unatt,
                       "n_pass": n_pass,
                       "pass_conditional": n_pass / n_eval,
                       "pass_unconditional": n_pass / n_total,
                       "share_no_eligible": n_noinc / n_total,
                       "tau_median": a["tau_median"], "spec_median": a["spec_median"],
                       "recall_median": a["recall_median"], "stored_pass_ci95": a["recall_p_ge_095_cp95_ci"]}
    return out


# ----------------------------------------------------------------------------- F9
def f9(A, revs):
    def med(key):
        return st.median(A[r][key] for r in revs)
    s = {
        "median_n_dev": med("n_dev"), "median_n_test": med("n_test"), "median_n_full": med("n_full"),
        "median_judge_test_reads": med("k_judge_test"),
        "median_frozen_test_reads": med("k_frozen_test"),
        "share_frozen_test": st.median(A[r]["k_frozen_test"] / A[r]["n_test"] for r in revs),
        "median_asr_test_only_test_reads": med("k_asr_test"),
        "median_asr_full_pool_reads": med("k_asr_full_pool"),
        "median_asr_from_full_test_reads": med("k_asr_from_full_test"),
        "median_asr_from_full_pool_reads": med("pos_full_reads_for_test"),
        "share_judge_test": st.median(A[r]["k_judge_test"] / A[r]["n_test"] for r in revs),
        "share_asr_test_only": st.median(A[r]["k_asr_test"] / A[r]["n_test"] for r in revs),
        "share_asr_full_pool": st.median(A[r]["k_asr_full_pool"] / A[r]["n_full"] for r in revs),
        "share_asr_from_full_pool": st.median(A[r]["pos_full_reads_for_test"] / A[r]["n_full"] for r in revs),
    }
    s["median_frozen_total_reads"] = s["median_n_dev"] + s["median_frozen_test_reads"]
    s["median_judge_total_reads"] = s["median_n_dev"] + s["median_judge_test_reads"]
    N = 2000
    nd, nt = int(0.6 * N), int(0.2 * N)
    worked = {"N_total": N, "n_dev": nd, "n_test": nt, "n_holdout": N - nd - nt,
              "judge_dev_labels": nd,
              "judge_test_reads": round(s["share_judge_test"] * nt),
              "judge_total_reads": nd + round(s["share_judge_test"] * nt),
              "asr_test_only_reads": round(s["share_asr_test_only"] * nt),
              "asr_test_only_dev_reads": 0,
              "asr_full_pool_reads": round(s["share_asr_full_pool"] * N),
              "asr_from_full_pool_reads_for_test": round(s["share_asr_from_full_pool"] * N),
              "method": "median per-review read share applied to a 60/20/20 split of 2,000 records (illustrative)"}
    # net-effort read shares exactly as net_effort.py defines them: both arms charged n_dev
    j_share = {r: (A[r]["n_dev"] + A[r]["k_judge_test"]) / (A[r]["n_dev"] + A[r]["n_test"]) for r in revs}
    a_share = {r: (A[r]["n_dev"] + A[r]["k_asr_test"]) / (A[r]["n_dev"] + A[r]["n_test"]) for r in revs}
    return {"summary": s, "worked_2000": worked,
            "net_judge_share": j_share, "net_asr_share": a_share}


def net_block(D, F9, A):
    """Net effort as results_net_effort.json computes it: read share (n_dev + reads) / (n_dev + n_test), both arms
    charged n_dev. That file uses the 3-dp-rounded stored WSS for the judge; full precision differs by ~1e-4."""
    ref, revs = D["net"], D["revs"]
    stored_j = {r: (A[r]["n_dev"] + A[r]["n_test"] * (TARGET - D["stored"][r]["wss95"])) / (A[r]["n_dev"] + A[r]["n_test"])
                for r in revs}
    check("net judge median read share from stored 3-dp WSS = results_net_effort",
          st.median(stored_j.values()), ref["wss95"]["median_read_share"], 1e-9)
    js, as_ = F9["net_judge_share"], F9["net_asr_share"]
    med_j = st.median(js.values())
    med_a = st.median(as_.values())
    check("net judge median net WSS, full precision vs results_net_effort (rounding only)",
          TARGET - med_j, ref["wss95"]["median_net_wss95"], 1e-4)
    check("net ASReview u4 median read share = results_net_effort", med_a, ref["asr_u4"]["median_read_share"], 1e-6)
    check("net ASReview u4 median net WSS = results_net_effort", TARGET - med_a, ref["asr_u4"]["median_net_wss95"], 1e-6)
    check("net judge positive reviews = results_net_effort", sum(TARGET - v > 0 for v in js.values()),
          ref["wss95"]["n_positive"], 0)
    check("net ASReview positive reviews = results_net_effort", sum(TARGET - v > 0 for v in as_.values()),
          ref["asr_u4"]["n_positive"], 0)
    return {"median_judge_net_full_precision": TARGET - med_j, "median_judge_net_stored_3dp": ref["wss95"]["median_net_wss95"],
            "median_asr_net": TARGET - med_a, "median_judge_read_share": med_j, "median_asr_read_share": med_a}


# ----------------------------------------------------------------------------- F10
def misses(R, revs, tau_of, strict=False):
    miss = below = 0
    for rev in revs:
        r = R[rev]
        tau = tau_of(rev)
        n1 = sum(r["y"])
        tp = sum(1 for p, y in zip(r["p"], r["y"]) if y and (p > tau if strict else p >= tau))
        miss += n1 - tp
        below += int(tp / n1 < TARGET)
    return {"misses": miss, "below_095": below}


def f10(D, R, revs):
    loo = {r: D["gaps"]["global_tau"]["per_review"][r]["tau_global"] for r in revs}
    out = {
        "n_eligible": sum(sum(R[r]["y"]) for r in revs),
        "n_test": sum(R[r]["n_test"] for r in revs),
        "per_review_tau_include_if_p_ge_tau": misses(R, revs, lambda r: R[r]["tau"]),
        "per_review_tau_include_only_if_p_gt_tau": misses(R, revs, lambda r: R[r]["tau"], strict=True),
        "fixed_global_007_include_if_p_ge": misses(R, revs, lambda r: 0.07),
        "loo_global_tau_include_if_p_ge": misses(R, revs, lambda r: loo[r]),
        "records_exactly_at_per_review_tau": sum(p == R[r]["tau"] for r in revs for p in R[r]["p"]),
        "records_exactly_at_007": sum(p == 0.07 for r in revs for p in R[r]["p"]),
        "loo_global_tau_values": {t: sum(1 for r in revs if loo[r] == t) for t in sorted(set(loo.values()))},
    }
    for key in ("fixed_global_007_include_if_p_ge", "loo_global_tau_include_if_p_ge", "per_review_tau_include_if_p_ge_tau"):
        out[key]["reviews_reaching_095"] = len(revs) - out[key]["below_095"]
    out["_note"] = "D3 pairs 35/39 with 50/577; 35/39 holds for fixed 0.07 (29 misses) and LOO global (50 misses)"
    check("F10 reviews reaching 0.95, fixed global 0.07 (D3 published 35)",
          out["fixed_global_007_include_if_p_ge"]["reviews_reaching_095"], 35, 0)
    check("F10 reviews reaching 0.95, LOO global tau (D3 published 35)",
          out["loo_global_tau_include_if_p_ge"]["reviews_reaching_095"], 35, 0)
    check("F10 misses at per-review tau (published 27)", out["per_review_tau_include_if_p_ge_tau"]["misses"], 27, 0)
    check("F10 reviews below 0.95 at per-review tau (published 14)",
          out["per_review_tau_include_if_p_ge_tau"]["below_095"], 14, 0)
    check("F10 misses at fixed global 0.07 (published 29)", out["fixed_global_007_include_if_p_ge"]["misses"], 29, 0)
    check("F10 misses at LOO-derived global tau (published 50)", out["loo_global_tau_include_if_p_ge"]["misses"], 50, 0)
    return out


# ----------------------------------------------------------------------------- F12
def f12(D, R, revs, n_draw=200):
    rng = np.random.default_rng(0)
    rows = []
    for rev in revs:
        r = R[rev]
        tau = r["tau"]
        correct = [int((p >= tau) == bool(y)) for p, y in zip(r["p"], r["y"])]
        c2 = np.array([jev_conf(p) for p in r["p"]])
        ct = np.array([abs(p - tau) for p in r["p"]])
        corr = np.array(correct)
        tb2, tbt, rnd = [], [], []
        for _ in range(n_draw):
            perm = rng.permutation(len(correct))
            tb2.append(jevlib.aurc(c2[perm].tolist(), corr[perm].tolist()))
            tbt.append(jevlib.aurc(ct[perm].tolist(), corr[perm].tolist()))
            rnd.append(jevlib.aurc(rng.random(len(correct)).tolist(), correct))
        rows.append({"review": rev, "n": len(correct), "error_rate": 1 - sum(correct) / len(correct),
                     "aurc_conf_2p_minus_1": jevlib.aurc(c2.tolist(), correct),
                     "aurc_conf_abs_p_minus_tau": jevlib.aurc(ct.tolist(), correct),
                     "aurc_conf_2p_minus_1_tiebreak_mean": st.mean(tb2),
                     "aurc_conf_abs_p_minus_tau_tiebreak_mean": st.mean(tbt),
                     "aurc_random_order_mean": st.mean(rnd),
                     "stored_aurc": D["stored"][rev]["aurc"]})
    check("F12 max |AURC(|2p-1|) - stored aurc| (3-dp rounding)",
          max(abs(x["aurc_conf_2p_minus_1"] - x["stored_aurc"]) for x in rows), 0.0, TOL)
    med = lambda k: st.median(x[k] for x in rows)
    return {"rows": rows, "summary": {k: med(k) for k in [
        "error_rate", "aurc_conf_2p_minus_1", "aurc_conf_abs_p_minus_tau",
        "aurc_conf_2p_minus_1_tiebreak_mean", "aurc_conf_abs_p_minus_tau_tiebreak_mean",
        "aurc_random_order_mean"]}}


# ----------------------------------------------------------------------------- F13
def f13(D, R, revs):
    rows, tau_mismatch = [], []
    for rev in revs:
        row = {"review": rev}
        for rule, fn in ablation.RULES.items():
            dev = [(fn(a), l) for _, a, l in R[rev]["dev"]]
            tau_s = sweep(dev)[0]
            if rule == "overall":
                tau = R[rev]["tau"]
                if abs(tau_s - tau) > 1e-9:
                    tau_mismatch.append(rev)
            else:
                tau = tau_s
            test = [(fn(a), l) for _, a, l in R[rev]["test"]]
            row[rule] = {"tau": tau, **ablation.score(test, tau)}
        rows.append(row)
    check("F13 overall tau: sweep(split dev) mismatches vs dev_tau.union (count)", len(tau_mismatch), 0, 0)
    base = [x["overall"]["wss"] for x in rows]
    out = {"tau_mismatch_reviews": tau_mismatch, "rows": rows, "summary": {}}
    for rule in ablation.RULES:
        w = [x[rule]["wss"] for x in rows]
        entry = {"median_wss": st.median(w),
                 "median_aurc": st.median(x[rule]["aurc"] for x in rows),
                 "reviews_rec_below_095": sum(x[rule]["rec"] < TARGET for x in rows),
                 "median_tau": st.median(x[rule]["tau"] for x in rows)}
        if rule != "overall":
            d = [a - b for a, b in zip(w, base)]
            entry.update({"median_paired_delta_vs_overall": st.median(d),
                          "rule_wins": sum(x > 0 for x in d), "rule_losses": sum(x < 0 for x in d),
                          "ties": sum(x == 0 for x in d),
                          "paired_bootstrap_p_B10000_seed0": jevlib.paired_bootstrap_p(st.median, w, base)})
        out["summary"][rule] = entry
    old = D["abl_old"]
    out["old_results_ablation39_median_wss"] = {
        rule: st.median(old[r][rule]["wss"] for r in revs) for rule in ablation.RULES}
    out["old_results_ablation39_tau_differs"] = {
        rule: [r for r, x in zip(revs, rows) if abs(old[r][rule]["tau"] - x[rule]["tau"]) > 1e-9] for rule in ablation.RULES}
    return out


# ----------------------------------------------------------------------------- F19
def hier_numpy(R, revs, B, seed):
    """Hierarchical bootstrap (reviews with replacement, then records within each drawn review),
    pooled recall and specificity at the dev tau. numpy default_rng(seed) (PCG64), not random.Random."""
    rng = np.random.default_rng(seed)
    data = [(np.array([p >= R[r]["tau"] for p in R[r]["p"]], dtype=bool),
             np.array(R[r]["y"], dtype=bool)) for r in revs]
    n_rev = len(data)
    rec, spec = np.empty(B), np.empty(B)
    for b in range(B):
        tp = n1 = tn = n0 = 0
        for j in rng.integers(0, n_rev, n_rev):
            pr, yy = data[j]
            m = len(yy)
            idx = rng.integers(0, m, m)
            ps, ys = pr[idx], yy[idx]
            tp += int(np.count_nonzero(ps & ys))
            n1 += int(np.count_nonzero(ys))
            tn += int(np.count_nonzero(~ps & ~ys))
            n0 += int(np.count_nonzero(~ys))
        rec[b] = tp / n1
        spec[b] = tn / n0
    return np.sort(rec), np.sort(spec)


def f19(D, R, revs):
    by_review = {r: [(int(p >= R[r]["tau"]), y) for p, y in zip(R[r]["p"], R[r]["y"])] for r in revs}
    flat = [x for v in by_review.values() for x in v]
    preds = [a for a, _ in flat]
    labels = [b for _, b in flat]

    def pooled_recall(pr, lb):
        return sum(1 for a, b in zip(pr, lb) if a == 1 and b == 1) / sum(lb)

    def pooled_spec(pr, lb):
        return sum(1 for a, b in zip(pr, lb) if a == 0 and b == 0) / sum(1 for b in lb if b == 0)

    out = {"point_recall": pooled_recall(preds, labels), "point_spec": pooled_spec(preds, labels)}
    lo, hi = jevlib.hier_boot_ci(pooled_recall, by_review, n=1000, seed=0)
    stored = D["gaps"]["hier_ci"]["pooled_recall"]["ci95"]
    check("F19 jevlib hier recall CI lower, B=1000 seed 0 (stored)", lo, stored[0], 1e-9)
    check("F19 jevlib hier recall CI upper, B=1000 seed 0 (stored)", hi, stored[1], 1e-9)
    out["jevlib_B1000_seed0_recall"] = [lo, hi]
    for seed in (0, 1, 2):
        rec, spec = hier_numpy(R, revs, 10000, seed)
        out[f"numpy_B10000_seed{seed}_recall_ci95"] = ci_sorted(rec)
        out[f"numpy_B10000_seed{seed}_spec_ci95"] = ci_sorted(spec)
    rec, spec = hier_numpy(R, revs, 1000, 0)
    out["numpy_B1000_seed0_recall_ci95"] = ci_sorted(rec)
    out["numpy_B1000_seed0_spec_ci95"] = ci_sorted(spec)
    return out


# ----------------------------------------------------------------------------- frozen threshold (deployable)
def frozen(R, revs, A, f7r):
    """WSS at achieved recall: (TN+FN)/N - (1 - recall_i), per review, at the frozen tau (the AUDIT convention;
    reproduces the published 0.64 and -0.057/13). The fixed-0.05 variant is reported only to show why it is not used."""
    rec = {x["review"]: x["recall"] for x in f7r["rows"]}
    excl = {r: sum(1 for p in R[r]["p"] if p < R[r]["tau"]) / R[r]["n_test"] for r in revs}
    fw = {r: excl[r] - (1 - rec[r]) for r in revs}
    d = [fw[r] - A[r]["asr_from_full"] for r in revs]
    out = {"median_wss_at_achieved_recall": st.median(fw.values()),
           "median_wss_fixed_005_variant_not_used": st.median(excl[r] - (1 - TARGET) for r in revs),
           "reviews_reaching_095": f7r["n_reach_095"],
           "median_paired_delta_vs_asr_from_full": st.median(d),
           "wins_vs_asr_from_full": sum(x > 0 for x in d), "losses_vs_asr_from_full": sum(x < 0 for x in d)}
    check("frozen: median WSS at achieved recall (AUDIT cites 0.64)", out["median_wss_at_achieved_recall"], 0.64, 0.0005)
    check("frozen: reviews reaching 0.95 (published 25)", out["reviews_reaching_095"], 25, 0)
    check("frozen - ASReview from-full median paired delta (published -0.057)",
          out["median_paired_delta_vs_asr_from_full"], -0.057, 0.0005)
    check("frozen - ASReview from-full wins (published 13)", out["wins_vs_asr_from_full"], 13, 0)
    return out


def frozen_paired(R, revs, A, f7r):
    """Paired CIs for the frozen-cut-off rows of tab:paired. Two WSS conventions for the frozen rule:
    achieved recall (credits overshoot above 0.95; matches frozen() and the 0.64 table row) and
    nominal target 0.95 (WSS = share unread - 0.05; same convention as the oracle and ASReview arms)."""
    rec = {x["review"]: x["recall"] for x in f7r["rows"]}
    excl = {r: sum(1 for p in R[r]["p"] if p < R[r]["tau"]) / R[r]["n_test"] for r in revs}
    conventions = {
        "achieved": {r: excl[r] - (1 - rec[r]) for r in revs},
        "nominal": {r: excl[r] - (1 - TARGET) for r in revs},
    }
    out = {}
    for conv, fw in conventions.items():
        for name, base in [("test_only", "asr_test"), ("full_pool", "asr_from_full")]:
            diffs = [round(fw[r] - A[r][base], 12) for r in revs]
            lo, hi = jevlib.bootstrap_ci(st.median, diffs, n=10000, seed=0)
            out[f"frozen_{conv}_minus_asr_{name}"] = {
                "median": st.median(diffs), "ci95_B10000_seed0": [lo, hi],
                "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs),
                "ties": sum(d == 0 for d in diffs)}
    a, n_ = out["frozen_achieved_minus_asr_test_only"], out["frozen_nominal_minus_asr_test_only"]
    check("frozen achieved - ASReview test-only median (published +0.103)", a["median"], 0.103, 0.0005)
    check("frozen achieved - ASReview test-only wins (published 28)", a["wins"], 28, 0)
    check("frozen nominal - ASReview test-only median (audit recompute +0.088)", n_["median"], 0.088, 0.0005)
    check("frozen nominal - ASReview full-pool median (audit recompute -0.023)",
          out["frozen_nominal_minus_asr_full_pool"]["median"], -0.023, 0.0005)
    for key, (w, l, t) in [("frozen_nominal_minus_asr_test_only", (26, 11, 2)),
                           ("frozen_nominal_minus_asr_full_pool", (18, 20, 1))]:
        x = out[key]
        check(f"{key} wins (integer k-count verified)", x["wins"], w, 0)
        check(f"{key} losses (integer k-count verified)", x["losses"], l, 0)
        check(f"{key} ties (integer k-count verified)", x["ties"], t, 0)
    return out


# ----------------------------------------------------------------------------- paired gaps and MDE
def paired(diffs):
    n = len(diffs)
    sd = st.stdev(diffs)
    lo, hi = jevlib.bootstrap_ci(st.median, diffs, n=10000, seed=0)
    return {"n": n, "median": st.median(diffs), "median_ci95_B10000_seed0": [lo, hi],
            "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs), "sd_paired": sd,
            "mde_mean_80pct": Z * sd / math.sqrt(n),
            "mde_median_80pct_normal_approx": Z * sd / math.sqrt(n) * math.sqrt(math.pi / 2)}


def gaps_and_mde(D, A, revs):
    out = {
        "synergy_judge_minus_asr_test_only": paired([A[r]["judge"] - A[r]["asr_test"] for r in revs]),
        "synergy_judge_minus_asr_from_full": paired([A[r]["judge"] - A[r]["asr_from_full"] for r in revs]),
    }
    chan = D["chan"]
    out["external_judge_minus_u4_22"] = paired([v["jev_wss95"] - v["elas_u4_wss95"] for v in chan.values()])
    out["external_judge_minus_u3_22_reference"] = paired([v["jev_wss95"] - v["elas_u3_wss95"] for v in chan.values()])
    s = out["synergy_judge_minus_asr_test_only"]
    check("SYNERGY median paired gap, judge - ASReview test-only (published +0.156)", s["median"], 0.156, TOL)
    check("SYNERGY wins, judge vs ASReview test-only (published 29)", s["wins"], 29, 0)
    check("SYNERGY median paired gap, judge - ASReview from-full (published +0.004)",
          out["synergy_judge_minus_asr_from_full"]["median"], 0.004, TOL)
    check("SYNERGY wins, judge vs ASReview from-full (published 20)",
          out["synergy_judge_minus_asr_from_full"]["wins"], 20, 0)
    check("SYNERGY median of ASReview test-only WSS (published 0.532)",
          st.median(A[r]["asr_test"] for r in revs), 0.532, TOL)
    check("SYNERGY median of ASReview from-full WSS (published 0.698)",
          st.median(A[r]["asr_from_full"] for r in revs), 0.698, TOL)
    e = out["external_judge_minus_u4_22"]
    check("external u4 median paired gap (published -0.0113)", e["median"], -0.0113, 0.00006)
    return out


# ----------------------------------------------------------------------------- audit 2026-11-24
LEGACY = ["Attai_2022", "Leenaars_2020", "Rinne_2021", "Viguie_2020", "van_de_Schoot_2025"]
T21 = 2.079613845  # t quantile 0.975, 21 degrees of freedom


def f20(D, R, revs, A, f7r):
    """Audit 2026-11-24: legacy-five overlap with the earlier partition and drop-check, clause counts,
    second-reader kappa, external mean paired gap, Clopper-Pearson interval widths."""
    out = {}
    old = json.load(open(path("data", "processed", "split.json")))["reviews"]
    new = json.load(open(path("data", "processed", "split.union.json")))["reviews"]
    ov = {}
    for r in LEGACY:
        part = {i: k for k, v in old[r]["ids"].items() for i in v}
        t = new[r]["ids"]["test"]
        c = {k: sum(part.get(i) == k for i in t) for k in ("dev", "test", "holdout")}
        ov[r] = {"n_test": len(t), **c, "in_earlier_partition": sum(c.values()),
                 "share_earlier_dev": c["dev"] / len(t)}
    out["legacy_overlap"] = ov
    check("F20 legacy test records lying in the earlier partition (all)", 
          sum(x["in_earlier_partition"] for x in ov.values()) / sum(x["n_test"] for x in ov.values()), 1.0, 0)

    def pack(rs):
        f = f7(R, rs)
        rec = {x["review"]: x["recall"] for x in f["rows"]}
        fw = [sum(1 for p in R[r]["p"] if p < R[r]["tau"]) / R[r]["n_test"] - (1 - rec[r]) for r in rs]
        return {"n_reviews": len(rs), "reviews_reaching_095": f["n_reach_095"],
                "missed": sum(x["n_eligible"] - x["n_caught"] for x in f["rows"]),
                "n_eligible": sum(x["n_eligible"] for x in f["rows"]),
                "median_wss_at_achieved_recall": st.median(fw)}

    rest = [r for r in revs if r not in LEGACY]
    out["frozen_drop_legacy"] = {"legacy_five": pack(LEGACY), "other_34": pack(rest)}
    check("F20 other-34 reviews reaching 0.95 (published 21)", out["frozen_drop_legacy"]["other_34"]["reviews_reaching_095"], 21, 0)
    check("F20 other-34 eligible studies missed (published 24)", out["frozen_drop_legacy"]["other_34"]["missed"], 24, 0)

    Q = load_questions()
    # same split as questions.split_clauses, counted before its [:8] cap
    raw = {r: len([p for p in re.split(r"(?<=[.;])\s+|\n+", Q[r]["criteria"].strip()) if len(p.strip(" ;-")) > 20])
           for r in revs}
    out["clause_counts_raw"] = {"per_review": raw, "n_reviews_over_8": sum(v > 8 for v in raw.values()), "max": max(raw.values())}
    check("F20 reviews with more than 8 raw clauses (published 13)", out["clause_counts_raw"]["n_reviews_over_8"], 13, 0)

    def verd(name):
        d = json.load(open(path("data", "processed", name)))
        return {x["rid"]: x.get("verdict") for x in d} if isinstance(d, list) else {r: x.get("verdict") for r, x in d.items()}

    key = json.load(open(path("data", "processed", "key_audit_key.json")))
    V, S = verd("key_audit_verdicts.json"), verd("key_audit_verdicts_second.json")
    cats = ("INCLUDE", "EXCLUDE", "UNSURE")
    ids = [r for r in key if key[r]["kind"] == "neg_hi" and r in S and S[r] in cats]
    a, b, n = [V[r] for r in ids], [S[r] for r in ids], len(ids)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum(a.count(c) * b.count(c) for c in cats) / n ** 2
    out["second_reader_neg_hi"] = {"n_valid": n, "same_verdict": sum(x == y for x, y in zip(a, b)), "kappa_3cat": (po - pe) / (1 - pe)}
    check("F20 second reader agrees on disputed records (published 81 of 99)", out["second_reader_neg_hi"]["same_verdict"], 81, 0)
    check("F20 second reader valid disputed verdicts (published 99)", n, 99, 0)
    check("F20 kappa second vs first reader (published 0.30)", out["second_reader_neg_hi"]["kappa_3cat"], 0.30, 0.005)

    d = [v["jev_wss95"] - v["elas_u4_wss95"] for v in D["chan"].values()]
    m, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    out["external_mean_gap_u4"] = {"n": len(d), "mean": m, "ci95_t": [m - T21 * se, m + T21 * se],
                                   "max_abs_diff": max(map(abs, d))}

    w = [x for x in f7r["rows"] if x["cp95"][1] - x["cp95"][0] > 0.3]
    out["cp_widths"] = {"n_wider_than_03": len(w), "n_wider_below_recall_1": sum(x["recall"] < 1 for x in w),
                        "n_zero_width": sum(x["cp95"][1] == x["cp95"][0] for x in f7r["rows"])}
    check("F20 recall intervals wider than 0.3 (published 24)", len(w), 24, 0)
    return out

# ----------------------------------------------------------------------------- main
def main():
    D = load()
    revs = D["revs"]
    R = build(D)
    A = arms(D, R)
    out = {}
    out["oracle_wss_median"] = st.median(A[r]["judge"] for r in revs)
    check("median oracle WSS@95 judge (published 0.743; oracle stop)", out["oracle_wss_median"], 0.743, TOL)
    out["f7_clopper_pearson"] = f7(R, revs)
    f7r = out["f7_clopper_pearson"]
    check("F7 reviews below 0.95 (published 14)", f7r["n_below_095"], 14, 0)
    out["f8_calibration_synergy"] = f8(D)
    out["f9_reads"] = f9(A, revs)
    out["f9_net"] = net_block(D, out["f9_reads"], A)
    out["f10_ties_reconciliation"] = f10(D, R, revs)
    out["f12_aurc"] = f12(D, R, revs)
    out["f13_ablation"] = f13(D, R, revs)
    check("F13 ablation overall median WSS on split ids (published 0.743)",
          out["f13_ablation"]["summary"]["overall"]["median_wss"], 0.743, TOL)
    out["f19_hierarchical_ci"] = f19(D, R, revs)
    out["frozen_threshold"] = frozen(R, revs, A, f7r)
    out["frozen_paired_cis"] = frozen_paired(R, revs, A, f7r)
    out["paired_gaps_and_mde"] = gaps_and_mde(D, A, revs)
    out["f20_audit"] = f20(D, R, revs, A, f7r)
    out["arms_per_review"] = A
    out["checks"] = CHECKS
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_md(out)
    n_fail = sum(not c["pass"] for c in CHECKS)
    print(f"wrote {OUT_JSON} and {OUT_MD}; checks {len(CHECKS) - n_fail}/{len(CHECKS)} PASS")
    return out


def write_md(o):
    f7r, f8r, f9r = o["f7_clopper_pearson"], o["f8_calibration_synergy"], o["f9_reads"]
    f10r, f12r, f13r = o["f10_ties_reconciliation"], o["f12_aurc"], o["f13_ablation"]
    f19r, fz, g = o["f19_hierarchical_ci"], o["frozen_threshold"], o["paired_gaps_and_mde"]
    s9, w9 = f9r["summary"], f9r["worked_2000"]
    s13 = f13r["summary"]
    syn, syn_f, ext = g["synergy_judge_minus_asr_test_only"], g["synergy_judge_minus_asr_from_full"], g["external_judge_minus_u4_22"]
    fails = [c for c in CHECKS if not c["pass"]]
    L = []
    add = L.append
    add("# v3 numbers (generated by v3_fixes.py; reproduce with `python3 v3_fixes.py` from x3/)")
    add("")
    add("Conventions: eligible = relevant test record (label 1). Frozen per-review tau from runs/jev/dev_tau.union.json; "
        "caught = p >= tau. WSS@95 = (TN+FN)/N - 0.05 (oracle stop unless stated). Stored numbers are 3-dp rounded; "
        "tolerance 0.0005.")
    add("")
    add("## Cross-check status")
    add(f"{len(CHECKS) - len(fails)}/{len(CHECKS)} PASS. " + ("No discrepancies." if not fails else
        "Discrepancies (report, do not hide):"))
    for c in fails:
        add(f"- FAIL: {c['check']} computed {c['computed']:.6g} vs published {c['published']:.6g}")
    add("")
    add("## F7 Clopper-Pearson 95% CIs on per-review recall (frozen tau, 39 reviews)")
    add(f"- Reviews reaching 0.95: {f7r['n_reach_095']}/39; below: {f7r['n_below_095']}.")
    add(f"- Of the {f7r['n_below_095']} below-target reviews, {f7r['n_below_095'] - f7r['n_below_with_cp_upper_below_095']} have a CP upper bound at or above 0.95, so their shortfall is not statistically established.")
    add(f"- Of the {f7r['n_reach_095']} reviews at or above 0.95, {f7r['n_reach_with_cp_lower_below_095']} have a CP lower bound below 0.95.")
    add(f"- Median CP half-width {f7r['median_cp_halfwidth']:.3f}; worst review {f7r['worst_review']} at recall {f7r['worst_recall']:.3f}.")
    add("- Sentence: \"Clopper-Pearson 95% intervals on per-review recall at the frozen threshold have median half-width "
        f"{f7r['median_cp_halfwidth']:.2f}; {f7r['n_below_095'] - f7r['n_below_with_cp_upper_below_095']} of the {f7r['n_below_095']} reviews below 0.95 "
        "have intervals that include 0.95, so per-review shortfalls are point estimates.\"")
    add("")
    add("## F8 SYNERGY label-count calibration (naive rule = frozen max-spec rule, B=200 draws per review)")
    add("| k | conditional pass, recall>=0.95 on eligible draws | unconditional pass, all draws | share with no eligible record | median tau | median spec |")
    add("|---|---|---|---|---|---|")
    for k in (50, 100, 200, 500):
        x = f8r[str(k)]
        add(f"| {k} | {x['pass_conditional']:.3f} | {x['pass_unconditional']:.3f} | {x['share_no_eligible']:.3f} | {x['tau_median']:.2f} | {x['spec_median']:.3f} |")
    add(f"- Dev-tau sanity: {f8r['sanity_dev_tau_reproduced']} reproduced from full dev.")
    add("- Sentence: \"With k=50 labelled development records, the calibrated threshold reaches 0.95 recall on "
        f"{f8r['50']['pass_conditional']:.0%} of eligible draws ({f8r['50']['pass_unconditional']:.0%} of all draws); "
        f"at k=500 it reaches {f8r['500']['pass_conditional']:.0%} ({f8r['500']['pass_unconditional']:.0%} of all draws).\"")
    add("")
    add("## F9 absolute reads per arm (median review; 39 reviews)")
    add(f"- Median dev labels {s9['median_n_dev']:.0f}; median test records {s9['median_n_test']:.0f}.")
    add(f"- Judge, oracle stop (records read in ranked order until 95% recall; the WSS basis): {s9['median_judge_test_reads']:.0f} test reads; with {s9['median_n_dev']:.0f} dev labels = {s9['median_judge_total_reads']:.0f}.")
    add(f"- Judge, frozen per-review threshold (every record with p >= tau is read; records below tau are auto-excluded): {s9['median_frozen_test_reads']:.0f} test reads (share {s9['share_frozen_test']:.3f}); with {s9['median_n_dev']:.0f} dev labels = {s9['median_frozen_total_reads']:.0f}. The two medians coincide at {s9['median_judge_test_reads']:.0f}; per review they differ.")
    add(f"- ASReview test-only: {s9['median_asr_test_only_test_reads']:.0f} test reads, 0 dev reads used (net WSS still charges n_dev).")
    add(f"- ASReview full pool: {s9['median_asr_full_pool_reads']:.0f} reads of {s9['median_n_full']:.0f} records; "
        f"test-scored from the full order: {s9['median_asr_from_full_pool_reads']:.0f} reads.")
    add(f"- Worked 2,000-record review (60/20/20, median share): judge {w9['judge_total_reads']} = {w9['judge_dev_labels']} dev + "
        f"{w9['judge_test_reads']} test; ASReview test-only {w9['asr_test_only_reads']} test reads; "
        f"ASReview full pool {w9['asr_full_pool_reads']} reads.")
    add(f"- Net WSS (both arms charged n_dev): judge {o['f9_net']['median_judge_net_full_precision']:.4f} at full precision "
        f"({o['f9_net']['median_judge_net_stored_3dp']:.4f} in results_net_effort.json from 3-dp WSS), ASReview u4 {o['f9_net']['median_asr_net']:.4f}.")
    add("- Sentence: \"Each arm's cost is in absolute reads: the judge reads "
        f"{s9['median_judge_total_reads']:.0f} records for the median review, of which {s9['median_n_dev']:.0f} are development labels; "
        "the net-effort gap shrinks mechanically because both arms are charged the same development reads, and the "
        "comparison charges ASReview development reads it does not use.\"")
    add("")
    add("## F10 tie sensitivity and 27 / 29 / 50 reconciliation")
    p = f10r["per_review_tau_include_if_p_ge_tau"]
    add(f"- Per-review tau, include if p >= tau: {p['misses']} missed of {f10r['n_eligible']} eligible; {p['below_095']} reviews below 0.95.")
    add(f"- Per-review tau, include only if p > tau (ties excluded): {f10r['per_review_tau_include_only_if_p_gt_tau']['misses']} missed; "
        f"{f10r['per_review_tau_include_only_if_p_gt_tau']['below_095']} reviews below 0.95.")
    add(f"- Fixed global 0.07, include if p >= 0.07: {f10r['fixed_global_007_include_if_p_ge']['misses']} missed.")
    add(f"- LOO-derived global tau (per-review, from results_gaps39.json): {f10r['loo_global_tau_include_if_p_ge']['misses']} missed; tau values {f10r['loo_global_tau_values']}.")
    add(f"- Records exactly at per-review tau: {f10r['records_exactly_at_per_review_tau']}; exactly at 0.07: {f10r['records_exactly_at_007']}.")
    add(f"- Reviews reaching 0.95: per-review tau {f10r['per_review_tau_include_if_p_ge_tau']['reviews_reaching_095']}/39; "
        f"fixed global 0.07 {f10r['fixed_global_007_include_if_p_ge']['reviews_reaching_095']}/39; "
        f"LOO global tau {f10r['loo_global_tau_include_if_p_ge']['reviews_reaching_095']}/39.")
    add("- 27 = per-review threshold. 29 = fixed global 0.07. 50 = LOO-derived global tau (gap_tests.py: each review's global tau fit on the pooled dev records of the other 38 reviews). The LOO figure is the one matching the paper's 50/577.")
    add("- DISCREPANCY vs D3: D3 attributes 50/577 to the label-free global threshold 0.07. The fixed 0.07 threshold misses 29/577; 50 belongs to the LOO global tau. Both reach 35/39 reviews. Use the threshold name explicitly.")
    add("")
    add("## F12 AURC under two confidence definitions")
    m = f12r["summary"]
    add(f"- Median AURC, confidence |2p-1|: {m['aurc_conf_2p_minus_1']:.3f} (stored results_test column reproduced; this is the convention used).")
    add(f"- Median AURC, confidence |p-tau|: {m['aurc_conf_abs_p_minus_tau']:.3f}, worse than random order ({m['aurc_random_order_mean']:.3f}); not used as a confidence.")
    add(f"- Random tie-break mean: |2p-1| {m['aurc_conf_2p_minus_1_tiebreak_mean']:.3f}; |p-tau| {m['aurc_conf_abs_p_minus_tau_tiebreak_mean']:.3f}.")
    add(f"- Random-order reference (AURC with random confidence, mean of 200 draws): {m['aurc_random_order_mean']:.3f}; median error rate {m['error_rate']:.3f}.")
    add("- Sentence: \"Ranking by confidence beats random order (AURC "
        f"{m['aurc_conf_2p_minus_1']:.3f} vs {m['aurc_random_order_mean']:.3f} at random), but confidence was not calibrated per record.\"")
    add("")
    add("## F13 clause ablation on split test ids (dev_tau.union for overall; sweep on split dev for AND/OR)")
    add("| rule | median WSS | median AURC | reviews rec<0.95 | paired delta vs overall (median) | wins | losses | ties | paired p (B=10000) |")
    add("|---|---|---|---|---|---|---|---|---|")
    for rule in ("overall", "and_min", "or_max"):
        x = s13[rule]
        if rule == "overall":
            add(f"| {rule} | {x['median_wss']:.3f} | {x['median_aurc']:.3f} | {x['reviews_rec_below_095']} | - | - | - | - | - |")
        else:
            add(f"| {rule} | {x['median_wss']:.3f} | {x['median_aurc']:.3f} | {x['reviews_rec_below_095']} | "
                f"{x['median_paired_delta_vs_overall']:+.3f} | {x['rule_wins']} | {x['rule_losses']} | {x['ties']} | "
                f"{x['paired_bootstrap_p_B10000_seed0']:.4f} |")
    add(f"- Old results_ablation39.json median WSS: {f13r['old_results_ablation39_median_wss']}. Reviews whose tau differs from the old run: {f13r['old_results_ablation39_tau_differs']}. The old ablation.py did not restrict dev or test to split ids (it iterates all of runs/jev/jev_dev.json and runs/jev/jev_test.json); this run does. Overall-rule median WSS is unchanged.")
    add("")
    add("## F19 hierarchical recall bootstrap (reviews, then records within review; pooled recall at dev tau)")
    add(f"- Point pooled recall {f19r['point_recall']:.4f}; pooled specificity {f19r['point_spec']:.4f}.")
    add(f"- jevlib, B=1000, seed 0 (matches stored gaps39 CI): recall {f19r['jevlib_B1000_seed0_recall'][0]:.4f} to {f19r['jevlib_B1000_seed0_recall'][1]:.4f}.")
    for seed in (0, 1, 2):
        lo, hi = f19r[f"numpy_B10000_seed{seed}_recall_ci95"]
        add(f"- numpy default_rng(seed={seed}), B=10000: recall {lo:.4f} to {hi:.4f}.")
    lo, hi = f19r["numpy_B1000_seed0_recall_ci95"]
    add(f"- numpy default_rng(seed=0), B=1000: recall {lo:.4f} to {hi:.4f}.")
    add("")
    add("## Frozen per-review threshold (deployable) and paired gaps")
    add(f"- WSS at achieved recall (TN+FN)/N - (1 - recall) at the frozen tau: median {fz['median_wss_at_achieved_recall']:.3f}; "
        f"reviews reaching 0.95: {fz['reviews_reaching_095']}/39. (The fixed-0.05 variant, median {fz['median_wss_fixed_005_variant_not_used']:.3f}, is not used.)")
    add(f"- Frozen minus ASReview from-full: paired median {fz['median_paired_delta_vs_asr_from_full']:+.3f}; frozen wins {fz['wins_vs_asr_from_full']}/39.")
    fp = o["frozen_paired_cis"]
    for key, label in [("frozen_achieved_minus_asr_test_only", "achieved-recall WSS, test partition"),
                       ("frozen_achieved_minus_asr_full_pool", "achieved-recall WSS, full review"),
                       ("frozen_nominal_minus_asr_test_only", "nominal-0.95 WSS, test partition"),
                       ("frozen_nominal_minus_asr_full_pool", "nominal-0.95 WSS, full review")]:
        x = fp[key]
        add(f"- Frozen cut-off vs ASReview Ultra ({label}): paired median {x['median']:+.3f}, "
            f"95% CI [{x['ci95_B10000_seed0'][0]:+.3f}, {x['ci95_B10000_seed0'][1]:+.3f}] (B=10,000, seed 0); "
            f"won/lost/tied {x['wins']}/{x['losses']}/{x['ties']}.")
    add(f"- SYNERGY judge minus ASReview test-only: median {syn['median']:+.3f}, B=10000 CI [{syn['median_ci95_B10000_seed0'][0]:.3f}, {syn['median_ci95_B10000_seed0'][1]:.3f}], wins {syn['wins']}/39, paired SD {syn['sd_paired']:.3f}.")
    add(f"- SYNERGY judge minus ASReview from-full: median {syn_f['median']:+.3f}, CI [{syn_f['median_ci95_B10000_seed0'][0]:.3f}, {syn_f['median_ci95_B10000_seed0'][1]:.3f}], wins {syn_f['wins']}/39.")
    add(f"- External (22 Cochrane) judge minus ASReview u4: median {ext['median']:+.4f}, CI [{ext['median_ci95_B10000_seed0'][0]:.4f}, {ext['median_ci95_B10000_seed0'][1]:.4f}], wins {ext['wins']}, losses {ext['losses']}, paired SD {ext['sd_paired']:.4f}.")
    add("")
    add("## MDE (80% power, two-sided alpha 0.05, normal approximation, paired by review)")
    add("Assumptions: reviews independent; observed paired SD of per-review WSS@95 differences; "
        "mean-based MDE = (1.960+0.842) x SD / sqrt(n); median-based MDE approximated by the mean MDE x sqrt(pi/2) = 1.2533 (normal asymptotics).")
    for name, x in (("SYNERGY judge-ASReview test-only, n=39", syn), ("SYNERGY judge-ASReview from-full, n=39", syn_f),
                    ("External judge-u4, n=22", ext)):
        add(f"- {name}: MDE mean {x['mde_mean_80pct']:.4f}; median (approx) {x['mde_median_80pct_normal_approx']:.4f}.")
    add("- Sentence: \"At 80% power the comparison could detect a paired median gap of about "
        f"{ext['mde_median_80pct_normal_approx']:.3f} in the external set and {syn['mde_median_80pct_normal_approx']:.3f} in SYNERGY; "
        "smaller gaps are not excluded.\"")
    add("")
    a = o["f20_audit"]
    add("## F20 Audit 2026-11-24 (legacy overlap, drop-check, clause counts, kappa, mean gap, interval widths)")
    for r, x in a["legacy_overlap"].items():
        add(f"- Legacy {r}: {x['n_test']} test records, all {x['in_earlier_partition']} in the earlier partition "
            f"(dev {x['dev']}, {x['share_earlier_dev']:.0%}; test {x['test']}; holdout {x['holdout']}).")
    for k in ("other_34", "legacy_five"):
        x = a["frozen_drop_legacy"][k]
        add(f"- Frozen cut-off, {k}: reach 0.95 {x['reviews_reaching_095']}/{x['n_reviews']}; missed {x['missed']} of {x['n_eligible']}; "
            f"median WSS at achieved recall {x['median_wss_at_achieved_recall']:.4f}.")
    c = a["clause_counts_raw"]
    if c:
        over = {r: v for r, v in c["per_review"].items() if v > 8}
        add(f"- Raw clause counts above the cap of 8: {c['n_reviews_over_8']} of 39 reviews, maximum {c['max']}; reviews and counts {over}.")
    s, m, w = a["second_reader_neg_hi"], a["external_mean_gap_u4"], a["cp_widths"]
    add(f"- Second reader on disputed records: agrees {s['same_verdict']} of {s['n_valid']}; kappa {s['kappa_3cat']:.3f}.")
    add(f"- External judge minus Ultra, mean paired gap {m['mean']:+.4f}, t interval [{m['ci95_t'][0]:+.4f}, {m['ci95_t'][1]:+.4f}] "
        f"(n={m['n']}, largest single difference {m['max_abs_diff']:.3f}).")
    add(f"- Recall intervals wider than 0.3: {w['n_wider_than_03']} of 39 ({w['n_wider_below_recall_1']} with recall below 1.0); zero-width: {w['n_zero_width']}.")
    add("")
    add("STATUS: DONE")
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
