"""Label-light calibration simulation on CACHED Jev probabilities (X3 item 3, redesigned).

$0 / zero API calls. Per-record noul probabilities are reconstructed offline from the
cache materialization (runs/jev/jev_dev.json + runs/jev/jev_test.json), filtered to the
dev/test ids in data/split.union.json. Any split record missing from the cache is
counted and dropped (never re-queried).

Design (per NEXT_STEPS.md item-3 redesign + Main steer):
  * 39 reviews (union39). For k in {50,100,200,500}: B=200 draws WITHOUT replacement
    from each review's DEV records (dev-only partition; test never touched for calibration).
    Draws are IDENTICAL across rules (same per-(k,review) seed).
  * Calibration rules (all max-specificity on the 0.01 grid, differing in the recall
    criterion they impose on the draw):
      naive_r0.95 : draw recall >= 0.95            (the frozen sweep.py rule)
      margin_r0.98: draw recall >= 0.98            (conservative)
      margin_r0.99: draw recall >= 0.99            (conservative)
      margin_r1.00: draw recall >= 1.00            (conservative: catch every eligible)
      cp95        : one-sided 95% Clopper-Pearson LOWER bound of draw recall >= 0.95;
                    if no grid tau satisfies it (typical when n1_draw is small) use
                    FALLBACK tau=0.01 (accept nearly everything) and count the fallback.
  * Evaluate tau on the review's TEST records: recall, specificity, auto@5% (conf |2p-1|,
    identical to score_test.py), AURC. WSS@95 is threshold-free -> never an outcome here.
  * "Unattainable" is SPLIT into no_include_draw (n1_draw==0) vs threshold_unattainable
    (n1_draw>0 but no grid tau meets the criterion). Reported per rule x k.
  * Each k is reported on the ALL-review set AND on the COMMON 29-review set
    (n_dev>=500) so k columns are like-for-like.
  * Infinite-label reference: full-dev tau (naive r*=0.95) test recall P(>=0.95)
    across the 39 reviews -- the design floor the conservative rules are measured against.

Outcomes per rule x k: median [IQR] test recall; P(test recall>=0.95) with Clopper-Pearson
95% CI (exact binomial, n = evaluated draw x review pairs); median spec / auto5 / AURC;
share unattainable (naive/margin) or share fallback (cp95); across-draw tau distribution;
and one hierarchical-shrinkage pooled recall estimate (formula stated in the JSON).
"""
import json, os, sys, time
import numpy as np
from scipy.stats import beta as beta_dist

from layout import P  # noqa: E402  (study-relative paths -> case folders; adds ../shared to sys.path)
import jevlib  # canonical metrics; used to validate the fast equivalents below

GRID = [round(x * 0.01, 2) for x in range(1, 100)]          # 0.01 .. 0.99 (sweep.py)
KS = [50, 100, 200, 500]
B = 200
RECALL_TARGET = 0.95
MAX_LOSS = 0.05

RULES = {
    "naive_r0.95":  {"type": "margin", "r_star": 0.95},
    "margin_r0.98": {"type": "margin", "r_star": 0.98},
    "margin_r0.99": {"type": "margin", "r_star": 0.99},
    "margin_r1.00": {"type": "margin", "r_star": 1.00},
    "cp95":         {"type": "cp", "alpha": 0.05, "target": 0.95, "fallback_tau": 0.01},
}

# CP lower bound is monotonic in k_caught (fixed n1): precompute k_min[n1] = smallest
# k with beta.ppf(alpha, k, n1-k+1) >= target. CP criterion then reduces to
# k_caught >= k_min[n1] -- O(1) per tau, no per-tau beta.ppf (was the bottleneck).
def _build_kmin(max_n1, alpha, target):
    kmin = {}
    for n1 in range(1, max_n1 + 1):
        km = n1 + 1  # impossible -> always fallback
        for k in range(n1, -1, -1):
            lb = 0.0 if k == 0 else float(beta_dist.ppf(alpha, k, n1 - k + 1))
            if lb >= target:
                km = k
            else:
                break
        kmin[n1] = km
    return kmin

KMIN = {}  # filled in main() once max n_dev is known


# ---------------------------------------------------------------- data loading

def load_pairs():
    """{rev: {"dev": [(p,y)], "test": [(p,y)]}} from cached JSON, filtered to split ids."""
    split = json.load(open(P("data/processed/split.union.json")))
    devj = json.load(open(P("runs/jev/jev_dev.json")))
    testj = json.load(open(P("runs/jev/jev_test.json")))
    out, n_miss_dev, n_miss_test = {}, 0, 0
    for rev, r in split["reviews"].items():
        dj, tj = devj.get(rev, {}), testj.get(rev, {})
        dev, test = [], []
        for rid in r["ids"]["dev"]:
            if rid in dj:
                dev.append((dj[rid]["answers"]["include"]["noul"], dj[rid]["label"]))
            else:
                n_miss_dev += 1
        for rid in r["ids"]["test"]:
            if rid in tj:
                test.append((tj[rid]["answers"]["include"]["noul"], tj[rid]["label"]))
            else:
                n_miss_test += 1
        out[rev] = {"dev": dev, "test": test}
    return out, n_miss_dev, n_miss_test


# ---------------------------------------------------------------- tau rules

def tau_rule(p, y, rule):
    """Max-specificity grid tau meeting the rule's draw-recall criterion.

    Returns (tau, status): status in {"ok","no_include_draw","threshold_unattainable"}.
    "no_include_draw": n1==0 (recall undefined). "threshold_unattainable": n1>0 but no
    grid tau meets the criterion. For cp95 the caller converts a non-ok status into the
    fallback tau. Tie-break matches sweep.py (smallest tau among max-spec).
    """
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0:
        return None, "no_include_draw"
    e_asc = np.sort(p[y == 1])
    n_asc = np.sort(p[y == 0])
    best_tau, best_spec = None, -1.0
    for tau in GRID:
        k_caught = n1 - np.searchsorted(e_asc, tau, side="left")
        if rule["type"] == "margin":
            met = (k_caught / n1) >= rule["r_star"]
        else:  # cp
            met = k_caught >= KMIN.get(n1, n1 + 1)
        if not met:
            continue
        spec = (np.searchsorted(n_asc, tau, side="left") / n0) if n0 else 0.0
        if best_tau is None or spec > best_spec:
            best_tau, best_spec = tau, spec
    if best_tau is None:
        return None, "threshold_unattainable"
    return best_tau, "ok"


# ---------------------------------------------------------------- test metrics (fast, jevlib-validated)

def auto_frac_fast(conf, corr, y, max_loss=MAX_LOSS):
    """Max kept fraction with include-loss<=max_loss. O(n log n); matches jevlib.auto_frac_at_loss."""
    n_inc = int(y.sum())
    if n_inc == 0:
        return float("nan")
    order = np.argsort(-conf, kind="stable")
    missed = (y == 1) & (~corr)
    prefix = np.cumsum(missed[order])
    ok = prefix / n_inc <= max_loss
    if not ok.any():
        return 0.0
    k_max = int(np.max(np.nonzero(ok)[0])) + 1
    return k_max / len(conf)


def aurc_fast(conf, corr):
    """Area under the risk-coverage curve. Matches jevlib.aurc."""
    order = np.argsort(-conf, kind="stable")
    err = (~corr).astype(float)
    prefix = np.cumsum(err[order])
    risks = prefix / np.arange(1, len(conf) + 1)
    return float(risks.mean())


def eval_test(p, y, tau):
    pred = (p >= tau).astype(int)
    n1 = int(y.sum()); n0 = len(y) - n1
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    rec = tp / n1 if n1 else float("nan")
    spec = tn / n0 if n0 else float("nan")
    conf = np.abs(2 * p - 1)
    corr = pred == y
    return rec, spec, auto_frac_fast(conf, corr, y), aurc_fast(conf, corr)


# ---------------------------------------------------------------- self-check: fast == jevlib

def selfcheck():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(2, 60))
        p = np.round(rng.random(n), 2)
        y = (rng.random(n) < 0.2).astype(int)
        if y.sum() == 0:
            y[0] = 1
        tau = float(rng.choice(GRID))
        pred = (p >= tau).astype(int)
        conf = np.abs(2 * p - 1)
        corr = pred == y
        assert abs(auto_frac_fast(conf, corr, y) - jevlib.auto_frac_at_loss(list(conf), list(corr), list(y))) < 1e-12
        assert abs(aurc_fast(conf, corr) - jevlib.aurc(list(conf), list(corr))) < 1e-12
    # tau rule vs brute-force sweep.py loop on random draws, for every rule
    for _ in range(200):
        n = int(rng.integers(5, 80))
        p = np.round(rng.random(n), 2)
        y = (rng.random(n) < 0.25).astype(int)
        if y.sum() == 0:
            y[0] = 1
        for rule in RULES.values():
            got, got_status = tau_rule(p, y, rule)
            n1 = int(y.sum()); n0 = n - n1
            if n1 == 0:
                assert got_status == "no_include_draw"
                continue
            best = None
            for tau in GRID:
                pr = [1 if pp >= tau else 0 for pp in p]
                kc = sum(1 for a, b in zip(pr, y) if a and b)
                if rule["type"] == "margin":
                    met = (k_c := kc) / n1 >= rule["r_star"]
                else:
                    lb = 0.0 if kc == 0 else float(beta_dist.ppf(rule["alpha"], kc, n1 - kc + 1))
                    met = lb >= rule["target"]
                if not met:
                    continue
                spec = (sum(1 for a, b in zip(pr, y) if not a and not b) / max(n0, 1)) if n0 else 0.0
                if best is None or spec > best[1]:
                    best = (tau, spec)
            ref_status = "ok" if best else "threshold_unattainable"
            assert got_status == ref_status, (rule, got_status, ref_status)
            if best:
                assert got == best[0], (rule, got, best[0])
    print("selfcheck OK: fast auto/aurc == jevlib; all tau rules == brute force")


# ---------------------------------------------------------------- sanity + infinite-label reference

def sanity_and_reference(pairs):
    """Reproduce published dev-tau (runs/jev/dev_tau.union.json) from FULL dev; also the
    infinite-label reference: test recall at that full-dev tau, P(>=0.95) across reviews."""
    pub = json.load(open(P("runs/jev/dev_tau.union.json")))["reviews"]
    sanity, ref = {}, {}
    max_diff = 0.0
    for rev, d in pairs.items():
        p_dev = np.array([pp for pp, _ in d["dev"]]); y_dev = np.array([l for _, l in d["dev"]]).astype(int)
        tau = tau_rule(p_dev, y_dev, RULES["naive_r0.95"])[0]
        exp = pub[rev]["tau"]
        max_diff = max(max_diff, abs((tau if tau is not None else -1) - exp))
        sanity[rev] = {"published": exp, "recomputed": tau, "match": tau == exp}
        p_test = np.array([pp for pp, _ in d["test"]]); y_test = np.array([l for _, l in d["test"]]).astype(int)
        rec, spec, auto, aurc = eval_test(p_test, y_test, tau)
        ref[rev] = {"tau": tau, "test_recall": round(rec, 4), "test_spec": round(spec, 4),
                    "test_auto5": round(auto, 4), "test_aurc": round(aurc, 4)}
    n_ge = sum(1 for v in ref.values() if v["test_recall"] >= RECALL_TARGET)
    return sanity, max_diff, ref, n_ge


# ---------------------------------------------------------------- Clopper-Pearson

def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


# ---------------------------------------------------------------- simulation

def simulate(pairs):
    revs = list(pairs.keys())
    common29 = [r for r in revs if len(pairs[r]["dev"]) >= 500]
    results = {}   # results[rule][k] = {"rows": [...], "counts": {...}}
    for k in KS:
        # generate draws once per (k, review); identical across rules
        draws_by_rev = {}
        for ri, rev in enumerate(revs):
            n_dev = len(pairs[rev]["dev"])
            if n_dev < k:
                continue
            rng = np.random.default_rng(k * 100003 + ri)
            draws_by_rev[rev] = [rng.choice(n_dev, size=k, replace=False) for _ in range(B)]
        for rule_name, rule in RULES.items():
            rows = []
            counts = {"n_total": 0, "n_no_include_draw": 0, "n_threshold_unattainable": 0,
                      "n_fallback": 0, "n_ok": 0}
            for rev in draws_by_rev:
                dev, test = pairs[rev]["dev"], pairs[rev]["test"]
                p_dev = np.array([pp for pp, _ in dev]); y_dev = np.array([l for _, l in dev]).astype(int)
                p_test = np.array([pp for pp, _ in test]); y_test = np.array([l for _, l in test]).astype(int)
                for idx in draws_by_rev[rev]:
                    tau, status = tau_rule(p_dev[idx], y_dev[idx], rule)
                    counts["n_total"] += 1
                    if status == "no_include_draw":
                        counts["n_no_include_draw"] += 1
                    elif status == "threshold_unattainable":
                        counts["n_threshold_unattainable"] += 1
                    if rule["type"] == "cp" and status != "ok":
                        tau = rule["fallback_tau"]
                        status = "fallback"
                        counts["n_fallback"] += 1
                    if status == "ok":
                        counts["n_ok"] += 1
                    if status in ("ok", "fallback"):
                        rec, spec, auto, aurc = eval_test(p_test, y_test, tau)
                        rows.append({"rev": rev, "rec": rec, "spec": spec, "auto": auto,
                                     "aurc": aurc, "tau": tau, "status": status})
            results.setdefault(rule_name, {})[k] = {"rows": rows, "counts": counts,
                                                    "common29": set(common29)}
    return results, common29


def aggregate(rule_name, k, R, common29):
    """Build the per-(rule,k) outcome dict for both the all-review and common-29 sets."""
    rows = R["rows"]; counts = R["counts"]
    out = {}
    for set_name, rev_set in (("all", None), ("common29", R["common29"])):
        if set_name == "common29":
            srows = [r for r in rows if r["rev"] in rev_set]
        else:
            srows = rows
        recs = np.array([r["rec"] for r in srows]); specs = np.array([r["spec"] for r in srows])
        autos = np.array([r["auto"] for r in srows]); aurcs = np.array([r["aurc"] for r in srows])
        taus = np.array([r["tau"] for r in srows])
        n_eval = len(srows)
        n_ge = int((recs >= RECALL_TARGET).sum())
        cp_lo, cp_hi = clopper_pearson(n_ge, n_eval)
        # hierarchical-shrinkage pooled recall over reviews in this set
        revs = sorted({r["rev"] for r in srows})
        r_i = np.array([np.mean([r["rec"] for r in srows if r["rev"] == rv]) for rv in revs])
        w_i = np.array([len(pairs_global[rv]["test"]) for rv in revs], dtype=float)
        r_bar = float(r_i.mean()); K = float(np.median(w_i))
        pooled = float(((w_i * r_i + K * r_bar) / (w_i + K)).mean())
        out[set_name] = {
            "n_reviews": len(revs), "n_evaluated": n_eval,
            "n_no_include_draw": counts["n_no_include_draw"] if set_name == "all" else None,
            "n_threshold_unattainable": counts["n_threshold_unattainable"] if set_name == "all" else None,
            "n_fallback": counts["n_fallback"] if set_name == "all" else None,
            "share_unattainable": round((counts["n_no_include_draw"] + counts["n_threshold_unattainable"]) / counts["n_total"], 4) if counts["n_total"] else None,
            "share_fallback": round(counts["n_fallback"] / counts["n_total"], 4) if counts["n_total"] else None,
            "recall_median": round(float(np.median(recs)), 4),
            "recall_iqr": [round(float(np.percentile(recs, 25)), 4), round(float(np.percentile(recs, 75)), 4)],
            "recall_p_ge_095": round(n_ge / n_eval, 4) if n_eval else None,
            "recall_p_ge_095_cp95_ci": [round(cp_lo, 4), round(cp_hi, 4)],
            "spec_median": round(float(np.median(specs)), 4),
            "auto5_median": round(float(np.median(autos)), 4),
            "aurc_median": round(float(np.median(aurcs)), 4),
            "tau_median": round(float(np.median(taus)), 4),
            "tau_iqr": [round(float(np.percentile(taus, 25)), 4), round(float(np.percentile(taus, 75)), 4)],
            "pooled_recall_shrinkage": round(pooled, 4),
            "pooled_recall_simple_review_mean": round(r_bar, 4),
        }
    return out


def main():
    global pairs_global
    t0 = time.time()
    selfcheck()
    pairs, n_miss_dev, n_miss_test = load_pairs()
    pairs_global = pairs
    print(f"loaded {len(pairs)} reviews; missing dev probs={n_miss_dev}, missing test probs={n_miss_test}")

    sanity, max_diff, ref, ref_n_ge = sanity_and_reference(pairs)
    n_match = sum(1 for v in sanity.values() if v["match"])
    print(f"sanity: dev-tau reproduced for {n_match}/{len(sanity)} reviews (max abs diff={max_diff})")
    if n_match != len(sanity):
        print("SANITY CHECK FAILED -- stopping. Mismatches:",
              {r: v for r, v in sanity.items() if not v["match"]})
        sys.exit(1)
    print(f"infinite-label reference: full-dev tau test recall P(>=0.95) = "
          f"{ref_n_ge}/{len(ref)} = {ref_n_ge/len(ref):.3f}")
    global KMIN
    max_n1 = max(len(d["dev"]) for d in pairs.values())
    KMIN = _build_kmin(max_n1, RULES["cp95"]["alpha"], RULES["cp95"]["target"])
    print(f"CP k_min lookup built for n1=1..{max_n1}")

    results, common29 = simulate(pairs)
    print(f"common-29 set: {len(common29)} reviews (n_dev>=500)")

    out_results = {}
    for rule_name in RULES:
        out_results[rule_name] = {}
        for k in KS:
            out_results[rule_name][str(k)] = aggregate(rule_name, k, results[rule_name][k], common29)

    payload = {
        "meta": {
            "description": "Label-light calibration simulation on cached Jev noul probabilities (X3 item 3 redesigned)",
            "k_values": KS, "B": B, "recall_target": RECALL_TARGET, "grid": "0.01..0.99 (0.01 step)",
            "rules": {n: (f"max-spec s.t. draw recall>={r['r_star']}" if r["type"] == "margin"
                          else f"max-spec s.t. CP95 lower bound of draw recall>=0.95; fallback tau=0.01")
                      for n, r in RULES.items()},
            "partition": "calibration draws from DEV only; evaluation on TEST (data/split.union.json); draws identical across rules (same per-(k,review) seed)",
            "probabilities_source": "runs/jev/jev_dev.json + runs/jev/jev_test.json (cache materialization), filtered to split.union.json ids; 0 API calls",
            "n_reviews": len(pairs), "n_common29": len(common29),
            "n_dev_missing_dropped": n_miss_dev, "n_test_missing_dropped": n_miss_test,
            "unattainable_split": "no_include_draw (n1==0) vs threshold_unattainable (n1>0, no grid tau meets criterion); cp95 converts both to fallback tau=0.01",
            "recall_proportion_unit": "evaluated draw x review pairs (naive/margin: unattainable excluded; cp95: fallback included)",
            "hierarchical_shrinkage_formula": "r_i=mean recall over evaluated draws; w_i=n_test_i; r_bar=mean(r_i); K=median(w_i); shrunk_i=(w_i*r_i+K*r_bar)/(w_i+K); pooled=mean(shrunk_i)",
            "outcomes": ["recall", "spec", "auto5", "aurc"],
            "wss95_note": "threshold-free; never a calibration outcome (per contract)",
            "zero_api_calls": True,
            "runtime_sec": round(time.time() - t0, 1),
        },
        "sanity_check": {
            "description": "reproduce runs/jev/dev_tau.union.json tau from FULL dev set per review",
            "n_reviews": len(sanity), "n_match": n_match, "max_abs_diff": max_diff,
            "all_match": n_match == len(sanity), "per_review": sanity,
        },
        "infinite_label_reference": {
            "description": "full-dev tau (naive r*=0.95) test recall across 39 reviews -- the design floor",
            "n_reviews": len(ref), "n_recall_ge_095": ref_n_ge,
            "p_recall_ge_095": round(ref_n_ge / len(ref), 4),
            "per_review": ref,
        },
        "results": out_results,
    }
    dest = P("results/jev/results_calibsim39.json")
    json.dump(payload, open(dest, "w"), indent=1)
    print(f"wrote {dest}")

    # ---- summary table: rule x k x set
    print("\n=== Label-light calibration simulation (39 reviews, dev-only draws, B=200) ===")
    print(f"infinite-label reference (full-dev tau): P(test recall>=0.95) = "
          f"{ref_n_ge}/{len(ref)} = {ref_n_ge/len(ref):.3f}\n")
    hdr = (f"{'rule':>13} {'k':>4} {'set':>8} {'n_rev':>5} {'n_eval':>6} {'rec_med':>7} "
           f"{'rec_IQR':>14} {'P(rec>=.95)':>11} {'CP95_CI':>16} {'spec':>6} {'auto5':>6} "
           f"{'aurc':>6} {'tau_med':>7} {'unatt%/fall%':>12} {'pool_rec':>8}")
    print(hdr)
    for rule_name in RULES:
        for k in KS:
            for set_name in ("all", "common29"):
                r = out_results[rule_name][str(k)][set_name]
                iqr = f"[{r['recall_iqr'][0]:.3f},{r['recall_iqr'][1]:.3f}]"
                ci = f"[{r['recall_p_ge_095_cp95_ci'][0]:.3f},{r['recall_p_ge_095_cp95_ci'][1]:.3f}]"
                share = r["share_fallback"] if rule_name == "cp95" else r["share_unattainable"]
                share_s = f"{share*100:.1f}%" if share is not None else "n/a"
                print(f"{rule_name:>13} {k:>4} {set_name:>8} {r['n_reviews']:>5} {r['n_evaluated']:>6} "
                      f"{r['recall_median']:>7.3f} {iqr:>14} {r['recall_p_ge_095']:>11.3f} {ci:>16} "
                      f"{r['spec_median']:>6.3f} {r['auto5_median']:>6.3f} {r['aurc_median']:>6.3f} "
                      f"{r['tau_median']:>7.2f} {share_s:>12} {r['pooled_recall_shrinkage']:>8.3f}")
        print("-" * len(hdr))
    print(f"\nruntime {time.time()-t0:.1f}s; zero API calls; sanity dev-tau match={n_match}/{len(sanity)}")


if __name__ == "__main__":
    main()
