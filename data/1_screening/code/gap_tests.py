"""Gap tests for X3: global tau, shortfall predictors, title-only subgroup, hierarchical CI.
Zero API spend — all from cached Jev outputs.
"""
import json, os, sys, csv
from layout import P  # noqa: E402  (study-relative paths -> case folders; adds ../shared to sys.path)
import jevlib
from sweep import sweep
from scipy.stats import spearmanr, mannwhitneyu




def load_all():
    jev_dev = json.load(open(P("runs/jev/jev_dev.json")))
    jev_test = json.load(open(P("runs/jev/jev_test.json")))
    split = json.load(open(P("data/processed/split.union.json")))
    results_test = json.load(open(P("results/jev/results_test.union39.json")))
    dev_tau = json.load(open(P("runs/jev/dev_tau.union.json")))
    return jev_dev, jev_test, split, results_test, dev_tau


def pairs_for(jev, split, rev, partition):
    """Get (noul, label) pairs for a review's partition."""
    recs = jev[rev]
    ids = set(split["reviews"][rev]["ids"].get(partition, []))
    return [(r["answers"]["include"]["noul"], r["label"])
            for rid, r in recs.items() if rid in ids]


def recall_spec_auto(pairs, tau):
    """Compute recall, specificity, auto-excluded share."""
    n = len(pairs)
    n1 = sum(l for _, l in pairs)
    n0 = n - n1
    tp = sum(1 for p, l in pairs if p >= tau and l == 1)
    tn = sum(1 for p, l in pairs if p < tau and l == 0)
    rec = tp / n1 if n1 else 0.0
    spec = tn / n0 if n0 else 0.0
    auto_excl = sum(1 for p, _ in pairs if p < tau) / n if n else 0.0
    return rec, spec, auto_excl


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def main():
    jev_dev, jev_test, split, results_test, dev_tau = load_all()
    reviews = sorted(jev_dev.keys())

    # ---- Section 1: global tau (leave-one-review-out) ----
    # Optimized: pre-compute TP/TN per tau, then leave-one-out by subtraction.
    print("Section 1: global tau (leave-one-review-out)...")
    global_tau = {"per_review": {}, "summary": {}}

    # Pre-compute per-review dev pairs
    dev_pairs_by_review = {r: pairs_for(jev_dev, split, r, "dev") for r in reviews}
    all_dev_pairs = [p for r in reviews for p in dev_pairs_by_review[r]]

    TAUS = [round(x * 0.01, 2) for x in range(1, 100)]
    n1_all = sum(l for _, l in all_dev_pairs)
    n0_all = len(all_dev_pairs) - n1_all

    # Pre-compute TP/TN per tau for all_dev and per review
    tp_all = {tau: sum(1 for p, l in all_dev_pairs if p >= tau and l == 1) for tau in TAUS}
    tn_all = {tau: sum(1 for p, l in all_dev_pairs if p < tau and l == 0) for tau in TAUS}
    tp_by_review = {r: {tau: sum(1 for p, l in dev_pairs_by_review[r] if p >= tau and l == 1) for tau in TAUS} for r in reviews}
    tn_by_review = {r: {tau: sum(1 for p, l in dev_pairs_by_review[r] if p < tau and l == 0) for tau in TAUS} for r in reviews}
    n1_by_review = {r: sum(l for _, l in dev_pairs_by_review[r]) for r in reviews}
    n0_by_review = {r: len(dev_pairs_by_review[r]) - n1_by_review[r] for r in reviews}

    # For each review, find optimal tau on leave-one-out pooled dev
    for r in reviews:
        best_tau, best_spec = None, -1.0
        for tau in TAUS:
            tp_loo = tp_all[tau] - tp_by_review[r][tau]
            tn_loo = tn_all[tau] - tn_by_review[r][tau]
            n1_loo = n1_all - n1_by_review[r]
            n0_loo = n0_all - n0_by_review[r]
            rec_loo = tp_loo / n1_loo if n1_loo else 0.0
            if rec_loo < 0.95:
                continue
            spec_loo = tn_loo / n0_loo if n0_loo else 0.0
            if spec_loo > best_spec:
                best_spec = spec_loo
                best_tau = tau
        if best_tau is None:
            best_tau = 0.5
        tau_global = best_tau

        test_pairs = pairs_for(jev_test, split, r, "test")
        rec, spec, auto_excl = recall_spec_auto(test_pairs, tau_global)
        global_tau["per_review"][r] = {
            "tau_global": tau_global,
            "rec": rec,
            "spec": spec,
            "auto_excluded": auto_excl,
            "n_test": len(test_pairs),
            "n_incl": sum(l for _, l in test_pairs),
        }
        print(f"  {r}: tau_global={tau_global:.2f} rec={rec:.3f} spec={spec:.3f} auto_excl={auto_excl:.3f}")

    recs = [v["rec"] for v in global_tau["per_review"].values()]
    specs = [v["spec"] for v in global_tau["per_review"].values()]
    autos = [v["auto_excluded"] for v in global_tau["per_review"].values()]
    missed = sum(v["n_incl"] * (1 - v["rec"]) for v in global_tau["per_review"].values())
    global_tau["summary"] = {
        "median_recall": median(recs),
        "n_recall_ge_095": sum(1 for r in recs if r >= 0.95),
        "n_recall_eq_100": sum(1 for r in recs if r == 1.0),
        "median_spec": median(specs),
        "median_auto_excluded": median(autos),
        "total_missed_eligible": round(missed, 1),
        "total_eligible": 577,
    }

    per_review_cmp = {}
    for r in reviews:
        rt = next(x for x in results_test if x["review"] == r)
        per_review_cmp[r] = {
            "rec_per_review_tau": rt["rec"],
            "spec_per_review_tau": rt["spec"],
            "rec_global_tau": global_tau["per_review"][r]["rec"],
            "spec_global_tau": global_tau["per_review"][r]["spec"],
        }
    global_tau["comparison"] = per_review_cmp

    # ---- Section 2: shortfall predictors ----
    print("Section 2: shortfall predictors...")
    shortfall = {}
    rec_list, n_incl_list, ece_list = [], [], []
    for r in reviews:
        rt = next(x for x in results_test if x["review"] == r)
        rec_list.append(rt["rec"])
        n_incl_list.append(rt["n_incl"])
        ece_list.append(rt["ece"])

    rho_nincl, p_nincl = spearmanr(rec_list, n_incl_list)
    shortfall["spearman_recall_vs_n_incl"] = {"rho": rho_nincl, "p": p_nincl}

    rho_ece, p_ece = spearmanr(rec_list, ece_list)
    shortfall["spearman_recall_vs_ece"] = {"rho": rho_ece, "p": p_ece}

    below = [n for n, r in zip(n_incl_list, rec_list) if r < 0.95]
    above = [n for n, r in zip(n_incl_list, rec_list) if r >= 0.95]
    if below and above:
        u_stat, u_p = mannwhitneyu(below, above, alternative="two-sided")
    else:
        u_stat, u_p = float("nan"), float("nan")
    shortfall["mann_whitney_n_incl"] = {
        "median_below_095": median(below) if below else None,
        "median_above_095": median(above) if above else None,
        "n_below": len(below),
        "n_above": len(above),
        "u_stat": u_stat,
        "p": u_p,
    }

    # ---- Section 3: title-only subgroup ----
    print("Section 3: title-only subgroup...")
    title_only = {"per_review": {}, "summary": {}}

    def subgroup_metrics(pairs, tau):
        if not pairs:
            return {"n": 0, "n_incl": 0, "rec": None,
                    "mean_noul_eligible": None, "mean_noul_ineligible": None}
        n = len(pairs)
        n1 = sum(l for _, l in pairs)
        tp = sum(1 for p, l in pairs if p >= tau and l == 1)
        rec = tp / n1 if n1 else 0.0
        eligible_nouls = [p for p, l in pairs if l == 1]
        ineligible_nouls = [p for p, l in pairs if l == 0]
        return {
            "n": n,
            "n_incl": n1,
            "rec": rec,
            "mean_noul_eligible": sum(eligible_nouls) / len(eligible_nouls) if eligible_nouls else None,
            "mean_noul_ineligible": sum(ineligible_nouls) / len(ineligible_nouls) if ineligible_nouls else None,
        }

    all_title_pairs = []
    all_withabs_pairs = []
    for r in reviews:
        csv_path = P(f"data/raw/synergy/{r}.test.csv")
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping")
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        test_pairs_dict = {}
        recs = jev_test[r]
        ids = set(split["reviews"][r]["ids"].get("test", []))
        for rid, rec in recs.items():
            if rid in ids:
                test_pairs_dict[rid] = (rec["answers"]["include"]["noul"], rec["label"])

        title_only_pairs = []
        with_abs_pairs = []
        for row in rows:
            rid = row["record_id"]
            if rid not in test_pairs_dict:
                continue
            noul, label = test_pairs_dict[rid]
            abstract = row.get("abstract", "").strip()
            if not abstract:
                title_only_pairs.append((noul, label))
            else:
                with_abs_pairs.append((noul, label))

        tau = dev_tau["reviews"][r]["tau"]
        title_only["per_review"][r] = {
            "tau": tau,
            "title_only": subgroup_metrics(title_only_pairs, tau),
            "with_abstract": subgroup_metrics(with_abs_pairs, tau),
        }
        for p, l in title_only_pairs:
            all_title_pairs.append((p, l, tau))
        for p, l in with_abs_pairs:
            all_withabs_pairs.append((p, l, tau))

    def pooled_subgroup_metrics(pairs_with_tau):
        if not pairs_with_tau:
            return {"n": 0, "n_incl": 0, "rec": None,
                    "mean_noul_eligible": None, "mean_noul_ineligible": None}
        n = len(pairs_with_tau)
        n1 = sum(l for _, l, _ in pairs_with_tau)
        tp = sum(1 for p, l, t in pairs_with_tau if p >= t and l == 1)
        rec = tp / n1 if n1 else 0.0
        eligible_nouls = [p for p, l, _ in pairs_with_tau if l == 1]
        ineligible_nouls = [p for p, l, _ in pairs_with_tau if l == 0]
        return {
            "n": n,
            "n_incl": n1,
            "rec": rec,
            "mean_noul_eligible": sum(eligible_nouls) / len(eligible_nouls) if eligible_nouls else None,
            "mean_noul_ineligible": sum(ineligible_nouls) / len(ineligible_nouls) if ineligible_nouls else None,
        }

    title_only["summary"] = {
        "title_only": pooled_subgroup_metrics(all_title_pairs),
        "with_abstract": pooled_subgroup_metrics(all_withabs_pairs),
    }

    # ---- Section 4: hierarchical bootstrap CI ----
    print("Section 4: hierarchical bootstrap CI...")
    by_review = {}
    for r in reviews:
        tau = dev_tau["reviews"][r]["tau"]
        test_pairs = pairs_for(jev_test, split, r, "test")
        by_review[r] = [(1 if p >= tau else 0, l) for p, l in test_pairs]

    def pooled_recall(preds, labels):
        n1 = sum(labels)
        if n1 == 0:
            return 0.0
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        return tp / n1

    def pooled_spec(preds, labels):
        n0 = sum(1 for l in labels if l == 0)
        if n0 == 0:
            return 0.0
        tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
        return tn / n0

    recall_ci = jevlib.hier_boot_ci(pooled_recall, by_review, n=1000, seed=0)
    spec_ci = jevlib.hier_boot_ci(pooled_spec, by_review, n=1000, seed=0)

    all_preds = [p for pairs in by_review.values() for p, _ in pairs]
    all_labels = [l for pairs in by_review.values() for _, l in pairs]
    point_recall = pooled_recall(all_preds, all_labels)
    point_spec = pooled_spec(all_preds, all_labels)

    hier_ci = {
        "pooled_recall": {"point": point_recall, "ci95": list(recall_ci)},
        "pooled_spec": {"point": point_spec, "ci95": list(spec_ci)},
        "n_resamples": 1000,
        "seed": 0,
    }

    # ---- Write output ----
    out = {
        "global_tau": global_tau,
        "shortfall": shortfall,
        "title_only": title_only,
        "hier_ci": hier_ci,
    }
    out_path = P("results/jev/results_gaps39.json")
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"\nWrote {out_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Global tau: median recall = {global_tau['summary']['median_recall']:.3f}")
    print(f"  #reviews recall>=0.95: {global_tau['summary']['n_recall_ge_095']}/39")
    print(f"  #reviews recall=1.00: {global_tau['summary']['n_recall_eq_100']}/39")
    print(f"  median spec = {global_tau['summary']['median_spec']:.3f}")
    print(f"  median auto-excluded = {global_tau['summary']['median_auto_excluded']:.3f}")
    print(f"  total missed eligible = {global_tau['summary']['total_missed_eligible']}/577")
    print(f"\nShortfall: Spearman recall vs n_incl rho={rho_nincl:.3f} p={p_nincl:.4f}")
    print(f"  Spearman recall vs ECE rho={rho_ece:.3f} p={p_ece:.4f}")
    print(f"  Mann-Whitney n_incl: below_095 median={shortfall['mann_whitney_n_incl']['median_below_095']}, "
          f"above_095 median={shortfall['mann_whitney_n_incl']['median_above_095']}")
    print(f"\nTitle-only: {title_only['summary']['title_only']['n']} records, "
          f"{title_only['summary']['title_only']['n_incl']} includes")
    print(f"  with abstract: {title_only['summary']['with_abstract']['n']} records, "
          f"{title_only['summary']['with_abstract']['n_incl']} includes")
    print(f"\nHier CI: pooled recall = {point_recall:.3f} [{recall_ci[0]:.3f}, {recall_ci[1]:.3f}]")
    print(f"  pooled spec = {point_spec:.3f} [{spec_ci[0]:.3f}, {spec_ci[1]:.3f}]")


if __name__ == "__main__":
    main()
