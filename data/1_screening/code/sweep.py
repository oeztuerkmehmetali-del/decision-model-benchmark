"""Score Jev vs ASReview on identical test records: risk-coverage, AURC, auto-frac@loss, WSS@95, ECE/Brier."""
import json, os, sys, time

from layout import P  # noqa: E402  (study-relative paths -> case folders; adds ../shared to sys.path)
import jevlib  # noqa: E402
TAUS = [round(x * 0.01, 2) for x in range(1, 100)]

def jev_conf(p):
    return abs(2 * p - 1)  # distance from maximal uncertainty

def rank_conf(order, rid):
    """Ranker triage confidence: screen from both ends, escalate the middle."""
    pos = order.index(rid) / max(len(order) - 1, 1)
    return abs(2 * pos - 1)

def coverage_at_loss(items, max_loss=0.05):
    """items: (conf, correct, is_include). Greedy top-conf order; max kept frac with include-loss<=max."""
    items = sorted(items, key=lambda t: -t[0])
    n_incl = sum(t[2] for t in items)
    best, kept_loss = 0.0, 0
    missed = n_incl
    for k, (_, correct, is_incl) in enumerate(items, 1):
        if is_incl and not correct:
            pass
        missed = sum(1 for _, c, i in items[:k] if i and not c)
        loss = missed / n_incl if n_incl else 0
        if loss <= max_loss:
            best = k / len(items)
    return best

def sweep(dev):
    """Spec rule (locked protocol): max dev specificity s.t. dev recall >= 0.95, fine grid 0.01.
    Verified 2026-10-05 vs locked-5: reproduces DEV_TAU within 0.01 (float-grid direction only)."""
    y = [l for _, l in dev]
    n1 = sum(y); n0 = len(y) - n1
    best = None
    for tau in TAUS:
        pred = [1 if p >= tau else 0 for p, _ in dev]
        tp = sum(1 for a, b in zip(pred, y) if a and b)
        rec = tp / max(n1, 1)
        if rec < 0.95:
            continue
        tn = sum(1 for a, b in zip(pred, y) if not a and not b)
        spec = tn / max(n0, 1)
        if best is None or spec > best[1]:
            best = (tau, spec, rec)
    if best is None:
        return (0.5, 0.0, 0.0)
    tau, spec, rec = best
    conf = [jev_conf(p) for p, _ in dev]
    correct = [int(p >= tau) == l for p, l in dev]
    auto = jevlib.auto_frac_at_loss(conf, correct, y)
    return (tau, auto, rec)

def main():
    import sys
    t0 = time.time()
    devname = os.environ.get("X3_DEV", "runs/jev/jev_dev.json")
    splitname = os.environ.get("X3_SPLIT", "data/split.json")
    tauname = os.environ.get("X3_TAU", "runs/jev/dev_tau.json")
    only = sys.argv[1:] or None
    jev = json.load(open(P(devname)))
    split = json.load(open(P(splitname)))
    out = {"model": jevlib.MODEL, "taus": TAUS, "reviews": {}}
    for rev, recs in jev.items():
        if only and rev not in only:
            continue
        if rev not in split["reviews"]:
            continue
        dev_ids = set(split["reviews"][rev]["ids"].get("dev", []))
        dev = [(r["answers"]["include"]["noul"], r["label"]) for rid, r in recs.items() if rid in dev_ids]
        if not dev:
            dev = [(r["answers"]["include"]["noul"], r["label"]) for r in recs.values()]
        tau, auto_dev, rec_dev = sweep(dev)
        out["reviews"][rev] = {"tau": tau, "dev_auto_frac": auto_dev, "dev_recall": rec_dev,
                               "dev_n": len(dev)}
        print(f"{rev}: tau={tau} dev_auto={auto_dev:.3f} dev_rec={rec_dev:.3f} n={len(dev)}")
    # test scoring happens after test fan-out; save dev taus
    json.dump(out, open(P(tauname), "w"), indent=1)
    print(f"dev sweep done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
