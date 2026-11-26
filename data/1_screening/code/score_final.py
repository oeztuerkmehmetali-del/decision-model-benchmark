"""Final locked scoring: Jev with fine-grid dev taus + Jev WSS@95 + bootstrap CIs + paired p vs ASReview."""
import json, os, sys, time

from layout import P, load_questions  # noqa: E402  (study-relative paths -> case folders; adds ../shared to sys.path)
import jevlib  # noqa: E402

CACHE = P(os.environ.get("X3_CACHE", "runs/jev/jev_cache.jsonl"))
def load_dev_tau():
    """Dev taus from sweep.py (spec rule); fails loudly if any scored review lacks one (0c)."""
    import sys
    tauname = os.environ.get("X3_TAU", "runs/jev/dev_tau.json")
    only = set(sys.argv[1:]) or None
    with open(P(tauname)) as f:
        d = json.load(f)
    taus = {rev: v["tau"] for rev, v in d["reviews"].items()
            if only is None or rev in only}
    return taus

def load_pairs(splitname):
    jevlib._load(CACHE)
    qs = load_questions()
    out = {}
    for rev, r in qs.items():
        clauses = r["clauses"]
        for rid, v in r["records"].items():
            if v["split"] != splitname:
                continue
            qq = {f"clause_{i}": {"type": "noul",
                  "instructions": f"Does this record satisfy this eligibility clause? Clause: {c}"}
                  for i, c in enumerate(clauses)}
            qq["include"] = {"type": "noul",
                "instructions": "Overall: should this record be INCLUDED for full-text review? "
                                f"Eligibility criteria: {v['state']['criteria']}"}
            p = jevlib.ask(v["state"], qq, cache=CACHE)["answers"]["include"]["noul"]
            out.setdefault(rev, []).append((p, v["label"]))
    return out

def main():
    t0 = time.time()
    DEV_TAU = load_dev_tau()
    jev = load_pairs("test")
    only = sys.argv[1:] or None
    rows = []
    miss_tau = [rev for rev in jev if (not only or rev in only) and rev not in DEV_TAU]
    assert not miss_tau, f"dev_tau.json missing taus for: {miss_tau}"
    for rev, pairs in jev.items():
        if only and rev not in only:
            continue
        tau = DEV_TAU[rev]
        y = [lab for _, lab in pairs]
        ps = [p for p, _ in pairs]
        pred = [1 if p >= tau else 0 for p in ps]
        tp = sum(1 for a, b in zip(pred, y) if a and b)
        rec = tp / sum(y)
        spec = sum(1 for a, b in zip(pred, y) if not a and not b) / (len(y) - sum(y))
        conf = [abs(2 * p - 1) for p in ps]
        corr = [a == b for a, b in zip(pred, y)]
        auto = jevlib.auto_frac_at_loss(conf, corr, y)
        aurc = jevlib.aurc(conf, corr)
        wss = jevlib.wss_at_recall(ps, y)
        ece = jevlib.ece_binary(ps, y)
        brier = jevlib.brier_binary(ps, y)
        lo, hi = jevlib.bootstrap_ci(
            lambda a, b: (sum(1 for x, z in zip(a, b) if x and z) / sum(b)) if sum(b) else 0.0,
            pred, y, n=1000, seed=0)
        asr = {}
        for ai in ("elas_u4", "elas_u3"):
            d = json.load(open(P(f"runs/asreview/{rev.lower()}_{ai}.json")))
            asr[ai] = d["test"]["wss95"]
        rows.append({"review": rev, "n_test": len(y), "n_incl": sum(y), "tau": tau,
                     "rec": round(rec, 3), "rec_ci": [round(lo, 3), round(hi, 3)],
                     "spec": round(spec, 3), "auto5": round(auto, 3),
                     "aurc": round(aurc, 3), "wss95": round(wss, 3),
                     "ece": round(ece, 3), "brier": round(brier, 4),
                     "asr_u4": round(asr["elas_u4"], 3), "asr_u3": round(asr["elas_u3"], 3)})
        print(f"{rev}: tau={tau} rec={rec:.3f}[{lo:.3f},{hi:.3f}] spec={spec:.3f} "
              f"auto5={auto:.3f} aurc={aurc:.3f} wss95={wss:.3f} ece={ece:.3f} "
              f"| asr_u4={asr['elas_u4']:.3f} u3={asr['elas_u3']:.3f}")
    dest = os.environ.get("X3_OUT", "results/results_test.json")
    json.dump(rows, open(P(dest), "w"), indent=1)
    print(f"done {time.time()-t0:.0f}s")
    # paired: Jev wss vs pre-registered Ultra comparator (elas_u4); NB (elas_u3) descriptive only (0f)
    jw = [r["wss95"] for r in rows]
    aw = [r["asr_u4"] for r in rows]
    print("Jev WSS:", jw, "Ultra WSS:", aw)
    print("paired bootstrap p (Jev vs Ultra):",
          round(jevlib.paired_bootstrap_p(lambda c: sum(c) / len(c), jw, aw), 4))
    print("NB elas_u3 (descriptive):", [r["asr_u3"] for r in rows])

if __name__ == "__main__":
    main()
