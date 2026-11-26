"""Run ASReview simulate (elas_u4 + elas_u3) per review, extract screen order + WSS@95."""
import csv, json, os, sqlite3, subprocess, tempfile, zipfile

from layout import P  # noqa: E402  (study-relative paths -> case folders)
ASREVIEW_BIN = os.environ.get("ASREVIEW_BIN", "asreview")  # asreview==3.0.8 (requirements.lock in the study)
AIS = ["elas_u4", "elas_u3"]

def test_csv(rev, test_ids):
    """Test-pool CSV: sim runs ONLY on test records. STILL LEAKY (ranker learns
    test labels mid-sim) -> optimistic upper bound, NOT test-blind. True test-blind
    needs train on dev+holdout + predict on unseen test (ASReview predict API);
    out of scope for canary. Jev test numbers (dev tau, no test-label use) stay clean;
    bias direction favours ASReview, so a Jev>=ASReview claim is conservative."""
    import csv as _csv
    src = P(f"data/raw/synergy/{rev}.csv")
    dest = P(f"data/raw/synergy/{rev}.test.csv")
    with open(src) as f:
        rows = list(_csv.reader(f))
    head, body = rows[0], [r for r in rows[1:] if r[0] in test_ids]
    with open(dest, "w", newline="") as f:
        w = _csv.writer(f); w.writerow(head); w.writerows(body)
    return dest

def run_simulate_path(csvpath, tag, ai):
    out = P(f"runs/asreview/{tag}_{ai}.asreview")
    if os.path.exists(out):
        print(f"skip {out} exists"); return out
    cmd = [ASREVIEW_BIN, "simulate",
           csvpath,
           "--n-prior-included", "1", "--n-prior-excluded", "1",
           "--seed", "0", "--ai", ai, "--n-stop", "-1", "--output", out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    print((r.stdout or "")[-300:], (r.stderr or "")[-300:])
    if r.returncode != 0:
        print("FAILED", tag, ai); return None
    return out

def extract_order(apath, order_ids):
    """Screen order as openalex ids: results.record_id = int row in exported CSV."""
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(apath) as z:
        z.extractall(tmp)
    con = sqlite3.connect(os.path.join(tmp, "results.db"))
    rows = con.execute("SELECT record_id, label FROM results ORDER BY time, rowid").fetchall()
    con.close()
    return [(order_ids[rid], lab) for rid, lab in rows]

def metrics(order):
    n = len(order)
    n_incl = sum(l for _, l in order)
    found = 0; wss95 = None; curve = []
    for i, (_, l) in enumerate(order, 1):
        found += l
        rec = found / n_incl if n_incl else 0
        curve.append(rec)
        if wss95 is None and rec >= 0.95:
            wss95 = (n - i) / n - 0.05
    return {"n": n, "n_incl": n_incl, "wss95": wss95, "recall_curve": curve,
            "order": [rid for rid, _ in order],
            "order_labels": [l for _, l in order]}

def main():
    import csv as _csv
    os.makedirs(P("runs/asreview/"), exist_ok=True)
    splitpath = P(os.environ.get("X3_SPLIT", "data/processed/split.json"))
    with open(splitpath) as f:
        split = json.load(f)
    revs = list(split["reviews"])
    assert len({r.lower() for r in revs}) == len(revs), "lowercased review tags collide"
    for rev in split["reviews"]:
        test_ids = set(split["reviews"][rev]["ids"]["test"])
        # full-CSV upper bound (leaky: learns test labels mid-sim)
        full_csv = P(f"data/raw/synergy/{rev}.csv")
        with open(full_csv) as f:
            full_ids = [r[0] for r in list(_csv.reader(f))[1:]]
        for ai in AIS:
            tag = rev.lower()
            fout = {"review": rev, "ai": ai, "asreview": "3.0.8",
                    "estimand": "optimistic_upper_bound_leaky"}
            ap = run_simulate_path(full_csv, tag + "_full", ai)
            if ap:
                m = metrics(extract_order(ap, full_ids))
                recs_pos = {rid: i for i, rid in enumerate(m["order"])}
                torder = sorted([rid for rid in test_ids if rid in recs_pos],
                                key=lambda r: recs_pos[r])
                lab = {rid: l for rid, l in zip(m["order"], m["order_labels"])}
                tm = metrics([(rid, lab[rid]) for rid in torder])
                fout["full"] = {k: v for k, v in m.items() if k != "recall_curve"}
                fout["test_from_full"] = {k: v for k, v in tm.items()
                                          if k not in ("order", "order_labels", "recall_curve")}
            # test-pool sim (primary; still leaky, optimistic upper bound)
            tcsv = test_csv(rev, test_ids)
            with open(tcsv) as f:
                tids = [r[0] for r in list(_csv.reader(f))[1:]]
            ap2 = run_simulate_path(tcsv, tag + "_test", ai)
            if not ap2:
                continue
            m2 = metrics(extract_order(ap2, tids))
            fout["test_order"] = m2["order"]
            fout["test"] = {k: v for k, v in m2.items()
                            if k not in ("order", "order_labels", "recall_curve")}
            dest = P(f"runs/asreview/{tag}_{ai}.json")
            with open(dest, "w") as f:
                json.dump(fout, f)
            w = lambda v: f"{v:.3f}" if v is not None else "None"
            print(f"{rev} {ai}: test_n={m2['n']} test_incl={m2['n_incl']} test_wss95={w(m2['wss95'])}")
if __name__ == "__main__":
    main()
