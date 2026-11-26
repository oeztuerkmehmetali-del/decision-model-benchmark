"""Jev fan-out: ONE ask() per record, dev split first. Usage: fanout.py [dev|test|holdout] [REV...]."""
import json, os, sys

from layout import P, load_questions  # noqa: E402  (study-relative paths -> case folders; adds ../shared to sys.path)
import jevlib  # noqa: E402

CACHE = P(os.environ.get("X3_CACHE", "runs/jev/jev_cache.jsonl"))
TAG = os.environ.get("X3_RUNTAG", "jev")  # output prefix; "d3u" for the d3-ultra arm

def main():
    splitname = sys.argv[1] if len(sys.argv) > 1 else "dev"
    only = sys.argv[2:] or None
    qs = load_questions()
    items, keys = [], []
    for rev, r in qs.items():
        if only and rev not in only and rev.lower() not in [o.lower() for o in only]:
            continue
        clauses = r["clauses"]
        qnames = [f"clause_{i}" for i in range(len(clauses))] + ["include"]
        for rid, v in r["records"].items():
            if v["split"] != splitname:
                continue
            qq = {k: {"type": "noul", "instructions": inst}
                  for k, inst in
                  ([(f"clause_{i}", f"Does this record satisfy this eligibility clause? Clause: {c}")
                    for i, c in enumerate(clauses)] +
                   [("include", v_state_include(v["state"]))])}
            items.append((v["state"], qq))
            keys.append((rev, rid, v["label"]))
    print(f"fan-out {splitname}: {len(items)} records")
    resps = jevlib.ask_many(items, cache=CACHE, workers=int(os.environ.get("X3_WORKERS", "16")))
    out = {}
    for (rev, rid, lab), resp in zip(keys, resps):
        ans = resp.get("answers", {})
        row = {k: {"noul": a.get("noul"), "confidence": a.get("confidence")}
               for k, a in ans.items() if isinstance(a, dict)}
        out.setdefault(rev, {})[rid] = {"label": lab, "answers": row}
    shard = os.environ.get("X3_SHARD", "")
    suffix = f".{shard}" if shard else ""
    dest = P(f"runs/{TAG}/{TAG}_{splitname}{suffix}.json")  # TAG = arm folder (jev|d3u)
    if os.path.exists(dest):
        with open(dest) as f:
            prior = json.load(f)
        for rev, recs in prior.items():
            for rid, row in recs.items():
                out.setdefault(rev, {}).setdefault(rid, row)
    json.dump(out, open(dest, "w"))
    # 0b assert: every requested review present
    qs_revs = {rev for rev, r in qs.items()
               if (not only or rev in only or rev.lower() in [o.lower() for o in only])
               and any(v["split"] == splitname for v in r["records"].values())}
    missing = qs_revs - set(out)
    assert not missing, f"fan-out missing reviews in {dest}: {missing}"
    nq = sum(len(v["answers"]) for rev in out.values() for v in rev.values())
    print(f"wrote {dest}: {len(keys)} records this batch, {sum(len(v) for v in out.values())} total, {nq} nouls, cost ${jevlib.cost_usd():.4f} this-proc, cache {os.path.getsize(CACHE)/1e6:.1f}MB cumulative, usage {jevlib.USAGE}")

def v_state_include(state):
    return ("Overall: should this record be INCLUDED for full-text review? "
            f"Eligibility criteria: {state['criteria']}")

if __name__ == "__main__":
    main()
