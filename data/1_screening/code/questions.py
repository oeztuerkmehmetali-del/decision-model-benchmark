"""Build Jev state + questions per record: verbatim criteria + one noul per clause + overall include noul."""
import json, os, re, sys

from layout import P  # noqa: E402  (study-relative paths -> case folders)
from synergy_dataset import Dataset  # noqa: E402  (pip: synergy-dataset 2.2, network)

def split_clauses(criteria):
    """Split criteria text into clauses on sentence/;/bullet boundaries; keep verbatim."""
    parts = re.split(r"(?<=[.;])\s+|\n+", criteria.strip())
    return [p.strip(" ;-") for p in parts if len(p.strip()) > 20][:8]

def build_questions(clauses, criteria):
    qs = {}
    for i, c in enumerate(clauses):
        qs[f"clause_{i}"] = {"type": "noul",
            "instructions": f"Does this record satisfy this eligibility clause? Clause: {c}"}
    qs["include"] = {"type": "noul",
        "instructions": ("Overall: should this record be INCLUDED for full-text review? "
                         f"Eligibility criteria: {criteria}")}
    return qs

def build_state(title, abstract, criteria):
    state = {"title": title or "", "criteria": criteria}
    state["abstract"] = abstract or "[no abstract; decide on title alone]"
    return state

def main():
    splitpath = P(os.environ.get("X3_SPLIT", "data/processed/split.json"))
    with open(splitpath) as f:
        split = json.load(f)
    out = {}
    for rev in split["reviews"]:
        d = Dataset(rev)
        criteria = d.metadata["publication"]["eligibility_criteria"]
        clauses = split_clauses(criteria)
        recs = d.to_dict()
        items = {}
        for splitname, ids in split["reviews"][rev]["ids"].items():
            for rid in ids:
                r = recs[rid]
                items[rid] = {"split": splitname,
                              "state": build_state(r.get("title"), r.get("abstract"), criteria),
                              "label": r.get("label_included")}
        out[rev] = {"criteria": criteria, "criteria_len": len(criteria),
                    "clauses": clauses, "n_questions": len(clauses) + 1,
                    "records": items}
        nq = len(clauses) + 1
        print(f"{rev}: {len(clauses)} clauses + overall = {nq} nouls/record, {len(items)} records")
    dest = P("data/processed/questions.json")
    keep = set(split["reviews"])  # staged merges stay within current split; locked-5 never dragged along
    out = {rev: v for rev, v in out.items() if rev in keep}
    if os.path.exists(dest):
        with open(dest) as f:
            prior = json.load(f)
        for rev, recs in prior.items():
            if rev in keep and rev not in out:
                out[rev] = recs
    with open(dest, "w") as f:
        json.dump(out, f)
    print("wrote", dest, f"{os.path.getsize(dest)/1e6:.1f} MB", f"{len(out)} reviews")

if __name__ == "__main__":
    main()
