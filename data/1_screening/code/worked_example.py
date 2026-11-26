"""Writes ../input/worked_example.json: the exact request (state + questions) sent to the model for one dev record,
rebuilt from input/questions with the same templates as fanout.py, plus the stored Jev 1.13 and d3-ultra answers.
Usage: python3 worked_example.py [Review_tag] [record_id]   (default: shortest-abstract positive dev record)"""
import json, sys
from layout import P, load_questions

qs = load_questions()
rev = sys.argv[1] if len(sys.argv) > 1 else "Ali_2024"
recs = qs[rev]["records"]
if len(sys.argv) > 2:
    rid = sys.argv[2]
else:
    rid = min((r for r, v in recs.items() if v["split"] == "dev" and v["label"] == 1 and v["state"]["abstract"].startswith("[") is False),
              key=lambda r: len(recs[r]["state"]["abstract"]))
v, clauses = recs[rid], qs[rev]["clauses"]
questions = {f"clause_{i}": {"type": "noul", "instructions": f"Does this record satisfy this eligibility clause? Clause: {c}"}
             for i, c in enumerate(clauses)}
questions["include"] = {"type": "noul", "instructions": "Overall: should this record be INCLUDED for full-text review? "
                        f"Eligibility criteria: {v['state']['criteria']}"}
out = {"review": rev, "record_id": rid, "split": v["split"], "label_included": v["label"],
       "request": {"state": v["state"], "questions": questions},
       "jev_1.13_answers": json.load(open(P("runs/jev/jev_dev.json")))[rev][rid]["answers"],
       "d3_ultra_answers": json.load(open(P("runs/d3u/d3u_dev.json")))[rev][rid]["answers"]}
json.dump(out, open(P("data/processed/worked_example.json"), "w"), indent=1, ensure_ascii=False)
print(rev, rid, len(json.dumps(out)), "bytes")
