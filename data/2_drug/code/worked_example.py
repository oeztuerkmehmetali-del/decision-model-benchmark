"""Write input/worked_example.json: the exact request (state + questions) code/run.py sends for the first detection pair and
the first explanation pair, with the stored Jev 1.13 and d3-ultra answers found by the sha256 cache key. Offline.
Usage (from paper/data/2_drug/):  python3 code/worked_example.py"""
import json, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(HERE, "..", "shared"), os.path.join(HERE, "code")]
import jevlib  # noqa: E402
from drugcon.evaluation import POSITIVE, NEGATIVE, CANONICAL_FACTORS  # noqa: E402

src = open(os.path.join(HERE, "code", "run.py")).read()
exec(src[src.index("DET_STATE ="):src.index("f = os.path.join(HERE")])  # the templates det_q / exp_q, unchanged
MODEL = {"jev": "jev-1.13.0", "d3u": "vllm-sr/d3-ultra"}


def answers(st, q):
    out = {}
    for m, mid in MODEL.items():
        store = jevlib._load(os.path.join(HERE, m, f"cache_{m}.jsonl"))
        out[m] = store[jevlib._key(st, q, mid)]["answers"]
    return out


ex = {}
r = next(pd.read_csv(os.path.join(HERE, "input/eval_sets/detection.csv")).itertuples())
st, q = det_q(r)
ex["detection"] = {"pair_id": int(r.pair_id), "gold_label": r.gold_label, "state": st, "questions": q, "answers": answers(st, q)}
r = next(pd.read_csv(os.path.join(HERE, "input/eval_sets/explanation.csv")).itertuples())
st, q = exp_q(r)
ex["explanation"] = {"id": int(r.id), "gold_factors": r.human_labeled_contextual_factors, "state": st, "questions": q,
                     "answers": answers(st, q)}
with open(os.path.join(HERE, "input/worked_example.json"), "w") as fh:
    json.dump(ex, fh, indent=1, ensure_ascii=False)
print("wrote input/worked_example.json")
