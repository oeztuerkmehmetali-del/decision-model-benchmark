"""x8 DrugCon-Bench: typed decision models (Jev 1.13, d3-ultra) on contradiction detection + explanation.

Usage (from paper/data/2_drug/):  python3 code/run.py --model jev|d3u --task detection|explanation [--limit N] [--tag _rep1]
Writes <model>/<task>__<model>.tsv (one row per pair, gold + probabilities) and prints metrics.
Task wording paraphrases the authors' prompts (counterfactual_appendix_v2, taxonomy_v1; see input/prompts.md);
scoring = the authors' evaluation.py (binary_prf, multilabel_metrics), so numbers are comparable to the paper's tables.
Decision rule fixed before any run: detection = argmax choice; explanation = factor kept if p >= 0.5.
"""
import argparse, ast, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # case folder (paper/data/2_drug)
sys.path[:0] = [os.path.join(HERE, "..", "shared"), os.path.join(HERE, "code")]  # jevlib; drugcon (authors' scorer)
MODELS = {"jev": ("https://api.typesafe.ai", "TYPESAFE_API_KEY", "jev-1.13.0"),
          "d3u": ("https://decision.vllm-sr.ai", "VLLM_SR_API_KEY", "vllm-sr/d3-ultra")}
a = argparse.ArgumentParser()
a.add_argument("--model", choices=MODELS, required=True)
a.add_argument("--task", choices=("detection", "explanation"), required=True)
a.add_argument("--limit", type=int, default=0)
a.add_argument("--workers", type=int, default=4)
a.add_argument("--tag", default="", help="suffix for cache + output; a new tag forces fresh API answers (repeatability run)")
a = a.parse_args()
os.environ["TYPESAFE_BASE_URL"], os.environ["JEV_KEY_ENV"], MODEL = MODELS[a.model]
import jevlib  # noqa: E402  (reads the env above at import)
from drugcon.evaluation import POSITIVE, NEGATIVE, CANONICAL_FACTORS, binary_prf, bootstrap_f1_ci, multilabel_metrics  # noqa: E402

DET_STATE = {"task": (
    "Decide whether two biomedical claims about the same drug-disease association present a contradiction, using "
    "counterfactual reasoning: assume Claim 1 is true and ask whether Claim 2's overall assessment of the drug for the "
    "disease still holds; then the reverse. A contradiction here is an apparent contradiction: the two claims give "
    "opposing overall assessments of the same drug for the same disease (effective vs. ineffective, beneficial vs. "
    "harmful, safe vs. unsafe); it need not be a strict logical impossibility. A contradiction may be partial: if one "
    "subclaim opposes the other claim, the pair is contradictory even if other parts agree. Do NOT dismiss a "
    "contradiction merely because the studies differ in outcome, patient subgroup, dose, duration, timing, comparator "
    "or study design.")}
DET_CRITERIA = {
    POSITIVE: "The overall assessments of the drug for the disease oppose each other, fully or in part.",
    NEGATIVE: "The claims concern different drugs or diseases, express the same overall direction, or make no clear assessment of the drug-disease association."}
EXP_STATE = {"task": (
    "Two biomedical claims report opposing outcomes for the same drug-disease association. Determine which contextual "
    "factor(s) explain the apparent contradiction. Base the analysis strictly on the provided claims and abstracts; do "
    "not use external knowledge.")}


def det_q(r):
    return DET_STATE, {"pair": {"type": "choice", "criteria": DET_CRITERIA,
                                "instructions": f"Claim 1: {r.claim_1}\nClaim 2: {r.claim_2}"}}


def exp_q(r):  # claims + abstracts go in the state (read once), questions stay short
    st = dict(EXP_STATE, claim_1=r.claim_1, claim_1_context=r.claim_abs_1, claim_2=r.claim_2, claim_2_context=r.claim_abs_2)
    return st, {f"f{i}": {"type": "noul", "instructions": f"Does the contextual factor '{f}' explain the apparent contradiction between the two claims?"}
                for i, f in enumerate(CANONICAL_FACTORS)}


f = os.path.join(HERE, "input", "eval_sets", f"{a.task}.csv")
d = pd.read_csv(f)
if a.limit:
    d = d.sample(n=a.limit, random_state=0).reset_index(drop=True)
det = a.task == "detection"
os.makedirs(os.path.join(HERE, a.model), exist_ok=True)
resps = jevlib.ask_many([(det_q if det else exp_q)(r) for r in d.itertuples()],
                        cache=os.path.join(HERE, a.model, f"cache_{a.model}{a.tag}.jsonl"), workers=a.workers, model=MODEL)
tag = f"{a.task}__{a.model}" + (f"_limit{a.limit}" if a.limit else "") + a.tag
if det:
    ans = [r["answers"]["pair"] for r in resps]
    d["p_contra"] = [x["probabilities"][POSITIVE] for x in ans]
    d["pred"] = [x["choice"] for x in ans]
    m = binary_prf(d.gold_label, d.pred)
    lo, hi = bootstrap_f1_ci(d.gold_label, d.pred)
    print(f"{MODEL} detection n={m['n']} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} [{lo:.3f},{hi:.3f}] acc {m['accuracy']:.3f}  tp/fp/fn/tn {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}")
    cols = ["pair_id", "pmid_1", "pmid_2", "gold_label", "pred", "p_contra"]
else:
    for i, fac in enumerate(CANONICAL_FACTORS):
        d[f"p_{fac}"] = [r["answers"][f"f{i}"]["noul"] for r in resps]
    d["pred"] = [str([fac for fac in CANONICAL_FACTORS if row[f"p_{fac}"] >= 0.5]) for _, row in d.iterrows()]
    t = multilabel_metrics(d.human_labeled_contextual_factors, d.pred)
    print(f"{MODEL} explanation n={len(d)}"); print(t.round(3).to_string())
    cols = ["id", "pmid_1", "pmid_2", "human_labeled_contextual_factors", "pred"] + [f"p_{fac}" for fac in CANONICAL_FACTORS]
d[cols].to_csv(os.path.join(HERE, a.model, f"{tag}.tsv"), sep="\t", index=False)
print("USAGE", jevlib.USAGE, f"tokens->cost at Jev price ${jevlib.cost_usd():.4f}")
