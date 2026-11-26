"""Write or check code/MANIFEST.json: sha256 of every input and stored model answer the case depends on, plus model/software provenance.
  python3 code/manifest.py          write code/MANIFEST.json   (from paper/data/2_drug/)
  python3 code/manifest.py --check  exit 1 if any listed file changed or is missing
"""
import hashlib, json, os, platform, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # case folder (paper/data/2_drug)
MAN = os.path.join(HERE, "code", "MANIFEST.json")
TRACK = ["code/run.py", "code/analyze.py", "baseline/paper_reference.csv", "../shared/jevlib.py"]
for d in ("input", "jev", "d3u", "code/drugcon"):
    for root, dirs, files in os.walk(os.path.join(HERE, d)):
        dirs[:] = [x for x in dirs if x != "__pycache__"]  # interpreter-specific bytecode
        TRACK += [os.path.relpath(os.path.join(root, f), HERE) for f in files]


def sha(p):
    with open(os.path.join(HERE, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


if "--check" in sys.argv:
    m = json.load(open(MAN))
    bad = [p for p, h in m["files"].items() if not os.path.exists(os.path.join(HERE, p)) or sha(p) != h]
    print("MANIFEST check:", "OK" if not bad else f"{len(bad)} mismatched: {bad}")
    sys.exit(1 if bad else 0)

import matplotlib, numpy, pandas  # noqa: E402
json.dump({
    "study": "x8 DrugCon-Bench: Jev 1.13 and d3-ultra as contradiction judges",
    "dataset": {"name": "DrugCon-Bench", "repo": "https://github.com/MengfeiLan/DrugCon_Drug_Contradiction",
                "commit": "2c2d5f5f4542cee0925a6b6b398694cc2433083d", "code_licence": "MIT (input/LICENSE)",
                "paper": "Lan et al. 2026, medRxiv doi:10.64898/2026.11.11.26367856 (posted 2026-11-12)",
                "copied": "input/eval_sets/, input/prompts.md, input/LICENSE, input/CITATION.cff, code/drugcon/{__init__,evaluation}.py"},
    "models": {"jev": {"id": "jev-1.13.0", "endpoint": "https://api.typesafe.ai", "release": "2026-09-15"},
               "d3u": {"id": "vllm-sr/d3-ultra", "endpoint": "https://decision.vllm-sr.ai",
                       "note": "pinned 'vllm-sr/d3' is rejected by the endpoint; d3-ultra answered"}},
    "runs": {"main": "2026-11-20, full sets, logs <model>/log_<task>__<model>.txt",
             "repeat": "compare/rep1_started_utc.txt; seeded subsets (pandas sample random_state=0): detection n=200, explanation n=50"},
    "decision_rules_fixed_before_runs": {"detection": "argmax of typed choice", "explanation": "factor kept if p>=0.5"},
    "software": {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
                 "matplotlib": matplotlib.__version__},
    "rebuild": "bash code/reproduce.sh --verify",
    "files": {p: sha(p) for p in sorted(TRACK)},
}, open(MAN, "w"), indent=1)
print("wrote code/MANIFEST.json with", len(TRACK), "files")
