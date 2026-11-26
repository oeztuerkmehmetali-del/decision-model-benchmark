"""Case-folder layout for 1_screening. The study scripts were written for the x3 tree (data/, runs/, results/);
P() maps those study-relative paths to the role folders here, so every script keeps its original path strings.
Importing this module also puts ../shared (jevlib.py, the shared model client) on sys.path."""
import gzip, json, os, sys

CASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # paper/data/1_screening
sys.path.insert(0, os.path.join(CASE, "..", "shared"))
sys.path.insert(0, os.path.join(CASE, "code"))

# (study-relative prefix, case-folder prefix); first match wins
_MAP = [
    ("data/processed/questions.json", "input/questions.json"),  # 281 MB: rebuilt from input/questions/*.json.gz, see load_questions()
    ("data/processed/", "input/"),
    ("data/raw/synergy/", "input/synergy/"),
    ("runs/jev/", "jev/"),
    ("runs/d3u/", "d3u/"),
    ("runs/asreview/", "baseline/"),
    ("results/jev/results_test.union39.json", "compare/results_test.union39.json"),  # Jev scores + ASReview columns
    ("results/jev/", "jev/"),
    ("results/results_d3u.json", "d3u/results_d3u.json"),
    ("results/d3u_numbers.md", "d3u/d3u_numbers.md"),
    ("results/", "compare/"),
]


def P(rel):
    rel = rel.replace(os.sep, "/")
    for a, b in _MAP:
        if rel.startswith(a):
            return os.path.join(CASE, b + rel[len(a):])
    raise KeyError(f"no layout rule for {rel}")


def load_questions():
    """questions.json (review -> {criteria, clauses, records}) reassembled from the per-review slices, in split order."""
    full = P("data/processed/questions.json")
    if os.path.exists(full):
        return json.load(open(full))
    out = {}
    for rev in json.load(open(P("data/processed/reviews_union.json"))):
        with gzip.open(os.path.join(CASE, "input", "questions", rev + ".json.gz"), "rt", encoding="utf-8") as f:
            out.update(json.load(f))
    return out
