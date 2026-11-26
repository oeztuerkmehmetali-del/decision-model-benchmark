"""Freeze stratified split for a review list -> data/<name>. Usage: freeze_split.py [list.json] [outname]."""
import hashlib, json, os, random, sys

from layout import P  # noqa: E402  (study-relative paths -> case folders)
from synergy_dataset import Dataset  # noqa: E402  (pip: synergy-dataset 2.2, network)
SEED = 0

def fracs_for(n_incl):
    """0d adaptive: small reviews get bigger test/holdout floors."""
    return (0.70, 0.15, 0.15) if n_incl >= 100 else (0.60, 0.20, 0.20)

def strat_split(ids0, ids1, seed, fracs):
    rng = random.Random(seed)
    out = {}
    for label, ids in ((0, ids0), (1, ids1)):
        rng.shuffle(ids)
        n = len(ids)
        n_dev = int(n * fracs[0]); n_test = int(n * fracs[1])
        out.setdefault("dev", []).extend(ids[:n_dev])
        out.setdefault("test", []).extend(ids[n_dev:n_dev + n_test])
        out.setdefault("holdout", []).extend(ids[n_dev + n_test:])
    for v in out.values():
        rng.shuffle(v)
    return out

def main():
    listfile = sys.argv[1] if len(sys.argv) > 1 else P("data/processed/reviews_union.json")  # list of the 39 review tags
    destname = sys.argv[2] if len(sys.argv) > 2 else "split.union.json"
    with open(listfile) as f:
        reviews = json.load(f)
    split = {"seed": SEED, "model": "jev-1.13.0", "reviews": {}}
    for i, rev in enumerate(reviews):
        d = Dataset(rev)
        recs = d.to_dict()
        ids0 = sorted(k for k, r in recs.items() if r.get("label_included") == 0)
        ids1 = sorted(k for k, r in recs.items() if r.get("label_included") == 1)
        fracs = fracs_for(len(ids1))
        seed = int(hashlib.sha256(f"{SEED}:{rev}".encode()).hexdigest()[:8], 16)
        parts = strat_split(ids0, ids1, seed, fracs)
        nincl = {k: sum(1 for _id in v if recs[_id].get("label_included") == 1)
                 for k, v in parts.items()}
        split["reviews"][rev] = {
            "n_total": len(recs), "n_incl": len(ids1),
            "fracs": {"dev": fracs[0], "test": fracs[1], "holdout": fracs[2]},
            "n": {k: len(v) for k, v in parts.items()},
            "n_incl_split": nincl,
            "ids": parts,
        }
        print(f"{rev}: total={len(recs)} incl={len(ids1)} fracs={fracs[0]:.2f}/{fracs[1]:.2f} "
              + " ".join(f"{k}={len(v)}(incl{nincl[k]})" for k, v in parts.items()))
    blob = json.dumps(split, sort_keys=True)
    split["split_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    dest = P("data/processed/" + destname)
    with open(dest, "w") as f:
        json.dump(split, f, indent=1)
    print("wrote", dest, split["split_sha256"])

if __name__ == "__main__":
    main()
