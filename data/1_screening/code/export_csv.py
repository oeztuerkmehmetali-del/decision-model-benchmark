"""Export per-review CSVs (title/abstract/label) from synergy to_dict for ASReview simulate."""
import csv, json, os, sys

from layout import P  # noqa: E402  (study-relative paths -> case folders)
from synergy_dataset import Dataset  # noqa: E402  (pip: synergy-dataset 2.2, network)

def main():
    splitpath = P(os.environ.get("X3_SPLIT", "data/processed/split.json"))
    with open(splitpath) as f:
        split = json.load(f)
    only = sys.argv[1:] or None
    outdir = P("data/raw/synergy/")
    os.makedirs(outdir, exist_ok=True)
    for rev in split["reviews"]:
        if only and rev not in only:
            continue
        d = Dataset(rev)
        recs = d.to_dict()
        dest = os.path.join(outdir, rev + ".csv")
        with open(dest, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["record_id", "title", "abstract", "label_included"])
            for rid, r in recs.items():
                w.writerow([rid, r.get("title", ""), r.get("abstract") or "",
                            r.get("label_included", "")])
        print(rev, "wrote", dest)

if __name__ == "__main__":
    main()
