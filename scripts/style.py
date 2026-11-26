"""House style for every merged-paper figure: import this, never set rcParams or colours locally."""
import json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J = Path(os.environ.get("MJ_STUDIES", Path(__file__).resolve().parents[1] / "data"))  # study files used by the paper: data/x2, data/x3, ...; collect_data.py sets MJ_STUDIES to the full studies/
OUT = Path(__file__).resolve().parents[1] / "figs"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.titlesize": 8, "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "axes.grid": True, "grid.alpha": .25, "legend.frameon": False, "legend.fontsize": 6.5,
                     "pdf.fonttype": 42})  # TrueType, not Type 3: journals require it and old Foxit stalls on Type 3
# Okabe-Ito; one meaning per colour in every figure
JEV, OTHER, REF, BASE = "#D55E00", "#0072B2", "#000000", "#999999"  # Jev, second decision model, established tool, no-model baseline
POSTHOC = "#56B4E9"  # comparator adjusted after results were seen (also hatch "///")
EXTRA = "#E69F00"    # Jev variant with added context
TRAINED = "#009E73"  # classifier trained on the task's own labels
WIDTH = 7.2          # full text width, inches


def save(f, name):
    f.tight_layout()
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        f.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=.03, **kw)
    plt.close(f)


def dump(key, obj):
    """Merge quoted numbers into figs/numbers.json under one top-level key."""
    p = OUT / "numbers.json"
    d = json.load(open(p)) if p.exists() else {}
    d[key] = obj
    json.dump(d, open(p, "w"), indent=1)
