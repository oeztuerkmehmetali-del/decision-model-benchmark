"""Build feature-matched PINs: only the Jev allow-list features (psm_serializer.SAGE_FEATURES [+ EVIDENCE_FEATURES]) + charge dummies.
Output: baseline/sage_fm.pin, baseline/sage_fm_ev.pin  (run mokapot on them with the same flags as tB). Run from anywhere."""
import pandas as pd
from pathlib import Path
from psm_serializer import SAGE_FEATURES, EVIDENCE_FEATURES
CASE = Path(__file__).resolve().parent.parent
T, OUT = CASE / "input", CASE / "baseline"
base = pd.read_csv(T / "sage_ev.pin", sep="\t")
meta = ["SpecId", "Label", "ScanNr", "ExpMass", "CalcMass", "FileName", "Peptide", "Proteins"]
feat = [k for k, *_ in SAGE_FEATURES] + ["z=2", "z=3", "z=4", "z=5", "z=6", "z=other"]
ev = [k for k, *_ in EVIDENCE_FEATURES]
for name, cols in (("sage_fm", feat), ("sage_fm_ev", feat + ev)):
    d = base[[c for c in meta if c != "Peptide" and c != "Proteins"] + cols + ["Peptide", "Proteins"]]
    d.to_csv(OUT / f"{name}.pin", sep="\t", index=False); print(name, len(cols), "features")
