"""PIN-row -> Jev question text. Leakage-safe by construction (allow-list, not ban-list).

Jev sees ONLY search-engine features. Never serialized: Label, SpecId (starts target_/decoy_),
Proteins (decoy_ prefix, species), Peptide (decoy peptides are shuffled -> detectable by
sequence "naturalness"; does not transfer to false target PSMs), any mokapot score/q/PEP.
Numbers are pre-bucketed (Jev jaggedness #2).
"""
import math

FEATURES = ("XCorr", "TailorScore", "deltCn", "deltLCn", "lnNumDSP", "lnNumSP", "absdM", "PepLen")
CHARGES = ("Charge1", "Charge2", "Charge3", "Charge4", "Charge5")
STATE = {"task": ("Tandem-MS peptide-spectrum match (PSM) quality control. Each question describes one "
                  "database-search PSM by search-engine features only. Estimate the probability that the "
                  "PSM is a correct identification.")}


def decile(value, sorted_ref):
    """1..10 rank bucket of value within a label-free reference distribution (ascending list)."""
    import bisect
    return 1 + min(9, int(10 * bisect.bisect_left(sorted_ref, value) / max(1, len(sorted_ref))))


HINTS = (" Guidance: higher XCorr/TailorScore and larger deltCn (gap to the next-best candidate) support a correct match;"
         " a precursor mass error near 0 Da supports it; fully tryptic termini support it. All PSMs here are near the"
         " 1% FDR cut-off, so roughly 1 in 5 is false: use the within-list percentiles to judge which are better than their neighbours.")
HINTS2 = (" Context: PSMs come from a database search against a target-decoy database. Only candidates near the 1% FDR"
          " cut-off are listed, where correct and incorrect matches overlap strongly and a substantial fraction are false."
          " Evidence FOR a correct match: high TailorScore/XCorr relative to the other listed PSMs; a clear margin to the"
          " runner-up (high deltCn/deltLCn); a small |precursor mass error|; fully tryptic termini. Evidence AGAINST: many"
          " competing candidate peptides (high lnNumDSP/lnNumSP), a low score for its length, a large mass error. Buckets"
          " compare this PSM with the other listed PSMs, not with all PSMs. Answer with the probability that this PSM is correct.")
NUM_FEATURES = (("XCorr", 2), ("TailorScore", 2), ("deltCn", 3), ("deltLCn", 3), ("lnNumDSP", 2), ("lnNumSP", 2),
                ("lnrSp", 2), ("lnExpect", 2), ("IonFrac", 3), ("Sp", 1), ("absdM", 4), ("PepLen", 0))
VARIANTS = ("base", "raw", "hints", "rel", "rel_hints", "rel_hints2", "sage_rel", "sage_rel_ev")

# Sage PIN columns (posterior_error dropped upstream: it is label-trained). (column, label, decimals)
SAGE_FEATURES = (("ln(hyperscore)", "ln hyperscore", 2), ("ln(delta_next)", "ln gap to next-best candidate (delta_next)", 2),
                 ("matched_peaks", "matched fragment peaks", 0),
                 ("longest_y_pct", "longest y run / peptide length", 2), ("ln(matched_intensity_pct)", "ln % of spectrum intensity matched", 2),
                 ("ln(-poisson)", "ln(-log10 Poisson score)", 2), ("fragment_ppm", "mean absolute fragment mass error (ppm)", 1),
                 ("ln(precursor_ppm)", "ln precursor mass error (ppm)", 2), ("sqrt(delta_rt_model)", "sqrt retention-time prediction error", 2),
                 ("scored_candidates", "candidate peptides scored for this spectrum", 0), ("peptide_len", "peptide length", 0),
                 ("missed_cleavages", "missed cleavages", 0))
EVIDENCE_FEATURES = (("top5_expl", "of the 5 most intense peaks explained by b/y ions", 0, 5), ("top10_expl", "of the 10 most intense peaks explained", 0, 10),
                     ("top20_expl", "of the 20 most intense peaks explained", 0, 20), ("tic_expl", "fraction of total ion current explained", 2, None),
                     ("b_frac", "fraction of possible b ions matched", 2, None), ("y_frac", "fraction of possible y ions matched", 2, None))
SAGE_HINT = (" Context: Sage search against a target-decoy database; only candidates near the 1% FDR cut-off are listed, so many are false."
             " A correct match usually explains the intense peaks of the spectrum with long contiguous b/y ladders (especially y);"
             " a false match typically matches many weak/noise peaks, scattered ions, few of the intense peaks. Buckets compare this PSM"
             " with the other listed PSMs (10 = highest). Answer with the probability that this PSM is correct.")


def describe(row, ref, variant="base"):
    """row: mapping of PIN columns -> number/str. ref: {'XCorr': sorted list, 'TailorScore': sorted list}."""
    f = lambda k: float(row[k])
    parts = []
    if variant.startswith("sage"):
        return _describe_sage(row, ref, variant.endswith("_ev"))
    if variant != "base":
        return _describe_v(row, ref, variant)
    for k in ("XCorr", "TailorScore"):
        if k in row:
            parts.append(f"{k}={f(k):.1f} (decile {decile(f(k), ref[k])}/10 among all top PSMs)")
    for k, nd in (("deltCn", 2), ("deltLCn", 2), ("lnNumDSP", 1), ("lnNumSP", 1)):
        if k in row:
            parts.append(f"{k}={f(k):.{nd}f}")
    if "absdM" in row:
        parts.append(f"|precursor mass error|={f('absdM'):.3f} Da")
    if "PepLen" in row:
        parts.append(f"peptide length={int(f('PepLen'))}")
    z = [c[-1] for c in CHARGES if c in row and float(row[c]) == 1.0]
    if z:
        parts.append(f"charge={z[0]}+")
    ends = [k for k in ("enzN", "enzC") if k in row and float(row[k]) == 1.0]
    parts.append(f"tryptic termini={len(ends)}/2")
    return "; ".join(parts)


def build_questions(rows, ref, prefix="psm", variant="base", qtype="noul"):
    """rows: list of (id, row). One question per PSM. qtype noul -> probability; choice -> hard yes/no verdict."""
    out = {}
    for i, r in rows:
        ins = ("Is this PSM a correct identification? " + describe(r, ref, variant)
               + (SAGE_HINT if variant.startswith("sage") else HINTS2 if variant.endswith("hints2") else HINTS if variant.endswith("hints") else ""))
        out[f"{prefix}_{i}"] = ({"type": "noul", "instructions": ins} if qtype == "noul"
                                else {"type": "choice", "instructions": ins, "criteria": {"yes": None, "no": None}})
    return out


def _describe_v(row, ref, variant):
    """Ablation variants. raw: undiscretised numbers. rel*: deciles within the candidate band (label-free).
    Only allow-listed NUM_FEATURES present in the PIN are used (Tide vs Comet column sets differ)."""
    f = lambda k: float(row[k])
    rel = variant.startswith("rel")
    parts = []
    for k, nd in NUM_FEATURES:
        if k not in row:
            continue
        txt = f"{k}={f(k):.{nd}f}"
        if rel:
            txt += f" (bucket {decile(f(k), ref['band'][k])}/10 among the listed PSMs, 10 = highest value)"
        parts.append(txt)
    z = [c[-1] for c in CHARGES if c in row and float(row[c]) == 1.0]
    if z:
        parts.append(f"charge={z[0]}+")
    parts.append(f"tryptic termini={sum(float(row[k]) == 1.0 for k in ('enzN', 'enzC') if k in row)}/2")
    return "; ".join(parts)


def _describe_sage(row, ref, with_ev):
    """Sage features (band-relative buckets) [+ fragment-ion evidence beyond the PIN]. Allow-list only."""
    parts = []
    for k, label, nd in SAGE_FEATURES:
        if k in row:
            v = float(row[k]); parts.append(f"{label}={v:.{nd}f} (bucket {decile(v, ref['band'][k])}/10)")
    z = [c for c in ("z=2", "z=3", "z=4", "z=5", "z=6") if c in row and float(row[c]) == 1.0]
    if z:
        parts.append(f"charge={z[0][-1]}+")
    if with_ev:
        for k, label, nd, of in EVIDENCE_FEATURES:
            v = float(row[k]); b = f" (bucket {decile(v, ref['band'][k])}/10)" if k in ref["band"] else ""
            parts.append(f"{int(v)}/{of} {label}" if of else f"{label}={v:.{nd}f}{b}")
        parts.append(f"matched ion ladders (ordinals from the N-terminus for b, C-terminus for y): {row['ladder']}")
    return "; ".join(parts)
