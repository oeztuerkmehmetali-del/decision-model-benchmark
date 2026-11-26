"""d3 verdict on every Tier B PSM (vllm-sr/d3-ultra; NOT jev-1.13), full context, labels masked.
Run from anywhere:  X3_MODEL=vllm-sr/d3-ultra python code/run_d3_all.py [--smoke]   (all answers cached in d3u/cache_all.jsonl: offline replay)
The model gets the experiment and search description (CONTEXT) and, per PSM, every number mokapot gets
(the paper's sage_rel_ev text + the six mokapot features that text leaves out). Masked: target/decoy status and
its proxies - peptide sequence, protein, species (the entrapment truth) - and mokapot's own score.
PSMs go in random order with neutral keys, so order and keys carry no information.
d3_p_thin = answers to the paper's thin-context wording (ps.STATE + SAGE_HINT) for the PSMs an earlier, stopped
run already scored; read from the cache only, never requested -> paired context-effect check in master_eval.py.
Writes compare/d3_all.tsv and d3u/sage_d3.pin (mokapot input with d3 as extra features). Resumable (cache).
Env: TYPESAFE_BASE_URL, JEV_KEY_ENV, X3_MODEL, JEV_TOKEN_CAP.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(os.path.dirname(ROOT), "shared")]  # shared model client: paper/data/shared/jevlib.py
import jevlib, psm_serializer as ps  # noqa: E402
from run_boundary import read_pin  # noqa: E402

IN, JEV, D3U, CMP = (os.path.join(ROOT, d) for d in ("input", "jev", "d3u", "compare")); CACHE = os.path.join(D3U, "cache_all.jsonl"); BATCH = 16

# Frozen before any full-context answer was seen (see the commit that introduced it). Sources: PRIDE PXD001468
# sample/acquisition protocol (Chick et al. 2015, doi 10.1038/nbt.3267), b1931.mzXML header, tierB_sage.json, paper_v2 methods.
CONTEXT = {
    "task": "Quality control of peptide-spectrum matches (PSMs) from one shotgun-proteomics run. Each question describes one PSM. "
            "Judge whether the peptide the search engine assigned is the peptide that produced the spectrum.",
    "sample": "Human HEK293T (293T) cell lysate. Proteins reduced with DTT, alkylated with iodoacetamide, digested with Lys-C and then "
              "trypsin; peptides separated into 24 basic-pH reversed-phase fractions. This run is one of those fractions.",
    "acquisition": "Thermo Q Exactive, nano-LC with a 3 h gradient, data-dependent top-20 method. MS1 in the Orbitrap at 70,000 "
                   "resolution (300-1500 m/z). MS2 by HCD, 2 m/z isolation window, read in the Orbitrap at 17,500 resolution; "
                   "centroided spectra. 55,352 MS2 spectra in the run.",
    "search": "Sage 0.14.6. Database: 20,416 human and 16,345 Arabidopsis thaliana SwissProt proteins plus reversed decoy sequences. "
              "No plant material is in the sample, so a match to an Arabidopsis-only or a decoy sequence is necessarily wrong. "
              "Trypsin (after K/R, not before P), up to 2 missed cleavages, peptide length 6-30, peptide mass 500-5000 Da; fixed "
              "carbamidomethyl-Cys, variable Met oxidation (max 2); precursor and fragment tolerance 10 ppm; isotope offsets -1 to +3; "
              "b and y ions, fragment charge up to 2; spectra deisotoped, 150 most intense peaks kept; only the top-ranked peptide per "
              "spectrum is reported.",
    "listed_psms": "All 46,191 top-ranked PSMs of the run are listed, unfiltered and in random order. From the number of decoy hits "
                   "(9,130), about 40% of the listed PSMs are expected to be wrong and about 60% correct.",
    "not_shown": "Peptide sequence, protein, species and target/decoy status are withheld for every PSM. Judge from the match evidence.",
    "how_to_read": "A correct match usually explains the intense peaks of the spectrum with long contiguous b/y ladders (especially y); "
                   "a false match typically matches many weak/noise peaks, scattered ions, few of the intense peaks. Buckets compare "
                   "this PSM with all listed PSMs (1 = lowest, 10 = highest).",
}
CRITERIA = {"yes": "the assigned peptide produced this spectrum: a correct identification",
            "no": "the assigned peptide did not produce this spectrum: a wrong match"}
# mokapot features (sage_nopep.pin) absent from the sage_rel_ev text; the other unused PIN features are constant in this run
EXTRA = (("retentiontime", "retention time (min)", 1), ("aligned_rt", "retention time on a 0-1 scale", 2),
         ("predicted_rt", "predicted retention time on the same scale", 2), ("isotope_error", "precursor isotope offset (Da)", 0),
         ("longest_b", "longest contiguous b run (ions)", 0), ("longest_y", "longest contiguous y run (ions)", 0))


def ask(qs):
    for _ in range(60):  # free endpoint has a tokens-per-minute limit: wait it out instead of dying
        try:
            return jevlib.ask(CONTEXT, qs, cache=CACHE)
        except RuntimeError as e:
            if "429" not in str(e):
                raise
            time.sleep(60)
    raise RuntimeError("rate limit did not clear in 60 min")


def main():
    a = argparse.ArgumentParser(); a.add_argument("--smoke", action="store_true"); a = a.parse_args()
    d = pd.read_csv(os.path.join(JEV, "tB_all_choice.tsv"), sep="\t").astype({"SpecId": str})  # paper row order + jev-1.13 answers
    pin = read_pin(f"{IN}/sage_nopep.pin"); pin["SpecId"] = pin.SpecId.astype(str)  # same load as run_boundary.py
    pin = pin.merge(pd.read_csv(f"{IN}/evidence.tsv", sep="\t").rename(columns={"psm_id": "SpecId"}).astype({"SpecId": str}), on="SpecId", how="left")
    feats = pin.drop_duplicates("SpecId").set_index("SpecId").loc[d.SpecId]
    cols = [k for k, *_ in ps.SAGE_FEATURES] + [k for k, *_ in ps.EVIDENCE_FEATURES]
    ref = {"band": {k: sorted(feats[k].astype(float)) for k in cols if k in feats}}
    n = len(d)
    text = lambda r: (ps.describe(r, ref, "sage_rel_ev") + "; "
                      + "; ".join(f"{label}={float(r[k]):.{nd}f}" for k, label, nd in EXTRA))
    full = lambda b: {f"psm_{k}": {"type": "choice", "instructions": "Is this PSM a correct identification? " + text(feats.iloc[i]),
                                   "criteria": CRITERIA} for k, i in b}
    thin = lambda b: ps.build_questions([(k, feats.iloc[i]) for k, i in b], ref, variant="sage_rel_ev", qtype="choice")

    perm = np.random.default_rng(20261119).permutation(n)  # key j = position in the shuffle, PSM = perm[j]
    shuf = [[(j, int(perm[j])) for j in range(b, min(b + BATCH, n))] for b in range(0, n, BATCH)]
    if a.smoke:
        shuf = shuf[:2]
        print(json.dumps(CONTEXT, indent=1)); print(json.dumps(next(iter(full(shuf[0]).values())), indent=1))
    s, t0 = {}, time.time()
    for m, b in enumerate(shuf, 1):  # one request at a time: the endpoint's token-per-minute limit is shared with the omp guards
        r = ask(full(b)); s.update((i, r["answers"][f"psm_{k}"]) for k, i in b)
        if m % 100 == 0:
            print(f"full-context {m}/{len(shuf)} batches  {time.time() - t0:.0f}s  retries {jevlib.USAGE['retries']}", file=sys.stderr, flush=True)
    done = jevlib._load(CACHE)
    old = {i: done[key]["answers"][f"psm_{k}"] for b in shuf for key in [jevlib._key(ps.STATE, thin(b), jevlib.MODEL)] if key in done for k, i in b}
    d["d3_p"] = [s[i]["probabilities"]["yes"] if i in s else np.nan for i in range(n)]
    d["d3_yes"] = [int(s[i]["choice"] == "yes") if i in s else np.nan for i in range(n)]
    d["d3_p_thin"] = [old[i]["probabilities"]["yes"] if i in old else np.nan for i in range(n)]
    print(f"model {jevlib.MODEL}  requests {jevlib.USAGE['requests']} cache_hits {jevlib.USAGE['cache_hits']} retries {jevlib.USAGE['retries']} "
          f"input_tokens {jevlib.USAGE['input_tokens']}  scored {int(d.d3_p.notna().sum())}/{n}  thin-context answers reused {int(d.d3_p_thin.notna().sum())}")
    if a.smoke:
        print(d[d.d3_p.notna()][["SpecId", "is_decoy", "jev_p", "d3_p", "d3_p_thin"]].head(8).to_string()); return
    d.to_csv(os.path.join(CMP, "d3_all.tsv"), sep="\t", index=False)
    # mokapot input: the 30-feature PIN + d3_p, d3_yes (same construction as jev/sage_jev.pin)
    add = d.set_index("SpecId")[["d3_p", "d3_yes"]]
    with open(f"{IN}/sage_nopep.pin") as f, open(os.path.join(D3U, "sage_d3.pin"), "w") as g:
        hdr = f.readline().rstrip("\n").split("\t"); k = hdr.index("Peptide")
        g.write("\t".join(hdr[:k] + ["d3_p", "d3_yes"] + hdr[k:]) + "\n")
        for line in f:
            x = line.rstrip("\n").split("\t")
            g.write("\t".join(x[:k] + [f"{add.d3_p[x[0]]:.6f}", str(int(add.d3_yes[x[0]]))] + x[k:]) + "\n")


if __name__ == "__main__":
    main()
