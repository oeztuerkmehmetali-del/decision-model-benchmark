"""x2 serializer unit test (TODO 0p-11): Jev PSM payload must carry NO leakage.

Ban list:
  - LabelClass / decoy flag / label column of any kind
  - Any baseline-derived score: Percolator/Mokapot q-value, PEP, SVM score, PosteriorError
  - Target/decoy marker itself (protein suffix _target/_decoy)

Run: python3 code/serializer_test.py  (exit 0 = pass)
"""
import json
import sys

BANNED_KEYS = {
    "LabelClass", "label", "decoy", "is_decoy", "q_value", "qvalue", "PEP",
    "pep", "posterior_error", "score", "svm_score", "PosteriorError_Probability",
    "Label", "FDP", "FDR", "n_p", "n_t", "FDP_1B", "index", "vt",
}
BANNED_SUBSTRINGS = ("_decoy", "decoy", "percolator", "mokapot", "q-value", "qvalue")


def serialize_psm(psm):
    """Convert one percolator-format PSM row -> Jev questions + state.

    state = text features the decision primitve may legally see.
    questions = {"accept": {"type": "noul", "instructions": ...}}
    """
    allowed = ("peptide", "mod_peptide", "charge", "ScanNr", "title", "precursor_mz", "retention")
    state = {k: str(psm[k]) for k in allowed if k in psm}
    state = json.loads(json.dumps(state))  # round-trip catches non-JSON-leaks
    questions = {"accept": {"type": "noul",
        "instructions": "Given the PSM features, should this PSM be accepted at standard FDR?"}}
    return state, questions


def test_no_banned_keys():
    psm = {"peptide": "PEPTIDEK", "mod_peptide": "PEPT[80]IDEK", "charge": "2",
           "ScanNr": "42", "LabelClass": "5", "q_value": "0.01", "decoy": "0",
           "score": "0.9", "FDP": "0.0"}
    state, questions = serialize_psm(psm)
    flat = json.dumps(state) + json.dumps(questions)
    banned = [k for k in BANNED_KEYS if f'"{k}"' in flat]
    assert not banned, f"leaked banned keys: {banned}"


def test_no_decoy_marker_substring():
    psm = {"peptide": "PEPTIDEK", "charge": "2",
           "protein": "sp|ABCDE_decoy|ABCDE_decoy;"}
    state, questions = serialize_psm(psm)
    flat = json.dumps(state) + json.dumps(questions).lower()
    for s in BANNED_SUBSTRINGS:
        assert s not in flat.lower(), f"leaked substring {s!r}"


def test_questions_shape_valid():
    state = {"peptide": "PEPTIDEK", "charge": "2"}
    qs = {"accept": {"type": "noul", "instructions": "accept?"}}
    flat = json.dumps(qs)
    assert '"type"' in flat and '"noul"' in flat, "questions must keep typed noul shape"
    assert '"instructions"' in flat, "questions must have instructions"


def test_roundtrip_lossless():
    psm = {"peptide": "PEPT[80]IDEK", "charge": "3", "ScanNr": "7"}
    state, questions = serialize_psm(psm)
    assert state["peptide"] == "PEPT[80]IDEK" and state["charge"] == "3"


def test_pin_serializer_allowlist_no_leak():
    import psm_serializer as ps
    row = {"Label": "-1", "SpecId": "decoy_0_1_2_1", "Proteins": "decoy_tr|X|X_CHLSB",
           "Peptide": "K.AAAAAAK.E", "mokapot q-value": "0.01", "XCorr": "1.234", "TailorScore": "1.1",
           "deltCn": "0.1", "absdM": "0.004", "PepLen": "7", "Charge2": "1", "enzN": "1", "enzC": "1"}
    ref = {"XCorr": [0.5, 1.0, 2.0], "TailorScore": [0.5, 1.0, 2.0]}
    qs = ps.build_questions([(0, row)], ref)
    flat = json.dumps(qs) + json.dumps(ps.STATE)
    for s in ("decoy", "target", "CHLSB", "AAAAAAK", "mokapot", "Label", "SpecId"):
        assert s not in flat, f"leaked {s!r}"
    assert "XCorr=1.2" in flat and qs["psm_0"]["type"] == "noul"


def test_sage_rel_ev_allowlist_no_leak():
    """Production Sage + fragment-evidence path (the variant behind every Tier B number)."""
    import psm_serializer as ps
    row = {"Label": "-1", "SpecId": "decoy_9", "Proteins": "rev_sp|P1|X_ARATH", "Peptide": "K.QQQQQQK.E",
           "mokapot q-value": "0.01", "mokapot score": "3.2", "ScanNr": "4242", "ladder": "b3,b4 y2,y5"}
    for k, *_ in ps.SAGE_FEATURES + ps.EVIDENCE_FEATURES:
        row[k] = "5"
    ref = {"band": {k: [1.0, 5.0, 9.0] for k, *_ in ps.SAGE_FEATURES + ps.EVIDENCE_FEATURES}}
    flat = json.dumps(ps.build_questions([(0, row)], ref, variant="sage_rel_ev", qtype="choice"))
    for s in ("decoy_", "ARATH", "QQQQQQK", "mokapot", "Label", "SpecId", "4242", "rev_sp"):
        assert s not in flat, f"leaked {s!r}"
    assert "b3,b4 y2,y5" in flat and "ladders" in flat  # evidence is emitted, labels are not


# ---- Tier B production path: run_boundary.py -> psm_serializer.build_questions(variant="sage_rel_ev") ----
# Real column names and values copied from data/tierB/sage_ev.pin and evidence.tsv, so no data file is read.
# Targets (Label 1): SpecId 34184, 4219. Decoy (Label -1): SpecId 2936. SpecIds are numeric, as in production.
REAL_PIN_HEADER = (
    "SpecId Label ScanNr ExpMass CalcMass FileName retentiontime ion_mobility rank "
    "z=2 z=3 z=4 z=5 z=6 z=other peptide_len missed_cleavages semi_enzymatic isotope_error "
    "ln(precursor_ppm) fragment_ppm ln(hyperscore) ln(delta_next) ln(delta_best) aligned_rt predicted_rt "
    "sqrt(delta_rt_model) predicted_mobility sqrt(delta_mobility) matched_peaks longest_b longest_y "
    "longest_y_pct ln(matched_intensity_pct) scored_candidates ln(-poisson) top5_expl top10_expl top20_expl "
    "tic_expl nb ny b_run y_run b_frac y_frac Peptide Proteins"
).split()
REAL_ROW_VALUES = [
    "34184\t1\t31512\t1722.7915\t1722.7883\tb1931.ms2.mzML\t95.81983\t0.0\t1\t1\t0\t0\t0\t0\t0\t16\t0\t0\t0.0\t1.0445998\t2.9216037\t4.027304983669136\t3.509228982612504\t0.0\t0.46067226\t0.45769238\t0.05458823\t0.0\t0.0\t17\t3\t14\t0.875\t4.077929\t219\t2.756067971924275\t1\t3\t7\t0.3647766422415681\t3\t14\t3\t14\t0.2\t0.9333333333333332\tEPSVPASC[+57.0215]AVSATYER\tsp|Q92615|LAR4B_HUMAN",
    "4219\t1\t5079\t1409.6832\t1409.6798\tb1931.ms2.mzML\t34.20233\t0.0\t1\t1\t0\t0\t0\t0\t0\t12\t0\t0\t0.0\t1.230996\t1.6612046\t3.830454418732278\t3.144626976368051\t0.0\t0.16443428\t0.16521302\t0.03162278\t0.0\t0.0\t13\t2\t10\t0.8333333\t4.276613\t209\t2.421560685991429\t4\t7\t8\t0.533077600966181\t3\t10\t2\t10\t0.2727272727272727\t0.9090909090909092\tYHTINGHNAEVR\tsp|P22626|ROA2_HUMAN",
    "2936\t-1\t57812\t1001.52545\t1001.53284\tb1931.ms2.mzML\t156.4835\t0.0\t1\t1\t0\t0\t0\t0\t0\t9\t0\t0\t0.0\t2.125129\t2.4312959\t3.7338144942487617\t2.5457173197346603\t0.0\t0.7523245\t0.57609415\t0.41979802\t0.0\t0.0\t12\t3\t7\t0.7777778\t3.6786761\t342\t2.2957654707804047\t3\t5\t7\t0.2607043603362258\t4\t8\t4\t8\t0.5\t1.0\tC[+57.0215]LAALADLR\trev_sp|Q96KR1|ZFR_HUMAN",
]
REAL_LADDER = {"34184": "b2-4 | y1-14", "4219": "b2-3,10 | y1-10", "2936": "b1-4 | y1-8"}  # evidence.tsv ladder
SAGE_ALLOW = ("ln(hyperscore) ln(delta_next) matched_peaks longest_y_pct ln(matched_intensity_pct) ln(-poisson) "
              "fragment_ppm ln(precursor_ppm) sqrt(delta_rt_model) scored_candidates peptide_len missed_cleavages").split()
EVIDENCE_ALLOW = "top5_expl top10_expl top20_expl tic_expl b_frac y_frac".split()
CHARGE_ALLOW = "z=2 z=3 z=4 z=5 z=6 z=other".split()  # 18 = 12 Sage + 6 charge dummies (make_fm_pins.py)
# z=other is the reference class: never emitted for Jev, so it is allow-listed but not in EMITTED_EXPECTED.
EMITTED_EXPECTED = frozenset(SAGE_ALLOW + EVIDENCE_ALLOW + ["ladder"] + [c for c in CHARGE_ALLOW if c != "z=other"])
BANNED_TOKENS = ("label", "proteins", "specid", "scannr", "filename", "posterior_error", "q-value",
                 "mokapot", "entrapment", "arath", "human", "decoy")
FIXED_HINT_EXEMPT = "target-decoy"  # the only exempt text: the phrase in psm_serializer.SAGE_HINT

def _real_rows():
    """Real rows merged with ladders, plus label-bearing canaries the production PIN does not carry."""
    rows = []
    for line in REAL_ROW_VALUES:
        vals = line.split("\t")
        assert len(vals) == len(REAL_PIN_HEADER), f"header/row width mismatch for {vals[0]}"
        d = dict(zip(REAL_PIN_HEADER, vals))
        assert d["Peptide"] and d["Proteins"][:3] in ("sp|", "rev"), "fixture columns misordered"
        d["ladder"] = REAL_LADDER[d["SpecId"]]
        d.update({"posterior_error": "0.04217", "mokapot q-value": "0.0102", "mokapot score": "3.1"})
        rows.append(d)
    return rows

def _band_ref(rows):
    """Mirror of run_boundary.py ref['band']: per-column sorted band values (label-free)."""
    import psm_serializer as ps
    cols = [k for k, *_ in ps.NUM_FEATURES + ps.SAGE_FEATURES + ps.EVIDENCE_FEATURES]
    return {"band": {k: sorted(float(r[k]) for r in rows) for k in cols if k in rows[0]}}

def _question(row, ref, qtype="noul"):
    import psm_serializer as ps
    return ps.build_questions([(0, row)], ref, variant="sage_rel_ev", qtype=qtype)["psm_0"]

def _leak_tokens(obj):
    import psm_serializer as ps
    text = (json.dumps(obj) + json.dumps(ps.STATE)).lower().replace(FIXED_HINT_EXEMPT, "")
    return [t for t in BANNED_TOKENS if t in text]

def _perturb(v):
    try:
        return str(float(v) + 1.0)
    except ValueError:
        return v + "X"

def test_production_declared_feature_tables_frozen():
    """(b1) Declared Sage and evidence tables are exactly the documented 12 + 6 columns."""
    import psm_serializer as ps
    assert sorted(k for k, *_ in ps.SAGE_FEATURES) == sorted(SAGE_ALLOW)
    assert sorted(k for k, *_ in ps.EVIDENCE_FEATURES) == sorted(EVIDENCE_ALLOW)

def test_production_text_has_no_label_identity():
    """(a) Production Tier B text (noul and choice) names no label/protein/score-model identity.
    Exempt: the fixed phrase 'target-decoy' in psm_serializer.SAGE_HINT, and nothing else."""
    import psm_serializer as ps
    assert ps.SAGE_HINT.count(FIXED_HINT_EXEMPT) == 1
    rows = _real_rows(); ref = _band_ref(rows)
    for qtype in ("noul", "choice"):
        qs = {f"psm_{i}": _question(r, ref, qtype) for i, r in enumerate(rows)}
        hits = _leak_tokens(qs)
        assert not hits, f"{qtype}: banned tokens in production text: {hits}"
        assert "ln hyperscore=" in json.dumps(qs)  # features really are emitted

def test_production_emitted_set_is_allowlist():
    """(b2) Perturbation sweep over every column a row carries (real PIN + canaries): a column counts as
    emitted iff changing its value changes the production question text. Must equal the documented set."""
    rows = _real_rows(); ref = _band_ref(rows); base = rows[0]
    base_txt = json.dumps(_question(base, ref))
    emitted = set()
    for col in base:
        if col.startswith("z="):
            continue  # charge dummies are probed below
        p = dict(base); p[col] = _perturb(base[col])
        if json.dumps(_question(p, ref)) != base_txt:
            emitted.add(col)
    for z in CHARGE_ALLOW:
        p = dict(base); p.update({c: "0" for c in CHARGE_ALLOW}); p[z] = "1"
        if f"charge={z[-1]}+" in json.dumps(_question(p, ref)):
            emitted.add(z)
    assert emitted == EMITTED_EXPECTED, (f"extra={sorted(emitted - EMITTED_EXPECTED)} "
                                         f"missing={sorted(EMITTED_EXPECTED - emitted)}")

def test_known_issue_f3_band_keys_encode_label():
    """KNOWN ISSUE F3 (documented, not fixed; run_boundary.py is not changed here). run_boundary.py builds the
    band as concat(targets, decoys) and keys questions psm_<band position>, so every decoy key exceeds every
    target key. This asserts the current scheme is label-ordered. When keys are shuffled or hashed, flip it."""
    import psm_serializer as ps
    rows = _real_rows(); ref = _band_ref(rows)
    band = [r for r in rows if r["Label"] == "1"] + [r for r in rows if r["Label"] == "-1"]
    keys = [int(k.rsplit("_", 1)[1]) for k in ps.build_questions(list(enumerate(band)), ref, variant="sage_rel_ev")]
    n_t = sum(r["Label"] == "1" for r in band)
    assert max(keys[:n_t]) < min(keys[n_t:]), "keys no longer label-ordered: update KNOWN ISSUE F3"


if __name__ == "__main__":
    for name in sorted(globals()):
        if name.startswith("test_"):
            globals()[name]()
            print(f"PASS {name}")
    print("ALL PASS")
