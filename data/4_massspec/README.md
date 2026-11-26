# Case 4: mass spectrometry (PSM re-scoring), study x2

**Question.** Does a zero-shot decision model (Jev 1.13, d3-ultra) accept as many correct peptide-spectrum matches (PSMs) at 1% estimated error as the trained re-scorer mokapot 0.10.0, on one PXD001468 HEK293 run (b1931, PRIDE run `b1931_293T_proteinID_11A`) searched with Sage 0.14.6 against a human + Arabidopsis (entrapment) + reversed-decoy database? Answer: no (paper Fig 5, Table 1 row 4).

## 2. The decision-model task
Both models score all 46,191 PSMs of the Sage search (37,061 targets, 9,130 decoys). Each question is one PSM; 16 PSMs go in one request (`psm_0`..`psm_15`). The model sees search-engine features only: never the target/decoy flag, species, q-value, PEP, mokapot score or peptide letters (`code/serializer_test.py` is the leak test; exit 0 = pass). Answer = binary choice yes/no with probabilities; the probability of "yes" is the score used for ranking. Ground truth is entrapment: an Arabidopsis-only target is known wrong (the sample has no plant); used only in evaluation (Proteins column of `input/sage_ev.pin`).

| | Jev 1.13 | d3-ultra |
|---|---|---|
| model id | `jev-1.13.0` (System One endpoint `https://api.typesafe.ai/v1/systemone`) | `vllm-sr/d3-ultra` (same wire; env `X3_MODEL`) |
| request | `{model, state, questions}` | same |
| state | one sentence (`task`) | `task`, `sample`, `acquisition`, `search` (full run context incl. Sage settings) |
| per-PSM text | feature values + decile buckets + b/y ladders | same plus RT, isotope offset, b/y run lengths (six extra numbers); criteria text "the assigned peptide produced this spectrum" |
| script | `code/run_boundary.py --all --qtype choice` (all PSMs); borderline band `--qlo 0.005 --qhi 0.05` (mokapot q 0.5-5%, 2,799 targets) | `code/run_d3_all.py` |
| stored answers | `jev/boundary_jev_cache_tB_all_choice.jsonl` (2,887 requests), `jev/boundary_jev_cache_tB_sage_rel_ev.jsonl` (259 requests) | `d3u/cache_all.jsonl` (3,252 stored responses: 2,887 full-context batches used in the paper, plus earlier thin-context batches reused only for the context-effect line) |
| settings | none sent: no temperature, seed or threshold (payload is only model/state/questions); cache key = sha256(model, state, questions) | same |
| when | call dates are not recorded in the caches or logs (gap); `d3u/run_d3_all.log` shows about 2.1 s per batch on a rate-limited endpoint | same |

Question wording comes from `code/psm_serializer.py` (variant `sage_rel_ev`). The first request of each run, exactly as sent, with its stored answer, is in `input/example_request_jev.json` and `input/example_request_d3u.json` (regenerate: `python3 code/show_requests.py jev|d3u`; it asserts that the rebuilt request hashes to a stored cache key).

Jev state (verbatim):
```
{"task": "Tandem-MS peptide-spectrum match (PSM) quality control. Each question describes one database-search PSM by search-engine features only. Estimate the probability that the PSM is a correct identification."}
```
Jev worked example `psm_0` (verbatim from the request that was sent):
```
{"type": "choice", "instructions": "Is this PSM a correct identification? ln hyperscore=4.56 (bucket 10/10); ln gap to next-best candidate (delta_next)=4.17 (bucket 10/10); matched fragment peaks=32 (bucket 10/10); longest y run / peptide length=0.59 (bucket 8/10); ln % of spectrum intensity matched=3.94 (bucket 9/10); ln(-log10 Poisson score)=3.50 (bucket 10/10); mean absolute fragment mass error (ppm)=2.3 (bucket 5/10); ln precursor mass error (ppm)=0.06 (bucket 1/10); sqrt retention-time prediction error=0.05 (bucket 1/10); candidate peptides scored for this spectrum=664 (bucket 8/10); peptide length=27 (bucket 10/10); missed cleavages=1 (bucket 7/10); charge=3+; 2/5 of the 5 most intense peaks explained by b/y ions; 6/10 of the 10 most intense peaks explained; 10/20 of the 20 most intense peaks explained; fraction of total ion current explained=0.36 (bucket 9/10); fraction of possible b ions matched=0.38 (bucket 8/10); fraction of possible y ions matched=0.73 (bucket 8/10); matched ion ladders (ordinals from the N-terminus for b, C-terminus for y): b2-4,9-12,14-16 | y2-17,22-24 Context: Sage search against a target-decoy database; only candidates near the 1% FDR cut-off are listed, so many are false. A correct match usually explains the intense peaks of the spectrum with long contiguous b/y ladders (especially y); a false match typically matches many weak/noise peaks, scattered ions, few of the intense peaks. Buckets compare this PSM with the other listed PSMs (10 = highest). Answer with the probability that this PSM is correct.", "criteria": {"yes": null, "no": null}}
```
stored answer: `{"type": "choice", "choice": "yes", "confidence": 0.46, "probabilities": {"yes": 0.73, "no": 0.27}}`

d3-ultra `search` context (part of its state, verbatim):
```
Sage 0.14.6. Database: 20,416 human and 16,345 Arabidopsis thaliana SwissProt proteins plus reversed decoy sequences. No plant material is in the sample, so a match to an Arabidopsis-only or a decoy sequence is necessarily wrong. Trypsin (after K/R, not before P), up to 2 missed cleavages, peptide length 6-30, peptide mass 500-5000 Da; fixed carbamidomethyl-Cys, variable Met oxidation (max 2); precursor and fragment tolerance 10 ppm; isotope offsets -1 to +3; b and y ions, fragment charge up to 2; spectra deisotoped, 150 most intense peaks kept; only the top-ranked peptide per spectrum is reported.
```
d3-ultra worked example `psm_0` (verbatim):
```
{"type": "choice", "instructions": "Is this PSM a correct identification? ln hyperscore=3.66 (bucket 7/10); ln gap to next-best candidate (delta_next)=2.53 (bucket 6/10); matched fragment peaks=11 (bucket 7/10); longest y run / peptide length=0.45 (bucket 7/10); ln % of spectrum intensity matched=2.38 (bucket 3/10); ln(-log10 Poisson score)=2.18 (bucket 7/10); mean absolute fragment mass error (ppm)=3.1 (bucket 8/10); ln precursor mass error (ppm)=2.11 (bucket 9/10); sqrt retention-time prediction error=0.35 (bucket 7/10); candidate peptides scored for this spectrum=608 (bucket 7/10); peptide length=11 (bucket 4/10); missed cleavages=0 (bucket 1/10); charge=2+; 0/5 of the 5 most intense peaks explained by b/y ions; 1/10 of the 10 most intense peaks explained; 3/20 of the 20 most intense peaks explained; fraction of total ion current explained=0.07 (bucket 3/10); fraction of possible b ions matched=0.30 (bucket 6/10); fraction of possible y ions matched=0.80 (bucket 8/10); matched ion ladders (ordinals from the N-terminus for b, C-terminus for y): b2-4 | y1-3,5-9; retention time (min)=55.8; retention time on a 0-1 scale=0.27; predicted retention time on the same scale=0.39; precursor isotope offset (Da)=2; longest contiguous b run (ions)=3; longest contiguous y run (ions)=5", "criteria": {"yes": "the assigned peptide produced this spectrum: a correct identification", "no": "the assigned peptide did not produce this spectrum: a wrong match"}}
```
stored answer: `{"type": "choice", "choice": "no", "probabilities": {"yes": 0.19443099531942634, "no": 0.8055690046805737}, "confidence": 0.6111380093611474}`

## 3. Comparator
mokapot 0.10.0 (conda env `x2pilot`), command from the logs: `mokapot input/sage_nopep.pin --dest_dir baseline --file_root tB --seed 1 --max_workers 4 --keep_decoys` -> `baseline/tB.mokapot.psms.txt` (standard, 30 features). `baseline/tB_fm_ev.*`: mokapot on the 24 features Jev sees ("trained reader"; pin built by `code/make_fm_pins.py`). `jev/tBjev.*` and `d3u/tBd3.*`: mokapot with the model score added as a feature (pins `jev/sage_jev.pin`, `d3u/sage_d3.pin`; logs `*.log`). Engine score alone = `ln(hyperscore)` of `input/sage_ev.pin`.

## 4. Steps (run from `paper/data/4_massspec/` unless stated)
| # | step | command | level |
|---|---|---|---|
| 1 | spectra -> mzML | `sh input/fetch.sh` (PRIDE download, see `input/EXTERNAL.tsv`; 1.46 GB + 266 MB) | needs HPC/external tool |
| 2 | Sage 0.14.6 search | settings `input/tierB_sage.json`; command line NOT recorded; stored output `input/sage_out/`, `input/sage_nopep.pin` | needs HPC/external tool (gap) |
| 3 | fragment evidence | `python3 code/tierB_evidence.py --sage-dir input/sage_out --mzml input/b1931.ms2.mzML --out input/evidence.tsv` (stored: `input/evidence.tsv`; `input/sage_ev.pin` = Sage pin + evidence columns, no script) | needs external file |
| 4 | mokapot baselines | commands in section 3 (stored: `baseline/`) | needs external tool (mokapot 0.10.0) |
| 5 | Jev answers, all PSMs | `python3 code/run_boundary.py --all --qtype choice --variant sage_rel_ev --pin input/sage_nopep.pin --evidence input/evidence.tsv --mok baseline/tB.mokapot.psms.txt --mok-decoy baseline/tB.mokapot.decoy.psms.txt --cache jev/boundary_jev_cache_tB_all_choice.jsonl --out jev/tB_all_choice.tsv` (fresh calls need `TYPESAFE_API_KEY`; with the cache: 0 requests, output byte-identical to stored, checked) | needs model API (cached answers given) |
| 5b | Jev borderline band | `python3 code/run_boundary.py --pin input/sage_nopep.pin --evidence input/evidence.tsv --mok baseline/tB.mokapot.psms.txt --mok-decoy baseline/tB.mokapot.decoy.psms.txt --variant sage_rel_ev --qlo 0.005 --qhi 0.05 --cache jev/boundary_jev_cache_tB_sage_rel_ev.jsonl --out jev/tB_boundary_sage_rel_ev.tsv` (byte-identical, checked; arguments reconstructed, not recorded) | needs model API (cached answers given) |
| 6 | d3-ultra answers | `X3_MODEL=vllm-sr/d3-ultra python3 code/run_d3_all.py` -> `compare/d3_all.tsv` (per-PSM scores of both models) and `d3u/sage_d3.pin` (both byte-identical to stored, checked; 0 requests) | needs model API (cached answers given) |
| 7 | re-score + compare | `python3 code/master_eval.py > compare/master_result.txt` (byte-identical, checked) | offline |
| 8 | shrink-the-run test | `python code/small_data.py > compare/small_data_result.txt` with the `x2pilot` python (retrains mokapot 120x; mokapot must sit next to the interpreter; `compare/small_data.tsv` byte-identical, checked) | offline |
| 9 | figure + Table 1 numbers | `cd ../../scripts && python3 fig5_massspec.py` -> `figs/fig5_massspec.*`, `numbers.json["massspec"]` | offline |

Headline (`compare/master_result.txt`), accepted at 1% entrapment FDP: engine 21,939; Jev 1.13 18,906 (-30.1%); d3-ultra 22,197 (-17.9%); trained reader 27,071; mokapot 27,047. Jev 1.13 and d3-ultra differ in prompt wording as well as in model.

## 5. Known gaps
- The Sage command line and the mzXML conversion are unrecorded; no script produces `input/sage_nopep.pin`, `input/sage_ev.pin`, `jev/sage_jev.pin` (the d3 pin is rebuilt by step 6). The stored files are the inputs.
- Call dates of both models are not recorded; Jev reruns are paid and not repeatable. The band-run arguments were reconstructed from the stored output.
- `compare/small_data_result.txt` is the stored stdout; the rerun was checked through `compare/small_data.tsv`.
- The figure script asserts that all 20 size-500 shrink-run mokapot rows failed; kept as is.
- Spectra are external; no offline step needs them.
