# Case 4: mass spectrometry (PSM re-scoring), study x2

**Question.** Does a zero-shot decision model (Jev 1.13, d3-ultra) accept as many correct peptide-spectrum matches (PSMs) at 1% estimated error as the trained re-scorer mokapot 0.10.0, on one PXD001468 HEK293 run (b1931, PRIDE run `b1931_293T_proteinID_11A`) searched with Sage 0.14.6 against a human + Arabidopsis (entrapment) + reversed-decoy database? Answer: no (paper Fig 5, Table 1 row 4).

## 2. The decision-model task
Both models score all 46,191 PSMs of the Sage search (37,061 targets, 9,130 decoys). Each question is one PSM; 16 PSMs go in one request, keyed by their running index (`psm_0`..`psm_15` in the first request) and grouped in the order of the mokapot output files (targets by score, then decoys), so a request holds PSMs of similar mokapot score. The model sees search-engine features only: never the target/decoy flag, species, q-value, PEP, mokapot score or peptide letters (`code/serializer_test.py` is the leak test; exit 0 = pass). Answer = binary choice yes/no with probabilities; the probability of "yes" is the score used for ranking. Ground truth is entrapment: an Arabidopsis-only target is known wrong (the sample has no plant); used only in evaluation (Proteins column of `input/sage_ev.pin`).

| | Jev 1.13 | d3-ultra |
|---|---|---|
| model id | `jev-1.13.0` (System One endpoint `https://api.typesafe.ai/v1/systemone`) | `vllm-sr/d3-ultra` (same wire; env `X3_MODEL`) |
| request | `{model, state, questions}` | same |
| state | one sentence (`task`) | same, byte for byte |
| per-PSM text | feature values + decile buckets + b/y ladders | same, byte for byte |
| script | `code/run_boundary.py --all --qtype choice` (all PSMs); borderline band `--qlo 0.005 --qhi 0.05` (mokapot q 0.5-5%, 2,799 targets) | the same all-PSM command with the d3-ultra endpoint (step 6) |
| stored answers | `jev/boundary_jev_cache_tB_all_choice.jsonl` (2,887 requests), `jev/boundary_jev_cache_tB_sage_rel_ev.jsonl` (259 requests) | `d3u/cache_tB_all_choice.jsonl` (2,887 requests), per-PSM table `d3u/tB_all_choice.tsv` |
| settings | none sent: no temperature, seed or threshold (payload is only model/state/questions); cache key = sha256(model, state, questions) | same |
| when | call dates are not recorded in the caches or logs | 2026-11-26 (`d3u/run_d3_same_question.log`: 2,887 requests, 45.8 M tokens; 48 PSMs asked twice first, 0 changed answers) |

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

d3-ultra received the same state and the same `psm_0` question; its stored answer: `{"type": "choice", "choice": "yes", "probabilities": {"yes": 0.917587461011761, "no": 0.08241253898823896}, "confidence": 0.8351749220235221}`

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
| 6 | d3-ultra answers | step 5 command with `--cache d3u/cache_tB_all_choice.jsonl --out d3u/tB_all_choice.tsv` and env `X3_MODEL=vllm-sr/d3-ultra TYPESAFE_BASE_URL=https://decision.vllm-sr.ai JEV_KEY_ENV=VLLM_SR_API_KEY` (with the cache: 0 requests); then `python code/make_d3_all.py` -> `compare/d3_all.tsv` (per-PSM scores of both models) and `d3u/sage_d3.pin` (both byte-identical to stored, checked); then mokapot as in section 3 on `d3u/sage_d3.pin` with `--file_root tBd3` | needs model API (cached answers given) |
| 7 | re-score + compare | `python3 code/master_eval.py > compare/master_result.txt` (byte-identical, checked) | offline |
| 8 | shrink-the-run test | `python code/small_data.py > compare/small_data_result.txt` with the `x2pilot` python (retrains mokapot 120x; mokapot must sit next to the interpreter). The stored `compare/small_data.tsv` and `small_data_result.txt` are copies of the study run, which used the same code and byte-identical inputs | offline |
| 9 | figure + Table 1 numbers | `cd ../../scripts && python3 fig5_massspec.py` -> `figs/fig5_massspec.*`, `numbers.json["massspec"]` | offline |

Headline (`compare/master_result.txt`), accepted at 1% entrapment FDP: engine 21,939; Jev 1.13 18,906 (-30.1%); d3-ultra 12,157 (-55.1%); trained reader 27,071; mokapot 27,047. Both models received identical requests, so the difference between them is the model.

## 5. Known gaps
- The Sage command line and the mzXML conversion are unrecorded; no script produces `input/sage_nopep.pin`, `input/sage_ev.pin`, `jev/sage_jev.pin` (the d3 pin is rebuilt by step 6). The stored files are the inputs.
- Jev call dates are not recorded; Jev reruns are paid and not repeatable. The band-run arguments were reconstructed from the stored output.
- mokapot 0.10.0 is not byte-reproducible run to run with the same seed (checked: three runs on one 1,000-match subset gave three different outputs), so the mokapot columns of the shrink-the-run test change slightly on a rerun (for example trained on 8 or 9 of the 20 subsets of 1,000); the model columns and the conclusions do not. The stored table is the run of 2026-11-26.
- Spectra are external; no offline step needs them.
- `input/human_arath.fasta` holds the reviewed UniProt Swiss-Prot proteins of human (UP000005640) and Arabidopsis thaliana (UP000006548), licence CC BY 4.0; the UniProt release was not recorded.
- In the four mokapot logs (`baseline/tB.log`, `baseline/tB_fm_ev.log`, `jev/tBjev.log`, `d3u/tBd3.log`) the local install path is replaced by `<env>`; otherwise they equal the study logs.
