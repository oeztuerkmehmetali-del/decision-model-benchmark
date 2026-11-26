# Reproducibility audit: merge_jev paper (`paper/paper.tex`)

Date: 2026-11-26. Repo: `~/projects/protein/jev/merge_jev` (HEAD 9eb6e9c plus uncommitted edits from another session).
Method: four read-only sci-reviewer audits, one per case. Reruns used /tmp copies with `PYTHONDONTWRITEBYTECODE=1`. No model APIs were called (`run.py` was not run). A fleet supervisor checked the workers: no off-scope writes.
Paper: 191 lines, compiles to 8 pages with no warnings and no undefined refs. Five bib entries are never cited: `field2007`, `chow1970`, `geifman2017`, `wen`, `lancode`.

## Verdict

| Case | Headline numbers | Figure rerun | Wording / method issues | Inputs inside `paper/` |
|---|---|---|---|---|
| 1 Screening | reproduce | fig2 PNG md5 + numbers.json identical | P2 x2, P3 x2 | yes (SYNERGY CSVs shipped). Steps 2 and 6 need network / ASReview 3.0.8; their outputs are shipped |
| 2 Drug | reproduce | `reproduce.sh --verify` exit 0; metrics, macros, fig3 byte-identical | P2 x4, P3 x3 | yes after this audit (upstream DrugCon tables + README added) |
| 3 Cell | reproduce | fig4 PNG + numbers.json identical; PDF md5 differs (embedded metadata, not pixel-diffed) | **FAIL** claim 11, P2 x4 | heart yes after this audit; pancreas h5ad only on bwHelix |
| 4 Mass spec | reproduce | fig5 PNG + numbers.json identical | **P1** error-estimate label, P2 x4 | yes after this audit (spectra placed locally, gitignored) |

Status vocabulary: every rerun is `rerun-identical` for the PNG/JSON outputs. Claims are checked against shipped data, not against the upstream raw sources.

## Case 1: Screening (SYNERGY, 39 reviews)

Reproduced: 39 reviews, 18,183 test records, 577 eligible; Jev WSS@95 0.7426, ASReview 0.6981; paired difference +0.00376, CI [-0.0655, +0.0585]. d3-ultra count 1,792 is correct (an older note said 1,568).

| Sev | Finding | Fix |
|---|---|---|
| P2 | The ASReview comparator learns from test labels (`optimistic_upper_bound_leaky`). "given every record to learn from" understates this. | Call it an optimistic, label-leaking upper bound. |
| P2 | d3-ultra minus ASReview is +0.040, CI [-0.015, +0.125], Wilcoxon p = 0.0475. Abstract and Fig 2 say "no detectable difference". d3-ultra is detectably better than Jev (+0.024, CI [+0.006, +0.051]), which the paper does not mention. | Report the d3u vs Jev result; make the d3u vs ASReview test statements agree. |
| P3 | "five reviews' test records seen during development" is only partly right: 68 to 72% of the new test records were in the legacy dev part, 14 to 16% in legacy test. | State the overlap fractions. |
| P3 | Only the favourable full-pool ASReview arm is shown. The test-only arm (median 0.53) and the frozen-threshold WSS (Jev 0.640, d3u 0.674) are missing. | Add them to the SI. |

## Case 2: Drug contradictions (DrugCon)

Reproduced: 1,731 candidate pairs / 542 gold, 217 clusters, 819 abstracts, d3u F1 0.7916. `manifest.py --check` OK.

| Sev | Finding | Fix |
|---|---|---|
| P2 | The paper says "probability >= 0.5", but Jev F1 0.7958 comes from argmax. The >= 0.5 rule gives 0.7944. | Fix the wording, or report 0.794. |
| P2 | "Detectably better" decisions use rounded bounds. d3u vs GPT-4.1-nano has lower bound +0.0002, so strictly detectable and the list would have 8 entries. Jev vs Qwen3-32B is borderline (-0.0004). | Decide on unrounded bounds and list the borderline cases. |
| P2 | Published models are treated as fixed constants (one-sample test), which ignores their own sampling error. This bias favours "better" claims. | Say so, or bootstrap both sides where per-item outputs exist. |
| P2 | Counts 494/615/48: "615" was in no shipped file. Its source is the upstream DrugCon README. | Fixed in this audit: README now shipped in `input/upstream/`. |
| P3 | `paper_reference.csv` is hand-transcribed. The nano counterfactual F1 is 0.744 in the CSV and 0.739 in the upstream README. | Reconcile. |
| P3 | The best-of-two-prompts reference is not disclosed. | Disclose it. |
| P3 | Nano seeded the annotation, which favours nano. Already disclosed. | none |

Provenance: upstream DrugCon pinned at commit 2c2d5f5 (code MIT; bench data CC BY 4.0, abstract text under each PMC OA article's licence). The local `contradictory_claim_pairs.csv` md5 is 85be6379… This matches the local clone at that commit. The b802cce2… / 10b37e45… md5s in the README refer to the eval_sets and were not re-checked against GitHub. ManConCorpus (external generalisation test) is not redistributed (licence).

## Case 3: Cell types (pancreas 2,440 / heart 7,075 test cells)

Reproduced: pancreas CellMarker 83.3% vs Jev 76.0% (Jev + literature 84.6%); heart CellMarker z-score 19.8% vs Jev 44.0%; majority baselines 39.63 / 23.15. CellMarker xlsx sha256 `bf52b8cd60df…` matches the README.

| Sev | Finding | Fix |
|---|---|---|
| **FAIL** | "same scoring rule" is false for heart. M2 uses a transductive z-score `((X-mean)/sd)@P`; M1, M1n and Jev use `W@P` on row-normalised log1p(X). (`eval_v31.py` L109-149, `donor_bootstrap_v38.py`) | Describe M2's scorer separately. |
| P2 | Heart 44.0 is the mean of 3 builds (43.72, 43.58, 44.72), while "+23.9" is build 1 minus M2. The mean-based gain is +24.2. | Use one convention. |
| P2 | The heart CI +11.1 to +36.3 comes from resampling only 6 donors and is anti-conservative. Per-donor gains: +23.0, -4.2, +83.0, +29.9, +28.1, +47.1. Donor t-interval +4.0 to +64.9 (df 5); leave-one-donor-out +20.9 to +29.7. | Call the CI approximate and state the 6-donor limit. |
| P2 | Against M1n (23.4%) the heart gain is +20.3, not detectable. On macro accuracy Jev is 28.7 vs M2 23.4 (CI -0.2 to +8.3) and below M1n (32.5). Pancreas macro: Jev 32.0 vs M1 42.6. "Led in heart" holds only for micro accuracy. | Qualify as micro accuracy; report macro. |
| P2 | Double log1p was applied as run. Under single log: pancreas M1 83.3 -> 81.8, Jev+lit 84.6 -> 83.8; heart M1n 23.4 -> 26.8, Jev 43.7 -> 47.9. Two formal verdicts flip (pancreas Jev+lit vs DB; heart Jev vs M1n). | "No comparison changed direction; two formal verdicts flipped." |
| P3 | Heart M1 at 0.27% is a scorer collapse: a 359-gene pDC list absorbs every cell. It is not a coverage gap (only 44 test cells lack a listed gene). | Say so. |
| P3 | Abstract mentions only Jev alone; Jev + literature (84.6, +1.3 vs DB, CI +0.2 to +2.5) is the stronger result. | Optional. |

Provenance: 69 of 81 `paper_path` files are byte-identical to their study copies. `log1p_sensitivity.json` was missing and is now added. The heart QC script `helix_qc.py` differs from the version that produced the stored heart.h5ad (gap documented in the README).

Rerun after placing heart.h5ad: `donor_bootstrap_v38.py --tissue heart` with `--comparator M1n` and `--comparator M2` both ran from `paper/data/3_cell` alone (about 70 s each). Every number matches `compare/jev_vs_baseline/heart_donor_bootstrap{,_M2}.json`. The only differences are two path labels (`catdir`, `donors/source`), which changed when the bundle was restructured. Reproducibility level: rerun-identical (numbers).

## Case 4: Mass spec (Chick et al., Sage + mokapot)

Reproduced: 37,061 targets; mokapot 27,047; d3u 22,197 (-17.9%); Jev 18,906 (-30.1%); engine score 21,939; trained arm 27,071 (+0.1%). No leakage in the trained arm (mokapot is cross-validated; Arabidopsis labels are used only for evaluation).

| Sev | Finding | Fix |
|---|---|---|
| **P1** | The paper (l.107, l.111) says "1% estimated error" via decoys, but the code uses an entrapment estimate: FDP = N_Arab * (1 + 1/r) / N_targets, r = 16345/20416. With decoy-based FDR the counts are: mokapot 27,070, d3u 23,101, Jev 21,439, engine 22,378, giving gaps of -14.7% and -20.8% (paper: -17.9 / -30.1). | Relabel as entrapment-estimated FDP, or switch to decoy FDR and update the numbers. |
| P2 | The Jev count depends on how ties are grouped (96 distinct probabilities). Without keeping ties together: 19,086 (-29.4%). CI [-33.8, -21.7] is wide and asymmetric. | Report tie handling. |
| P2 | The "18 to 30% fewer" range comes from entrapment only. The d3u [-21.2, -14.2] and Jev [-33.8, -21.7] intervals meet near -21, so the two models are not clearly different. | Soften the ranking. |
| P2 | Small-run claim: at n = 500 it rests on a >= 15/20-draw rule. At 1,000, mokapot failed to train in 11/20 draws, but the Fig 5c mokapot line starts at 1k. The 20k draws overlap heavily. | Mark the failed draws; state the overlap. |
| P2 | The bootstrap uses 300 PSM-level resamples and ignores peptide/spectrum dependence. A 2,000-resample rerun agrees. | Note the dependence. |
| P3 | The d3-ultra prompt reveals the entrapment design. | Disclose it. |

Notes: Sage's own 1% cut (27,278, sage.log) is a different statistic from the paper's engine score (21,939, plain ln hyperscore). `sage_nopep.pin` and `sage_ev.pin` have no producing script. The model call dates are not recorded. Run b1931 is linked to Chick et al. / PXD001468 only through EXTERNAL.tsv/README.

## Data availability inside the project (owner requirement)

Changes made in this audit. Files were placed by hardlink (no extra disk). The three >1 GB files are gitignored, and all placed files are listed in each case's `SOURCES.tsv`:

| File placed in `paper/data/` | From | Check |
|---|---|---|
| `4_massspec/input/b1931.mzXML` (1.46 GB) | `studies/x2/data/raw/tierB/` | `fetch.sh` sha256 ok |
| `4_massspec/input/b1931.ms2.mzML` (266 MB) | `studies/x2/data/processed/tierB/` | `fetch.sh` sha256 ok |
| `3_cell/input/external/heart.h5ad` (1.07 GB) | `studies/x7/data/processed/` | `fetch.sh` sha256 ok |
| `3_cell/compare/log1p_sensitivity.json` | `studies/x7/results/jev/` | n/a |
| `2_drug/input/upstream/` (README + 4 bench CSVs; CC BY 4.0, abstracts per-article) | upstream DrugCon clone @2c2d5f5 | md5 recorded above |

Still external (cannot be met from this machine):

| Input | Needed for | Where it is | Why not placed |
|---|---|---|---|
| `3_cell/input/external/pancreas.h5ad` + `raw/{heart,pancreas}.h5ad` | rebuilding splits / QC only; figures and stats run without them | bwHelix workspace `fr_mo1039-af3c5/x7/data/{raw,processed}` | helix unreachable (no SSH master); `cellxgene_census` not installed locally |
| SYNERGY network fetch, ASReview 3.0.8 | screening steps 2 and 6 (outputs shipped) | PyPI / network | tool installs, not data |
| ManConCorpus | drug external test | upstream | not redistributable |

The gitignored GB files exist only on this machine. A git clone will not contain them; `fetch.sh` reports them missing and prints how to get them.

## Recommended paper edits (priority order)

1. Mass spec: relabel the error estimate as entrapment FDP (P1).
2. Cell: remove "same scoring rule"; fix the +23.9 vs 44.0 mismatch; call the heart CI approximate (6 donors).
3. Screening: describe ASReview as label-leaking; report d3u vs Jev.
4. Drug: argmax vs >= 0.5; decide on unrounded bounds.
5. Remove or cite the five unused bib entries.

## Claim check and paper edits (2026-11-26, main session)

Each finding was re-checked against the shipped files in `paper/data/` before editing `paper.tex`. "Not re-checked" means accepted on the audit's evidence without an independent rerun.

| Finding | Check | Verdict | Paper edit |
|---|---|---|---|
| MS P1 entrapment, not decoy | `4_massspec/code/master_eval.py`: `FDP = N_ent(1+1/r)/N_T`, `r = 16345/20416` | accepted | Error estimate described as entrapment (cites `wen`); counts unchanged. Decoy-FDR counts (27,070 / 23,101 / 21,439 / 22,378) not re-checked |
| MS P3 d3-ultra prompt reveals design | `input/example_request_d3u.json` states Arabidopsis/decoy matches are wrong; Jev request does not | accepted | Disclosed in the methods sentence |
| MS P2 small runs at 1,000 | `compare/small_data.tsv`: size 1,000 = 9 trained, 10 best-feature, 1 failed | accepted | Fig 5c caption: 11 of 20 not trained |
| MS P2 ties, CI asymmetry, PSM-level bootstrap, ranking | not re-checked | open | none |
| Cell FAIL "same scoring rule" | `eval_v31.py` L109-149: M1/M1n/Jev score `W@P` on row-normalised log1p; M2 scores z-scored raw X | accepted | Same rule stated for pancreas only; heart z-score baseline described as a different rule |
| Cell 44.0 mean vs +23.9 first build | bootstrap JSON: builds 43.72/43.58/44.72; M2 19.79 | accepted | 44.0 labelled mean of three builds; gain labelled first build |
| Cell heart CI from 6 donors | bootstrap resamples donors | accepted | Called approximate, cites `field2007`; t-interval / leave-one-out values not re-checked |
| Cell macro accuracy | `heart_donor_bootstrap_M2.json`: Jev r1 28.8 vs M2 23.4, CI -0.25 to +8.3 | accepted | Added: not detectable on macro accuracy |
| Cell double log1p flips | `compare/log1p_sensitivity.json`: verdicts flip for pancreas M4 and heart M3, both toward Jev; no direction change | accepted | Limits sentence updated |
| Screening ASReview label leakage | `baseline/*.json` estimand `optimistic_upper_bound_leaky`; `run_baseline.py` comment | accepted | Leakage stated, favours ASReview |
| Screening d3u vs Jev not reported | +0.024 [+0.006, +0.051] from earlier verified numbers | accepted | Reported in Results |
| Screening d3u vs ASReview Wilcoxon p = 0.0475 | paper decides on the paired interval, which includes zero | refuted as an error | none; abstract wording consistent with the stated rule |
| Drug argmax vs >= 0.5 | `detection__jev.tsv`: F1 0.7958 stored vs 0.7944 at p >= 0.5, 2 disagreements, 3 pairs at 0.5; d3u identical | accepted | Rule stated as "more likely answer"; 0.794 given |
| Drug unrounded bounds | `compare/metrics.json`: d3u vs GPT-4.1-nano lower +0.0002; Jev vs Qwen3-32B lower -0.0004 | accepted | Both edge cases named |
| Drug published scores fixed | one-sample comparison | accepted | Intervals stated as somewhat too narrow |
| Uncited bib entries | `paper.tex` | accepted | `lancode`, `field2007`, `wen` now cited; `chow1970`, `geifman2017` removed |
| Screening P3 overlap fractions, test-only arm; drug P3 items; cell P3 items | not re-checked | open | none (main text only; no SI) |
