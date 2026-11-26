# Case 1: screening (source study x3) -> Fig 2, Table 1 row "screening"

## 1. Question
Can a decision model (Jev 1.13, d3-ultra) rank title/abstract records of 39 SYNERGY systematic reviews well enough to save screening work at 95% recall (WSS@95), compared with ASReview LAB 3.0.8?

## 2. Decision-model task
Roles of folders: `input/` (given to models + labels), `jev/` (Jev 1.13 answers/results), `d3u/` (d3-ultra answers/results), `baseline/` (ASReview), `compare/` (both models and/or baseline), `code/`.

One call per record, built by `code/questions.py` (criteria split into clauses) and `code/fanout.py`:

* `state` = `{"title": <record title>, "criteria": <review's eligibility criteria, verbatim>, "abstract": <record abstract>}`
* `questions` = for each criteria clause i (max 8, sentence/;/bullet split, > 20 chars): `clause_i`, type `noul`, instructions
  `Does this record satisfy this eligibility clause? Clause: <clause text>`; plus one `include`, type `noul`, instructions
  `Overall: should this record be INCLUDED for full-text review? Eligibility criteria: <criteria text>`.
* answer per question = `noul` (probability-like score in [0,1]) (+ `confidence`, always null in these runs). The record score is the `include` noul, clause answers feed the gap/ablation analyses.
* Worked example, verbatim from the stored inputs (rebuilt by `code/worked_example.py`): `input/worked_example.json` (review Harms_2024, record `https://openalex.org/w2896253248`, label included=1; contains the request, the stored Jev answers (include 0.88) and d3-ultra answers).
* Number of calls (cache keys): Jev `jev/jev_cache.jsonl` 84,119 keys; d3-ultra `d3u/d3u_cache.jsonl` 81,708 keys. Scored per model: dev 62,093 + test 18,183 records (409,686 and 120,511 noul answers). `jev/jev_dev.json` as stored holds 64,947 dev and 19,902 test records (see gaps); scoring uses only the records in `input/split.union.json`.
* Threshold: per-review tau chosen on the dev split only (`code/sweep.py`: lowest-cost tau reaching dev recall >= 0.95; tau grid 0.01..0.99), frozen, then applied to test. Split: `input/split.union.json` (seed 0, dev/holdout/test per review; sha256 field `split_sha256`).
* When: Jev calls 2026-10-05/06 (cache mtime 2026-10-06 00:50), d3-ultra calls 2026-11-24 (dev 19:27, test 20:30; cache 20:45).
* Number of calls (cache keys): Jev `jev/jev_cache.jsonl` 84,119 keys; d3-ultra `d3u/d3u_cache.jsonl` 81,708 keys. Answers used: dev 62,093 + test 18,183 records per model (x7.. questions/record: dev 409,686 nouls; test 120,511 nouls for each model in the re-fanned files). `jev/jev_dev.json` stores 64,947 dev and 19,902 test records (superseded split members, see gaps); scoring uses only the records in `input/split.union.json`.

## 3. Comparator
ASReview LAB 3.0.8 simulate, model `elas_u4` (and `elas_u3`), 1 prior include + 1 prior exclude, per review. Outputs `baseline/<review>_elas_u{3,4}.json` (screen order and WSS@95; produced on HPC/workstation by `code/run_baseline.py`; the `.asreview` intermediates and per-review test CSVs are not kept). Arms used in the paper: `test-only` (simulation on test records only, own order; leaky/optimistic), `equal footing` / full-pool (ASReview trained on all reviewed records and scored on test only), see `compare/results_v3_fixes.json`.

## 4. Steps (run from `paper/data/1_screening/code/` unless stated)
Study-relative paths used inside the scripts (`runs/jev/...`, `data/processed/...`) are mapped to this layout by `code/layout.py`.

| # | Step | Command | Level |
|---|------|---------|-------|
| 1 | SYNERGY records -> `input/synergy/<rev>.csv` (copied; 39 reviews + their `.test.csv`) | `input/fetch.sh` lists the source (pip `synergy-dataset==2.2`); files are included | offline |
| 2 | split + questions | `python3 freeze_split.py` (-> `input/split.union.json`), `python3 questions.py` (-> `input/questions/<rev>.json.gz`, shipped gzipped per review; `layout.load_questions()` reassembles) | needs network (synergy_dataset); outputs given |
| 3 | Jev answers | `X3_RUNTAG=jev python3 fanout.py dev` and `... test` (-> `jev/jev_dev.json`, `jev/jev_test.json`) | cached answers given: offline (cache hits, no key); fresh: needs model API |
| 4 | d3-ultra answers | `bash d3u_run.sh` (dev fan-out, tau frozen, test fan-out; -> `d3u/d3u_dev.json`, `d3u/d3u_test.json`, `d3u/dev_tau.d3u.json`) | offline from `d3u/d3u_cache.jsonl`; fresh: needs model API |
| 5 | dev threshold | `X3_DEV=runs/jev/jev_dev.json X3_SPLIT=data/processed/split.union.json X3_TAU=runs/jev/dev_tau.union.json python3 sweep.py` (Jev); d3-ultra: `X3_MODEL=vllm-sr/d3-ultra X3_DEV=runs/d3u/d3u_dev.json X3_SPLIT=data/processed/split.union.json X3_TAU=runs/d3u/dev_tau.d3u.json python3 sweep.py` | offline |
| 6 | ASReview baseline | `ASREVIEW_BIN=<asreview 3.0.8> python3 run_baseline.py` (-> `baseline/`) | needs HPC/external tool (asreview==3.0.8; outputs given) |
| 7 | Jev test scoring | `X3_TAU=runs/jev/dev_tau.union.json X3_OUT=results/jev/results_test.union39.json python3 score_final.py` (-> `compare/results_test.union39.json`, Jev vs ASReview) | offline |
| 8 | analyses | `python3 ablation.py` (-> `jev/results_ablation39.json`), `python3 calib_sim.py` (-> `jev/results_calibsim39.json`), `python3 gap_tests.py` (-> `jev/results_gaps39.json`), `python3 net_effort.py` (-> `compare/results_net_effort.json`) | offline |
| 9 | combined Jev/ASReview results with published-value asserts | `python3 v3_fixes.py` (-> `compare/results_v3_fixes.json`, `compare/v3_numbers.md`) | offline |
| 10 | d3-ultra arm (asserts it reproduces the published Jev numbers first) | `python3 d3u_arm.py` (-> `d3u/results_d3u.json`, `d3u/d3u_numbers.md`) | offline |
| 11 | figure + numbers.json key `screening` | from `paper/scripts/`: `python3 fig2_screening.py` (re-derives d3-ultra per review from `d3u/` answers with `code/v3_fixes.py`, asserts equality with `d3u/results_d3u.json`) | offline |

Offline re-run check done for this folder: see `RERUN.md`.

## 5. Known gaps
* `input/reviews_union.json` and `input/split.json` are copied as they are; the original `data/raw/synergy` CSVs come from SYNERGY (OpenAlex ids); no external files > 50 MB, so `input/EXTERNAL.tsv` has no data rows and `input/fetch.sh` only documents the SYNERGY source.
* Steps 2 and 6 cannot run here without network / ASReview 3.0.8; their outputs are shipped. The ASReview `.asreview` intermediates and logs are not shipped.
* d3-ultra: 1,792 test records were queried (cached, never scored) before the threshold was frozen (`pre_tau_test_queries_disclosed` in `d3u/results_d3u.json`); the d3-ultra cache holds 1,434 keys that match no current record (origin unknown, never read).
* Jev answer files hold more dev/test records (64,947 / 19,902) than the frozen split (62,093 / 18,183); only records in the split are scored.
* ASReview "test-only" arm is weaker than the equal-footing arm (see paper SI); the d3-ultra vs equal-footing ASReview paired difference is +0.040, CI [-0.015, 0.125] (includes 0).
* `paper.tex` audit issues about the screening claims (frozen-threshold WSS, P2 net-effort) are not addressed in this folder.
* No temperature/seed is exposed by the model services; reruns with fresh calls may differ from the cached answers.
