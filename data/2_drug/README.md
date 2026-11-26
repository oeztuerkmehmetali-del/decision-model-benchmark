# Case 2: drug-trial claim contradictions (DrugCon-Bench)

Source study `studies/x8_drugcon`. Paper item: Fig. 3 (`paper/scripts/fig3_drug.py`), Table 1 drug row, numbers.json key `drug`.
Everything below is relative to this folder (`paper/data/2_drug/`). Nothing outside `paper/` is needed.

## 1. Question

Does a typed decision model (Jev 1.13 or d3-ultra) detect contradictory drug-trial claim pairs as well as the eleven language models published with DrugCon-Bench (Lan et al. 2026, medRxiv doi:10.64898/2026.11.11.26367856)?
Metric: detection F1 on 1,731 pairs (542 contradictory, 217 PMID-connected groups), 95% CI from a cluster bootstrap over the groups.

## 2. The decision-model task

Two tasks, run on both models; the figure uses **detection** only (explanation feeds the numbers in `compare/metrics.json` that the paper text may cite).

| | |
|---|---|
| Models | Jev 1.13 = `jev-1.13.0` at `https://api.typesafe.ai` (released 2026-09-15; key `TYPESAFE_API_KEY`); d3-ultra = `vllm-sr/d3-ultra` at `https://decision.vllm-sr.ai` (key `VLLM_SR_API_KEY`; the pinned `vllm-sr/d3` is rejected by the endpoint, d3-ultra answered). |
| Settings | None sent. The request body is exactly `{"model", "state", "questions"}` (`code/run.py`, `paper/data/shared/jevlib.py::_post`): no temperature, seed or threshold. Answers are cached by sha256 of model+state+questions, so a rerun from the cache is offline. |
| Decision rule (fixed before any run) | Detection: argmax of the typed `choice` (`pred`); `p_contra` = probability of "Contradiction exists" (Jev rounds probabilities to 0.01). Explanation: a factor is kept if its probability is >= 0.5. |
| When | Main runs 2026-11-20 (logs `jev/log_*__jev.txt`, `d3u/log_*__d3u.txt`). Repeat run (fresh answers, seeded subsets, `pandas.sample(random_state=0)`: detection n=200, explanation n=50) from 2026-11-20T13:26:18Z (`compare/rep1_started_utc.txt`, hand-written). |
| Calls | Per model: detection 1,711 requests + 20 cache hits (1,731 pairs); explanation 532 requests + 10 cache hits (542 pairs, 10 questions each in one request); repeat 200 + 50 requests. Totals from the logs; cost about $0.09 per model at the Jev token price. d3-ultra needed 529 retries on detection and 410 on explanation (rate limits), Jev none. |

**Detection template** (`code/run.py`, `DET_STATE`/`DET_CRITERIA`/`det_q`). One request per pair; state and one `choice` question:
```
state.task  = "Decide whether two biomedical claims about the same drug-disease association present a contradiction, using counterfactual reasoning: assume Claim 1 is true and ask whether Claim 2's overall assessment of the drug for the disease still holds; then the reverse. A contradiction here is an apparent contradiction: the two claims give opposing overall assessments of the same drug for the same disease (effective vs. ineffective, beneficial vs. harmful, safe vs. unsafe); it need not be a strict logical impossibility. A contradiction may be partial: if one subclaim opposes the other claim, the pair is contradictory even if other parts agree. Do NOT dismiss a contradiction merely because the studies differ in outcome, patient subgroup, dose, duration, timing, comparator or study design."
questions.pair = {type: "choice",
  criteria: {"Contradiction exists": "The overall assessments of the drug for the disease oppose each other, fully or in part.",
             "Claims can coexist":  "The claims concern different drugs or diseases, express the same overall direction, or make no clear assessment of the drug-disease association."},
  instructions: "Claim 1: <claim_1>\nClaim 2: <claim_2>"}
```
Worked example, `pair_id` 0 of `input/eval_sets/detection.csv` (gold "Contradiction exists"), verbatim from the stored inputs (full request and both stored answers: `input/worked_example.json`):
```
Claim 1: Adding nebulized magnesium sulfate to standard therapy in patients with moderate to severe asthma attacks leads to greater and faster improvement in PEFR, respiratory rate, oxygen saturation and respiratory rate.It also reduces hospitalization rates in this patient population.
Claim 2: The use of nebulised magnesium sulphate in moderate asthma exacerbation as adjuvant treatment showed no benefit to standard treatment in our study.
Jev 1.13  -> {"choice": "Contradiction exists", "confidence": 1.0,  "probabilities": {"Contradiction exists": 1.0,    "Claims can coexist": 0.0}}
d3-ultra  -> {"choice": "Contradiction exists", "confidence": 0.99978, "probabilities": {"Contradiction exists": 0.99989, "Claims can coexist": 0.00011}}
```
**Explanation template** (`EXP_STATE`/`exp_q`): state = task text ("Two biomedical claims report opposing outcomes for the same drug-disease association. Determine which contextual factor(s) explain the apparent contradiction. Base the analysis strictly on the provided claims and abstracts; do not use external knowledge.") + `claim_1`, `claim_1_context` (abstract), `claim_2`, `claim_2_context`; ten `noul` questions `f0..f9`, one per canonical factor (`drugcon.evaluation.CANONICAL_FACTORS`): "Does the contextual factor '<factor>' explain the apparent contradiction between the two claims?". Worked example (id 0) is in `input/worked_example.json`.

The wording paraphrases, and is not identical to, the authors' prompts (`input/prompts.md`: `counterfactual_appendix_v2`, `taxonomy_v1`). The authors use one multi-label prompt for explanation; we use ten independent questions.

## 3. Comparator (baseline)

Eleven published models from Lan et al. 2026, Table 3 (T=0; GPT-4.1-nano detection = mean over T 0/0.5/1 of Table 2): `baseline/paper_reference.csv`. **Hand-transcribed on 2026-11-20 from the image** `https://www.medrxiv.org/content/medrxiv/early/2026/11/12/2026.11.11.26367856/T3.large.jpg`; no script, header comments in the file. Detection value per model = the better of its two prompt columns (`cf_f1`, `dc_f1`); the strongest is Claude-Sonnet-4.5, 0.793. Published values carry no sampling error, so differences to them carry only our bootstrap uncertainty.

## 4. Steps, input -> model calls -> results -> figure

Run all from `paper/data/2_drug/` unless stated. Levels: `offline` = from files in this folder; `needs model API` = fresh calls (cached answers are given); `needs HPC/external tool` = none here.

| # | Step | Command | Input -> output | Level |
|---|---|---|---|---|
| 1 | Check input and answer checksums | `python3 code/manifest.py --check` | `code/MANIFEST.json` lists sha256 of `input/`, `jev/`, `d3u/`, `baseline/`, `code/`, `../shared/jevlib.py` | offline |
| 2 | Model calls, detection | `python3 code/run.py --model jev --task detection --workers 6` (and `--model d3u`) | `input/eval_sets/detection.csv` -> `jev/cache_jev.jsonl`, `jev/detection__jev.tsv` (`d3u/...`). With the shipped caches this makes no call. | offline from cache; `needs model API` for fresh answers (new `--tag` forces fresh calls) |
| 3 | Model calls, explanation | `python3 code/run.py --model jev --task explanation --workers 6` (and `--model d3u`) | `input/eval_sets/explanation.csv` -> `jev/explanation__jev.tsv`, `d3u/explanation__d3u.tsv` | as step 2 |
| 4 | Repeat run (fresh answers) | `python3 code/run.py --model jev --task detection --limit 200 --tag _rep1`; `--task explanation --limit 50 --tag _rep1`; same with `--model d3u` | -> `*_limit200_rep1.tsv`, `*_limit50_rep1.tsv`, `cache_*_rep1.jsonl` | `needs model API` (stored) |
| 5 | Metrics, CIs, paired differences, vs. published models | `python3 code/analyze.py` (about 65 s) or `bash code/reproduce.sh [--verify]` | `{jev,d3u}/*.tsv`, `{jev,d3u}/log_*.txt`, `baseline/paper_reference.csv`, `compare/rep1_started_utc.txt` -> `compare/metrics.json`, `compare/macros.tex` (cluster bootstrap B=2000, seed 0, same replicates for both models) | offline |
| 6 | Figure and numbers | from `paper/scripts/`: `python3 fig3_drug.py` | `compare/metrics.json`, `compare/macros.tex`, `jev/detection__jev.tsv` (gold labels), `baseline/paper_reference.csv` -> `paper/figs/fig3_drug.{pdf,png}`, `numbers.json["drug"]` | offline |
| - | Worked example | `python3 code/worked_example.py` | -> `input/worked_example.json` | offline |

`fig3_drug.py` asserts its headline numbers against `compare/macros.tex`; stale or differently regenerated results trip the asserts.

## 5. Known gaps

- `baseline/paper_reference.csv` is a transcription of a figure image; no script and no independent check beyond `fig3_drug.py` asserting it against `compare/macros.tex`.
- The frozen evaluation sets `input/eval_sets/*.csv` are pinned upstream (github.com/MengfeiLan/DrugCon_Drug_Contradiction @ `2c2d5f5f4542cee0925a6b6b398694cc2433083d`, MIT) with source md5 in `input/eval_sets/PROVENANCE.txt` (`contradictory_claim_pairs.csv` b802cce2..., `candidate_pairs.csv` 10b37e45...). The upstream builder `scripts/build_eval_sets.py` and those raw CSVs are not copied; a local upstream copy has a different md5 for `contradictory_claim_pairs.csv` (85be6379...), so the eval-set build is not rerun here. No file exceeds 50 MB, so there is no `input/EXTERNAL.tsv`.
- Fresh model calls need paid keys and can drift; the repeat run reproduced 99.5% (Jev) and 100% (d3-ultra) of detection labels on 200 pairs, 98.6% / 100% of explanation decisions on 50 pairs (`compare/metrics.json`, key `repeatability`).
- Gold explanation factors were annotated after the authors saw GPT-4.1-nano predictions; abstracts are likely in the models' pretraining data; clinical claims only. The upstream pair bootstrap (CI .769-.821) is too narrow because pairs share PMIDs; we use the cluster bootstrap.
- `code/MANIFEST.json` hashes `../shared/jevlib.py`; editing that file breaks `--check` until `python3 code/manifest.py` is rerun.
- `code/analyze.py` is the study script minus its standalone-figure section (`fig_*.pdf`, not used by the paper); numbers are unchanged. The authors' scorer is kept as `code/drugcon/evaluation.py` (+ `__init__.py`); other upstream modules are not used.
- Model timestamps beyond the log dates are not recorded per call.
- Process note: the study's own `reproduce.sh`/`MANIFEST.json` tracked the vendored layout; here they are adapted to this folder (`code/reproduce.sh`, `code/manifest.py`, `code/MANIFEST.json`).
