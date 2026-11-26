# Pretrained Decision Models in Bioinformatics: Comparable to Active-Learning Screening, Baseline-Dependent in Cell-Type Annotation, Behind a Trained Mass-Spectrometry Re-scorer

Mehmet Ali Öztürk and Simon Elsässer. Data, code and LaTeX source for the paper (`paper.pdf`).

## What this project is

Much of biomedical research consists of small decisions with a fixed set of answers, repeated many times: does this
study belong in the review, do these two trial claims contradict each other, which type is this cell, is this
peptide-spectrum match correct. A decision model is a language model used for such closed questions. It is given a
case and a list of allowed answers and returns a probability for each answer, with no training on the task.

We tested two pretrained decision models, Jev 1.13 and d3-ultra, on four public bioinformatics tasks with known
answers. The tasks are ordered from text input to numeric input, and each is compared with the tool a practitioner
would otherwise use.

![Figure 1. What a decision model is, and the four tests.](figs/fig1_gradient.png)

**Figure 1.** The same question put to a chatbot and to a decision model, and the four tests ordered from text to
numeric input.

| Case | Folder | Task | Compared against |
|---|---|---|---|
| 1 | `data/1_screening/` | rank titles/abstracts of 39 systematic reviews | ASReview LAB active learning |
| 2 | `data/2_drug/` | detect contradictions between drug-trial claims | eleven published LLMs (DrugCon-Bench) |
| 3 | `data/3_cell/` | name cell types from marker genes | CellMarker 2.0 database |
| 4 | `data/4_massspec/` | re-score peptide-spectrum matches | mokapot trained on the same run |

## Folder layout

Each case folder is self-contained: `README.md` (the decision-model task: question, inputs, settings, outputs, rerun
commands, known gaps), `input/`, `jev/` and `d3u/` (each model's cached answers and results), `baseline/`, `compare/`,
`code/`, and `SOURCES.tsv` (one row per file: path, source in the original study, role, whether modified, note).
`data/shared/jevlib.py` is the model client.

## Rebuild figures and paper (offline, no model calls)

```
python3 scripts/fig1_gradient.py && python3 scripts/fig2_screening.py && python3 scripts/fig3_drug.py && python3 scripts/fig4_cell.py && python3 scripts/fig5_massspec.py
mkdir -p build && pdflatex -interaction=nonstopmode -output-directory=build paper.tex && pdflatex -interaction=nonstopmode -output-directory=build paper.tex && cp build/paper.pdf .
```

Needs numpy, pandas, matplotlib and a TeX Live install. Each figure script asserts its headline numbers against the
stored results and writes every quoted value to `figs/numbers.json`.

## Limits

- Fresh model answers need paid API keys; with the cached answers no call is made.
- Raw files over 50 MB (mass-spectrometry spectra; single-cell `.h5ad` files) are not included. They are listed with
  size and sha256 in `data/<case>/input/EXTERNAL.tsv`; `input/fetch.sh` checks them after download.
- Case-specific gaps are listed in each case README.

## Third-party data

- Case 1: SYNERGY dataset (`synergy-dataset==2.2`, CC0 1.0).
- Case 2: DrugCon-Bench annotations (CC BY 4.0); abstract text is from the PMC Open Access subset under each article's licence (`data/2_drug/input/upstream/README.md`).
- Case 3: Tabula Sapiens cells and CellMarker 2.0 markers, cited in the paper.
- Case 4: UniProt Swiss-Prot protein sequences (CC BY 4.0).

## Results

| Case | Measure | Jev 1.13 | d3-ultra | Comparator | Reading |
|---|---|---|---|---|---|
| 1 Study screening | work saved at 95% recall, ideal stopping point (an upper bound) | 0.74 | 0.74 | ASReview 0.70 | no detectable difference between Jev and ASReview, without training |
| 2 Drug-trial claims | F1 | 0.796 | 0.792 | best published model 0.793 | close to the best published model, not ahead |
| 3 Cell types, pancreas | cells named correctly | 75.6% | 79.4% | database 81.8% | behind where the database is good |
| 3 Cell types, heart | cells named correctly | 48.1% | 41.0% | database 19.8% (z-score version), 26.8% (size-normalised version, tried after the results were known) | ahead of the usual use of the database; Jev's lead over the stronger use is not detectable |
| 4 Mass spectrometry | matches accepted at 1% error | 18,906 | 12,157 | mokapot 27,047 | 30 to 55% fewer than a re-scorer trained on the run |

The two decision models showed no detectable difference from the comparators on the two text tasks, gained in
cell-type annotation only where the curated database is weak, and stayed behind a trained re-scorer on numeric
mass-spectrometry data. Intervals, the pre-set cut-off for screening and all caveats are in the paper.
