# Decision-model benchmark

Data, code and LaTeX source for the paper testing two pretrained decision models (Jev 1.13 and d3-ultra) on four
bioinformatics tasks, ordered from text to numeric data, each against the established tool for that task.

| Case | Folder | Task | Compared against |
|---|---|---|---|
| 1 | `data/1_screening/` | rank titles/abstracts of 39 systematic reviews | ASReview LAB active learning |
| 2 | `data/2_drug/` | detect contradictions between drug-trial claims | eleven published LLMs (DrugCon-Bench) |
| 3 | `data/3_cell/` | name cell types from marker genes | CellMarker 2.0 database |
| 4 | `data/4_massspec/` | re-score peptide-spectrum matches | mokapot trained on the same run |

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
