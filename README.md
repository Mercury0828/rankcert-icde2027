# Ranking Queries with Confidence Guarantees over Quantum Circuit-Cutting Reconstructions: code and data

Code, result rows and extended version (full proofs and experimental configuration, `extended_version.pdf`) for the
IEEE ICDE 2027 submission by Jiachen Shen (University of Houston) and Hui Zhong (Miami University).
Archived on Zenodo: https://doi.org/10.5281/zenodo.24120178

## Layout

- `rankcert/`: the method. Confidence regions on raw measurement categories (`regions/`), interval propagation
  (`propagate/interval.py`), refined score and pair bounds, pilot selection, allocation and labelling
  (`certify/rpair.py`), appended-shot rounds (`certify/seq.py`), and the branch-and-bound searches (`search/`).
- `baselines/bl1.py`: BL1 (uniform Hoeffding / DKW bound).
- `bench/`: benchmark circuits (adder, QAOA, Bernstein-Vazirani) and their exact reference distributions;
  `bench/a6.py` lists every instance with its split (development seeds 1001-1005, held-out seeds 1006-1010).
- `stats/labels.py`: label evaluation against the exact answer (sure members, sure non-members, wrong labels).
- `derisk/`: evaluation runners and the scripts that turn rows into the paper's tables and figures.
- `data/`: the result rows used in the paper (JSON lines, one row per instance, budget, repetition and method):
  `data/a6/runs` and `data/a6/runs6` (development), `data/a6/heldout` (held-out), `data/h3/check_rows_tight.jsonl`
  (search beyond enumeration). `derisk/results/a6/` holds the generated summaries.
- `derisk/A6_FREEZE.json`: the method, parameters and code version frozen on the development instances before the
  held-out instances were evaluated.

## Environment

Python 3.11 with the packages in `requirements.txt`, plus QCut 2.4.0
(`pip install git+https://github.com/FiQCI/QCut@v2.4.0`). Run every command from the repository root with
`PYTHONPATH=.` and one thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`).
The paper's runs used one thread of an Intel Core i9-13900K with 64 GB of memory.

## Paper item -> how it is produced

| Paper item | Command | Output |
|---|---|---|
| Tables II, III, IV and the numbers in the text | `python -m derisk.a6_paper_numbers` then `python -m derisk.a6_paper_tables` | `derisk/results/a6/paper_numbers.json`, `paper/tables/{cov,a6,abl,search,numbers}.tex` |
| Table V (search) | rows in `data/h3/`; same two commands | `paper/tables/search.tex` |
| Figure 1 and Examples 1-3 | `python -m derisk.a6_fig1` then `python figs/fig1.py` | `derisk/results/a6/fig1.json`, `figs/fig1.pdf` |
| Figure 2 | `python figs/fig_shots.py` (reads `paper_numbers.json`) | `figs/fig_shots.pdf` |
| Table I | descriptive; the sizes come from the instances listed in `bench/a6.py` | |
| Simulator check of the references (Section VI-A) | `python -m derisk.a6_validate <instance>` | `derisk/results/a6/validation.json` |

The table scripts read the rows from `data/` and write LaTeX into `paper/tables/` (created if absent).

## Re-running the experiments

1. Build the exact reference distributions of an instance (`bench/a6.py`, written under `data/a6/ref` and
   `data/a6/embargo` for held-out seeds).
2. Development instances: `python -m derisk.a6_run <instance> --Ns 1000,10000,100000,1000000 --reps 0-9
   --methods R-CP,BL1,R-PAIR,R-PAIR-S,R-PAIR6,R-PAIR-S6`. Held-out instances (N = 1e5, reps 0-9):
   `bash derisk/run_a6_heldout.sh <shard> <instance ...>`.
3. Search beyond enumeration: `python -m derisk.h3_search` (see its docstring for the grid m in {32, 36, 40}).
Draws are seeded per (family, m, seed, N, repetition, stream), so reruns reproduce the rows.

## Tests

`python -m pytest tests`. Tests that need reference distributions skip when `data/a6/ref` has not been built.
