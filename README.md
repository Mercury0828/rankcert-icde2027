# Ranking Queries with Confidence Guarantees over Quantum Circuit-Cutting Reconstructions: code and data

This archive accompanies the IEEE ICDE 2027 submission of the same title by Jiachen Shen (University of Houston) and
Hui Zhong (Miami University). It contains the implementation, the benchmark generator, the evaluation scripts, the
result rows behind every number of the paper, and the extended version with the full proofs (`extended_version.pdf`).

## Contents

- `rankcert/`: the method. Confidence regions on raw measurement categories (`regions/`), interval propagation
  (`propagate/`), refined score and pair bounds with the pilot, allocation and labelling (`certify/rpair.py`), and the
  branch-and-bound searches (`search/`).
- `baselines/`: BL1 (uniform Hoeffding / DKW bound).
- `bench/`: generation of the benchmark circuits (adder, QAOA, Bernstein-Vazirani) and their exact reference
  distributions; `bench/a6.py` lists every instance with its split (development or held-out).
- `stats/`: label evaluation against the exact answer (sure members, sure non-members, wrong labels).
- `derisk/a6_run.py`: the evaluation runner (all methods, equal total shots, seeded draws).
  `derisk/run_a6_heldout.sh` runs the held-out evaluation. `derisk/h3_search.py` runs the search beyond enumeration.
- `derisk/a6_paper_numbers.py` and `derisk/a6_paper_tables.py`: compute every table and number of Section VI from the
  rows; `derisk/a6_fig1.py`, `figs/fig1.py` and `figs/fig_shots.py` produce the figures.
- `rows/`: the result rows (JSON lines, one row per instance, budget, repetition and method) for the development
  instances, the held-out instances and the search runs. `derisk/results/a6/` holds the generated summaries.
- `derisk/A6_FREEZE.json`: the record of the method and parameters frozen on the development instances before the
  held-out instances were evaluated.
- `tests/`: unit tests (`python -m pytest tests`).

## Requirements

Python 3.11, NumPy 2.4, SciPy 1.17, Matplotlib 3.11, pytest, and the circuit-cutting library QCut 2.4.0
(https://github.com/FiQCI/QCut/releases/tag/v2.4.0) with Qiskit; Qiskit Aer only for `derisk/a6_validate.py`.
All runs in the paper used one thread of an Intel Core i9-13900K with 64 GB of memory.

## Reproducing the paper

1. Generate the reference distributions of the benchmark: see `bench/a6.py` (writes the instances under the directory
   configured as `A6` in that file).
2. Run the evaluation: `python -m derisk.a6_run <instance> --Ns 1000,10000,100000,1000000 --reps 0-9
   --methods R-CP,BL1,R-PAIR,R-PAIR-S,R-PAIR6,R-PAIR-S6` for development instances, and
   `bash derisk/run_a6_heldout.sh <shard> <instance ...>` for held-out instances. The paper's own rows are in `rows/`.
3. Regenerate the numbers and tables: `python -m derisk.a6_paper_numbers` then `python -m derisk.a6_paper_tables`
   (both read the rows from the `A6` directory; copy `rows/` there or point `A6` at it).
4. Figures: `python -m derisk.a6_fig1`, `python figs/fig1.py`, `python figs/fig_shots.py`.

## License

To be set by the authors on the Zenodo record.
