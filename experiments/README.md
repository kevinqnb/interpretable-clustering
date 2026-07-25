## Running Experiments

This directory reproduces our experiments. Each dataset (`aniso`, `anuran`, `climate`, `protein`, `yeast`, `mnist`, `fashion`) has its own subdirectory with an `experiment_runner.py` and `config.py`, and follows the same pipeline:

1. **`mine_rules.py`** -- mines and caches a candidate rule ensemble for `PEC`, filtered to a confidence threshold (`CONFIDENCE_DEFAULT` in that dataset's `config.py`).
2. **`alphas.py` / `ids_lambda_search.py`** -- search for PEC's `alpha` hyperparameter and IDS's lambda weights (run concurrently, independent of each other). Note that the IDS search space is large and this will take a while to run for larger datasets.
3. **`select_alphas.py`** -- picks a final alpha per objective from `alphas.py`'s sweep (elbow method).
4. **`max_rules.py` / `lambda.py` / `confidence.py`** -- the actual experiments: performance as the rule budget, IDS's lambda tradeoff, and the mining confidence threshold are each varied in turn. Some datasets add further sweeps at this stage (e.g. aniso's `input_sensitivity.py`, varying rule-pool composition) -- check `experiment_runner.py`'s final stage for the exact set.

Run a dataset's full pipeline with, e.g.:
```
uv run python experiments/climate/experiment_runner.py --total-cpu-count 8
```
(omit the flag to auto-detect and use every core). Each stage caches its output under `data/experiments/{dataset}/`, which isn't included in this repo (too large).

`mnist`/`fashion` split several stages across multiple per-algorithm scripts (some algorithms are much slower to fit than others) and merge the parts with a `*_combine.py` script. This split changes as algorithms are added or dropped, so check each dataset's own scripts for the current breakdown rather than relying on this README.

Every dataset also has `ids_lambda_search_alt.py`/`max_rules_ids_alt.py`, an optional alternate pipeline that tunes IDS against PEC's own objective instead of held-out AUC. It isn't wired into `experiment_runner.py` -- run it by hand after `select_alphas.py`. See the comments in `ids_lambda_search_alt.py` for details.

### Config and output tags

Each dataset's `config.py` holds its fixed parameters (cluster count, rule-mining thresholds, CPU budget, etc.) -- see that file for the exact values used. Some things worth knowing about:
- `CONFIDENCE_DEFAULT` -- the confidence threshold `mine_rules.py` filters the ensemble with.
- `OUTFILE_REF` -- a filename suffix threaded through every confidence-dependent artifact. Give it a new value whenever you change `CONFIDENCE_DEFAULT` (or anything else you want a fresh, non-overwriting run for); otherwise the new run silently overwrites the old one's output.

### Reproducibility

Every script seeds a single global `seed` once near the top (see the `REMINDER` comment in each file) and threads it explicitly into every model with internal randomness, since `joblib` worker processes don't inherit the main process's seeded global NumPy state. Stochastic models are refit across several trial seeds and reported as `{mean, std, values}`; deterministic models are fit once. See `Experiment` in `experiment.py` and each dataset's `max_rules.py`/`confidence.py` for more detail.
