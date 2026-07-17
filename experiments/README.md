## Running Experiments:
The following directory contains all code for reproducing our experiments. 
Each sub-directory pertains to an individual dataset, and contains components for mining rules (`mine_rules.py`), choosing $\alpha$ values(`alphas.py`), and running experiment for which the maximum number of allowed rules is varied (`max_rules.py`) (performed in that order). Each is designed to cache results and make them reusable for subsequent experiments and plotting. These are saved to the `data/experiments` folder. WE DO NOT provide this data, since it is not memory efficient for uploading to a shared repository, and so to recompute our experiments one will need to run ALL of the steps outlined below.

#### 1. Mining for Rules:
The first step is creating an ensemble of rules to use for `PEC`. For each data directory, this may be done by running `mine_rules.py`. 

Note that for larger datasets, the algorithm which creates the discretized version of the dataset (called `bin_df`) for input to apriori may take a long time to run (~24 hours). It's best to cache this for future use, and we do so by saving to `data/experiments/{dataset}/rules/bin_df.csv`. 

Each of the four active miners (`decision_tree_rules.pkl`, `exkmc_rules.pkl`, `forest_rules.pkl`, `class_association_rules.pkl`) is loaded from its cached `.pkl` file instead of being re-mined whenever that cache is still valid for `data/experiments/{dataset}/rules/` -- this guarantees the pre-filter rule pool stays bit-for-bit identical across repeated runs, on top of the seeding already described under "Reproducibility" below. Cache validity is tracked by a `.params.json` sidecar saved next to each `.pkl`/`bin_df.csv` (e.g. `class_association_rules.params.json`), recording the exact parameters (`n_clusters`, `seed`, plus that miner's own hyperparameters -- e.g. `car_min_support`/`car_min_confidence`/`car_max_rule_length` for the CAR miner) that produced it. On each run the sidecar is compared against the current config values; a mismatch (including a missing sidecar, e.g. from a cache produced before this mechanism existed) is treated as stale and triggers a re-mine, so changing a single dataset parameter (like `CAR_MIN_SUPPORT`) only invalidates the miner(s) it actually affects, not the whole pool -- no manual deletion required. Delete a miner's `.pkl`/`.params.json` (or the whole `rules/` directory) to force a fresh mine regardless. This caching only covers the pre-filter mining stage -- the confidence-filtered pool (`ensemble_rules{OUTFILE_REF}.pkl`) is always recomputed from the cached pre-filter pool, since it must reflect the current `CONFIDENCE_DEFAULT`.

Shallow-tree rule mining (`shallow_rules.pkl`) is disabled (commented out in `mine_rules.py`) and no longer contributes to the pre-filter ensemble.

`mine_rules.py` filters the mined pool of rules by a confidence threshold (the fraction of a rule's covered points that share its majority cluster label), using a single fixed threshold, `CONFIDENCE_DEFAULT` in that dataset's `experiments/{dataset}/config.py` (currently `0.5` everywhere). There is no CLI flag and no sweep -- every dataset now follows `aniso`'s original single-confidence design. To run at a different confidence, edit `CONFIDENCE_DEFAULT` (and, to avoid overwriting the previous run's artifacts, `OUTFILE_REF` -- see below) and re-run the pipeline from `mine_rules.py` onward.

`mine_rules.py` saves:
- `ensemble_rules{OUTFILE_REF}.pkl` -- the filtered rule pool, a pickled list of rule objects.
- `ensemble_labels{OUTFILE_REF}.pkl` -- each rule's majority-cluster label.
- `cost_info_dict{OUTFILE_REF}.pkl.gz` -- $k$-Means cost, `mistake_info_dict{OUTFILE_REF}.pkl.gz` -- mistakes cost, `pairwise_distance_info_dict{OUTFILE_REF}.pkl.gz` -- pairwise distance cost. These are cached, pickled dictionaries keyed to the exact rule pool used to build them, so a differently-filtered pool needs its own cache -- they are not interchangeable across confidence values.

The unfiltered pool mined before any confidence filtering is saved once, untagged (i.e. without `OUTFILE_REF`), as `pre_filter_ensemble_rules.pkl` (and `pre_filter_ensemble_labels.pkl`, `bin_df.csv`, and the raw per-miner rule files) in every dataset -- these don't depend on the confidence threshold at all, so retagging them would force an expensive recompute (see the ~24h `bin_df` note above) for no benefit. This is what `confidence.py` (step 4 below) reads.

For more information about saving / loading rules, see `intercluster/rules.py` or `intercluster/decision_sets/objectives/objective.py` (which caches coverage and cost values).

#### `OUTFILE_REF`: distinguishing separate runs' artifacts

Every dataset's `config.py` defines `OUTFILE_REF` (currently `'_conf_50'` everywhere), a plain string suffix threaded through every confidence-*dependent* artifact filename any stage script saves or reads -- not just the top-level `exp*.json` results, but the filtered rule pool, its precomputed cost/mistake/pairwise-distance caches, IDS lambdas/caches, and selected alphas too. It is **not** tied to confidence specifically -- it's a general "which run is this" tag, so it's up to you to keep it in sync with whatever you change between runs (most commonly `CONFIDENCE_DEFAULT`, but it could be anything). If you switch confidence and don't also change `OUTFILE_REF`, the new run's artifacts will silently overwrite the old ones; give it a new value (e.g. `'_conf_75'`) to keep both around side by side. The confidence-*independent* pre-filter mining artifacts described above are the one exception -- they never get `OUTFILE_REF`.

#### 2. Choosing $\alpha$:
After creating a set of rules for `PEC`, we perform a hyperparameter search fo $\alpha$. This takes as input the cached rule information from the previous step, which is loaded at the beginning of each `alphas.py` file. Every script downstream of `mine_rules.py` in every dataset (`alphas.py`, `select_alphas.py`, `ids_lambda_search.py`, `max_rules.py`/`max_rules_exkmc.py`/`max_rules_combine.py`, `lambda.py`/`lambda_exkmc.py`/`lambda_combine.py`) reads its inputs and tags its own output files using that dataset's `OUTFILE_REF` (e.g. `exp{OUTFILE_REF}.json`, or -- for the mnist/fashion `*_combine.py` scripts -- the `main_ref`/`combine_refs`/`out_ref` they merge, each of which layers a `_dscluster`/`_exkmc` sub-tag underneath `OUTFILE_REF`). Scripts that dispatch fits through `Experiment`'s `joblib.Parallel` (`alphas.py`, `max_rules.py`, `max_rules_exkmc.py`, `lambda.py`, `lambda_exkmc.py`) read their worker count from `CPU_COUNT` in that dataset's `experiments/{dataset}/config.py` (see the "Pipeline runner" section below for where that value comes from). Results are then saved to the `data/experiments/{dataset}/alphas/` directory according to the `outfile` variable.

After running `alphas.py`, alpha selection by the elbow method can be done either interactively in `examples/experiments.ipynb` (the `select_alphas` cell) or, in every dataset, by running `select_alphas.py` -- the two implement the identical selection logic, so either can produce the `selected_alphas{OUTFILE_REF}.json` file the next step expects; `select_alphas.py` exists so the pipeline can be scripted end to end without a manual notebook pass.

#### Pipeline runner:
Every dataset (`aniso`/`anuran`/`climate`/`protein`/`yeast`/`mnist`/`fashion`) has its own `experiments/{dataset}/experiment_runner.py` and `experiments/{dataset}/config.py`, replacing the old per-dataset `run_confidence_sweep.sh` bash scripts. `config.py` holds the dataset's fixed constants, including the CPU-budget knobs:
- `TOTAL_CPU_COUNT` -- the whole-pipeline CPU budget, auto-detected via `os.cpu_count()` and overridable via a per-dataset env var (e.g. `ANURAN_TOTAL_CPU_COUNT`) or `experiment_runner.py`'s `--total-cpu-count` flag.
- `CPU_COUNT` -- the per-script default (defaults to `TOTAL_CPU_COUNT`, i.e. a standalone script run uses the whole budget); `experiment_runner.py` overrides this per subprocess via the dataset's `_CPU_COUNT` env var, dividing `TOTAL_CPU_COUNT` evenly across however many scripts run concurrently at each pipeline stage so no stage oversubscribes the machine (e.g. `total_cpu_count // len(concurrent_scripts)`) -- rather than each concurrent script hardcoding its own fixed share, as the old bash scripts did.

`experiment_runner.py` drives the full pipeline once (no confidence loop, no CLI confidence flag anywhere), running independent stages concurrently (`alphas.py`/`ids_lambda_search.py` don't depend on each other; neither do `max_rules.py`/`lambda.py`) while waiting on real dependencies (`select_alphas.py` needs `alphas.py`'s output; `max_rules.py`/`lambda.py` need both `select_alphas.py`'s and `ids_lambda_search.py`'s output), then runs `confidence.py` once at the end (it does its own internal 0.0-0.95 sweep over the pre-filter pool -- see step 4 below -- and only needs `CONFIDENCE_DEFAULT`'s selected-alphas/ids-lambdas files already on disk, not `max_rules.py`'s or `lambda.py`'s output):
- `aniso`/`anuran`/`climate`/`protein`/`yeast`: `mine_rules -> [alphas || ids_lambda_search] -> select_alphas -> [max_rules || lambda || confidence]`.
- `mnist`/`fashion`: same shape, but the final stage is two concurrently-running, internally-sequential families instead of a flat 3-way stage: `(max_rules -> max_rules_exkmc -> max_rules_combine)` and `(lambda -> lambda_exkmc -> lambda_combine)` run concurrently with each other, then `confidence.py` runs once after both finish -- each family runs as one thread so the `*_combine.py` merge always happens after both of its inputs exist, and the two families split the CPU budget between them. (`max_rules_exp.py`/`lambda_exp.py`, which used to fit a third companion, Exp-Tree, were removed -- see the note at the end of this section.)

Run e.g. `uv run python experiments/anuran/experiment_runner.py --total-cpu-count 15` to use 15 of a machine's cores for the whole run; omit the flag to auto-detect and use every core. To run at a different confidence value, edit `CONFIDENCE_DEFAULT`/`OUTFILE_REF` in `config.py` first (see above) -- there is no CLI override.

**`mnist`/`fashion` also have `experiment_runner_prep.py`**, an alternative runner that stops after `select_alphas.py` (i.e. `mine_rules -> [alphas || ids_lambda_search] -> select_alphas`) and does *not* run `max_rules.py`, `lambda.py`, `confidence.py`, or their companion `_exkmc`/`_combine` scripts. Use it when you'd rather run those three as their own individual experiments -- e.g. to time or debug them separately, or split them across machines -- instead of letting `experiment_runner.py` drive the whole thing end to end. It takes the same `--total-cpu-count` flag and reads the same `config.py`.

**Dead-weight cleanup (all datasets):** `examples/experiments.ipynb`'s active plotting cells only ever read `comparison_modules = ['Decision-Tree', 'ExKMC', 'IDS', 'CBA', 'CN2']` and `objective_names` restricted to the 3 unweighted PEC objectives (weighted objectives are only read once, at a single fixed rule budget, by the Uncertainty section). Every dataset's `max_rules.py`/`lambda.py`/`confidence.py` used to also fit Exp-Tree, Shallow-Tree, and WRA-weighted (plus, in `lambda.py`, plain WRA, and in `confidence.py`, the weighted objectives across every one of 20 confidence levels) despite none of that ever being plotted -- all now removed. `max_rules.py`/`lambda.py` also now restrict the weighted-objective PEC fits to the single rule budget the Uncertainty section actually reads, instead of sweeping the full rule-budget/lambda range for no consumer. For `mnist`/`fashion` specifically, this meant deleting `max_rules_exp.py`/`lambda_exp.py` entirely -- their sole purpose was fitting Exp-Tree -- and updating `max_rules_combine.py`/`lambda_combine.py` to no longer expect an `_exp` ref to merge.

#### 3. Varying Maximum Rules
We evaluate our algorithms across settings where the maximum number of allowed rules is varied by running `max_rules.py`. This takes as input both the mined rules from step 1 and the alpha parameters selected in step 2. Results are then saved to `data/experiments/'dataset'/max_rules/` directory according to the `outfile` variable. These may then be loaded to plot results in `examples/experiments.ipynb`. 

NOTE: That for the `mnist` and `fashion` datasets we split computation across different files, since some algorithms took much longer to run. In these cases, one would run `max_rules.py` and `max_rules_exkmc.py` in any order, and then combine with `max_rules_combine.py`.

#### 4. Varying the minimum confidence threshold
`confidence.py` sweeps the minimum-confidence threshold used to filter mined tree rules (0.0 to 1.0 in steps of 0.05), refitting all algorithms at each threshold to see how rule quality/filtering affects results. It takes as input the mined rules from step 1 and the alpha values selected in step 2, and saves results to `data/experiments/{dataset}/confidence/exp_confidence{OUTFILE_REF}.json`. Unlike `max_rules.py`, the rule pool changes at every threshold, so pool-dependent algorithms (PEC, CBA, IDS) are refit from scratch each iteration rather than relying on precomputed coverage/cost caches.

## Reproducibility

Every experiment script sets a single global `seed` once near the top (see the "REMINDER" comment in each file) and threads it into every class that accepts a `random_state`/seed parameter (`KMeansBase`, `DecisionTree`, `ShallowTree`, `ExplanationTree`, `IDS`, `RandomForestMiner`, `entropy_bin`). Two properties matter for reproducing a result exactly:

1. **Every class with internal randomness accepts its own explicit seed.** Nothing should rely on the ambient global NumPy RNG state (`np.random.seed(...)` alone). This matters in particular because `alphas.py`/`max_rules.py` dispatch model fitting across `joblib.Parallel(backend='loky')` worker processes (see `Experiment.run()` in `experiment.py`), and those workers do **not** inherit the main process's seeded global state. A class that only calls bare `np.random.*` will silently produce different results on every run once it's fit inside a `Parallel` worker, even with a fixed `seed` in `fixed_parameters`.
2. **Models whose fitted solution has inherent randomness are evaluated across multiple trials.** A single point estimate from one arbitrarily-seeded fit doesn't tell you how sensitive the result is to that choice of seed.

**Stochastic vs. deterministic models**, as currently used in `experiments/climate/`:

| Model | Stochastic? | Source of randomness | Seed parameter |
|---|---|---|---|
| `PEC` (all objective variants) | No | distorted/lazy greedy have no randomness | n/a |
| `ExkmcTree` (ExKMC/IMM) | No, as used here | only random when it fits its own internal KMeans; climate always passes a pre-fit `kmeans` object | n/a |
| `CN2`, `CBA`, `WRABaseline` | No | pure greedy/scoring, no RNG | n/a |
| `DecisionTree` | Yes | sklearn's internal tie-breaking | `random_state` |
| `IDS` | Yes | randomized-greedy / SLS selection (and, if lambdas aren't fixed, coordinate-ascent lambda search) | `random_state` |

For the climate experiments, `max_rules.py` and `confidence.py` derive a fixed list of per-trial seeds (`trial_seeds = [seed + i for i in range(n_trials)]`, `n_trials = 10`) from the single master `seed`, and refit each stochastic model once per trial seed. Deterministic models are fit once.

**`ShallowTree`/`ExplanationTree`** are stochastic in the same way (`kmeans_random_state`/`random_state` respectively -- see the caveat below), but no longer appear in any dataset's `max_rules.py`/`lambda.py`/`confidence.py`: neither is in `examples/experiments.ipynb`'s `comparison_modules`, so fitting/refitting them across every rule budget or confidence level was pure waste (see the "Dead-weight cleanup" note above). They remain available as library classes (`intercluster.decision_trees`) for ad hoc use outside the experiment scripts.

**Known caveat -- Cython-level randomness in `ExplanationTree`.** `ExplanationSplitter`'s compiled Cython backend (`split_cy` and `get_split_outliers_cy` in `decision_trees/splitters/cython/explanation.pyx`) has its own tie-breaks (`np.random.randint`/`np.random.uniform`) that read the *global* NumPy RNG state directly -- they are not parameterized by `ExplanationTree`'s `random_state` argument at all (that argument only controls the base `Tree` class's heap tie-break). The mitigation used here: `max_rules.py`/`confidence.py` fit these stochastic modules single-process (not through `Experiment`'s joblib dispatch) specifically so that calling `np.random.seed(trial_seed)` immediately before each trial's `fit()` call -- which they do -- fully pins down both the Python-level and Cython-level randomness for that trial. The same pattern (`InformationGainSplitter`/`ObliqueInformationGainSplitter` in `information_gain.pyx`) exists in `ID3Tree`/`ObliqueTree`, which climate's scripts don't use, but a future user of those classes should apply the same mitigation.

**Output schema:** for stochastic models, every metric in the output JSON (`exp_rule_length.json`, `exp_confidence.json`) is reported as `{"mean": ..., "std": ..., "values": [...]}` across the `n_trials` trials, instead of a bare float. Deterministic models (`PEC`, `ExKMC`, `CN2`, `CBA`, `WRA`) keep bare floats.

`mine_rules.py` (rule mining) and `alphas.py` ($\alpha$ selection) are one-time, cached hyperparameter-selection steps rather than models under evaluation, so they are run once under the master seed rather than repeated across trials -- the same treatment as `ids_lambda_search.py`'s IDS-lambda coordinate ascent.

## Selected parameters:
For reference we outline the parameters used in each of our experiments. `mine_rules.py` filters at the single `filter_confidence` scalar shown below (`CONFIDENCE_DEFAULT` in that dataset's `config.py`) -- see the `OUTFILE_REF` note above for how to switch to a different confidence value without overwriting a previous run's artifacts.
```
{
    'n': dataset size,
    'd': dataset features,
    'n_clusters': selected number of clusters,
    'n_select': number of rules to select (when running alphas.py),
    'max_rules': maximum number of rules (when incrementing in max_rules.py),
    'shallow_tree_depth_factor': depth for the Shallow-Tree algorithm,
    'n_forest': number of trees to use in the random forest,
    'forest_max_depth': maximum depth to use in the random forest,
    'car_min_support': minimum support for the apriori algorithm,
    'car_min_confidence': minimum confidence for the apriori algorithm,
    'car_max_rule_length': maximum rule length for the apriori algorithm, 
    'filter_confidence': confidence level at which we filter tree rules,
    'seed': random seed generator
}
```

* Number of clusters are chosen either by an elbow heuristic in `examples/datasets.ipynb`, or (in the case of MNIST and Fashion MNIST) to simply match the number of ground truth labels for the dataset.
* The minimum support and confidence parameters for class association rule mining are chosen to produce a set of rules which are diverse enough to be effective, while still maintaining efficient computational performance. 
* The depth factor for a shallow tree is consistently chosen as 0.03, as suggested in the original paper.
* The number of trees in our random forest is consistently chosen as 100. We limit the depth of these trees to 6, since we wouldn't want an explainable rule to be much longer than this. 
* Before passing the ensemble of rules to PIC, we first filter by confidence. This is done to remove lower quality rules from nodes early on in the mined trees, and is chosen consistently with the confidence for apriori.

### Climate, kmeans clustering:
```
{
    'n': n,
    'd': d,
    'n_clusters': 6,
    'n_select': 6,
    'max_rules': 12,
    'shallow_tree_depth_factor': 0.03,
    'n_forest': 100,
    'forest_max_depth': 6,
    'car_min_support': 0.025,
    'car_min_confidence': 0.85,
    'car_max_rule_length': 3, # (really means 6 by pyfim convention)
    'filter_confidence': 0.85,
    'seed': seed
}
```

### Anuran, kmeans clustering:
```
{
    'n': n,
    'd': d,
    'n_clusters': 5,
    'n_select': 5,
    'max_rules': 11,
    'shallow_tree_depth_factor': 0.03,
    'n_forest': 100,
    'forest_max_depth': 6,
    'car_min_support': 0.025,
    'car_min_confidence': 0.85,
    'car_max_rule_length': 3, # (really means 6 by pyfim convention)
    'filter_confidence': 0.85,
    'seed': seed
}
```

### Protein, kmeans clustering:
```
{
    'n': n,
    'd': d,
    'n_clusters': 6,
    'n_select': 6,
    'max_rules': 12,
    'shallow_tree_depth_factor': 0.03,
    'n_forest': 100,
    'forest_max_depth': 6,
    'car_min_support': 0.05, # Little bit larger for this dataset, which explodes in rule length otherwise
    'car_min_confidence': 0.85,
    'car_max_rule_length': 3, # (really means 6 by pyfim convention)
    'filter_confidence': 0.85,
    'seed': seed
}
```

### Yeast, kmeans clustering:
```
{
    'n': n,
    'd': d,
    'n_clusters': 9,
    'n_select': 9,
    'max_rules': 15,
    'shallow_tree_depth_factor': 0.03,
    'n_forest': 100,
    'forest_max_depth': 6,
    'car_min_support': 0.025,
    'car_min_confidence': 0.85,
    'car_max_rule_length': 3, # (really means 6 by pyfim convention)
    'filter_confidence': 0.85,
    'seed': seed
}
```

### MNIST, kmeans clustering:
```
{
    'n': n,
    'd': d,
    'n_clusters': 10,
    'n_select': 10,
    'max_rules': 16,
    'shallow_tree_depth_factor': 0.03,
    'n_forest': 100,
    'forest_max_depth': 6,
    'car_min_support': 0.025,
    'car_min_confidence': 0.65,
    'car_max_rule_length': 2, # (really means 4 by pyfim convention)
    'filter_confidence': 0.65,
    'seed': seed
}
```

### Fashion MNIST, kmeans clustering:
```
{
    'n': n,
    'd': d,
    'n_clusters': 10,
    'n_select': 10,
    'max_rules': 16,
    'shallow_tree_depth_factor': 0.03,
    'n_forest': 100,
    'forest_max_depth': 6,
    'car_min_support': 0.025,
    'car_min_confidence': 0.75,
    'car_max_rule_length': 2, # (really means 4 by pyfim convention)
    'filter_confidence': 0.75,
    'seed': seed
}
```