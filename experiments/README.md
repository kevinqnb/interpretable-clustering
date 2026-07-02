## Running Experiments:
The following directory contains all code for reproducing our experiments. 
Each sub-directory pertains to an individual dataset, and contains components for mining rules (`mine_rules.py`), choosing $\alpha$ values(`alphas.py`), and running experiment for which the maximum number of allowed rules is varied (`max_rules.py`) (performed in that order). Each is designed to cache results and make them reusable for subsequent experiments and plotting. These are saved to the `data/experiments` folder. WE DO NOT provide this data, since it is not memory efficient for uploading to a shared repository, and so to recompute our experiments one will need to run ALL of the steps outlined below.

#### 1. Mining for Rules:
The first step is creating an ensemble of rules to use for `PEC`. For each data directory, this may be done by running `mine_rules.py`. 

Note that for larger datasets, the algorithm which creates the discretized version of the dataset (called `bin_df`) for input to apriori may take a long time to run (~24 hours). It's best to cache this for future use, and we do so by saving to `data/experiments/{dataset}/rules/bin_df.csv`. 

Likewise, rather than recomputing the coverage and cost 
scores for our mined rules, we pre-compute and cache these 
values using pickled dictionaries saved to the same rules directory (`cost_info_dict.pkl.gz` --$k$-Means cost, `mistake_info_dict.pkl.gz` -- mistakes cost, `pairwise_info_dict.pkl.gz` -- pairwise distance cost)

Most importantly the mined set of ensemble rules is also saved to this directory as a pickled list of rule objects: `ensemble_rules.pkl`. 

For more information about saving / loading rules, see `intercluster/rules.py` or `intercluster/decision_sets/objectives/objective.py` (which caches coverage and cost values).

#### 2. Choosing $\alpha$:
After creating a set of rules for `PEC`, we perform a hyperparameter search fo $\alpha$. This takes as input the cached rule information from the previous step, which is loaded at the beginning of each `alphas.py` file. Results are 
then saved to the `data/experiments/{dataset}/alphas/` directory according to the `outfile` variable.

After running `alphas.py`, we plot the results and select values based upon an elbow heurisitic in `examples/experiments.ipynb`. These may then be saved and used for the next step.

#### 3. Varying Maximum Rules
We evaluate our algorithms across settings where the maximum number of allowed rules is varied by running `max_rules.py`. This takes as input both the mined rules from step 1 and the alpha parameters selected in step 2. Results are then saved to `data/experiments/'dataset'/max_rules/` directory according to the `outfile` variable. These may then be loaded to plot results in `examples/experiments.ipynb`. 

NOTE: That for the `mnist` and `fashion` datasets we split computation across different files, since some algorithms took much longer to run. In these cases, one would run `max_rules.py`, `max_rules_exkmc.py`, and `max_rules_exp.py` in any order, and then comine with `max_rules_combine.py`.

#### 4. Varying the minimum confidence threshold
`confidence.py` sweeps the minimum-confidence threshold used to filter mined tree rules (0.0 to 1.0 in steps of 0.05), refitting all algorithms at each threshold to see how rule quality/filtering affects results. It takes as input the mined rules from step 1 and the alpha values selected in step 2, and saves results to `data/experiments/{dataset}/confidence/exp_confidence.json`. Unlike `max_rules.py`, the rule pool changes at every threshold, so pool-dependent algorithms (PEC, WRA, CBA, IDS) are refit from scratch each iteration rather than relying on precomputed coverage/cost caches.

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
| `ShallowTree` | Yes | internal KMeans re-initialization | `kmeans_random_state` |
| `ExplanationTree` | Yes | heap tie-break for equal-gain splits | `random_state` |
| `IDS` | Yes | randomized-greedy / SLS selection (and, if lambdas aren't fixed, coordinate-ascent lambda search) | `random_state` |

For the climate experiments, `max_rules.py` and `confidence.py` derive a fixed list of per-trial seeds (`trial_seeds = [seed + i for i in range(n_trials)]`, `n_trials = 10`) from the single master `seed`, and refit each stochastic model once per trial seed. Deterministic models are fit once.

**Known caveat -- Cython-level randomness in `ExplanationTree`.** `ExplanationSplitter`'s compiled Cython backend (`split_cy` and `get_split_outliers_cy` in `decision_trees/splitters/cython/explanation.pyx`) has its own tie-breaks (`np.random.randint`/`np.random.uniform`) that read the *global* NumPy RNG state directly -- they are not parameterized by `ExplanationTree`'s `random_state` argument at all (that argument only controls the base `Tree` class's heap tie-break). The mitigation used here: `max_rules.py`/`confidence.py` fit these stochastic modules single-process (not through `Experiment`'s joblib dispatch) specifically so that calling `np.random.seed(trial_seed)` immediately before each trial's `fit()` call -- which they do -- fully pins down both the Python-level and Cython-level randomness for that trial. The same pattern (`InformationGainSplitter`/`ObliqueInformationGainSplitter` in `information_gain.pyx`) exists in `ID3Tree`/`ObliqueTree`, which climate's scripts don't use, but a future user of those classes should apply the same mitigation.

**Output schema:** for stochastic models, every metric in the output JSON (`exp_rule_length.json`, `exp_confidence.json`) is reported as `{"mean": ..., "std": ..., "values": [...]}` across the `n_trials` trials, instead of a bare float. Deterministic models (`PEC`, `ExKMC`, `CN2`, `CBA`, `WRA`) keep bare floats.

`mine_rules.py` (rule mining) and `alphas.py` ($\alpha$ selection) are one-time, cached hyperparameter-selection steps rather than models under evaluation, so they are run once under the master seed rather than repeated across trials -- the same treatment as `ids_lambda_search.py`'s IDS-lambda coordinate ascent.

## Selected parameters:
For reference we outline the parameters used in each of our experiments. 
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