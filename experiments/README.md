## Running Experiments:
The following directory contains all code for reproducing our experiments. 
Each sub-directory pertains to an individual dataset, and contains components for mining rules (`mine_rules.py`), choosing $\alpha$ values(`alphas.py`), and running experiment for which the maximum number of allowed rules is varied (`max_rules.py`) (performed in that order). Each is designed to cache results and make them reusable for subsequent experiments and plotting. These are saved to the `data/experiments` folder. WE DO NOT provide this data, since it is not memory efficient for uploading to a shared repository, and so to recompute our experiments one will need to run ALL of the steps outlined below.

#### 1. Mining for Rules:
The first step is creating an ensemble of rules to use for `PEC`. For each data directory, this may be done by running `mine_rules.py`. 

Note that for larger datasets, the algorithm which creates the discretized version of the dataset (called `bin_df`) for input to apriori may take a long time to run (~24 hours). It's best to cache this for future use, and we do so by saving to `data/experiments/'dataset'/rules/bin_df.csv`. 

Likewise, rather than recomputing the coverage and cost 
scores for our mined rules, we pre-compute and cache these 
values using pickled dictionaries saved to the same rules directory (`cost_info_dict.pkl.gz` --$k$-Means cost, `mistake_info_dict.pkl.gz` -- mistakes cost, `pairwise_info_dict.pkl.gz` -- pairwise distance cost)

Most importantly the mined set of ensemble rules is also saved to this directory as a pickled list of rule objects: `ensemble_rules.pkl`. 

For more information about saving / loading rules, see `intercluster/rules.py` or `intercluster/decision_sets objectives/objective.py` (which caches coverage and cost values).

#### 2. Choosing $\alpha$:
After creating a set of rules for `PEC`, we perform a hyperparameter search fo $\alpha$. This takes as input the cached rule information from the previous step, which is loaded at the beginning of each `alphas.py` file. Results are 
then saved to the `data/experiments/'dataset'/alphas/` directory according to the `outfile` variable.

After running `alphas.py`, we plot the results and select values based upon an elbow heurisitic in `examples/experiments.ipynb`. These may then be saved and used for the next step.

#### 3. Varying Maximum Rules
Finally, we evaluate our algorithms across settings where the maximum number of allowed rules is varied by running `max_rules.py`. This takes as input both the mined rules from step 1 and the alpha parameters selected in step 2. Results are then saved to `data/experiments/'dataset'/max_rules/` directory according to the `outfile` variable. These may then be loaded to plot results in `examples/experiments.ipynb`. 

NOTE: That for the `mnist` and `fashion` datasets we split computation across different files, since some algorithms took much longer to run. In these cases, one would run `max_rules.py`, `max_rules_exkmc.py`, and `max_rules_exp.py` in any order, and then comine with `max_rules_combine.py`.

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