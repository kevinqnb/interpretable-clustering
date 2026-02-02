## Selected parameters:

* Number of clusters are chosen either to minimize clustering cost whilst avoiding clust overfit,
    or (in the case of MNIST and Fashion MNIST) to simply match the number of ground truth labels for the dataset.
    Our analysis is shown in `examples/datasets.ipynb` or in `examples/climate.ipynb`.
* The minimum support and confidence parameters for class association rule mining are chosen to 
    produce a set of rules which we see as being large enough to be effective, while still maintaining efficient
    computational performance. 
* The depth factor for a shallow tree is consistently chosen as 0.03, as suggested in the 
    original paper.
* The number of trees in our random forest is consistently chosen as 100. We limit the depth of 
    these trees to 6, since we wouldn't want an explainable rule to be much longer than this. 
* Before passing the ensemble of rules to PIC, we first filter to include rules with confidence 
    $\geq 50\%$. This is mainly to to improve computational efficiency, especially for 
    larger datasets. It also helps filter out early nodes in a decision tree. 

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

### KDD Cup, kmeans clustering:
```
{
    'n': n,
    'd': d,
    'n_clusters': 20,
    'n_select': 20,
    'max_rules': 26,
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