import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import copy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd
from ExKMC.Tree import Tree as ExTree
from ruleclustering import *


# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 13982
np.random.seed(seed)


####################################################################################################
# Read and process data:

shapefile_path = '../data/climate_divisions/climate_divisions.shp'
dtype_dict = {'CLIMDIV': 'str'}
gdf = gpd.read_file(shapefile_path, dtype = dtype_dict)
gdf['CLIMDIV'] = gdf['CLIMDIV'].apply(lambda x: f'{int(x):04d}')

climate_data = pd.read_csv('../data/climate.csv', dtype={'ID': str, 'Year': str})
climate_data.set_index(['ID', 'Year'], inplace=True)

historical_years = [str(i) for i in range(1900,2000)]
recent_years = [str(i) for i in range(2013,2024)]

historical = climate_data.loc[pd.IndexSlice[:, historical_years], :]
recent = climate_data.loc[pd.IndexSlice[:, recent_years], :]

historical_avg = historical.groupby(level='ID').mean()
recent_avg = recent.groupby(level='ID').mean()

climate_change = (recent_avg - historical_avg)/historical_avg
climate_change = climate_change.loc[gdf.CLIMDIV,:]

# change months to seasons\n",
groupings = {'pcpn_winter': ['pcpn_dec', 'pcpn_jan', 'pcpn_feb'],
            'pcpn_spring': ['pcpn_mar', 'pcpn_apr', 'pcpn_may'],
            'pcpn_summer': ['pcpn_june', 'pcpn_july', 'pcpn_aug'],
            'pcpn_fall': ['pcpn_sept', 'pcpn_oct', 'pcpn_nov'],
            'temp_winter': ['temp_dec', 'temp_jan', 'temp_feb'],
            'temp_spring': ['temp_mar', 'temp_apr', 'temp_may'],
            'temp_summer': ['temp_june', 'temp_july', 'temp_aug'],
            'temp_fall': ['temp_sept', 'temp_oct', 'temp_nov']}

seasonal_historical = pd.DataFrame()
seasonal_recent = pd.DataFrame()
seasonal_climate_change = pd.DataFrame()

# Calculate the average for each group of months
for group_name, columns in groupings.items():
    seasonal_historical[group_name] = historical_avg[columns].mean(axis=1)
    seasonal_recent[group_name] = recent_avg[columns].mean(axis=1)
    seasonal_climate_change[group_name] = climate_change[columns].mean(axis=1)
    
# Normalize the data
data = climate_change.to_numpy()
feature_labels = climate_change.columns

scaler = MinMaxScaler()
X = scaler.fit_transform(data)


####################################################################################################
# Define Parameters and experiment modules:

n_clusters = 7
min_rules = n_clusters
max_rules = 2*n_clusters
min_depth = 3
max_depth = 4

# Base module
base = KMeansBase(n_clusters)
A,C = base.assign(X)

cluster_model = KMeansRuleClustering
cluster_params = {
    'k_clusters' : n_clusters,
    'init' : 'manual',
    'center_init' : C,
    'max_iterations' : 500,
}

# Modules:

# Tree:
tree_params = {
    'splits' : 'axis',
    'max_leaf_nodes' : max_rules,
    'max_depth' : max_depth,
    'norm' : 2,
    'center_init' : 'k-means',
    'n_centers' : n_clusters
}

mod1 = TreeMod(
    tree_model = CentroidTree,
    tree_params = tree_params,
    cluster_model = cluster_model,
    cluster_params = cluster_params,
    min_rules = min_rules,
    min_depth = min_depth,
    name = 'Centroid'
)


# Forest Tree:
'''
forest_tree_params = {
    'splits' : 'axis',
    'max_leaf_nodes' : max_rules,
    'max_depth' : 2,
    'norm' : 2,
    'center_init' : 'k-means',
    'n_centers' : n_clusters
}
'''

forest_tree_params = {
    'data_labels' : base.clustering.labels_,
    'max_leaf_nodes' : max_rules,
    'max_depth' : 2
}


# Forest:
forest_params = {
    'tree_model' : SklearnTree,
    'tree_params' : forest_tree_params,
    'num_trees' : 10000,
    'max_features' : 6,
    'feature_pairings' : [list(range(12))] + [list(range(12,24))],
    'train_size' : 0.75
}

prune_params = {
    'search_range' : np.linspace(0,1,100),
    'coverage_threshold' : 0.9 * len(X)
}

mod2 = ForestMod(
    forest_model = DecisionForest,
    forest_params = forest_params,
    cluster_model = cluster_model,
    cluster_params = cluster_params,
    assignment_method = 'vote',
    prune_params = None,
    min_rules = min_rules,
    min_depth = min_depth,
    max_rules = max_rules,
    max_depth = max_depth,
    name = 'Forest-Vote'
)

mod3 = ForestMod(
    forest_model = DecisionForest,
    forest_params = forest_params,
    cluster_model = cluster_model,
    cluster_params = cluster_params,
    assignment_method = 'vote',
    prune_params = prune_params,
    min_rules = min_rules,
    min_depth = min_depth,
    max_rules = max_rules,
    max_depth = max_depth,
    name = 'Forest-Vote-Pruned'
)

mod4 = ForestMod(
    forest_model = DecisionForest,
    forest_params = forest_params,
    cluster_model = cluster_model,
    cluster_params = cluster_params,
    assignment_method = 'min',
    prune_params = None,
    min_rules = min_rules,
    min_depth = min_depth,
    max_rules = max_rules,
    max_depth = max_depth,
    name = 'Forest-Min'
)

mod5 = ForestMod(
    forest_model = DecisionForest,
    forest_params = forest_params,
    cluster_model = cluster_model,
    cluster_params = cluster_params,
    assignment_method = 'min',
    prune_params = prune_params,
    min_rules = min_rules,
    min_depth = min_depth,
    max_rules = max_rules,
    max_depth = max_depth,
    name = 'Forest-Min-Pruned'
)


baseline_list = [base]
module_list = [mod1, mod2, mod3, mod4, mod5]


cost_fns = {
    'standard' : kmeans_cost,
    'normalized' : kmeans_cost,
    'normalized_v2' : kmeans_cost
}

cost_fn_params = {
    'standard' : {},
    'normalized' : {'normalize' : True},
    'normalized_v2' : {'normalize' : True, 'square' : 'True'}
}


####################################################################################################
# Changing the number of rules experiment:

Ex1 = RulesExperiment(
    data = X,
    baseline_list = baseline_list,
    module_list = module_list,
    cost_fns = cost_fns,
    cost_fn_params = cost_fn_params,
    verbose = True
)

Ex1_results = Ex1.run(min_rules = min_rules, max_rules = max_rules)
Ex1.save_results('../data/experiments/decision_sets/', '_climate_sklearn_d2')


####################################################################################################
# Changing the depth experiment

'''
Ex2 = DepthExperiment(
    data = X,
    baseline_list = baseline_list,
    module_list = module_list,
    cost_fns = cost_fns,
    cost_fn_params = cost_fn_params,
    random_seed = seed,
    verbose = True
)

Ex2_results = Ex2.run(min_depth = min_depth, max_depth = max_depth)
Ex2.save_results('../data/experiments/decision_sets/', '_climate_rand_d6')
'''