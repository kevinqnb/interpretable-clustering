import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd
from ExKMC.Tree import Tree as ExTree
from intercluster.rules import *
from intercluster.pruning import *
from intercluster import *
from intercluster.experiments import *


# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 919817
np.random.seed(seed)


####################################################################################################
# Read and process data:

shapefile_path = 'data/climate_divisions/climate_divisions.shp'
dtype_dict = {'CLIMDIV': 'str'}
gdf = gpd.read_file(shapefile_path, dtype = dtype_dict)
gdf['CLIMDIV'] = gdf['CLIMDIV'].apply(lambda x: f'{int(x):04d}')

climate_data = pd.read_csv('data/climate.csv', dtype={'ID': str, 'Year': str})
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

# Parameters:
k = 7
n_clusters = k
min_rules = k
max_rules = 2*k
min_depth = 3
max_depth = 6
n_trees = 1000

####################################################################################################
# Baselines:

# KMeans:
kmeans_base = KMeansBase(n_clusters)
A,C = kmeans_base.assign(data)
y = kmeans_base.clustering.labels_

# Exkmc:
exkmc_mod = ExkmcMod(
    n_clusters = n_clusters,
    kmeans_model = kmeans_base.clustering,
    base_tree = 'IMM',
    min_rules = min_rules
)

####################################################################################################

# Decision Forest with Sklearn Trees:

# 1) depth 2, 90% coverage:
forest_tree_params1 = {
    'max_leaf_nodes' : max_rules,
    'max_depth' : 2
}

forest_params1 = {
    'tree_model' : SklearnTree,
    'tree_params' : forest_tree_params1,
    'num_trees' : n_trees,
    'max_features' : 6,
    'max_labels' : 1,
    'feature_pairings' : [list(range(12))] + [list(range(12,24))],
    'train_size' : 0.75
}

prune_objective1 = KmeansObjective(
    X = data,
    centers = C,
    normalize = True,
    threshold = 0.9
)

prune_params1 = {
    'k' : k,
    'X' : data,
    'y' : [[l] for l in y],
    'objective' : prune_objective1,
    'lambda_search_range' : np.linspace(0,2,101)
}

mod1 = ForestMod(
    forest_model = DecisionForest,
    forest_params = forest_params1,
    clustering = kmeans_base,
    prune_params = prune_params1,
    min_rules = min_rules,
    min_depth = min_depth,
    max_rules = max_rules,
    max_depth = max_depth,
    name = 'Forest'
)

####################################################################################################

# List of Modules and Measurements:
comparison_module = mod1
baseline_list = [kmeans_base]
module_list = [exkmc_mod]

measurement_fns = [
    ClusteringCost(average = True)
]


####################################################################################################
# Running the Experiment:

n_samples = 1000

Ex1 = RulesExperimentV2(
    data = data,
    comparison_module = comparison_module,
    baseline_list = baseline_list,
    module_list = module_list,
    measurement_fns = measurement_fns,
    n_samples = n_samples,
    labels = y,
    random_seed = seed,
    verbose = False
)

Ex1_results = Ex1.run(min_rules = min_rules, max_rules = max_rules)
Ex1.save_results('data/experiments/decision_sets/', '_climate')


####################################################################################################
