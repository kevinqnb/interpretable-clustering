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

np.seterr(all='raise')
cpu_count = 16

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 918717
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
n_rules = k
min_frac_cover = 0.5
n_trees = 1000
n_sets = 1000

####################################################################################################
# Baselines:

# KMeans:
kmeans_base = KMeansBase(n_clusters)
A,C = kmeans_base.assign(data)
y = labels_format(kmeans_base.clustering.labels_.astype(int))

# IMM:
imm_base = IMMBase(
    n_clusters = n_clusters,
    kmeans_model = kmeans_base.clustering
)

####################################################################################################
# Module Parameters:

# Depth 2 Forest:
forest_tree_params_depth_2 = {
    'max_depth' : 2
}

forest_params_depth_2 = {
    'tree_model' : SklearnTree,
    'tree_params' : forest_tree_params_depth_2,
    'num_trees' : n_trees,
    'max_features' : 6,
    'max_labels' : 1,
    'feature_pairings' : [list(range(12))] + [list(range(12,24))],
    'train_size' : 0.75
}

# Depth 5 Forest:
forest_tree_params_depth_5 = {
    'max_depth' : 5
}

forest_params_depth_5 = {
    'tree_model' : SklearnTree,
    'tree_params' : forest_tree_params_depth_5,
    'num_trees' : n_trees,
    'max_features' : 6,
    'max_labels' : 1,
    'feature_pairings' : [list(range(12))] + [list(range(12,24))],
    'train_size' : 0.75
}

# SVM Forest:
forest_tree_params_svm = {
    'max_depth' : 1
}

forest_params_svm = {
    'tree_model' : SVMTree,
    'tree_params' : forest_tree_params_svm,
    'num_trees' : n_trees,
    'max_features' : 2,
    'max_labels' : 1,
    'feature_pairings' : [list(range(12))] + [list(range(12,24))],
    'train_size' : 0.75
}


# ExKMC Forest:
forest_tree_params_exkmc = {
    'k' : k,
    'kmeans' : kmeans_base.clustering,
    'max_leaf_nodes' : 2*k,
    'imm' : True
}

forest_params_exkmc = {
    'tree_model' : ExkmcTree,
    'tree_params' : forest_tree_params_exkmc,
    'num_trees' : 1,
    'max_features' : 24,
    'max_labels' : k,
    'feature_pairings' : [list(range(24))],
    'train_size' : 1
}


# Voronoi Decision Set:
voronoi_params = {
    'centers' : C,
    'num_sets' : n_sets,
    'num_conditions' : k-1,
    'feature_pairings' : [list(range(12))] + [list(range(12,24))]
}


prune_objective = KmeansObjective(
    X = data,
    centers = C,
    average = True,
    normalize = True
)

prune_params = {
    'n_rules' : n_rules,
    'n_clusters' : n_clusters,
    'X' : data,
    'y' : y,
    'objective' : prune_objective,
    'lambda_search_range' : np.linspace(0,100,1001),
    'cpu_count' : cpu_count
}


####################################################################################################

# Decision Forest with Sklearn Trees:

# 1) depth 2:
mod1 = DecisionSetMod(
    decision_set_model = DecisionForest,
    decision_set_params = forest_params_depth_2,
    clustering = kmeans_base,
    prune_params = prune_params,
    min_frac_cover = min_frac_cover,
    name = 'Forest'
)


# 2) depth 5:
mod2 = DecisionSetMod(
    decision_set_model = DecisionForest,
    decision_set_params = forest_params_depth_5,
    clustering = kmeans_base,
    prune_params = prune_params,
    min_frac_cover = min_frac_cover,
    name = 'Forest-Depth-5'
)


####################################################################################################

# Forests with SVM Trees:

# 3) SVM Tree depth 1:
mod3 = DecisionSetMod(
    decision_set_model = DecisionForest,
    decision_set_params = forest_params_svm,
    clustering = kmeans_base,
    prune_params = prune_params,
    min_frac_cover = min_frac_cover,
    name = 'SVM-Forest'
)

####################################################################################################

# Forests with ExKMC Trees:

# 4) ExKMC Tree:
mod4 = DecisionSetMod(
    decision_set_model = DecisionForest,
    decision_set_params = forest_params_exkmc,
    clustering = kmeans_base,
    prune_params = prune_params,
    min_frac_cover = min_frac_cover,
    name = 'ExKMC-Forest'
)


####################################################################################################


# 5) Voronoi decision set
mod5 = DecisionSetMod(
    decision_set_model = VoronoiSet,
    decision_set_params = voronoi_params,
    clustering = kmeans_base,
    prune_params = prune_params,
    min_frac_cover = min_frac_cover,
    name = 'Voronoi-Set'
)


####################################################################################################

# List of Modules and Measurements:

baseline_list = [kmeans_base, imm_base]
module_list = [mod1, mod2, mod3, mod4, mod5]

measurement_fns = [
    ClusteringCost(average = True, normalize = False),
    ClusteringCost(average = True, normalize = True),
    Overlap(),
    Coverage(),
    DistanceRatio(),
]


####################################################################################################
# Running the Experiment:

n_samples = 1

Ex1 = CoverageExperiment(
    data = data,
    baseline_list = baseline_list,
    module_list = module_list,
    measurement_fns = measurement_fns,
    n_samples = n_samples,
    labels = y,
    verbose = True
)

Ex1_results = Ex1.run(n_steps = 11, step_size = 0.05)
Ex1.save_results('data/experiments/climate/', '_test')


####################################################################################################
