import os
import numpy as np
from intercluster.rules import *
from intercluster.pruning import *
from intercluster import *
from intercluster.experiments import *

# Prevents memory leakage for KMeans:
os.environ["OMP_NUM_THREADS"] = "1"

#np.seterr(all='raise')
prune_cpu_count = 1
experiment_cpu_count = 16

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 342
np.random.seed(seed)


####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_anuran('data/anuran')

# Parameters:
k = 10
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
    'max_features' : data.shape[1]//4,
    'max_labels' : 1,
    'feature_pairings' : [list(range(data.shape[1]))],
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
    'max_features' : data.shape[1]//4,
    'max_labels' : 1,
    'feature_pairings' : [list(range(data.shape[1]))],
    'train_size' : 0.75
}

# Oblique Forest:
oblique_forest_tree_params = {
    'max_depth' : 2
}

oblique_forest_params = {
    'tree_model' : ObliqueTree,
    'tree_params' : oblique_forest_tree_params,
    'num_trees' : n_trees,
    'max_features' : 2,
    'max_labels' : 1,
    'feature_pairings' : [list(range(data.shape[1]))],
    'train_size' : 0.75
}


# SVM Forest:
forest_tree_params_svm = {}

forest_params_svm = {
    'tree_model' : SVMTree,
    'tree_params' : forest_tree_params_svm,
    'num_trees' : n_trees,
    'max_features' : 2,
    'max_labels' : 1,
    'feature_pairings' : [list(range(data.shape[1]))],
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
    'max_features' : data.shape[1],
    'max_labels' : k,
    'feature_pairings' : [list(range(data.shape[1]))],
    'train_size' : 1
}


# Voronoi Decision Set:
voronoi_params = {
    'centers' : C,
    'num_sets' : n_sets,
    'num_conditions' : 2,
    'feature_pairings' : [list(range(data.shape[1]))]
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
    'full_search' : False,
    'cpu_count' : prune_cpu_count
}


####################################################################################################

# Decision forest with axis aligned trees:

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

# Oblique rule sets:

# 3) Forest with simple oblique trees depth 2:
mod3 = DecisionSetMod(
    decision_set_model = DecisionForest,
    decision_set_params = oblique_forest_params,
    clustering = kmeans_base,
    prune_params = prune_params,
    min_frac_cover = min_frac_cover,
    name = 'Oblique-Forest'
)

# 4) SVM decision set
mod4 = DecisionSetMod(
    decision_set_model = DecisionForest,
    decision_set_params = forest_params_svm,
    clustering = kmeans_base,
    prune_params = prune_params,
    min_frac_cover = min_frac_cover,
    name = 'SVM'
)

# 5) Voronoi decision set
mod5 = DecisionSetMod(
    decision_set_model = VoronoiSet,
    decision_set_params = voronoi_params,
    clustering = kmeans_base,
    prune_params = prune_params,
    min_frac_cover = min_frac_cover,
    name = 'Voronoi'
)


####################################################################################################

# Forests with ExKMC Trees:

# 6) ExKMC Tree:
mod6 = DecisionSetMod(
    decision_set_model = DecisionForest,
    decision_set_params = forest_params_exkmc,
    clustering = kmeans_base,
    prune_params = prune_params,
    min_frac_cover = min_frac_cover,
    name = 'ExKMC-Forest'
)


####################################################################################################

# List of Modules and Measurements:

baseline_list = [kmeans_base, imm_base]
module_list = [mod1, mod2, mod4, mod6]

measurement_fns = [
    ClusteringCost(average = True, normalize = False),
    ClusteringCost(average = True, normalize = True),
    Overlap(),
    Coverage(),
    DistanceRatio(),
]


####################################################################################################
# Running the Experiment:

n_samples = 100

Ex1 = CoverageExperiment(
    data = data,
    baseline_list = baseline_list,
    module_list = module_list,
    measurement_fns = measurement_fns,
    n_samples = n_samples,
    labels = y,
    cpu_count = experiment_cpu_count,
    verbose = False
)

Ex1_results = Ex1.run(n_steps = 11, step_size = 0.05)
Ex1.save_results('data/experiments/anuran/', '')

####################################################################################################
