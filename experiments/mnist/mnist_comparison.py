import os
import numpy as np
from intercluster.rules import *
from intercluster.pruning import *
from intercluster.utils import *
from intercluster.experiments import *

# Prevents memory leakage for KMeans:
os.environ["OMP_NUM_THREADS"] = "1"

#np.seterr(all='raise')
prune_cpu_count = 1
experiment_cpu_count = 12

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 342
np.random.seed(seed)


####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_mnist()
n,d = data.shape

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
imm_assign, imm_centers = imm_base.assign(data)
imm_depth = int(imm_base.weighted_average_rule_length)

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
    'max_features' : d,
    'max_labels' : 1,
    'max_depths' : [2],
    'feature_pairings' : [list(range(d))],
    'train_size' : 0.75
}

# Depth IMM Forest:
forest_tree_params_depth_imm = {
    'max_depth' : imm_depth
}

forest_params_depth_imm = {
    'tree_model' : SklearnTree,
    'tree_params' : forest_tree_params_depth_imm,
    'num_trees' : n_trees,
    'max_features' : d,
    'max_labels' : 1,
    'max_depths' : [imm_depth],
    'feature_pairings' : [list(range(d))],
    'train_size' : 0.75
}

# SVM set:
svm_params = {
    'num_rules' : n_sets,
    'num_features' : 2,
    'feature_pairings' : [list(range(d))],
    'train_size' : 0.75
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
    'lambda_search_range' : np.linspace(0,10,101),
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


# 2) depth match:
mod2 = DecisionSetMod(
    decision_set_model = DecisionForest,
    decision_set_params = forest_params_depth_imm,
    clustering = kmeans_base,
    prune_params = prune_params,
    min_frac_cover = min_frac_cover,
    name = 'Forest-Depth-IMM'
)


####################################################################################################

# Oblique rule sets:

# 3) SVM decision set
mod3 = DecisionSetMod(
    decision_set_model = SVMSet,
    decision_set_params = svm_params,
    clustering = kmeans_base,
    prune_params = prune_params,
    min_frac_cover = min_frac_cover,
    name = 'SVM'
)


####################################################################################################

# IMM with outliers removed:
mod4 = IMMMod(
    n_clusters=k,
    kmeans_model=kmeans_base.clustering,
    min_frac_cover=min_frac_cover,
    name = "IMM-outliers"
)


####################################################################################################

# List of Modules and Measurements:

baseline_list = [kmeans_base, imm_base]
module_list = [mod1, mod2, mod3, mod4]

measurement_fns = [
    ClusteringCost(average = True, normalize = True),
    Overlap(),
    Coverage(),
    DistanceRatio(),
    Silhouette(),
]


####################################################################################################
# Running the Experiment:

n_samples = 10

Ex1 = CoverageComparisonExperiment(
    data = data,
    baseline_list = baseline_list,
    module_list = module_list,
    measurement_fns = measurement_fns,
    n_samples = n_samples,
    labels = y,
    cpu_count = experiment_cpu_count,
    verbose = False
)

import time 
start = time.time()
Ex1_results = Ex1.run(n_steps = 11, step_size = 0.05)
Ex1.save_results('data/experiments/mnist/', '')
end = time.time()
print(end - start)

####################################################################################################
