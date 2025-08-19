import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.pruning import *
from intercluster.experiments import *

# Prevents memory leakage for KMeans:
os.environ["OMP_NUM_THREADS"] = "1"

experiment_cpu_count = 1

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 342
np.random.seed(seed)


####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_digits()
data_labels = labels_format(data_labels)
data = data[:50]
data_labels = data_labels[:50]
n,d = data.shape

# Parameters:
n_clusters = 10

n_core = 5
true_assignment = labels_to_assignment(data_labels, n_clusters)
density_distances = density_distance(data, n_core = n_core)
euclidean_distances = pairwise_distances(data)
epsilon = min_inter_cluster_distance(density_distances, true_assignment) - 0.01

lambda_val = 10

max_rules = 20
n_rules_list = np.arange(n_clusters, max_rules + 1)

####################################################################################################

# Experiment 1: KMeans reference clustering:
n_clusters = 10
n_rules_list = np.arange(n_clusters, max_rules + 1)
kmeans_base = KMeansBase(n_clusters = n_clusters)
kmeans_assignment = kmeans_base.assign(data)

# Decision Tree
decision_tree_params = [{'max_leaf_nodes' : i} for i in n_rules_list]
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)

# Removal Tree
rem_tree_params = [{'num_clusters' : n_clusters}]
rem_tree_mod = DecisionTreeMod(
    model = RemovalTree,
    name = 'Removal-Tree'
)

# ExKMC
exkmc_params = [
    {'k' : n_clusters, 'kmeans': kmeans_base.clustering, 'max_leaf_nodes': i} for i in n_rules_list
]
exkmc_mod = DecisionTreeMod(
    model = ExkmcTree,
    name = 'ExKMC'
)

# IDS
lambda_search_dict = {
    'l1': (0, 1e5),
    'l2': (0, 1e5),
    'l3': (0, 1e5),
    'l4': (0, 1e5),
    'l5': (0, 1e5),
    'l6': (0, 1e5),
    'l7': (0, 1e5)
}
ids_params = [
    {'bins' : 5, 'n_mine' : 15, 'lambda_search_dict' : lambda_search_dict,
     'ternary_search_precision' : 10.0, 'max_iterations' : 10, 'quantiles' : True}
]

ids_mod = DecisionSetMod(
    model = IdsSet,
    name = 'IDS'
)


# Decision Set Clustering
dsclust_params = [
    {'lambd' : lambda_val, 'n_features' : 2, 'rules_per_point' : 10, 'n_rules' : i}
    for i in n_rules_list
]
dsclust_mod = DecisionSetMod(
    model = DSCluster,
    name = 'Decision-Set-Clustering'
)

baseline = kmeans_base
module_list = [
    (decision_tree_mod, decision_tree_params),
    (rem_tree_mod, rem_tree_params),
    (exkmc_mod, exkmc_params),
    #(ids_mod, ids_params),
    (dsclust_mod, dsclust_params)
]

coverage_mistake_measure = CoverageMistakeScore(
    lambda_val = lambda_val,
    ground_truth_assignment = kmeans_assignment,
    name = 'Coverage-Mistake-Score'
)

silhouette_measure = Silhouette(
    distances = euclidean_distances,
    name = 'Silhouette-Score'
)

measurement_fns = [
    coverage_mistake_measure,
    silhouette_measure,
]

exp1 = MaxRulesExperiment(
    data = data,
    n_rules_list = n_rules_list,
    baseline = baseline,
    module_list = module_list,
    measurement_fns = measurement_fns,
    n_samples = 10,
    cpu_count = experiment_cpu_count
)

import time 
start = time.time()
exp1_results = exp1.run()
exp1.save_results('data/experiments/digits/max_rules/', '_kmeans')
end = time.time()
print("Experiment 1 time:", end - start)

####################################################################################################

# Experiment 2: DBSCAN reference clustering:

# Baseline DBSCAN
dbscan_base = DBSCANBase(eps=epsilon, n_core=n_core)
dbscan_assignment = dbscan_base.assign(data)
n_clusters = len(unique_labels(dbscan_base.labels)) # NOTE: Really need to think more about this...
n_rules_list = np.arange(n_clusters, max_rules + 1)

# Decision Tree
decision_tree_params = [{'max_leaf_nodes' : i} for i in n_rules_list]
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)

# Removal Tree
rem_tree_params = [{'num_clusters' : n_clusters}]
rem_tree_mod = DecisionTreeMod(
    model = RemovalTree,
    name = 'Removal-Tree'
)

# IDS
lambda_search_dict = {
    'l1': (0, 1e5),
    'l2': (0, 1e5),
    'l3': (0, 1e5),
    'l4': (0, 1e5),
    'l5': (0, 1e5),
    'l6': (0, 1e5),
    'l7': (0, 1e5)
}
ids_params = [
    {'bins' : 5, 'n_mine' : 15, 'lambda_search_dict' : lambda_search_dict,
     'ternary_search_precision' : 10.0, 'max_iterations' : 10, 'quantiles' : True}
]

ids_mod = DecisionSetMod(
    model = IdsSet,
    name = 'IDS'
)


# Decision Set Clustering
dsclust_params = [
    {'lambd' : lambda_val, 'n_features' : 2, 'rules_per_point' : 10, 'n_rules' : i}
    for i in n_rules_list
]
dsclust_mod = DecisionSetMod(
    model = DSCluster,
    name = 'Decision-Set-Clustering'
)

baseline = dbscan_base
module_list = [
    (decision_tree_mod, decision_tree_params),
    (rem_tree_mod, rem_tree_params),
    #(ids_mod, ids_params),
    (dsclust_mod, dsclust_params)
]

coverage_mistake_measure = CoverageMistakeScore(
    lambda_val = lambda_val,
    ground_truth_assignment = dbscan_assignment,
    name = 'Coverage-Mistake-Score'
)

silhouette_measure = Silhouette(
    distances = density_distances,
    name = 'Silhouette-Score'
)

measurement_fns = [
    coverage_mistake_measure,
    silhouette_measure
]

exp2 = MaxRulesExperiment(
    data = data,
    n_rules_list = n_rules_list,
    baseline = baseline,
    module_list = module_list,
    measurement_fns = measurement_fns,
    n_samples = 10,
    cpu_count = experiment_cpu_count
)

import time 
start = time.time()
exp2_results = exp2.run()
exp2.save_results('data/experiments/digits/max_rules/', '_dbscan')
end = time.time()
print("Experiment 2 time:", end - start)

####################################################################################################
