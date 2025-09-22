import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.pruning import *
from intercluster.mining import *
from intercluster.experiments import *

# Prevents memory leakage for KMeans:
os.environ["OMP_NUM_THREADS"] = "1"

experiment_cpu_count = 4

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 342

####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_climate('data/climate')
n,d = data.shape

# Parameters:
n_clusters = 6
lambda_val = 5.0
max_rules = 20
n_samples = 10

# KMeans:
kmeans_n_clusters = n_clusters
kmeans_n_rules_list = list(np.arange(kmeans_n_clusters, max_rules + 1))
euclidean_distances = pairwise_distances(data)

# Shallow Tree
depth_factor = 0.03

# Association Rule Mining:
min_support = 0.01
min_confidence = 0.5
max_length = 10
association_rule_miner = ClassAssociationMiner(
    min_support = min_support,
    min_confidence = min_confidence,
    max_length = max_length
)

# Pointwise Rule Mining:
pointwise_samples_per_point = 10
pointwise_prob_dim = 1/2
pointwise_prob_stop = 8/10
pointwise_rule_miner = PointwiseMinerV2(
    samples = pointwise_samples_per_point,
    prob_dim = pointwise_prob_dim,
    prob_stop = pointwise_prob_stop
)

# IDS:
ids_lambdas = [1,0,0,0,0,1,1]


####################################################################################################

# Experiment 1: KMeans reference clustering:
np.random.seed(seed)

# Baseline KMeans
kmeans_base = KMeansBase(n_clusters = kmeans_n_clusters, random_seed = seed)
kmeans_assignment = kmeans_base.assign(data)

# Decision Tree
decision_tree_params = {(i,) : {'max_leaf_nodes' : i} for i in kmeans_n_rules_list}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)

# Removal Tree
rem_tree_params = {tuple(kmeans_n_rules_list) : {'num_clusters' : kmeans_n_clusters}}
rem_tree_mod = DecisionTreeMod(
    model = RemovalTree,
    name = 'Exp-Tree'
)

# ExKMC
exkmc_params = {
    (i,) : {
        'k' : kmeans_n_clusters,
        'kmeans': kmeans_base.clustering,
        'max_leaf_nodes': i
    } for i in kmeans_n_rules_list
}
exkmc_mod = DecisionTreeMod(
    model = ExkmcTree,
    name = 'ExKMC'
)

# Shallow Tree
shallow_tree_params = {
    tuple(kmeans_n_rules_list) : {
        'n_clusters' : kmeans_n_clusters,
        'depth_factor' : depth_factor,
        'kmeans_random_state' : seed
    } for i in kmeans_n_rules_list
}
shallow_tree_mod = DecisionTreeMod(
    model = ShallowTree,
    name = 'Shallow-Tree'
)


# CBA
cba_params = {
    tuple(kmeans_n_rules_list) : {
        'rule_miner' : association_rule_miner,
    }
}
cba_mod = DecisionSetMod(
    model = CBA,
    rule_miner = association_rule_miner,
    name = 'CBA'
)

# IDS
ids_params = {
    tuple(kmeans_n_rules_list) : {
        'lambdas' : ids_lambdas,
        'rule_miner' : association_rule_miner,
    }
}

ids_mod = DecisionSetMod(
    model = IDS,
    rule_miner = association_rule_miner,
    name = 'IDS'
)


# Decision Set Clustering (1) -- Entropy Association Rules (same as IDS)
dsclust_params1 = {
    (i,) : {
        'lambd' : lambda_val,
        'n_rules' : i,
        'rule_miner' : association_rule_miner,
    }
    for i in kmeans_n_rules_list
}
dsclust_mod1 = DecisionSetMod(
    model = DSCluster,
    rule_miner = association_rule_miner,
    name = 'DSCluster-Association-Rules'
)

# Decision Set Clustering (2) -- Pointwise Rules
dsclust_params2 = {
    (i,) : {
        'lambd' : lambda_val,
        'n_rules' : i,
        'rule_miner' : pointwise_rule_miner,
    }
    for i in kmeans_n_rules_list
}
dsclust_mod2 = DecisionSetMod(
    model = DSCluster,
    rule_miner = pointwise_rule_miner,
    name = 'DSCluster-Pointwise-Rules'
)

baseline = kmeans_base
module_list = [
    (decision_tree_mod, decision_tree_params),
    (rem_tree_mod, rem_tree_params),
    (exkmc_mod, exkmc_params),
    (shallow_tree_mod, shallow_tree_params),
    (cba_mod, cba_params),
    #(ids_mod, ids_params),
    (dsclust_mod1, dsclust_params1),
    (dsclust_mod2, dsclust_params2)
]

coverage_mistake_measure = CoverageMistakeScore(
    lambda_val = lambda_val,
    ground_truth_assignment = kmeans_assignment,
    name = 'coverage-mistake-score'
)

silhouette_measure = Silhouette(
    distances = euclidean_distances,
    name = 'silhouette-score'
)

measurement_fns = [
    coverage_mistake_measure,
    silhouette_measure,
]

exp = MaxRulesExperiment(
    data = data,
    n_rules_list = kmeans_n_rules_list,
    baseline = baseline,
    module_list = module_list,
    measurement_fns = measurement_fns,
    n_samples = n_samples,
    cpu_count = experiment_cpu_count
)

import time 
start = time.time()
exp1_results = exp.run()
exp.save_results('data/experiments/climate/max_rules/', '_kmeans2')
end = time.time()
print("Experiment 1 time:", end - start)

####################################################################################################
