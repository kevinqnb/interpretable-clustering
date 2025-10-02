import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.selection import *
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
data, labels, feature_labels, scaler = load_preprocessed_anuran('data/anuran')
n,d = data.shape

##### Parameters #####

# Agglomerative Clustering
n_clusters = 6
euclidean_distances = pairwise_distances(data)

# General
lambda_val = 5.0
max_rules = n_clusters + 20

# Shallow Tree
depth_factor = 0.03

# Association Rule Mining:
min_support = 0.01
min_confidence = 0.8
max_length = 10

####################################################################################################

np.random.seed(seed)

# Agglomerative reference clustering:
agglomerative_base = AgglomerativeBase(n_clusters=n_clusters, linkage='single')
agglo_assignment = agglomerative_base.assign(data)
agglo_labels = agglomerative_base.labels
agglo_n_rules_list = list(np.arange(n_clusters, max_rules + 1))


# Decision Tree
decision_tree_params = {(i,) : {'max_leaf_nodes' : i, 'random_state' : seed}
                        for i in agglo_n_rules_list}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)


# Explanation Tree
exp_tree_params = {tuple(agglo_n_rules_list) : {'num_clusters' : n_clusters}}
exp_tree_mod = DecisionTreeMod(
    model = ExplanationTree,
    name = 'Exp-Tree'
)


# Rule Generation 
association_rule_miner = ClassAssociationMiner(
    min_support = min_support,
    min_confidence = min_confidence,
    max_length = max_length,
    random_state = seed
)
association_rule_miner.fit(data, agglo_labels)
association_n_mine = len(association_rule_miner.decision_set)

association_rule_miner = ClassAssociationMiner(
    min_support = min_support,
    min_confidence = min_confidence,
    max_length = max_length,
    random_state = seed
)


# CBA
cba_params = {
    tuple(agglo_n_rules_list) : {}
}
cba_mod = DecisionSetMod(
    model = CBA,
    rule_miner = association_rule_miner,
    name = 'CBA'
)


# IDS
ids_lambdas = [
    1/association_n_mine,
    1/(2 * data.shape[1] * association_n_mine),
    1/(len(data) * (association_n_mine**2)),
    1/(len(data) * (association_n_mine**2)),
    1/n_clusters,
    1/(data.shape[0] * association_n_mine),
    1/(data.shape[0])
]

ids_params = {
    tuple(agglo_n_rules_list) : {
        'lambdas' : ids_lambdas,
    }
}
ids_mod = DecisionSetMod(
    model = IDS,
    rule_miner = association_rule_miner,
    name = 'IDS'
)


# Decision Set Clustering
dsclust_params = {
    (i,) : {
        'lambd' : lambda_val,
        'n_rules' : i,
    }
    for i in agglo_n_rules_list
}
dsclust_mod = DecisionSetMod(
    model = DSCluster,
    rule_miner = association_rule_miner,
    name = 'DSCluster'
)



baseline = agglomerative_base
module_list = [
    (decision_tree_mod, decision_tree_params),
    (exp_tree_mod, exp_tree_params),
    (cba_mod, cba_params),
    #(ids_mod, ids_params),
    (dsclust_mod, dsclust_params)
]
n_samples = [1,1,1,1]

coverage_mistake_measure = CoverageMistakeScore(
    lambda_val = lambda_val,
    ground_truth_assignment = agglo_assignment,
    name = 'coverage-mistake-score'
)

silhouette_measure = Silhouette(
    distances = euclidean_distances,
    name = 'silhouette-score'
)

clustering_dist = ClusteringDistance(
    ground_truth_assignment = agglo_assignment,
    name = 'clustering-distance'
)


measurement_fns = [
    coverage_mistake_measure,
    silhouette_measure,
    clustering_dist
]

exp = MaxRulesExperiment(
    data = data,
    n_rules_list = agglo_n_rules_list,
    baseline = baseline,
    module_list = module_list,
    measurement_fns = measurement_fns,
    n_samples = n_samples,
    cpu_count = experiment_cpu_count,
    verbose = True
)

exp_results = exp.run()
exp.save_results('data/experiments/anuran/max_rules/', '_agglo')

####################################################################################################

