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

experiment_cpu_count = 12

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 342

####################################################################################################
# Read and process data:
data, labels, feature_labels, scaler = load_preprocessed_ansio()
n,d = data.shape

##### Parameters #####

# Agglomerative Clustering
n_clusters = 12
euclidean_distances = pairwise_distances(data)

# General
lambda_val = 5.0
max_rules = n_clusters + 20

# Shallow Tree
depth_factor = 0.03

# Association Rule Mining:
min_support = 0.001
min_confidence = 0.5
max_length = 4

# IDS:
ids_samples = 10

# Pointwise Rule Mining:
pointwise_generation_samples = 10
samples_per_point = 5
prob_dim = 1/2
prob_stop = 3/4

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


# Pre-generated association rules
association_rule_miner = ClassAssociationMiner(
    min_support = min_support,
    min_confidence = min_confidence,
    max_length = max_length,
    random_state = seed
)
association_rules, association_rule_labels = association_rule_miner.fit(data, agglo_labels)
association_n_mine = len(association_rule_miner.decision_set)


# CBA
cba_params = {
    tuple(agglo_n_rules_list) : {}
}
cba_mod = DecisionSetMod(
    model = CBA,
    rules = association_rules,
    rule_labels = association_rule_labels,
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

ids_module_list = []
for s in range(ids_samples):
    ids_params = {
        tuple(agglo_n_rules_list) : {
            'lambdas' : ids_lambdas
        }
    }
    ids_mod = DecisionSetMod(
        model = IDS,
        rules = association_rules,
        rule_labels = association_rule_labels,
        rule_miner = association_rule_miner,
        name = f"IDS_{s}"
    )
    ids_module_list.append((ids_mod, ids_params))


# Decision Set Clustering
dsclust_params_assoc = {
    (i,) : {
        'lambd' : lambda_val,
        'n_rules' : i
    }
    for i in agglo_n_rules_list
}
dsclust_mod_assoc = DecisionSetMod(
    model = DSCluster,
    rules = association_rules,
    rule_labels = association_rule_labels,
    rule_miner = association_rule_miner,
    name = 'DSCluster-Assoc'
)


# Pre-generated pointwise rules
pointwise_module_list = []
for s in range(pointwise_generation_samples):
    pointwise_rule_miner = PointwiseMinerV2(
        samples = samples_per_point,
        prob_dim = prob_dim,
        prob_stop = prob_stop
    )
    pointwise_rules, pointwise_rule_labels = pointwise_rule_miner.fit(data, agglo_labels)

    # Decision Set Clustering : Pointwise Rules
    dsclust_params = {
        (i,) : {
            'lambd' : lambda_val,
            'n_rules' : i
        }
        for i in agglo_n_rules_list
    }
    dsclust_mod = DecisionSetMod(
        model = DSCluster,
        rules = pointwise_rules,
        rule_labels = pointwise_rule_labels,
        rule_miner = pointwise_rule_miner,
        name = f"DSCluster_{s}"
    )

    pointwise_module_list.append((dsclust_mod, dsclust_params))



baseline = agglomerative_base
module_list = [
    (decision_tree_mod, decision_tree_params),
    (exp_tree_mod, exp_tree_params),
    (cba_mod, cba_params),
    (dsclust_mod_assoc, dsclust_params_assoc)
] + ids_module_list + pointwise_module_list
#n_samples = [1,1,1,10,1] + pointwise_sample_list

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
    cpu_count = experiment_cpu_count,
    verbose = True
)

exp_results = exp.run()
exp.save_results('data/experiments/aniso/max_rules/', '_agglo')

####################################################################################################

