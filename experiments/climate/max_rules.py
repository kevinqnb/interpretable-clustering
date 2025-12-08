import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.objectives import *
from intercluster.decision_sets.mining import *
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
data, data_labels, feature_labels, scaler = load_preprocessed_climate('data/climate')
n,d = data.shape

##### Parameters #####

# KMeans
n_clusters = 6
euclidean_distances = pairwise_distances(data)

# General
lambda_val = 1.0
max_rules = n_clusters + 20
kmeans_n_rules_list = list(np.arange(n_clusters, max_rules + 1))

# Shallow Tree
depth_factor = 0.03

# Cluster mining:
per_cluster_cost = 0.1

# Association Rule Mining:
min_support = 0.05
min_confidence = 0.9
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

# Baseline KMeans
kmeans_base = KMeansBase(n_clusters = n_clusters, random_seed = seed)
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels


# Decision Tree
decision_tree_params = {(i,) : {'max_leaf_nodes' : i, 'random_state' : seed}
                        for i in kmeans_n_rules_list}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)


# Explanation Tree
exp_tree_params = {tuple(kmeans_n_rules_list) : {'num_clusters' : n_clusters}}
exp_tree_mod = DecisionTreeMod(
    model = ExplanationTree,
    name = 'Exp-Tree'
)


# ExKMC
exkmc_params = {
    (i,) : {
        'k' : n_clusters,
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
        'n_clusters' : n_clusters,
        'depth_factor' : depth_factor,
        'kmeans_random_state' : seed
    } for i in kmeans_n_rules_list
}
shallow_tree_mod = DecisionTreeMod(
    model = ShallowTree,
    name = 'Shallow-Tree'
)

'''
# Pre-generated association rules
association_rule_miner = ClassAssociationMiner(
    min_support = min_support,
    min_confidence = min_confidence,
    max_length = max_length,
    random_state = seed
)
association_rules, association_rule_labels = association_rule_miner.fit(data, kmeans_labels)
association_n_mine = len(association_rule_miner.decision_set)


# CBA
cba_params = {
    tuple(kmeans_n_rules_list) : {}
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
        tuple(kmeans_n_rules_list) : {
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
'''

rule_miner = ClusterMiner(
    cluster_cost=per_cluster_cost,
    method = "kmeans"
)
rules, rule_labels = rule_miner.fit(
    X = data, y = kmeans_base.labels
)

# Decision Set Clustering: Coverage Mistake Objective
dsclust_params_cov_mistake = {
    (i,) : {
        'objective' : CoverageMistakeObjective(
            n_rules = i,
            lambda_val = lambda_val
        )
    }
    for i in kmeans_n_rules_list
}
dsclust_mod_cov_mistake = DecisionSetMod(
    model = DSCluster,
    rules = rules,
    rule_labels = rule_labels,
    rule_miner = rule_miner,
    name = 'DSCluster'
)

# Decision Set Clustering: Total Coverage Mistake Objective
dsclust_params_total_cov_mistake = {
    (i,) : {
        'objective' : TotalCoverageMistakeObjective(
            n_rules = i,
            lambda_val = lambda_val
        )
    }
    for i in kmeans_n_rules_list
}
dsclust_mod_total_cov_mistake = DecisionSetMod(
    model = DSCluster,
    rules = rules,
    rule_labels = rule_labels,
    rule_miner = rule_miner,
    name = 'DSCluster-TotalCoverageMistake'
)

# Decision Set Clustering: Coverage Cost Objective
dsclust_params_cov_cost = {
    (i,) : {
        'objective' : CoverageCostObjective(
            cluster_centers = kmeans_base.centers,
            n_rules = i,
            lambda_val = lambda_val,
            method = "kmeans"
        )
    }
    for i in kmeans_n_rules_list
}
dsclust_mod_cov_cost = DecisionSetMod(
    model = DSCluster,
    rules = rules,
    rule_labels = rule_labels,
    rule_miner = rule_miner,
    name = 'DSCluster-CoverageCost'
)

# Decision Set Clustering: Coverage Cost Objective
dsclust_params_total_cov_cost = {
    (i,) : {
        'objective' : TotalCoverageCostObjective(
            cluster_centers = kmeans_base.centers,
            n_rules = i,
            lambda_val = lambda_val,
            method = "kmeans"
        )
    }
    for i in kmeans_n_rules_list
}
dsclust_mod_total_cov_cost = DecisionSetMod(
    model = DSCluster,
    rules = rules,
    rule_labels = rule_labels,
    rule_miner = rule_miner,
    name = 'DSCluster-TotalCoverageCost'
)



baseline = kmeans_base
module_list = [
    (decision_tree_mod, decision_tree_params),
    (exp_tree_mod, exp_tree_params),
    (exkmc_mod, exkmc_params),
    (shallow_tree_mod, shallow_tree_params),
    #(cba_mod, cba_params),
    (dsclust_mod_cov_mistake, dsclust_params_cov_mistake),
    (dsclust_mod_total_cov_mistake, dsclust_params_total_cov_mistake),
    (dsclust_mod_cov_cost, dsclust_params_cov_cost),
    (dsclust_mod_total_cov_cost, dsclust_params_total_cov_cost),
]


objective1 = CoverageMistakeObjective(
    n_rules = max_rules,
    lambda_val = lambda_val
)

objective2 = TotalCoverageMistakeObjective(
    n_rules = max_rules,
    lambda_val = lambda_val
)

objective3 = CoverageCostObjective(
    cluster_centers = kmeans_base.centers,
    n_rules = max_rules,
    lambda_val = lambda_val,
    method = "kmeans"
)
objective3.set_data(data)

objective4 = TotalCoverageCostObjective(
    cluster_centers = kmeans_base.centers,
    n_rules = max_rules,
    lambda_val = lambda_val,
    method = "kmeans"
)
objective4.set_data(data)

measurement_fns = [
    ObjectiveValue(
        objective = objective1,
        baseline_assignment = kmeans_assignment,
        name = 'objective-value'
    ),
    ObjectiveValue(
        objective = objective2,
        baseline_assignment = kmeans_assignment,
        name = 'total-coverage-mistake-objective-value'
    ),
    ObjectiveValue(
        objective = objective3,
        baseline_assignment = kmeans_assignment,
        name = 'coverage-cost-objective-value'
    ),
    ObjectiveValue(
        objective = objective4,
        baseline_assignment = kmeans_assignment,
        name = 'total-coverage-cost-objective-value'
    ),
    ObjectiveGain(
        objective = objective1,
        baseline_assignment = kmeans_assignment,
        name = 'per-cluster-coverage'
    ),
    ObjectiveCost(
        objective = objective1,
        baseline_assignment = kmeans_assignment,
        name = 'per-rule-mistakes'
    ),
    ObjectiveGain(
        objective = objective4,
        baseline_assignment = kmeans_assignment,
        name = 'total-coverage'
    ),
    ObjectiveCost(
        objective = objective4,
        baseline_assignment = kmeans_assignment,
        name = 'per-rule-clustering-cost'
    ),
    ClusteringCost(
        data = data,
        method = 'kmeans',
        average = True,
        normalize = True,
        name = 'normalized-clustering-cost'
    )
]

exp = MaxRulesExperiment(
    data = data,
    n_rules_list = kmeans_n_rules_list,
    baseline = baseline,
    module_list = module_list,
    measurement_fns = measurement_fns,
    cpu_count = experiment_cpu_count,
    verbose = True
)

exp_results = exp.run()
exp.save_results('data/experiments/climate/max_rules/', '_coverage_mistake')

####################################################################################################

