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

experiment_cpu_count = 1

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 342

####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_digits()
data_labels = labels_format(data_labels)
data = data
data_labels = data_labels
n,d = data.shape
n_unique_labels = len(unique_labels(data_labels))

# Parameters:
lambda_val = 5.0
max_rules = 20
n_samples = 1

# KMeans:
kmeans_n_clusters = n_unique_labels
kmeans_n_rules_list = list(np.arange(kmeans_n_clusters, max_rules + 1))

# DBSCAN
n_core = 20
# Use only if there is some ground truth labeling of the data:
true_assignment = labels_to_assignment(data_labels, n_unique_labels)
density_distances = density_distance(data, n_core = n_core)
euclidean_distances = pairwise_distances(data)
#epsilon = min_inter_cluster_distance(density_distances, true_assignment) - 0.01
epsilon = 1.5

# IDS
ids_n_mine = 10000
ids_lambdas = [
    1/ids_n_mine,
    1/(2 * data.shape[1] * ids_n_mine),
    1/(len(data) * (ids_n_mine**2)),
    1/(len(data) * (ids_n_mine**2)),
    0,
    1/(data.shape[0] * ids_n_mine),
    1/(data.shape[0])
]
'''
ids_lambda_search_dict = {
    'l1': (0, 10000),
    'l2': (0, 10000),
    'l3': (0, 10000),
    'l4': (0, 10000),
    'l5': (0, 10000),
    'l6': (0, 10000),
    'l7': (0, 10000)
}
ids_ternary_search_precision = 10.0
ids_max_iterations = 10
'''

# Decision Set Clustering
dsclust_n_features = 3
dsclust_rules_per_point = 50

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
    name = 'Removal-Tree'
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
        'depth_factor' : 1.0,
        'kmeans_random_state' : seed
    } for i in kmeans_n_rules_list
}
shallow_tree_mod = DecisionTreeMod(
    model = ShallowTree,
    name = 'Shallow-Tree'
)

# IDS
'''
ids_params = {
    tuple(kmeans_n_rules_list) : {
        'bins' : ids_bins,
        'n_mine' : ids_n_mine,
        'lambda_search_dict' : ids_lambda_search_dict,
        'ternary_search_precision' : ids_ternary_search_precision,
        'max_iterations' : ids_max_iterations,
        'quantiles' : ids_quantiles
    }
}
'''
association_rule_miner_ids = AssociationRuleMiner(max_rules = ids_n_mine, bin_type = 'mdlp')
ids_params = {
    tuple(kmeans_n_rules_list) : {
        'lambdas' : ids_lambdas,
        'rule_miner' : association_rule_miner_ids,
    }
}

ids_mod = DecisionSetMod(
    model = IdsSet,
    rule_miner = association_rule_miner_ids,
    name = 'IDS'
)


# Decision Set Clustering (1) -- Entropy Association Rules (same as IDS)
association_rule_miner_dscluster = AssociationRuleMiner(max_rules = ids_n_mine, bin_type = 'mdlp')
dsclust_params1 = {
    (i,) : {
        'lambd' : lambda_val,
        'n_rules' : i,
        'rule_miner' : association_rule_miner_dscluster,
    }
    for i in kmeans_n_rules_list
}
dsclust_mod1 = DecisionSetMod(
    model = DSCluster,
    rule_miner = association_rule_miner_dscluster,
    name = 'DSCluster-Association-Rules'
)

# Decision Set Clustering (2) -- Pointwise Rules
pointwise_rule_miner = PointwiseMiner(
    lambd = lambda_val,
    n_features = dsclust_n_features,
    rules_per_point = dsclust_rules_per_point
)
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
    #(decision_tree_mod, decision_tree_params),
    #(rem_tree_mod, rem_tree_params),
    #(exkmc_mod, exkmc_params),
    #(shallow_tree_mod, shallow_tree_params),
    (ids_mod, ids_params),
    #(dsclust_mod1, dsclust_params1),
    #(dsclust_mod2, dsclust_params2)
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
    n_rules_list = kmeans_n_rules_list,
    baseline = baseline,
    module_list = module_list,
    measurement_fns = measurement_fns,
    n_samples = n_samples,
    cpu_count = experiment_cpu_count
)

import time 
start = time.time()
exp1_results = exp1.run()
#exp1.save_results('data/experiments/digits/max_rules/', '_kmeans')
end = time.time()
print("Experiment 1 time:", end - start)

####################################################################################################

'''
# Experiment 2: DBSCAN reference clustering:
np.random.seed(seed)

# Baseline DBSCAN
dbscan_base = DBSCANBase(eps=epsilon, n_core=n_core)
dbscan_assignment = dbscan_base.assign(data)
dbscan_n_clusters = len(unique_labels(dbscan_base.labels))
dbscan_n_rules_list = list(np.arange(dbscan_n_clusters, max_rules + 1))

if dbscan_n_clusters < 2:
    raise ValueError("DBSCAN found less than 2 clusters. Try changing n_core or epsilon.")

# Decision Tree
decision_tree_params = {(i,) : {'max_leaf_nodes' : i} for i in dbscan_n_rules_list}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)

# Removal Tree
rem_tree_params = {tuple(dbscan_n_rules_list) : {'num_clusters' : dbscan_n_clusters}}
rem_tree_mod = DecisionTreeMod(
    model = RemovalTree,
    name = 'Removal-Tree'
)


# IDS
#ids_params = {
#    tuple(kmeans_n_rules_list) : {
#        'bins' : ids_bins,
#        'n_mine' : ids_n_mine,
#        'lambda_search_dict' : ids_lambda_search_dict,
#        'ternary_search_precision' : ids_ternary_search_precision,
#        'max_iterations' : ids_max_iterations,
#        'quantiles' : ids_quantiles
#    }
#}

association_rule_miner_ids = AssociationRuleMiner(max_rules = ids_n_mine, bin_type = 'mdlp')
ids_params = {
    tuple(dbscan_n_rules_list) : {
        'lambdas' : ids_lambdas,
        'rule_miner' : association_rule_miner_ids,
    }
}
ids_mod = DecisionSetMod(
    model = IdsSet,
    rule_miner = association_rule_miner_ids,
    name = 'IDS'
)


# Decision Set Clustering (1) -- Entropy Association Rules (same as IDS)
association_rule_miner_dscluster = AssociationRuleMiner(max_rules = ids_n_mine, bin_type = 'mdlp')
dsclust_params1 = {
    (i,) : {
        'lambd' : lambda_val,
        'n_rules' : i,
        'rule_miner' : association_rule_miner_dscluster,
    }
    for i in dbscan_n_rules_list
}
dsclust_mod1 = DecisionSetMod(
    model = DSCluster,
    rule_miner = association_rule_miner_dscluster,
    name = 'DSCluster-Association-Rules'
)

# Decision Set Clustering (2) -- Pointwise Rules
pointwise_rule_miner = PointwiseMiner(
    lambd = lambda_val,
    n_features = dsclust_n_features,
    rules_per_point = dsclust_rules_per_point
)
dsclust_params2 = {
    (i,) : {
        'lambd' : lambda_val,
        'n_rules' : i,
        'rule_miner' : pointwise_rule_miner,
    }
    for i in dbscan_n_rules_list
}
dsclust_mod2 = DecisionSetMod(
    model = DSCluster,
    rule_miner = pointwise_rule_miner,
    name = 'DSCluster-Pointwise-Rules'
)

baseline = dbscan_base
module_list = [
    (decision_tree_mod, decision_tree_params),
    (rem_tree_mod, rem_tree_params),
    #(ids_mod, ids_params),
    (dsclust_mod1, dsclust_params1),
    (dsclust_mod2, dsclust_params2)
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
    n_rules_list = dbscan_n_rules_list,
    baseline = baseline,
    module_list = module_list,
    measurement_fns = measurement_fns,
    n_samples = n_samples,
    cpu_count = experiment_cpu_count
)

import time 
start = time.time()
exp2_results = exp2.run()
exp2.save_results('data/experiments/digits/max_rules/', '_dbscan')
end = time.time()
print("Experiment 2 time:", end - start)

####################################################################################################
'''
