import os
import pandas as pd
import numpy as np
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
data = pd.read_csv('../data/synthetic/D31.csv', index_col = 0).to_numpy()[:,0:2]
n,d = data.shape

# Parameters:
lambda_val = 2.0
n_rules = 31
n_samples = 10000
std_dev = 0.1

# KMeans
n_clusters = 31

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
kmeans_base = KMeansBase(n_clusters = n_clusters, random_seed = seed)
kmeans_assignment = kmeans_base.assign(data)


# Decision Tree
decision_tree_params = {'max_leaf_nodes' : n_rules}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)


# Removal Tree
rem_tree_params = {'num_clusters' : n_clusters}
rem_tree_mod = DecisionTreeMod(
    model = RemovalTree,
    name = 'Exp-Tree'
)


# ExKMC
exkmc_params = {
    'k' : n_clusters,
    'kmeans': kmeans_base.clustering,
    'max_leaf_nodes': n_rules
}
exkmc_mod = DecisionTreeMod(
    model = ExkmcTree,
    name = 'ExKMC'
)


# Shallow Tree
shallow_tree_params = {
    'n_clusters' : n_clusters,
    'depth_factor' : depth_factor,
    'kmeans_random_state' : seed
}
shallow_tree_mod = DecisionTreeMod(
    model = ShallowTree,
    name = 'Shallow-Tree'
)


# CBA
cba_params = {
    'rule_miner' : association_rule_miner,
}
cba_mod = DecisionSetMod(
    model = CBA,
    rule_miner = association_rule_miner,
    name = 'CBA'
)


# IDS
ids_params = {
    'lambdas' : ids_lambdas,
    'rule_miner' : association_rule_miner,
}
ids_mod = DecisionSetMod(
    model = IDS,
    rule_miner = association_rule_miner,
    name = 'IDS'
)


# Decision Set Clustering (1) -- Entropy Association Rules (same as IDS)
dsclust_params1 = {
    'lambd' : lambda_val,
    'n_rules' : n_rules,
    'rule_miner' : association_rule_miner
}
dsclust_mod1 = DecisionSetMod(
    model = DSCluster,
    rule_miner = association_rule_miner,
    name = 'DSCluster-Association-Rules'
)


# Decision Set Clustering (2) -- Pointwise Rules
dsclust_params2 = {
    'lambd' : lambda_val,
    'n_rules' : n_rules,
    'rule_miner' : pointwise_rule_miner,
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
    (ids_mod, ids_params),
    (dsclust_mod1, dsclust_params1),
    (dsclust_mod2, dsclust_params2)
]

exp = RobustnessExperiment(
    data = data,
    baseline = baseline,
    module_list = module_list,
    std_dev = std_dev,
    n_samples = n_samples
)

exp_results = exp.run()
exp.save_results('data/experiments/blobs/robustness/', '_kmeans')

exp_no_outliers = RobustnessExperiment(
    data = data,
    baseline = baseline,
    module_list = module_list,
    std_dev = std_dev,
    n_samples = n_samples,
    ignore = {-1}
)

exp_no_outliers_results = exp_no_outliers.run()
exp_no_outliers.save_results('data/experiments/blobs/robustness/', '_kmeans_no_outliers')


####################################################################################################