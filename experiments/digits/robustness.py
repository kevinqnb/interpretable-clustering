import os
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
data, data_labels, feature_labels, scaler = load_preprocessed_digits()
data_labels = labels_format(data_labels)
data = data
data_labels = data_labels
n,d = data.shape
n_unique_labels = len(unique_labels(data_labels))

# Parameters:
lambda_val = 5.0
n_rules = 10
n_samples = 10000
std_dev = 0.1

# KMeans:
kmeans_n_clusters = n_unique_labels

# DBSCAN
n_core = 20
epsilon = 1.5

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
decision_tree_params = {'max_leaf_nodes' : n_rules}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)

# Removal Tree
rem_tree_params = {'num_clusters' : kmeans_n_clusters}
rem_tree_mod = DecisionTreeMod(
    model = RemovalTree,
    name = 'Exp-Tree'
)

# ExKMC
exkmc_params = {
    'k' : kmeans_n_clusters,
    'kmeans': kmeans_base.clustering,
    'max_leaf_nodes': n_rules
}
exkmc_mod = DecisionTreeMod(
    model = ExkmcTree,
    name = 'ExKMC'
)

# Shallow Tree
shallow_tree_params = {
    'n_clusters' : kmeans_n_clusters,
    'depth_factor' : depth_factor,
    'kmeans_random_state' : seed
}
shallow_tree_mod = DecisionTreeMod(
    model = ShallowTree,
    name = 'Shallow-Tree'
)

# CBA
cba_params = {'rule_miner' : association_rule_miner}
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
    #(ids_mod, ids_params),
    (dsclust_mod1, dsclust_params1),
    (dsclust_mod2, dsclust_params2)
]

exp1 = RobustnessExperiment(
    data = data,
    baseline = baseline,
    module_list = module_list,
    std_dev = std_dev,
    n_samples = n_samples
)

exp1_results = exp1.run()
exp1.save_results('data/experiments/digits/robustness/', '_kmeans')

exp1_no_outliers = RobustnessExperiment(
    data = data,
    baseline = baseline,
    module_list = module_list,
    std_dev = std_dev,
    n_samples = n_samples,
    ignore = {-1}
)

exp1_no_outliers_results = exp1_no_outliers.run()
exp1_no_outliers.save_results('data/experiments/digits/robustness/', '_kmeans_no_outliers')

####################################################################################################

# Experiment 2: DBSCAN reference clustering:
np.random.seed(seed)

# Baseline DBSCAN
dbscan_base = DBSCANBase(eps=epsilon, n_core=n_core)
dbscan_assignment = dbscan_base.assign(data)
dbscan_n_clusters = len(unique_labels(dbscan_base.labels))

if dbscan_n_clusters < 2:
    raise ValueError("DBSCAN found less than 2 clusters. Try changing n_core or epsilon.")

# Decision Tree
decision_tree_params = {'max_leaf_nodes' : n_rules}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)

# Removal Tree
rem_tree_params = {'num_clusters' : dbscan_n_clusters}
rem_tree_mod = DecisionTreeMod(
    model = RemovalTree,
    name = 'Exp-Tree'
)

# CBA
cba_params = {'rule_miner' : association_rule_miner}
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
    'rule_miner' : association_rule_miner,
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

baseline = dbscan_base
module_list = [
    (decision_tree_mod, decision_tree_params),
    (rem_tree_mod, rem_tree_params),
    (cba_mod, cba_params),
    #(ids_mod, ids_params),
    (dsclust_mod1, dsclust_params1),
    (dsclust_mod2, dsclust_params2)
]

exp2 = RobustnessExperiment(
    data = data,
    baseline = baseline,
    module_list = module_list,
    std_dev = std_dev,
    n_samples = n_samples
)

exp2_results = exp2.run()
exp2.save_results('data/experiments/digits/robustness/', '_dbscan')


exp2_no_outliers = RobustnessExperiment(
    data = data,
    baseline = baseline,
    module_list = module_list,
    std_dev = std_dev,
    n_samples = n_samples,
    ignore = {-1}
)

exp2_no_outliers_results = exp2_no_outliers.run()
exp2_no_outliers.save_results('data/experiments/digits/robustness/', '_dbscan_no_outliers')

####################################################################################################

