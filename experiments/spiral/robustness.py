import os
import pandas as pd
import numpy as np
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.selection import *
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
data = pd.read_csv('data/synthetic/spiral_noisy.csv', index_col = 0).to_numpy()[:,0:2]
n,d = data.shape

# Parameters:
lambda_val = 5.0
n_rules = 6
n_samples = 10000
std_dev = np.std(data) / 20

# DBSCAN
epsilon = 0.225
n_core = 10
density_distances = density_distance(data, n_core = n_core)

# Shallow Tree
depth_factor = 0.03

# Association Rule Mining:
min_support = 0.01
min_confidence = 0.5
max_length = 10


####################################################################################################

# Experiment 2: DBSCAN reference clustering:
np.random.seed(seed)

# Baseline DBSCAN
dbscan_base = DBSCANBase(eps=epsilon, n_core=n_core)
dbscan_assignment = dbscan_base.assign(data)
dbscan_labels = dbscan_base.labels
dbscan_n_clusters = len(unique_labels(dbscan_base.labels))

if dbscan_n_clusters < 2:
    raise ValueError("DBSCAN found less than 2 clusters. Try changing n_core or epsilon.")


# Decision Tree
decision_tree_params = {'max_leaf_nodes' : n_rules, 'random_state' : seed}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)

# Removal Tree
exp_tree_params = {'num_clusters' : dbscan_n_clusters}
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
association_rule_miner.fit(data, dbscan_labels)
association_n_mine = len(association_rule_miner.decision_set)

association_rule_miner = ClassAssociationMiner(
    min_support = min_support,
    min_confidence = min_confidence,
    max_length = max_length,
    random_state = seed
)


# CBA
cba_params = {}
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
    1/dbscan_n_clusters,
    1/(data.shape[0] * association_n_mine),
    1/(data.shape[0])
]

ids_params = {
    'lambdas' : ids_lambdas
}
ids_mod = DecisionSetMod(
    model = IDS,
    rule_miner = association_rule_miner,
    name = 'IDS'
)

# Decision Set Clustering (1) -- Entropy Association Rules (same as IDS)
dsclust_params1 = {
    'lambd' : lambda_val,
    'n_rules' : n_rules
}
dsclust_mod1 = DecisionSetMod(
    model = DSCluster,
    rule_miner = association_rule_miner,
    name = 'DSCluster'
)

baseline = dbscan_base
module_list = [
    (decision_tree_mod, decision_tree_params),
    (exp_tree_mod, exp_tree_params),
    (cba_mod, cba_params),
    (ids_mod, ids_params),
    (dsclust_mod1, dsclust_params1)
]


exp = RobustnessExperiment(
    data = data,
    baseline = baseline,
    module_list = module_list,
    std_dev = std_dev,
    n_samples = n_samples
)

exp_results = exp.run()
exp.save_results('data/experiments/spiral/robustness/', '_dbscan')


exp_no_outliers = RobustnessExperiment(
    data = data,
    baseline = baseline,
    module_list = module_list,
    std_dev = std_dev,
    n_samples = n_samples,
    ignore = {-1}
)

exp_no_outliers_results = exp_no_outliers.run()
exp_no_outliers.save_results('data/experiments/spiral/robustness/', '_dbscan_no_outliers')

####################################################################################################

