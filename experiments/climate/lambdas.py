import os
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

experiment_cpu_count = 1

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
n_rules = n_clusters

# Shallow Tree
depth_factor = 0.03

# Association Rule Mining:
frequency_threshold = 0.1

# One-d cluster mining:
per_cluster_cost = 0.1


####################################################################################################

np.random.seed(seed)

# Baseline clustering
kmeans_base = KMeansBase(n_clusters = n_clusters, random_seed = seed)
kmeans_assignment = kmeans_base.assign(data)

# Objective function:
objective = CoverageMistakeObjective(
    n_rules = n_rules
)

objective2 = TotalCoverageCostObjective(
    cluster_centers = kmeans_base.centers,
    n_rules = n_rules,
    method = "kmeans"
)
objective2.set_data(data)

rule_miner = ClusterMiner(
    cluster_cost=per_cluster_cost,
    method = "kmeans"
)

rules, rule_labels = rule_miner.fit(
    X = data, y = kmeans_base.labels
)

# Find minimum lambda value:
dsclust = DSCluster(
    objective = objective,
    rule_miner = rule_miner,
    rules = rules,
    rule_labels = rule_labels,
)

min_lambda_val = dsclust.compute_lambda(data, kmeans_base.labels) + 1e-10

# Lambda array:
lambda_array = np.linspace(min_lambda_val, min_lambda_val * 2, 100)

# Decision Set Clustering Modules:
dsclust_params = {
    (i,) : {
        'objective' : CoverageMistakeObjective(
            n_rules = n_rules,
            lambda_val = lambda_array[i]
        ),
        'rule_miner' : rule_miner,
    }
    for i in range(len(lambda_array))
}
dsclust_mod = DecisionSetMod(
    model = DSCluster,
    rule_miner = rule_miner,
    name = 'DSCluster'
)

module_list = [
    (dsclust_mod, dsclust_params)
]


measurement_fns = [
    ObjectiveGain(
        objective = objective,
        name = 'per-cluster-coverage'
    ),
    ObjectiveCost(
        objective = objective,
        name = 'mistakes-cost'
    ),
    ObjectiveGain(
        objective = objective2,
        name = 'total-coverage'
    ),
    ObjectiveCost(
        objective = objective2,
        name = 'clustering-cost'
    )
]

exp = LambdaExperiment(
    data = data,
    ground_truth_assignment = kmeans_assignment,
    lambda_array = lambda_array,
    baseline = kmeans_base,
    module_list = module_list,
    measurement_fns= measurement_fns,
    cpu_count = experiment_cpu_count,
    verbose = True
)

import time 
start = time.time()
exp1_results = exp.run()
exp.save_results('data/experiments/climate/lambdas/', '_kmeans')
end = time.time()
print("Experiment 1 time:", end - start)

####################################################################################################