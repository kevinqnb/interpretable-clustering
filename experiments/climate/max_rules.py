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

experiment_cpu_count = 6

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 342
np.random.seed(seed)


####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_climate('data/climate')
#true_assignment = labels_to_assignment(data_labels, n_labels = 6)
n,d = data.shape

# Parameters:
n_clusters = 6
n_core = 5
epsilon = 1.9
density_distances = density_distance(data, n_core = n_core)
euclidean_distances = pairwise_distances(data)

lambda_val = 10

max_rules = 20

####################################################################################################

# Experiment 1: KMeans reference clustering:

kmeans_base = KMeansBase(n_clusters = n_clusters)
kmeans_assignment = kmeans_base.assign(data)

# Decision Tree
decision_tree_params = {}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    fitting_params = decision_tree_params,
    min_rules = n_clusters,
    name = 'Decision-Tree'
)

# Removal Tree
rem_tree_params = {'num_clusters' : n_clusters}
rem_tree_mod = DecisionTreeMod(
    model = RemovalTree,
    fitting_params = rem_tree_params,
    min_rules = n_clusters,
    name = 'Removal-Tree'
)

# ExKMC
exkmc_params = {'k' : n_clusters, 'kmeans': kmeans_base.clustering}
exkmc_mod = DecisionTreeMod(
    model = ExkmcTree,
    fitting_params = exkmc_params,
    min_rules = n_clusters,
    name = 'ExKMC'
)

# Decision Set Clustering
dsclust_params = {'lambd' : lambda_val, 'n_features' : 2, 'rules_per_point' : 10}
dsclust_mod = DecisionSetMod(
    model = DSCluster,
    fitting_params = dsclust_params,
    min_rules = n_clusters,
    name = 'Decision-Set-Clustering'
)

baseline = kmeans_base
module_list = [decision_tree_mod, rem_tree_mod, exkmc_mod, dsclust_mod]

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
    baseline = baseline,
    module_list = module_list,
    measurement_fns = measurement_fns,
    n_samples = 100,
    cpu_count = experiment_cpu_count
)

import time 
start = time.time()
exp1_results = exp1.run(n_steps = max_rules - n_clusters)
exp1.save_results('data/experiments/climate/max_rules/', '_kmeans')
end = time.time()
print("Experiment 1 time:", end - start)

####################################################################################################

# Experiment 2: DBSCAN reference clustering:

# Baseline DBSCAN
dbscan_base = DBSCANBase(eps=epsilon, n_core=n_core)
dbscan_assignment = dbscan_base.assign(data)
n_clusters = len(unique_labels(dbscan_base.labels)) # NOTE: Really need to think more about this...

# Decision Treef
decision_tree_params = {}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    fitting_params = decision_tree_params,
    min_rules = n_clusters,
    name = 'Decision-Tree'
)

# Removal Tree
rem_tree_params = {'num_clusters' : n_clusters}
rem_tree_mod = DecisionTreeMod(
    model = RemovalTree,
    fitting_params = rem_tree_params,
    min_rules = n_clusters,
    name = 'Removal-Tree'
)

# Decision Set Clustering
dsclust_params = {'lambd' : lambda_val, 'n_features' : 2, 'rules_per_point' : 10}
dsclust_mod = DecisionSetMod(
    model = DSCluster,
    fitting_params = dsclust_params,
    min_rules = n_clusters,
    name = 'Decision-Set-Clustering'
)

baseline = dbscan_base
module_list = [decision_tree_mod, rem_tree_mod, dsclust_mod]

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
    baseline = baseline,
    module_list = module_list,
    measurement_fns = measurement_fns,
    n_samples = 100,
    cpu_count = experiment_cpu_count
)

import time 
start = time.time()
exp2_results = exp2.run(n_steps = max_rules - n_clusters)
exp2.save_results('data/experiments/climate/max_rules/', '_dbscan')
end = time.time()
print("Experiment 2 time:", end - start)

####################################################################################################
