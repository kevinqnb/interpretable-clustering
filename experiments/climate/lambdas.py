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

experiment_cpu_count = 8

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 342

####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_climate('data/climate')
n,d = data.shape

fixed_parameters = {
    'n' : n,
    'd' : d,
    'n_clusters': 6,
    'n_rules': 6,
    'n_bins': 5,
    'min_support': 0.1,
    'max_rule_length': 10,
    'per_cluster_cost': 0.1,
}


####################################################################################################

np.random.seed(seed)

# Baseline clustering
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = seed)
kmeans_assignment = kmeans_base.assign(data)

# Rule Mining:
uniform_rule_miner = FrequentItemsetMiner(
    min_support = fixed_parameters['min_support'],
    max_length = fixed_parameters['max_rule_length'],
    binning_method = "uniform",
    bin_params = {
        'n_bins': fixed_parameters['n_bins'],
    }
)

uniform_rules, uniform_rule_labels = uniform_rule_miner.fit(
    X = data, y = kmeans_base.labels
)


cluster_rule_miner = FrequentItemsetMiner(
    min_support = fixed_parameters['min_support'],
    max_length = fixed_parameters['max_rule_length'],
    binning_method = "cluster",
    bin_params = {
        'cluster_cost': fixed_parameters['per_cluster_cost'],
        'method': 'kmeans'
    }
)

cluster_rules, cluster_rule_labels = cluster_rule_miner.fit(
    X = data, y = kmeans_base.labels
)


####################################################################################################
# Module 1. Uniform Bin Rules, Coverage Mistake Objective

objective = CoverageMistakeObjective(
    n_rules = fixed_parameters['n_rules']
)

# Find minimum lambda value:
dsclust = DSCluster(
    objective = objective,
    rule_miner = uniform_rule_miner,
    rules = uniform_rules,
    rule_labels = uniform_rule_labels,
)

lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
lambda_array = lambda_array[np.isfinite(lambda_array)]
# Subsample lambda array at even intervals:
indices = np.linspace(0, len(lambda_array) - 1, num = 100, dtype=int)
lambda_array = lambda_array[indices]

# Decision Set Clustering:
dsclust_params1 = {
    (l,) : {
        'objective' : CoverageMistakeObjective(
            n_rules = fixed_parameters['n_rules'],
            lambda_val = l
        )
    }
    for i,l in enumerate(lambda_array)
}

dsclust_mod1 = DecisionSetMod(
    model = DSCluster,
    rule_miner = uniform_rule_miner,
    rules = uniform_rules,
    rule_labels = None,
    name = 'Uniform-Bin; Coverage-Mistake-Obj'
)


####################################################################################################
# Module 2. Cluster Bin Rules, Coverage Mistake Objective

objective = CoverageMistakeObjective(
    n_rules = fixed_parameters['n_rules']
)

# Find minimum lambda value:
dsclust = DSCluster(
    objective = objective,
    rule_miner = cluster_rule_miner,
    rules = cluster_rules,
    rule_labels = cluster_rule_labels,
)

lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
lambda_array = lambda_array[np.isfinite(lambda_array)]
# Subsample lambda array at even intervals:
indices = np.linspace(0, len(lambda_array) - 1, num = 100, dtype=int)
lambda_array = lambda_array[indices]

# Decision Set Clustering:
dsclust_params2 = {
    (l,) : {
        'objective' : CoverageMistakeObjective(
            n_rules = fixed_parameters['n_rules'],
            lambda_val = l
        )
    }
    for i,l in enumerate(lambda_array)
}

dsclust_mod2 = DecisionSetMod(
    model = DSCluster,
    rule_miner = cluster_rule_miner,
    rules = cluster_rules,
    rule_labels = None,
    name = 'Cluster-Bin; Coverage-Mistake-Obj'
)


####################################################################################################
# Module 3. Uniform Bin Rules, Coverage Cost Objective

objective = CoverageCostObjective(
    cluster_centers = kmeans_base.centers,
    n_rules = fixed_parameters['n_rules'],
    method = "kmeans"
)
objective.set_data(data)

# Find minimum lambda value:
dsclust = DSCluster(
    objective = objective,
    rule_miner = uniform_rule_miner,
    rules = uniform_rules,
    rule_labels = uniform_rule_labels,
)

lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
lambda_array = lambda_array[np.isfinite(lambda_array)]
# Subsample lambda array at even intervals:
indices = np.linspace(0, len(lambda_array) - 1, num = 100, dtype=int)
lambda_array = lambda_array[indices]

# Decision Set Clustering:
dsclust_params3 = {
    (l,) : {
        'objective' : CoverageCostObjective(
            cluster_centers = kmeans_base.centers,
            n_rules = fixed_parameters['n_rules'],
            lambda_val = l,
            method = "kmeans"
        )
    }
    for i,l in enumerate(lambda_array)
}

dsclust_mod3 = DecisionSetMod(
    model = DSCluster,
    rule_miner = uniform_rule_miner,
    rules = uniform_rules,
    rule_labels = None,
    name = 'Uniform-Bin; Coverage-Cost-Obj'
)


####################################################################################################
# Module 4. Cluster Bin Rules, Coverage Cost Objective

objective = CoverageCostObjective(
    cluster_centers = kmeans_base.centers,
    n_rules = fixed_parameters['n_rules'],
    method = "kmeans"
)
objective.set_data(data)

# Find minimum lambda value:
dsclust = DSCluster(
    objective = objective,
    rule_miner = cluster_rule_miner,
    rules = cluster_rules,
    rule_labels = cluster_rule_labels,
)

lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
lambda_array = lambda_array[np.isfinite(lambda_array)]
# Subsample lambda array at even intervals:
indices = np.linspace(0, len(lambda_array) - 1, num = 100, dtype=int)
lambda_array = lambda_array[indices]

# Decision Set Clustering:
dsclust_params4 = {
    (l,) : {
        'objective' : CoverageCostObjective(
            cluster_centers = kmeans_base.centers,
            n_rules = fixed_parameters['n_rules'],
            lambda_val = l,
            method = "kmeans"
        )
    }
    for i,l in enumerate(lambda_array)
}

dsclust_mod4 = DecisionSetMod(
    model = DSCluster,
    rule_miner = cluster_rule_miner,
    rules = cluster_rules,
    rule_labels = None,
    name = 'Cluster-Bin; Coverage-Cost-Obj'
)


####################################################################################################
# Module 5. Uniform Bin Rules, Total Coverage Mistake Objective

objective = TotalCoverageMistakeObjective(
    n_rules = fixed_parameters['n_rules']
)

# Find minimum lambda value:
dsclust = DSCluster(
    objective = objective,
    rule_miner = uniform_rule_miner,
    rules = uniform_rules,
    rule_labels = uniform_rule_labels,
)

lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
lambda_array = lambda_array[np.isfinite(lambda_array)]
# Subsample lambda array at even intervals:
indices = np.linspace(0, len(lambda_array) - 1, num = 100, dtype=int)
lambda_array = lambda_array[indices]

# Decision Set Clustering:
dsclust_params5 = {
    (l,) : {
        'objective' : TotalCoverageMistakeObjective(
            n_rules = fixed_parameters['n_rules'],
            lambda_val = l
        )
    }
    for i,l in enumerate(lambda_array)
}

dsclust_mod5 = DecisionSetMod(
    model = DSCluster,
    rule_miner = uniform_rule_miner,
    rules = uniform_rules,
    rule_labels = None,
    name = 'Uniform-Bin; Total-Coverage-Mistake-Obj'
)


####################################################################################################
# Module 6. Cluster Bin Rules, Coverage Mistake Objective

objective = TotalCoverageMistakeObjective(
    n_rules = fixed_parameters['n_rules']
)

# Find minimum lambda value:
dsclust = DSCluster(
    objective = objective,
    rule_miner = cluster_rule_miner,
    rules = cluster_rules,
    rule_labels = cluster_rule_labels,
)

lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
lambda_array = lambda_array[np.isfinite(lambda_array)]
# Subsample lambda array at even intervals:
indices = np.linspace(0, len(lambda_array) - 1, num = 100, dtype=int)
lambda_array = lambda_array[indices]

# Decision Set Clustering:
dsclust_params6 = {
    (l,) : {
        'objective' : TotalCoverageMistakeObjective(
            n_rules = fixed_parameters['n_rules'],
            lambda_val = l
        )
    }
    for i,l in enumerate(lambda_array)
}

dsclust_mod6 = DecisionSetMod(
    model = DSCluster,
    rule_miner = cluster_rule_miner,
    rules = cluster_rules,
    rule_labels = None,
    name = 'Cluster-Bin; Total-Coverage-Mistake-Obj'
)


####################################################################################################
# Module 7. Uniform Bin Rules, Coverage Cost Objective

objective = TotalCoverageCostObjective(
    cluster_centers = kmeans_base.centers,
    n_rules = fixed_parameters['n_rules'],
    method = "kmeans"
)
objective.set_data(data)

# Find minimum lambda value:
dsclust = DSCluster(
    objective = objective,
    rule_miner = uniform_rule_miner,
    rules = uniform_rules,
    rule_labels = uniform_rule_labels,
)

lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
lambda_array = lambda_array[np.isfinite(lambda_array)]
# Subsample lambda array at even intervals:
indices = np.linspace(0, len(lambda_array) - 1, num = 100, dtype=int)
lambda_array = lambda_array[indices]

# Decision Set Clustering:
dsclust_params7 = {
    (l,) : {
        'objective' : TotalCoverageCostObjective(
            cluster_centers = kmeans_base.centers,
            n_rules = fixed_parameters['n_rules'],
            lambda_val = l,
            method = "kmeans"
        )
    }
    for i,l in enumerate(lambda_array)
}

dsclust_mod7 = DecisionSetMod(
    model = DSCluster,
    rule_miner = uniform_rule_miner,
    rules = uniform_rules,
    rule_labels = None,
    name = 'Uniform-Bin; Total-Coverage-Cost-Obj'
)


####################################################################################################
# Module 8. Cluster Bin Rules, Coverage Cost Objective

objective = TotalCoverageCostObjective(
    cluster_centers = kmeans_base.centers,
    n_rules = fixed_parameters['n_rules'],
    method = "kmeans"
)
objective.set_data(data)

# Find minimum lambda value:
dsclust = DSCluster(
    objective = objective,
    rule_miner = cluster_rule_miner,
    rules = cluster_rules,
    rule_labels = cluster_rule_labels,
)

lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
lambda_array = lambda_array[np.isfinite(lambda_array)]
# Subsample lambda array at even intervals:
indices = np.linspace(0, len(lambda_array) - 1, num = 100, dtype=int)
lambda_array = lambda_array[indices]

# Decision Set Clustering:
dsclust_params8 = {
    (l,) : {
        'objective' : TotalCoverageCostObjective(
            cluster_centers = kmeans_base.centers,
            n_rules = fixed_parameters['n_rules'],
            lambda_val = l,
            method = "kmeans"
        )
    }
    for i,l in enumerate(lambda_array)
}

dsclust_mod8 = DecisionSetMod(
    model = DSCluster,
    rule_miner = cluster_rule_miner,
    rules = cluster_rules,
    rule_labels = None,
    name = 'Cluster-Bin; Total-Coverage-Cost-Obj'
)


####################################################################################################

module_list = [
    (dsclust_mod1, dsclust_params1),
    (dsclust_mod2, dsclust_params2),
    (dsclust_mod3, dsclust_params3),
    (dsclust_mod4, dsclust_params4),
    (dsclust_mod5, dsclust_params5),
    (dsclust_mod6, dsclust_params6),
    (dsclust_mod7, dsclust_params7),
    (dsclust_mod8, dsclust_params8),
]


measurement_fns = [
    TotalCoverage(),
    ClusterCoverage(baseline_assignment = kmeans_assignment),
    Mistakes(baseline_assignment = kmeans_assignment),
    ClusteringCost(data = data, method = "kmeans"),
    RuleClusteringCost(data = data, method = "kmeans"),
    PairwiseDistance(baseline_assignment = kmeans_assignment),
    RulePairwiseDistance(baseline_assignment = kmeans_assignment),
]

exp = Experiment(
    data = data,
    baseline = kmeans_base,
    module_list = module_list,
    measurement_fns= measurement_fns,
    fixed_parameters = fixed_parameters,
    cpu_count = experiment_cpu_count,
    verbose = True
)

import time 
start = time.time()
exp_results = exp.run()
exp.save_results('data/experiments/climate/lambdas/', '')
end = time.time()
print("Experiment time:", end - start)


####################################################################################################