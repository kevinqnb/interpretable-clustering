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

experiment_cpu_count = 8

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 342

####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_climate('data/climate')
euclidean_distances = pairwise_distances(data)
n,d = data.shape

##### Parameters #####
fixed_parameters = {
    'n' : n,
    'd' : d,
    'n_clusters': 6,
    'max_rules': 6 + 20,
    'min_support': 0.025,
    'max_rule_length': 3,
    'n_bins': 6,
    'per_cluster_cost': 0.25,
    'depth_factor': 0.03,
    'ids_samples': 10,
    'lambdas' : {}
}

n_rules_list = list(range(fixed_parameters['n_clusters'], fixed_parameters['max_rules'] + 1))

####################################################################################################

np.random.seed(seed)

# Baseline KMeans
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = seed)
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels


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
# Comparison Modules:

# Decision Tree
decision_tree_params = {(i,) : {'max_leaf_nodes' : i, 'random_state' : seed}
                        for i in n_rules_list}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)


# Explanation Tree
exp_tree_params = {tuple(n_rules_list) : {'num_clusters' : fixed_parameters['n_clusters']}}
exp_tree_mod = DecisionTreeMod(
    model = ExplanationTree,
    name = 'Exp-Tree'
)


# ExKMC
exkmc_params = {
    (i,) : {
        'k' : fixed_parameters['n_clusters'],
        'kmeans': kmeans_base.clustering,
        'max_leaf_nodes': i
    } for i in n_rules_list
}
exkmc_mod = DecisionTreeMod(
    model = ExkmcTree,
    name = 'ExKMC'
)


# Shallow Tree
shallow_tree_params = {
    tuple(n_rules_list) : {
        'n_clusters' : fixed_parameters['n_clusters'],
        'depth_factor' : fixed_parameters['depth_factor'],
        'kmeans_random_state' : seed
    } for i in n_rules_list
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
'''

# IDS
rule_comb = len(uniform_rules) * fixed_parameters['n_clusters']
ids_lambdas = [
    1/rule_comb,
    1/(2 * data.shape[1] * rule_comb),
    1/(len(data) * (rule_comb**2)),
    1/(len(data) * (rule_comb**2)),
    1/fixed_parameters['n_clusters'],
    1/(data.shape[0] * rule_comb),
    1/(data.shape[0])
]

ids_module_list = []
for s in range(fixed_parameters['ids_samples']):
    ids_params = {
        tuple(n_rules_list) : {
            'lambdas' : ids_lambdas
        }
    }
    ids_mod = DecisionSetMod(
        model = IDS,
        rules = uniform_rules,
        rule_labels = None,
        rule_miner = uniform_rule_miner,
        name = f"IDS_{s}"
    )
    ids_module_list.append((ids_mod, ids_params))


####################################################################################################
# Decision Set Clustering Modules:

####################################################################################################
# Module 1. Uniform Bin Rules, Coverage Mistake Objective

objective = CoverageMistakeObjective(
    n_rules = fixed_parameters['n_clusters']
)

# Find minimum lambda value:
dsclust = DSCluster(
    objective = objective,
    rule_miner = uniform_rule_miner,
    rules = uniform_rules,
    rule_labels = uniform_rule_labels,
)

#lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
#lambda_val = lambda_array[0]
lambda_val = 2.143

# Decision Set Clustering:
dsclust_params1 = {
    (r,) : {
        'objective' : CoverageMistakeObjective(
            n_rules = r,
            lambda_val = lambda_val
        )
    }
    for i,r in enumerate(n_rules_list)
}

dsclust_mod1 = DecisionSetMod(
    model = DSCluster,
    rule_miner = uniform_rule_miner,
    rules = uniform_rules,
    rule_labels = None,
    name = 'DSCluster; Uniform-Bin; Coverage-Mistake-Obj'
)

fixed_parameters['lambdas'][dsclust_mod1.name] = lambda_val


####################################################################################################
# Module 2. Cluster Bin Rules, Coverage Mistake Objective

objective = CoverageMistakeObjective(
    n_rules = fixed_parameters['n_clusters']
)

# Find minimum lambda value:
dsclust = DSCluster(
    objective = objective,
    rule_miner = cluster_rule_miner,
    rules = cluster_rules,
    rule_labels = cluster_rule_labels,
)

#lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
#lambda_val = lambda_array[0]
lambda_val = 1.765

# Decision Set Clustering:
dsclust_params2 = {
    (r,) : {
        'objective' : CoverageMistakeObjective(
            n_rules = r,
            lambda_val = lambda_val
        )
    }
    for i,r in enumerate(n_rules_list)
}

dsclust_mod2 = DecisionSetMod(
    model = DSCluster,
    rule_miner = cluster_rule_miner,
    rules = cluster_rules,
    rule_labels = None,
    name = 'DSCluster; Cluster-Bin; Coverage-Mistake-Obj'
)

fixed_parameters['lambdas'][dsclust_mod2.name] = lambda_val


####################################################################################################
# Module 3. Uniform Bin Rules, Coverage Cost Objective

objective = CoverageCostObjective(
    cluster_centers = kmeans_base.centers,
    n_rules = fixed_parameters['n_clusters'],
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
lambda_val = 0.04193

# Decision Set Clustering:
dsclust_params3 = {
    (r,) : {
        'objective' : CoverageCostObjective(
            cluster_centers = kmeans_base.centers,
            n_rules = r,
            lambda_val = lambda_val,
            method = "kmeans"
        )
    }
    for i,r in enumerate(n_rules_list)
}

dsclust_mod3 = DecisionSetMod(
    model = DSCluster,
    rule_miner = uniform_rule_miner,
    rules = uniform_rules,
    rule_labels = None,
    name = 'DSCluster; Uniform-Bin; Coverage-Cost-Obj'
)

fixed_parameters['lambdas'][dsclust_mod3.name] = lambda_val


####################################################################################################
# Module 4. Cluster Bin Rules, Coverage Cost Objective

objective = CoverageCostObjective(
    cluster_centers = kmeans_base.centers,
    n_rules = fixed_parameters['n_clusters'],
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

#lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
#lambda_val = lambda_array[0]\
lambda_val = 0.03254

# Decision Set Clustering:
dsclust_params4 = {
    (r,) : {
        'objective' : CoverageCostObjective(
            cluster_centers = kmeans_base.centers,
            n_rules = r,
            lambda_val = lambda_val,
            method = "kmeans"
        )
    }
    for i,r in enumerate(n_rules_list)
}

dsclust_mod4 = DecisionSetMod(
    model = DSCluster,
    rule_miner = cluster_rule_miner,
    rules = cluster_rules,
    rule_labels = None,
    name = 'DSCluster; Cluster-Bin; Coverage-Cost-Obj'
)

fixed_parameters['lambdas'][dsclust_mod4.name] = lambda_val


####################################################################################################
# Module 5. Uniform Bin Rules, Total Coverage Mistake Objective

objective = TotalCoverageMistakeObjective(
    n_rules = fixed_parameters['n_clusters']
)

# Find minimum lambda value:
dsclust = DSCluster(
    objective = objective,
    rule_miner = uniform_rule_miner,
    rules = uniform_rules,
    rule_labels = uniform_rule_labels,
)

#lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
#lambda_val = lambda_array[0]
lambda_val = 3.143

# Decision Set Clustering:
dsclust_params5 = {
    (r,) : {
        'objective' : TotalCoverageMistakeObjective(
            n_rules = r,
            lambda_val = lambda_val
        )
    }
    for i,r in enumerate(n_rules_list)
}

dsclust_mod5 = DecisionSetMod(
    model = DSCluster,
    rule_miner = uniform_rule_miner,
    rules = uniform_rules,
    rule_labels = None,
    name = 'DSCluster; Uniform-Bin; Total-Coverage-Mistake-Obj'
)

fixed_parameters['lambdas'][dsclust_mod5.name] = lambda_val


####################################################################################################
# Module 6. Cluster Bin Rules, Total Coverage Mistake Objective

objective = TotalCoverageMistakeObjective(
    n_rules = fixed_parameters['n_clusters']
)

# Find minimum lambda value:
dsclust = DSCluster(
    objective = objective,
    rule_miner = cluster_rule_miner,
    rules = cluster_rules,
    rule_labels = cluster_rule_labels,
)

#lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
#lambda_val = lambda_array[0]
lambda_val = 3.2174

# Decision Set Clustering:
dsclust_params6 = {
    (r,) : {
        'objective' : TotalCoverageMistakeObjective(
            n_rules = r,
            lambda_val = lambda_val
        )
    }
    for i,r in enumerate(n_rules_list)
}

dsclust_mod6 = DecisionSetMod(
    model = DSCluster,
    rule_miner = cluster_rule_miner,
    rules = cluster_rules,
    rule_labels = None,
    name = 'DSCluster; Cluster-Bin; Total-Coverage-Mistake-Obj'
)

fixed_parameters['lambdas'][dsclust_mod6.name] = lambda_val


####################################################################################################
# Module 7. Uniform Bin Rules, Total Coverage Cost Objective

objective = TotalCoverageCostObjective(
    cluster_centers = kmeans_base.centers,
    n_rules = fixed_parameters['n_clusters'],
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

#lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
#lambda_val = lambda_array[0]
lambda_val = 0.03629

# Decision Set Clustering:
dsclust_params7 = {
    (r,) : {
        'objective' : TotalCoverageCostObjective(
            cluster_centers = kmeans_base.centers,
            n_rules = r,
            lambda_val = lambda_val,
            method = "kmeans"
        )
    }
    for i,r in enumerate(n_rules_list)
}

dsclust_mod7 = DecisionSetMod(
    model = DSCluster,
    rule_miner = uniform_rule_miner,
    rules = uniform_rules,
    rule_labels = None,
    name = 'DSCluster; Uniform-Bin; Total-Coverage-Cost-Obj'
)

fixed_parameters['lambdas'][dsclust_mod7.name] = lambda_val


####################################################################################################
# Module 8. Cluster Bin Rules, Coverage Cost Objective

objective = TotalCoverageCostObjective(
    cluster_centers = kmeans_base.centers,
    n_rules = fixed_parameters['n_clusters'],
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

#lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
#lambda_val = lambda_array[0]
lambda_val = 0.03502

# Decision Set Clustering:
dsclust_params8 = {
    (r,) : {
        'objective' : TotalCoverageCostObjective(
            cluster_centers = kmeans_base.centers,
            n_rules = r,
            lambda_val = lambda_val,
            method = "kmeans"
        )
    }
    for i,r in enumerate(n_rules_list)
}

dsclust_mod8 = DecisionSetMod(
    model = DSCluster,
    rule_miner = cluster_rule_miner,
    rules = cluster_rules,
    rule_labels = None,
    name = 'DSCluster; Cluster-Bin; Total-Coverage-Cost-Obj'
)

fixed_parameters['lambdas'][dsclust_mod8.name] = lambda_val


####################################################################################################


baseline = kmeans_base
module_list = [
    (decision_tree_mod, decision_tree_params),
    (exp_tree_mod, exp_tree_params),
    (exkmc_mod, exkmc_params),
    (shallow_tree_mod, shallow_tree_params),
    #(cba_mod, cba_params),
    (dsclust_mod1, dsclust_params1),
    (dsclust_mod2, dsclust_params2),
    (dsclust_mod3, dsclust_params3),
    (dsclust_mod4, dsclust_params4),
    (dsclust_mod5, dsclust_params5),
    (dsclust_mod6, dsclust_params6),
    (dsclust_mod7, dsclust_params7),
    (dsclust_mod8, dsclust_params8)
] #+ ids_module_list


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
exp.save_results('data/experiments/climate/max_rules/', '_rule_tuning')
end = time.time()
print("Experiment time:", end - start)


####################################################################################################

