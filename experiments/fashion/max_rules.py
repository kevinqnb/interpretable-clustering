import os
import json
import math
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

def _memoryview_safe(x):
    """
    Make array safe to run in a Cython memoryview-based kernel. 
    As far as I can tell, this sometimes is an issue when data is pickled in 
    multiprocessing environments.
    """
    if not x.flags.writeable:
        if not x.flags.owndata:
            x = x.copy(order='C')
        x.setflags(write=True)
    return x

####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_fashion()
data = _memoryview_safe(data)
n,d = data.shape

fixed_parameters = {
    'n' : n,
    'd' : d,
    'n_clusters': 10,
    'max_rules': 20,
    'min_support': 0.05,
    'min_confidence': 0.85,
    'car_max_rule_length': 4,
    'n_forest': 100,
    'max_depth': 10,
    'depth_factor': 0.03,
    'ids_samples': 1,
    'seed': seed,
}

n_rules_list = list(range(fixed_parameters['n_clusters'], fixed_parameters['max_rules'] + 1))

np.random.seed(fixed_parameters['seed'])

# Baseline KMeans
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = fixed_parameters['seed'])
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

# Weights for uncertainty objectives
weights = distance_ratio_score(data, kmeans_base.centers)
fixed_parameters['weights'] = weights.tolist()

# Alpha values for objectives:
# Hard coded for now; will be selected via separate experiment later:
kmeans_distances = pairwise_distances(data, kmeans_base.centers)
max_distances = np.max(kmeans_distances, axis=1)
max_distance = np.max(max_distances)

alpha_dict = {
    'dscluster; coverage-mistake; ensemble': 0.01 * n * 1.0,
    'dscluster; total-coverage-mistake; ensemble': 0.01 * n * 1.0,
    'dscluster; coverage-cost; ensemble': 0.01 * n * max_distance,
    'dscluster; total-coverage-cost; ensemble': 0.01 * n * max_distance,
    'dscluster; coverage-pairwise-distance; ensemble': 0.01 * math.comb(n, 2),
    'dscluster; total-coverage-pairwise-distance; ensemble': 0.01 * math.comb(n, 2),
    'dscluster; coverage-mistake-weighted; ensemble': 0.01 * n * 1.0,
    'dscluster; total-coverage-mistake-weighted; ensemble': 0.01 * n * 1.0,
    'dscluster; coverage-cost-weighted; ensemble': 0.01 * n * max_distance,
    'dscluster; total-coverage-cost-weighted; ensemble': 0.01 * n * max_distance,
    'dscluster; coverage-pairwise-distance-weighted; ensemble': 0.01 * math.comb(n, 2),
    'dscluster; total-coverage-pairwise-distance-weighted; ensemble': 0.01 * math.comb(n, 2),
}

#with open("data/experiments/anuran/alphas/selected_alphas.json") as f:
#    selected_alpha_dict = json.load(f)
#alpha_dict = alpha_dict | selected_alpha_dict
fixed_parameters['alpha'] = alpha_dict

bin_df = pd.read_csv("data/experiments/fashion/bin_df.csv")

outfile = 'data/experiments/fashion/max_rules/'
outfile_ref = '_update'

####################################################################################################
# Rule Mining:

decision_tree_rule_miner = TreeMiner(
    tree = DecisionTree(random_state = fixed_parameters['seed']),
)
decision_tree_rules, decision_tree_rule_labels = decision_tree_rule_miner.fit(
    X = data, y = kmeans_base.labels
)


exkmc_rule_miner = TreeMiner(
    tree = ExkmcTree(
        k = fixed_parameters['n_clusters'],
        max_leaf_nodes = fixed_parameters['max_rules'],
        kmeans = kmeans_base.clustering,
        imm = True
    )
)
exkmc_rules, exkmc_rule_labels = exkmc_rule_miner.fit(
    X = data, y = kmeans_base.labels
)


shallow_tree_miner = TreeMiner(
    tree = ShallowTree(
        n_clusters = fixed_parameters['n_clusters'],
        depth_factor = fixed_parameters['depth_factor'],
        kmeans_random_state = fixed_parameters['seed']
    )
)
shallow_rules, shallow_rule_labels = shallow_tree_miner.fit(
    X = data, y = kmeans_labels
)


forest_rule_miner = RandomForestMiner(
    forest_params = {
        'n_estimators': fixed_parameters['n_forest'],
        'max_depth': fixed_parameters['max_depth'],
        'random_state': fixed_parameters['seed']
    }
)
forest_rules, forest_rule_labels = forest_rule_miner.fit(data, kmeans_base.labels)


class_association_rule_miner = ClassAssociationRuleMiner(
    min_support = fixed_parameters['min_support'],
    min_confidence = fixed_parameters['min_confidence'],
    max_length = fixed_parameters['car_max_rule_length'],
    bin_df = bin_df
)
class_association_rules, class_association_rule_labels = class_association_rule_miner.fit(
    X = data, y = kmeans_base.labels
)

ensemble_rules = decision_tree_rules + exkmc_rules + shallow_rules + forest_rules + class_association_rules
ensemble_rules = filter_rules(
    ensemble_rules, data, kmeans_labels, confidence = fixed_parameters['min_confidence']
)

rule_miner_dict = {
    'ensemble': (None, ensemble_rules, None),
}


####################################################################################################
# Comparison Modules:

# Decision Tree
decision_tree_params = {(i,) : {'max_leaf_nodes' : i, 'random_state' : fixed_parameters['seed']}
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
        'kmeans_random_state' : fixed_parameters['seed']
    } for i in n_rules_list
}
shallow_tree_mod = DecisionTreeMod(
    model = ShallowTree,
    name = 'Shallow-Tree'
)


# IDS:
'''
rule_comb = len(class_association_rules) * fixed_parameters['n_clusters']
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
        fitting_params = {'bin_df': class_association_rule_miner.bin_df},
        rules = class_association_rules,
        name = f"IDS_{s}"
    )
    ids_module_list.append((ids_mod, ids_params))
'''

####################################################################################################
# Objectives for Decision Set Clustering:

objective_dict = {
    'coverage-mistake': {
        'objective_type': 'coverage-mistake'
    },
    'total-coverage-mistake': {
        'objective_type': 'total-coverage-mistake'
    },
    'coverage-cost': {
        'cluster_centers': kmeans_base.centers,
        'objective_type': 'coverage-cost',
        'cluster_cost_method': 'kmeans'
    },
    'total-coverage-cost': {
        'cluster_centers': kmeans_base.centers,
        'objective_type': 'total-coverage-cost',
        'cluster_cost_method': 'kmeans'
    },
    'coverage-pairwise-distance': {
        'objective_type': 'coverage-pairwise-distance',
    },
    'total-coverage-pairwise-distance': {
        'objective_type': 'total-coverage-pairwise-distance',
    },
    'coverage-mistake-weighted': {
        'weights': weights,
        'objective_type': 'coverage-mistake'
    },
    'total-coverage-mistake-weighted': {
        'weights': weights,
        'objective_type': 'total-coverage-mistake'
    },
    'coverage-cost-weighted': {
        'cluster_centers': kmeans_base.centers,
        'weights': weights,
        'objective_type': 'coverage-cost',
        'cluster_cost_method': 'kmeans'
    },
    'total-coverage-cost-weighted': {
        'cluster_centers': kmeans_base.centers,
        'weights': weights,
        'objective_type': 'total-coverage-cost',
        'cluster_cost_method': 'kmeans'
    },
    'coverage-pairwise-distance-weighted': {
        'weights': weights,
        'objective_type': 'coverage-pairwise-distance',
    },
    'total-coverage-pairwise-distance-weighted': {
        'weights': weights,
        'objective_type': 'total-coverage-pairwise-distance',
    },
}

####################################################################################################
# Decision Set Clustering Modules:

dscluster_module_list = []
for obj_name, obj_params in objective_dict.items():
    for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
        module_name = f'dscluster; {obj_name}; {rule_miner_name}'
        alpha_val = alpha_dict[module_name]
        dsclust_params = {
            (r,) : {'n_select' : r, 'alpha_val' : alpha_val} | obj_params
            for i,r in enumerate(n_rules_list)
        }
        dsclust_mod = DecisionSetMod(
            model = DSCluster,
            rules = rules,
            name = module_name
        )
        dscluster_module_list.append((dsclust_mod, dsclust_params))


####################################################################################################


baseline = kmeans_base
module_list = [
    (decision_tree_mod, decision_tree_params),
    (exp_tree_mod, exp_tree_params),
    (exkmc_mod, exkmc_params),
    (shallow_tree_mod, shallow_tree_params),
] + dscluster_module_list #+ ids_module_list

measurement_fns = [
    TotalCoverage(),
    TotalCoverage(weights = weights, name = 'total-coverage-weighted'),
    TotalCoverageSet(),
    ClusterCoverage(baseline_assignment = kmeans_assignment),
    ClusterCoverage(
        baseline_assignment = kmeans_assignment, weights = weights, name = 'cluster-coverage-weighted'
    ),
    ClusterCoverageSet(baseline_assignment = kmeans_assignment),
    Overlap(),
    Mistakes(baseline_assignment = kmeans_assignment),
    ClusteringCost(data = data, average = True, normalize = True, method = "kmeans"),
    RuleClusteringCost(data = data, cluster_centers = kmeans_base.centers, method = "kmeans"),
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
exp.save_results(outfile, outfile_ref)
end = time.time()
print("Experiment time:", end - start)


####################################################################################################

