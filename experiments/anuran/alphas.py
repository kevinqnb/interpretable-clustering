import os
import math
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
data, labels, feature_labels, scaler = load_preprocessed_anuran('data/anuran')
n,d = data.shape

fixed_parameters = {
    'n' : n,
    'd' : d,
    'n_clusters': 6,
    'n_select': 6,
    'min_support': 0.05, 
    'min_confidence': 0.85,
    'max_rule_length': 4,
    'n_forest': 100,
    'max_depth': None,
    'depth_factor': 0.03,
    'seed': seed,
}

np.random.seed(fixed_parameters['seed'])

# Do baseline clustering
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = fixed_parameters['seed'])
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels
kmeans_distances = pairwise_distances(data, kmeans_base.centers)
furthest_distance = np.max(np.max(kmeans_distances, axis=1))

# Weights for uncertainty objectives
weights = distance_ratio_score(data, kmeans_base.centers)
fixed_parameters['weights'] = weights.tolist()

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
    max_length = fixed_parameters['max_rule_length'],
    binning_method = "entropy",
    bin_params = {
        'random_state': fixed_parameters['seed'],
    }
)
class_association_rules, class_association_rule_labels = class_association_rule_miner.fit(
    X = data, y = kmeans_base.labels
)


ensemble_rules = decision_tree_rules + exkmc_rules + shallow_rules + forest_rules + class_association_rules
ensemble_rules = filter_rules(
    ensemble_rules, data, kmeans_labels, confidence = fixed_parameters['min_confidence']
)

rule_miner_dict = {
    'ensemble' : (None, ensemble_rules, None),
}


####################################################################################################
# Define Objectives:


objective_dict = {
    'coverage-mistake': {
        'n_select': fixed_parameters['n_select'],
        'objective_type': 'coverage-mistake'
    },
    'total-coverage-mistake': {
        'n_select': fixed_parameters['n_select'],
        'objective_type': 'total-coverage-mistake'
    },
    'coverage-cost': {
        'n_select': fixed_parameters['n_select'],
        'cluster_centers': kmeans_base.centers,
        'objective_type': 'coverage-cost',
        'cluster_cost_method': 'kmeans'
    },
    'total-coverage-cost': {
        'n_select': fixed_parameters['n_select'],
        'cluster_centers': kmeans_base.centers,
        'objective_type': 'total-coverage-cost',
        'cluster_cost_method': 'kmeans'
    },
    'coverage-pairwise-distance': {
        'n_select': fixed_parameters['n_select'],
        'objective_type': 'coverage-pairwise-distance',
    },
    'total-coverage-pairwise-distance': {
        'n_select': fixed_parameters['n_select'],
        'objective_type': 'total-coverage-pairwise-distance',
    },
    'coverage-mistake-weighted': {
        'n_select': fixed_parameters['n_select'],
        'weights': weights,
        'objective_type': 'coverage-mistake'
    },
    'total-coverage-mistake-weighted': {
        'n_select': fixed_parameters['n_select'],
        'weights': weights,
        'objective_type': 'total-coverage-mistake'
    },
    'coverage-cost-weighted': {
        'n_select': fixed_parameters['n_select'],
        'cluster_centers': kmeans_base.centers,
        'weights': weights,
        'objective_type': 'coverage-cost',
        'cluster_cost_method': 'kmeans'
    },
    'total-coverage-cost-weighted': {
        'n_select': fixed_parameters['n_select'],
        'cluster_centers': kmeans_base.centers,
        'weights': weights,
        'objective_type': 'total-coverage-cost',
        'cluster_cost_method': 'kmeans'
    },
    'coverage-pairwise-distance-weighted': {
        'n_select': fixed_parameters['n_select'],
        'weights': weights,
        'objective_type': 'coverage-pairwise-distance',
    },
    'total-coverage-pairwise-distance-weighted': {
        'n_select': fixed_parameters['n_select'],
        'weights': weights,
        'objective_type': 'total-coverage-pairwise-distance',
    },
}


# List of alpha values to try for each objective
n_compare = 25
objective_alpha_dict = {
    'coverage-mistake': np.linspace(0.0, fixed_parameters['n'], num = n_compare),
    'total-coverage-mistake': np.linspace(0.0, fixed_parameters['n'], num = n_compare),
    'coverage-cost': np.linspace(0.0, furthest_distance * n, num = n_compare),
    'total-coverage-cost': np.linspace(0.0, furthest_distance * n, num = n_compare),
    'coverage-pairwise-distance': np.linspace(0.0, math.comb(fixed_parameters['n'], 2), num = n_compare),
    'total-coverage-pairwise-distance': np.linspace(0.0, math.comb(fixed_parameters['n'], 2), num = n_compare),
    'coverage-mistake-weighted': np.linspace(0.0, fixed_parameters['n'], num = n_compare),
    'total-coverage-mistake-weighted': np.linspace(0.0, fixed_parameters['n'], num = n_compare),
    'coverage-cost-weighted': np.linspace(0.0, furthest_distance * n, num = n_compare),
    'total-coverage-cost-weighted': np.linspace(0.0, furthest_distance * n, num = n_compare),
    'coverage-pairwise-distance-weighted': np.linspace(0.0, math.comb(fixed_parameters['n'], 2), num = n_compare),
    'total-coverage-pairwise-distance-weighted': np.linspace(0.0, math.comb(fixed_parameters['n'], 2), num = n_compare),
}


####################################################################################################
# Create experiment modules:

module_list = []
for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
    for obj_name, obj_params in objective_dict.items():
        module_name = f'dscluster; {obj_name}; {rule_miner_name}'
        alpha_vals = objective_alpha_dict[obj_name]
        for alpha in alpha_vals:
            obj_parameterized = {
                (alpha_val,): obj_params | {'alpha_val': alpha_val}
                for alpha_val in objective_alpha_dict[obj_name]
            }

            module = DecisionSetMod(
                model = DSCluster,
                rules = rules,
                name = f'dscluster; {obj_name}; {rule_miner_name}'
            )
            module_list.append((module, obj_parameterized))


####################################################################################################
# Run Experiment:

measurement_fns = [
    TotalCoverage(),
    TotalCoverage(weights = weights, name = 'total-coverage-weighted'),
    ClusterCoverage(baseline_assignment = kmeans_assignment),
    ClusterCoverage(
        baseline_assignment = kmeans_assignment, weights = weights, name = 'cluster-coverage-weighted'
    ),
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
exp.save_results('data/experiments/anuran/alphas/', '_update')
end = time.time()
print("Experiment time:", end - start)


####################################################################################################