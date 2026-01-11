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
    'n_select': 6,
    'min_support': 0.05, 
    'min_confidence': 0.85,
    'max_rule_length': 4,
    'depth_factor': 0.03,
    'lambdas': {}
}

np.random.seed(seed)

# Do baseline clustering
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = seed)
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

# Find average distance of points to their closest cluster center
kmeans_distances = pairwise_distances(data, kmeans_base.centers)
furthest_distance = np.max(np.max(kmeans_distances, axis=1))

####################################################################################################
# Rule Mining:

decision_tree_rule_miner = TreeMiner(
    tree = DecisionTree(random_state = seed),
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
        kmeans_random_state = seed
    )
)
shallow_rules, shallow_rule_labels = shallow_tree_miner.fit(
    X = data, y = kmeans_labels
)


forest_rule_miner = RandomForestMiner(forest_params = {'n_estimators': 100, 'random_state': seed})
forest_rules, forest_rule_labels = forest_rule_miner.fit(data, kmeans_base.labels)


class_association_rule_miner = ClassAssociationRuleMiner(
    min_support = fixed_parameters['min_support'],
    min_confidence = fixed_parameters['min_confidence'],
    max_length = fixed_parameters['max_rule_length'],
    binning_method = "entropy",
    bin_params = {
        'random_state': seed,
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
    }
}


# List of alpha values to try for each objective
objective_alpha_dict = {
    'coverage-mistake': np.linspace(0.0, fixed_parameters['n'], num = 100),
    'total-coverage-mistake': np.linspace(0.0, fixed_parameters['n'], num = 100),
    'coverage-cost': np.linspace(0.0, furthest_distance * n, num = 100),
    'total-coverage-cost': np.linspace(0.0, furthest_distance * n, num = 100),
}


####################################################################################################
# Create experiment modules:

module_list = []
for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
    for obj_name, obj_params in objective_dict.items():
        module_name = f'dscluster; {obj_name}; {rule_miner_name}'
        if module_name not in fixed_parameters['lambdas']:
            fixed_parameters['lambdas'][module_name] = {}
        alpha_vals = objective_alpha_dict[obj_name]
        for alpha in alpha_vals:
            dsclust = DSCluster(
                rules = rules,
                alpha_val = alpha,
                **obj_params
            )
            dsclust.fit(data, kmeans_labels)
            lambda_val = dsclust.objective.lambda_val
            fixed_parameters['lambdas'][f'dscluster; {obj_name}; {rule_miner_name}'][alpha] = lambda_val

            obj_parameterized = {
                (alpha_val,): obj_params | {'alpha_val': alpha_val, 'lambda_val': lambda_val}
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
    ClusterCoverage(baseline_assignment = kmeans_assignment),
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
exp.save_results('data/experiments/climate/alphas/', '')
end = time.time()
print("Experiment time:", end - start)


####################################################################################################