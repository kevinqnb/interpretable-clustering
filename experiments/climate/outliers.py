import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.ensemble import IsolationForest
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
#lambdas_fname = 'data/experiments/climate/lambdas/selected_lambdas_alpha_zero.json'
#with open(lambdas_fname, 'r') as f:
#    selected_lambdas = json.load(f)

fixed_parameters = {
    'n' : n,
    'd' : d,
    'n_clusters': 6,
    'n_select': 6,
    'max_rules': 12,
    'min_support': 0.05,
    'min_confidence': 0.85,
    'max_rule_length': 4,
    'depth_factor': 0.03,
    'alpha_mistakes': 0.0,
    'lambdas' : {},
}

fixed_parameters['alpha_rule_clustering_cost'] = 0.0

np.random.seed(seed)

# Baseline KMeans
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = seed)
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

# Find outlier scores:
clf = IsolationForest(random_state=0).fit(data)
isolation_scores = clf.score_samples(data) + 1
ratio_scores = distance_ratio_score(data, kmeans_base.centers)

fixed_parameters['weights'] = {
    'baseline': np.ones(n),
    'isolation-tree': isolation_scores,
    'distance-ratio': ratio_scores
}

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
    'ensemble': (None, ensemble_rules, None),
}


####################################################################################################
# Objectives for Decision Set Clustering:

objective_dict = {
    'coverage-mistake': {
        'n_select': fixed_parameters['n_select'],
        'alpha_val': fixed_parameters['alpha_mistakes'],
        'objective_type': 'coverage-mistake'
    },
    'total-coverage-mistake': {
        'n_select': fixed_parameters['n_select'],
        'alpha_val': fixed_parameters['alpha_mistakes'],
        'objective_type': 'total-coverage-mistake'
    },
    'coverage-cost': {
        'n_select': fixed_parameters['n_select'],
        'alpha_val': fixed_parameters['alpha_rule_clustering_cost'],
        'cluster_centers': kmeans_base.centers,
        'objective_type': 'coverage-cost',
        'cluster_cost_method': 'kmeans'
    },
    'total-coverage-cost': {
        'n_select': fixed_parameters['n_select'],
        'alpha_val': fixed_parameters['alpha_rule_clustering_cost'],
        'cluster_centers': kmeans_base.centers,
        'objective_type': 'total-coverage-cost',
        'cluster_cost_method': 'kmeans'
    }
}


####################################################################################################
# Find covered set for all objective, rule miner, and weight combinations:

module_results = {}
for obj_name, obj_params in objective_dict.items():
    for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
        module_name = f'dscluster; {obj_name}; {rule_miner_name}'
        if module_name not in module_results:
            module_results[module_name] = {}
        for weight_name, weights in fixed_parameters['weights'].items():
            dsclust = DSCluster(
                rules = rules,
                weights = weights,
                **obj_params
            )
            dsclust.fit(data, kmeans_labels)
            dclust_labels = dsclust.predict(data)

            lambda_val = dsclust.objective.lambda_val
            fixed_parameters['lambdas'][module_name] = lambda_val


            dsclust_labels = dsclust.predict(data)
            dsclust_data_to_rule_assignment = dsclust.get_data_to_rules_assignment(data)
            dsclust_rule_labels = dsclust.decision_set_labels
            dsclust_rule_to_cluster_assignment = labels_to_assignment(
                dsclust_rule_labels, n_labels = fixed_parameters['n_select'], ignore = {-1}
            )
            dsclust_data_to_cluster_assignment = dsclust_data_to_rule_assignment @ dsclust_rule_to_cluster_assignment
            covered = covered_mask(dsclust_data_to_cluster_assignment)
            module_results[module_name][weight_name] = covered.tolist()


####################################################################################################
# Save results:

# Convert numpy arrays to lists for JSON serialization
fixed_parameters_serializable = fixed_parameters.copy()
fixed_parameters_serializable['weights'] = {    
    k: v.tolist() if isinstance(v, np.ndarray) else v 
    for k, v in fixed_parameters['weights'].items()
}

experiment_results = {
    'fixed_parameters': fixed_parameters_serializable,
    'modules': module_results
}   

results_fname = 'data/experiments/climate/outliers/exp.json'
with open(results_fname, 'w') as f:
    json.dump(experiment_results, f)


####################################################################################################

