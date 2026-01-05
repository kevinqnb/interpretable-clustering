import os
import json
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
data, labels, feature_labels, scaler = load_preprocessed_anuran('data/anuran')
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
    'max_rules': 6 + 20,
    'min_support': 0.05,
    'min_confidence': 0.8,
    'max_rule_length': 5,
    'depth_factor': 0.03,
    'lambdas' : {},
    'alpha_mistakes': 0.5 * n * 1.0
}

n_rules_list = list(range(fixed_parameters['n_clusters'], fixed_parameters['max_rules'] + 1))

np.random.seed(seed)

# Baseline KMeans
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = seed)
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

# Find average distance of points to their closest cluster center
kmeans_distances = pairwise_distances(data, kmeans_base.centers)
closest_distances = np.min(kmeans_distances, axis=1)
average_distance = np.mean(closest_distances)
fixed_parameters['alpha_rule_clustering_cost'] = 0.5 * n * average_distance

####################################################################################################
# Rule Mining:

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


forest_rule_miner = RandomForestMiner(forest_params = {'n_estimators': 100, 'random_state': seed})
forest_rules, forest_rule_labels = forest_rule_miner.fit(data, kmeans_base.labels)

rule_miner_dict = {
    'decision-tree': (decision_tree_rule_miner, decision_tree_rules, None),
    'exkmc': (exkmc_rule_miner, exkmc_rules, None),
    'random-forest': (forest_rule_miner, forest_rules, None),
    'car-entropy': (class_association_rule_miner, class_association_rules, None),
}


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


####################################################################################################
# Decision Set Clustering Modules:

objective1 = CoverageMistakeObjective(
    n_rules = -1, # Placeholder, will be set later
    lambda_val = -1.0, # Placeholder, will be set later
    alpha_val = fixed_parameters['alpha_mistakes']
)

objective2 = CoverageCostObjective(
    data = data,
    cluster_centers = kmeans_base.centers,
    n_rules = -1, # Placeholder, will be set later
    lambda_val = -1.0, # Placeholder, will be set later
    alpha_val = fixed_parameters['alpha_rule_clustering_cost'],
    method = "kmeans"
)


objective_dict = {
    'coverage-mistake': objective1,
    'coverage-cost': objective2,
}

dscluster_module_list = []
for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
    for obj_name, obj in objective_dict.items():
        module_name = f'dscluster; {rule_miner_name}; {obj_name}'
        #lambda_val = fixed_parameters['lambdas'][module_name]
        obj_eval = type(obj)(
            **{k: v for k, v in obj.__dict__.items()
               if k not in ['n_rules','lambda_val','data_to_center_distances']},
            n_rules = n_rules_list[0],
            lambda_val = -1 # Placeholder, will be set later
        )

        # Find lambda_val for this module
        dsclust = DSCluster(
            objective = obj_eval,
            rule_miner = rule_miner,
            rules = rules,
            rule_labels = rule_labels,
        )
        if rule_miner_name == 'random-forest':
            dsclust.filter_rules(data, kmeans_labels, remove_top = 0.05)
        lambda_vals = dsclust.compute_lambdas(data, kmeans_labels)
        lambda_val = lambda_vals[0]

        # Decision Set Clustering Parameters:
        dsclust_params = {
            (r,) : {
                'objective' : type(obj)(
                    **{k: v for k, v in obj.__dict__.items()
                       if k not in ['n_rules','lambda_val', 'data_to_center_distances']},
                    n_rules = r,
                    lambda_val = lambda_val
                )
            }
            for i,r in enumerate(n_rules_list)
        }

        dsclust_mod = DecisionSetMod(
            model = DSCluster,
            rule_miner = rule_miner,
            rules = rules,
            rule_labels = rule_labels,
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
] + dscluster_module_list


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
exp.save_results('data/experiments/anuran/max_rules/', '_tuned')
end = time.time()
print("Experiment time:", end - start)


####################################################################################################

