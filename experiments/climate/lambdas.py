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
    'n_bins': 6,
    'min_support': 0.05,
    'min_confidence': 0.8,
    'max_rule_length': 4,
    'per_cluster_cost': 0.25,
    'alpha_mistakes': 0.01 * n * 1.0,
}

np.random.seed(seed)

# Do baseline clustering
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = seed)
kmeans_assignment = kmeans_base.assign(data)


# Find average distance of points to their closest cluster center
kmeans_distances = pairwise_distances(data, kmeans_base.centers)
closest_distances = np.min(kmeans_distances, axis=1)
average_distance = np.mean(closest_distances)
fixed_parameters['alpha_rule_clustering_cost'] = 0.01 * n * average_distance
fixed_parameters['alpha_rule_mean_cost'] = 0.01 * n * average_distance


####################################################################################################
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

'''
clique_rule_miner = CliqueMiner(
    density = 20/len(data),
    bin_method = 'adaptive_grid',
    bin_params = {
        'initial_bins': 100,
    }
)
clique_rules, clique_rule_labels = clique_rule_miner.fit(data)
'''


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
    #'fim-uniform': (uniform_rule_miner, uniform_rules, None),
    #'fim-cluster': (cluster_rule_miner, cluster_rules, None),
    #'clique': (clique_rule_miner, clique_rules, None),
    'car-entropy': (class_association_rule_miner, class_association_rules, None),
    'decision-tree': (decision_tree_rule_miner, decision_tree_rules, None),
    'exkmc': (exkmc_rule_miner, exkmc_rules, None),
    'random-forest': (forest_rule_miner, forest_rules, None)
}


####################################################################################################
# Find Rule-Mined centers:

center_objective = TotalCoverageRuleCostObjective(
    data = data,
    n_rules = fixed_parameters['n_rules'],
    alpha_val = fixed_parameters['alpha_rule_mean_cost'],
    method = "kmeans"
)

dsclust = DSCluster(
    objective = center_objective,
    rule_miner = forest_rule_miner,
    rules = forest_rules,
    rule_labels = None,
)

lambda_array = dsclust.compute_lambdas(data, None)
dsclust.set_lambda(lambda_array[0])
dsclust.fit(data, None)
dsclust_labels = dsclust.predict(data)

labels_dict = {-1:-1}
new_dsclust_labels = []
for l in dsclust_labels:
    lint = list(l)[0]
    if lint not in labels_dict:
        labels_dict[lint] = len(labels_dict.keys()) - 1
    new_dsclust_labels.append({labels_dict[lint]})
dsclust_labels = new_dsclust_labels
n_labels = fixed_parameters['n_rules']
dsclust_assignment = labels_to_assignment(
    dsclust_labels, n_labels = n_labels, ignore = {-1}
)
dsclust_centers = np.zeros((n_labels, d))
for r in range(n_labels):
    if np.sum(dsclust_assignment[:, r]) > 0:
        dsclust_centers[r] = np.mean(
            data[dsclust_assignment[:, r] == 1], axis = 0
        )
    else:
        dsclust_centers[r] = data[np.random.choice(data.shape[0], 1)[0]]


####################################################################################################
# Define Objectives:

objective1 = CoverageMistakeObjective(
    n_rules = fixed_parameters['n_rules'],
    alpha_val = fixed_parameters['alpha_mistakes']
)

objective2 = TotalCoverageMistakeObjective(
    n_rules = fixed_parameters['n_rules'],
    alpha_val = fixed_parameters['alpha_mistakes']
)

objective3 = CoverageCostObjective(
    data = data,
    cluster_centers = kmeans_base.centers,
    n_rules = fixed_parameters['n_rules'],
    alpha_val = fixed_parameters['alpha_rule_clustering_cost'],
    method = "kmeans"
)

objective4 = TotalCoverageCostObjective(
    data = data,
    cluster_centers = kmeans_base.centers,
    n_rules = fixed_parameters['n_rules'],
    alpha_val = fixed_parameters['alpha_rule_clustering_cost'],
    method = "kmeans"
)

objective5 = TotalCoverageCostObjective(
    data = data,
    cluster_centers = dsclust_centers,
    n_rules = fixed_parameters['n_rules'],
    alpha_val = fixed_parameters['alpha_rule_clustering_cost'],
    method = "kmeans"
)


objective_dict = {
    'coverage-mistake': objective1,
    'total-coverage-mistake': objective2,
    'coverage-cost': objective3,
    'total-coverage-cost': objective4,
    'total-coverage-cost-unsupervised': objective5
}


####################################################################################################
# Create experiment modules:

module_list = []
for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
    for obj_name, obj in objective_dict.items():
        # Find minimum lambda value:
        dsclust = DSCluster(
            objective = obj,
            rule_miner = rule_miner,
            rules = rules,
            rule_labels = rule_labels,
        )

        # NOTE: Does this affect the total coverage cost objective, which doesn't use labels?
        lambda_array = dsclust.compute_lambdas(data, kmeans_base.labels)
        lambda_array = lambda_array[np.isfinite(lambda_array)]
        # Subsample lambda array at even intervals:
        indices = np.linspace(0, len(lambda_array) - 1, num = min(len(lambda_array), 100), dtype=int)
        lambda_array = lambda_array[indices]
        # Add on a few values between 0 and min lambda:
        #prelim = np.linspace(0, lambda_array[0], num = 10)
        #lambda_array = np.concatenate((prelim, lambda_array))

        dsclust_params = {
            (l,) : {
                'objective' : type(obj)(
                    **{k: v for k, v in obj.__dict__.items() if k not in ['lambda_val', 'data_to_center_distances']},
                    lambda_val = l
                )
            }
            for i,l in enumerate(lambda_array)
        }

        dsclust_mod = DecisionSetMod(
            model = DSCluster,
            rule_miner = rule_miner,
            rules = rules,
            rule_labels = rule_labels,
            name = f'dscluster; {rule_miner_name}; {obj_name}'
        )
        module_list.append((dsclust_mod, dsclust_params))


####################################################################################################
# Run Experiment:

measurement_fns = [
    TotalCoverage(),
    ClusterCoverage(baseline_assignment = kmeans_assignment),
    Mistakes(baseline_assignment = kmeans_assignment),
    ClusteringCost(data = data, average = True, normalize = True, method = "kmeans"),
    RuleClusteringCost(data = data, cluster_centers = kmeans_base.centers, method = "kmeans"),
    RuleClusteringCost(data = data, cluster_centers = dsclust_centers, method = "kmeans", name = "rule-mean-cost"),
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
exp.save_results('data/experiments/climate/lambdas/', '_alpha')
end = time.time()
print("Experiment time:", end - start)


####################################################################################################