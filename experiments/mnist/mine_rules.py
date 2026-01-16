import os
import numpy as np
import pandas as pd
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.objectives import *
from intercluster.decision_sets.mining import *
from intercluster.experiments import *
from intercluster.rules import save_rules, load_rules

# Prevents memory leakage for KMeans:
os.environ["OMP_NUM_THREADS"] = "1"

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 342

####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_mnist()
n,d = data.shape

fixed_parameters = {
    'n' : n,
    'd' : d,
    'n_clusters': 10,
    'max_rules': 16,
    'min_support': 0.05,
    'min_confidence': 0.90,
    'car_max_rule_length': 4,
    'n_forest': 100,
    'max_depth': 10,
    'depth_factor': 0.03,
    'ids_samples': 1,
    'seed': seed,
}

np.random.seed(fixed_parameters['seed'])

# Do baseline clustering
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = fixed_parameters['seed'])
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

####################################################################################################
# Create bin_df for rule mining:

#bin_df = entropy_bin(
#    data, kmeans_labels, random_state = fixed_parameters['seed']
#)

#bin_df.to_csv('data/experiments/fashion/rules/bin_df.csv', index = False)

bin_df = pd.read_csv('data/experiments/mnist/rules/bin_df.csv')

####################################################################################################
# Mine for rules:


decision_tree_rule_miner = TreeMiner(
    tree = DecisionTree(random_state = fixed_parameters['seed']),
)
decision_tree_rules, decision_tree_rule_labels = decision_tree_rule_miner.fit(
    X = data, y = kmeans_base.labels
)

print("Mined DT rules:", len(decision_tree_rules))


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

print("Mined ExKMC rules:", len(exkmc_rules))

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


print("Mined Shallow rules:", len(shallow_rules))


forest_rule_miner = RandomForestMiner(
    forest_params = {
        'n_estimators': fixed_parameters['n_forest'],
        'max_depth': fixed_parameters['max_depth'],
        'random_state': fixed_parameters['seed']
    }
)
forest_rules, forest_rule_labels = forest_rule_miner.fit(data, kmeans_base.labels)

print("Mined Forest rules:", len(forest_rules))

class_association_rule_miner = ClassAssociationRuleMiner(
    min_support = fixed_parameters['min_support'],
    min_confidence = fixed_parameters['min_confidence'],
    max_length = fixed_parameters['car_max_rule_length'],
    bin_df = bin_df
)
class_association_rules, class_association_rule_labels = class_association_rule_miner.fit(
    X = data, y = kmeans_base.labels
)

print("Mined CAR rules:", len(class_association_rules))

ensemble_rules = decision_tree_rules + exkmc_rules + shallow_rules + forest_rules + class_association_rules
ensemble_rules = filter_rules(
    ensemble_rules, data, kmeans_labels, confidence = fixed_parameters['min_confidence']
)

print("Total ensemble rules after filtering:", len(ensemble_rules))

# Save mined rules and their labels:
save_rules(decision_tree_rules, 'data/experiments/mnist/rules/decision_tree_rules.pkl')
save_rules(exkmc_rules, 'data/experiments/mnist/rules/exkmc_rules.pkl')
save_rules(shallow_rules, 'data/experiments/mnist/rules/shallow_rules.pkl')
save_rules(forest_rules, 'data/experiments/mnist/rules/forest_rules.pkl')
save_rules(class_association_rules, 'data/experiments/mnist/rules/class_association_rules.pkl')
save_rules(ensemble_rules, 'data/experiments/mnist/rules/ensemble_rules.pkl')