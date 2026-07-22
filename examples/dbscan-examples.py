# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: interclusternb-py3.12
#     language: python
#     name: interclusternb-py3.12
# ---

# %%
import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.objectives import *
from intercluster.decision_sets.mining import *
# %load_ext autoreload
# %autoreload 2

# %%
####################################################################################################
# Path setup

import sys
from pathlib import Path

# Ensure the repository root (the folder that contains `data/`) is on sys.path.
# This makes `from data.preprocessing import ...` work when running this file directly.
_HERE = Path.cwd().resolve()
PROJECT_ROOT = next((p for p in _HERE.parents if (p / "data").is_dir()), None)
if PROJECT_ROOT is None:
    raise ModuleNotFoundError(
        "Could not locate repository root."
    )
sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import *
from experiments.experiment import *
from experiments.modules import *

####################################################################################################

# %%
# NOTE: The following is important for recreating results, 
# But can be changed for randomized analysis.
seed = 170
np.random.seed(seed)

# %%
palette = sns.color_palette("husl", 8)
cmap = ListedColormap(palette)

color_dict = {i: cmap(7 - i) for i in range(8)}
color_dict[-1] = 'grey'

cmap2 = ListedColormap(sns.color_palette("tab20", 20))
color_dict2 = {i: cmap2(i) for i in range(len(cmap2.colors))}
color_dict2[-1] = 'grey'

# This assumes tex is installed in your system, 
# if not, you may simply remove most of this, aside from font.size 
# (doing so, however, will break certain plotting functions)
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.serif": [],
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 20
})

# %% [markdown]
# In the following notebook, we study interpretable clustering methods applicable to 
# `DBSCAN` clusterings with difficult, non-spherical clusters.

# %% [markdown]
# ## Aniso

# %%
# Anisotropicly distributed data
n_samples = 500
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

data,labels = aniso
n,d = data.shape
k = 3
true_assignment = labels_to_assignment(labels_format(labels), k)

n_core = 10
epsilon = 0.4
n_rules = k


# DBSCAN
dbscan = DBSCAN(eps=epsilon, min_samples=n_core)
dbscan.fit(data)
dbscan_labels_ = dbscan.labels_
dbscan_labels = labels_format(dbscan_labels_)
dbscan_n_clusters = len(unique_labels(dbscan_labels))
dbscan_assignment = labels_to_assignment(dbscan_labels, n_labels = dbscan_n_clusters)

# %%
fig, axs = plt.subplots(1, 1, figsize = (4,4))
axs.scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)
plt.show()

# %%
####################################################################################################
# Decision Tree
dtree = DecisionTree(
    max_leaf_nodes = n_rules
)
dtree.fit(data, dbscan_labels)
dtree_labels = dtree.predict(data)
dtree_labels_ = flatten_labels(dtree_labels)
dtree_leaf_labels = dtree.get_leaf_labels()
# This should ignore any rules which are assigned to the outlier class 
dtree_rule_assignment = labels_to_assignment(dtree_leaf_labels, n_labels = dbscan_n_clusters)
dtree_data_to_rule_assignment = dtree.get_data_to_rules_assignment(data)
dtree_data_to_cluster_assignment = dtree_data_to_rule_assignment @ dtree_rule_assignment


# ExKMC
exkmc_kmeans = KMeansBase(n_clusters = dbscan_n_clusters, random_seed = 342)
exkmc_kmeans.assign(data)
exkmc_tree = ExkmcTree(
    k = dbscan_n_clusters,
    kmeans = exkmc_kmeans.clustering,
    max_leaf_nodes = n_rules
)
exkmc_tree.fit(data, dbscan_labels)
exkmc_tree_labels = exkmc_tree.predict(data)
exkmc_tree_labels_ = flatten_labels(exkmc_tree_labels)
exkmc_tree_leaf_labels = exkmc_tree.get_leaf_labels()
exkmc_tree_rule_assignment = labels_to_assignment(exkmc_tree_leaf_labels, n_labels = dbscan_n_clusters)
exkmc_tree_data_to_rule_assignment = exkmc_tree.get_data_to_rules_assignment(data)
exkmc_tree_data_to_cluster_assignment = exkmc_tree_data_to_rule_assignment @ exkmc_tree_rule_assignment

# %%
####################################################################################################
# Mine for rules:

decision_tree_rule_miner = TreeMiner(
    tree = DecisionTree(random_state = 342),
)
decision_tree_rules, decision_tree_rule_labels = decision_tree_rule_miner.fit(
    X = data, y = dbscan_labels
)

forest_rule_miner = RandomForestMiner(
    forest_params = {
        'n_estimators': 100,
        'max_depth': None,
        'random_state': 342
    }
)
forest_rules, forest_rule_labels = forest_rule_miner.fit(data, dbscan_labels)

class_association_rule_miner = ClassAssociationRuleMiner(
    min_support = 0.01,
    min_confidence = 0.85,
    max_length = 2,
)
class_association_rules, class_association_rule_labels = class_association_rule_miner.fit(
    X = data, y = dbscan_labels
)

ensemble_rules = decision_tree_rules + forest_rules + class_association_rules
ensemble_rules = filter_rules(
    ensemble_rules, data, dbscan_labels, confidence = 0.85, support = 0.0
)

print("Total ensemble rules after filtering:", len(ensemble_rules))

# %%
####################################################################################################
# IDS
# lambdas = None triggers the full IDS procedure: IDSCoordinateAscent searches the 7-dimensional
# lambda space (ternary search per coordinate, scored by held-out AUC on a stratified train/val
# split), then the winning lambdas are used for the final random-greedy rule selection.
ids_set = IDS(
    rules = class_association_rules,
    rule_labels = class_association_rule_labels,
    n_select = n_rules,
    lambdas = None,
    lambda_search_dict = [(0.01, 1.0)] * 7,
    ternary_search_precision = 0.01,
    max_iterations = 5,
    tol = 1e-3,
    optimizer = 'random_greedy',
    random_state = 342,
)
ids_set.fit(data, dbscan_labels)
ids_set_labels = ids_set.predict(data)
ids_set_labels_ = np.array([min(labs) if len(labs) > 0 else -1 for labs in ids_set_labels])
ids_set_rule_assignment = ids_set.get_rules_to_clusters_assignment(n_labels = dbscan_n_clusters)
ids_set_data_to_rule_assignment = ids_set.get_data_to_rules_assignment(data)
ids_set_data_to_cluster_assignment = ids_set_data_to_rule_assignment @ ids_set_rule_assignment

# %%
####################################################################################################
# Partial Explanation Clustering (PEC)
pec = PEC(
    rules = ensemble_rules,
    objective_type = 'coverage-pairwise-distance',
    n_select = n_rules,
    alpha_val = 0,
)
pec.fit(data, dbscan_labels)
pec_labels = pec.predict(data)
pec_labels_ = flatten_labels(pec_labels)
# This should ignore any rules which are assigned to the outlier class, but note that we already preventing outlier rules
# in the DSCluster algorithm. This is here mostly for consistency.
pec_rule_assignment = pec.get_rules_to_clusters_assignment(n_labels = dbscan_n_clusters)
pec_data_to_rule_assignment = pec.get_data_to_rules_assignment(data)
pec_data_to_cluster_assignment = pec_data_to_rule_assignment @ pec_rule_assignment

# %%
fig,axs = plt.subplots(nrows = 1, ncols = 5, figsize = (20,3), dpi = 100)

axs[0].scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)
axs[1].scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)
axs[2].scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)
axs[3].scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)
axs[4].scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)

#axs[0].set_xlabel(r'$x_1$')
#axs[0].set_title(r"\texttt{DBSCAN}")


# Decision Tree
plot_decision_boundaries(dtree, data, color_dict, ax = axs[1], resolution = 1000, label_array = False)
#axs[1].set_xlabel(r'$x_1$')
#axs[1].set_title(r"\texttt{Decision-Tree}")


# ExKMC
plot_decision_boundaries(exkmc_tree, data, color_dict, ax = axs[2], resolution = 1000, label_array = False)
#axs[2].set_xlabel(r'$x_1$')
#axs[2].set_title(r"\texttt{ExKMC}")

# IDS
plot_rule_decision_boundaries(ids_set, data, color_dict, axs[3])
#axs[3].set_xlabel(r'$x_1$')
#axs[3].set_title(r"\texttt{IDS}")

# Partial Explanation Clustering (PEC)
plot_rule_decision_boundaries(pec, data, color_dict, axs[4])
#axs[4].set_xlabel(r'$x_1$')
#axs[4].set_title(r"\texttt{PEC}")

axs[0].set_ylabel(r'$x_2$')
axs[1].yaxis.set_visible(False)
axs[2].yaxis.set_visible(False)
axs[3].yaxis.set_visible(False)
axs[4].yaxis.set_visible(False)

axs[0].set_xticklabels([])
axs[0].set_yticklabels([])
axs[1].set_xticklabels([])
axs[2].set_xticklabels([])
axs[3].set_xticklabels([])
axs[4].set_xticklabels([])

#plt.savefig("../figures/experiments/aniso_example_pairwise.pdf", bbox_inches = 'tight', dpi = 300)

# %% [markdown]
# ## Moons

# %%
# Moons
n_samples = 500
seed = 30
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)

data,labels = noisy_moons
n,d = data.shape
k = 2
true_assignment = labels_to_assignment(labels_format(labels), k)

n_core = 10
epsilon = 0.105
n_rules = k


# DBSCAN
dbscan = DBSCAN(eps=epsilon, min_samples=n_core)
dbscan.fit(data)
dbscan_labels_ = dbscan.labels_
dbscan_labels = labels_format(dbscan_labels_)
dbscan_n_clusters = len(unique_labels(dbscan_labels))
dbscan_assignment = labels_to_assignment(dbscan_labels, n_labels = dbscan_n_clusters)

# %%
fig, axs = plt.subplots(1, 1, figsize = (4,4))
axs.scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)
plt.show()

# %%
####################################################################################################
# Decision Tree
dtree = DecisionTree(
    max_leaf_nodes = n_rules
)
dtree.fit(data, dbscan_labels)
dtree_labels = dtree.predict(data)
dtree_labels_ = flatten_labels(dtree_labels)
dtree_leaf_labels = dtree.get_leaf_labels()
# This should ignore any rules which are assigned to the outlier class 
dtree_rule_assignment = labels_to_assignment(dtree_leaf_labels, n_labels = dbscan_n_clusters)
dtree_data_to_rule_assignment = dtree.get_data_to_rules_assignment(data)
dtree_data_to_cluster_assignment = dtree_data_to_rule_assignment @ dtree_rule_assignment


# ExKMC
exkmc_kmeans = KMeansBase(n_clusters = dbscan_n_clusters, random_seed = 342)
exkmc_kmeans.assign(data)
exkmc_tree = ExkmcTree(
    k = dbscan_n_clusters,
    kmeans = exkmc_kmeans.clustering,
    max_leaf_nodes = n_rules
)
exkmc_tree.fit(data, dbscan_labels)
exkmc_tree_labels = exkmc_tree.predict(data)
exkmc_tree_labels_ = flatten_labels(exkmc_tree_labels)
exkmc_tree_leaf_labels = exkmc_tree.get_leaf_labels()
exkmc_tree_rule_assignment = labels_to_assignment(exkmc_tree_leaf_labels, n_labels = dbscan_n_clusters)
exkmc_tree_data_to_rule_assignment = exkmc_tree.get_data_to_rules_assignment(data)
exkmc_tree_data_to_cluster_assignment = exkmc_tree_data_to_rule_assignment @ exkmc_tree_rule_assignment

# %%
####################################################################################################
# Mine for rules:

decision_tree_rule_miner = TreeMiner(
    tree = DecisionTree(random_state = 342, max_leaf_nodes = None), leaf_rules = False
)
decision_tree_rules, decision_tree_rule_labels = decision_tree_rule_miner.fit(
    X = data, y = dbscan_labels
)

forest_rule_miner = RandomForestMiner(
    forest_params = {
        'n_estimators': 100,
        'max_depth': None,
        'random_state': 342,
        'max_leaf_nodes': None,
    },
    leaf_rules=False
)
forest_rules, forest_rule_labels = forest_rule_miner.fit(data, dbscan_labels)

class_association_rule_miner = ClassAssociationRuleMiner(
    min_support = 0.01,
    min_confidence = 0.85,
    max_length = 2,
)
class_association_rules, class_association_rule_labels = class_association_rule_miner.fit(
    X = data, y = dbscan_labels
)

ensemble_rules = decision_tree_rules + class_association_rules + forest_rules
ensemble_rules = filter_rules(
    ensemble_rules, data, dbscan_labels, confidence = 0.85,
)

print("Total ensemble rules after filtering:", len(ensemble_rules))

# %%
####################################################################################################
# IDS
# lambdas = None triggers the full IDS procedure: IDSCoordinateAscent searches the 7-dimensional
# lambda space (ternary search per coordinate, scored by held-out AUC on a stratified train/val
# split), then the winning lambdas are used for the final random-greedy rule selection.
ids_set = IDS(
    rules = class_association_rules,
    rule_labels = class_association_rule_labels,
    n_select = n_rules,
    lambdas = None,
    lambda_search_dict = [(0.01, 1.0)] * 7,
    ternary_search_precision = 0.01,
    max_iterations = 5,
    tol = 1e-3,
    optimizer = 'random_greedy',
    random_state = 342,
)
ids_set.fit(data, dbscan_labels)
ids_set_labels = ids_set.predict(data)
ids_set_labels_ = np.array([min(labs) if len(labs) > 0 else -1 for labs in ids_set_labels])
ids_set_rule_assignment = ids_set.get_rules_to_clusters_assignment(n_labels = dbscan_n_clusters)
ids_set_data_to_rule_assignment = ids_set.get_data_to_rules_assignment(data)
ids_set_data_to_cluster_assignment = ids_set_data_to_rule_assignment @ ids_set_rule_assignment

# %%
pec = PEC(
    rules = ensemble_rules,
    objective_type = 'coverage-pairwise-distance',
    n_select = n_rules,
    alpha_val = 0,
)
pec.fit(data, dbscan_labels)
pec_labels = pec.predict(data)
pec_labels_ = flatten_labels(pec_labels)
# This should ignore any rules which are assigned to the outlier class, but note that we already preventing outlier rules
# in the PEC algorithm. This is here mostly for consistency.
pec_rule_assignment = pec.get_rules_to_clusters_assignment(n_labels = dbscan_n_clusters)
pec_data_to_rule_assignment = pec.get_data_to_rules_assignment(data)
pec_data_to_cluster_assignment = pec_data_to_rule_assignment @ pec_rule_assignment

# %%
fig,axs = plt.subplots(nrows = 1, ncols = 5, figsize = (20,3), dpi = 100)

axs[0].scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)
axs[1].scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)
axs[2].scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)
axs[3].scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)
axs[4].scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)

#axs[0].set_xlabel(r'$x_1$')
axs[0].set_title(r"\texttt{DBSCAN}")


# Decision Tree
plot_decision_boundaries(dtree, data, color_dict, ax = axs[1], resolution = 1000, label_array = False)
#axs[1].set_xlabel(r'$x_1$')
axs[1].set_title(r"\texttt{Decision-Tree}")


# ExKMC
plot_decision_boundaries(exkmc_tree, data, color_dict, ax = axs[2], resolution = 1000, label_array = False)
#axs[2].set_xlabel(r'$x_1$')
axs[2].set_title(r"\texttt{ExKMC}")

# IDS
plot_rule_decision_boundaries(ids_set, data, color_dict, axs[3])
#axs[3].set_xlabel(r'$x_1$')
axs[3].set_title(r"\texttt{IDS}")


# Partial Explanation Clustering (PEC)
plot_rule_decision_boundaries(pec, data, color_dict, axs[4])
#axs[4].set_xlabel(r'$x_1$')
axs[4].set_title(r"\texttt{PEC}")


axs[0].set_ylabel(r'$x_2$')
axs[1].yaxis.set_visible(False)
axs[2].yaxis.set_visible(False)
axs[3].yaxis.set_visible(False)
axs[4].yaxis.set_visible(False)

axs[0].set_xticklabels([])
axs[0].set_yticklabels([])
axs[1].set_xticklabels([])
axs[2].set_xticklabels([])
axs[3].set_xticklabels([])
axs[4].set_xticklabels([])

plt.savefig("../figures/experiments/moon_example_pairwise.pdf", bbox_inches = 'tight', dpi = 300)

# %% [markdown]
# ## Aggregation

# %%
data, labels, feature_labels, scaler = load_preprocessed_aggregation("../data/synthetic")
labels = labels.astype(int) - 1
n,d = data.shape
k = len(np.unique(labels))
true_assignment = labels_to_assignment(labels_format(labels), k)

n_core = 10
epsilon = 0.25
n_rules = 5


# DBSCAN
dbscan = DBSCAN(eps=epsilon, min_samples=n_core)
dbscan.fit(data)
dbscan_labels_ = dbscan.labels_
dbscan_labels = labels_format(dbscan_labels_)
dbscan_n_clusters = len(unique_labels(dbscan_labels))
dbscan_assignment = labels_to_assignment(dbscan_labels, n_labels = dbscan_n_clusters)

# %%
fig, axs = plt.subplots(1, 1, figsize = (4,4))
axs.scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)
plt.show()

# %%
####################################################################################################
# Decision Tree
dtree = DecisionTree(
    max_leaf_nodes = n_rules
)
dtree.fit(data, dbscan_labels)
dtree_labels = dtree.predict(data)
dtree_labels_ = flatten_labels(dtree_labels)
dtree_leaf_labels = dtree.get_leaf_labels()
# This should ignore any rules which are assigned to the outlier class 
dtree_rule_assignment = labels_to_assignment(dtree_leaf_labels, n_labels = dbscan_n_clusters)
dtree_data_to_rule_assignment = dtree.get_data_to_rules_assignment(data)
dtree_data_to_cluster_assignment = dtree_data_to_rule_assignment @ dtree_rule_assignment


# ExKMC
exkmc_kmeans = KMeansBase(n_clusters = dbscan_n_clusters, random_seed = 342)
exkmc_kmeans.assign(data)
exkmc_tree = ExkmcTree(
    k = dbscan_n_clusters,
    kmeans = exkmc_kmeans.clustering,
    max_leaf_nodes = n_rules
)
exkmc_tree.fit(data, dbscan_labels)
exkmc_tree_labels = exkmc_tree.predict(data)
exkmc_tree_labels_ = flatten_labels(exkmc_tree_labels)
exkmc_tree_leaf_labels = exkmc_tree.get_leaf_labels()
exkmc_tree_rule_assignment = labels_to_assignment(exkmc_tree_leaf_labels, n_labels = dbscan_n_clusters)
exkmc_tree_data_to_rule_assignment = exkmc_tree.get_data_to_rules_assignment(data)
exkmc_tree_data_to_cluster_assignment = exkmc_tree_data_to_rule_assignment @ exkmc_tree_rule_assignment

# %%
####################################################################################################
# Mine for rules:

decision_tree_rule_miner = TreeMiner(
    tree = DecisionTree(random_state = 342, max_leaf_nodes = None), leaf_rules = False
)
decision_tree_rules, decision_tree_rule_labels = decision_tree_rule_miner.fit(
    X = data, y = dbscan_labels
)


forest_rule_miner = RandomForestMiner(
    forest_params = {
        'n_estimators': 100,
        'max_leaf_nodes': None,
        'max_depth': None,
        'random_state': 342
    },
    leaf_rules = False
)
forest_rules, forest_rule_labels = forest_rule_miner.fit(data, dbscan_labels)

class_association_rule_miner = ClassAssociationRuleMiner(
    min_support = 0.01,
    min_confidence = 0.85,
    max_length = 2,
)
class_association_rules, class_association_rule_labels = class_association_rule_miner.fit(
    X = data, y = dbscan_labels
)

ensemble_rules = decision_tree_rules + class_association_rules + forest_rules
ensemble_rules = filter_rules(
    ensemble_rules, data, dbscan_labels, confidence = 0.85,
)

print("Total ensemble rules after filtering:", len(ensemble_rules))

# %%
####################################################################################################
# IDS
# lambdas = None triggers the full IDS procedure: IDSCoordinateAscent searches the 7-dimensional
# lambda space (ternary search per coordinate, scored by held-out AUC on a stratified train/val
# split), then the winning lambdas are used for the final random-greedy rule selection.
ids_set = IDS(
    rules = class_association_rules,
    rule_labels = class_association_rule_labels,
    n_select = n_rules,
    lambdas = None,
    lambda_search_dict = [(0.01, 1.0)] * 7,
    ternary_search_precision = 0.01,
    max_iterations = 5,
    tol = 1e-3,
    optimizer = 'random_greedy',
    random_state = 342,
)
ids_set.fit(data, dbscan_labels)
ids_set_labels = ids_set.predict(data)
ids_set_labels_ = np.array([min(labs) if len(labs) > 0 else -1 for labs in ids_set_labels])
ids_set_rule_assignment = ids_set.get_rules_to_clusters_assignment(n_labels = dbscan_n_clusters)
ids_set_data_to_rule_assignment = ids_set.get_data_to_rules_assignment(data)
ids_set_data_to_cluster_assignment = ids_set_data_to_rule_assignment @ ids_set_rule_assignment

# %%
pec = PEC(
    rules = ensemble_rules,
    objective_type = 'coverage-pairwise-distance',
    n_select = n_rules,
    alpha_val = 0,
)
pec.fit(data, dbscan_labels)
pec_labels = pec.predict(data)
pec_labels_ = flatten_labels(pec_labels)
# This should ignore any rules which are assigned to the outlier class, but note that we already preventing outlier rules
# in the PEC algorithm. This is here mostly for consistency.
pec_rule_assignment = pec.get_rules_to_clusters_assignment(n_labels = dbscan_n_clusters)
pec_data_to_rule_assignment = pec.get_data_to_rules_assignment(data)
pec_data_to_cluster_assignment = pec_data_to_rule_assignment @ pec_rule_assignment  

# %%
fig,axs = plt.subplots(nrows = 1, ncols = 5, figsize = (20,3), dpi = 100)

axs[0].scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)
axs[1].scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)
axs[2].scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)
axs[3].scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)
axs[4].scatter(data[:,0], data[:,1], c = [color_dict[l] for l in dbscan_labels_], s = 20)

axs[0].set_xlabel(r'$x_1$')
#axs[0].set_title(r"\texttt{DBSCAN}")


# Decision Tree
plot_decision_boundaries(dtree, data, color_dict, ax = axs[1], resolution = 1000, label_array = False)
axs[1].set_xlabel(r'$x_1$')
#axs[1].set_title(r"\texttt{Decision-Tree}")


# ExKMC
plot_decision_boundaries(exkmc_tree, data, color_dict, ax = axs[2], resolution = 1000, label_array = False)
axs[2].set_xlabel(r'$x_1$')
#axs[2].set_title(r"\texttt{ExKMC}")

# IDS
plot_rule_decision_boundaries(ids_set, data, color_dict, axs[3])
axs[3].set_xlabel(r'$x_1$')
#axs[3].set_title(r"\texttt{IDS}")


# Partial Explanation Clustering (PEC)
plot_rule_decision_boundaries(pec, data, color_dict, axs[4])
axs[4].set_xlabel(r'$x_1$')
#axs[4].set_title(r"\texttt{PEC}")


axs[0].set_ylabel(r'$x_2$')
axs[1].yaxis.set_visible(False)
axs[2].yaxis.set_visible(False)
axs[3].yaxis.set_visible(False)
axs[4].yaxis.set_visible(False)

axs[0].set_xticklabels([])
axs[0].set_yticklabels([])
axs[1].set_xticklabels([])
axs[2].set_xticklabels([])
axs[3].set_xticklabels([])
axs[4].set_xticklabels([])

#plt.savefig("../figures/experiments/aggregation_example_pairwise.pdf", bbox_inches = 'tight', dpi = 300)
