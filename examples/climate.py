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
#     display_name: intercluster-py3.12.4
#     language: python
#     name: intercluster-py3.12.4
# ---

# %%
import os
import copy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import seaborn as sns
import geopandas as gpd
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.objectives import *
from intercluster.decision_sets.mining import *
from intercluster.measurements import *

# Helps with KMeans memory leak issues.
os.environ["OMP_NUM_THREADS"] = "1"

# %load_ext autoreload
# %autoreload 2

# %%
####################################################################################################
# Path setup (for loading the data)

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
seed = 342
np.random.seed(seed)

# %%
# This assumes tex is installed in your system, 
# if not, you may simply remove most of this, aside from font.size 
# (although this will break certain plotting functions)
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.serif": [],
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 24
})

palette = sns.color_palette("husl", 8)
cmap = ListedColormap(palette)

color_dict = {
    -2: 'grey',
    -1: 'white',
    0: cmap(0),
    1: cmap(1),
    2: cmap(3),
    3: cmap(4),
    4: cmap(5),
    5: cmap(6),
    6: cmap(2),
    7: cmap(7)
}

subset_colors = [cmap(0), cmap(1), cmap(3), cmap(4), cmap(5), cmap(6)]
cmap_subset = ListedColormap(subset_colors)

uncovered_color = (1,1,1)
overlap_color = (0.45, 0.5, 0.5)

# %%
cmap_subset

# %% [markdown]
# # A Case Study with Climate Data
#
# Throughout the following notebook we showcase a study of interpretable clustering 
# applied to a climate dataset attributed to NOAA. This should also serve as 
# an example for how to work with the repository and create clustering models. 
#
# NOAA National Centers for Environmental information, Climate at a Glance: Regional Time Series, 
# published March 2025, retrieved on March 13, 2025 from 
# https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/regional/time-series

# %% [markdown]
# ### Load Data

# %%
# We first load and preprocess the data, the first part which is in the form of 
# geographical shape files (used for plotting maps).
filepath = '../data/climate'
shape_file_path = os.path.join(filepath, 'climate_divisions/climate_divisions.shp')
dtype_dict = {'CLIMDIV': 'str'}
gdf = gpd.read_file(shape_file_path, dtype = dtype_dict)
gdf['CLIMDIV'] = gdf['CLIMDIV'].apply(lambda x: f'{int(x):04d}')

# Next the temperature and precipitation data:
data_file_path = os.path.join(filepath, 'climate.csv')
climate_data = pd.read_csv(data_file_path, dtype={'ID': str, 'Year': str})
climate_data.set_index(['ID', 'Year'], inplace=True)

# This is grouped into historical and recent time periods, for which we take averages.
historical_years = [str(i) for i in range(1900,2000)]
recent_years = [str(i) for i in range(2013,2024)]
historical = climate_data.loc[pd.IndexSlice[:, historical_years], :]
recent = climate_data.loc[pd.IndexSlice[:, recent_years], :]
historical_avg = historical.groupby(level='ID').mean()
recent_avg = recent.groupby(level='ID').mean()

# The percent change in values is then taken to be the relative difference between averages.
climate_change = (recent_avg - historical_avg)/historical_avg
climate_change = climate_change.loc[gdf.CLIMDIV,:]
    
# Finally, we normalize the data using standard scaling.
data = climate_change.to_numpy()
feature_labels = list(climate_change.columns)
location_ids = list(climate_change.index)

scaler = StandardScaler()
data = scaler.fit_transform(data)

# %%
# The result is a dataframe in which indices represent 344 climate division locations 
# and columns represent 24 feature variables for temperature and percipitation in each of 12 months. 
pd.DataFrame(data, columns = feature_labels, index = location_ids)


# %%
# This function is used to plot the climate division map, 
# with divisions colored according to the provided labels.
def plot_individual_climate_map(labels : list[set[int]], outfile = None):
    color_label_array = np.empty(len(gdf))
    color_label_array[:] = np.nan
    for i,id in enumerate(location_ids):
        idx = gdf.loc[gdf.CLIMDIV == id].index
        if len(idx) == 1:
            l = list(labels[i])
            if len(l) > 1:
                color_label_array[idx[0]] = -2
            elif len(l) == 1:
                color_label_array[idx[0]] = l[0]
            else:
                color_label_array[idx[0]] = -1

    fig,ax = plt.subplots(figsize=(8,8))
    gdf['color'] = color_label_array
    gdf.plot(column='color', cmap=None, ax = ax, legend=False, edgecolor = 'black', alpha = 1)
    uni_labels = gdf['color'].unique()

    for l in uni_labels:
        gdf_subset = gdf[gdf['color'] == l]
        gdf_subset.plot(
            column='color',
            cmap=ListedColormap([color_dict[l]]),
            ax = ax, legend=False,
            edgecolor = 'black',
            alpha = 1
        )

    plt.xticks([])
    plt.yticks([])
    if outfile is not None:
        plt.savefig(outfile, bbox_inches = 'tight', dpi = 300)


# %% [markdown]
# ### KMeans
#
# As a reference, we compute a standard clustering. 

# %%
# To select a number of clusters, we'll simply analyze the change in cost as k is increased.
samples = 100
num_clusters_trials = np.array(range(2,20))
num_clusters_costs = np.zeros((len(num_clusters_trials), samples))

for i, nclusters in enumerate(num_clusters_trials):
    for j in range(samples):
        kmeans = KMeans(n_clusters=nclusters, n_init="auto", random_state = None).fit(data)
        kmeans_labels = labels_format(kmeans.labels_)
        kmeans_assignment = labels_to_assignment(kmeans_labels, n_labels = nclusters)
        cost = kmeans_cost(data, kmeans.cluster_centers_, kmeans_assignment)
        num_clusters_costs[i,j] = cost

# %%
x = num_clusters_trials
y = np.mean(num_clusters_costs, axis = 1)
plt.plot(
    x,
    y,
    marker = 'o', 
    linewidth = 6,
    alpha = 0.8,
    markersize = 12
)

plt.scatter(
    x[4:5],
    y[4:5],
    alpha=1.0,
    s=500,
    color='black',
)
plt.ylabel(r'SSE Cost')
plt.xlabel(r'$k$')
#plt.savefig('../figures/experiments/climate_k.pdf', bbox_inches = 'tight', dpi = 300)

# %%
# We'll say that a good clustering comes at around 6 clusters,
# and use this throughout our experiments

np.random.seed(342)
k = 6
kmeans = KMeans(n_clusters=k, n_init="auto", random_state = None).fit(data)
kmeans_labels = labels_format(kmeans.labels_)
kmeans_assignment = labels_to_assignment(kmeans_labels, n_labels = k)

# Note that to compute a normalized clustering cost (to account for coverage and overlap)
# set the average and normalize parameters to `True`
cost = kmeans_cost(
    data,
    kmeans.cluster_centers_,
    kmeans_assignment,
    average = True,
    normalize = True
)
print("Cost: " + str(cost))

# %%
plot_individual_climate_map(kmeans_labels, outfile = None)

# %% [markdown]
# ### Decision Tree

# %%
dtree = DecisionTree(
    max_leaf_nodes = k,
    random_state = seed
)

dtree.fit(data, kmeans_labels)
dtree_labels = dtree.predict(data)

# %%
plot_individual_climate_map(dtree_labels)

# %%
# NOTE: The following code is used to draw the decision tree, and save it as a figure.
# This requires installation of pygraphviz on your system! 
# (see installation instructions in the README)

#fname = '../figures/experiments/climate_dtree.png'
fname = None

# NOTE: To draw an tree with readable data, we need to give it labels to use in place of 
# features, and we'll also need to 'unscale' the data, since it was previously normalized. 
# The following function will do these things for you if the appropriate parameters are passed.
plot_tree(
    root = dtree.root,
    feature_labels = feature_labels,
    data_scaler = scaler,
    color_dict = color_dict,
    output_file = fname
)

# %% [markdown]
# ### IMM / ExKMC
# Next, we show an iterative mistake minimization decision tree clustering of the dataset. 
# This utilizes an existing implementation: https://github.com/navefr/ExKMC

# %%
exkmc_tree = ExkmcTree(
    k = k,
    kmeans = kmeans,
    max_leaf_nodes = k,
    imm = True
)

exkmc_tree.fit(data)

exkmc_labels = exkmc_tree.predict(data, leaf_labels = False)
exkmc_assignment = labels_to_assignment(exkmc_labels, n_labels = k)

# %%
plot_individual_climate_map(exkmc_labels)

# %%
# NOTE: The following code is used to draw the decision tree, and save it as a figure.
# This requires installation of pygraphviz on your system! 
# (see installation instructions in the README)


#fname = '../figures/experiments/climate_imm_tree.png'
fname = None

# NOTE: To draw an tree with readable data, we need to give it labels to use in place of 
# features, and we'll also need to 'unscale' the data, since it was previously normalized. 
# The following function will do these things for you if the appropriate parameters are passed.
plot_tree(
    root = exkmc_tree.root,
    feature_labels = feature_labels,
    data_scaler = scaler,
    color_dict = color_dict,
    output_file = fname
)

# %%
# From this we may compute a few interpretability measurements:
print("Max rule length: " + str(exkmc_tree.depth))
print("Weighted average rule length: " + str(exkmc_tree.get_weighted_average_depth(data)))

# %% [markdown]
# ## Removal Tree
#
# Next, we show an implementation for the what we call the Explanation Tree (or Removal Tree) algorithm, 
# designed by Bandyapadhyay et al. in their work "How to find a good explanation for clustering?"
# The algorithm works by removing outliers during the training process, aiming to find a minimal 
# number of points to remove so that the resulting tree exactly replicates KMeans on everything 
# that remains.

# %%
exp_tree = ExplanationTree(num_clusters = k)
exp_tree.fit(data, kmeans_labels)
exp_labels = exp_tree.predict(data, remove_outliers = False)

# %%
plot_individual_climate_map(exp_labels, outfile = None)

# %%
plot_tree(
    root = exp_tree.root,
    feature_labels = feature_labels,
    data_scaler = scaler,
    color_dict = color_dict,
    output_file = None
)

# %%
# From this we may compute a few interpretability measurements:
print("Max rule length: " + str(exp_tree.depth))
print("Weighted average rule length: " + str(exp_tree.get_weighted_average_depth(data)))

# %% [markdown]
# ## Shallow Tree

# %%
shallow_tree = ShallowTree(
    n_clusters = k,
    depth_factor = 0.03,
    kmeans_random_state = seed
)
shallow_tree.fit(data, kmeans_labels)
shallow_labels = shallow_tree.predict(data)

# %%
plot_individual_climate_map(shallow_labels, outfile = None)

# %% [markdown]
# ## PEC
# Next, we show how to create a partial interpretable clustering model,
# trained with an ensemble of rules. 

# %%
# We use the following set of fixed parameters throughout our experiments.
fixed_parameters = {
    'n': len(data),
    'd': data.shape[1],
    'n_clusters': 6,
    'n_select': 6,
    'max_rules': 12,
    'shallow_tree_depth_factor': 0.03,
    'n_forest': 100,
    'forest_max_depth': 6,
    'car_min_support': 0.025,
    'car_min_confidence': 0.85,
    'car_max_rule_length': 3, # (really means 6 by pyfim convention)
    'filter_confidence': 0.85,
    'seed': seed
}


# Distance ratio weights:
weights = distance_ratio_score(data, kmeans.cluster_centers_)


# Class association rule mining:
class_association_rule_miner = ClassAssociationRuleMiner(
    min_support = fixed_parameters['car_min_support'],
    min_confidence = fixed_parameters['car_min_confidence'],
    max_length = fixed_parameters['car_max_rule_length'],
    binning_method = "entropy",
    bin_params = {
        'random_state': seed,
    }
)
class_association_rules, class_association_rule_labels = class_association_rule_miner.fit(
    X = data, y = kmeans_labels
)


# Decision tree rule mining:
decision_tree_rule_miner = TreeMiner(
    tree = DecisionTree(random_state = seed),
)
decision_tree_rules, decision_tree_rule_labels = decision_tree_rule_miner.fit(
    X = data, y = kmeans_labels
)


# ExKMC rule mining:
exkmc_rule_miner = TreeMiner(
    tree = ExkmcTree(
        k = fixed_parameters['n_clusters'],
        kmeans = kmeans,
        imm = True
    )
)
exkmc_rules, exkmc_rule_labels = exkmc_rule_miner.fit(
    X = data, y = kmeans_labels
)


# Shallow tree rule mining:
shallow_tree_miner = TreeMiner(
    tree = ShallowTree(
        n_clusters = fixed_parameters['n_clusters'],
        depth_factor = fixed_parameters['shallow_tree_depth_factor'],
        kmeans_random_state = fixed_parameters['seed']
    )
)
shallow_rules, shallow_rule_labels = shallow_tree_miner.fit(
    X = data, y = kmeans_labels
)


# Random forest rule mining:
forest_rule_miner = RandomForestMiner(
    forest_params = {
        'criterion': 'entropy',
        'n_estimators': fixed_parameters['n_forest'],
        'random_state': fixed_parameters['seed'],
    }
)
forest_rules, forest_rule_labels = forest_rule_miner.fit(data, kmeans_labels)


# Filtering rules by confidence, and creating an ensemble of the remaining rules.
ensemble_rules = decision_tree_rules + exkmc_rules + shallow_rules + forest_rules + class_association_rules
ensemble_rules = filter_rules(
    ensemble_rules, data, kmeans_labels, confidence = fixed_parameters['filter_confidence']
)

# %%
# The following show parameterization options for PEC, for each 
# our different objectives (with the only changes being in the cost function or the coverage weights).
# These are mostly the same, aside from KMeans which also requires 
# the cluster centers and cost method to be specified as part of the objective parameters.

# NOTE: The alpha values here are set to be the same as those used in 
# other experiments for our paper, which were found through a hyperparameter sweep.
objective_dict = {
    # Mistake Cost Function:
    'coverage-mistake': {
        'objective_type': 'coverage-mistake',
        'n_select': fixed_parameters['n_select'],
        'alpha_val': 7.17,
        'selection_algorithm': 'distorted-greedy'
    },
    # KMeans Cost Function:
    'coverage-cost': {
        'objective_type': 'coverage-cost',
        'n_select': fixed_parameters['n_select'],
        'alpha_val': 103.35,
        'cluster_centers': kmeans.cluster_centers_,
        'cluster_cost_method': 'kmeans',
        'selection_algorithm': 'distorted-greedy'
    },
    # Pairwise Distance Cost Function:
    'coverage-pairwise-distance': {
        'objective_type': 'coverage-pairwise-distance',
        'n_select': fixed_parameters['n_select'],
        'alpha_val': 275.2,
        'selection_algorithm': 'distorted-greedy'
    },
    # Weighted Coverage + Mistake Cost Function:
    'coverage-mistake-weighted': {
        'objective_type': 'coverage-mistake',
        'n_select': fixed_parameters['n_select'],
        'alpha_val': 7.17,
        'weights': weights,
        'selection_algorithm': 'distorted-greedy'
    },
    # Weighted Coverage + KMeans Cost Function:
    'coverage-cost-weighted': {
        'objective_type': 'coverage-cost',
        'n_select': fixed_parameters['n_select'],
        'alpha_val': 103.35,
        'cluster_centers': kmeans.cluster_centers_,
        'cluster_cost_method': 'kmeans',
        'weights': weights,
        'selection_algorithm': 'distorted-greedy'
    },
    # Weighted Coverage + Pairwise Distance Cost Function:
    'coverage-pairwise-distance-weighted': {
        'objective_type': 'coverage-pairwise-distance',
        'n_select': fixed_parameters['n_select'],
        'alpha_val': 275.2,
        'weights': weights,
        'selection_algorithm': 'distorted-greedy'
    },
}

# %%
# Compute clustering
objective_type = 'coverage-cost-weighted'

pec = PEC(
    rules = ensemble_rules,
    **objective_dict[objective_type]
)
pec.fit(data, kmeans_labels)
lambda_val = pec.objective.lambda_val
print("lambda value: " + str(lambda_val))

pec_labels = pec.predict(data)

# %%
plot_individual_climate_map(pec_labels, outfile = None)

# %%
pec.get_weighted_average_rule_length(data)

# %%
# Plotting the decision set of pec as a list of if then rules.

#fname = '../figures/experiments/climate_pec.png'
fname = None

plot_decision_set(
    pec.decision_set,
    feature_labels = feature_labels,
    data_scaler = scaler,
    color_dict = color_dict,
    vertical = True,
    size_factor = None,
    filename = fname
)


# %% [markdown]
# ### Experiment Plots

# %%
def plot_climate_maps(
    labels,
    outfile=None,
    *,
    figsize=(8, 8),
    title_fontsize=18,
):
    """Plot climate division maps.

    Parameters
    ----------
    labels:
        Either:
        - list[set[int]]: one labeling for all locations
        - dict[str, list[set[int]]]: multiple labelings, keyed by plot title

    outfile:
        If provided, save the figure to this path.

    figsize:
        Matplotlib figsize passed directly to plt.subplots.

    title_fontsize:
        Font size for subplot titles.
    """

    def _labels_to_color_array(one_labels: list[set[int]]):
        color_label_array = np.empty(len(gdf))
        color_label_array[:] = np.nan

        for i, id in enumerate(location_ids):
            idx = gdf.loc[gdf.CLIMDIV == id].index
            if len(idx) == 1:
                l = list(one_labels[i])
                if len(l) > 1:
                    color_label_array[idx[0]] = -2
                elif len(l) == 1:
                    color_label_array[idx[0]] = l[0]
                else:
                    color_label_array[idx[0]] = -1

        return color_label_array

    # Normalize input to a dict[str, list[set[int]]]
    if isinstance(labels, dict):
        labels_dict = labels
    else:
        labels_dict = {"": labels}

    n_plots = len(labels_dict)

    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[0]

    # Work on a copy so we don't mutate gdf globally.
    base = gdf.copy()

    for ax, (name, one_labels) in zip(axes, labels_dict.items()):
        base["color"] = _labels_to_color_array(one_labels)

        # Base layer (ensures full geometry render)
        base.plot(column="color", cmap=None, ax=ax, legend=False, edgecolor="black", alpha=1)

        # Overlay each unique label with its specific color
        for l in base["color"].unique():
            subset = base[base["color"] == l]
            subset.plot(
                column="color",
                cmap=ListedColormap([color_dict[l]]),
                ax=ax,
                legend=False,
                edgecolor="black",
                alpha=1,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        if name:
            ax.set_title(rf"\texttt{{{name}}}", fontsize=title_fontsize)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight", dpi=300)

    return fig, axes


# %%
plot_dict = {
    "k-Means": kmeans_labels,
    "Decision-Tree": dtree_labels,
    "ExKMC": exkmc_labels,
    #"Shallow Tree": shallow_labels,
    "PEC": pec_labels,
}

plot_climate_maps(
    plot_dict,
    outfile='../figures/experiments/climate_maps.png',
    figsize=(20, 3),
    title_fontsize=16,
)

# %%
fig, ax = plt.subplots(figsize = (6,1))
legend_elements = [
    mlines.Line2D([], [], marker = 'o', markersize=20, color=cmap_subset(i), lw=0, label=f'Cluster {i}', alpha=0.9)
    for i in range(k)
]
ax.legend(handles=legend_elements, ncol=6)
ax.axis('off')
plt.savefig('../figures/experiments/climate_maps_legend.pdf', bbox_inches = 'tight', dpi = 300)
