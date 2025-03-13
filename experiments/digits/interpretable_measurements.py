import os
import copy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import seaborn as sns
from intercluster import *
from intercluster.rules import *
from intercluster.pruning import *
from intercluster.experiments import *

# This assumes tex is installed in your system, 
# if not you may simply remove most of this, aside from font.size:
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

# Prevents memory leakage for KMeans:
os.environ["OMP_NUM_THREADS"] = "1"

#np.seterr(all='raise')
prune_cpu_count = 1
experiment_cpu_count = 1

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 342
np.random.seed(seed)


####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_digits()
n,d = data.shape

# Parameters:
k = 10
n_clusters = k
n_rules = k
n_trees = 1000

####################################################################################################
# Run Baselines:

kmeans = KMeans(n_clusters=k, n_init="auto", random_state = None).fit(data)
kmeans_labels = labels_format(kmeans.labels_)
kmeans_assignment = labels_to_assignment(kmeans_labels, n_labels = k)
centers = kmeans.cluster_centers_


exkmc_tree = ExkmcTree(
    k = k,
    kmeans = kmeans,
    max_leaf_nodes = k,
    imm = True
)
exkmc_tree.fit(data)
exkmc_labels = exkmc_tree.predict(data, leaf_labels = False)
exkmc_assignment = labels_to_assignment(exkmc_labels, n_labels = k)
exkmc_centers = update_centers(data, kmeans.cluster_centers_, exkmc_assignment)
imm_depth = exkmc_tree.depth

####################################################################################################
# Module Parameters:
y = kmeans_labels

# Depth 2 Forest:
forest_tree_params_depth_2 = {
    'max_depth' : 2
}

forest_params_depth_2 = {
    'tree_model' : SklearnTree,
    'tree_params' : forest_tree_params_depth_2,
    'num_trees' : n_trees,
    'max_features' : d,
    'max_labels' : 1,
    'max_depths' : list(range(1, 2 + 1)),
    'feature_pairings' : [list(range(d))],
    'train_size' : 0.75
}

# Depth 3 Forest:
forest_tree_params_depth_3 = {
    'max_depth' : 3
}

forest_params_depth_3 = {
    'tree_model' : SklearnTree,
    'tree_params' : forest_tree_params_depth_3,
    'num_trees' : n_trees,
    'max_features' : d,
    'max_labels' : 1,
    'max_depths' : list(range(1, 3 + 1)),
    'feature_pairings' : [list(range(d))],
    'train_size' : 0.75
}


# Depth 4 Forest:
forest_tree_params_depth_4 = {
    'max_depth' : 4
}

forest_params_depth_4 = {
    'tree_model' : SklearnTree,
    'tree_params' : forest_tree_params_depth_4,
    'num_trees' : n_trees,
    'max_features' : d,
    'max_labels' : 1,
    'max_depths' : list(range(1, 4 + 1)),
    'feature_pairings' : [list(range(d))],
    'train_size' : 0.75
}


# Depth IMM Forest:
forest_tree_params_depth_imm = {
    'max_depth' : imm_depth
}

forest_params_depth_imm = {
    'tree_model' : SklearnTree,
    'tree_params' : forest_tree_params_depth_imm,
    'num_trees' : n_trees,
    'max_features' : d,
    'max_labels' : 1,
    'max_depths' : list(range(1, imm_depth + 1)),
    'feature_pairings' : [list(range(d))],
    'train_size' : 0.75
}


prune_objective = KmeansObjective(
    X = data,
    centers = centers,
    average = True,
    normalize = True
)

prune_params = {
    'n_rules' : n_rules,
    'n_clusters' : n_clusters,
    'X' : data,
    'y' : y,
    'objective' : prune_objective,
    'lambda_search_range' : np.linspace(0,5,51),
    'full_search' : True,
    'cpu_count' : prune_cpu_count
}


####################################################################################################

# Create and fit the decision set models
forest_depth_2 = DecisionForest(**forest_params_depth_2)
forest_depth_2.fit(data, kmeans_labels)

forest_depth_3 = DecisionForest(**forest_params_depth_3)
forest_depth_3.fit(data, kmeans_labels)

forest_depth_4 = DecisionForest(**forest_params_depth_4)
forest_depth_4.fit(data, kmeans_labels)

forest_depth_imm = DecisionForest(**forest_params_depth_imm)
forest_depth_imm.fit(data, kmeans_labels)

####################################################################################################

# Pruning 
frac_cover = 0.8
frac_remove = 1- frac_cover

prune_objective = KmeansObjective(
    X = data,
    centers = centers,
    average = True,
    normalize = True
)

prune_params = {
    'n_rules' : n_rules,
    'frac_cover' : frac_cover,
    'n_clusters' : n_clusters,
    'X' : data,
    'y' : y,
    'objective' : prune_objective,
    'lambda_search_range' : np.linspace(0,5,51),
    'full_search' : True,
    'cpu_count' : 1
}


forest_depth_2.prune(**prune_params)
if forest_depth_2.prune_status:
    forest_depth_2_prune_predictions = forest_depth_2.pruned_predict(data, rule_labels = False)
else:
    print("Forest depth 2 pruning failed.")

forest_depth_3.prune(**prune_params)
if forest_depth_3.prune_status:
    forest_depth_3_prune_predictions = forest_depth_3.pruned_predict(data, rule_labels = False)
else:
    print("Forest depth 3 pruning failed.")

forest_depth_4.prune(**prune_params)
if forest_depth_4.prune_status:
    forest_depth_4_prune_predictions = forest_depth_4.pruned_predict(data, rule_labels = False)
else:
    print("Forest depth 4 pruning failed.")

forest_depth_imm.prune(**prune_params)
if forest_depth_imm.prune_status:
    forest_depth_imm_prune_predictions = forest_depth_imm.pruned_predict(data, rule_labels = False)
else:
    print("Forest depth IMM pruning failed.")

####################################################################################################

# Create assignments:
forest_depth_2_assignment = labels_to_assignment(
    forest_depth_2_prune_predictions,
    n_labels = n_clusters
)

forest_depth_3_assignment = labels_to_assignment(
    forest_depth_3_prune_predictions,
    n_labels = n_clusters
)

forest_depth_4_assignment = labels_to_assignment(
    forest_depth_4_prune_predictions,
    n_labels = n_clusters
)

forest_depth_imm_assignment = labels_to_assignment(
    forest_depth_imm_prune_predictions,
    n_labels = n_clusters
)

outliers = outlier_mask(data, centers = centers, frac_remove=frac_remove)
non_outliers= np.where(~outliers)[0]
exkmc_outlier_assignment = copy.deepcopy(exkmc_assignment)
exkmc_outlier_assignment[outliers,:] = False

method_assignment_dict = {
    "forest_depth_2" : (forest_depth_2, forest_depth_2_assignment),
    "forest_depth_3" : (forest_depth_3, forest_depth_3_assignment),
    "forest_depth_4" : (forest_depth_4, forest_depth_4_assignment),
    "forest_depth_imm" : (forest_depth_imm, forest_depth_imm_assignment),
    "outlier" : (exkmc_tree, exkmc_outlier_assignment)
}


####################################################################################################

# Record Measurements

measurement_fns = [
    ClusteringCost(average = True, normalize = True),
    Overlap(),
    Coverage(),
]

measurement_dict = {}

for mname, (method, massign) in method_assignment_dict.items():
    for measure in measurement_fns:
        measurement_dict[(mname, measure.name)] = measure(data, massign, centers)

    if hasattr(method, "depth"):
        measurement_dict[(mname, "max-rule-length")] = method.depth
    elif hasattr(method, "max_rule_length"):
        measurement_dict[(mname,"max-rule-length")] = method.max_rule_length
    else:
        raise ValueError("No rule length attribute for the given method.")
    
    if hasattr(method, "get_weighted_average_depth"):
        if mname == 'outlier':
            measurement_dict[(mname, 'weighted-average-rule-length')] = (
                method.get_weighted_average_depth(data[non_outliers,:])
            )
        else:
            measurement_dict[(mname, 'weighted-average-rule-length')] = (
                method.get_weighted_average_depth(data)
            )
    elif hasattr(method, "get_weighted_average_rule_length"):
        measurement_dict[(mname, 'weighted-average-rule-length')] = (
            method.get_weighted_average_rule_length(data)
        )
    else:
        raise ValueError("No weighted rule length attribute for the given method.")

measurement_df = pd.DataFrame(
    [(k[0], k[1], v) for k, v in measurement_dict.items()], columns=["Row", "Column", "Value"]
)
measurement_df = measurement_df.pivot(index="Row", columns="Column", values="Value")
measurement_df.to_csv("data/experiments/digits/measurements.csv")


####################################################################################################

# Plotting 
distance_ratios = distance_ratio(data, centers)

for mname, (method, massign) in method_assignment_dict.items():
    # Single Covers:
    single_cover_mask = np.sum(massign, axis = 1) == 1
    single_cover_size = np.sum(single_cover_mask)
    single_cover_distance_ratios = distance_ratios[single_cover_mask]

    # Overlaps:
    overlap_mask = np.sum(massign, axis = 1) > 1
    overlap_size = np.sum(overlap_mask)
    overlap_distance_ratios = distance_ratios[overlap_mask]

    # Uncovereds:
    uncovered_mask = np.sum(massign, axis = 1) < 1
    uncovered_size = np.sum(uncovered_mask)
    uncovered_distance_ratios = distance_ratios[uncovered_mask]

    binwidth = 0.15
    cdict = {"Unique" : 5, "Overlapping" : 1, "Uncovered" : 7}
    fname = 'figures/digits/' + mname + '_cover_dist_.png'
    plt.figure()
    if single_cover_size > 1:
        sns.histplot(
            single_cover_distance_ratios,
            stat = 'probability',
            binwidth = binwidth,
            alpha = 1,
            label = "Unique",
            color = cmap(cdict["Unique"]),
            fill = False,
            linewidth = 6
        )
    if overlap_size > 1:
        sns.histplot(
            overlap_distance_ratios,
            stat = 'probability',
            binwidth = binwidth,
            alpha = 1,
            label = "Overlap",
            color = cmap(cdict["Overlapping"]),
            fill = False,
            linewidth = 6
        )
    if uncovered_size > 1:
        sns.histplot(
            uncovered_distance_ratios,
            stat = 'probability',
            binwidth = binwidth,
            alpha = 1,
            label = "Uncovered",
            color = cmap(cdict["Uncovered"]),
            multiple = "stack",
            fill = False,
            linewidth = 6
        )

    if mname == "forest_depth_2":
        yaxis = True
    else:
        yaxis = False
    if yaxis:
        plt.ylabel("Density")
    else:
        plt.ylabel("")
        plt.yticks([])

    xaxis = False
    if xaxis:
        plt.xlabel("Distance Ratio")
    else:
        plt.xlabel("")
        plt.xticks([])

    legend_elements = [
        mlines.Line2D(
            [], [],
            marker = 's',
            markersize=10,
            color=cmap(cdict["Unique"]),
            lw=0,
            label= "Size: " + str(single_cover_size),
            alpha=1
        ),
        mlines.Line2D(
            [], [],
            marker = 's',
            markersize=10,
            color=cmap(cdict["Overlapping"]),
            lw=0,
            label= "Size: " + str(overlap_size),
            alpha=1
        ),
        mlines.Line2D(
            [], [],
            marker = 's',
            markersize=10,
            color=cmap(cdict["Uncovered"]),
            lw=0,
            label= "Size: " + str(uncovered_size),
            alpha=1
        ),
    ]

    plt.ylim(0,0.75)
    plt.xlim(0.95,3)
    plt.legend(loc = "upper right", handles=legend_elements, ncol = 1)
    plt.savefig(fname, bbox_inches = 'tight', dpi = 300)
    #plt.show()
    plt.close()
