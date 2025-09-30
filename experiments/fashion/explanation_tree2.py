import os
import copy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import seaborn as sns
import geopandas as gpd
from intercluster import *
from intercluster.rules import *
from intercluster.selection import *
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
experiment_cpu_count = 24

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 342
np.random.seed(seed)


####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_fashion()

import math
size = math.ceil(0.25 * len(data))
random_samples = np.sort(np.random.choice(len(data), size = size, replace = False))
data = data[random_samples, :]
data_labels = data_labels[random_samples]

n,d = data.shape

# Parameters:
k = 10

####################################################################################################
# Run Baselines:

kmeans = KMeans(n_clusters=k, n_init="auto", random_state = None).fit(data)
kmeans_labels = labels_format(kmeans.labels_)
kmeans_assignment = labels_to_assignment(kmeans_labels, n_labels = k)
centers = kmeans.cluster_centers_

####################################################################################################
# Run Explanation Tree:

exp_tree = ExplanationTree(num_clusters = k, cpu_count = experiment_cpu_count)
exp_tree.fit(data, kmeans_labels)
exp_labels = exp_tree.predict(data, remove_outliers = True)
exp_assignment = labels_to_assignment(exp_labels, n_labels = k)
exp_centers = update_centers(data, centers, exp_assignment)

outliers = list(exp_tree.outliers)
non_outliers = [i for i in range(len(data)) if i not in exp_tree.outliers]

# Record Measurements:
measurement_fns = [
    ClusteringCost(average = True, normalize = True),
    Overlap(),
    Coverage(),
]
measurement_dict = {}
for measure in measurement_fns:
    measurement_dict[measure.name] = [measure(data, exp_assignment, exp_centers)]

measurement_dict['max-rule-legnth'] = [exp_tree.depth]
measurement_dict['weighted-average-rule-length'] = [
    exp_tree.get_weighted_average_depth(data, remove_outliers = True)
]

measurement_df = pd.DataFrame(measurement_dict)
measurement_df.to_csv("data/experiments/fashion/explanation_tree_.csv")


####################################################################################################

# Plotting 
distance_ratios = distance_ratio(data, centers)

massign = exp_assignment

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
fname = 'figures/fashion/explanation_tree_cover_dist_.png'

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
        fill = False,
        linewidth = 6
    )

yaxis = False
if yaxis:
    plt.ylabel("Density")
else:
    plt.ylabel("")
    plt.yticks([])

xaxis = True
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

plt.ylim(0,0.7)
plt.xlim(0.95,3)
plt.legend(loc = "upper right", handles=legend_elements, ncol = 1)
plt.savefig(fname, bbox_inches = 'tight', dpi = 300)
plt.close()


# Save data:
np.savez_compressed("data/experiments/fashion/explanation_tree_assignment.npz", arr=exp_assignment)
np.savez_compressed("data/experiments/fashion/reference_centers.npz", arr=centers)

