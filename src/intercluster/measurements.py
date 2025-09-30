import numpy as np
import networkx as nx
from itertools import combinations
from sklearn.metrics.pairwise import pairwise_distances
from typing import List, Set
from numpy.typing import NDArray
from .utils import (
    covered_mask,
    divide_with_zeros,
    tiebreak
)
import warnings


####################################################################################################


def mode(x : NDArray) -> float:
    """
    Returns the mode of a list of values.
    
    Args:
        x (np.ndarray): List of values.
    """
    unique, counts = np.unique(x, return_counts=True)
    return unique[tiebreak(counts)[-1]]


####################################################################################################


def entropy(x : NDArray) -> float:
    """
    Returns the entropy of a list of labels.
    
    Args:
        x (np.ndarray): List of labels.
    """
    unique, counts = np.unique(x, return_counts=True)
    p = counts / len(x)
    return -np.sum(p * np.log2(p))


####################################################################################################


def coverage(assignment : np.ndarray, percentage : bool = True) -> float:
    """
    Computes the coverage of a point assignment. 
    
    Args:
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to class j and False (0) otherwise. 

        percentage (bool, optional): If True, returns the coverage as a percentage of points
            covered by at least one cluster. If False, returns the total number of points covered.
        
    Returns:
        coverage (float): Fraction of points covered by at least one cluster.
    """
    n,k = assignment.shape
    #coverage = np.sum(np.sum(assignment, axis = 1) > 0) / n
    if percentage:
        coverage = np.sum(covered_mask(assignment)) / n
    else:
        coverage = np.sum(covered_mask(assignment))
    return coverage


####################################################################################################


def overlap(assignment : np.ndarray) -> float:
    """
    Computes the overlap of a point assignment, as the average of the 
    number of clusters to which points are assigned. 
    
    NOTE: The average is taken over all points which are assigned to at least one cluster.
    
    Args:
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to class j and False (0) otherwise. 
        
    Returns:
        overlap (float): Average over number of overlaps for each point.
    """
    covered_mask = np.sum(assignment, axis = 1) > 0
    if np.sum(covered_mask) > 0:
        return np.mean(np.sum(assignment[covered_mask, :], axis = 1))
    else:
        # undefined when no points are covered. 
        return np.nan


####################################################################################################


def center_dists(X : NDArray, centers : NDArray, norm : int = 2, square : bool = False) -> NDArray:
    """
    Computes the distance of each point in a dataset to a set of centers. 
    
    Args:
        X (np.ndarray): (n x d) Dataset.
        
        centers (np.ndarray): (k x d) Set of representative centers.
        
        norm (int, optional): Norm to use for computing distances. 
            Takes values 1 or 2. Defaults to 2.
            
        square (bool, optional): If using the 2 norm, optionally returns the squared distances. 
            Defaults to False which gives the standard notion of L2 distance.
            
    Returns:
        distances (np.ndarray): (n x k) Distance matrix where entry (i,j) is the distance
            from point i to center j.
    """
    n,d = X.shape
    k,d2 = centers.shape
    
    if d != d2:
        raise ValueError("Data points and centers have different dimensionality.")
    
    if (norm != 2) and (norm != 1):
        raise ValueError("Invalid norm, accepts values 1 or 2.")
    
    if norm == 2:
        diffs = X[np.newaxis, :, :] - centers[:, np.newaxis, :]
        if square:
            distances = np.sum((diffs)**2, axis=-1)
        else:
            distances = np.sqrt(np.sum(diffs**2, axis=-1))
    elif norm == 1:
        diffs = X[np.newaxis, :, :] - centers[:, np.newaxis, :]
        distances = np.sum(np.abs(diffs), axis=-1)
    return distances.T


####################################################################################################


def kmeans_cost(
    X : NDArray,
    centers : NDArray,
    assignment : NDArray,
    average : bool = False,
    normalize : bool = False,
    norm : int = 2
) -> float:
    """
    Computes the squared L2 norm cost of a clustering with an associated set of centers.

    Args:
        X (np.ndarray): (n x d) Dataset
        
        centers (np.ndarray): (k x d) Set of representative centers for each of the k clusters.
        
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to cluster j and False (0) otherwise. 
        
        average (bool, optional): Whether to average the per-point cost by the number of clusters
            that the point is assigned to. Defaults to False.
        
        normalize (bool, optional): Whether to normalize the cost by the number of points
            covered in the clustering. Defaults to False.
            
        norm (int) : Norm with which to compute the distances to cluster centers. With norm = 2,
            this produces the standard kmeans cost, whereas norm = 1 gives a kmedians cost. 
            Defaults to 2.

    Returns:
        cost (float): Total cost of the clustering.
    """
        
    n,d= X.shape
    n2,k = assignment.shape
    
    if n != n2:
        raise ValueError(f"Shape of data {n} does not match shape of shape of assignment {n2}.")
    
    center_dist_arr = center_dists(X, centers, norm = norm, square = True)
    center_dist_arr = center_dist_arr * assignment
    center_dist_sum = np.sum(center_dist_arr, axis = 1)
    n_assigns = np.sum(assignment, axis = 1)
    
    cost = None
    if average:
        point_costs = np.divide(
            center_dist_sum,
            n_assigns,
            out=np.zeros_like(center_dist_sum),
            where=n_assigns!=0
        )
        cost = np.sum(point_costs)
    else:
        cost = np.sum(center_dist_sum)
            
    if normalize:
        covered = coverage(assignment) * n
        if covered == 0:
            cost = np.inf
        else:
            cost /= covered
        
    return cost

    
####################################################################################################


def distance_ratio(X : NDArray, centers : NDArray) -> NDArray:
    """
    For each data point, computes the ratio of the distance to its second closest cluster center
    and the distance to its closest cluster center.

    Args:
        X (np.ndarray): (n x d) Dataset.
        
        centers (np.ndarray): (k x d) Set of representative centers for each of the k clusters.

    Returns:
        (np.ndarray): Length n distance ratio array.
    """
    n,d = X.shape
    center_dist_matrix = center_dists(X, centers, norm = 2, square = False)
    sorted_dist_matrix = np.argsort(center_dist_matrix, axis = 1)
    closest_dists = np.array(
        [center_dist_matrix[i, sorted_dist_matrix[i, 0]] for i in range(n)]
    )
    second_closest_dists = np.array(
        [center_dist_matrix[i, sorted_dist_matrix[i, 1]] for i in range(n)]
    )
    return divide_with_zeros(second_closest_dists, closest_dists)


####################################################################################################


def _point_silhouette_score(
        distances : NDArray,
        assignment : NDArray,
        idx : int,
        cluster_idx : int
) -> float:
    """
    Computes the silhouette score of a single point from a distance matrix and an assignment matrix.
    This is a private function, and should not be used directly. Use silhouette_score instead.
    
    Args:
        distances (np.ndarray): n x n array of pairwise distances between points in the dataset.
        
        assignment (np.ndarray: bool): n x k boolean matrix with entry (i,j) being True
            if point i belongs to cluster j and False otherwise.

        idx (int): Index of point to compute the score for.

        cluster_idx (int): Index for the cluster which the point at idx belongs to,
            and should be evaluated with.

    Returns:
        silhouette (float): Silhouette score.
    """
    n, k = assignment.shape
    Ci = np.where(assignment[:,cluster_idx])[0]
    assert idx in Ci, "Point at given index is not within given cluster."

    if len(Ci) == 1:
        # Singleton cluster, assuming 0/0 = 0:
        avg_intracluster_distance = 0
    else:
        avg_intracluster_distance = distances[idx, Ci].sum() / (len(Ci) - 1)

    avg_intercluster_distance = np.inf
    for j in range(k):
        if j != cluster_idx:
            # Every point in cluster j, excluding the point at idx, if present.
            Cj = np.where(assignment[:,j])[0]
            if len(Cj) > 0:
                inter_dist = np.inf
                if idx in Cj:
                    if len(Cj) == 1:
                        # Assuming 0/0 = 0:
                        inter_dist = 0
                    else:
                        inter_dist = distances[idx, Cj].sum() / (len(Cj) - 1) 
                else:
                    inter_dist = distances[idx, Cj].sum() / len(Cj)

                if inter_dist < avg_intercluster_distance:
                    avg_intercluster_distance = inter_dist


    if avg_intercluster_distance == avg_intracluster_distance:
        # No way to distinguish the point from at least one other cluster, return a score of 0.0.
        return 0.0
    
    elif avg_intracluster_distance == np.inf:
        # Otherwise, if the intra-cluster distance is infinite, 
        # the point must be better distinguished by some other cluster.
        return -1.0
    
    elif avg_intercluster_distance == np.inf:
        # Otherwise, if the inter-cluster distance is infinite, 
        # the point is best distinguished by its own cluster.
        return 1.0
    
    else:
        score = (
            (avg_intercluster_distance - avg_intracluster_distance) / 
            np.max([avg_intercluster_distance, avg_intracluster_distance])
        )
        return score



####################################################################################################


def silhouette_score(
        distances : NDArray,
        assignment : NDArray
) -> float:
    """
    Computes the silhouette score from a distance matrix and an point to cluster assignment matrix.

    Args:
        distances (np.ndarray): n x n array of pairwise distances between points in the dataset.
        
        assignment (np.ndarray: bool): n x k boolean matrix with entry (i,j) being True
            if point i belongs to cluster j and False otherwise.
    
    Returns:
        silhouette (float): Silhouette score.
    """
    n,n_ = distances.shape
    if n != n_:
        raise ValueError("Distance matrix must be square.")
    
    if np.any(distances) < 0:
        raise ValueError("Distance matrix must only contain non-negative values.")
    
    n_, k = assignment.shape
    if n != n_:
        raise ValueError(f"Shape of data {n} does not match shape of shape of assignment {n_}.")

    if k < 2 or (np.sum(assignment, axis = 0) > 0).sum() < 2:
        warnings.warn("Silhoutte score is only defined for instances with two or more clusters.")
        return np.nan
    
    
    covered = coverage(assignment, percentage = False)
    score_sum = 0.0
    for i in range(k):
        cluster_points = np.where(assignment[:, i])[0]
        for j in cluster_points:
            score = _point_silhouette_score(distances, assignment, j, i)
            score_sum += score / np.sum(assignment[j,:])

    return score_sum / covered


####################################################################################################



def mutual_reachability_distance(X : NDArray, n_core : int = 1) -> NDArray:
    """
    For each pair of points (i,j) in a dataset, computes the distance between them as the maximum of 
    their {euclidean distance, the euclidean distance to the mu-th nearest neighbor of i, and 
    the euclidean distance to the mu-th nearest neighbor of j}. In the context of DBSCAN, this is 
    a value of epsilon at which the points i and j must belong to the same cluster (but not yet the 
    minimum most).

    Args:
        X (np.ndarray): (n x d) Dataset.
        
        n_core (int, optional): Number of nearest neighbors to consider for a core point.
            Defaults to 1.

    Returns:
        distances (np.ndarray): n x n array of mutual reachability distances.
    """
    n, d = X.shape
    distances = np.zeros((n, n))
    euclidean_distances = pairwise_distances(X, metric='euclidean')

    # mu-th nearest neighbor distance for each point
    euclidean_distance_sorted = np.sort(euclidean_distances, axis=1)
    mu_distance = euclidean_distance_sorted[:, n_core - 1]

    for i in range(n):
        for j in range(n):
            if i == j:
                distances[i, j] = 0
            else:
                mutual_distance = max(
                    euclidean_distances[i, j],
                    mu_distance[i],
                    mu_distance[j]
                )
                distances[i, j] = mutual_distance

    return distances


####################################################################################################


def density_distance(X : NDArray, n_core : int = 1) -> NDArray:
    """
    Computes the density distance between each pair of points in a dataset. The density distance 
    between points i and j is the computed as the largest edge weight on the path 
    between them in the minimum spanning tree of the graph where edge weights are given 
    by euclidean distances. 

    This is an implemenation of work seen in:
    "Unsupervised representation learning with Minimax distance measures"
    by Morteza Haghir Chehreghani 2020, https://arxiv.org/abs/1904.13223
    
    Args:
        X (np.ndarray): (n x d) Dataset.

        n_core (int, optional): Number of nearest neighbors to consider for a core point.
            Defaults to 1.
        
    Returns:
        distances (np.ndarray): n x n array of density distances.
    """
    n, d = X.shape
    density_distances = np.zeros((n,n))
    reachability_distances = mutual_reachability_distance(X, n_core)

    # Create a graph from the distance matrix
    G = nx.from_numpy_array(reachability_distances)
    T = nx.minimum_spanning_tree(G)

    # Extract the edges of the minimum spanning tree, sorted by weight
    sorted_mst_edges = sorted(T.edges(data=True), key=lambda x: x[2]['weight'])

    # Compute minimax path distances by dynamic programming
    component_list = [{i} for i in range(n)]
    component_id = list(range(n))
    current_id = n - 1
    for u, v, data in sorted_mst_edges:
        if component_id[u] != component_id[v]:
            first_side = component_list[component_id[u]]
            second_side = component_list[component_id[v]]
            new_component = first_side.union(second_side)

            component_list.append(new_component)
            current_id += 1
            component_id.append(current_id)
            for i in new_component:
                component_id[i] = current_id

            weight = data['weight']
            for i in first_side:
                for j in second_side:
                    density_distances[i, j] = weight
                    density_distances[j, i] = weight

    return density_distances


####################################################################################################


def pairwise_distance_threshold(D : NDArray, indices : NDArray, threshold : float) -> bool:
    """
    Given a distance matrix D and a subset of indices, determines if each pair of points 
    within indices satisfies the distance threshold.
    
    Args:
        D (np.ndarray): Distance matrix of size n x n.
        
        indices (np.ndarray): Indices of points to consider.
        
        threshold (float): Distance threshold to satisfy.
    
    Returns:
        (bool) : True if all pairs satisfy the distance threshold, False otherwise.
    """
    if D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")
    
    if np.any(D < 0):
        raise ValueError("Distance matrix must only contain non-negative values.")
    
    if len(indices) == 0 or len(indices) == 1:
        # If no indices or only one index, trivially satisfied.
        return True
    
    if len(indices) > D.shape[0]:
        raise ValueError("Indices length exceeds distance matrix size.")
    
    if threshold < 0:
        raise ValueError("Threshold must be non-negative.")
    
    return np.all(D[np.ix_(indices, indices)] <= threshold)


####################################################################################################


def max_intra_cluster_distance(
        distances : NDArray,
        assignment : NDArray
    ) -> float:
    """
    Computes the maximum intra-cluster distance for a given assignment of points to clusters.
    Args:
        distances (np.ndarray): n x n array of pairwise distances between points in the dataset.
        
        assignment (np.ndarray: bool): n x k boolean matrix with entry (i,j) being True
            if point i belongs to cluster j and False otherwise.
    Returns:
        max_dist (float): Maximum intra-cluster distance.
    """
    max_dist = 0
    for i in range(assignment.shape[1]):
        i_pts = np.where(assignment[:,i] == 1)[0]
        if len(i_pts) > 0:
            distance_sub = distances[np.ix_(i_pts, i_pts)]
            sub_max = np.max(distance_sub)
            if np.max(distance_sub) > max_dist:
                max_dist = sub_max

    return max_dist

    
####################################################################################################
    

def min_inter_cluster_distance(
        distances : NDArray,
        assignment : NDArray
    ) -> float:
    """
    Computes the minimum inter-cluster distance for a given assignment of points to clusters.

    Args:
        distances (np.ndarray): n x n array of pairwise distances between points in the dataset.
        
        assignment (np.ndarray: bool): n x k boolean matrix with entry (i,j) being True
            if point i belongs to cluster j and False otherwise.
    Returns:
        min_dist (float): Minimum inter-cluster distance.
    """
    n,k = assignment.shape
    min_dist = np.inf
    for (i,j) in combinations(list(range(k)), 2):
        i_pts = np.where(assignment[:,i] == 1)[0]
        j_pts = np.where(assignment[:,j] == 1)[0]

        if len(i_pts) > 0 and len(j_pts) > 0:
            distance_sub = distances[np.ix_(i_pts, j_pts)]
            np.fill_diagonal(distance_sub, np.inf)
            new_min = np.min(distance_sub)
            if new_min < min_dist:
                min_dist = new_min

    return min_dist


####################################################################################################

def mistakes(
    ground_truth_assignment : NDArray,
    data_to_rule_assignment : NDArray,
    rule_to_cluster_assignment : NDArray
) -> int:
    """
    Computes the number of mistakes made by a given point to cluster assignment, with respect to a ground truth.
    """
    n, k = ground_truth_assignment.shape
    m = rule_to_cluster_assignment.shape[0]
    assert rule_to_cluster_assignment.shape[1] == k, \
        "The number of clusters in the rule assignment must match the data assignment."
    assert np.all(np.sum(rule_to_cluster_assignment, axis=1) <= 1), \
        "Each rule must belong to exactly one cluster."
    assert data_to_rule_assignment.shape == (n, m), \
        ("The data to rules assignment must have shape (n, m) where n is the number of data "
        "points and m is the number of rules.")
    
    mistakes = 0
    for i, rule_points in enumerate(data_to_rule_assignment.T):
        rule_clusters = np.where(rule_to_cluster_assignment[i])[0]
        if len(rule_clusters) > 0:
            cluster_points = ground_truth_assignment[:, rule_clusters[0]]
            mistakes += np.sum(rule_points & ~cluster_points)
            
    return mistakes


####################################################################################################


def coverage_mistake_score(
        lambda_val : float,
        ground_truth_assignment : NDArray,
        data_to_rule_assignment : NDArray,
        rule_to_cluster_assignment : NDArray
):
    """
    Sum of covered points minus lambda times the number of mistakes made by a given 
    point to cluster assignment, with respect to a ground truth.

    Args:
        lambda_val (float): Weighting factor for the mistakes term in the objective function.
        ground_truth_assignment (np.ndarray: bool): n x k boolean (or binary) matrix 
            with entry (i,j) being True (1) if point i belongs to cluster j and False (0) 
            otherwise. This should correspond to a ground truth labeling of the data. 
        rule_to_cluster_assignment (np.ndarray: bool): m x k boolean (or binary) matrix 
            with entry (i,j) being True (1) if rule i belongs to cluster j and False (0) 
            otherwise. NOTE: each rule must belong to exactly one cluster.
        data_to_rules_assignment (np.ndarray: bool): n x m boolean (or binary) matrix 
            with entry (i,j)  being True (1) if point i satisfies rule j and False (0) 
            otherwise.
    """
    n, k = ground_truth_assignment.shape
    m = rule_to_cluster_assignment.shape[0]
    assert rule_to_cluster_assignment.shape[1] == k, \
        "The number of clusters in the rule assignment must match the data assignment."
    assert np.all(np.sum(rule_to_cluster_assignment, axis=1) <= 1), \
        "Each rule must belong to exactly one cluster."
    assert data_to_rule_assignment.shape == (n, m), \
        ("The data to rules assignment must have shape (n, m) where n is the number of data "
        "points and m is the number of rules.")
    
    cover = coverage(data_to_rule_assignment, percentage = False)
    
    mistakes = 0
    for i, rule_points in enumerate(data_to_rule_assignment.T):
        rule_clusters = np.where(rule_to_cluster_assignment[i])[0]
        if len(rule_clusters) > 0:
            cluster_points = ground_truth_assignment[:, rule_clusters[0]]
            mistakes += np.sum(rule_points & ~cluster_points)

    return cover - lambda_val * mistakes


####################################################################################################


def uncovered_mistake_score(
        ground_truth_assignment : NDArray,
        data_to_rule_assignment : NDArray,
        rule_to_cluster_assignment : NDArray
):
    """
    Sum of uncovered points and the number of mistakes made by a given 
    point to cluster assignment, with respect to a ground truth.

    Args:
        ground_truth_assignment (np.ndarray: bool): n x k boolean (or binary) matrix 
            with entry (i,j) being True (1) if point i belongs to cluster j and False (0) 
            otherwise. This should correspond to a ground truth labeling of the data. 
        rule_to_cluster_assignment (np.ndarray: bool): m x k boolean (or binary) matrix 
            with entry (i,j) being True (1) if rule i belongs to cluster j and False (0) 
            otherwise. NOTE: each rule must belong to exactly one cluster.
        data_to_rules_assignment (np.ndarray: bool): n x m boolean (or binary) matrix 
            with entry (i,j)  being True (1) if point i satisfies rule j and False (0) 
            otherwise.
    """
    n, k = ground_truth_assignment.shape
    m = rule_to_cluster_assignment.shape[0]
    assert rule_to_cluster_assignment.shape[1] == k, \
        "The number of clusters in the rule assignment must match the data assignment."
    assert np.all(np.sum(rule_to_cluster_assignment, axis=1) <= 1), \
        "Each rule must belong to exactly one cluster."
    assert data_to_rule_assignment.shape == (n, m), \
        ("The data to rules assignment must have shape (n, m) where n is the number of data "
        "points and m is the number of rules.")
    
    uncovered = n - coverage(data_to_rule_assignment, percentage = False)
    
    mistakes = 0
    for i, rule_points in enumerate(data_to_rule_assignment.T):
        rule_clusters = np.where(rule_to_cluster_assignment[i])[0]
        if len(rule_clusters) > 0:
            cluster_points = ground_truth_assignment[:, rule_clusters[0]]
            mistakes += np.sum(rule_points & ~cluster_points)

    return uncovered + mistakes


####################################################################################################


def label_differences(
        true_labels : list[Set[int]],
        pred_labels : list[Set[int]],
        percentage : bool = False,
        ignore : Set[int] = None
) -> int:
    """
    Computes the number of points which are assigned differently in two labelings.
    This is a helper function for computing the robustness of a clustering.

    Args:
        true_labels (np.ndarray): Length n array of ground truth labels.
        
        pred_labels (np.ndarray): Length n array of predicted labels.

        percentage (bool, optional): If True, returns the fraction of points which differ
            between the two labelings. If False, returns the total number of points which differ.
            Defaults to False.

    Returns:
        differences (int): Number of points assigned to different clusters in the two labelings.
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("Label arrays must have the same length.")
    
    differences = 0
    for i in range(len(true_labels)):
        if ignore is not None and (true_labels[i] == ignore or pred_labels[i] == ignore):
            continue
        else:
            if true_labels[i] != pred_labels[i]:
                differences += 1

    if percentage:
        if ignore is None:
            return differences / len(true_labels)
        else:
            return differences / len([l for l in true_labels if l not in ignore])
    else:
        return differences
    

####################################################################################################


def clustering_distance(
    labels1 : List[Set[int]],
    labels2 : List[Set[int]],
    percentage : bool = False,
    ignore : Set[int] = None
) -> float:
    """
    Computes the distance between two clusterings as the number (or percentage) of pairs which are 
    assigned to the same cluster in one clustering and different clusters in the other.

    Args:
        labels1 (list[set[int]]): Length n list of ground truth labels.
        
        labels2 (list[set[int]]): Length n list of predicted labels.

        percentage (bool, optional): If True, returns the fraction of pairs which differ
            between the two labelings. If False, returns the total number. Defaults to False.

        ignore (set[int], optional): If provided, ignores points with this label in both
            labelings. Defaults to None.

    Returns:
        distance (float): Number (or percentage) of pairs assigned to different clusters 
            in the two labelings.
    """
    n = len(labels1)
    if n != len(labels2):
        raise ValueError("Label arrays must have the same length.")
    
    non_outliers = [
        i for i in range(n) if (ignore is None or (labels1[i] != ignore and labels2[i] != ignore))
    ]
    n = len(non_outliers)
    if n < 2:
        raise ValueError("Not enough non-outlier points to compute clustering distance.")
    
    n_pairs = n * (n - 1) / 2
    
    differences = 0
    for (i,j) in combinations(non_outliers, 2):
        same1 = True if (labels1[i] & labels1[j]) else False
        same2 = True if (labels2[i] & labels2[j]) else False
        if same1 != same2:
            differences += 1

    if percentage:
        return differences / n_pairs
    else:
        return differences