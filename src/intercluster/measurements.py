import numpy as np
import networkx as nx
from itertools import combinations
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from typing import List
from numpy.typing import NDArray
from .utils import (
    covered_mask,
    divide_with_zeros,
    tiebreak
)


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




def point_silhouette(X : NDArray, assignment : NDArray, idx : int, cluster_idx : int) -> float:
    """
    Computes the silhouette score of a single point.

    Args:
        X (np.ndarray): Input dataset. 

        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to cluster j and False otherwise.

        idx (int): Index of point to compute the score for.

        cluster_idx (int): Index for the cluster which the point at idx belongs to
            and should be evaluated with.

    Returns:
        silhouette (float): Silhouette score.
    """
    n,d = X.shape
    n2, k = assignment.shape

    assert k > 1, "Must have at least 2 clusters."

    if n != n2:
        raise ValueError(f"Shape of data {n} does not match shape of shape of assignment {n2}.")
    
    Xi = X[idx, :]
    Ci = np.where(assignment[:,cluster_idx])[0]

    assert idx in Ci, "Point at given index is not within given cluster index."

    X_Ci = X[Ci, :]

    # Singleton cluster:
    if len(X_Ci) == 1:
        return 0

    intercluster_distance = (
        np.sum(np.linalg.norm(Xi - X_Ci, axis = 1, ord = 2)) / (len(Ci) - 1)
    )

    intracluster_distance = np.inf
    for j in range(k):
        if j != cluster_idx:
            Cj = np.where(assignment[:,j])[0]

            if len(Cj) > 0:
                X_Cj = X[Cj, :]
                intra_dist = (
                    np.sum(np.linalg.norm(Xi - X_Cj, axis = 1, ord = 2)) / len(Cj)
                )
                if intra_dist < intracluster_distance:
                    intracluster_distance = intra_dist

    score = (
        (intracluster_distance - intercluster_distance) / 
        np.max([intracluster_distance, intercluster_distance])
    )

    return score


####################################################################################################

'''
def silhouette_score(X : NDArray, assignment : NDArray) -> float:
    """
    Computes the silhouette score as the mean of scores for the entire dataset.

    Args:
        X (np.ndarray): Input dataset. 

        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to cluster j and False otherwise.

    Returns:
        silhouette (float): Silhouette score.
    """
    n,d = X.shape
    scores = []
    for idx in range(n):
        i_clusters = np.where(assignment[idx,:])[0]
        # Only compute for covered points!
        if len(i_clusters) > 0:
            cluster_total = 0
            for cluster_idx in i_clusters:
                cluster_total += point_silhouette(X, assignment, idx, cluster_idx)
            scores.append(cluster_total / len(i_clusters))

    return np.mean(scores)
'''

def silhouette_score(
        distances : NDArray,
        labels : NDArray,
        ignore : List[int] = []
    ) -> float:
    """
    Given a precomputed set of pairwise distances and a set of labels, 
    computes the silhouette score as the mean of silhouette values for each point.

    Args:
        distances (np.ndarray): n x n array of pairwise distances between points in the dataset.

        labels (np.ndarray): Length n array of labels.

        ignore (List[int], optional): List of labels to ignore in the silhouette score computation.
            Defaults to an empty list.

    Returns:
        silhouette (float): Silhouette score.
    """
    '''
    assignment = {
        label : np.where(labels == label)[0] for label in unique_labels if label not in ignore
    }

    count = 0
    silhouette = 0.0
    for i in range(n):
        i_label = labels[i]
        if i_label not in ignore:
            mean_intra = np.sum(distances[i, assignment[i_label]]) / (len(assignment[i_label]) - 1)
            min_mean_inter = np.inf
            for j in unique_labels:
                if j != i_label and j not in ignore:
                    mean_inter = np.sum(distances[i, assignment[j]]) / len(assignment[j])
                    if mean_inter < min_mean_inter:
                        min_mean_inter = mean_inter
            count += 1
            silhouette += (min_mean_inter - mean_intra) / max(min_mean_inter, mean_intra)

    return silhouette / count
    '''
    assert len(distances.shape) == 2, "Distances must be a 2D array."
    assert distances.shape[0] == distances.shape[1], "Distances must be a square matrix."
    assert len(labels) == distances.shape[0], "Labels must match the number of points in distances."

    non_ignore = np.where(~np.isin(labels, ignore))[0]
    labels_ = labels[non_ignore]
    distances_ = distances[np.ix_(non_ignore, non_ignore)]

    unique_labels = np.unique(labels_)
    if len(unique_labels) == 1 or len(non_ignore) == 0:
        # If there is only one label, the silhouette score is undefined.
        return np.nan
    else:
        # Compute the silhouette score using sklearn's implementation.
        # This requires a precomputed distance matrix.
        return sklearn_silhouette_score(X = distances_, labels = labels_, metric = 'precomputed')
            


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