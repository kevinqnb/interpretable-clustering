import math
import copy
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from collections.abc import Iterable
from typing import List, Dict, Set, Callable, Tuple, Iterator
from numpy.typing import NDArray
from .node import Node


####################################################################################################


def tiebreak(scores : NDArray, proxy : NDArray = None) -> NDArray:
    """
    Breaks ties in a length m array of scores by:
    1) (IF given) Comparing values in a same-sized proxy array.
    2) Otherwise breaking ties randomly. 

    NOTE: By default preferences are taken in ascending order.
    
    Args:
        scores (np.ndarray): Length m array of scores.
        proxy (np.ndarray, optional): Length m array of proxy values to use for tiebreaking.
            Defaults to None.
            
    Returns:
        argsort (np.ndarray): Length m argsort of scores with ties broken.
    """
    m = len(scores)
    random_tiebreakers = np.random.rand(m)
    if proxy is not None:
        return np.lexsort((random_tiebreakers, proxy, scores))
    else:
        return np.lexsort((random_tiebreakers, scores))


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


def divide_with_zeros(x : NDArray, y : NDArray) -> NDArray:
    """
    Given two arrays, divide element-wise with the convention that 0/0 = 1 and 1/0 = infty. 

    Args:
        x (np.ndarray): Numerator array.
        y (np.ndarray): Denominator array.

    Returns:
        (np.ndarray): New array with element-wise divisions. 
    """
    assert x.shape == y.shape, "Input arrays do not match in size."

    ones_mask = np.zeros(x.shape, dtype = bool)
    ones_mask = (x == 0) & (y == 0)

    infty_mask = np.zeros(x.shape, dtype = bool)
    infty_mask = (x != 0) & (y == 0)

    xcopy = copy.deepcopy(x)
    ycopy = copy.deepcopy(y)
    xcopy[ones_mask] = 1
    ycopy[ones_mask] = 1
    xcopy[infty_mask] = np.inf
    ycopy[infty_mask] = 1

    return np.divide(xcopy, ycopy)


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


def covered_mask(assignment : np.ndarray) -> NDArray:
    """
    Finds a boolean array describing data coverage.
    
    Args:
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to class j and False (0) otherwise. 
        
    Returns:
        coverage (np.ndarray): Size n array with index i being true if point i
            is covered by at least one cluster, and false otherwise.
    """
    return np.sum(assignment, axis = 1) > 0


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


def update_centers(X : NDArray, current_centers : NDArray, assignment : NDArray) -> NDArray:
    """
    Given a dataset and a current assignment to cluster centers, update the centers by finding 
    the mean of the points assigned to each original center.
    
    Args:
        X (np.ndarray): Input (n x d) dataset.
        
        current_centers (np.ndarray): Current set of cluster centers represented as a (k x d) array.
        
        assignment (np.ndarray): Boolean assignment matrix of size (n x k). Entry (i,j) is 
            `True` if point i is assigned to cluster j and `False` otherwise.
            
    Returns:
        updated_centers (np.ndarray): Size (k x d) array of updated centers.
    """
    n,d = X.shape
    k,d_ = current_centers.shape
    n_,k_ = assignment.shape
    
    assert d == d_, f"Dimensionality of data {d} and cluster centers {d_} do not match."
    assert n == n_, f"Shape of data {n} does not match shape of shape of assignment {n_}."
    assert k == k_, f"Shape of current centers {k} doesn't match shape of shape of assignment {k_}."

    updated_centers = np.zeros((k,d))
    for i in range(k):
        assigned = np.where(assignment[:,i])[0]
        if len(assigned) > 0:
            new_center = np.mean(X[assigned,:], axis = 0)
        else:
            new_center = current_centers[i,:]
            
        updated_centers[i,:] = new_center
        
    return updated_centers


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


def outlier_mask(
        X : NDArray,
        centers : NDArray,
        frac_remove : float
    ) -> NDArray:
    """
    Finds outliers to remove. Specifically, we take the convention that outliers are 
    points for which the following ratio is small. 

        (distance to second closest center / distance to closest center)

    Args:
        X (np.ndarray): Input (n x d) dataset. 

        centers (np.ndarray): Input (k x d) array of cluster centers. 

    Returns:
        outliers_mask (np.ndarray): Array of boolean values in which a True value indicates that 
            the point is an outlier and False indicates otherwise.
    """
    assert frac_remove <= 1, "Fractional size must be <= 1."
    assert frac_remove >= 0, "Fractional size must be >= 0."
    n,d = X.shape
    out_mask = np.zeros(n, dtype = bool)
    if frac_remove > 0:
        dratios = distance_ratio(X, centers)
        n_removes = int(n * frac_remove)
        outliers = np.argsort(dratios)[:n_removes]
        out_mask[outliers] = True
    return out_mask


####################################################################################################


def labels_format(labels : NDArray) -> List[Set[int]]:
    """
    Takes a 1 dimensional array of labels and forms it into a 2d label list, which is the 
    default form used in this library.
    
    Args:
        labels (np.ndarray): Length n array of labels.
    """
    return [{i} for i in labels]


####################################################################################################


def can_flatten(labels : List[Set[int]]) -> bool:
    """
    Determines if a 2d list of labels can be flattened so as to be
    exactly represented by a 1 dimensional array.
    
    Args:
        labels (List[Set[int]]): 2d list of integers where the inner list at index i 
            labels of the item with index i.
            
    Returns:
        (bool): True if the labels can be flattened and False otherwise.
    """
    label_lengths = np.array([len(labs) for labs in labels])
    return np.all(label_lengths == 1)


####################################################################################################


def flatten_labels(labels : List[Set[int]]) -> NDArray:
    """
    Given a 2d labels list, returns a flattened list of labels. 
    
    Args:
        labels (List[Set[int]]): 2d list of integers where the inner list at index i 
            labels of the item with index i.
            
    Returns:
        flattened (List[int]): Flattened list of labels.
    """
    flattened = np.array([j for _,labs in enumerate(labels) for j in labs], dtype = np.int64)
    return flattened


####################################################################################################


def unique_labels(labels : List[Set[int]]) -> Set[int]:
    """
    Given a 2d labels list, returns a set of unique labels. 
    
    Args:
        labels (List[Set[int]]): 2d list of integers where the inner list at index i 
            labels of the item with index i.
            
    Returns:
        unique_labels (Set[int]): Set of unique labels.
    """
    unique_labels = set(flatten_labels(labels))
    return unique_labels


####################################################################################################


def labels_to_assignment(
        labels : List[Set[int]],
        n_labels : int,
        ignore : Set[int] = set()
    ) -> NDArray:
    """
    Takes an input list of labels and returns its associated clustering matrix.
    NOTE: By convention, clusters are indexed [0...k-1] and items are indexed [0...n-1].
    This is how they should be labeled in the input label array.
    
    Args:
        labels (List[int] OR List[Set[int]]): List of integers where an entry at index i has value 
            j if the item associated with index i is given label j. Alternatively,
            in a soft labeling where points have multiple labels, labels[i] is a list of 
            cluster labels j.
            
        n_labels (int, optional): Total number of unique labels to create the assignment matrix 
            with. Helfpul for cases where points aren't assigned to any label (empty list).

        ignore (Set[int], optional): Set of labels to ignore in the assignment matrix. For example,
            this might be set to the set {-1} in the case that a label of -1 indicates that a point
            is not assigned to any cluster. Defaults to an empty set.

    Returns:
        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to label j and False otherwise.
    """
    assignment_matrix = np.zeros((len(labels), n_labels), dtype = bool)
    for i,labs in enumerate(labels):
        for j in labs:
            if j not in ignore:
                assignment_matrix[i, j] = True
        
    return assignment_matrix


####################################################################################################


def assignment_to_labels(assignment : NDArray) -> List[Set[int]]:
    """
    Takes an input n x k boolean assignment matrix, and outputs a list of labels for the 
    datapoints.
     
    NOTE: By convention, clusters are indexed [0...k-1] and items are indexed [0...n-1].
    This is how they will be represented in the output label array.
    
    Args:
        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to label j and False otherwise.

    Returns:
        labels (List[Set[int]]): List of integers where an entry at index i has value 
            j if the item associated with index i is present within label j. Alternatively,
            in a soft labeling where points have multiple labels, labels[i] is a list of 
            labels j.
    """
    labels = []
    for _, assign in enumerate(assignment):
        l = np.where(assign)[0]
        labels.append(set(l))
            
    return labels


####################################################################################################


def assignment_to_dict(
    assignment_matrix : NDArray
) -> Dict[int, NDArray]:
    """
    Given a 2d labels list, returns a dictionary where the keys are labels,
    and the values are the sets of indices for the inner lists which contain the unique label.
    
    Args:
        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to label j and False otherwise.
    
    Returns:
        assignment_dict (Dict[int, set[int]]): Dictionary where the keys are integers (labels) and 
            values are the sets of data point indices covered by the label.
    """        
    assignment_dict = {l: np.array([]) for l in range(assignment_matrix.shape[1])}
    for i in range(assignment_matrix.shape[1]):
        #assignment_dict[i] = set(np.where(assignment_matrix[:,i])[0])
        assignment_dict[i] = set(assignment_matrix[:,i].nonzero()[0]) 
    return assignment_dict


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


def traverse(node : Node, path : List[Node] = None) -> Iterator[List[Node]]:
    """
    Traverses a binary tree in a depth-first manner, yielding paths as as they are discovered.
    The function itself utilizes an iterative, yield from approach. For example, the following 
    command creates an iterator object over the set of all paths. Iterating through it 
    with a loop then prints all paths in a depth first manner:
    
    ```
    for path in traverse(root):
        print(path)
    ```
    
    
    Args:
        node (Node): Root node of the subtree to recurse into.
        
        path (List[Node], optional): List of node objects visited so far. Defaults to None 
            which starts a new traversal.
    
    Yields:
        path (List[Node]): List of node objects visited on the current path.
            If the path followed a left child, the corresponding string is 'left'.
            Otherwise, the string is 'right'.
    """
    if path is None:
        path = []
    
    # Yield the path up to and including the current node.
    path_update = path + [node]
    yield path_update
    
    # Yield paths with children
    if node.left_child is not None:
        left_condition_node = copy.deepcopy(node)
        left_condition_node.condition.set_direction(-1)
        left_path = path + [left_condition_node]
        yield from traverse(node.left_child, left_path)
        
    # NOTE: This creates a copy of the nodes added to the path, 
    # and depending on the direction taken in the tree, switches the node's logical condition to 
    # the correct direction. By default tree nodes will have direction -1 (<= condition), which is 
    # intended to move left if True, so switching is especially helpful for paths moving right.
    if node.right_child is not None:
        right_condition_node = copy.deepcopy(node)
        right_condition_node.condition.set_direction(1)
        right_path = path + [right_condition_node]
        yield from traverse(node.right_child, right_path)
        

####################################################################################################


def collect_nodes(root : Node) -> List[Node]:
    """
    Given the root of a tree, finds all nodes in the tree.
    
    Args:
        root (Node): Root of the tree.
    
    Returns:
        nodes (List[Node]): List of nodes in the tree. 
    """
    
    nodes = []
    for path in traverse(root):
        last_node = path[-1]
        nodes.append(last_node)
            
    return nodes


####################################################################################################


def collect_leaves(root : Node) -> List[Node]:
    """
    Given the root of a tree, finds all leaf nodes in the tree.
    
    Args:
        root (Node): Root of the tree.
    
    Returns:
        leaves (List[Node]): List of leaf nodes in the tree. 
    """
    
    leaves = []
    for path in traverse(root):
        last_node = path[-1]
        if last_node.type == 'leaf':
            leaves.append(last_node)
            
    return leaves


####################################################################################################


def get_decision_paths(
    root : Node
) -> List[List[Node]]:
    """
    Given the root of a tree, finds all decision paths 
    used to reach leaf nodes in the tree. Optionally, this takes an array y of training data labels
    AND an array of specific labels to look for. In that case, only paths with leaf nodes
    which have a majority of a label within the labels array are returned.
    
    Args:
        root (Node): Root of the tree.
    Returns:
        paths (List[List[Node]]): List of decision paths in the tree, where each decision path 
            is represented as a list of Node objects. 
    """
    paths = []
    for path in traverse(root):
        last_node = path[-1]
        if last_node.type == 'leaf':
            paths.append(path)
            
    return paths


####################################################################################################


def get_decision_paths_with_labels(
    root : Node,
    #labels : List[Set[int]],
    select_labels : NDArray,
) -> Tuple[List[List[Node]], List[Set[int]]]:
    """
    Given the root of a tree, finds all decision paths 
    used to reach leaf nodes in the tree. Optionally, this takes an input set of data labels
    AND an array of selected labels to look for. In that case, whenever a leaf 
    node is found, consider the data points associated with it. If there is a majority 
    for a selected label, keep that path. Otherwise discard it.
    
    NOTE: Perhaps this should take the trained node's class label instead...

    Args:
        root (Node): Root of the tree.
    
        labels (List[Set[int]]): Training Data labels.
            
        select_labels (np.ndarray): Labels to filter by.
    Returns:
        paths (List[List[Node]]): List of decision paths in the tree, where each decision path 
            is represented as a list of Node objects. 
        path_labels (List[Set[int]]): List of labels corresponding to each path.
    """
    paths = []
    path_labels = []
    for path in traverse(root):
        last_node = path[-1]
        if last_node.type == 'leaf' and len(last_node.indices) > 0:
            '''
            indices_labels = [labels[i] for i in last_node.indices]
            indices_labels = flatten_labels(indices_labels)
            majority_label = mode(indices_labels)
            if majority_label in select_labels:
                paths.append(path)
                path_labels.append({majority_label})
            '''
            if last_node.label in select_labels:
                paths.append(path)
                path_labels.append({last_node.label})
            
    return paths, path_labels


####################################################################################################


def get_depth(root : Node) -> int:
    """
    Given the root of a tree, finds the maximum depth of the tree.
    
    Args:
        root (Node): Root of the tree.
    
    Returns:
        depth (int): Maximum depth of the tree. 
    """
    depths = []
    for path in traverse(root):
        depths.append(len(path) - 1)    
    return max(depths)


####################################################################################################


def satisfies_conditions(X : NDArray, condition_list : List) -> NDArray:
    """
    Given a dataset X and a list of conditions, determines 
    which data indices satisfy them simultaneously.
    
    Args:
        X (np.ndarray): Dataset to evaluate.
        
        condition_list (List[Condition]): List of conditions to evaluate with.
    
    Returns:
        (np.ndarray): Integer array of data indices satisfying the decision path. 
    """
    satisfies_mask = np.zeros((X.shape[0], len(condition_list)), dtype = bool)
    for i, cond in enumerate(condition_list):
        assert (cond.direction is not None), "Condition has no associated inequality direction."
        
        satisfies_cond_mask = cond.evaluate(X)
        satisfies_mask[:,i] = satisfies_cond_mask
        
    return np.where(np.all(satisfies_mask, axis = 1))[0]


####################################################################################################
        

def satisfies_path(X : NDArray, path : List) -> NDArray:
    """
    Given a dataset X and a decision path, determines 
    which data indices satisfy the path.
    
    Args:
        X (np.ndarray): Dataset to evaluate.
        
        path (List[(Node, str)]): Decision path to evaluate.
    
    Returns:
        (np.ndarray): Integer array of data indices satisfying the decision path. 
    """
    condition_list = [node.condition for node in path[:-1]]
    return satisfies_conditions(X, condition_list)


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
