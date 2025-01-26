import numpy as np
from collections.abc import Iterable
from typing import List, Dict, Set
from numpy.typing import NDArray


####################################################################################################


def tiebreak(scores : NDArray, proxy : NDArray = None) -> NDArray:
    """
    Breaks ties in a length m array of scores by:
    1) (IF given) Comparing values in a same-sized proxy array.
    2) Otherwise breaking ties randomly. 
    
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


def center_dists(X : NDArray, centers : NDArray, norm : int = 2) -> NDArray:
    """
    Computes the distance of each point in a dataset to a set of centers.
    
    Args:
        X (np.ndarray): (n x d) Dataset.
        
        centers (np.ndarray): (k x d) Set of representative centers.
        
        norm (int, optional): Norm to use for computing distances. 
            Takes values 1 or 2. Defaults to 2.
            
    Returns:
        distances (np.ndarray): (n x k) Distance matrix where entry (i,j) is the distance
            from point i to center j.
    """
    if norm == 2:
        diffs = X[np.newaxis, :, :] - centers[:, np.newaxis, :]
        distances = np.sum((diffs)**2, axis=-1)
    elif norm == 1:
        diffs = X[np.newaxis, :, :] - centers[:, np.newaxis, :]
        distances = np.sum(np.abs(diffs), axis=-1)
    return distances.T


####################################################################################################


def kmeans_cost(
    X : np.ndarray,
    assignment : np.ndarray,
    centers : np.ndarray,
    normalize : bool = False
) -> float:
    """
    Computes the squared L2 norm cost of a clustering with an associated set of centers

    Args:
        X (np.ndarray): (n x d) Dataset
        
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to cluster j and False (0) otherwise. 
            
        centers (np.ndarray): (k x d) Set of representative centers for each of the k clusters.
        
        normalize (bool, optional): Whether to normalize the cost by the number of points
            covered and the number of overlapping cluster assignments for each point. 
            Defaults to False.

    Returns:
        cost (float): Total cost of the clustering.
    """
        
    n,d= X.shape
    cost = 0
    covered = 0
    for i in range(n):
        i_centers = centers[assignment[i,:] == 1, :]
        if len(i_centers) > 0:
            point_cost = np.sum(np.linalg.norm(i_centers - X[i,:], axis = 1)**2)
            if normalize:
                point_cost /= len(i_centers) 
                
            cost += point_cost
            covered += 1
            
    if normalize:
        cost /= covered * d
    return cost

    
####################################################################################################


def kmedians_cost(
    X : np.ndarray,
    assignment : np.ndarray,
    centers : np.ndarray,
    normalize : bool = False
) -> float:
    """
    Computes the L1 norm cost of a clustering with an associated set of centers

    Args:
        X (np.ndarray): (n x d) Dataset
        
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to cluster j and False (0) otherwise. 
            
        centers (np.ndarray): (k x d) Set of representative centers for each of the k clusters.
        
        normalize (bool, optional): Whether to normalize the cost by the number of points
            covered and the number of overlapping cluster assignments for each point. 
            Defaults to False.

    Returns:
        cost (float): Total cost of the clustering.
    """
    n,d = X.shape
    cost = 0
    covered = 0
    for i in range(n):
        i_centers = centers[assignment[i,:] == 1, :]
        if len(i_centers) > 0:
            point_cost = np.sum(np.abs(i_centers - X[i,:]))
            if normalize:
                point_cost /= len(i_centers)
                
            cost += point_cost
            covered += 1
            
    if normalize:
        cost /= covered * d
    return cost


####################################################################################################


def overlap(
    assignment : np.ndarray
):
    """
    Computes the overlap of a point assignment. 
    
    Args:
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to class j and False (0) otherwise. 
        
    Returns:
        overlap (float): Fraction of overlapping cluster assignments for each point.
    """
    '''
    n,k = assignment.shape
    overlap = 0
    class_combinations = list(combinations(range(k), 2))
    for (i,j) in class_combinations:
        overlap += np.sum(assignment[:,i] * assignment[:,j])/n
    return overlap / len(class_combinations)
    '''
    covered_mask = np.sum(assignment, axis = 1) > 0
    return np.mean(np.sum(assignment[covered_mask, :], axis = 1))


####################################################################################################


def coverage(
    assignment : np.ndarray
):
    """
    Computes the coverage of a point assignment. 
    
    Args:
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to class j and False (0) otherwise. 
        
    Returns:
        coverage (float): Fraction of points covered by at least one cluster.
    """
    n,k = assignment.shape
    coverage = np.sum(np.sum(assignment, axis = 1) > 0) / n
    return coverage


####################################################################################################

def kmeans_plus_plus_initialization(X, k, random_seed = None):
    """
    Implements KMeans++ initialization to choose k initial cluster centers.

    Args:
    X (np.ndarray): Data points (n_samples, n_features)
    k (int): Number of clusters
    
    Returns:
    centers (np.ndarray): Initialized cluster centers (k, n_features)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    n, m = X.shape
    centers = np.empty((k, m))
    centers[:] = np.inf
    
    # Randomly choose the first center
    first_center = np.random.choice(n)
    centers[0,:] = X[first_center,:]

    for i in range(1, k):
        diffs = X[np.newaxis, :, :] - centers[:, np.newaxis, :]
        distances = np.sum(diffs ** 2, axis=-1)
        min_distances = np.min(distances, axis = 0)
        
        # Choose the next center with probability proportional to the squared distance
        probabilities = min_distances / min_distances.sum()
        next_center_index = np.random.choice(n, p=probabilities)
        centers[i,:] = X[next_center_index,:]

    return centers


####################################################################################################


def labels_to_assignment(labels, n_labels = None):
    """
    Takes an input list of labels and returns its associated clustering matrix.
    NOTE: By convention, clusters are indexed [0,k) and items are indexed [0, n).
    
    Args:
        labels (List[int] OR List[List[int]]): List of integers where an entry at index i has value 
            j if the item associated with index i is given label j. Alternatively,
            in a soft labeling where points have multiple labels, labels[i] is a list of 
            cluster labels j.
            
        n_labels (int, optional): Number of unique labels to give to points. Defaults to None,
            in which case the number of inferred from the input set of data labels.

    Returns:
        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to label j and False otherwise.
    """
    # Infer if n_labels is not provided
    if n_labels is None:
        unique_labels = set()
        for l in labels:
            if isinstance(l, (int, float, np.integer)):
                unique_labels.add(l)
            elif isinstance(l, Iterable) and not isinstance(l, (str, bytes)):
                unique_labels = unique_labels.union(set(l))
            else:
                raise ValueError("Invalid label type")         
        n_labels = len(unique_labels)
        
    assignment_matrix = np.zeros((len(labels), n_labels), dtype = bool)
    for i,j in enumerate(labels):
        if isinstance(j, (int, float, np.integer)):
            if not np.isnan(j):
                assignment_matrix[i, int(j)] = True
        elif isinstance(j, Iterable) and not isinstance(j, (str, bytes)):
            for l in j:
                assignment_matrix[i, int(l)] = True
        else:
            raise ValueError("Invalid label type")
        
    return assignment_matrix


####################################################################################################


def flatten_labels(labels : List[List[int]]) -> List[int]:
    """
    Given a 2d labels list, returns a flattened list of labels. 
    
    Args:
        labels (List[List[int]]): 2d list of integers where the inner list at index i 
            labels of the item with index i.
            
    Returns:
        flattened (List[int]): Flattened list of labels.
    """
    flattened = [j for i,labs in enumerate(labels) for j in labs]
    return flattened


####################################################################################################


def assignment_to_labels(assignment):
    """
    Takes an input n x k boolean assignment matrix, and outputs a list of labels for the 
    datapoints.
     
    NOTE: By convention, labels are indexed [0,k) and items are indexed [0, n).
    
    Args:
        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to label j and False otherwise.

    Returns:
        labels (List[List[int]]): List of integers where an entry at index i has value 
            j if the item associated with index i is present within label j. Alternatively,
            in a soft labeling where points have multiple labels, labels[i] is a list of 
            labels j.
    """
    labels = []
    for _, assign in enumerate(assignment):
        l = np.where(assign)[0]
        labels.append(list(l))
            
    return labels


####################################################################################################


def assignment_to_dict(
    assignment_matrix : NDArray
) -> Dict[int, Set[int]]:
    """
    Given a 2d labels list, returns a dictionary where the keys are the unique labels,
    and the values are the sets of indices for the inner lists which contain the unique label.
    
    Args:
        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to label j and False otherwise.
    
    Returns:
        assignment_dict (Dict[int, List[int]]): Dictionary where the keys are integers (labels) and 
            values are the sets of data point indices covered by the label.
    """        
    assignment_dict = {l: [] for l in range(assignment_matrix.shape[1])}
    for i in range(assignment_matrix.shape[1]):
        points = set(np.where(assignment_matrix[:,i])[0])
        assignment_dict[i] = points  
    return assignment_dict


####################################################################################################


def num_assigned(assignment_matrix : NDArray) -> int:
    """
    Given an assignment matrix, returns the total number of data points assigned to 
    some label.
    
    Args:
        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to label j and False otherwise.
    """
    unique_assigned = set()
    for i in range(assignment_matrix.shape[1]):
        points = set(np.where(assignment_matrix[:,i])[0])
        unique_assigned = unique_assigned.union(points)
    n_assigned = len(unique_assigned)
    return n_assigned


####################################################################################################