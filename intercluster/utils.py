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


def coverage(assignment : np.ndarray) -> float:
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


def update_centers(X : NDArray, assignment : NDArray) -> NDArray:
    """
    Given a dataset and a current assignment to cluster centers, update the centers by finding 
    the mean of the points assigned to each original center.
    
    Args:
        X (np.ndarray): Input (n x d) dataset.
        
        assignment (np.ndarray): Boolean assignment matrix of size (n x k). Entry (i,j) is 
            `True` if point i is assigned to cluster j and `False` otherwise.
            
    Returns:
        updated_centers (np.ndarray): Size (k x d) array of updated centers.
    """
    n,d = X.shape
    n2,k = assignment.shape
    
    if n != n2:
        raise ValueError(f"Shape of data {n} does not match shape of shape of assignment {n2}.")

    updated_centers = np.zeros((k,d))
    for i in range(k):
        assigned = np.where(assignment[:,i])[0]
        new_center = np.mean(X[assigned,:], axis = 0)
        updated_centers[i,:] = new_center
        
    return updated_centers


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


def can_flatten(labels : List[Set[int]]) -> NDArray:
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
    flattened = np.array([j for _,labs in enumerate(labels) for j in labs])
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


def labels_to_assignment(labels : List[Set[int]], n_labels):
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

    Returns:
        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to label j and False otherwise.
    """
    '''
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
    '''    
    assignment_matrix = np.zeros((len(labels), n_labels), dtype = bool)
    for i,labs in enumerate(labels):
        for j in labs:
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
        assignment_dict[i] = set(np.where(assignment_matrix[:,i])[0])  
    return assignment_dict


####################################################################################################