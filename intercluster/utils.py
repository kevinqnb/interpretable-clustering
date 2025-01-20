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
    normalize : bool = False,
    square : bool = False
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
        if square:
            cost /= covered **2
        else:
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


def labels_to_assignment(labels, k = None):
    """
    Takes an input list of labels and returns its associated clustering matrix.
    NOTE: By convention, clusters are indexed [0,k) and items are indexed [0, n).
    
    Args:
        labels (List[int] OR List[List[int]]): List of integers where an entry at index i has value 
            j if the item associated with index i is present within cluster j. Alternatively,
            in a soft clustering where points have multiple labels, labels[i] is a list of 
            cluster labels j.
            
        k (int, optional): Number of clusters. Defaults to None, in which case the number of
            clusters is inferred from the input labels.

    Returns:
        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to cluster j and False otherwise.
    """
    # Infer if k is not provided
    if k is None:
        s = -1
        for l in labels:
            if np.max(l) > s:
                s = np.max(l) 
        k = int(s) + 1
        
    assignment_matrix = np.zeros((len(labels), k), dtype = bool)
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


def assignment_to_labels(assignment):
    """
    Takes an input n x k boolean assignment matrix, and outputs a list of labels for the 
    datapoints.
     
    NOTE: By convention, clusters are indexed [0,k) and items are indexed [0, n).
    
    Args:
        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to cluster j and False otherwise.

    Returns:
        labels (List[List[int]]): List of integers where an entry at index i has value 
            j if the item associated with index i is present within cluster j. Alternatively,
            in a soft clustering where points have multiple labels, labels[i] is a list of 
            cluster labels j.
    """
    labels = []
    for _, assign in enumerate(assignment):
        l = np.where(assign)[0]
        labels.append(list(l))
            
    return labels


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


def label_covers_dict(
    labels : List[List[int]],
    unique_labels : NDArray = None
) -> Dict[int, Set[int]]:
    """
    Given a 2d labels list, returns a dictionary where the keys are the unique labels,
    and the values are the sets of indices for the inner lists which contain the unique label.
    
    Args:
        labels (List[List[int]]): 2d list of integers where the inner list at index i 
            labels of the item with index i.
            
        unique_labels (np.ndarray): Array of unique labels that should form the keys of 
            the label covers dictionary. Useful if certain labels do not show up in 
            the labels list. If None, the unique labels are inferred from the input.
    
    Returns:
        covers_dict (Dict[int, Set[int]]): Dictionary where the keys are integers (labels) and 
            values are the sets of data point indices covered by the label.
    """
    if unique_labels is None:
        unique_labels = np.unique(flatten_labels(labels))
        
    covers_dict = {l: set() for l in unique_labels}
    for i, labs in enumerate(labels):
        for l in labs:
            covers_dict[l].add(i)
            
    return covers_dict


####################################################################################################

def traverse(node, path=None):
    """
    Traverses a binary tree in a depth-first manner, yielding nodes as they are visited.
    
    Args:
        node (Node): Root node of the tree.
    
    Yields:
        path (List[(Node, str)]): List of node objects visited on the current path.
            If the path followed a left child, the corresponding string is 'left'.
            Otherwise, the string is 'right'.
    """
    if path is None:
        path = []
    
    path_update = path + [node]
    
    yield path_update
    
    if node.left_child is not None:
        left_path = path + [(node, 'left')]
        yield from traverse(node.left_child, left_path)
    if node.right_child is not None:
        right_path = path + [(node, 'right')]
        yield from traverse(node.right_child, right_path)

####################################################################################################

def find_leaves(root):
    """
    Given the root of a tree, finds all paths to leaf nodes in the tree.
    
    Args:
        root (Node): Root of the tree.
    
    Returns:
        leaves (dict[int: List[Node]]): Dictionary where the key is the label of the leaf node,
            and the value is a list of nodes along the path from root to leaf (inclusive). 
    """
    
    leaves = {}
    
    for path in traverse(root):
        last_node = path[-1]
        if last_node.type == 'leaf':
            leaves[last_node.label] = path
            
    return leaves


####################################################################################################


def rule_grid(X, g):
    """
    Builds a g x g grid of rules around a given dataset. 

    Args:
        data (np.ndarray): Input (n x m) dataset. 
        g (int): _description_

    Returns:
        grid (List[Rule]): List of Rule objects describing the grid.
    """
    # Step 1: Find the min and max of the dataset along both dimensions
    x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
    y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])

    # Step 2: Calculate the step size for both dimensions
    x_step = (x_max - x_min) / g
    y_step = (y_max - y_min) / g

    # Step 3: Create the grid cells with logical conditions
    grid_cells = []

    for i in range(g):
        for j in range(g):
            x_start = x_min + i * x_step
            x_end = x_start + x_step
            y_start = y_min + j * y_step
            y_end = y_start + y_step
            
            # Logical conditions defining the current cell
            ineq = ['>', '<', '>', '<']
            if i == 0:
                ineq[0] = '>='
            elif i == g - 1:
                ineq[1] = '<='
            if j == 0:
                ineq[2] = '>='
            elif j == g - 1:
                ineq[3] = '<='
            
            cell_conditions = [
                Condition(0, ineq[0], x_start),
                Condition(0, ineq[1], x_end),
                Condition(1, ineq[2], y_start),
                Condition(1, ineq[3], y_end)
            ]
            
            grid_cells.append(Rule([Term(cell_conditions)]))

    return grid_cells

####################################################################################################

def remove_rows_cols(matrix, indices):
    """
    Given a data matrix, and a list of integer valued indices,
    removes the rows and columns corresponding to the indices.

    Args:
        matrix (np.ndarray): (n x m) Dataset in the form of a numpy array.
        indices (_type_): List of indices where each index i corresponds to both a row 
            and a column to be removed from the matrix.

    Returns:
        (np.ndarray): Modified dataset with rows/columns removed.
    """
    
    indices = sorted(indices, reverse=True)
    for idx in indices:
        # Remove the specified row
        matrix = np.delete(matrix, idx, axis=0)
        # Remove the specified column
        matrix = np.delete(matrix, idx, axis=1)
    return matrix

####################################################################################################

def add_row_col(matrix, new_row, new_col):
    """
    Given a data matrix, add a new row and a new column.
    The new column/row is added as the last column/row in the matrix.

    Args:
        matrix (np.ndarray): (n x m) Dataset in the form of a numpy array.
        new_row (np.ndarray): Length m array to be added as a new row.
        new_col (np.ndarray): Length n + 1 array to be added as a new column.

    Returns:
        (np.ndarray): Modified (n + 1 x m + 1) dataset with rows/columns added.
    """
    # Ensure the new row is a 1D array of the correct length
    new_row = np.array(new_row)
    assert new_row.shape[0] == matrix.shape[1], "New row length must match number of columns in the matrix"
    
    # Append the new row to the matrix
    matrix = np.vstack([matrix, new_row])
    
    # Ensure the new column is a 1D array of the correct length
    new_col = np.array(new_col)
    assert new_col.shape[0] == matrix.shape[0], "New column length must match number of rows in the updated matrix"
    
    # Append the new column to the matrix
    matrix = np.hstack([matrix, new_col.reshape(-1, 1)])
    
    return matrix
    
####################################################################################################


def mode(x : NDArray) -> float:
    """
    Returns the mode of a list of values.
    
    Args:
        x (np.ndarray): List of values.
    """
    unique, counts = np.unique(x, return_counts=True)
    return unique[np.argmax(counts)]


####################################################################################################