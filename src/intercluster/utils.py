import copy
import numpy as np
from typing import List, Dict, Set, Tuple, Iterator
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

'''

NOTE: I think this should be re-worked into a more general function that takes a 1d array of 
distance measurements and returns a mask of outliers based on a threshold or a fraction to remove.

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
'''


####################################################################################################


def labels_format(labels : NDArray, ignore : Set = set()) -> List[Set[int]]:
    """
    Takes a 1 dimensional array of labels and forms it into a 2d label list, which is the 
    default form used in this library.
    
    Args:
        labels (np.ndarray): Length n array of labels.

        ignore (Set[int], optional): Set of labels to ignore in the output. 
            Defaults to an empty set.
    """
    return [{i} if i not in ignore else set() for i in labels]


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


def unique_labels(labels : List[Set[int]], ignore : set = set()) -> Set[int]:
    """
    Given a 2d labels list, returns a set of unique labels. 
    
    Args:
        labels (List[Set[int]]): 2d list of integers where the inner list at index i 
            labels of the item with index i.
            
    Returns:
        unique_labels (Set[int]): Set of unique labels.
    """
    unique_labels = set(flatten_labels(labels)) - ignore
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