import copy
import numpy as np
from typing import List, Set, Tuple, Iterator
from numpy.typing import NDArray
from intercluster.utils import flatten_labels, mode
from ._node import Node


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
    labels : List[Set[int]],
    select_labels : NDArray,
) -> Tuple[List[List[Node]], List[Set[int]]]:
    """
    Given the root of a tree, finds all decision paths 
    used to reach leaf nodes in the tree. Optionally, this takes an input set of data labels
    AND an array of selected labels to look for. In that case, whenever a leaf 
    node is found, consider the data points associated with it. If there is a majority 
    for a selected label, keep that path. Otherwise discard it.
    
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
            indices_labels = [labels[i] for i in last_node.indices]
            indices_labels = flatten_labels(indices_labels)
            majority_label = mode(indices_labels)
            if majority_label in select_labels:
                paths.append(path)
                path_labels.append({majority_label})
            
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