import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray
from intercluster.utils import flatten_labels, mode
from ._node import Node


####################################################################################################


def traverse(node : Node, path : List[Tuple[Node, str]] = None):
    """
    Traverses a binary tree in a depth-first manner, yielding nodes as they are visited.
    
    Args:
        node (Node): Root node of the subtree to recurse into.
        
        path (List[(Node, str)], optional): List of node objects visited so far.
    
    Yields:
        path (List[(Node, str)]): List of node objects visited on the current path.
            If the path followed a left child, the corresponding string is 'left'.
            Otherwise, the string is 'right'.
    """
    if path is None:
        path = []
    
    path_update = path + [(node, None)]
    
    yield path_update
    
    if node.left_child is not None:
        left_path = path + [(node, 'left')]
        yield from traverse(node.left_child, left_path)
    if node.right_child is not None:
        right_path = path + [(node, 'right')]
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
        last_node = path[-1][0]
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
        last_node = path[-1][0]
        if last_node.type == 'leaf':
            leaves.append(last_node)
            
    return leaves


####################################################################################################


def get_decision_paths(
    root : Node
) -> List[List[Tuple[Node, str]]]:
    """
    Given the root of a tree, finds all decision paths 
    used to reach leaf nodes in the tree. Optionally, this takes an array y of training data labels
    AND an array of specific labels to look for. In that case, only paths with leaf nodes
    which have a majority of a label within the labels array are returned.
    
    Args:
        root (Node): Root of the tree.
    Returns:
        paths (List[(Node, str)]): List of decision paths in the tree. 
    """
    paths = []
    for path in traverse(root):
        last_node = path[-1][0]
        if last_node.type == 'leaf':
            paths.append(path)
            
    return paths


####################################################################################################


def get_decision_paths_with_labels(
    root : Node,
    labels : List[List[int]],
    select_labels : NDArray,
) -> Tuple[List[List[Tuple[Node, str]]], List[List[int]]]:
    """
    Given the root of a tree, finds all decision paths 
    used to reach leaf nodes in the tree. Optionally, this takes an input set of data labels
    AND an array of selected labels to look for. In that case, whenever a leaf 
    node is found, consider the data points associated with it. If there is a majority 
    for a selected label, keep that path. Otherwise discard it.
    
    Args:
        root (Node): Root of the tree.
    
        y (np.ndarray, optional): Training Data labels.
            
        labels (np.ndarray, optional): Labels to filter by.
    Returns:
        paths (List[(Node, str)]): List of decision paths in the tree. 
        path_labels (List[List[int]]): List of labels corresponding to each path.
    """
    paths = []
    path_labels = []
    for path in traverse(root):
        last_node = path[-1][0]
        if last_node.type == 'leaf' and len(last_node.indices) > 0:
            indices_labels = [labels[i] for i in last_node.indices]
            indices_labels = flatten_labels(indices_labels)
            majority_label = mode(indices_labels)
            if majority_label in select_labels:
                paths.append(path)
                path_labels.append([majority_label])
            
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
    '''
    weights = np.zeros((X.shape[1], len(path) - 1))
    thresholds = np.zeros(len(path) - 1)
    directions = np.ones(len(path) - 1)
    
    for i, (node, direction) in enumerate(path[:-1]):
        weights[node.features, i] = node.weights
        thresholds[i] = node.threshold
        if direction == 'left':
            directions[i] = -1

    results = np.sign(np.dot(X, weights) - thresholds)
    results[results == 0] = -1
    satisfies_mask = np.all(results == directions, axis=1)
    return np.where(satisfies_mask)[0]
    '''


####################################################################################################