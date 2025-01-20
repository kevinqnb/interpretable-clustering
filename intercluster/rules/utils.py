import numpy as np
from typing import List, Any, Tuple
import numpy.typing as npt
from ._node import Node
from ..utils import mode


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
    y : npt.NDArray,
    labels : npt.NDArray,
) -> Tuple[List[List[Tuple[Node, str]]], List[List[int]]]:
    """
    Given the root of a tree, finds all decision paths 
    used to reach leaf nodes in the tree. Optionally, this takes an array y of training data labels
    AND an array of specific labels to look for. In that case, only paths with leaf nodes
    which have a majority of a label within the labels array are returned.
    
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
        if last_node.type == 'leaf':
            leaf_label = mode(y[last_node.indices[0]])
            if leaf_label in labels:
                paths.append(path)
                path_labels.append([leaf_label])
            
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


def satisfies_path(X, path):
    """
    Given a dataset X and a decision path, determines 
    which data indices satisfy the path.
    
    Args:
        X (np.ndarray): Dataset to evaluate.
        
        path (List[(Node, str)]): Decision path to evaluate.
    
    Returns:
        (np.ndarray): Integer array of data indices satisfying the decision path. 
    """
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