import numpy as np
import copy
import heapq
from numpy.typing import NDArray
from typing import List, Callable
from intercluster.utils import mode
from ._node import Node
from .utils import *


class Tree():
    """
    Base class for a Tree object. 
    """
    def __init__(
        self,
        splitter : Callable,
        base_tree : Node = None,
        max_leaf_nodes : int = None,
        max_depth : int = None,
        min_points_leaf : int = 1,
        feature_labels : List[str] = None
    ):
        """
        Args:
            splitter (Callable): Function/Object which determines how to split leaf nodes.
            
            base_tree (Node, optional): Root node of a baseline tree to start from. 
                Defaults to None, in which case the tree is grown from root.
            
            max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
                Defaults to None.
                
            max_depth (int, optional): Optional constraint for maximum depth. 
                Defaults to None.
                
            min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
                within a single leaf. Defaults to 1.
                
            feature_labels (List[str]): Iterable object with strings representing feature names. 
            
            
        Attributes:            
            root (Node): Root node of the tree.
        
            heap (heapq list): Maintains the heap structure of the tree.
            
            leaf_count (int): Number of leaves in the tree.
            
            node_count (int): Number of nodes in the tree.
                
            depth (int): The maximum depth of the tree.
                
        """
        self.splitter = splitter
        self.base_tree = copy.deepcopy(base_tree)
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_points_leaf = min_points_leaf
        self.feature_labels = feature_labels

        self.root = None
        self.heap = []
        self.leaf_count = 0
        self.node_count = 0
        self.depth = 0
        

    def fit(
        self,
        X : NDArray,
        y : NDArray = None
    ):
        """
        Initiates and builds a decision tree around a given dataset. 
        Keeps a heap queue for all current leaf nodes in the tree,
        which prioritizes splitting the leaves with the largest costs (gain in cost performance).
        NOTE: I store -1*costs because heapq pops items with minimum cost by default.
        For example, heap leaf object looks like:
            (-1*gain, random_tiebreaker, node object, split info)
        Where split info is a tuple of precomputed (features, weights, thresholds) information.

        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. Defaults to None.
        """
        # Reset the heap and tree:
        self.heap = []
        self.leaf_count = 0
        self.node_count = 0
        
        # Set feature labels if not set already:
        if self.feature_labels is None:
            self.feature_labels = [None]*X.shape[1]
        else:
            if not len(self.feature_labels) == X.shape[1]:
                raise ValueError('Feature labels must match the shape of the data.')
        
        # if stopping criteria weren't provided, set to the maximum possible
        if self.max_leaf_nodes is None:
            self.max_leaf_nodes = len(X)
        
        if self.max_depth is None:
            self.max_depth = len(X) - 1
            
        # Initialize the dataset and splitter
        self.X = X
        self.y = y
        self.splitter.fit(X, y)
        
        if self.base_tree is None:
            self.root = Node()
            if y is None:
                root_label = self.leaf_count
            else:
                root_label = mode(y)
            
            root_indices = np.arange(len(X))
            root_cost = self.splitter.cost(root_indices)
            root_depth = 0
            self.root.leaf_node(
                label = root_label,
                cost = root_cost,
                indices = root_indices,
                depth = root_depth
            )
            
            self.add_leaf_node(self.root)
            
        else:
            self.root = self.base_tree
            decision_paths = get_decision_paths(self.base_tree)
            for path in decision_paths:
                l_indices = satisfies_path(X, path)
                l_node = path[-1][0]
                l_label = l_node.label
                l_cost = self.splitter.cost(l_indices)
                l_depth = l_node.depth
                
                l_node.leaf_node(
                    label = l_label,
                    cost = l_cost,
                    indices = l_indices,
                    depth = l_depth
                )
                self.add_leaf_node(l_node)   
        
        
        while len(self.heap) > 0:
            self.grow()
            
            
    def add_leaf_node(
        self,
        node : Node
    ):
        """
        Adds a new leaf node to the heap.
        
        Args:
            node (Node): Leaf node to add to the heap.
        """
        gain, split_info = self.splitter.split(indices = node.indices)
        random_tiebreaker = np.random.rand()
        leaf_obj = (-1*gain, random_tiebreaker, node, split_info)
        heapq.heappush(self.heap, leaf_obj)
        self.leaf_count += 1
        if node.depth > self.depth:
            self.depth = node.depth
        
    
    def branch(
        self,
        node : Node,
        split_info : Tuple[NDArray, NDArray, float]
    ):
        """
        Splits a leaf node into two new leaf nodes.
        
        Args:
            node (Node): Leaf node to add to the heap.
            
            split_info (Tuple[np.ndarray, np.ndarray, float]): 
                Precomputed information for the split.
        """            
        left_indices, right_indices = self.splitter.get_split_indices(node.indices, split_info)
        
        y_l = None
        y_r = None
        if self.y is not None:
            y_l = self.y[left_indices]
            y_r = self.y[right_indices]
        
        # Calculate cost of left and right branches
        left_cost = self.splitter.cost(left_indices)
        right_cost = self.splitter.cost(right_indices)
        left_depth = node.depth + 1
        right_depth = node.depth + 1
        
        if self.y is None:
            l_label = node.label
            r_label = self.leaf_count
        else:
            l_label = mode(y_l)
            r_label = mode(y_r)
        
        # Create New leaf nodes
        left_node = Node()
        left_node.leaf_node(
            label = l_label,
            cost = left_cost,
            indices = left_indices,
            depth = left_depth
        )
        right_node = Node()
        right_node.leaf_node(
            label = r_label,
            cost = right_cost,
            indices = right_indices,
            depth = right_depth
        )
        
        # And push them into the heap:
        self.add_leaf_node(left_node)
        self.add_leaf_node(right_node)
        
        # Transform the splitted node from a leaf node to an internal tree node:
        features, weights, threshold = split_info
        node.tree_node(
            left_child=left_node,
            right_child=right_node,
            features=features,
            weights=weights,
            threshold=threshold,
            cost=node.cost,
            indices=node.indices,
            depth=node.depth,
            feature_labels = [self.feature_labels[f] for f in features]
        )
        # Adjust counts:
        self.leaf_count -= 1
        self.node_count += 1
        
        

    def grow(
        self
    ):
        """
        Builds the decision tree by splitting leaf nodes.
        """
        # pop an object from the heap
        heap_leaf_obj = heapq.heappop(self.heap)
        gain = -1*heap_leaf_obj[0]
        node = heap_leaf_obj[2]
        split_info = heap_leaf_obj[3] 
        
        # If we've reached any of the maximum conditions, stop growth. 
        # NOTE: This should also stop if the splitter decides there is no more gain to be had.
        if (
            (gain == -np.inf) or
            (node.depth >= self.max_depth) or 
            (self.leaf_count >= (self.max_leaf_nodes)) or
            (len(node.indices) <= self.min_points_leaf)
        ):
            pass

        else:
            self.branch(node, split_info)

        
    def predict(self, X : NDArray) -> NDArray:
        """
        Predicts the class labels of an input dataset X by recursing through the tree to 
        find where data points fall into leaf nodes.

        Args:
            X (np.ndarray): Input n x m dataset

        Returns:
            labels (np.ndarray): Length n array of class labels. 
        """
        labels = np.zeros(X.shape[0])
        decision_paths = get_decision_paths(self.root)
        for path in decision_paths:
            leaf = path[-1][0]
            satisfies = satisfies_path(X, path)
            labels[satisfies] = leaf.label
            
        return labels
    
    def get_nodes(self) -> List[Node]:
        """
        Returns all leaf nodes in the tree.
        
        Returns:
            leaves (List[Node]): List of leaf nodes in the tree.
        """
        nodes = collect_nodes(self.root)
        return nodes
    
    def get_leaves(self) -> List[Node]:
        """
        Returns all leaf nodes in the tree.
        
        Returns:
            leaves (List[Node]): List of leaf nodes in the tree.
        """
        leaves = collect_leaves(self.root)
        return leaves
            