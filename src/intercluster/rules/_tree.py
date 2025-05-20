import numpy as np
import copy
import heapq
from numpy.typing import NDArray
from typing import List, Set, Callable
from intercluster.utils import mode, can_flatten, flatten_labels
from ._conditions import Condition
from ._node import Node
from .utils import *

import time
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
        min_points_leaf : int = 1
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

        self.root = None
        self.heap = []
        self.leaf_count = 0
        self.node_count = 0
        self.depth = 0
        

    def fit(
        self,
        X : NDArray,
        y : List[Set[int]] = None
    ):
        """
        Initiates and builds a decision tree around a given dataset. 
        Keeps a heap queue for all current leaf nodes in the tree,
        which prioritizes splitting the leaves with the largest costs (gain in cost performance).

        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. Defaults to None.
        """
        # Reset the heap and tree:
        self.heap = []
        self.leaf_count = 0
        self.node_count = 0
        
        # if stopping criteria weren't provided, set to the maximum possible
        if self.max_leaf_nodes is None:
            self.max_leaf_nodes = len(X)
        
        if self.max_depth is None:
            self.max_depth = len(X) - 1
            
        # Initialize the dataset and splitter
        if not can_flatten(y):
            raise ValueError("Each data point must have exactly one label.")

        self.X = X
        self.y = y
        self.splitter.fit(X, y)
        
        if self.base_tree is None:
            self.root = Node()
            if y is None:
                root_label = None
            else:
                root_label = mode(flatten_labels(y))
            
            root_indices = np.arange(len(X))
            root_cost = self.splitter.cost(root_indices)
            root_depth = 0
            self.root.leaf_node(
                leaf_num = self.leaf_count,
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
                l_num = l_node.leaf_num
                l_label = l_node.label
                l_cost = self.splitter.cost(l_indices)
                l_depth = l_node.depth
                
                l_node.leaf_node(
                    leaf_num = l_num,
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

        NOTE: I store -1*gain because heapq pops items with minimum value by default, whereas
        we really want to search for largest gain.
        
        Args:
            node (Node): Leaf node to add to the heap.
        """
        gain, condition = self.splitter.split(indices = node.indices)
        random_tiebreaker = np.random.rand()
        leaf_obj = (-1*gain, random_tiebreaker, node, condition)
        heapq.heappush(self.heap, leaf_obj)
        self.leaf_count += 1
        if node.depth > self.depth:
            self.depth = node.depth
        
    
    def branch(
        self,
        node : Node,
        condition : Condition
    ):
        """
        Splits a leaf node into two new leaf nodes.
        
        Args:
            node (Node): Leaf node to add to the heap.
            
            condition (Condition): Logical or functional condition for evaluating and 
                splitting the data points.
        """            
        left_indices, right_indices = self.splitter.get_split_indices(node.indices, condition)
        
        y_l = None
        y_r = None
        if self.y is not None:
            y_l = [self.y[idx] for idx in left_indices]
            y_r =  [self.y[idx] for idx in right_indices]
        
        # Calculate cost of left and right branches
        left_cost = self.splitter.cost(left_indices)
        right_cost = self.splitter.cost(right_indices)
        left_depth = node.depth + 1
        right_depth = node.depth + 1
        l_leaf_num = node.leaf_num
        r_leaf_num = self.leaf_count

        if self.y is None:
            l_label = None
            r_label = None
        else:
            l_label = mode(flatten_labels(y_l))
            r_label = mode(flatten_labels(y_r))
        
        # Create New leaf nodes
        left_node = Node()
        left_node.leaf_node(
            leaf_num = l_leaf_num,
            label = l_label,
            cost = left_cost,
            indices = left_indices,
            depth = left_depth
        )
        right_node = Node()
        right_node.leaf_node(
            leaf_num = r_leaf_num,
            label = r_label,
            cost = right_cost,
            indices = right_indices,
            depth = right_depth
        )
        
        # And push them into the heap:
        self.add_leaf_node(left_node)
        self.add_leaf_node(right_node)
        
        # Transform the splitted node from a leaf node to an internal tree node:
        node.tree_node(
            left_child=left_node,
            right_child=right_node,
            condition = condition,
            cost=node.cost,
            indices=node.indices,
            depth=node.depth
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
        condition = heap_leaf_obj[3] 
        
        # If we've reached any of the maximum conditions, stop growth. 
        # NOTE: This should also stop if the splitter decides there is no more gain to be had.
        # In that case, splitters in this package will output a gain of -np.inf.
        if (
            (gain == -np.inf) or
            (node.depth >= self.max_depth) or 
            (self.leaf_count >= (self.max_leaf_nodes)) or
            (len(node.indices) <= self.min_points_leaf)
        ):
            pass

        else:
            self.branch(node, condition)

        
    def predict(self, X : NDArray, leaf_labels = False) -> List[Set[int]]:
        """
        Predicts the class labels of an input dataset X by recursing through the tree to 
        find where data points fall into leaf nodes.

        Args:
            X (np.ndarray): Input n x m dataset

            leaf_labels (bool, optional): If true, gives labels based soley upon 
                leaf membership. Otherwise, returns the orignal class label predictions from 
                the fitted tree. Defaults to False.  

        Returns:
            labels (List[Set[int]]): Length n list of sets where the set at index i 
                represents the class labels for point i. 
        """
        labels = [set() for _ in range(len(X))]
        decision_paths = get_decision_paths(self.root)
        for path in decision_paths:
            leaf = path[-1]
            satisfies = satisfies_path(X, path)
            for idx in satisfies:
                if leaf_labels:
                    labels[idx].add(leaf.leaf_num)
                else:
                    labels[idx].add(leaf.label)
            
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
    
    def get_weighted_average_depth(self, X : NDArray) -> float:
        """
        Finds the weighted average depth of the tree, which is adjusted by the number 
        data points which fall into each leaf node. 

        Args:
            X : Input dataset to predict with. 

        Returns:
            wad (float): Weighted average depth.
        """
        wad = 0
        total_covers = 0
        decision_paths = get_decision_paths(self.root)
        for path in decision_paths:
             satisfies = satisfies_path(X, path)
             total_covers += len(satisfies)
             if len(satisfies) != 0:
                wad += len(satisfies) * (len(path) - 1)

        return wad/total_covers
            