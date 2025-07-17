import heapq
import numpy as np
from typing import List, Set, Tuple, Callable
from numpy.typing import NDArray
from intercluster import Condition, LinearCondition
from intercluster import (
    Node,
    mode,
    can_flatten,
    flatten_labels,
    get_decision_paths,
    satisfies_path
)
from .splitters import ExplanationSplitter
from .tree import Tree


class ExplanationTree(Tree):
    """
    Implements a tree in which outliers are removed during the fitting process
    in order to create an explainable clustering for the remaining set of points.
    
    This follows the explainable clustering algorithm outlied by Bandyapadhyay et al. 
    in their paper titled:
    "How to Find a Good Explanation for Clustering?" (2022 AAAI).
    """
    def __init__(
        self,
        num_clusters : int, 
        min_points_leaf : int = 1,
        cpu_count : int = 1
    ):
        """
        Args:
            num_clusters (int): The total number of observed clusters.
            
            cpu_count (int, optional): Number of processors to use. Defaults to 1.
            
        Attributes:            
            root (Node): Root node of the tree.
        
            heap (heapq list): Maintains the heap structure of the tree.
            
            leaf_count (int): Number of leaves in the tree.
            
            node_count (int): Number of nodes in the tree.
                
            depth (int): The maximum depth of the tree.

            outliers (Set[int]): Set of outlier data points to remove.
                
        """
        self.num_clusters = num_clusters
        splitter = ExplanationSplitter(
            num_clusters = num_clusters,
            min_points_leaf = min_points_leaf,
            #cpu_count = cpu_count
        )
        super().__init__(
            splitter = splitter,
            base_tree = None,
            max_leaf_nodes = num_clusters,
            max_depth = None,
            min_points_leaf = min_points_leaf
        )

        self.outliers = set()

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
        new_outliers = self.splitter.get_split_outliers(left_indices, right_indices)
        self.splitter.update_outliers(new_outliers)
        self.outliers = self.outliers.union(new_outliers)
        # This is really important, keeps track of outlier removal!
        left_indices_remain = np.array(list(set(left_indices) - self.outliers))
        right_indices_remain = np.array(list(set(right_indices) - self.outliers))

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
            indices = left_indices_remain,
            depth = left_depth
        )
        right_node = Node()
        right_node.leaf_node(
            leaf_num = r_leaf_num,
            label = r_label,
            cost = right_cost,
            indices = right_indices_remain,
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
        

    def predict(
        self, X : NDArray,
        leaf_labels = False,
        remove_outliers : bool = False
    ) -> List[Set[int]]:
        """
        Predicts the class labels of an input dataset X by recursing through the tree to 
        find where data points fall into leaf nodes.

        Args:
            X (np.ndarray): Input n x m dataset

            leaf_labels (bool, optional): If true, gives labels based soley upon 
                leaf membership. Otherwise, returns the orignal class label predictions from 
                the fitted tree. Defaults to False. 
                
            remove_outliers (bool, optional): If true, ignore indices which appear in self.outliers.
                This should really only be used for prediciting upon the same set of 
                data that was trained upon. Since that's often how it is used,
                this defaults to True. 

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

        # Remove outliers
        if remove_outliers:
            for outlier in self.outliers:
                labels[outlier] = {-1}
            
        return labels


    def get_weighted_average_depth(self, X : NDArray, remove_outliers : bool = True) -> float:
        """
        Finds the weighted average depth of the tree, which is adjusted by the number 
        data points which fall into each leaf node. 

        Args:
            X : Input dataset to predict with. 

            remove_outliers (bool, optional): If true, ignore indices which appear in self.outliers.
                This should really only be used for calling upon the same set of 
                data that was originally trained upon. Since that's often how it is used,
                this defaults to True. 

        Returns:
            wad (float): Weighted average depth.
        """
        X_ = X
        if remove_outliers:
            non_outliers = [i for i in range(len(X)) if i not in self.outliers]
            X_ = X[non_outliers, :]

        wad = 0
        total_covers = 0
        decision_paths = get_decision_paths(self.root)
        for path in decision_paths:
             satisfies = satisfies_path(X_, path)
             total_covers += len(satisfies)
             if len(satisfies) != 0:
                wad += len(satisfies) * (len(path) - 1)

        return wad/total_covers
        
 
        