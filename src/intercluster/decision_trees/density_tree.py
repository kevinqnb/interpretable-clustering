import numpy as np
from typing import List,Set
from numpy.typing import NDArray
from intercluster import (
    get_decision_paths,
    satisfies_path,
    density_distance,
    pairwise_distance_threshold
)
from .splitters import DummySplitter
from .entropy_tree import SklearnTree


class DensityTree(SklearnTree):
    """
    A decision tree that uses entropy or gini as the splitting criterion, while removing leaf nodes 
    that don't satisfy density constraints.
    """
    
    def __init__(
        self,
        epsilon, 
        n_core,
        criterion : str = 'entropy',
        max_leaf_nodes=None,
        max_depth=None,
        min_points_leaf=1,
        random_state = None
    ):
        """
        Args:
            epsilon (float): Density threshold distance.

            n_core (int): Minimum number of points within an epsilon distance for a point to be 
                considered a dense, core point.

            criterion (str): The function to measure the quality of a split. Supported criteria
                are 'gini' for the Gini impurity and 'entropy' for the information gain.
                
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
        self.epsilon = epsilon
        self.n_core = n_core
        super().__init__(
            criterion=criterion,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_points_leaf=min_points_leaf,
            random_state=random_state
        )


    def predict(self, X : NDArray, leaf_labels = False) -> List[Set[int]]:
        """
        Predicts the class labels of an input dataset X by recursing through the tree to 
        find where data points fall into leaf nodes.

        Removes leaf nodes that do not satisfy density constraints. Points that fall into 
        removed leaf nodes are assigned a label of -1.

        Args:
            X (np.ndarray): Input n x m dataset

            leaf_labels (bool, optional): If true, gives labels based soley upon 
                leaf membership. Otherwise, returns the orignal class label predictions from 
                the fitted tree. Defaults to False.  

        Returns:
            labels (List[Set[int]]): Length n list of sets where the set at index i 
                represents the class labels for point i. 
        """
        distances = density_distance(X, self.n_core)
        labels = [set() for _ in range(len(X))]
        decision_paths = get_decision_paths(self.root)
        for path in decision_paths:
            leaf = path[-1]
            satisfies = satisfies_path(X, path)

            if pairwise_distance_threshold(distances, satisfies, self.epsilon):
                for idx in satisfies:
                    if leaf_labels:
                        labels[idx].add(leaf.leaf_num)
                    else:
                        labels[idx].add(leaf.label)

            else:
                # If the path does not satisfy the density condition, leaf node is 
                # removed and points are assigned a label of -1.
                for idx in satisfies:
                    labels[idx].add(-1)
            
        return labels
    

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
        distances = density_distance(X, self.n_core)
        decision_paths = get_decision_paths(self.root)
        for path in decision_paths:
             satisfies = satisfies_path(X, path)

             if pairwise_distance_threshold(distances, satisfies, self.epsilon):
                total_covers += len(satisfies)
                if len(satisfies) != 0:
                    wad += len(satisfies) * (len(path) - 1)

        if total_covers == 0:
            return np.nan
        else:
            return wad/total_covers


