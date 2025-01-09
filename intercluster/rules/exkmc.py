import numpy as np
from ExKMC.Tree import Tree as ExTree
from typing import List, Callable
import numpy.typing as npt
from ._node import Node
from ._splitter import Splitter
from ._tree import Tree
from .utils import *


class ExkmcTree(Tree):
    """ 
    The ExKMC tree is based around work from [Frost, Moshkovitz, Rashtchian '20] in their 
    paper titled 'ExKMC: Expanding Explainable k-Means Clustering.'
    The following processes a tree created by their implementation, 
    which may be found at: https://github.com/navefr/ExKMC.
    """
    
    def __init__(
        self,
        k : int,
        kmeans : Callable,
        max_leaf_nodes : int = None,
        imm : bool = True,
        feature_labels : List[str] = None
    ):
        """
        Args:
            k (int): Number of clusters.
            
            kmeans (Callable): Trained Sklearn KMeans model.
            
            max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
                Defaults to None.
                
            imm (bool, optional): If True, the base of the tree is built with the IMM algorithm.
                Defaults to True.
                
            feature_labels (List[str]): Iterable object with strings representing feature names. 
                
                
        Attributes:
            root (Node): Root node of the tree.
            
            heap (heapq list): Maintains the heap structure of the tree.
            
            leaf_count (int): Number of leaves in the tree.
            
            node_count (int): Number of nodes in the tree.
                
            depth (int): The maximum depth of the tree.
        """
        self.k = k
        self.kmeans = kmeans
        self.imm = imm
        splitter = Splitter()
        super().__init__(
            splitter = splitter,
            max_leaf_nodes = max_leaf_nodes,
            feature_labels = feature_labels
        )
    
    
    def fit(
        self,
        X : npt.NDArray,
        y : npt.NDArray = None
    ):
        """
        Fits and Exkmc tree to a dataset X. 

        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. Defaults to None.
        """
        # Reset if needed:
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
            
        base_tree = 'IMM' if self.imm else 'NONE'
        self.exkmc_tree = ExTree(
            k = self.k,
            max_leaves = self.max_leaf_nodes,
            base_tree = base_tree
        )
        self.exkmc_tree.fit(X, self.kmeans)
        
        self.root = Node()
        indices = np.arange(len(X))        
        self.grow(X, indices, self.exkmc_tree.tree, self.root, 0)
        self.node_count += 1
        
        
    def grow(
        self,
        X : npt.NDArray,
        indices : npt.NDArray,
        exkmc_node : Callable,
        node_obj: Node,
        depth : int
    ):
        """
        Traverses through the ExKMC tree.
        
        Args:
            X (np.ndarray): Input dataset.
            
            indices (np.ndarray): Subset of data indices to build node with.
            
            exkmc_node (ExKMC.Tree.Node): Node of an ExKMC tree.
            
            node_obj (Node): Corresponding Node object to copy conditions into.  
            
            depth (int): current depth of the tree
        """
        X_ = X[indices, :]
        if depth > self.depth:
            self.depth = depth
        
        if exkmc_node.is_leaf():
            class_label = self.leaf_count 
            self.leaf_count += 1
            node_obj.leaf_node(
                label = class_label,
                score = -1,
                indices = indices,
                depth = depth
            )
        else:
            feature, threshold = exkmc_node.feature, exkmc_node.value
            left_mask = X_[:, feature] <= threshold
            right_mask = ~left_mask
            
            left_child = Node()
            right_child = Node()
            
            node_obj.tree_node(
                left_child = left_child,
                right_child = right_child,
                features = [feature],
                weights = [1],
                threshold = threshold,
                score = -1,
                indices = indices,
                depth = depth,
                feature_labels = [self.feature_labels[feature]]
            )
            
            self.grow(
                X,
                indices[left_mask],
                exkmc_node.left,
                left_child,
                depth + 1
            )
            self.grow(
                X,
                indices[right_mask],
                exkmc_node.right,
                right_child,
                depth + 1
            )
            
            self.node_count += 2
            
    def predict(
        self,
        X : npt.NDArray,
        leaf_labels : bool = True
    ) -> npt.NDArray:
        """
        Predicts the labels of a dataset X.
        
        Args:
            X (np.ndarray): Input dataset.
            
            leaf_labels (bool, optional): If true, gives labels based soley upon 
                leaf membership. Otherwise, returns the orignal predictions from 
                the fitted tree. Defaults to True.  

        """
        if leaf_labels:
            labels = np.zeros(X.shape[0])
            decision_paths = get_decision_paths(self.root)
            for path in decision_paths:
                leaf = path[-1][0]
                satisfies = satisfies_path(X, path)
                labels[satisfies] = leaf.label
            return labels
        else:
            return self.exkmc_tree.predict(X)