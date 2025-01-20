import numpy as np
import heapq
from ExKMC.Tree import Tree as ExTree
from typing import List, Callable
import numpy.typing as npt
from intercluster.splitters import ImmSplitter, DummySplitter
from ._node import Node
from ._tree import Tree
from .utils import *


class ImmTree(Tree):
    """
    Inherits from the Tree class to 
    create a tree in which leaf nodes are split so that 
    1) An input set of reference cluster centers are separated from each other, and
    2) The number of points separated from their closest cluster center is minimized. 
    
    This an implementation of the following work:
    "Explainable k-Means and k-Medians Clustering"
    Dasgupta, Frost, Moshkovitz, Rashtchian, 2020
    (https://arxiv.org/abs/2002.12538)
    
    Args:
        centers (np.ndarray, optional): Input list of reference centers to calculate cost with. 
            Defaults to None.
            
        norm (int, optional): Takes values 1 or 2. If norm = 1, compute distances using the 
            1 norm. If 2, compute distances with the squared two norm. Defaults to 2.
            
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
    
    def __init__(
        self,
        centers : npt.NDArray = None,
        norm : int = 2,
        max_leaf_nodes : int = None,
        max_depth : int = None,
        min_points_leaf : int = 1,
        feature_labels : List[str] = None
    ):
        self.centers = centers
        splitter = ImmSplitter(
            centers = centers,
            norm = norm,
            min_points_leaf = min_points_leaf
        )  
        super().__init__(
            splitter = splitter,
            max_leaf_nodes = max_leaf_nodes,
            max_depth = max_depth, 
            min_points_leaf = min_points_leaf,
            feature_labels = feature_labels
        )
        
    def fit(
        self,
        X : npt.NDArray,
        y : npt.NDArray = None
    ):
        """
        Initiates and builds a decision tree around a given dataset. 
        Keeps a heap queue for all current leaf nodes in the tree,
        which prioritizes splitting the leaves with the largest costs (gain in cost performance).
        NOTE: I store -1*costs because heapq pops items with minimum cost by default.
        For example, heap leaf object looks like:
            (-1*cost, random_tiebreaker, node object, split info)
        Where split info is a tuple of precomputed (features, weights, thresholds) information.

        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Dummy variable, defaults to None
                in which case the labels are manually set within the
                following method by assigning each data point to its closest center.
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
            
        # Initialize splitter
        self.X = X
        self.splitter.fit(X, y)
        
        self.root = Node()
        root_label = 0
        root_indices = np.arange(len(X))
        root_centroid_indices = np.arange(len(self.centers))
        root_cost = 0
        root_depth = 0
        self.root.leaf_node(
            label = root_label,
            cost = root_cost,
            indices = root_indices,
            depth = root_depth,
            centroid_indices = root_centroid_indices
        )
        
        self.add_leaf_node(self.root)
        
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
            
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. Defaults to None.
        """
        gain, split_info = self.splitter.split(
            indices = node.indices,
            centroid_indices = node.centroid_indices
        )
        random_tiebreaker = np.random.rand()
        leaf_obj = (-1*gain, random_tiebreaker, node, split_info)
        heapq.heappush(self.heap, leaf_obj)
        self.leaf_count += 1
        if node.depth > self.depth:
            self.depth = node.depth
            
            
    def branch(
        self,
        node : Node,
        split_info : Tuple[npt.NDArray, npt.NDArray, float]
    ):
        """
        Splits a leaf node into two new leaf nodes.
        
        Args:
            node (Node): Leaf node to add to the heap.
            
            split_info (Tuple[np.ndarray, np.ndarray, float]): 
                Precomputed information for the split.
            
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. Defaults to None.
        """
        features, weights, threshold = split_info
        
        (   left_indices,
            right_indices,
            left_centroid_indices,
            right_centroid_indices
        ) = self.splitter.get_split_indices(node.indices, node.centroid_indices, split_info)
        
        # Calculate cost of left and right branches
        left_cost = self.splitter.cost(
            left_indices,
            left_centroid_indices,
            node.centroid_indices
        )
        right_cost = self.splitter.cost(
            right_indices,
            right_centroid_indices,
            node.centroid_indices
        )
        
        left_depth = node.depth + 1
        right_depth = node.depth + 1
        
        l_label = left_centroid_indices[0]
        r_label = right_centroid_indices[0]
        
        # Create New leaf nodes
        left_node = Node()
        left_node.leaf_node(
            label = l_label,
            cost = left_cost,
            indices = left_indices,
            depth = left_depth,
            centroid_indices = left_centroid_indices
        )
        right_node = Node()
        right_node.leaf_node(
            label = r_label,
            cost = right_cost,
            indices = right_indices,
            depth = right_depth,
            centroid_indices = right_centroid_indices
        )
        
        # And push them into the heap:
        self.add_leaf_node(left_node)
        self.add_leaf_node(right_node)
        
        # Transform the splitted node from a leaf node to an internal tree node:
        node.tree_node(
            left_child=left_node,
            right_child=right_node,
            features=features,
            weights=weights,
            threshold=threshold,
            cost=node.cost,
            indices=node.indices,
            depth=node.depth,
            feature_labels = [self.feature_labels[f] for f in features],
            centroid_indices=node.centroid_indices
        )
        # Adjust counts:
        self.leaf_count -= 1
        self.node_count += 1
        
        
    def grow(
        self
    ):
        """
        Builds the decision tree by splitting leaf nodes.
        
        Args:
            X (np.ndarray): Input dataset.
                
            y (np.ndarray, optional): Target labels. Defaults to None.
        """
        # pop an object from the heap
        heap_leaf_obj = heapq.heappop(self.heap)
        node = heap_leaf_obj[2]
        split_info = heap_leaf_obj[3] 
        
        # If we've reached any of the maximum conditions, stop growth.
        if (
            (node.depth >= self.max_depth) or 
            (self.leaf_count >= (self.max_leaf_nodes)) or
            (len(node.indices) <= self.min_points_leaf) or
            (len(node.centroid_indices) <= 1)
        ):
            pass

        else:
            self.branch(node, split_info)
        
                    
####################################################################################################


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
        splitter = DummySplitter()
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
                cost = -1,
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
                cost = -1,
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