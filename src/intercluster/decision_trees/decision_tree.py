import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import List, Set, Callable
from numpy.typing import NDArray
from intercluster import Condition, LinearCondition
from intercluster import (
    Node,
    mode,
    labels_format,
    can_flatten,
    flatten_labels,
    traverse,
    get_decision_paths,
    satisfies_path
)
from .splitters import InformationGainSplitter, DummySplitter, ObliqueInformationGainSplitter
from .tree import Tree


####################################################################################################   


class DecisionTree(Tree):
    """
    Class designed to interface with an Sklearn Decision Tree Classifier. 
    For more information about Sklearn trees, please visit 
    (https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    #sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py)
    """
    
    def __init__(
        self,
        criterion : str = 'entropy',
        max_leaf_nodes : int = None,
        max_depth : int = None,
        min_points_leaf : int = 1,
        random_state : int = None,
        pruner : Callable = None, 
    ):
        """
         Args:
            criterion (str): The function to measure the quality of a split. Supported criteria
                are 'gini' for the Gini impurity and 'entropy' for the information gain.
                
            max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
                Defaults to None.
                
            max_depth (int, optional): Optional constraint for maximum depth. 
                Defaults to None.
                
            min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
                within a single leaf. Defaults to 1.

            pruner (Callable, optional): Function/Object used to prune branches of the tree.
                Defaults to None, in which case no pruning is performed.
            
        Attributes:
            root (Node): Root node of the tree.
        
            heap (heapq list): Maintains the heap structure of the tree.
            
            leaf_count (int): Number of leaves in the tree.
            
            node_count (int): Number of nodes in the tree.
                
            depth (int): The maximum depth of the tree.
        """
        self.criterion = criterion
        self.random_state = random_state
        splitter = DummySplitter()

        super().__init__(
            splitter = splitter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_points_leaf=min_points_leaf,
            pruner=pruner
        )

        if self.pruner is not None:
            supported_pruners = ['CoverageMistakePruner']
            if self.pruner.__name__ not in supported_pruners:
                raise ValueError(
                    f"Pruner {pruner.__name__} is not supported. "
                    "Supported pruners are: {supported_pruners}"
                )
        
            
    def fit(
        self,
        X : NDArray,
        y : List[Set[int]] = None
    ):
        """
        Fits a Sklearn tree to a dataset X and labels y.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels. Defaults to None.
        """ 
        # Reset if needed:
        self.heap = []
        self.leaf_count = 0
        self.node_count = 0

        # Remove any points with no label (outliers):
        if y is not None:
            X, y = zip(*[(x, label) for x, label in zip(X, y) if len(label) > 0])
            X = np.array(X)
            y = list(y)
        
        # if stopping criteria weren't provided, set to the maximum possible
        if self.max_leaf_nodes is None:
            self.max_leaf_nodes = len(X)
        
        if self.max_depth is None:
            self.max_depth = len(X) - 1
            
        # Initialize dataset and fit Sklearn tree:
        self.X = X
        self.y = y
        
        if not can_flatten(y):
            raise ValueError("Each data point must have exactly one label.")
        self.y_array = flatten_labels(y)
        
        self.sklearn_tree = DecisionTreeClassifier(
            criterion = self.criterion,
            max_depth = self.max_depth,
            min_samples_leaf = self.min_points_leaf,
            max_leaf_nodes = self.max_leaf_nodes,
            random_state = self.random_state
        )
        self.sklearn_tree.fit(X, self.y_array)
        self.classes = self.sklearn_tree.classes_
        self.tree_info = self.sklearn_tree.tree_
        self.root = Node()
        indices = np.arange(len(X))
        self.grow(indices, 0, self.root, 0)
        self.node_count += 1
        
        
    def grow(
        self,
        indices : NDArray,
        sklearn_node : int,
        node_obj : Node, 
        depth : int
    ):
        """
        Builds the decision tree by traversing the Sklearn tree.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray): Data labels.
            
            indices (np.ndarray): Subset of data indices to build node with.
            
            sklearn_node (int): Integer index of node in sklearn tree.
            
            node_obj (Node): Corresponding Node object to copy conditions into.  
            
            depth (int): current depth of the tree
        """
        X_ = self.X[indices, :]
        y_ = self.y_array[indices]
        
        if depth > self.depth:
            self.depth = depth
        
        if (self.tree_info.children_left[sklearn_node] < 0 and 
            self.tree_info.children_right[sklearn_node] < 0):
            class_label = self.classes[np.argmax(self.tree_info.value[sklearn_node])]
            node_obj.leaf_node(
                leaf_num = self.leaf_count,
                label = class_label,
                cost = -1,
                indices = indices,
                depth = depth
            )
            self.leaf_count += 1

        else:
            feature = self.tree_info.feature[sklearn_node]
            threshold = self.tree_info.threshold[sklearn_node]
            left_mask = X_[:, feature] <= threshold
            right_mask = ~left_mask
            
            left_child = Node()
            right_child = Node()
            
            condition = LinearCondition(
                features = np.array([feature]),
                weights = np.array([1]),
                threshold = threshold,
                direction = -1
            )
            
            node_obj.tree_node(
                left_child = left_child,
                right_child = right_child,
                condition = condition,
                cost = -1,
                indices = indices,
                depth = depth
            )
            
            self.grow(
                indices[left_mask],
                self.tree_info.children_left[sklearn_node],
                left_child,
                depth + 1
            )
            
            self.grow(
                indices[right_mask],
                self.tree_info.children_right[sklearn_node],
                right_child,
                depth + 1
            )
            
            self.node_count += 2

    
    def prune(self):
        """
        Prunes the decision tree by selecting a subset of leaf nodes which best satisfy the 
        pruning objective.
        """
        pass
            
    def predict(
        self,
        X : NDArray,
        leaf_labels : bool = False
    ) -> List[Set[int]]:
        """
        Predicts the labels of a dataset X.
        
        Args:
            X (np.ndarray): Input dataset.
            
            leaf_labels (bool, optional): If true, gives labels based soley upon 
                leaf membership. That is, each leaf is given a unique label. 
                Otherwise, returns the orignal predictions from 
                the fitted tree. Defaults to True.  

        Returns:
            labels (List[Set[int]]): List of label sets where the set at index i represents 
                the class labels of data point i.
        """
        if leaf_labels:
            labels = [set() for _ in range(X.shape[0])]
            decision_paths = get_decision_paths(self.root)
            for path in decision_paths:
                leaf = path[-1][0]
                satisfies = satisfies_path(X, path)
                for idx in satisfies:
                    labels[idx].add(leaf.leaf_num)
                    
            return labels
        
        else:
            return labels_format(self.sklearn_tree.predict(X))


####################################################################################################


class ID3Tree(Tree):
    """
    Inherits from the Tree class to implement a decision tree in which 
    axis aligned split criterion are chosen so that points in any leaf node 
    have small entropy with respect to their labels.

    This is a custom implementation of the decision tree class above, which does not 
    rely upon Sklearn's DecisionTreeClassifier (although is designed to be identical).
    
    Args:            
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
    
    def __init__(
        self,
        base_tree : Node = None,
        max_leaf_nodes : int = None,
        max_depth : int = None,
        min_points_leaf : int = 1
    ):
        splitter = InformationGainSplitter(
            min_points_leaf = min_points_leaf
        )  
        super().__init__(
            splitter = splitter,
            base_tree = base_tree, 
            max_leaf_nodes = max_leaf_nodes,
            max_depth = max_depth, 
            min_points_leaf = min_points_leaf
        )


####################################################################################################


class ObliqueTree(Tree):
    """
    Inherits from the Tree class to implement a decision tree in which 
    simple, linear split criterion are chosen so that points in any leaf node 
    have small entropy with respect to their labels.
    
    Args:            
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
    
    def __init__(
        self,
        base_tree : Node = None,
        max_leaf_nodes : int = None,
        max_depth : int = None,
        min_points_leaf : int = 1
    ):
        splitter = ObliqueInformationGainSplitter(
            min_points_leaf = min_points_leaf
        )  
        super().__init__(
            splitter = splitter,
            base_tree = base_tree, 
            max_leaf_nodes = max_leaf_nodes,
            max_depth = max_depth, 
            min_points_leaf = min_points_leaf
        )