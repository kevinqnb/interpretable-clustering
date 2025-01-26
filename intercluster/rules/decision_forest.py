import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Any, Tuple
from ._node import Node
from ._decision_set import DecisionSet
from .utils import get_decision_paths, get_decision_paths_with_labels, satisfies_path

class DecisionForest(DecisionSet):
    """
    Class for a decision set which trains multiple decision trees according to a 
    random forest model, and returns the set of all leaf nodes as rules.
    """
    def __init__(
        self,
        tree_model : Any,
        tree_params : Dict[str, Any],
        num_trees : int,
        max_features : int = None,
        max_labels : int = None,
        feature_pairings : List[List[int]] = None,
        train_size : float = 1.0,
        feature_labels : List[str] = None
    ):
        """
        Args:
            tree_model (Tree Object): Decision tree model to be used in the 
                creation of decision sets.
                
            tree_params (dict[str: any]): Dictionary of parameters to be passed as input 
                to the tree_model.
                
            num_trees (int): Number of trees to be trained.
            
            max_features (int, optional): Maximum number of features to be used in each tree. 
                Defaults to None, in which case all features are used.
                
            max_labels (int, optional): Maximum number of labels for each individual tree 
                distinguish. Defaults to None, in which case each tree is trained on the 
                entire set of labels.
                
            feature_pairings (List[List[int]]): List of feature indices representing sets 
                features which can be used together in a decision tree. 

        """
        super().__init__(feature_labels = feature_labels)
        self.tree_model = tree_model
        self.tree_params = tree_params
        self.num_trees = num_trees
        self.max_features = max_features
        self.max_labels = max_labels
        self.train_size = train_size
        
        for pairing in feature_pairings:
            if not all(isinstance(i, int) for i in pairing):
                raise ValueError('Feature pairings must be a list of lists of integers.')
            
            #if len(pairing) < max_features:
            #    raise ValueError('Number of features must be less than or equal to the number of 
            # features in each pairing.')
            
        self.feature_pairings = feature_pairings
        self.costs = {}
        self.depth = 0
        
    
    def _random_parameters(
        self,
        X : NDArray,
        y : NDArray = None
    ):
        """
        Randomly selects features, samples, and labels for training.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. Defaults to None.
            
        Returns:
        """
        n,d = X.shape
        
        # Random labels, choose the clusters to distinguish:
        rand_labels = None
        if y is not None:
            rand_labels = np.random.choice(
                np.unique(y), 
                size = min(self.max_labels, len(np.unique(y))),
                replace = False
            )
            
        # Random samples, choose the data points to train on::
        rand_samples = np.random.choice(
            n, 
            size = int(self.train_size * n),
            replace = False
        )
        
        # Random features, choose the features to train on:
        rand_pairing = np.random.randint(len(self.feature_pairings))
        pairing = self.feature_pairings[rand_pairing]
        rand_features = np.random.choice(
            pairing, 
            size = min(self.max_features, len(pairing)),
            replace = False
        )
            
        '''
        # Random depth
        rand_depth = np.random.randint(2, self.tree_params['max_depth'] + 1)
        tree = self.tree_model(
            feature_labels = rand_feature_labels,
            **dict(self.tree_params, max_depth=rand_depth)
        )
        '''
            
        return rand_samples, rand_features, rand_labels
        
        
        
    
    
    def _fit_tree(
        self,
        X : NDArray,
        y : NDArray = None
    ):
        """
        Fits dataset to a single tree.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. Defaults to None.
            
        returns:
            rules (List[List[(Node, str)]]): List decision tree paths where each item in the path is
                a tuple of a node and the direction (left <= or right >) taken on it.
        """
        rand_samples, rand_features, rand_labels = self._random_parameters(X, y)
        
        train_labels = None
        if rand_labels is not None:
            train_labels = np.array([i if i in rand_labels else -1 for i in y])
            train_labels = train_labels[rand_samples]
            
        train_data = X[rand_samples, :]
        train_data = train_data[:, rand_features]
        
        train_feature_labels = None
        if self.feature_labels is not None:
            train_feature_labels = [self.feature_labels[i] for i in rand_features]
            
        tree = self.tree_model(
                feature_labels = train_feature_labels,
                **self.tree_params
        )
        tree.fit(train_data, train_labels)
        
        # Translate back to orignal indices:
        node_list = tree.get_nodes()
        for node in node_list:
            node.indices = rand_samples[node.indices]
            if node.type == 'internal':
                node.features = rand_features[node.features]
                
        return tree
        
    
        
    def _fitting(
        self,
        X : NDArray,
        y : NDArray = None
    ) -> List[List[Tuple[Node, str]]]:
        """
        Fits a decision set by training a forest of decision trees, 
        and using collecting their leaf nodes as rules.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. Defaults to None.
            
        returns:
            rules (List[List[(Node, str)]]): List decision tree paths where each item in the path is
                a tuple of a node and the direction (left <= or right >) taken on it.
        """
        n,d = X.shape
        
        if self.max_features is None:
            self.max_features = d
            
        if (y is not None) and (self.max_labels is None):
            self.max_labels = len(np.unique(y))
            
        if self.feature_pairings is None:
            self.feature_pairings = [list(range(d))]
            
        rules = []
        rule_labels = []
        tree = None
        for _ in range(self.num_trees):
            tree = self._fit_tree(X, y)   
            new_rules = []
            new_labels = []
            if y is not None:
                selected_labels = np.unique(tree.y)
                selected_labels = selected_labels[selected_labels != -1]
                y_filter = np.array([i if i in selected_labels else -1 for i in y])
                new_rules, new_labels = get_decision_paths_with_labels(
                    tree.root,
                    y_filter,
                    selected_labels
                )
            else:
                new_rules = get_decision_paths(tree.root)
                new_labels = [[l] for l in 
                              range(len(rule_labels), len(rule_labels) + len(new_rules))]

            #import pdb; pdb.set_trace()
            rules = rules + new_rules
            rule_labels = rule_labels + new_labels
            if tree.depth > self.depth:
                self.depth = tree.depth
                
        return rules, rule_labels
    
    
    def get_data_to_rules_assignment(self, X : NDArray) -> NDArray:
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            assignment (np.ndarray): n x n_rules boolean matrix with entry (i,j) being True
                if point i is covered by rule j and False otherwise.
        """
        assignment = np.zeros((X.shape[0], len(self.decision_set)))
        for i, rule in enumerate(self.decision_set):
            data_points_satisfied = satisfies_path(X, rule)
            assignment[data_points_satisfied, i] = True
        return assignment