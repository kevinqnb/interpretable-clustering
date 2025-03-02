import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Any, Tuple, Set
from intercluster.utils import labels_format,unique_labels, can_flatten
from ._conditions import Condition
from ._decision_set import DecisionSet
from .utils import get_decision_paths, get_decision_paths_with_labels, satisfies_conditions

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
        max_depths : List[int] = None,
        train_size : float = 1.0
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

            max_depths (List[int]): Available maximum depths for each tree in the forest. 
                During the random forest process, a maximum depth is selected randomly from this
                list. 

            train_size (float): Fraction of random data points to train each tree with. 
        """
        super().__init__()
        self.tree_model = tree_model
        self.tree_params = tree_params
        self.num_trees = num_trees
        self.max_features = max_features
        self.max_labels = max_labels
        self.max_depths = max_depths

        assert train_size <= 1, "Fractional training size must be <= 1."
        assert train_size >= 0, "Fractional training size must be >= 0."
        self.train_size = train_size
        
        for pairing in feature_pairings:
            if not all(isinstance(i, int) for i in pairing):
                raise ValueError('Feature pairings must be a list of lists of integers.')
            
            #if len(pairing) < max_features:
            #    raise ValueError('Number of features must be less than or equal to the number of 
            # features in each pairing.')
            
        self.feature_pairings = feature_pairings
        self.costs = {}
        
    
    def _random_parameters(
        self,
        X : NDArray,
        y : List[Set[int]] = None
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
        rand_label_selects = None
        if y is not None:
            unique = unique_labels(y)
            rand_label_selects = np.random.choice(
                list(unique), 
                size = min(self.max_labels, len(unique)),
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
            
        # Random maximum depth
        rand_depth = None
        if self.max_depths is not None:
            rand_depth = np.random.choice(self.max_depths)
            
        return rand_samples, rand_features, rand_label_selects, rand_depth
    
    
    def _fit_tree(
        self,
        X : NDArray,
        y : List[Set[int]] = None
    ):
        """
        Fits dataset to a single tree.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels. Defaults to None.
            
        returns:
            rules (List[List[(Node, str)]]): List decision tree paths where each item in the path is
                a tuple of a node and the direction (left <= or right >) taken on it.
        """
        rand_samples, rand_features, rand_label_selects, rand_depth = self._random_parameters(X, y)
        
        train_labels = None
        if rand_label_selects is not None:
            train_labels = [set() for _ in range(len(y))]
            for i, label_set in enumerate(y):
                for l in label_set:
                    if l in rand_label_selects:
                        train_labels[i].add(l)
                    else:
                        train_labels[i].add(-1)
            train_labels = [train_labels[i] for i in rand_samples]
            
        train_data = X[rand_samples, :]
        train_data = train_data[:, rand_features]
        
        params = None
        if rand_depth is not None:
            params = dict(self.tree_params, max_depth=rand_depth)
        else:
            params = self.tree_params

        tree = self.tree_model(
            **params
        )
        tree.fit(train_data, train_labels)
        
        # Translate back to orignal indices:
        node_list = tree.get_nodes()
        for node in node_list:
            node.indices = rand_samples[node.indices]
            if node.type == 'internal':
                node.condition.features = rand_features[node.condition.features]
                
        return tree
        
    
        
    def _fitting(
        self,
        X : NDArray,
        y : List[Set[int]] = None
    ) -> List[List[Condition]]:
        """
        Fits a decision set by training a forest of decision trees, 
        and using collecting their leaf nodes as rules.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. Defaults to None.
            
        returns:
            rules (List[List[Condition]]): List where each item is a list of logical conditions. 
                In this case, each inner list represents a series of logical conditions used 
                within a certain path of the decision tree.
        """
        n,d = X.shape

        if not can_flatten(y):
            raise ValueError("Each data point must have exactly one label.")
        
        if self.max_features is None:
            self.max_features = d
            
        if (y is not None) and (self.max_labels is None):
            self.max_labels = len(unique_labels(y))
            
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
                select_labels = unique_labels(tree.y)
                select_labels = {l for l in select_labels if l != -1}
                filter_labels = [select_labels.intersection(label_set)
                                 if len(select_labels.intersection(label_set)) > 0
                                 else {-1} for label_set in y]
                
                new_rules, new_labels = get_decision_paths_with_labels(
                    root = tree.root,
                    #labels = filter_labels,
                    select_labels = select_labels
                )
            else:
                new_rules = get_decision_paths(tree.root)
                new_labels = [[l] for l in 
                              range(len(rule_labels), len(rule_labels) + len(new_rules))]
                
            # Take only the conditions from each node.
            new_rules = [[node.condition for node in rule[:-1]] for rule in new_rules]
            rules = rules + new_rules
            rule_labels = rule_labels + new_labels
            if tree.depth > self.max_rule_length:
                self.max_rule_length = tree.depth
                
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
        for i, condition_list in enumerate(self.decision_set):
            data_points_satisfied = satisfies_conditions(X, condition_list)
            assignment[data_points_satisfied, i] = True
        return assignment