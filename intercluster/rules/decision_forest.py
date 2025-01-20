from typing import List, Dict, Any, Tuple
from ._decision_sets import DecisionSet
from .utils import *
from ..utils import *

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
        
        self.covers = {}
        self.costs = {}
        
        
        
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
            
        returns:
            rules (List[List[(Node, str)]]): List decision tree paths where each item in the path is
                a tuple of a node and the direction (left <= or right >) taken on it.
        """
        n,d = X.shape
        if self.feature_pairings is None:
            self.feature_pairings = [list(range(d))]
            
        rules = []
        rule_labels = []
        tree = None
        for _ in range(self.num_trees):
            rand_pairing = np.random.randint(len(self.feature_pairings))
            pairing = self.feature_pairings[rand_pairing]
            rand_features = np.random.choice(
                pairing, 
                size = min(self.max_features, len(pairing)),
                replace = False
            )
            rand_samples = np.random.choice(
                n, 
                size = int(self.train_size * n),
                replace = False
            )
            train_data = X[rand_samples, :]
            train_data = train_data[:, rand_features]
            
            train_labels = None
            rand_labels = None
            if y is not None:
                train_labels = y[rand_samples]
                rand_labels = np.random.choice(
                    np.unique(train_labels), 
                    size = min(self.max_labels, len(np.unique(train_labels))),
                    replace = False
                )
                train_labels = np.array([i if i in rand_labels else -1 for i in train_labels])
            
            rand_feature_labels = None
            if self.feature_labels is not None:
                rand_feature_labels = [self.feature_labels[i] for i in rand_features]
            
            
            tree = self.tree_model(
                feature_labels = rand_feature_labels,
                **self.tree_params
            )
        
            '''
            rand_depth = np.random.randint(2, self.tree_params['max_depth'] + 1)
            tree = self.tree_model(
                feature_labels = rand_feature_labels,
                **dict(self.tree_params, max_depth=rand_depth)
            )
            '''
            
            tree.fit(train_data, train_labels)
            
            new_rules = []
            new_labels = []
            if y is not None:
                # NOTE: Not sure about this here... I think this function is too specific
                new_rules, new_labels = get_decision_paths_with_labels(
                    tree.root,
                    train_labels,
                    rand_labels
                )
            else:
                new_rules = get_decision_paths(tree.root)
                new_labels = [[l] for l in 
                              range(len(rule_labels), len(rule_labels) + len(new_rules))]

            #import pdb; pdb.set_trace()
            rules = rules + new_rules
            rule_labels = rule_labels + new_labels
            
            # translate back to orignal indices
            node_list = tree.get_nodes()
            for node in node_list:
                node.indices = rand_samples[node.indices]
                if node.type == 'internal':
                    node.features = rand_features[node.features]
                
        return rules, rule_labels
    
    
    def get_covers(self, X : NDArray) -> Dict[int, List[int]]:
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            covers (dict[int, List[int]]): Dictionary with rules labels 
            as keys and arrays of indices of X as values.
        """
        covers = {}
        for i, rule in enumerate(self.decision_set):
            covers[i] = satisfies_path(X, rule)
        return covers