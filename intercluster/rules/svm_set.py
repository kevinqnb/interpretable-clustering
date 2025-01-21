import numpy as np 
from sklearn.svm import LinearSVC
from typing import List, Tuple, Any, Dict
from numpy.typing import NDArray
from ._decision_set import DecisionSet


class SVMSet(DecisionSet):
    """
    Class for a decision set which trains multiple Support Vector Machine models 
    on random subsets of an input dataset, and uses the linear decision boundaries
    as rules.
    """
    def __init__(
        self,
        svm_params : Dict[str, Any],
        num_rules : int,
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
        self.svm_params = svm_params
        self.num_rules = num_rules
        self.max_features = max_features
        self.max_labels = max_labels
        self.train_size = train_size
        
        for pairing in feature_pairings:
            if not all(isinstance(i, int) for i in pairing):
                raise ValueError('Feature pairings must be a list of lists of integers.')
            
        self.feature_pairings = feature_pairings
        
        self.covers = {}
        self.costs = {}
    
    def _random_parameters(
        self,
        X : NDArray,
        y : NDArray
    ):
        """
        Randomly selects features, samples, and labels for training.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels.
            
        Returns:
        """
        n,d = X.shape
        
        # Random labels, choose the clusters to distinguish:
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
    
    
    def _fit_svm(
        self,
        X : NDArray,
        y : NDArray
    ) -> Any:
        """
        Fits dataset to a single tree.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. Defaults to None.
            
        returns:
            rules NOTE: FILL in the type of rule returned!!
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
        
        pass
        
    
        
    def _fitting(
        self,
        X : NDArray,
        y : NDArray
    ) -> Any:
        """
        Fits a decision set by training a forest of decision trees, 
        and using collecting their leaf nodes as rules.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. NOTE: May need to be 0/1 for SVM.
            
        returns:
            rules NOTE: FILL in the type of rule returned!!
        """
        n,d = X.shape
        
        if self.max_features is None:
            self.max_features = d
            
        if self.max_labels is None:
            self.max_labels = len(np.unique(y))
            
        if self.feature_pairings is None:
            self.feature_pairings = [list(range(d))]
            
        rules = []
        rule_labels = []
        for _ in range(self.num_trees):
            svm = self._fit_svm(X, y)   
            new_rules = []
            new_labels = []
            
            
            rules = rules + new_rules
            rule_labels = rule_labels + new_labels
                
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
        return covers