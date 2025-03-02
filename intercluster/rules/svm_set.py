import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from numpy.typing import NDArray
from typing import List, Any, Tuple, Set
from intercluster.utils import unique_labels, can_flatten, flatten_labels, mode
from ._conditions import Condition, LinearCondition
from ._decision_set import DecisionSet
from .utils import satisfies_conditions


class SVMSet(DecisionSet):
    """
    Implements a decision set which is formed as a collection of 
    one class versus all SVM partitions. To create concise explainable decision boundaries, 
    recursive feature elimination is used within the training of each SVM model.
    """
    def __init__(
        self,
        num_rules : int,
        num_features : int = 2,
        feature_pairings : List[List[int]] = None,
        train_size : float = 1.0
    ):
        """
        Args:
            num_rules (int): Number of rules to create. 

            num_features (int) : Number of features to select with recurisve feature elimination.
                Defaults to 2, in which case decision boundaries are lines in two dimensions.

            feature_pairings (List[List[int]]): List of feature indices representing sets 
                features which may be used together within a decision boundary. 

            train_size (float): Fraction of random data points to train each model with. 
        """
        super().__init__()
        self.num_rules = num_rules
        self.num_features = num_features

        # SVM rules are always single linear conditions (just varied on the number of 
        # features one chooses to use).
        self.max_rule_length = 1
        
        for pairing in feature_pairings:
            if not all(isinstance(i, int) for i in pairing):
                raise ValueError('Feature pairings must be a list of lists of integers.')
            
            if len(pairing) < self.num_features:
                raise ValueError("Pairings must be at least as long as n_features.")
            
        self.feature_pairings = feature_pairings

        assert train_size <= 1, "Fractional training size must be <= 1."
        assert train_size >= 0, "Fractional training size must be >= 0."
        self.train_size = train_size
        
        
    def _random_parameters(
            self,
            X : NDArray,
            y : List[Set[int]],
        ) -> Tuple[int, NDArray, int]:
        """
        Randomly selects samples and features for training.
        
        Returns:
            rand_label (int): Cluster label to distinguish.

            rand_samples (np.ndarray): Array of sample indices to use for training. 

            rand_pairing (int): Index of feature pairing to use for training. 
        """
        # NOTE: Should think about the case where chosen sample has no positive labels. 

        n,d = X.shape
        
        # Random labels, choose a single cluster to distinguish:
        rand_label = None
        if y is not None:
            unique = unique_labels(y)
            rand_label = np.random.choice(
                list(unique), 
                size = 1,
                replace = False
            )[0]
            
        # Random samples, choose the data points to train on::
        rand_samples = np.random.choice(
            n, 
            size = int(self.train_size * n),
            replace = False
        )

        # Choose the pairing of features to train on:
        rand_pairing = np.random.randint(len(self.feature_pairings))
            
        return rand_label, rand_samples, rand_pairing
    
    
    def _create_rule(self, X : NDArray, y : List[Set[int]]) -> Condition:
        """
        Finds a new voronoi rule.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels.
        
        Returns:
            condition (Condition): Rule object for a linear decision boundary. 

            label (int): Label for the rule.
        """
        rand_label, rand_samples, rand_pairing = self._random_parameters(X, y)
        rand_features = np.array(self.feature_pairings[rand_pairing])
        y_ = (self.y_array == rand_label).astype(int)
        y_ = y_[rand_samples]

        # Make sure we're not trying to distinguish an empty set:
        while np.sum(y_) < 1:
            rand_label, rand_samples, rand_pairing = self._random_parameters(X, y)
            rand_features = np.array(self.feature_pairings[rand_pairing])
            y_ = (self.y_array == rand_label).astype(int)
            y_ = y_[rand_samples]

        X_ = X[rand_samples, :]
        X_ = X_[:, rand_features]

        svm = LinearSVC()
        selector = RFE(svm, n_features_to_select=self.num_features, step=1)
        selector = selector.fit(X_, y_)
        selected_features = selector.support_

        X_select = X_[:, selected_features]
        svm.fit(X_select, y_)
        svm_labels = svm.predict(X_select)

        # translate back to original feature indices
        features = rand_features[selected_features]
        weights = np.ndarray.flatten(svm.coef_)
        threshold = -svm.intercept_[0]

        positive_predicts = svm_labels == 1
        if np.sum(positive_predicts) == 0:
            # Empty prediction set -- take the opposite direction of the other classes' boundary
            decision_values = svm.decision_function(X_select[~positive_predicts, :])
            direction = -1 if mode(decision_values) > 0 else 1
        else:
            # Otherwise choose the direction corresponding to the distinguishing cluster.
            decision_values = svm.decision_function(X_select[positive_predicts, :])
            direction = 1 if mode(decision_values) > 0 else -1

        condition = LinearCondition(
                features = features,
                weights = weights,
                threshold = threshold,
                direction = direction
        )
        return condition, rand_label
        
    
    
    def _fitting(self, X : NDArray, y : List[Set[int]]) -> Tuple[List[Any], List[int]]:
        """
        Privately used, custom fitting function.
        Fits a decision set to an input dataset. 
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels.
            
        returns:
            decision_set (List[Any]): List of rules.
            
            decision_set_labels (List[int]): List of labels corresponding to each rule.
        """
        if not can_flatten(y):
            raise ValueError("Each data point must have exactly one label.")
        
        #self.X = X
        self.y_array = flatten_labels(y)
        
        decision_set = [None for _ in range(self.num_rules)]
        decision_set_labels = [None for _ in range(self.num_rules)]
        for i in range(self.num_rules):
            condition, label = self._create_rule(X, y)
            decision_set[i] = [condition]
            decision_set_labels[i] = {label}
            
        return decision_set, decision_set_labels
    
    
    def get_data_to_rules_assignment(self, X : NDArray) -> NDArray:
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            assignment (np.ndarray): n x n_rules boolean matrix with entry (i,j) being True
                if point i is covered by rule j and False otherwise.
        """
        n = X.shape[0]
        n_rules = len(self.decision_set)

        rows = []
        cols = []
        values = []
        for j, rule in enumerate(self.decision_set):
            data_indices_satisfied = satisfies_conditions(X, rule)
            for i in data_indices_satisfied:
                rows.append(i)
                cols.append(j)
                values.append(True)
        
        assignment = coo_matrix((values, (rows, cols)), shape=(n, n_rules), dtype = bool)
        assignment = assignment.tocsc()
        return assignment
        '''
        assignment = np.zeros((X.shape[0], len(self.decision_set)))
        for i, rule in enumerate(self.decision_set):
            data_points_satisfied = satisfies_conditions(X, rule)
            assignment[data_points_satisfied, i] = True
        return assignment
        '''