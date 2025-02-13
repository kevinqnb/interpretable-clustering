import numpy as np
from numpy.typing import NDArray
from typing import List, Any, Tuple
from ._conditions import Condition, LinearCondition
from ._decision_set import DecisionSet
from .utils import satisfies_conditions


class VoronoiSet(DecisionSet):
    """
    Implements a decision set which is formed as a collection of 
    voronoi partitions in randomized subsets of the feature space.
    
    Specifically, for each decision set, choose a random centroid to 
    distinguish from all others and a size 2 subset of features on which to do so.  
    With k centroids, finding the voronoi conditions which split this centroid from 
    all the rest amounts to finding k - 1 linear conditions in 2 dimensional space.
    From among this set of conditions, randomly select up to at most num_conditions.
    """
    def __init__(
        self,
        centers : NDArray,
        num_sets : int,
        num_conditions : int = None,
        feature_pairings : List[List[int]] = None,
        feature_labels : List[str] = None
    ):
        """
        Args:
            centers (np.ndarray): Array of centroids to compute the voronoi partitions with.
            
            num_sets (int): Number of randomized voronoi partitions to collect.
            
            num_conditions
                
            feature_pairings (List[List[int]]): List of feature indices representing sets 
                features which can be used together in a decision tree. 

        """
        super().__init__(feature_labels = feature_labels)
        self.centers = centers
        self.num_sets = num_sets
        
        if num_conditions is None:
            self.num_conditions = len(centers) - 1
        elif num_conditions > len(centers) - 1:
            raise ValueError("Requested more conditions than there are pairwise comparisons.")
        else:
            self.num_conditions = num_conditions

        self.num_features = 2
        
        for pairing in feature_pairings:
            if not all(isinstance(i, int) for i in pairing):
                raise ValueError('Feature pairings must be a list of lists of integers.')
            
            if len(pairing) < 2:
                raise ValueError("Feature pairings must have at least two features...for now.")
            
        self.feature_pairings = feature_pairings
        
        
    def _random_parameters(self) -> Tuple[int, NDArray, NDArray]:
        """
        Randomly selects centroids and features for training.
        
        Returns:
            rand_center (int): Randomly chosen center index.
            
            rand_features (np.ndarray): Randomly chosen subset of features.
            
            rand_others (np.ndarray): Other centroids to compare against.
        """
        k,d = self.centers.shape
        
        # Choose the centroid to distinguish:
        rand_center = np.random.randint(k)
        
        # Random features, choose the features to train on:
        rand_pairing = np.random.randint(len(self.feature_pairings))
        pairing = self.feature_pairings[rand_pairing]
        rand_features = np.random.choice(
            pairing, 
            size = min(self.num_features, len(pairing)),
            replace = False
        )
        
        # Group of other centroids to compare against:
        rand_others = np.random.choice(
            [i for i in range(k) if i != rand_center],
            self.num_conditions,
            replace = False
        )
            
        return rand_center, rand_features, rand_others
    
    
    def _create_condition(
        self,
        target_center : NDArray,
        other_center : NDArray,
        features : NDArray
    ) -> Condition:
        """
        Given two centroids, finds the perpendicular bisector between them, and 
        creates a linear condition to distinguish the target center.
        
        Args:
            target_center (np.ndarray): Target centroid.
            other_center (np.ndarray): The other centroid for computing the bisector with.
            features (np.ndarray): Subset of features to compute the bisector on.
            
        Returns:
            condition (Condition): Linear condition for the perpendicular bisector.
        """
        c1 = target_center[features]
        c2 = other_center[features]
        slope = (c2[1] - c1[1])/(c2[0] - c1[0])
        midpoint = np.array([(c1[0] + c2[0])/2, (c1[1] + c2[1])/2])
        
        perpendicular_slope = -1/slope
        threshold = midpoint[1] - perpendicular_slope * midpoint[0]

        # -1 implies less than or equal to b, 1 implies greater than or equal to.
        inequality_direction = np.sign(c1[1] - perpendicular_slope*c1[0] - threshold)
        
        condition = LinearCondition(
            features = features,
            weights = np.array([-perpendicular_slope, 1]),
            threshold = threshold,
            direction=inequality_direction
        )
        
        return condition
    
    
    def _create_rule(self):
        """
        Finds a new voronoi rule.
        
        Returns:
            conditions (List[Condition]): List of linear conditions for the rule. 
                
            center (int): Associated label of the center.
        """
        k,d = self.centers.shape
        rand_center, rand_features, rand_others = self._random_parameters()
        
        conditions = []
        for j in rand_others:
            cond = self._create_condition(
                self.centers[rand_center,:],
                self.centers[j, :],
                features = rand_features
            )
            conditions.append(cond)
                
        return conditions, rand_center
        
    
    
    def _fitting(self, X : NDArray, y : NDArray = None) -> Tuple[List[Any], List[int]]:
        """
        Privately used, custom fitting function.
        Fits a decision set to an input dataset. 
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. Defaults to None.
            
        returns:
            decision_set (List[Any]): List of rules.
            
            decision_set_labels (List[int]): List of labels corresponding to each rule.
        """
        decision_sets = [None for _ in range(self.num_sets)]
        decision_set_labels = [None for _ in range(self.num_sets)]
        for i in range(self.num_sets):
            conditions, center = self._create_rule()
            decision_sets[i] = conditions
            decision_set_labels[i] = [center]
            
        return decision_sets, decision_set_labels
    
    
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
            data_points_satisfied = satisfies_conditions(X, rule)
            assignment[data_points_satisfied, i] = True
        return assignment
    
    
    
    