import numpy as np
import networkx as nx
from typing import List, Set, Any, Tuple, Callable
from numpy.typing import NDArray
from sklearn.metrics.pairwise import pairwise_distances
from intercluster import labels_format, satisfies_conditions, density_distance
from intercluster import Condition, LinearCondition
from .decision_set import DecisionSet
from .pruning import greedy

class DBSet(DecisionSet):
    """
    Collection of rules drawn as boxes (rules) around collections of dense points in the dataset.
    """
    def __init__(
        self,
        n_rules,
        n_features,
        epsilon,
        rules_per_point : int = 1
    ):
        """
        Args:
            n_rules (int): Number of rules to use in the decision set.
            
            n_features (int): Number of randomly chosen features to use for each rule.

            epsilon (float): Maximum width between any pair of points in a box.

            rules_per_point (int): Number of random rules to create for each point in the dataset.
        """
        super().__init__()
        self.n_rules = n_rules
        self.n_features = n_features
        self.epsilon = epsilon
        self.rules_per_point = rules_per_point


    def create_rules(self, X : NDArray) -> None:
        """
        Creates rules for the decision set by drawing boxes around dense sets of points 
        in randomly chosen dimensions. 

        Args:
            X (np.ndarray): Input dataset.
            y (List[Set[int]], optional): Target labels.
        Returns:
            decision_set (List[Condition]): List of rules created.
        """        
        n,d = X.shape
        X_sorted = np.argsort(X, axis=0)
        distances = density_distance(X)

        decision_set = []
        for i in range(n):
            for _ in range(self.rules_per_point):
                # Randomly select features to create a box around the point
                features = np.random.choice(d, self.n_features, replace=False)
                point_set = {i}
                satisfies = np.zeros((n, self.n_features), dtype=bool)
                satisfies[i, :] = True

                # Expand the box around the point until no more points can be added
                point_loc = np.where(X_sorted[:, features] == i)[0]
                lower_idx = np.copy(point_loc) - 2 # One step more, since the lower bound is a strict inequality (>).
                lower_max = np.zeros(self.n_features, dtype=bool)
                upper_idx = np.copy(point_loc) + 1
                upper_max = np.zeros(self.n_features, dtype=bool)
                any_movement = True
                while any_movement:
                    any_movement = False
                    # Check each feature to see if we can expand the box
                    for j, f in enumerate(features):
                        feature_vec = X_sorted[:, f]

                        # Move backward until points are no longer within epsilon distance
                        if lower_idx[j] >= 0 and lower_max[j] == False:
                            new_point = feature_vec[lower_idx[j] + 1]
                            satisfies[new_point, j] = True

                            if np.all(satisfies[new_point, :]):
                                new_point_distances = distances[:, new_point]
                                if np.all(new_point_distances[list(point_set)] <= self.epsilon):
                                    point_set.add(new_point)
                                    lower_idx[j] -= 1
                                    any_movement = True
                                else:
                                    lower_max[j] = True
                                    satisfies[new_point, j] = False
                                    lower_idx[j] += 1
                            else:
                                lower_idx[j] -= 1
                                any_movement = True

                        # Move forward until points are no longer within epsilon distance
                        if upper_idx[j] < n and upper_max[j] == False:
                            new_point = feature_vec[upper_idx[j]]
                            satisfies[new_point, j] = True

                            if np.all(satisfies[new_point, :]):
                                new_point_distances = distances[:, new_point]
                                if np.all(new_point_distances[list(point_set)] <= self.epsilon):
                                    point_set.add(new_point)
                                    upper_idx[j] += 1
                                    any_movement = True
                                else:
                                    upper_max[j] = True
                                    satisfies[new_point, j] = False
                                    upper_idx[j] -= 1
                            else:
                                upper_idx[j] += 1
                                any_movement = True

                        #import pdb; pdb.set_trace()

                #if np.sum(satisfies[:,0] & satisfies[:,1], axis=0) == 1:
                #if len(point_set) == 1:
                #    print(point_loc)
                #    print(lower_idx)
                #    print(upper_idx)
                #    print(satisfies[i])
                #    print()

                # Add the conditions to the rule, and the rule to the decision set
                rule = []
                for j, f in enumerate(features):
                    feature_vec = X_sorted[:, f]
                    lower_bound = (
                        X[feature_vec[lower_idx[j]], f]
                        if lower_idx[j] >= 0 else -np.inf
                    )
                    upper_bound = (
                        X[feature_vec[upper_idx[j]], f]
                        if upper_idx[j] < n else np.inf
                    )

                    condition1 = LinearCondition(
                        features=np.array([f]),
                        weights=np.array([1.0]),
                        threshold=lower_bound,
                        direction=1
                    )
                    condition2 = LinearCondition(
                        features=np.array([f]),
                        weights=np.array([1.0]),
                        threshold=upper_bound,
                        direction=-1
                    )
                    rule.append(condition1)
                    rule.append(condition2)

                decision_set.append(rule)

        return decision_set


    def prune_rules(
            self,
            X : NDArray,
            decision_set : List[List[Condition]]
        ) -> List[List[Condition]]:
        """
        Prunes the decision set by removing rules that do not cover any points in the dataset.

        Args:
            X (np.ndarray): Input dataset.
            decision_set (List[List[Condition]]): List of rules to prune.
        Returns:
            pruned_set (List[List[Condition]]): Selected subset of rules.
        """
        data_to_rules_assignment = self.get_data_to_rules_assignment(X, decision_set)
        #print(np.sum(data_to_rules_assignment, axis=0))
        selected_rules = greedy(self.n_rules, data_to_rules_assignment)
        pruned_set = [decision_set[i] for i in selected_rules]
        return pruned_set


    def assign_rules(self, X : NDArray, decision_set : List[List[Condition]]) -> None:
        """
        Assigns clusters labels to the rules in the decision set. Considering the graph 
        where each rule is joined by an edge if they cover a common point, unique labels 
        are assigned to each connected component.
        
        Args:
            X (np.ndarray): Input dataset.
        """
        # NOTE: This is inefficient to compute (already computed it in the rule creation step)
        distances = density_distance(X)
        assignment = self.get_data_to_rules_assignment(X, decision_set)
        edges = []
        for i in range(assignment.shape[1]):
            for j in range(i + 1, assignment.shape[1]):
                # If the two rules satisfy epsilon distance, join them
                first_rule_points = np.where(assignment[:, i])[0]
                second_rule_points = np.where(assignment[:, j])[0]
                if len(first_rule_points) > 0 and len(second_rule_points) > 0:
                    rule_distances = distances[first_rule_points][:, second_rule_points]
                    if np.all(rule_distances <= self.epsilon):
                        edges.append((i, j))

                #if np.any(assignment[:, i] & assignment[:, j]):
                #    edges.append((i, j))

        G = nx.Graph()
        G.add_nodes_from(range(len(decision_set)))
        G.add_edges_from(edges)
        connected_components = list(nx.connected_components(G))
        #print(connected_components)
        decision_set_labels = np.zeros(len(decision_set), dtype=int)
        for i, component in enumerate(connected_components):
            for rule_index in component:
                decision_set_labels[rule_index] = i

        decision_set_labels = labels_format(decision_set_labels)
        return decision_set_labels
    

    def _fitting(
        self,
        X : NDArray,
        y : List[Set[int]] = None,
    ) -> Tuple[List[List[Condition]], List[Set[int]]]:
        """
        Fits a point set to an input dataset by creating rules around individual points.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels.
            
        Returns:
            decision_set (List[Condition]): List of rules.
            
            decision_set_labels (List[int]): List of labels corresponding to each rule.
        """
        decision_set = self.create_rules(X)
        pruned_set = self.prune_rules(X, decision_set)
        #pruned_set = decision_set
        pruned_set_labels = self.assign_rules(X, pruned_set)
        #pruned_set_labels = [{0} for _ in range(len(pruned_set))]
        return pruned_set, pruned_set_labels


    def get_data_to_rules_assignment(
            self,
            X : NDArray,
            decision_set : List[List[Condition]] = None
        ) -> NDArray:
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
            rule_list (List[List[Condition]], optional): List of rules to use for assignment.
        Returns:
            assignment (np.ndarray): n x n_rules boolean matrix with entry (i,j) being True
                if point i is covered by rule j and False otherwise.
        """
        if decision_set is None:
            decision_set = self.decision_set
        assignment = np.zeros((X.shape[0], len(decision_set)), dtype=bool)
        for i, condition_list in enumerate(decision_set):
            data_points_satisfied = satisfies_conditions(X, condition_list)
            assignment[data_points_satisfied, i] = True
        return assignment