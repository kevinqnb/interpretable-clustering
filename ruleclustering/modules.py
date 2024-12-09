import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from ExKMC.Tree import Tree as ExTree
from .tree import *
from .rule_clustering import *
from .decision_sets import *
from .rule_pruning import *
from .utils import *


####################################################################################################


class Baseline:
    def __init__(self, name = None):
        self.name = name or self.__class__.__name__
    
    def assign(self):
        pass


class Module:
    def __init__(
        self, 
        min_rules,
        min_depth,
        name = None
    ):
        self.min_rules = min_rules
        self.min_depth = min_depth
        self.n_rules = min_rules
        self.n_depth = min_depth
        self.name = name or self.__class__.__name__
    
    def step_num_rules(self):
        self.n_rules += 1
        # fit model with new number of rules
        pass
    
    def step_depth(self):
        self.n_depth += 1
        # fit model with new depth
        pass
    
    def reset(self):
        self.n_rules = self.min_rules
        self.n_depth = self.min_depth
    
    
####################################################################################################

    
class KMeansBase(Baseline):
    def __init__(self, n_clusters, seed, name = 'KMeans'):
        super().__init__(name)
        self.clustering = KMeans(n_clusters=n_clusters,
                        random_state=seed
                    )
    def assign(self, X):
        self.clustering.fit(X)
        assignment = labels_to_assignment(self.clustering.labels_)
        centers = self.clustering.cluster_centers_
        return assignment, centers
    
    
####################################################################################################

    
class ExKMCMod(Module):
    """
    NOTE: ExKMC does not allow depth control.
    """
    def __init__(
        self,
        n_clusters,
        base_tree,
        kmeans,
        min_rules,
        max_rules,
        min_depth,
        max_depth,
        name = 'ExKMC'
    ):
        super().__init__(min_rules, max_rules, min_depth, max_depth, name)
        self.n_clusters = n_clusters
        self.base_tree = base_tree
        self.kmeans = kmeans
    
    def step_num_rules(self, X):
        tree = ExTree(self.n_clusters, max_leaves = self.n_rules, base_tree = self.base_tree)
        labels = tree.fit_predict(X, self.kmeans)
        assignment = labels_to_assignment(labels)
        centers = tree.all_centers
        self.n_rules += 1
        return assignment, centers
    
    
####################################################################################################
    
    
class CentroidMod(Module):
    def __init__(
        self,
        tree_params,
        clusterer,
        cluster_params,
        min_rules,
        min_depth,
        name = 'Centroid'
    ):
    
        self.tree_params = tree_params
        self.clusterer = clusterer
        self.cluster_params = cluster_params
        super().__init__(min_rules, min_depth, name)
        self.tree = None
        
    def reset(self):
        self.n_rules = self.min_rules
        self.n_depth = self.min_depth
        self.tree = None
    
    def step_num_rules(self, X):
        if self.tree is None:
            self.tree = CentroidTree(
                **self.tree_params
            )
            # SHOULD THIS BE n_rules - 1???
            self.tree.fit(X, iterative = True, init_steps = self.n_rules - 1)
            
        self.tree.fit_step()
        clustering = self.clusterer(
            rules = self.tree,
            **self.cluster_params
        )
        clustering.fit(X, fit_rules = False)
        assignment = clustering.predict(X)
        centers = clustering.centers
        self.n_rules += 1
        return assignment, centers
    
    
    def step_depth(self, X):
        original_depth = self.tree_params['max_depth']
        self.tree_params['max_depth'] = self.n_depth
        self.tree = CentroidTree(
            **self.tree_params,
        )
        self.tree_params['max_depth'] = original_depth
        
        self.tree.fit(X)
        clustering = self.clusterer(
            rules = self.tree,
            **self.cluster_params
        )
        clustering.fit(X, fit_rules = False)
        assignment = clustering.predict(X)
        centers = clustering.centers
        self.n_depth += 1
        return assignment, centers
        
        
####################################################################################################


class ForestMod(Module):
    def __init__(
        self,
        forest_params,
        clusterer,
        cluster_params,
        assignment_method,
        prune_params,
        min_rules,
        max_rules,
        min_depth,
        name = 'Forest'
    ):
        self.forest_params = forest_params
        self.clusterer = clusterer
        self.cluster_params = cluster_params
        self.assignment_method = assignment_method
        self.prune_params = prune_params
        super().__init__(min_rules, min_depth, name)
        self.max_rules = max_rules
        
        self.forest = None
        self.rule_labels = None
        self.data_labels = None
        self.centers = None
        
        
    def reset(self):
        self.n_rules = self.min_rules
        self.n_depth = self.min_depth
        self.forest = None
        self.rule_labels = None
        self.data_labels = None
        self.centers = None
    
    
    def step_num_rules(self, X):
        if self.forest is None:
            self.forest = DecisionForest(
                **self.forest_params
            )
            self.forest.fit(X)
            
            self.clustering = self.clusterer(
                rules = self.forest,
                **self.cluster_params
            )
            self.clustering.fit(X, fit_rules = False)
            self.rule_labels = assignment_to_labels(self.clustering.rule_assignment)
            full_assignment = self.clustering.predict(X, assignment_method = self.assignment_method)
            self.data_labels = assignment_to_labels(full_assignment)
            self.centers = self.clustering.centers
            
        def clustering_objective(S):
            A = self.clustering.points_to_rules[:, S]
            B = self.clustering.rule_assignment[S, :]
            pruned_assignment = np.dot(A,B)
            return kmeans_cost(X, pruned_assignment, self.centers, normalize = False)
            
        selected_rules = prune_with_grid_search(
            q = self.n_rules,
            data_labels = self.data_labels,
            rule_labels = self.rule_labels,
            rule_covers_dict = self.forest.covers,
            objective = clustering_objective,
            **self.prune_params
        )
        
        A = self.clustering.points_to_rules[:, selected_rules]
        B = self.clustering.rule_assignment[selected_rules, :]
        assignment = np.dot(A,B)
        self.n_rules += 1
        return assignment, self.centers
    
    
    def step_depth(self, X):
        original_depth = self.forest_params['max_depth']
        self.forest_params['max_depth'] = self.n_depth
        self.forest = DecisionForest(
            **self.forest_params,
        )
        self.forest_params['max_depth'] = original_depth
        self.forest.fit(X)
        
        self.clustering = self.clusterer(
            rules = self.forest,
            **self.cluster_params
        )
        self.clustering.fit(X, fit_rules = False)
        self.rule_labels = assignment_to_labels(self.clustering.rule_assignment)
        full_assignment = self.clustering.predict(X, assignment_method = self.assignment_method)
        self.data_labels = assignment_to_labels(full_assignment)
        self.centers = self.clustering.centers
            
        def clustering_objective(S):
            A = self.clustering.points_to_rules[:, S]
            B = self.clustering.rule_assignment[S, :]
            pruned_assignment = np.dot(A,B)
            return kmeans_cost(X, pruned_assignment, self.centers, normalize = False)
            
        selected_rules = prune_with_grid_search(
            q = self.max_rules,
            data_labels = self.data_labels,
            rule_labels = self.rule_labels,
            rule_covers_dict = self.forest.covers,
            objective = clustering_objective,
            **self.prune_params
        )
        
        A = self.clustering.points_to_rules[:, selected_rules]
        B = self.clustering.rule_assignment[selected_rules, :]
        assignment = np.dot(A,B)
        self.n_depth += 1
        return assignment, self.centers