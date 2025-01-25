import numpy as np
from sklearn.cluster import KMeans
from ExKMC.Tree import Tree as ExTree
from typing import Tuple, Dict, Any
from numpy.typing import NDArray
from intercluster.rules import *
from intercluster.pruning import *
from intercluster.utils import labels_to_assignment, kmeans_cost


####################################################################################################


class Baseline:
    """
    Experiment module for a non-variable baseline method. 
    
    Args:
        name (str, optional): Name of the baseline method. Defaults to Class name.
    """
    def __init__(self, name : str = None):
        self.name = name or self.__class__.__name__
    
    def assign(self):
        """
        Assigns data points to clusters.
        """
        pass


class Module:
    """
    Experiment module for a clustering method in which the number of rules or 
    depth of rules may be varied. This is designed to be dynamic in the sense that 
    the number of rules or depth of rules can be increased in steps. For certain types of 
    modules, this means we don't have to retrain an entire model for every step in 
    an experiment.
    
    Args:
        min_rules (int): Minimum number of rules.
        
        min_depth (int): Minimum depth of rules.
        
        name (str, optional): Name of the module. Defaults to Class name.
        
    Attributes:
        n_rules (int): Current number of rules.
        
        n_depth (int): Current depth of rules.
    """
    def __init__(
        self, 
        min_rules : int = 1,
        min_depth : int = 1,
        name : str = None
    ):
        self.min_rules = min_rules
        self.min_depth = min_depth
        self.n_rules = min_rules
        self.n_depth = min_depth
        self.name = name or self.__class__.__name__
    
    def step_num_rules(self):
        """
        Increases the number of rules by one and fits the model.
        """
        self.n_rules += 1
        # fit model with new number of rules
        pass
    
    def step_depth(self):
        """
        Increases the depth of rules by one and fits the model.
        """
        self.n_depth += 1
        # fit model with new depth
        pass
    
    def reset(self):
        """
        Resets the module to its initial state with minimal number of rules and depth.
        """
        self.n_rules = self.min_rules
        self.n_depth = self.min_depth
    
    
####################################################################################################

    
class KMeansBase(Baseline):
    """
    Baseline KMeans clustering method.
    
    Args:
        n_clusters (int): Number of clusters.
        
        random_seed (int): Random seed. Defaults to None.                                            
        
        name (str, optional): Name of the baseline method. Defaults to 'KMeans'.
    """
    def __init__(
        self,
        n_clusters : int,
        name : str = 'KMeans'
    ):
        super().__init__(name)
        self.n_clusters = n_clusters
        self.clustering = KMeans(
            n_clusters=n_clusters
        )
        
    def assign(self, X : NDArray) -> Tuple[NDArray, NDArray]:
        """
        Fits the KMeans model and returns the cluster assignment.
        
        Args:
            X (np.ndarray): Data matrix.
            
        Returns:
            assignment (np.ndarray): Cluster assignment boolean array of size n x k
                with entry (i,j) being True if point i belongs to cluster j and False otherwise.
            
            centers (np.ndarray): Size k x d array of cluster centers. 
        """
        self.clustering.fit(X)
        self.assignment = labels_to_assignment(self.clustering.labels_)
        self.centers = self.clustering.cluster_centers_
        return self.assignment, self.centers
    
    
####################################################################################################

    
class ExkmcMod(Module):
    """
    Experiment module for the ExKMC clustering method. For more information on 
    ExKMC please see the ExKMC package:
    (https://github.com/navefr/ExKMC)
    
    NOTE: ExKMC does not allow depth control, so this module only allows for 
    variability in the number of rules used.
    
    Args:
        n_clusters (int): Number of clusters.
        
        base_tree (str): Determines if the first n_clusters number of nodes in the tree 
            are initialized with IMM or not. Options are 'IMM' or 'NONE', default is 'IMM'.
        
        kmeans_model (Any): Pretrained SKLearn KMeans model.
        
        min_rules (int): Minimum number of rules.
        
        min_depth (int): Minimum depth of rules.
        
        max_rules (int): Maximum number of rules.
        
        max_depth (int): Maximum depth of rules.
        
        name (str, optional): Name of the module. Defaults to 'ExKMC'.
        
    Attributes:
        n_rules (int): Current number of rules.
        
        n_depth (int): Current depth of rules.
    """
    def __init__(
        self,
        n_clusters : int,
        kmeans_model : Any,
        base_tree : str,
        min_rules : int,
        name : str = 'ExKMC'
    ):
        super().__init__(min_rules = min_rules, name = name)
        self.n_clusters = n_clusters
        self.base_tree = base_tree
        self.kmeans_model = kmeans_model
    
    def step_num_rules(self, X : NDArray, y : NDArray = None) -> Tuple[NDArray, NDArray]:
        """
        Increases the number of rules by one and fits the model.
        
        Args:
            X (np.ndarray): Data matrix.
            
            y (np.ndarray, optional): Data labels. Defaults to None.
            
        Returns:
            assignment (np.ndarray): Cluster assignment boolean array of size n x k
                with entry (i,j) being True if point i belongs to cluster j and False otherwise.
            
            centers (np.ndarray): Size k x d array of cluster centers. 
        """
        tree = ExTree(self.n_clusters, max_leaves = self.n_rules, base_tree = self.base_tree)
        labels = tree.fit_predict(X, kmeans=self.kmeans_model)
        assignment = labels_to_assignment(labels)
        centers = tree.all_centers
        self.n_rules += 1
        self.n_depth = tree._max_depth()
        return assignment, centers
    
    

####################################################################################################


class ForestMod(Module):
    """
    Experiment module for random decision forests.
    This method trains a random forest model, and then prunes the set of leaf node rules 
    found in the forest with the help of an input set of clustering labels. 
    
    Args:
        forest_model (Any): Random forest model.
        
        forest_params (dict[str, Any]): Parameters for the random forest model.
        
        clustering (Baseline): Trained Baseline clustering model.
        
        prune_params (dict[str, Any]): Parameters for the rule pruning method. If None, 
            no pruning will be done. 
        
        min_rules (int): Minimum number of rules.
        
        min_depth (int): Minimum depth of rules.
        
        max_rules (int): Maximum number of rules.
        
        name (str, optional): Name of the module. Defaults to 'Forest'.
        
    Attributes:
        n_rules (int): Current number of rules.
        
        n_depth (int): Current depth of rules.
        
        forest (Any): The fitted random forest model.
        
        rule_labels (np.ndarray): Array of rule labels.
        
        data_labels (np.ndarray): Array of data labels.
        
        centers (np.ndarray): Array of cluster centers.
    """
    def __init__(
        self,
        forest_model : Any,
        forest_params : Dict[str, Any],
        clustering : Baseline,
        prune_params : Dict[str, Any],
        min_rules : int,
        min_depth : int,
        max_rules : int,
        max_depth : int,
        name = 'Forest'
    ):
        self.forest_model = forest_model
        self.forest_params = forest_params
        self.clustering = clustering
        self.prune_params = prune_params
        super().__init__(min_rules, min_depth, name)
        self.max_rules = max_rules
        self.max_depth = max_depth
        
        self.forest = None
        self.points_to_rules = None
        self.rule_assignment = None
        self.rule_labels = None
        self.data_labels = None
        self.centers = self.clustering.centers
        
        
    def reset(self):
        """
        Resets the module to its initial state with minimal number of rules and depth.
        Also resets the fitted forest model, rule labels, data labels, and centers.
        """
        self.n_rules = self.min_rules
        self.n_depth = self.min_depth
        self.forest = None
        self.rule_labels = None
        self.data_labels = None
        self.centers = None
        
    
    def assign(self, X : NDArray) -> Tuple[NDArray, NDArray]:
        """
        Prunes the model (if necessary) and returns the cluster assignment.
        
        Args:
            X (np.ndarray): Data matrix.
            
        Returns:
            assignment (np.ndarray): Cluster assignment boolean array of size n x k
                with entry (i,j) being True if point i belongs to cluster j and False otherwise.
            
            centers (np.ndarray): Size k x d array of cluster centers.
        """
        if self.prune_params is not None:
            def clustering_objective(S):
                A = self.points_to_rules[:, S]
                B = self.rule_assignment[S, :]
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
            
            A = self.points_to_rules[:, selected_rules]
            B = self.rule_assignment[selected_rules, :]
            assignment = np.dot(A,B)
        else:
            A = self.points_to_rules
            B = self.rule_assignment
            assignment = np.dot(A,B)
            
        return assignment, self.centers
    
    
    def step_num_rules(self, X : NDArray, y : NDArray) -> Tuple[NDArray, NDArray]:
        """
        Increases the number of rules by one and fits the model.
        
        Args:
            X (np.ndarray): Data matrix.
            
            y (np.ndarray, optional): Data labels.
        
        Returns:
            assignment (np.ndarray): Cluster assignment boolean array of size n x k
                with entry (i,j) being True if point i belongs to cluster j and False otherwise.
            
            centers (np.ndarray): Size k x d array of cluster centers.
        """
        if self.forest is None:
            self.forest = self.forest_model(
                **self.forest_params
            )
            self.forest.fit(X,  y)
            
            data_to_rules_labels = self.forest.predict(X, rule_labels = True)
            self.points_to_rules = labels_to_assignment(data_to_rules_labels,
                                                        k = len(self.forest.decision_set))
            self.data_labels = [[l] for l in y]
            
            self.rule_labels = self.forest.decision_set_labels
            self.rule_assignment = labels_to_assignment(self.rule_labels,
                                                        k = self.clustering.n_clusters)
            self.rule_labels = self.forest.decision_set_labels
            self.n_depth = self.forest.depth
            
        assignment,centers = self.assign(X)
        self.n_rules += 1
        return assignment, centers
    
    
####################################################################################################