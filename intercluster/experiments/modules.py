import numpy as np
from sklearn.cluster import KMeans
from ExKMC.Tree import Tree as ExTree
from typing import Tuple, Dict, Any
from numpy.typing import NDArray
from intercluster.rules import *
from intercluster.pruning import *
from intercluster.utils import labels_to_assignment, update_centers


####################################################################################################


class Baseline:
    """
    Experiment module for a non-variable, baseline method. 
    
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
    Experiment module for a clustering method run over selected parameter settings.
    
    Args:        
        name (str, optional): Name of the module. Defaults to Class name.
        
    """
    def __init__(
        self, 
        name : str = None
    ):
        self.name = name or self.__class__.__name__
    
    def reset(self):
        """
        Resets the module to its initial state.
        """
        pass
    
    
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
        self.assignment = labels_to_assignment(self.clustering.labels_, n_labels = self.n_clusters)
        self.centers = self.clustering.cluster_centers_
        self.max_rule_length = np.nan
        return self.assignment, self.centers
    
    
####################################################################################################


class IMMBase(Baseline):
    """
    Baseline IMM clustering method.
    
    Args:
        n_clusters (int): Number of clusters.
        
        kmeans_model (Any): Pretrained SKLearn KMeans model.
        
        name (str, optional): Name of the baseline method. Defaults to 'KMeans'.
    """
    def __init__(
        self,
        n_clusters : int,
        kmeans_model : Any,
        name : str = 'IMM'
    ):
        self.n_clusters = n_clusters
        self.kmeans_model = kmeans_model
        super().__init__(name)
        
    def assign(self, X : NDArray) -> Tuple[NDArray, NDArray]:
        """
        Fits the IMM model and returns the cluster assignment.
        
        Args:
            X (np.ndarray): Data matrix.
            
        Returns:
            assignment (np.ndarray): Cluster assignment boolean array of size n x k
                with entry (i,j) being True if point i belongs to cluster j and False otherwise.
            
            centers (np.ndarray): Size k x d array of cluster centers. 
        """
        tree = ExTree(self.n_clusters, max_leaves = self.n_clusters, base_tree = 'IMM')
        exkmc_labels = tree.fit_predict(X, kmeans=self.kmeans_model)
        assignment = labels_to_assignment(exkmc_labels, n_labels = self.n_clusters)
        #centers = tree.all_centers
        updated_centers = update_centers(X, assignment)
        self.max_rule_length = tree._max_depth()
        return assignment, updated_centers
    
    
####################################################################################################

    
class ExkmcMod(Module):
    """
    Experiment module for the ExKMC clustering method. For more information on 
    ExKMC please see the ExKMC package:
    (https://github.com/navefr/ExKMC)
    
    NOTE: ExKMC does not allow depth control, so this module only allows for 
    variability in the number of rules used... that is until I get my own version of ExKMC working.
    
    Args:
        n_clusters (int): Number of clusters.
        
        base_tree (str): Determines if the first n_clusters number of nodes in the tree 
            are initialized with IMM or not. Options are 'IMM' or 'NONE', default is 'IMM'.
        
        kmeans_model (Any): Pretrained SKLearn KMeans model.
        
        min_rules (int): Minimum number of rules.
        
        name (str, optional): Name of the module. Defaults to 'ExKMC'.
        
    Attributes:
        n_rules (int): Current number of rules. Defaults to min rules before running any experiment.
        
        max_rule_length (int): Current maximum rule length (equivalent to tree depth). 
            Defaults to -1 before running any experiment.
    """
    def __init__(
        self,
        n_clusters : int,
        kmeans_model : Any,
        base_tree : str,
        min_rules : int,
        name : str = 'ExKMC'
    ):
        super().__init__(name = name)
        self.n_clusters = n_clusters
        self.kmeans_model = kmeans_model
        self.base_tree = base_tree
        self.min_rules = min_rules
        self.reset()
        
        
    def reset(self):
        """
        Resets experiments by returning parametrs to their default values.
        """
        self.n_rules = self.min_rules
        self.max_rule_length = -1
        
    
    def step_num_rules(self, X : NDArray, y : NDArray = None) -> Tuple[NDArray, NDArray]:
        """
        Increases the number of rules by one and re-fits the model.
        
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
        assignment = labels_to_assignment(labels, n_labels = self.n_clusters)
        #centers = tree.all_centers
        updated_centers = update_centers(X, assignment)
        self.n_rules += 1
        self.max_rule_length = tree._max_depth()
        return assignment, updated_centers   
    

####################################################################################################


class DecisionSetMod(Module):
    """
    Experiment module for decision sets.
    
    Args:
        decision_set_model (Any): Decision set model.
        
        decision_set_params (dict[str, Any]): Parameters for the decision set model.
        
        clustering (Baseline): Trained Baseline clustering model.
        
        prune_params (dict[str, Any]): Parameters for the rule pruning method. If None, 
            no pruning will be done. 
        
        min_rules (int): Minimum number of rules.
        
        min_frac_cover (float): Minimum fraction of data points to cover.
        
        name (str, optional): Name of the module. Defaults to 'Decision-Set'.
        
    Attributes:
        model (Any): The fitted decision set model.
        
        n_rules (int): Current number of rules.
        
        frac_cover (float): Current coverage threshold.
        
        max_rule_length (int): Current maximum rule length among all collected rules.
        
        centers (np.ndarray): Array of cluster centers.
    """
    def __init__(
        self,
        decision_set_model : Any,
        decision_set_params : Dict[str, Any],
        clustering : Baseline,
        prune_params : Dict[str, Any],
        min_rules : int = None,
        min_frac_cover : int = None,
        name = 'Decision-Set'
    ):
        self.decision_set_model = decision_set_model
        self.decision_set_params = decision_set_params
        self.clustering = clustering
        self.prune_params = prune_params
        self.min_rules = min_rules
        self.min_frac_cover = min_frac_cover
        super().__init__(name)
        
        self.reset()
        self.centers = self.clustering.centers
        
        
    def reset(self):
        """
        Resets experiments by returning parametrs to their default values.
        """
        self.model = None
        self.n_rules = self.min_rules
        self.frac_cover = self.min_frac_cover
        self.max_rule_length = np.nan
    
    
    def step_num_rules(
        self,
        X : NDArray,
        y : NDArray,
        step_size : int = 1
    )-> Tuple[NDArray, NDArray]:
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
        if self.model is None:
            self.model = self.decision_set_model(
                **self.decision_set_params
            )
            self.model.fit(X,  y)
            
            if hasattr(self.model, "max_rule_length"):
                self.max_rule_length = self.model.max_rule_length
            elif hasattr(self.model, "depth"):
                self.max_rule_length = self.model.depth
            elif hasattr(self.model, "num_conditions"):
                self.max_rule_length = self.model.num_conditions
            else:
                raise ValueError("Decision set model has no rule length parameter.")

        if self.prune_params is not None:
            self.model.prune(n_rules = self.n_rules, **self.prune_params)
            assignment = labels_to_assignment(
                self.model.pruned_predict(X, rule_labels = False),
                n_labels = self.clustering.n_clusters
            )
        else:
            assignment = labels_to_assignment(
                self.model.predict(X, rule_labels = False),
                n_labels = self.clustering.n_clusters
            )
        updated_centers = update_centers(X, assignment)
        
        self.n_rules += step_size
        return assignment, updated_centers
    
    
    def step_coverage(
        self,
        X : NDArray,
        y : NDArray,
        step_size : float = 0.05
    ) -> Tuple[NDArray, NDArray]:
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
        if self.model is None:
            self.model = self.decision_set_model(
                **self.decision_set_params
            )
            self.model.fit(X,  y)
            
            if hasattr(self.model, "max_rule_length"):
                self.max_rule_length = self.model.max_rule_length
            elif hasattr(self.model, "depth"):
                self.max_rule_length = self.model.depth
            elif hasattr(self.model, "num_conditions"):
                self.max_rule_length = self.model.num_conditions
            else:
                raise ValueError("Decision set model has no rule length parameter.")

        assignment = None
        updated_centers = None
        if self.prune_params is not None:
            self.model.prune(frac_cover = self.frac_cover, **self.prune_params)
            if self.model.prune_status:
                assignment = labels_to_assignment(
                    self.model.pruned_predict(X, rule_labels = False),
                    n_labels = self.clustering.n_clusters
                )
                updated_centers = update_centers(X, assignment)
        else:
            assignment = labels_to_assignment(
                self.model.predict(X, rule_labels = False),
                n_labels = self.clustering.n_clusters
            )
            updated_centers = update_centers(X, assignment)
        
        self.frac_cover += step_size
        if self.frac_cover > 1:
            raise ValueError("Stepped beyond 100 percent coverage.")
        return assignment, updated_centers
    
    
####################################################################################################