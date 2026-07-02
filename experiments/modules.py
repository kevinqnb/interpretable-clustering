import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from typing import Tuple, Dict, Any
from numpy.typing import NDArray
from typing import List, Set, Any
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.objectives import *
from intercluster.utils import (
    labels_format,
    labels_to_assignment,
    unique_labels,
)
from intercluster import Rule


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
        random_seed : int = None,
        name : str = 'KMeans'
    ):
        super().__init__(name)
        self.n_clusters = n_clusters
        self.clustering = KMeans(
            n_clusters=n_clusters,
            random_state=random_seed
        )
        self.fitted = False
        self.assignment = None
        self.centers = None
        self.labels = None
        self.max_rule_length = np.nan
        self.sum_rule_length = np.nan
        self.weighted_average_rule_length = np.nan
        
    def assign(self, X : NDArray) -> NDArray:
        """
        Fits the KMeans model and returns the cluster assignment.
        
        Args:
            X (np.ndarray): Data matrix.
            
        Returns:
            assignment (np.ndarray): Cluster assignment boolean array of size n x k
                with entry (i,j) being True if point i belongs to cluster j and False otherwise.
        """
        if not self.fitted:
            self.clustering.fit(X)
            self.clustering.cluster_centers_ = self.clustering.cluster_centers_.copy()
            self.labels = labels_format(self.clustering.labels_)
            self.assignment = labels_to_assignment(
                self.labels,
                n_labels = self.n_clusters
            )
            self.centers = self.clustering.cluster_centers_
            self.fitted = True
        
        return self.assignment
    
    
####################################################################################################


class DBSCANBase(Baseline):
    """
    Baseline DBSCAN clustering method.
    
    Args:
        eps (float): The maximum distance between two samples for one to be considered 
            as in the neighborhood of the other.
        
        min_samples (int): The number of samples in a neighborhood for a point to be 
            considered as a core point.
        
        name (str, optional): Name of the baseline method. Defaults to 'DBSCAN'.
    """
    def __init__(
        self,
        eps : float,
        n_core : int,
        name : str = 'DBSCAN'
    ):
        super().__init__(name)
        self.eps = eps
        self.n_core = n_core
        self.clustering = DBSCAN(eps=eps, min_samples=n_core)
        self.fitted = False
        self.assignment = None
        self.max_rule_length = np.nan
        self.sum_rule_length = np.nan
        self.weighted_average_rule_length = np.nan

        
    def assign(self, X : NDArray) -> NDArray:
        """
        Fits the DBSCAN model and returns the cluster assignment.
        
        Args:
            X (np.ndarray): Data matrix.
            
        Returns:
            assignment (np.ndarray): Cluster assignment boolean array of size n x k
                with entry (i,j) being True if point i belongs to cluster j and False otherwise.
        """
        if not self.fitted:
            self.clustering.fit(X)
            self.labels = labels_format(self.clustering.labels_)
            n_unique = len(unique_labels(self.labels))
            self.assignment = labels_to_assignment(
                self.labels,
                n_labels = n_unique,
            )
            self.fitted = True
        
        return self.assignment
    

####################################################################################################


class AgglomerativeBase(Baseline):
    """
    Baseline Agglomerative clustering method.
    
    Args:
        n_clusters (int): Number of clusters.

        linkage (str): Linkage criterion to use.
        
        name (str, optional): Name of the baseline method. Defaults to 'DBSCAN'.
    """
    def __init__(
        self,
        n_clusters : int,
        linkage : str = 'single',
        name : str = 'Agglomerative'
    ):
        super().__init__(name)
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.fitted = False
        self.assignment = None
        self.max_rule_length = np.nan
        self.sum_rule_length = np.nan
        self.weighted_average_rule_length = np.nan

        
    def assign(self, X : NDArray) -> NDArray:
        """
        Fits the Agglomerative model and returns the cluster assignment.
        
        Args:
            X (np.ndarray): Data matrix.
            
        Returns:
            assignment (np.ndarray): Cluster assignment boolean array of size n x k
                with entry (i,j) being True if point i belongs to cluster j and False otherwise.
        """
        if not self.fitted:
            self.clustering.fit(X)
            self.labels = labels_format(self.clustering.labels_)
            n_unique = len(unique_labels(self.labels))
            self.assignment = labels_to_assignment(
                self.labels,
                n_labels = n_unique,
            )
            self.fitted = True
        
        return self.assignment


####################################################################################################


class DecisionTreeMod(Module):
    """
    Experiment module for a decision tree clustering method.
    
    Args:
        model (Any): Tree model. 

        fitting_params (Dict[str, Any]): Dictionary of parameters to pass to the tree model 
            prior to fitting. 
        
        name (str, optional): Name of the module. Defaults to 'Decision-Tree'.
    """
    def __init__(
        self,
        model : Any,
        fitting_params : Dict[str, Any] = None,
        name : str = 'Decision-Tree'
    ):
        self.model = model
        self.fitting_params = fitting_params
        super().__init__(name)
        self.reset()


    def reset(self):
        """
        Resets experiments by returning parametrs to their default values.
        """
        self.n_rules = np.nan
        self.max_rule_length = np.nan
        self.sum_rule_length = np.nan
        self.weighted_average_rule_length = np.nan
        self.tree = None

    
    def update_fitting_params(self, fitting_params : Dict[str, Any]):
        """
        Updates the fitting parameters for the model.
        
        Args:
            kwargs (Dict[str, Any]): Dictionary of parameters to update.
        """
        self.fitting_params = fitting_params


    def fit(self, X : NDArray, y : NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Increases the number of rules by one and fits the model.
        
        Args:
            X (np.ndarray): Data matrix.
            
            y (np.ndarray, optional): Data labels.
        
        Returns:
            data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is `True` if 
                data point i is assigned to rule j and `False` otherwise.

            rule_to_cluster_assignment (np.ndarray): Size (r x k) boolean array where entry (i,j) is 
                `True` if rule i is assigned to cluster j and `False` otherwise. Each rule must 
                be assigned to a single cluster.

            data_to_cluster_assignment (np.ndarray): Size (n x k) boolean array where entry (i,j) is 
                `True` if point i is assigned to cluster j and `False` otherwise. Data points may be 
                assigned to multiple clusters. 
        """
        n_unique = len(unique_labels(y))
        # Fit the model with the current number of rules
        self.tree = self.model(**self.fitting_params)
        self.tree.fit(X, y)
        tree_labels = self.tree.predict(X)
        tree_leaf_labels = self.tree.get_leaf_labels()
        tree_rule_assignment = labels_to_assignment(
            tree_leaf_labels, n_labels = n_unique
        )
        tree_data_to_rule_assignment = self.tree.get_data_to_rules_assignment(X)
        tree_data_to_cluster_assignment = labels_to_assignment(
            tree_labels, n_labels = n_unique
        )

        # A few data things to record:
        self.n_rules = self.tree.leaf_count
        self.max_rule_length = self.tree.depth
        self.sum_rule_length = self.tree.get_sum_of_depths()
        self.weighted_average_rule_length = self.tree.get_weighted_average_depth(X)

        return (
            tree_data_to_rule_assignment,
            tree_rule_assignment,
            tree_data_to_cluster_assignment
        )
    
    def predict(self, X : NDArray) -> List[Set[int]]:
        """
        Predicts cluster assignments for new data points.
        
        Args:
            X (np.ndarray): Data matrix.
        Returns:
            labels (List[Set[int]]): Length n list of predicted labels.
        """
        if self.tree is None:
            raise ValueError("Model has not been fit yet.")
        return self.tree.predict(X)


####################################################################################################


class DecisionSetMod(Module):
    """
    Experiment module for a decision tree clustering method.
    
    Args:
        model (Any): Decision Set Model to use.

        fitting_params (Dict[str, Any]): Dictionary of parameters to pass to the tree model 
            prior to fitting.

        rules (List[List[Condition]], optional): Pre-mined rules to use. If None, rules will be mined
            using the rule_miner. Defaults to None.

        rule_labels (List[Set[int]], optional): Pre-mined rule labels to use. If None, rule labels
            will be mined using the rule_miner. Defaults to None.

        rule_miner (Any): Rule miner used to generate the rules.
        
        name (str, optional): Name of the module. Defaults to 'Decision-Tree'.
    """
    def __init__(
        self,
        model : Any,
        fitting_params : Dict[str, Any] = None,
        rules : List[Rule] = None,
        rule_labels : List[Set[int]] = None,
        name : str = 'Decision-Set'
    ):
        self.model = model
        self.fitting_params = fitting_params
        self.rules = rules
        self.rule_labels = rule_labels
        super().__init__(name)
        self.reset()
    

    def reset(self):
        """
        Resets experiments by giving previous outputs their default values.
        """
        self.n_rules = np.nan
        self.max_rule_length = np.nan
        self.sum_rule_length = np.nan
        self.weighted_average_rule_length = np.nan
        self.lambda_val = np.nan
        self.dset = None


    def update_fitting_params(self, fitting_params : Dict[str, Any] = None):
        """
        Updates the fitting parameters for the model.
        
        Args:
            kwargs (Dict[str, Any]): Dictionary of parameters to update.
        """
        self.fitting_params = fitting_params


    def fit(self, X : NDArray, y : NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Increases the number of rules by one and fits the model.
        
        Args:
            X (np.ndarray): Data matrix.
            
            y (np.ndarray, optional): Data labels.
        
        Returns:
            data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is `True` if 
                data point i is assigned to rule j and `False` otherwise.

            rule_to_cluster_assignment (np.ndarray): Size (r x k) boolean array where entry (i,j) is 
                `True` if rule i is assigned to cluster j and `False` otherwise. Each rule must 
                be assigned to a single cluster.

            data_to_cluster_assignment (np.ndarray): Size (n x k) boolean array where entry (i,j) is 
                `True` if point i is assigned to cluster j and `False` otherwise. Data points may be 
                assigned to multiple clusters. 
        """
        # Fit the model with the current number of rules
        self.dset = self.model(
            **(self.fitting_params | {'rules' : self.rules, 'rule_labels' : self.rule_labels})
        )

        self.dset.fit(X, y)
        self.lambda_val = self.dset.lambda_val if hasattr(self.dset, 'lambda_val') else np.nan
        dset_labels = self.dset.predict(X)
        n_unique = len(unique_labels(y))

        dset_data_to_rule_assignment = self.dset.get_data_to_rules_assignment(X)
        dset_rule_to_cluster_assignment = self.dset.get_rules_to_clusters_assignment(n_labels = n_unique)
        dset_data_to_cluster_assignment = labels_to_assignment(
            dset_labels, n_labels = n_unique
        )

        self.n_rules = len(self.dset.decision_set)
        self.max_rule_length = self.dset.max_rule_length
        self.sum_rule_length = self.dset.get_sum_of_rule_lengths()
        self.weighted_average_rule_length = self.dset.get_weighted_average_rule_length(X)

        return (
            dset_data_to_rule_assignment,
            dset_rule_to_cluster_assignment,
            dset_data_to_cluster_assignment
        )
    

    def predict(self, X : NDArray) -> List[Set[int]]:
        """
        Predicts cluster assignments for new data points.
        
        Args:
            X (np.ndarray): Data matrix.
        Returns:
            labels (List[Set[int]]): Length n list of predicted labels.
        """
        if self.dset is None:
            raise ValueError("Model has not been fit yet.")
        return self.dset.predict(X)


####################################################################################################


def aggregate_trials(trial_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregates a list of per-trial result dictionaries (all sharing the same keys,
    e.g. one dict of measurement values per random seed) into per-key summary
    statistics.

    Used for modules whose fitted solution has inherent randomness (e.g. IDS,
    ExplanationTree, DecisionTree, ShallowTree), where a single fit is not
    representative and results should instead be reported across several
    random trials.

    Args:
        trial_results (List[Dict[str, Any]]): Length n_trials list of result
            dictionaries, each mapping metric name to a scalar value.

    Returns:
        aggregated (Dict[str, Dict[str, Any]]): Dictionary mapping each metric name
            to {'mean': float, 'std': float, 'values': List[Any]} computed across
            trials. Non-numeric or NaN entries are passed through in 'values' and
            excluded from the mean/std computation.
    """
    if not trial_results:
        return {}

    aggregated = {}
    for key in trial_results[0].keys():
        values = [t[key] for t in trial_results]
        numeric_values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        aggregated[key] = {
            'mean': float(np.mean(numeric_values)) if numeric_values else np.nan,
            'std': float(np.std(numeric_values)) if numeric_values else np.nan,
            'values': values,
        }
    return aggregated


####################################################################################################