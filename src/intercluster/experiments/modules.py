import numpy as np
from sklearn.cluster import KMeans
from ExKMC.Tree import Tree as ExTree
from typing import Tuple, Dict, Any
from numpy.typing import NDArray
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.pruning import *
from intercluster.utils import labels_format, labels_to_assignment, update_centers, outlier_mask


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
        self.fitted = False
        self.assignment = None
        self.original_centers = None
        self.centers = None
        self.max_rule_length = np.nan
        self.weighted_average_rule_length = np.nan
        
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
        if not self.fitted:
            self.clustering.fit(X)
            clustering_labels = labels_format(self.clustering.labels_)
            self.assignment = labels_to_assignment(clustering_labels, n_labels = self.n_clusters)
            self.centers = self.clustering.cluster_centers_
            self.original_centers = self.centers
            self.fitted = True
        
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
        self.original_centers = kmeans_model.cluster_centers_
        super().__init__(name)
        self.fitted = False
        self.assignment = None
        self.centers = None
        self.max_rule_length = np.nan
        self.weighted_average_rule_length = np.nan
        
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
        if not self.fitted:
            exkmc_tree = ExkmcTree(
                k=self.n_clusters,
                kmeans=self.kmeans_model,
                max_leaf_nodes=self.n_clusters,
                imm=True
            )
            exkmc_tree.fit(X)
            exkmc_labels = exkmc_tree.predict(X, leaf_labels = False)
            self.assignment = labels_to_assignment(exkmc_labels, n_labels = self.n_clusters)
            self.centers = update_centers(
                X = X,
                current_centers = self.original_centers,
                assignment = self.assignment
            )
            self.max_rule_length = exkmc_tree.depth
            self.weighted_average_rule_length = exkmc_tree.get_weighted_average_depth(X)

        return self.assignment, self.centers


####################################################################################################


class IMMMod(Module):
    """
    IMM clustering module, which allows for training upon a dataset with outliers removed 
    (i.e. varies coverage).
    
    Args:
        n_clusters (int): Number of clusters.
        
        kmeans_model (Any): Pretrained SKLearn KMeans model.
        
        min_frac_cover (float): Minimum fraction of data points to cover.
        
        name (str, optional): Name of the baseline method. Defaults to 'KMeans'.
    """
    def __init__(
        self,
        n_clusters : int,
        kmeans_model : Any,
        min_frac_cover : float,
        name : str = 'IMM'
    ):
        self.n_clusters = n_clusters
        self.kmeans_model = kmeans_model
        self.min_frac_cover = min_frac_cover
        self.original_centers = kmeans_model.cluster_centers_
        super().__init__(name)
        self.reset()
    
    def reset(self):
        """
        Resets experiments by returning parametrs to their default values.
        """
        self.frac_cover = self.min_frac_cover
        self.max_rule_length = np.nan
        self.weighted_average_rule_length = np.nan

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
        n, d = X.shape 

        if self.frac_cover > 1:
            raise ValueError("Stepped beyond 100 percent coverage.")
        
        # remove outliers:
        frac_remove = 1 - self.frac_cover
        outliers = outlier_mask(X, centers = self.original_centers, frac_remove=frac_remove)
        non_outliers_idx = np.where(~outliers)[0]

        X_ = X[~outliers]
        # NOTE: after thinking a lot about this, I don't think centers should be updated yet.
        #updated_kmeans = KMeans(n_clusters = self.n_clusters).fit(X_)
        exkmc_tree = ExkmcTree(
                k=self.n_clusters,
                kmeans=self.kmeans_model,
                max_leaf_nodes=self.n_clusters,
                imm=True
        )
        exkmc_tree.fit(X_)
        exkmc_labels = exkmc_tree.predict(X_, leaf_labels = False)

        # Leave outliers excluded from assignment completely (they should not play a role in cost.)
        exkmc_full_labels = [{} for _ in range(n)]
        for i,idx in enumerate(non_outliers_idx):
            exkmc_full_labels[idx] = exkmc_labels[i]
        
        assignment = labels_to_assignment(exkmc_full_labels, n_labels = self.n_clusters)
        updated_centers = update_centers(
            X = X,
            current_centers = self.original_centers,
            #current_centers = updated_kmeans.cluster_centers_,
            assignment = assignment
        )

        self.max_rule_length = exkmc_tree.depth
        # NOTE: Non-outlier subset is used in computation of weighted depth
        self.weighted_average_rule_length = exkmc_tree.get_weighted_average_depth(X_)
        self.frac_cover = np.round(self.frac_cover + step_size, 5)
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
        min_frac_cover : int = None,
        name = 'Decision-Set'
    ):
        self.decision_set_model = decision_set_model
        self.decision_set_params = decision_set_params
        self.clustering = clustering
        self.prune_params = prune_params
        self.min_frac_cover = min_frac_cover
        self.original_centers = self.clustering.centers
        super().__init__(name)
        
        self.reset()
        
        
    def reset(self):
        """
        Resets experiments by returning parametrs to their default values.
        """
        self.model = None
        self.frac_cover = self.min_frac_cover
        self.max_rule_length = np.nan
        self.weighted_average_rule_length = np.nan
    
    
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
        if self.frac_cover > 1:
            raise ValueError("Stepped beyond 100 percent coverage.")
        
        if self.model is None:
            self.model = self.decision_set_model(
                **self.decision_set_params
            )
            self.model.fit(X,  y)
            self.max_rule_length = self.model.max_rule_length

        assignment = None
        updated_centers = None
        if self.prune_params is not None:
            self.model.prune(frac_cover = self.frac_cover, **self.prune_params)
            if self.model.prune_status:
                assignment = labels_to_assignment(
                    self.model.pruned_predict(X, rule_labels = False),
                    n_labels = self.clustering.n_clusters
                )
                updated_centers = update_centers(
                    X = X,
                    current_centers = self.clustering.original_centers,
                    assignment = assignment
                )
        else:
            assignment = labels_to_assignment(
                self.model.predict(X, rule_labels = False),
                n_labels = self.clustering.n_clusters
            )
            updated_centers = update_centers(
                X = X,
                current_centers = self.clustering.original_centers,
                assignment = assignment
            )
        
        self.weighted_average_rule_length = self.model.get_weighted_average_rule_length(X)
        self.frac_cover = np.round(self.frac_cover + step_size,5)
        return assignment, updated_centers
    
    
####################################################################################################