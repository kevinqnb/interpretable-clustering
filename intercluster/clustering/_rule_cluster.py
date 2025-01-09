import numpy as np
from ..utils import *


class RuleClustering:
    """
    Default class for a set of Rule objects which are intended to be clustered.
    For more info on Rule objects, please see rules.py.
    """
    
    def __init__(self, rules, k_clusters):
        """
        Args:
            rules (Object): Rule Model equipped with fit() and predict() methods.
            
            k_clusters (int): Number of clusters.
            
        Attributes:
            rule_list (List[List[int]]): 2d list of length (# rules). Once fitted to a dataset X, 
                each inner list i will contain the indices of points from X which satisfy 
                rule i.
        
            clustering (List[List[int]]): 2d list describing cluster assignments for *Rules. 
                Each inner list represents a single cluster and contains the indices of the rules 
                from self.rule_list which are contained within the cluster. 
                
            clustering (np.ndarray): n x k binary matrix with entry (i,j) being 1 if point 
                i belongs to cluster j and 0 otherwise.
                
            labels (List[int]): List of cluster labels for each rule in rule_list.
            
        """
        
        self.rules = rules
        self.k_clusters = k_clusters
        self.points_to_rules = None
        self.rule_assignment = None
        #self.labels = None
    

    def initialize_clustering(self):
        """
        Initializes the clustering by giving every rule its own cluster.
        NOTE: This should only be called after rule_list is initialized (happens in .fit())
        """
        #self.clustering = [[i] for i in range(len(self.rule_list))]
        #self.labels = [i for i in range(len(self.rule_list))]
        self.rule_assignment = np.zeros((self.points_to_rules.shape[1], self.k_clusters))
        #for i in range(len(self.rule_matrix)):
        #    self.clustering[i, i] = 1
        
    
    def update_clustering(self, new_assignment):
        """
        Update the current clustering with new assignments.

        Args:
            C (List[List[int]]): 2d list describing cluster assignments. Each inner list represents 
                a single cluster and contains the indices of the rules from self.rule_list
                which are contained within the cluster. 
        """
        self.rule_assignment = new_assignment
        #self.labels = clustering_to_labels(self.clustering, n = len(self.rule_list))
            
    def cluster(self):
        """
        Performs the process of clustering the data.
        """
        pass
            
    def fit(self, X, fit_rules = True):
        """
        Fits and clusters the input dataset X.

        Args:
            X (np.ndarray): Input dataset.
            
            fit_rules (bool): If True, fits rules to the dataset as well.
        """
        
        # Ensures that the rule model is fitted to X,
        # But allows the rule model room to breathe if it's still in
        # its fitting process.
        if fit_rules:
            self.rules.fit(X)
            
        rule_model_labels = self.rules.predict(X)
        
        self.points_to_rules = labels_to_assignment(rule_model_labels)
        
        # ETC... compute clustering ...
        
    
    def predict(
        self,
        X : np.ndarray,
        assignment_method : str = None
    ) -> np.ndarray:
        """
        Assigns cluster labels to an input dataset.

        Args:
            X (np.ndarray): Input n x m dataset to predict upon.
            
            assignment_method (str, optional): If None, returns the full point to cluster
                assignment. Otherwise accepts values 'vote' or 'min.' If 'vote',
                returns the cluster assignment for which each point x is assigned to the
                cluster that appears most often in the rules which cover x. If 'min',
                returns the cluster assignment for which each point x is assigned to the
                cluster which is closest among the assignments for rules which cover x. 

        Returns:
            point_assignment (np.ndarray): n x k boolean matrix with entry (i,j) being 1 if point
                i belongs to cluster j and 0 otherwise.
        """
        
        rule_model_labels = self.rules.predict(X)
        
        points_to_rules_matrix = labels_to_assignment(rule_model_labels)
        
        if assignment_method is None:
            # Boolean matrix multiplication to get the full cluster assignment of the data
            point_assignment = np.dot(points_to_rules_matrix, self.rule_assignment)
            
        elif assignment_method == 'vote':
            # Standard matrix multiplication to get the voted clustering of the data
            rule_assignment_int = self.rule_assignment.astype(int)
            point_assignment = np.dot(points_to_rules_matrix, rule_assignment_int)
            
            # Convert the integer point assignment array back to boolean
            n, m = point_assignment.shape
            voted_assignment = np.zeros((n, m), dtype=bool)
            
            for i in range(n):
                # If a data point belongs to multiple clusters, find the cluster
                # it belongs to most often. 
                
                if point_assignment[i,:].max() > 0:
                    max_indices = np.where(point_assignment[i,:] == point_assignment[i,:].max())[0]
                    if len(max_indices) > 1:
                        # In the event of a tie, randomly choose one of the assigned clusters.
                        max_index = np.random.choice(max_indices)
                    else:
                        max_index = max_indices[0]
                        
                    voted_assignment[i, max_index] = True
                
            point_assignment = voted_assignment
            
            
        elif assignment_method == 'min':
            # Boolean matrix multiplication to get the full cluster assignment of the data
            point_assignment = np.dot(points_to_rules_matrix, self.rule_assignment)
            
            n, m = point_assignment.shape
            min_assignment = np.zeros((n, m), dtype=bool)
            
            for i in range(n):
                min_dist = np.inf
                min_j = None
                for j in np.where(point_assignment[i,:] == 1)[0]:
                    
                    # THIS SHOULD REALLY ACCOUNT FOR KMEDIANS DISTANCE as well...
                    dist = np.sum((X[i,:] - self.centers[j,:])**2)
                    if dist < min_dist:
                        min_dist = dist
                        min_j = j
                        
                min_assignment[i, min_j] = True
                
            point_assignment = min_assignment
            
            
        return point_assignment