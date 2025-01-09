import numpy as np
import copy
from sklearn.cluster import KMeans
from ._rule_cluster import RuleClustering
from ..utils import *

class KMediansRuleClustering(RuleClustering):
    """
    Clusters a set of rules via a rule constrained version of Lloyd's algorithm.
    """
    def __init__(self, rule_list, k_clusters, init = 'k-medians', n_init = 10, center_init = None,
                 max_iterations = 100, random_seed = None):
        """
        Args:
            rule_list (List[Rule]): List of Rule objects to cluster.
            
            k_clusters (int): Number of clusters.
            
            init (str, optional): Center initialization method. Included options are 
                'k-means' which runs a k-means algorithm and uses the output centers or 
                'random++' which uses a randomized k-means++ initialization. 
                
            n_init (int, optional): If using 'random' init, this parameter controls 
                the number of different initializations to try. The clustering object 
                retains the clustering with the best cost performance. Defaults to 10.
                
            center_init (np.ndarray, optional): (k x m) Array of starting centers 
                if using 'manual' initialization method.
            
            max_iterations (int, optional): Maximum number of iterations to run the 
                clustering step for. Defaults to 500.
                
            random_seed (int, optional): Random seed to use for any randomized processes.
                                        
        Attributes:
            rule_list (List[List[int]]): 2d list of length (# rules). Once fitted to a dataset X, 
                each inner list i will contain the indices of points from X which satisfy 
                rule i.
        
            clustering (List[List[int]]): 2d list describing cluster assignments for *Rules. 
                Each inner list represents a single cluster and contains the indices of the rules 
                from self.rule_list which are contained within the cluster. 
                
            labels (List[int]): List of cluster labels for each rule in rule_list.
            
            centers (np.ndarray): k x m Array of representative points in the clustering.
            
            cost_per_iteration (List[float]): List with values at index i describing the cost
                of the clustering at iteration i of the algorithm. 
                
            cost (float): Clustering cost in the final iteration of the algorithm.
        """
        
        super().__init__(rule_list, k_clusters)
        
        if init in ['k-means', 'random++', 'manual']:
            self.init = init
        else:
            raise ValueError('Unsupported initialization method.')
        
        self.n_init = n_init
        
        if self.init == 'manual' and (center_init is not None):
            self.centers = copy.deepcopy(center_init)
        elif self.init == 'manual':
            raise ValueError('Must give an input array of centers for manual initialization.')
        
        self.max_iterations = max_iterations
        self.random_seed = random_seed

        self.centers = None
        self.iterations = 0
        self.cost_per_iteration = []
        
    def cluster_assign(self, X):
        """
        Computes a rule constrained cluster assignment by assigning all points
        in every rule to a center which is currently closest to the mean of the points belonging to
        the rule.
        
        Args:
            X (np.ndarray): Input dataset.
        """
        
        new_clustering = [[] for i in range(self.k_clusters)]
        for i, rule in enumerate(self.rule_list):
            Xi = X[rule, :]
            if len(Xi) != 0:
                diffs = Xi[np.newaxis, :, :] - self.centers[:, np.newaxis, :]
                distances = np.sum(np.abs(diffs), axis=-1)
                sum_array = np.sum(distances, axis=1)
                closest_center = int(np.argmin(sum_array))
                new_clustering[closest_center].append(i)
                
        self.update_clustering(new_clustering)
        
                
    def update_centers(self, X):
        """
        Updates the array of representatives by choosing the mean to be the representative of
        each cluster.
        
        Args:
            X (np.ndarray): Input dataset.
        """
        reassigns = []
        for i,cluster in enumerate(self.clustering):
            if len(cluster) > 0:
                cluster_data = [X[self.rule_list[j], :] for j in cluster]
                Xi = np.vstack(cluster_data)
                self.centers[i,:] = np.median(Xi, axis = 0)
            else:
                # If there are no rules in the cluster, create a new center 
                # to be the mean of the rule which is furthest away from its current center
                max_dist = -1
                max_idx = -1
                new_center = np.zeros(self.centers.shape[1])
                for j, rule in enumerate(self.rule_list):
                    if j not in reassigns:
                        assignment = int(self.labels[j])
                        rule_dist = np.sum(np.abs(X[rule,:] - self.centers[assignment,:]))
                        if rule_dist > max_dist:
                            max_dist = rule_dist
                            max_idx = j
                            new_center = np.median(X[rule,:], axis = 0)
                        
                self.centers[i,:] = new_center
                reassigns.append(max_idx)
            
            
    def cluster(self, X):
        """
        Performs the iterative k-means clustering process.
        
        Args:
            X (np.ndarray): Dataset to cluster rules upon.
        """
        
        self.cluster_assign(X)
        
        if self.update:
            prev_clustering = set({-1})
            new_clustering = set(frozenset(cluster) for cluster in self.clustering)
            
            while new_clustering != prev_clustering and self.iterations < self.max_iterations:
                self.update_centers(X)
                self.cluster_assign(X)
                
                prev_clustering = new_clustering
                new_clustering = set(frozenset(cluster) for cluster in self.clustering)
                data_clustering, data_labels = self.predict(X, return_clustering = True)
                self.cost_per_iteration.append(kmedians_cost(X, data_clustering, self.centers))
                self.iterations += 1

        else:
            data_clustering, data_labels = self.predict(X, return_clustering = True)
            self.cost_per_iteration.append(kmedians_cost(X, data_clustering, self.centers))
    
        self.cost = self.cost_per_iteration[-1]
        
            
    def fit(self, X, fit_rules = True, update = True):
        """
        Fits rules to a given dataset X and runs the clustering process.
        Args:
            X (np.ndarray): Input dataset.
            
            fit_rules (bool): If True, fits rules to the dataset as well.
        """
        self.update = update

        # Ensures that the rule model is fitted to X,
        # But allows the rule model room to breath if it's still in
        # its fitting process.
        if fit_rules:
            self.rules.fit(X)
            
        rule_model_labels = self.rules.predict(X)
        
        # NEED TO FIGURE OUT HOW TO ACCOUNT FOR OVERLAPS!
        self.rule_list = labels_to_clustering(rule_model_labels)
        
        # Initialize clustering and labels:
        self.initialize_clustering()
        
        if self.init == 'k-means':
            kmeans = KMeans(n_clusters=self.k_clusters, random_state=self.random_seed, n_init="auto").fit(X)
            self.centers = kmeans.cluster_centers_
            self.cluster(X)
            
        elif self.init == 'random++':
            # Store results from best run
            best_cost = np.inf
            best_rule_list = None
            best_clustering = None
            best_labels = None
            best_iterations = None
            best_cost_per_iteration = None
            
            for init in range(self.n_init):
                # Reset 
                self.initialize_clustering()
                self.iterations = 0
                self.cost_per_iteration = []
                
                # Run clustering with random centers
                self.centers = kmeans_plus_plus_initialization(X, self.k_clusters, self.random_seed)
                self.cluster(X)
                
                # Record if improved
                if self.cost < best_cost:
                    best_cost = self.cost
                    best_rule_list = self.rule_list
                    best_clustering = self.clustering
                    best_labels = self.labels
                    best_iterations = self.iterations
                    best_cost_per_iteration = self.cost_per_iteration
                    
            self.cost = best_cost
            self.rule_list = best_rule_list
            self.clustering = best_clustering
            self.labels = best_labels
            self.iterations = best_iterations
            self.cost_per_iteration = best_cost_per_iteration
            
        else:
            # manual initialization
            self.centers = self.center_init
            self.cluster(X)
        