import numpy as np
from sklearn.cluster import KMeans
import copy
from rules import Rule
from utils import *


####################################################################################################
class RuleClustering:
    """
    Default class for a set of Rule objects which are intended to be clustered.
    For more info on Rule objects, please see rules.py.
    
    NOTE: Although this work was intended to support overlapping Rules, it currently only
    works with a set that is non-overlapping. While overlapping Rules will still run without 
    error, it may be the case that during cluster prediction on a given dataset
    the output clustering and labels will not be reflective of the true 
    overlapping cluster structure. I may fix this in the future.
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
                
            labels (List[int]): List of cluster labels for each rule in rule_list.
            
        """
        
        self.rules = rules
        self.k_clusters = k_clusters
        self.rule_list = None
        self.clustering = None
        self.labels = None
    

    def initialize_clustering(self):
        """
        Initializes the clustering by giving every rule its own cluster.
        NOTE: This should only be called after rule_list is initialized (happens in .fit())
        """
        self.clustering = [[i] for i in range(len(self.rule_list))]
        self.labels = [i for i in range(len(self.rule_list))]
        
    
    def update_clustering(self, C):
        """
        Update the current clustering with new assignments.

        Args:
            C (List[List[int]]): 2d list describing cluster assignments. Each inner list represents 
                a single cluster and contains the indices of the rules from self.rule_list
                which are contained within the cluster. 
        """
        self.clustering = C
        new_labels = np.empty(len(self.rule_list))
        new_labels[:] = np.nan
        self.labels = clustering_to_labels(self.clustering, new_labels)
            
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
        # But allows the rule model room to breath if it's still in
        # its fitting process.
        if fit_rules:
            self.rules.fit(X)
            
        rule_model_labels = self.rules.predict(X)
        
        # NEED TO FIGURE OUT HOW TO ACCOUNT FOR OVERLAPS!
        self.rule_list = labels_to_clustering(rule_model_labels)
        
        # ETC... compute clustering ...
        
    
    def predict(self, X, return_clustering = False):
        """
        Assigns cluster labels to an input dataset.

        Args:
            X (np.ndarray): Input n x m dataset to predict upon.
            
            return_clustering (bool): If true, return both the clustering of indices from X,
                and the label array. Defaults to False, in which case only the label 
                array is returned.

        Returns:
            data_clustering (List[List[int]]): 2d List describing cluster assignments 
                for data points. clustering[i] is a list of indices j for the data points
                within cluster i.
                
            data_labels (List[int]): Length n array of assigned labels for each data point.
        """
        
        rule_model_labels = self.rules.predict(X)
        data_labels = np.array([self.labels[i] for i in rule_model_labels])
        
        data_clustering = labels_to_clustering(data_labels)
        
        if return_clustering:
            return data_clustering, data_labels
        else:
            return data_labels

####################################################################################################

class KMeansRuleClustering(RuleClustering):
    """
    Clusters a set of rules via a rule constrained version of Lloyd's algorithm.
    """
    def __init__(self, rules, k_clusters, init = 'k-means', n_init = 10, center_init = None, 
                 max_iterations = 100, random_seed = None):
        """
        Args:
            rules (Object): Rule Model equipped with fit() and predict() methods.
            
            k_clusters (int): Number of clusters.
            
            init (str, optional): Center initialization method. Included options are 
                'k-means' which runs a k-means algorithm and uses the output centers,
                'random++' which uses a randomized k-means++ initialization, or 
                'manual' which assumes an input array of centers. 
                
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
            
            iterations (int): Iteration counter.
            
            cost_per_iteration (List[float]): List with values at index i describing the cost
                of the clustering at iteration i of the algorithm. 
                
            cost (float): Clustering cost in the final iteration of the algorithm.
        """
        
        super().__init__(rules, k_clusters)
        if init in ['k-means', 'random++', 'manual']:
            self.init = init
        else:
            raise ValueError('Unsupported initialization method.')

        self.n_init = n_init
        
        if self.init == 'manual' and (center_init is not None):
            if len(center_init) != self.k_clusters:
                raise ValueError("Input centers must be an array of size k x m")
            
            self.center_init = copy.deepcopy(center_init)
            
        elif self.init == 'manual':
            raise ValueError('Must give an input array of centers for manual initialization.')
        
            
        self.random_seed = random_seed
        self.max_iterations = max_iterations
        
        self.centers = None
        self.iterations = 0
        self.cost_per_iteration = []
        self.cost = np.inf
            
    
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
                distances = np.sum(diffs ** 2, axis=-1)
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
                self.centers[i,:] = np.mean(Xi, axis = 0)
            else:
                # If there are no rules in the cluster, create a new center 
                # to be the mean of the rule which is furthest away from its current center
                max_dist = -1
                max_idx = -1
                new_center = np.zeros(self.centers.shape[1])
                for j, rule in enumerate(self.rule_list):
                    if j not in reassigns:
                        assignment = int(self.labels[j])
                        rule_dist = np.sum((X[rule,:] - self.centers[assignment,:])**2)
                        if rule_dist > max_dist:
                            max_dist = rule_dist
                            max_idx = j
                            new_center = np.mean(X[rule,:], axis = 0)
                        
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
                self.cost_per_iteration.append(kmeans_cost(X, data_clustering, self.centers))
                self.iterations += 1

        else:
            data_clustering, data_labels = self.predict(X, return_clustering = True)
            self.cost_per_iteration.append(kmeans_cost(X, data_clustering, self.centers))
    
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
            
            
    
####################################################################################################

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
        
####################################################################################################

'''
class AgglomerativeRuleClustering(RuleClustering):
    """
    Agglomerative, hierarchical clustering of rules.
    """
    def __init__(self, rule_list, k_clusters, linkage = 'single'):
        """
        Args:
            rule_list (List[Rule]): List of Rule objects to cluster.
            k_clusters (int): Number of clusters.
            linkage (str, optional): Linkage method to use. Implemented options 
                are currently ['single','complete']. Defaults to 'single'.
                
        Attributes:
            D (np.ndarray): Pairwise distance matrix storing distances between clusters. Should 
                initially be the distances between all rules in self.rule_list.
        """
        super().__init__(rule_list, k_clusters)
        
        if linkage == 'single':
            self.linkage = self.single_linkage
        elif linkage == 'complete':
            self.linkage = self.complete_linkage
        else:
            raise ValueError('Unknown linkage function')

        self.D = None
        
        
    def single_linkage(self, idx, joined_pair):
        """
        Compute the single linkage distance between a cluster indexed by idx and the newly 
        formed cluster which is a combination of items in joined_pair.

        Args:
            idx (int): Index of a cluster in self.clustering.
            joined_pair (Tuple(int,int)): Pair of indices for clusters which
                                        have most recently been joined together. 

        Returns:
            (float): Distance between cluster idx and the newly formed cluster.
        """
        
        # The idea is that the new single linkage distance is the minimum of previous
        # single linkage distances.
        dist1 = self.D[min(idx, joined_pair[0]), max(idx, joined_pair[0])]
        dist2 = self.D[min(idx, joined_pair[1]), max(idx, joined_pair[1])]
        return min(dist1, dist2)
    

    def complete_linkage(self, idx, joined_pair):
        """
        Compute the complete linkage distance between a cluster indexed by idx and the newly 
        formed cluster which is a combination of items in joined_pair.

        Args:
            idx (int): Index of a cluster in self.clustering.
            joined_pair (Tuple(int,int)): Pair of indices for clusters which
                                        have most recently been joined together. 

        Returns:
            (float): Distance between cluster idx and the newly formed cluster.
        """

        # The idea is that the new complete linkage distance is the maximum of previous
        # single linkage distances.
        dist1 = self.D[min(idx, joined_pair[0]), max(idx, joined_pair[0])]
        dist2 = self.D[min(idx, joined_pair[1]), max(idx, joined_pair[1])]
        return max(dist1, dist2)
    
    
    def distance_update(self, joined_pair):
        """
        Updates the distance matrix D to maintain pairwise distances between clusters.
        Specifically, this is designed to be called after joining two clusters.

        Args:
            joined_pair (Tuple(int,int)): Pair of indices for clusters which
                                        have most recently been joined together. 
        """

        # This will remove the rows/columns associated with the joined cluster, 
        # and add a new row/column with computed distances for the new cluster
        # onto the end of the matrix. 
        new_dists = []
        for i in range(self.D.shape[1]):
            if i != joined_pair[0] and i != joined_pair[1]:
                new_dists.append(self.linkage(i, joined_pair))
                
            
        self.D = remove_rows_cols(self.D, joined_pair)
        new_row = np.array(new_dists)
        new_col = np.array(new_dists + [np.inf])
        self.D = add_row_col(self.D, new_row, new_col)
            
    
    def cluster_update(self, joined_pair):
        """
        Updates the clustering after joining a pair of clusters.

        Args:
            joined_pair (Tuple(int,int)): Pair of indices for clusters which
                                        have most recently been joined together. 
        """
        
        c1 = joined_pair[0]
        c2 = joined_pair[1]
        new_cluster = [self.clustering[c1] + self.clustering[c2]]
        new_clustering = [self.clustering[i] for i in 
                          range(len(self.clustering)) if i != c1 and i != c2] + new_cluster
        self.update_clustering(new_clustering)
        
        
    def cluster(self):
        """
        Performs the process of hierarchically clustering the data.
        """
        while len(self.clustering) > self.k_clusters:
            flat_index = np.argmin(self.D)
            c1, c2 = np.unravel_index(flat_index, self.D.shape)
            joined_pair = (min(c1, c2), max(c1, c2))
            if self.D[joined_pair[0], joined_pair[1]] != np.inf:
                self.cluster_update(joined_pair)
                self.distance_update(joined_pair)
            else:
                print('More than k clusters, no more non inf distance clusters to join!')
                break
            
    
    def fit(self, X, D):
        """ 
        Fits and clusters the input dataset X.
        
        IMPORTANT: self.rule_list and D need to be in the same relative order,
        in other words rule_list[i] must correspond to D[i,:] and D[:,i]

        Args:
            X (np.ndarray): Input n x m dataset.
            D (np.ndarray): D (np.ndarray): Pairwise distance matrix storing 
                distances between clusters. Should initially be the distances between all rules
                in self.rule_list.
        """
        self.fit_rules(X)
        self.D = copy.deepcopy(D)
        self.cluster()
        self.update_rules()
    
'''
####################################################################################################