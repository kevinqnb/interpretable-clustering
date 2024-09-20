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
    
    NOTE: Although this work was intended to support overlapping Rules, it currently 
    works best with a set that is non-overlapping. While overlapping Rules will still run without 
    error, it may be the case that during cluster prediction on a given dataset
    the output clustering and labels will not be reflective of the true 
    overlapping cluster structure. I may fix this in the future.
    """
    
    def __init__(self, rule_list, k_clusters):
        """
        Args:
            rule_list (List[Rule]): List of Rule objects to cluster.
            k_clusters (int): Number of clusters.
            
        Attributes:
            clustering (List[List[int]]): 2d list describing cluster assignments for *Rules. 
                Each inner list represents a single cluster and contains the indices of the rules 
                from self.rule_list which are contained within the cluster. 
                
            labels (List[int]): List of cluster labels for each rule in rule_list.
                
            clustered_rule_list (List[Rule]): Length k list of Rule objects, which are formed from 
                combinations of rules according to the rule clustering.
        """
        
        self.rule_list = rule_list
        self.k_clusters = k_clusters
        self.clustering = None
        self.labels = None
        self.clustered_rule_list = None
        self.initialize_clustering()
        
    
    def __str__(self):
        """
        Gives a user-friendly string representation of the clustering boundaries.
        """
        cluster_str = "\n".join(f"IF:\n{repr(rule)}\nTHEN CLUSTER {i}\n" 
                                for i,rule in enumerate(self.clustered_rule_list))
        return cluster_str
    
            
    def initialize_clustering(self):
        """
        Initializes the clustering by giving every rule its own cluster.
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
        self.labels = clustering_to_labels(self.clustering)
        
    def fit_rules(self, X):
        """
        Fits a dataset X to each rule in the given rule list.

        Args:
            X (np.ndarray): Input dataset.
        """
        for r in self.rule_list:
            r.fit(X)
            
    def update_rules(self):
        """
        Updates the clustered_rule_list to fit the current rule clustering.
        """
        new_rule_list = []
        for cluster in self.clustering:
            cluster_term_list = []
            for i in cluster:
                cluster_term_list += self.rule_list[i].term_list
                
            R = Rule(cluster_term_list)
            R.simplify()
            new_rule_list.append(R)
            
        self.clustered_rule_list = new_rule_list
            
            
    def cluster(self):
        """
        Performs the process of clustering the data.
        """
        pass
            
    def fit(self, X):
        """
        Fits and clusters the input dataset X.

        Args:
            X (np.ndarray): Input dataset.
        """
        pass
    
    def predict(self, X):
        """
        Assigns cluster labels to an input dataset.

        Args:
            X (np.ndarray): Input n x m dataset.

        Returns:
            data_clustering (List[List[int]]): 2d List describing cluster assignments 
                for data points. clustering[i] is a list of indices j for the data points
                within cluster i.
                
            data_labels (List[int]): Length n array of assigned labels for each data point.
        """
        data_labels = np.zeros(len(X)) - 1
        for i, cluster in enumerate(self.clustering):
            for j in cluster:
                rule = self.rule_list[j]
                idxs = rule.find_satisfied_indices(X)
                for idx in idxs:
                    # NOTE: Right now this only supports labeling of non-overlapping clusters. 
                    if data_labels[idx] == -1:
                        data_labels[idx] = i
        
        data_clustering = labels_to_clustering(data_labels)
        return data_clustering, data_labels

####################################################################################################

class KMeansRuleClustering(RuleClustering):
    """
    Clusters a set of rules via a rule constrained version of Lloyd's algorithm.
    """
    def __init__(self, rule_list, k_clusters, init = 'k-means', max_iterations = 1000,
                 random_seed = None, cost_tracker = False):
        """
        Args:
            rule_list (List[Rule]): List of Rule objects to cluster.
            
            k_clusters (int): Number of clusters.
            
            init (str, optional): Center initialization method. Included options are 
                'k-means' which runs a k-means algorithm and uses the output centers or 
                'k-means++' which uses a randomized k-means++ initialization. 
                
            max_iterations (int, optional): Maximum number of iterations to run the 
                clustering step for. Defaults to 1,000.
                
            random_seed (int, optional): Random seed to use for any randomized processes.
                
            cost_tracker (bool): Boolean variable deciding whether or not to keep track
                of the clustering cost at each iteration. 
                                        
        Attributes:
            centers (np.ndarray): k x m Array of representative points in the clustering.
            
            cost_per_iteration (List[float]): List with values at index i describing the cost
                of the clustering at iteration i of the algorithm. 
            
            iterations (int): Iteration counter.
        """
        super().__init__(rule_list, k_clusters)
        if init in ['k-means', 'k-means++']:
            self.init = init
        else:
            raise ValueError('Unsupported initialization method.')
        
        self.random_seed = random_seed
        self.max_iterations = max_iterations
        self.cost_tracker = cost_tracker
        if cost_tracker:
            self.cost_per_iteration = []
            
        self.centers = None
        self.iterations = 0
            
    
    def cluster_assign(self):
        """
        Computes a rule constrained cluster assignment by assigning all points
        in every rule to a center which is currently closest to the mean of the points belonging to
        the rule.
        """
        
        new_clustering = [[] for i in range(self.k_clusters)]
        for i, rule in enumerate(self.rule_list):
            Xi = rule.satisfied_points
            if len(Xi) != 0:
                diffs = Xi[np.newaxis, :, :] - self.centers[:, np.newaxis, :]
                distances = np.sum(diffs ** 2, axis=-1)
                sum_array = np.sum(distances, axis=1)
                closest_center = int(np.argmin(sum_array))
                new_clustering[closest_center].append(i)
                
        self.update_clustering(new_clustering)
            
        
    def update_centers(self):
        """
        Updates the array of representatives by choosing the mean to be the representative of
        each cluster.
        """
        reassigns = []
        for i,cluster in enumerate(self.clustering):
            if len(cluster) > 0:
                cluster_data = [self.rule_list[j].satisfied_points for j in cluster]
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
                        assignment = self.labels[j]
                        rule_dist = np.sum((rule.satisfied_points - self.centers[assignment,:])**2)
                        if rule_dist > max_dist:
                            max_dist = rule_dist
                            max_idx = j
                            new_center = np.mean(rule.satisfied_points, axis = 0)
                        
                self.centers[i,:] = new_center
                reassigns.append(max_idx)
    
    
    def cluster(self, X):
        """
        Performs the iterative k-means clustering process.
        """
        
        self.cluster_assign()
        prev_clustering = set({-1})
        new_clustering = set(frozenset(cluster) for cluster in self.clustering)
        
        while new_clustering != prev_clustering and self.iterations < self.max_iterations:
            self.update_centers()
            self.cluster_assign()
            
            prev_clustering = new_clustering
            new_clustering = set(frozenset(cluster) for cluster in self.clustering)
            if self.cost_tracker:
                data_clustering, data_labels = self.predict(X)
                self.cost_per_iteration.append(kmeans_cost(X, data_clustering, self.centers))
            self.iterations += 1
    
    def fit(self, X):
        """
        Fits rules to a given dataset X and runs the clustering process.
        Args:
            X (np.ndarray): Input dataset.
        """
        self.fit_rules(X)
        if self.init == 'k-means':
            kmeans = KMeans(n_clusters=self.k_clusters, random_state=self.random_seed, n_init="auto").fit(X)
            self.centers = kmeans.cluster_centers_
        else:
            self.centers = kmeans_plus_plus_initialization(X, self.k_clusters, self.random_seed)
            
        self.cluster(X)
        self.update_rules()
    
####################################################################################################

class KMediansRuleClustering(RuleClustering):
    """
    Clusters a set of rules via a rule constrained version of Lloyd's algorithm.
    """
    def __init__(self, rule_list, k_clusters, init = 'k-medians', max_iterations = 1000,
                 random_seed = None):
        """
        Args:
            rule_list (List[Rule]): List of Rule objects to cluster.
            
            k_clusters (int): Number of clusters.
            
            init (str, optional): Center initialization method. Included options are 
                'k-medians' which runs a k-means algorithm and uses the output centers or 
                'k-means++' which uses a randomized k-means++ initialization. 
                
            max_iterations (int, optional): Maximum number of iterations to run the 
                clustering step for. Defaults to 1,000.
                
            random_seed (int, optional): Random seed to use for any randomized processes.
                                        
        Attributes:
            centers (np.ndarray): k x m Array of representative points in the clustering.
        """
        super().__init__(rule_list, k_clusters)
        
        if init in ['k-medians', 'k-means++']:
            self.init = init
        else:
            raise ValueError('Unsupported initialization method.')
        
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        
        self.iterations = 0
        self.centers = None
            
    
    def cluster_assign(self):
        """
        Computes a rule constrained cluster assignment by assigning all points
        in every rule to the center which is currently closest to the median
        of the points belonging to the rule.
        """
        
        new_clustering = [[] for i in range(self.k_clusters)]
        for i, rule in enumerate(self.rule_list):
            Xi = rule.satisfied_points
            if len(Xi) != 0:
                diffs = Xi[np.newaxis, :, :] - self.centers[:, np.newaxis, :]
                distances = np.sum(np.abs(diffs), axis=-1)
                sum_array = np.sum(distances, axis=1)
                closest_center = int(np.argmin(sum_array))
                new_clustering[closest_center].append(i)
                
        self.update_clustering(new_clustering)
            
        
    def update_centers(self):
        """
        Updates the array of representatives by choosing the mean to be the representative of
        each cluster.
        """
        reassigns = []
        for i,cluster in enumerate(self.clustering):
            if len(cluster) > 0:
                cluster_data = [self.rule_list[j].satisfied_points for j in cluster]
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
                        assignment = self.labels[j]
                        rule_dist = np.sum(np.abs(rule.satisfied_points - self.centers[assignment,:]))
                        if rule_dist > max_dist:
                            max_dist = rule_dist
                            max_idx = j
                            new_center = np.mean(rule.satisfied_points, axis = 0)
                        
                self.centers[i,:] = new_center
                reassigns.append(max_idx)
    
    
    def cluster(self):
        """
        Performs the iterative k-means clustering process.
        """
        self.cluster_assign()
        prev_clustering = set({-1})
        new_clustering = set(frozenset(cluster) for cluster in self.clustering)
        
        while new_clustering != prev_clustering and self.iterations < self.max_iterations:
            self.update_centers()
            self.cluster_assign()
            
            prev_clustering = new_clustering
            new_clustering = set(frozenset(cluster) for cluster in self.clustering)
            
            self.iterations += 1
            
    
    def fit(self, X):
        """
        Fits rules to a given dataset X and runs the clustering process.
        Args:
            X (np.ndarray): Input dataset.
        """
        self.fit_rules(X)
        self.centers = kmeans_plus_plus_initialization(X, self.k_clusters)
        if self.init == 'k-means':
            kmeans = KMeans(n_clusters=self.k_clusters, random_state=self.random_seed, n_init="auto").fit(X)
            self.centers = kmeans.cluster_centers_
        else:
            self.centers = kmeans_plus_plus_initialization(X, self.k_clusters, self.random.seed)
        self.cluster()
        self.update_rules()
        
####################################################################################################

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
    
####################################################################################################