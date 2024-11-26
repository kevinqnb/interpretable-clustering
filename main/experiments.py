import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from ExKMC.Tree import Tree as ExTree
from tree import *
from decision_sets import *
from rule_clustering import *
from utils import *


####################################################################################################

class Experiment:
    """
    Performs suite of experiments on an input dataset which measure clustering cost with parameter 
    changes.
    """
    def __init__(self, data, n_clusters, random_seed = None, verbose = True):
        """
        Args:
            data (np.ndarray): Input dataset.
            
            n_clusters (int): Number of clusters to use.
            
            random_seed (int, optional): Random seed for experiments. Defaults to None.
            
            verbose (bool, optional): Allows for printing of status.
            
        Attrs:
            kmeans_centers (np.ndarray): (n x k) Array of reference centers found from a 
                run of kmeans.
                
            
        """
        self.data = data
        self.n_clusters = n_clusters
        self.seed = random_seed
        self.verbose = verbose
        np.random.seed(random_seed)
        
        self.kmeans = None
        self.kmeans_centers = None
        self.kmeans_cost = np.inf
        
        self.imm_tree = None
        self.imm_cost = np.inf
        self.imm_depth = 0
        
        self.randomized_imm_tree = None
        self.randomized_imm_cost = np.inf
        self.randomized_imm_depth = 0
        
        self.leaves_cost_df = None
        self.leaves_depth_df = None
        self.leaves_iteration_df = None
        
        self.depths_cost_df = None
        self.depths_iteration_df = None
        
    def save_results(self, directory, identifier = ''):
        leaves_cost_filename = os.path.join(directory, 'leaves_cost' + str(identifier) + '.csv')
        leaves_depth_filename = os.path.join(directory, 'leaves_depth' + str(identifier) + '.csv')
        leaves_iteration_filename = os.path.join(directory, 'leaves_iteration' + str(identifier) + '.csv')
        depth_cost_filename = os.path.join(directory, 'depths_cost' + str(identifier) +  '.csv')
        
        self.leaves_cost_df.to_csv(leaves_cost_filename)
        self.leaves_depth_df.to_csv(leaves_depth_filename)
        self.leaves_iteration_df.to_csv(leaves_iteration_filename)
        self.depths_cost_df.to_csv(depth_cost_filename)
        
        
    def initialize(self):
        """
        Initializes the experiments by computing results for a few 
        baseline algorithms (KMeans, IMM, Randomized IMM).
        """
        if self.verbose:
            print('Initializing k-means')
            
        kmeans = KMeans(n_clusters=self.n_clusters,
                        init = 'k-means++',
                        random_state=self.seed,
                        n_init="auto").fit(self.data)
        self.kmeans_centers = kmeans.cluster_centers_
        kmeans_assignment = labels_to_assignment(kmeans.labels_)
        self.kmeans_cost = kmeans_cost(self.data, kmeans_assignment, self.kmeans_centers)
        self.kmeans = kmeans
        
        
        if self.verbose:
            print('Initializing IMM')
            
        IMM_tree = ExTree(self.n_clusters, max_leaves = self.n_clusters, base_tree = "IMM",
                            random_state = self.seed)
        imm_labels = IMM_tree.fit_predict(self.data, kmeans)
        imm_assignment = labels_to_assignment(imm_labels)
        imm_centers = IMM_tree.all_centers
        self.imm_cost = kmeans_cost(self.data, imm_assignment, imm_centers)
        self.imm_tree = ConvertExKMC(IMM_tree.tree, self.data)
        self.imm_depth = self.imm_tree.depth
        
        if self.verbose:
            print('Initializing Randomized IMM')
            
        # Revert the seed in case it has been advanced anywhere else:
        np.random.seed(self.seed)

        random_tree_cost = np.inf
        random_tree = None

        for i in range(10000):
            random_tree_ = RandomTree(max_leaf_nodes = self.n_clusters, min_points_leaf = 1)
            random_tree_.fit(kmeans.cluster_centers_)
            random_tree_labels_ = random_tree_.predict(self.data)
            random_tree_assignment_ = labels_to_assignment(random_tree_labels_)
            #random_tree_centers_ = np.vstack([self.data[cluster,:].mean(axis = 0) for cluster in random_tree_clustering_])
            
            random_tree_centers_ = np.vstack([
                self.data[np.where(cluster)[0], :].mean(axis=0) if len(np.where(cluster)[0]) > 0 
                else np.zeros(self.data.shape[1]) 
                for i,cluster in enumerate(random_tree_assignment_.T)
            ])
            
            #rcost = kmeans_cost(self.data, random_tree_clustering_, kmeans.cluster_centers_)
            rcost = kmeans_cost(self.data, random_tree_assignment_, random_tree_centers_)
            if rcost < random_tree_cost:
                random_tree_cost = rcost
                random_tree = random_tree_
                
        self.randomized_imm_cost = random_tree_cost
        self.randomized_imm_tree = random_tree
        self.randomized_imm_depth = random_tree.depth
            
        
        
        
    def run_leaves_cost(self, min_leaves, max_leaves, clusterer):
        """
        Runs the leaves x cost experiment by iterating through the maximum leaves parameter
        for each algorithm.
        
        Args:
            min_leaves (int): Minimum number of leaves. 
            max_leaves (int): Maximum number of leaves.
            clusterer (RuleClustering): Rule clustering object to cluster leaves.
        """
        # Revert the seed in case it has been advanced anywhere else:
        np.random.seed(self.seed)
        
        if min_leaves < self.n_clusters:
            raise ValueError('Minimum leaves must be greater than the number of clusters. ')
        
        leaves = list(range(min_leaves,max_leaves + 1))
        
        
        if self.verbose:
            print("Initializing trees...")
        # Define some models to be iteratively built (instead of re-computing them from 
        # scratch every iteration)
        unsupervised_tree_ = UnsupervisedTree(splits = 'axis', max_leaf_nodes = max_leaves, 
                                              norm = 2)
        
        # Here we initialize a tree, which will expand later
        unsupervised_tree_.fit(self.data, iterative = True, init_steps = self.n_clusters - 1)

        # Unsupervised Oblique
        unsupervised_oblique_tree_ = UnsupervisedTree(splits = 'oblique', 
                                                      max_leaf_nodes = max_leaves, norm = 2)
        unsupervised_oblique_tree_.fit(self.data, iterative = True, 
                                       init_steps = self.n_clusters - 1)

        # Centroid tree which should be equivalent to ExKMC (if ExKMC doesn't use IMM base)
        centroid_tree_ = CentroidTree(splits = 'axis', max_leaf_nodes = max_leaves, norm = 2,
                                      center_init = 'manual', centers = self.kmeans_centers)
        centroid_tree_.fit(self.data, iterative = True, init_steps = self.n_clusters - 1)

        # Centroid Oblique Tree
        centroid_oblique_tree_ = CentroidTree(splits = 'oblique', max_leaf_nodes = max_leaves, 
                                              norm = 2, center_init = 'manual', 
                                              centers = self.kmeans_centers)
        centroid_oblique_tree_.fit(self.data, iterative = True, init_steps = self.n_clusters - 1)


        # Rule Clustered Tree
        '''
        clustered_tree_ = RuleClusteredTree(splits = 'axis', max_leaf_nodes = max_leaves, norm = 2, 
                                            center_init = 'manual', 
                                            centers = self.kmeans_centers,
                                        clusterer = clusterer, 
                                        start_centers = self.kmeans_centers)
        clustered_tree_.fit(self.data, iterative = True, init_steps = self.n_clusters - 1)


        # Rule Clustered Oblique Tree
        clustered_oblique_tree_ = RuleClusteredTree(splits = 'oblique', max_leaf_nodes = max_leaves,
                                                    norm = 2, center_init = 'manual', 
                                                    centers = self.kmeans_centers,
                                                    clusterer = clusterer, 
                                                    start_centers = self.kmeans_centers)
        clustered_oblique_tree_.fit(self.data, iterative = True, init_steps = self.n_clusters - 1)
        '''
        
        
        # Need to make feature pairings as a parameter somewhere!
        feature_pairings = [list(range(12))] + [list(range(12,24))]
        fd = 3
        num_trees = 1000
        
        unsupervised_forest = DecisionForest(UnsupervisedTree, tree_params = {'splits':'axis', 'max_depth':fd}, 
                            num_trees = num_trees, num_features = fd, feature_pairings = feature_pairings)
        
        unsupervised_forest.fit(self.data)
        
        # Centroid Decision Forest:
        centroid_forest = DecisionForest(CentroidTree, tree_params = {'splits':'axis', 'max_depth':fd}, 
                            num_trees = num_trees, num_features = fd, feature_pairings = feature_pairings,
                            center_init='manual', centers = self.kmeans_centers)
        centroid_forest.fit(self.data)
        
        
        
        # Track cost by leaves:
        exkmc_costs = []
        rule_cart_costs =[]
        rule_unsupervised_costs = []
        rule_unsupervised_oblique_costs = []
        rule_centroid_costs = []
        rule_centroid_oblique_costs = []
        #rule_clustered_costs = []
        #rule_clustered_oblique_costs = []
        rule_unsupervised_forest_costs = []
        rule_centroid_forest_costs = []

        # Track depths
        exkmc_depths = []
        rule_cart_depths =[]
        rule_unsupervised_depths = []
        rule_unsupervised_oblique_depths = []
        rule_centroid_depths = []
        rule_centroid_oblique_depths = []
        #rule_clustered_depths = []
        #rule_clustered_oblique_depths = []
        rule_unsupervised_forest_depths = []
        rule_centroid_forest_depths = []
        
        # Track iterations used for rule clustering
        rule_cart_iterations =[]
        rule_unsupervised_iterations = []
        rule_unsupervised_oblique_iterations = []
        rule_centroid_iterations = []
        rule_centroid_oblique_iterations = []
        #rule_clustered_iterations = []
        #rule_clustered_oblique_iterations = []
        rule_unsupervised_forest_iterations = []
        rule_centroid_forest_iterations = []
        
        # Expand the trees and compute cost:
        for l in leaves:
            
            if self.verbose:
                print("Number of leaves: " + str(l))

            # ExKMC
            ExKMC_tree_ = ExTree(self.n_clusters, max_leaves = l, base_tree = "IMM")
            exkmc_labs_ = ExKMC_tree_.fit_predict(self.data, self.kmeans)
            exkmc_assignment_ = labels_to_assignment(exkmc_labs_)
            exkmc_centers_ = ExKMC_tree_.all_centers
            exkmc_cost_ = kmeans_cost(self.data, exkmc_assignment_, exkmc_centers_)
            exkmc_costs.append(exkmc_cost_)
            exkmc_depths.append(ExKMC_tree_._max_depth())
            
            # Rule CART:
            dtree_ = DecisionTreeClassifier(criterion='gini', max_leaf_nodes = l)
            dtree_.fit(self.data, self.kmeans.labels_)
            cart_tree_ = ConvertSklearn(dtree_.tree_, self.data)
            rule_cart_ = clusterer(cart_tree_, k_clusters = self.n_clusters, init = 'manual', 
                                   center_init = self.kmeans_centers)
            rule_cart_.fit(self.data)
            rule_cart_costs.append(rule_cart_.cost)
            rule_cart_depths.append(cart_tree_.depth)
            rule_cart_iterations.append(rule_cart_.iterations)
            

            # Rule Clustering for Unsupervised Tree
            unsupervised_tree_.fit_step()
            rule_kmeans_ = clusterer(unsupervised_tree_, k_clusters = self.n_clusters,
                                     init = 'manual', center_init = self.kmeans_centers,
                                     random_seed = self.seed)
            rule_kmeans_.fit(self.data, fit_rules = False)
            rule_unsupervised_costs.append(rule_kmeans_.cost)
            rule_unsupervised_depths.append(unsupervised_tree_.depth)
            rule_unsupervised_iterations.append(rule_kmeans_.iterations)


            # Rule Clustering for Unsupervised Oblique Tree
            unsupervised_oblique_tree_.fit_step()
            rule_kmeans_ = clusterer(unsupervised_oblique_tree_, k_clusters = self.n_clusters,
                                     init = 'manual', center_init = self.kmeans_centers,
                                     random_seed = self.seed)
            rule_kmeans_.fit(self.data, fit_rules = False)
            rule_unsupervised_oblique_costs.append(rule_kmeans_.cost)
            rule_unsupervised_oblique_depths.append(unsupervised_oblique_tree_.depth)
            rule_unsupervised_oblique_iterations.append(rule_kmeans_.iterations)

            # Rule Clustering for Centroid Tree
            centroid_tree_.fit_step()
            rule_kmeans_ = clusterer(centroid_tree_, k_clusters = self.n_clusters, init = 'manual',
                                     center_init = self.kmeans_centers, random_seed = self.seed)
            rule_kmeans_.fit(self.data, fit_rules = False)
            rule_centroid_costs.append(rule_kmeans_.cost)
            rule_centroid_depths.append(centroid_tree_.depth)
            rule_centroid_iterations.append(rule_kmeans_.iterations)
            
            
            # Rule Clustering for Centroid Oblique Tree
            centroid_oblique_tree_.fit_step()
            rule_kmeans_ = clusterer(centroid_oblique_tree_, k_clusters = self.n_clusters, 
                                     init = 'manual', center_init = self.kmeans_centers, 
                                     random_seed = self.seed)
            rule_kmeans_.fit(self.data, fit_rules = False)
            rule_centroid_oblique_costs.append(rule_kmeans_.cost)
            rule_centroid_oblique_depths.append(centroid_oblique_tree_.depth)
            rule_centroid_oblique_iterations.append(rule_kmeans_.iterations)
            
            # Clustered Tree
            '''
            clustered_tree_.fit_step()
            rule_clustered_costs.append(clustered_tree_.clustering_cost)
            rule_clustered_depths.append(clustered_tree_.depth)
            rule_clustered_iterations.append(clustered_tree_.clustering_iterations)

            # Clustered Oblique Tree
            clustered_oblique_tree_.fit_step()
            rule_clustered_oblique_costs.append(clustered_oblique_tree_.clustering_cost)
            rule_clustered_oblique_depths.append(clustered_oblique_tree_.depth)
            rule_clustered_oblique_iterations.append(clustered_oblique_tree_.clustering_iterations)
            '''
            
            # Unsupervised Decision Forest:
            unsupervised_forest.prune(self.data, l)
            
            rule_kmeans_ = clusterer(unsupervised_forest, k_clusters = self.n_clusters,
                                     init = 'manual', center_init = self.kmeans_centers,
                                     random_seed = self.seed)
            rule_kmeans_.fit(self.data, fit_rules = False)
            rule_unsupervised_forest_costs.append(rule_kmeans_.cost)
            rule_unsupervised_forest_depths.append(fd)
            rule_unsupervised_forest_iterations.append(rule_kmeans_.iterations)
            
            # Centroid Decision Forest:
            centroid_forest.prune(self.data, l)
            
            rule_kmeans_ = clusterer(centroid_forest, k_clusters = self.n_clusters,
                                     init = 'manual', center_init = self.kmeans_centers,
                                     random_seed = self.seed)
            rule_kmeans_.fit(self.data, fit_rules = False)
            rule_centroid_forest_costs.append(rule_kmeans_.cost)
            rule_centroid_forest_depths.append(fd)
            rule_centroid_forest_iterations.append(rule_kmeans_.iterations)
            
            
            
        
    
        data_frame = {
            'k-means': [self.kmeans_cost]*len(leaves),
            'IMM': [self.imm_cost]*len(leaves),
            'RandomIMM': [self.randomized_imm_cost]*len(leaves),
            'ExKMC': exkmc_costs,
            'CART': rule_cart_costs,
            'Unsupervised': rule_unsupervised_costs,
            'Unsupervised Obl.': rule_unsupervised_oblique_costs,
            'Centroid': rule_centroid_costs,
            'Centroid Obl.': rule_centroid_oblique_costs,
            #'Hybrid': rule_clustered_costs,
            #'Hybrid Obl.': rule_clustered_oblique_costs
            'Unsupervised Forest': rule_unsupervised_forest_costs,
            'Centroid Forest': rule_centroid_forest_costs
        }
        
        self.df = data_frame
        self.leaves_cost_df = pd.DataFrame(data_frame)
        self.leaves_cost_df.index = leaves
        
        data_frame2 = {
            'IMM': [self.imm_depth]*len(leaves),
            'RandomIMM': [self.randomized_imm_depth]*len(leaves),
            'ExKMC': exkmc_depths,
            'CART': rule_cart_depths,
            'Unsupervised': rule_unsupervised_depths,
            'Unsupervised Obl.': rule_unsupervised_oblique_depths,
            'Centroid': rule_centroid_depths,
            'Centroid Obl.': rule_centroid_oblique_depths,
            #'Hybrid': rule_clustered_depths,
            #'Hybrid Obl.': rule_clustered_oblique_depths
            'Unsupervised Forest': rule_unsupervised_forest_depths,
            'Centroid Forest': rule_centroid_forest_depths
        }
        
        self.leaves_depth_df = pd.DataFrame(data_frame2)
        self.leaves_depth_df.index = leaves
        
        data_frame3 = {
            'CART': rule_cart_iterations,
            'Unsupervised': rule_unsupervised_iterations,
            'Unsupervised Obl.': rule_unsupervised_oblique_iterations,
            'Centroid': rule_centroid_iterations,
            'Centroid Obl.': rule_centroid_oblique_iterations,
            #'Hybrid': rule_clustered_iterations,
            #'Hybrid Obl.': rule_clustered_oblique_iterations
            'Unsupervised Forest': rule_unsupervised_forest_iterations,
            'Centroid Forest': rule_centroid_forest_iterations
        }
        
        self.leaves_iteration_df = pd.DataFrame(data_frame3)
        self.leaves_iteration_df.index = leaves
        
        
        
        
        
        
    def run_depths_cost(self, min_depth, max_depth, max_leaves, clusterer):
        """
        Runs the depths x cost experiment by iterating through the maximum depths parameter
        for each algorithm.
        
        Args:
            min_depth (int): Minimum depth. 
            max_depth (int): Maximum depth.
            max_leaves (int): Maximum number of leaves to include in the tree.
            clusterer (RuleClustering): Rule clustering object to cluster leaves.
        """
        # Revert the seed in case it has been advanced anywhere else:
        np.random.seed(self.seed)
        
        if 2**min_depth < self.n_clusters:
            raise ValueError('2^(minimum depth) must be greater than the number of clusters. ')
        
        depths = list(range(min_depth,max_depth + 1))
        
        # Track cost by leaves:
        rule_cart_costs =[]
        rule_unsupervised_costs = []
        rule_unsupervised_oblique_costs = []
        rule_centroid_costs = []
        rule_centroid_oblique_costs = []
        #rule_clustered_costs = []
        #rule_clustered_oblique_costs = []
        
        # Track iterations used for rule clustering
        rule_cart_iterations =[]
        rule_unsupervised_iterations = []
        rule_unsupervised_oblique_iterations = []
        rule_centroid_iterations = []
        rule_centroid_oblique_iterations = []
        #rule_clustered_iterations = []
        #rule_clustered_oblique_iterations = []

        # Expand the trees and compute cost:
        for d in depths:
            print("Depth: " + str(d))
            
            # Rule CART:
            dtree_ = DecisionTreeClassifier(criterion='gini', max_leaf_nodes = max_leaves, 
                                            max_depth = d)
            dtree_.fit(self.data, self.kmeans.labels_)
            cart_tree_ = ConvertSklearn(dtree_.tree_, self.data)

            rule_cart_ = clusterer(cart_tree_, k_clusters = self.n_clusters, init = 'manual',
                                   center_init = self.kmeans_centers)
            rule_cart_.fit(self.data)
            rule_cart_costs.append(rule_cart_.cost)
            rule_cart_iterations.append(rule_cart_.iterations)

            # Rule Clustering for Unsupervised Tree
            unsupervised_tree_ = UnsupervisedTree(splits = 'axis', max_leaf_nodes = max_leaves, 
                                                  max_depth = d, norm = 2)
            unsupervised_tree_.fit(self.data)
            rule_kmeans_ = clusterer(unsupervised_tree_, k_clusters = self.n_clusters,
                                     init = 'manual', 
                                     center_init = self.kmeans_centers, random_seed = self.seed)
            rule_kmeans_.fit(self.data, fit_rules = False)
            rule_unsupervised_costs.append(rule_kmeans_.cost)
            rule_unsupervised_iterations.append(rule_kmeans_.iterations)
            
            
            # Rule Clustering for Unsupervised Oblique Tree
            unsupervised_oblique_tree_ = UnsupervisedTree(splits = 'oblique', 
                                                          max_leaf_nodes = max_leaves, 
                                                          max_depth = d, norm = 2)
            unsupervised_oblique_tree_.fit(self.data)
            rule_kmeans_ = clusterer(unsupervised_oblique_tree_, k_clusters = self.n_clusters,
                                     init = 'manual', center_init = self.kmeans_centers, 
                                     random_seed = self.seed)
            rule_kmeans_.fit(self.data, fit_rules = False)
            rule_unsupervised_oblique_costs.append(rule_kmeans_.cost)
            rule_unsupervised_oblique_iterations.append(rule_kmeans_.iterations)

            # Rule Clustering for Centroid Tree
            centroid_tree_ = CentroidTree(splits = 'axis', max_leaf_nodes = max_leaves, 
                                          max_depth = d, norm = 2, center_init = 'manual', 
                                          centers = self.kmeans_centers)
            centroid_tree_.fit(self.data)
            rule_kmeans_ = clusterer(centroid_tree_, k_clusters = self.n_clusters, init = 'manual',
                                     center_init = centroid_tree_.centers, random_seed = self.seed)
            rule_kmeans_.fit(self.data, fit_rules = False)
            rule_centroid_costs.append(rule_kmeans_.cost)
            rule_centroid_iterations.append(rule_kmeans_.iterations)

            # Rule Clustering for Centroid Oblique Tree
            centroid_oblique_tree_ = CentroidTree(splits = 'oblique', max_leaf_nodes = max_leaves,
                                                  max_depth = d, norm = 2, center_init = 'manual',
                                                  centers = self.kmeans_centers)
            centroid_oblique_tree_.fit(self.data)
            rule_kmeans_ = clusterer(centroid_oblique_tree_, k_clusters = self.n_clusters,
                                     init = 'manual', center_init = centroid_oblique_tree_.centers,
                                     random_seed = self.seed)
            rule_kmeans_.fit(self.data, fit_rules = False)
            rule_centroid_oblique_costs.append(rule_kmeans_.cost)
            rule_centroid_oblique_iterations.append(rule_kmeans_.iterations)
            
            # Clustered Tree
            '''
            clustered_tree_ = RuleClusteredTree(splits = 'axis', max_leaf_nodes = max_leaves, 
                                                max_depth = d, norm = 2, center_init = 'manual',
                                                centers = self.kmeans_centers,
                                        clusterer = clusterer, start_centers = self.kmeans_centers)

            clustered_tree_.fit(self.data)
            rule_clustered_costs.append(clustered_tree_.clustering_cost)
            rule_clustered_iterations.append(clustered_tree_.clustering_iterations)
            
            
            # Clustered Oblique Tree
            clustered_oblique_tree_ = RuleClusteredTree(splits = 'oblique', 
                                                        max_leaf_nodes = max_leaves, max_depth = d,
                                                        norm = 2, center_init = 'manual',
                                                        centers = self.kmeans_centers, 
                                                        clusterer = clusterer, 
                                                        start_centers = self.kmeans_centers)
            clustered_oblique_tree_.fit(self.data)
            rule_clustered_oblique_costs.append(clustered_oblique_tree_.clustering_cost)
            rule_clustered_oblique_iterations.append(clustered_oblique_tree_.clustering_iterations)
            '''


        data_frame = {
            'IMM': [self.imm_cost if d >= self.imm_depth else np.nan for d in depths],
            'RandomIMM': ([self.randomized_imm_cost if d >= self.randomized_imm_depth
                           else np.nan for d in depths]),
            'CART': rule_cart_costs,
            'Unsupervised': rule_unsupervised_costs,
            'Unsupervised Obl.': rule_unsupervised_oblique_costs,
            'Centroid': rule_centroid_costs,
            'Centroid Obl.': rule_centroid_oblique_costs
            #'Hybrid': rule_clustered_costs,
            #'Hybrid Obl.': rule_clustered_oblique_costs
        }
        
        self.depths_cost_df = pd.DataFrame(data_frame)
        self.depths_cost_df.index = depths
        
        
        data_frame3 = {
            'CART': rule_cart_iterations,
            'Unsupervised': rule_unsupervised_iterations,
            'Unsupervised Obl.': rule_unsupervised_oblique_iterations,
            'Centroid': rule_centroid_iterations,
            'Centroid Obl.': rule_centroid_oblique_iterations
            #'Hybrid': rule_clustered_iterations,
            #'Hybrid Obl.': rule_clustered_oblique_iterations
        }
        
        self.depths_iteration_df = pd.DataFrame(data_frame3)
        self.depths_iteration_df.index = depths