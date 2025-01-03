import numpy as np
from ._tree import Tree

class RandomTree(Tree):
    """
    Inherits from the Tree class to implement a decision tree in which axis aligned 
    split criterion are randomly chosen from the input space.
    
    This model is a product of work by [Galmath, Jia, Polak, Svensson 2021] in their paper 
    titled "Nearly-Tight and Oblivious Algorithms for Explainable Clustering."
    
    NOTE: In order to imitate results from [Galmath, Jia, Polak, Svensson 2021], fit this model on 
        a dataset consisting only of centers or representative points.
    """
    def __init__(self, max_leaf_nodes=None, max_depth=None, min_points_leaf = 1, random_seed = None,
                 feature_labels = None):
        """ 
        Args:
            max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
                Defaults to None.
                
            max_depth (int, optional): Optional constraint for maximum depth. 
                Defaults to None.
                
            min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
                within a single leaf. Defaults to 1.
                
            random_seed (int, optional): Random seed. In the Tree object randomness is only ever 
                used for breaking ties between nodes, or if you are using a RandomTree!
                
            feature_labels (List[str]): Iterable object with strings representing feature names. 
            
        Attributes:
            heap (heapq list): Maintains the heap structure of the tree.
            
            leaf_count (int): Number of leaves in the tree.
            
            node_count (int): Number of nodes in the tree.
        
        """
        super().__init__(max_leaf_nodes = max_leaf_nodes, max_depth = max_depth, 
                         min_points_leaf = min_points_leaf, random_seed = random_seed,
                         feature_labels = feature_labels)
        
    def _cost(self, indices):
        """
        Assigns a random cost of a subset of data points.

        Args:
            indices (np.ndarray[int]): Indices of data points to compute cost with. 
            
        Returns:
            (float): Cost of the given data. 
        """
        
        # Costs are given by a uniform random sample, meaning nodes will be chosen randomly 
        # to split from 
        return np.random.uniform()


    def _find_best_split(self, indices):
            """
            Randomly chooses an axis aligned threshold value to split upon 
            given some input dataset X. The randomly sampled split must separate at least one
            pair of points from X_.

            Args:
                X_ (np.ndarray): Input dataset.
                
            Returns:
                Tuple(int, float): A (feature, threshold) split pair. 
            """
            X_ = self.X[indices, :]
            found = False
            split = None
            split_cost = None
            while not found:
                # Randomly sample a feature and a value for an axis aligned split:
                rand_feature = np.random.choice(range(X_.shape[1]))
                interval = (np.min(X_[:, rand_feature]), np.max(X_[:, rand_feature]))
                rand_cut = np.random.uniform(interval[0], interval[1])
                left_mask = X_[:, rand_feature] <= rand_cut
                right_mask = ~left_mask
                split_cost = self._cost(X_[left_mask]) + self._cost(X_[right_mask])
                
                # If sampled split separates at least two centers, then accept:
                left_branch = X_[left_mask]
                right_branch = X_[right_mask]
                if len(left_branch) > 0 and len(right_branch) > 0:
                    found = True
                    split = ([rand_feature], [1], rand_cut)

            return split_cost, split
        
####################################################################################################