import numpy as np
import itertools
from joblib import Parallel, delayed
from multiprocessing import shared_memory
from ._tree import Tree
from ..utils import *




####################################################################################################

def oblique_pair(args):
    pair, slopes, indices, min_points_leaf, shm_name, shape, cost_func = args
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    X = np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)
    X_ = X[indices, :]
    X_pair = X_[:, pair]
    best_split_val = np.inf
    best_split = None

    for i, x in enumerate(X_pair):
        for j, slope in enumerate(slopes):
            weights = np.array([slope[1], -slope[0]])
            threshold = slope[1] * x[0] - slope[0] * x[1]
            split = (list(pair), weights, threshold)

            left_mask = np.dot(X_pair, weights) <= threshold
            right_mask = ~left_mask
            left_indices = indices[left_mask]
            right_indices = indices[right_mask]

            if (np.sum(left_mask) < min_points_leaf or
                np.sum(right_mask) < min_points_leaf):
                split_val = np.inf
            else:
                X1_cost = cost_func(left_indices)
                X2_cost = cost_func(right_indices)
                split_val = X1_cost + X2_cost

            if split_val < best_split_val:
                best_split_val = split_val
                best_split = split

    return best_split_val, best_split


def oblique_pair2(args):
    pair, indices, min_points_leaf, shm_name, shape, cost_func = args
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    X = np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)
    X_ = X[indices, :]
    X_pair = X_[:, pair]
    # This makes sure axis aligned splits are accounted for:
    X_pair_ = np.vstack((X_pair, np.array([[1,0], [0,1]])))
    
    best_split_val = np.inf
    best_split = None
    
    for i,x in enumerate(X_pair_):
        # Draw unit vector through x from origin
        unit_vec = x/np.linalg.norm(x)
        
        # Find projections onto unit vector:
        proj_dists = np.sort(np.dot(X_pair, unit_vec))
        proj_dists_idx = np.argsort(np.dot(X_pair, unit_vec))
        
        for j, dist in enumerate(proj_dists[:-1]):
            weights = dist*unit_vec
            threshold = weights[0]**2 + weights[1]**2
            
            split = (list(pair), weights, threshold)
            
            left_indices = indices[proj_dists_idx[:(j + 1)]]
            right_indices = indices[proj_dists_idx[(j + 1):]]
            
            if (len(left_indices) < min_points_leaf or
                len(right_indices) < min_points_leaf):
                split_val = np.inf
            else:
                X1_cost = cost_func(left_indices)
                
                X2_cost = cost_func(right_indices)
                
                split_val = X1_cost + X2_cost
            
            #print(split_val)
            if split_val < best_split_val:
                best_split_val = split_val
                best_split = split
                
    return best_split_val, best_split
    



class LinearTree(Tree):
    """
    Base class for a Tree object which splits leaf nodes based upon linear conditions. 
    Allows for a choice between axis aligned splits, or oblique splits.
    """
    
    def __init__(self, splits = 'axis', max_leaf_nodes=None, max_depth=None, min_points_leaf = 1, 
                 norm = 2,center_init = None, centers = None, n_centers = None,
                 cluster_leaves = False, clusterer = None, random_seed = None,
                 feature_labels = None):
        """
        New Args:
            splits (str): May take values 'axis' or 'oblique' that
                decide on how to compute leaf splits.
        """
       
        # Initialize everything according to the base class.
        super().__init__(max_leaf_nodes, max_depth, min_points_leaf, norm, center_init, centers,
                        n_centers,
                        cluster_leaves, clusterer, random_seed, feature_labels)
        
        
        if splits not in ['axis', 'oblique', 'oblique-full']:
            raise ValueError("Must choose either 'axis' or 'oblique' for split method")
        else:
            self.splits = splits
    
    
    def _find_best_split(self, indices):
        """
        Finds the best axis aligned or oblique split by searching through possibilities
        outputting one with the best cost.

        Args:
            indices (np.ndarray[int]): Indices of data points to compute cost with. 

        Returns:
        
            Tuple(List[int], List[float], float): A (features, weights, threshold) split pair. 
        """
        # Reset the indices
        self.indices = indices
        
        X_ = self.X[indices,:]
        n_, d = X_.shape
        
        best_split = None
        best_split_val = np.inf
        
        if self.splits == 'axis' and self.norm == 2 and self.centers is None:
            """
            The following optimized version is attributed to 
            [Dasgupta, Frost, Moshkovitz, Rashtchian '20] in their paper
            'Explainable k-Means and k-Medians Clustering' 
            """
            n, d = X_.shape
            u = np.linalg.norm(X_)**2
            
            best_split = None
            best_split_val = np.inf
            for i in range(X_.shape[1]):
                s = np.zeros(d)
                r = np.sum(X_, axis = 0)
                order = np.argsort(X_[:, i])
                
                for j, idx in enumerate(order[:-1]):
                    threshold = X_[idx, i]
                    s = s + X_[idx, :]
                    r = r - X_[idx, :]
                    split_val = u - np.sum(s**2)/(j + 1) - np.sum(r**2)/(n - j - 1)
                    
                    if split_val < best_split_val:
                        best_split_val = split_val
                        # Axis aligned split:
                        best_split = ([i], [1], threshold)
                        
        
        elif self.splits == 'axis':
            for feature in range(d):
                unique_vals = np.unique(X_[:,feature])
                for threshold in unique_vals:
                    split = ([feature], [1], threshold)

                    left_mask = X_[:, feature] <= threshold
                    right_mask = ~left_mask
                    left_indices = indices[left_mask]
                    right_indices = indices[right_mask]
                    
                    if (np.sum(left_mask) < self.min_points_leaf or 
                        np.sum(right_mask) < self.min_points_leaf):
                        split_val = np.inf
                    else:
                        X1 = X_[left_mask, :]
                        X1_cost = self._cost(left_indices)
                        
                        X2 = X_[right_mask, :]
                        X2_cost = self._cost(right_indices)
                        
                        split_val = X1_cost + X2_cost
                    
                    if split_val < best_split_val:
                        best_split_val = split_val
                        best_split = split
                        
                        
        elif self.splits == 'oblique':
            feature_pairs = list(itertools.combinations(list(range(d)), 2))
            slopes = np.array([[0,1],
                    [1,2],
                    [1,1],
                    [2,1],
                    [1,0],
                    [2,-1],
                    [1,-1],
                    [1,-2]])
            
            '''
            for pair in feature_pairs:
                X_pair = X_[:, pair]
                for i,x in enumerate(X_pair):
                    for j, slope in enumerate(slopes):
                        weights = np.array([slope[1], -slope[0]])
                        threshold = slope[1]*x[0] - slope[0]*x[1]
                        split = (list(pair), weights, threshold)

                        left_mask = np.dot(X_pair, weights) <= threshold
                        right_mask = ~left_mask
                        left_indices = indices[left_mask]
                        right_indices = indices[right_mask]
                        
                        if (np.sum(left_mask) < self.min_points_leaf or
                            np.sum(right_mask) < self.min_points_leaf):
                            split_val = np.inf
                        else:
                            X1_cost = self._cost(left_indices)
                            
                            X2_cost = self._cost(right_indices)
                            
                            split_val = X1_cost + X2_cost
                        
                        if split_val < best_split_val:
                            best_split_val = split_val
                            best_split = split
            '''
            
            X_shm = shared_memory.SharedMemory(create=True, size=self.X.nbytes)
            X_shared = np.ndarray(self.X.shape, dtype=self.X.dtype, buffer=X_shm.buf)
            np.copyto(X_shared, self.X)

            try:
                results = Parallel(n_jobs=15)(
                    delayed(oblique_pair)(
                        (pair, slopes, indices, self.min_points_leaf, X_shm.name, self.X.shape, self._cost)
                    ) for pair in feature_pairs
                )
                best_split_val, best_split = min(results, key=lambda x: x[0])
            finally:
                X_shm.close()
                X_shm.unlink()
                         
                            
        elif self.splits == 'oblique-full':
            feature_pairs = list(itertools.combinations(list(range(d)), 2))
            X_shm = shared_memory.SharedMemory(create=True, size=self.X.nbytes)
            X_shared = np.ndarray(self.X.shape, dtype=self.X.dtype, buffer=X_shm.buf)
            np.copyto(X_shared, self.X)

            try:
                results = Parallel(n_jobs=15)(
                    delayed(oblique_pair2)(
                        (pair, indices, self.min_points_leaf, X_shm.name, self.X.shape, self._cost)
                    ) for pair in feature_pairs
                )
                best_split_val, best_split = min(results, key=lambda x: x[0])
            finally:
                X_shm.close()
                X_shm.unlink()
                
                            
        return best_split_val, best_split