import numpy as np
import heapq
import itertools
import copy
from joblib import Parallel, delayed
#from multiprocessing import Pool
from multiprocessing import shared_memory
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from intercluster import *
from ..utils import *


####################################################################################################


class Node():
    """
    Node object to be used within a logical decision tree, with simple 
    axis aligned splitting conditions.
    
    Args:
        None
    Attributes:
        random_val (float): Uniform random sample used to split ties / and create
            a relative order of nodes. Note that in the creation of the tree, nodes are chosen
            based on their cost, and this value is simply used to break ties. 
            
        type (str): Internally set to 'node' or 'leaf' depending on if the node is a
            normal node or a leaf node.
            
        label (int): (For leaf nodes only) Prediction label to be associated with this node.
        
        features (List[int]): The axis aligned feature to split on. 
        
        feature_labels (List[str]): Names for the given feature (for printing and display). 
        
        features (List[int]): The axis aligned features to split on. 
        
        weights (List[float]): Weights for each of the splitting features
        
        threshold (float): The threshold value to split on.
        
        left_child (Node): (For non-leaf nodes only) Pointer to the left branch of the current node.
        
        right_child (Node): (For non-leaf nodes only) Pointer to the right branch of the 
            current node. 
            
        indices (np.ndarray): The indices of data points belonging to this node.
        
        size (int): The number of data points belonging to this node
        
        cost (float): The cost associated with points belonging to this node. 
        
    """
    def __init__(self):
        self.random_val = np.random.uniform()
        self.type = None
        self.label = None
        self.features = None
        self.feature_labels = None
        self.weights = None
        self.threshold = None
        self.left_child = None
        self.right_child = None
        self.indices = None
        self.size = None
        self.cost = None
    
    def __lt__(self, other):
        """
        Creates an ordering of nodes by defining a < comparison. 
        This simply order nodes randomly via randomly sampled values. 

        Args:
            other (Node): Another node to compare with.

        Returns:
            (bool): Evaluation of the comparison. 
        """
        return self.random_val < other.random_val
    
    def tree_node(self, features, weights, threshold, left_child, right_child, 
                  indices, cost, feature_labels):
        """
        Initializes this as a normal node in the tree.

        Args:
            features (List[int]): The axis aligned features to split on. 
        
            weights (List[float]): Weights for each of the splitting features
        
            threshold (float): The threshold value to split on.
            
            left_child (Node): Pointer to the left branch of the current node.
            
            right_child (Node): Pointer to the right branch of the 
                current node. 
                
            indices (np.ndarray): The subset of data indices belonging to this node.
             
            cost (float): The cost associated with points belonging to this node. 
            
            feature_label (str): Name for the given feature (for printing and display). 
        """
        self.type = 'node'
        self.features = features
        self.weights = weights
        self.threshold = threshold
        self.left_child = left_child 
        self.right_child = right_child
        self.indices = indices
        self.cost = cost
        self.feature_labels = feature_labels
        self.size = len(indices)
        
        # reset label if this was previously a leaf node
        self.label = None
        
    def leaf_node(self, label, indices, cost):
        """
        Initializes this to be a leaf node in the tree.

        Args:
            label (int): Prediction label to be associated with this node.
            
            indices (np.ndarray): The subset of data indices belonging to this node.
            
            cost (float): The cost associated with points belonging to this node. 
        """
        self.type = 'leaf'
        self.label = label
        self.indices = indices
        self.cost = cost
        self.size = len(indices)


####################################################################################################


class Tree():
    """
    Base class for a Tree object. 
    """
    def __init__(self, max_leaf_nodes=None, max_depth=None, min_points_leaf = 1, norm = 2,
                 center_init = None, centers = None, n_centers = None, cluster_leaves = False, 
                 clusterer = None, random_seed = None, feature_labels = None):
        """
        Args:
            max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
                Defaults to None.
                
            max_depth (int, optional): Optional constraint for maximum depth. 
                Defaults to None.
                
            min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
                within a single leaf. Defaults to 1.
                
            norm (int, optional): Takes values 1 or 2. If norm = 1, compute distances using the 
                1 norm. If 2, compute distances with the squared two norm. Defaults to 2.
                
            center_init (str, optional): Center initialization method. Included options are 
                'k-means' which runs a k-means algorithm and uses the output centers,
                'random++' which uses a randomized k-means++ initialization, or 
                'manual' which assumes an input array of centers (in the next parameter). 
                Defaults to None in which case no centers are initialized.
                
            centers (np.ndarray, optional): Input list of reference centers to calculate cost with. 
                Defaults to None.
                
            n_centers (int, optional): Number of centers to use. Defaults to None. 
                
            cluster_leaves (bool, optional): If True, update the array centers after every 
                leaf expansion by clustering leaves. Only performs updates once the 
                number of leaves is > n_centers.
            
            clusterer (RuleClustering, optional): Clustering object used to cluster leaves 
                and find new centers. Defaults to None, not used if center_updates is False.
                
            random_seed (int, optional): Random seed. In the Tree object randomness is only ever 
                used for breaking ties between nodes, or if you are using a RandomTree!
                
            feature_labels (List[str]): Iterable object with strings representing feature names. 
            
            
        Attributes:
            X (np.ndarray): Size (n x m) dataset passed as input in the fitting process.
            
            root (Node): Root node of the tree.
        
            heap (heapq list): Maintains the heap structure of the tree.
            
            leaf_count (int): Number of leaves in the tree.
            
            node_count (int): Number of nodes in the tree.
            
            n_centers (int): The number of centers to use for computation of cost 
                or center updates.
                
            depth (int): The maximum depth of the tree.
            
            center_dists (np.ndarray): If being fitted to an (n x m) dataset, computes a
                n x k array of distances (measured either with squared 2 norm or 1 norm) 
                from every point to every center.
                
        """
        
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_points_leaf = min_points_leaf
        
        if norm in [1,2]:
            self.norm = norm
        else:
            raise ValueError('Norm must either be 1 or 2.')
        
        if center_init in [None, 'k-means', 'random++', 'manual']:
            self.center_init = center_init
        else:
            raise ValueError('Unsupported initialization method.')
        
        
        if self.center_init == 'manual' and (centers is not None):            
            self.centers = copy.deepcopy(centers)
            self.n_centers = len(self.centers)
            
        elif self.center_init == 'manual':
            raise ValueError('Must provide an input array of centers for manual initialization.')
        
        else:
            self.centers = centers
            self.n_centers = n_centers
        
        
        if cluster_leaves and (clusterer is None):
            raise ValueError('Must provide clustering object in order to cluster leaves.')
            
        self.cluster_leaves = cluster_leaves
        self.clusterer = clusterer
        
        self.feature_labels = feature_labels
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.random_seed = random_seed
        
        self.X = None
        self.root = None
        self.heap = []
        self.leaf_count = 0
        self.node_count = 0
        self.depth = 0
        self.center_dists = None
        self.indices = None
        self.clustering_iterations = 0
        
        
    def _update_center_dists(self):
        """
        Updates the center_dists array, which tracks distances from all points in self.X
        to all centers.
        """
        pass
    
    def _cost(self, indices):
        """
        Calculates the cost of a subset of data points.

        Args:
            indices (np.ndarray[int]): Indices of data points to compute cost with. 
            
        Returns:
            (float): Cost of the given data subset. 
        """
        return


    def _find_best_split(self, indices):
        """
        Chooses features, weights and a threshold to split an input data subset X_, defined by the
        a list of indices with which to access the dataset X.

        Args:
            indices (np.ndarray[int]): Indices of data points to compute cost with.
            
        Returns:
                Tuple(List[int], List[float], float): A (features, weights, threshold) split pair. 
        """
        return
    

    def fit(self, X, iterative = False, init_steps = None):
        """
        Initiates and builds a decision tree around a given dataset. 

        Args:
            X (np.ndarray): Input dataset.
            
            iterative (bool, optional): If True, allows the user to expand the tree iteratively by 
                manually calling .fit_step() until maximum conditions have been reached.
                Defaults to False.
                
            init_steps (int, optional): If init_steps is provided and iterative is True, the tree
                algorithm will first perform the input number of fitting steps, before 
                defaulting to a manual, iterative fitting process.
        """
        # Create Dataset
        self.X = X
        
        # Reset the heap and tree:
        self.heap = []
        self.leaf_count = 0
        self.node_count = 0
        
        # Initialize centers:
        if self.center_init == 'k-means':
            kmeans = KMeans(n_clusters=self.n_centers, random_state=self.random_seed,
                            n_init="auto").fit(X)
            self.centers = kmeans.cluster_centers_
            self.n_centers = len(self.centers)
            
        elif self.center_init == 'random++':
            self.centers = kmeans_plus_plus_initialization(X, self.n_centers, self.random_seed)
            self.n_centers = len(self.centers)
        
        # If using reference centers, initialize center_dists array:
        self._update_center_dists()
        
        # Set feature labels if not set already:
        if self.feature_labels is None:
            self.feature_labels = [None]*X.shape[1]
        else:
            if not len(self.feature_labels) == X.shape[1]:
                raise ValueError('Feature labels must match the shape of the data.')
        
        # if stopping criteria weren't provided, set to the maximum possible
        if self.max_leaf_nodes is None:
            self.max_leaf_nodes = len(X)
        
        if self.max_depth is None:
            self.max_depth = len(X) - 1
        
        # Keeping a heap queue for the current leaf nodes in the tree.
        # It prioritizes splitting the leaves with the largest scores (gain in cost performance).
        # NOTE: I store -1*scores because heapq pops items with minimum cost by default.
        # Heap leaf object: (-1*score, Node object, data indices subset, depth, split)
        # Each split (features, weights, thresholds) is pre-computed,
        # before placement into the heap.
        all_indices = np.array(range(len(self.X)))
        leaf_cost = self._cost(all_indices)
        self.root = Node()
        self.root.leaf_node(self.leaf_count, all_indices, leaf_cost)
        split_cost, split = self._find_best_split(all_indices)
        score = (leaf_cost - split_cost)
        heapq.heappush(self.heap, (-1*score, self.root, split, all_indices, 0))
        self.leaf_count += 1
        
        
        if not iterative:
            # Fits the tree in one go.
            while len(self.heap) > 0:
                self.fit_step()
        else:
            # Fits the tree iteratively
            if init_steps is not None:
                # If we need to find n centers before clustering leaves, 
                # we should do so before anything else.
                for _ in range(init_steps - 1):
                    self.fit_step()
                    
            else:
                # Otherwise, allows the user to control the entire fitting process.
                pass
                                
            
            
    def fit_step(self):
        """
        Performs a single fitting step in which a leaf is split, and depending on the 
        type of tree being used, the leaves may be clustered and centers may be updated.
        """
        if len(self.heap) == 0:
            raise ValueError("Empty heap, no more leaves to split or max conditions reached.")
        
        self.split_leaf()
                
        if self.cluster_leaves and self.leaf_count >= self.n_centers:
            clustering = self.clusterer(self, k_clusters = self.n_centers, 
                                        init = 'manual', center_init = self.centers, 
                                        random_seed = self.random_seed)
                
            clustering.fit(self.X, fit_rules = False)
            self.centers = clustering.centers
            self.clustering_cost = clustering.cost
            self.clustering_iterations = clustering.iterations
            self._update_center_dists()
        

    def split_leaf(self):
        """
        Builds the decision tree by iteratively selecting conditions and splitting leaf nodes.
        """
        
        # pop an object from the heap
        leaf_obj = heapq.heappop(self.heap)
        node_obj : Node = leaf_obj[1]
        split_obj = leaf_obj[2]
        indices_ = leaf_obj[3]
        depth = leaf_obj[4]
        
        n_ = len(indices_)
        X_ = self.X[indices_, :]
        
        # If we've reached the max depth or max number of leaf nodes -- stop recursing 
        if (depth >= self.max_depth) or (self.leaf_count >= (self.max_leaf_nodes)) or (n_ <= 1):
            #class_label = self.leaf_count 
            #self.leaf_count += 1
            #node_obj.leaf_node(class_label, indices_, self._cost(indices_))
            pass


        # Otherwise, find the best split and recurse into left and right branches
        else:
            features, weights, threshold = split_obj
            
            # Find new split
            left_mask = np.dot(X_[:, features], weights) <= threshold
            right_mask = ~left_mask
            left_indices = indices_[left_mask]
            right_indices = indices_[right_mask]
            
            # Calculate cost of left and right branches
            left_leaf_cost = self._cost(left_indices)
            right_leaf_cost = self._cost(right_indices)
            
            # Create New leaf nodes
            left_node = Node()
            left_node.leaf_node(label = node_obj.label, indices = left_indices, 
                                cost = left_leaf_cost)
            right_node = Node()
            right_node.leaf_node(label = self.leaf_count, indices = right_indices, 
                                cost = right_leaf_cost)
            self.leaf_count += 1
            
            # And push them into the heap:
            if len(left_indices) > 1:
                left_split_cost, left_split = self._find_best_split(left_indices)
            else:
                left_split_cost = 0
                left_split = None
                
            left_score = left_leaf_cost - left_split_cost
            left_obj = (-1*left_score, left_node, left_split, left_indices, depth + 1)
            
            if len(right_indices) > 1:
                right_split_cost, right_split = self._find_best_split(right_indices)
            else:
                right_split_cost = 0
                right_split = None
                
            right_score = right_leaf_cost - right_split_cost
            right_obj = (-1*right_score, right_node, right_split, right_indices, depth + 1)
            
            heapq.heappush(self.heap, left_obj)
            heapq.heappush(self.heap, right_obj)
            
            
            # Transform the splitted node into an internal tree node:
            node_obj.tree_node(features, weights, threshold, left_node, right_node, 
                               indices_, self._cost(indices_),
                               feature_labels = [self.feature_labels[f] for f in features])
            
            self.node_count += 1
            if depth + 1 > self.depth:
                self.depth = depth + 1


    def _predict_sample(self, sample, node):
        """
        Predicts the class label of a single sample from the data.

        Args:
            sample (np.ndarray): Single m dimensional data point. 
            node (Node): Starting node to recurse through the tree with. 

        Returns:
            (int): Class label for sample.
        """
        if node.type == 'leaf':
            return node.label
        
        if np.dot(sample[node.features], node.weights) <= node.threshold:
            return self._predict_sample(sample, node.left_child)
        else:
            return self._predict_sample(sample, node.right_child)
        
    def predict(self, X):
        """
        Predicts the class labels of an input dataset X by recursing through the tree to 
        find where data points fall into leaf nodes.

        Args:
            X (np.ndarray): Input n x m dataset

        Returns:
            (np.ndarray): Length n array of class labels. 
        """
        return np.array([self._predict_sample(sample, self.root) for sample in X])

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
        
####################################################################################################

class UnsupervisedTree(LinearTree):
    """
    Inherits from the Tree class to implement a decision tree in which 
    split criterion are chosen so that points in any leaf node are close
    in distance to their means or medians. 
   
    Args:
        splits (str): May take values 'axis' or 'oblique' that decide on how to compute leaf splits.
        
        max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
            Defaults to None.
            
        max_depth (int, optional): Optional constraint for maximum depth. 
            Defaults to None.
            
        min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
            within a single leaf. Defaults to 1.
            
        norm (int, optional): Takes values 1 or 2. If norm = 1, compute distances using the 
            1 norm. If 2, compute distances with the squared two norm. Defaults to 2.
            
        random_seed (int, optional): Random seed. In the Tree object randomness is only ever 
            used for breaking ties between nodes, or if you are using a RandomTree!
            
        feature_labels (List[str]): Iterable object with strings representing feature names. 
            
            
        Attributes:
            X (np.ndarray): Size (n x m) dataset passed as input in the fitting process.
            
            root (Node): Root node of the tree.
        
            heap (heapq list): Maintains the heap structure of the tree.
            
            leaf_count (int): Number of leaves in the tree.
            
            node_count (int): Number of nodes in the tree.
                
        """
    def __init__(self, splits = 'axis', max_leaf_nodes=None, max_depth=None, min_points_leaf = 1, 
                 norm = 2, random_seed = None, feature_labels = None):
        
        super().__init__(splits = splits, max_leaf_nodes = max_leaf_nodes, max_depth = max_depth, 
                         min_points_leaf = min_points_leaf, norm = norm, random_seed = random_seed,
                         feature_labels = feature_labels)
        
        
    def _cost(self, indices):
        """
        Given a set of indices defining a data subset X_ --
        which may be thought of as a subset of points reaching a given node in the tree --
        compute a cost for X_.
        
        In an unsupervised tree this amounts to find the sum of distances to 
        means or medians (using squared 2 norm or 1 norm respectively).

        Args:
            indices (np.ndarray[int]): Indices of data points to compute cost with. 
            
        Returns:
            (float): Cost of the given data. 
        """
        
        if len(indices) == 0:
            return np.inf
        else:
            if self.norm == 2:
                X_ = self.X[indices,:]
                mu = np.mean(X_, axis = 0)
                cost = np.sum(np.linalg.norm(X_ - mu, axis = 1)**2) #/len(X_)
                
            elif self.norm == 1:
                X_ = self.X[indices, :]
                eta = np.median(X_, axis = 0)
                cost = np.sum(np.abs(X_ - eta))
                
            return cost

    
####################################################################################################

class CentroidTree(LinearTree):
    """
    Inherits from the Tree class to implement a decision tree in which 
    axis aligned split criterion are chosen so that points in any leaf node are close
    in distance to their closest center or centroid, from a set of input centers.
    
    When input centers are those from the output of a k-means algorithm, this is 
    equivalent to the ExKMC algorithm, as designed by [Frost, Moshkovitz, Rashtchian '20]
    in their paper titled: 'ExKMC: Expanding Explainable k-Means Clustering.'
    
    Args:
        splits (str): May take values 'axis' or 'oblique' that decide on how to compute leaf splits.
        
        max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
            Defaults to None.
            
        max_depth (int, optional): Optional constraint for maximum depth. 
            Defaults to None.
            
        min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
            within a single leaf. Defaults to 1.
            
        norm (int, optional): Takes values 1 or 2. If norm = 1, compute distances using the 
            1 norm. If 2, compute distances with the squared two norm. Defaults to 2.
            
        center_init (str, optional): Center initialization method. Included options are 
            'k-means' which runs a k-means algorithm and uses the output centers,
            'random++' which uses a randomized k-means++ initialization, or 
            'manual' which assumes an input array of centers (in the next parameter). 
            Defaults to None in which case no centers are initialized.
            
        centers (np.ndarray, optional): Input list of reference centers to calculate cost with. 
            Defaults to None.
            
        random_seed (int, optional): Random seed. In the Tree object randomness is only ever 
            used for breaking ties between nodes, or if you are using a RandomTree!
            
        feature_labels (List[str]): Iterable object with strings representing feature names. 
            
            
    Attributes:
        X (np.ndarray): Size (n x m) dataset passed as input in the fitting process.
        
        root (Node): Root node of the tree.
    
        heap (heapq list): Maintains the heap structure of the tree.
        
        leaf_count (int): Number of leaves in the tree.
        
        node_count (int): Number of nodes in the tree.
        
        n_centers (int, optional): The number of centers to use for computation of cost 
            or center updates.
        
        center_dists (np.ndarray): If being fitted to an (n x m) dataset, computes a
            n x k array of distances (measured either with squared 2 norm or 1 norm) 
            from every point to every center.
                
    """
    
    def __init__(self, splits = 'axis', max_leaf_nodes=None, max_depth=None, min_points_leaf = 1, norm = 2,
                 center_init = 'k-means', centers = None, n_centers = None, random_seed = None,
                 feature_labels = None):
        
        super().__init__(splits = splits, max_leaf_nodes = max_leaf_nodes, max_depth = max_depth, 
                         min_points_leaf = min_points_leaf, norm = norm, center_init = center_init,
                         centers = centers, n_centers = n_centers, random_seed = random_seed, 
                         feature_labels = feature_labels)
        
        if center_init is None:
            raise ValueError('Must provide some method of initializing centers.')
        
    
    def _update_center_dists(self):
        """
        Updates the center_dists array, which tracks distances from all points in self.X
        to all centers. This helps with computational efficiency, since we won't 
        need to recompute this when searching through splits.
        """
        if self.norm == 2:
            diffs = self.X[np.newaxis, :, :] - self.centers[:, np.newaxis, :]
            distances = np.sum((diffs)**2, axis=-1)
            self.center_dists = distances.T
            
        elif self.norm == 1:
            diffs = self.X[np.newaxis, :, :] - self.centers[:, np.newaxis, :]
            distances = np.sum(np.abs(diffs), axis=-1)
            self.center_dists = distances.T
            
        
    def _cost(self, indices):
        """
        Given a set of indices defining a data subset X_ --
        which may be thought of as a subset of points reaching a given node in the tree --
        compute a cost for X_.
        
        In a centroid tree this amounts to find the sum of distances to 
        centers (using squared 2 norm or 1 norm respectively).
        
        If self.centers is None and no input reference centers have been given, 
        calculate the cost of a data subset X_ by finding the sum of 
        squared distances to mean(X_).
        
        Otherwise, 

        Args:
            indices (np.ndarray[int]): Indices of data points to compute cost with. 
            
        Returns:
            (float): Cost of the given data. 
        """
        
        if len(indices) == 0:
            return np.inf
        else:
            dists_ = self.center_dists[indices,:]
            sum_array = np.sum(dists_, axis = 0)
            return np.min(sum_array)

    
####################################################################################################
    
class RuleClusteredTree(LinearTree):
    """
    Inherits from the Tree class to implement a decision tree in which 
    axis aligned split criterion are chosen so that points in any leaf node are close
    in distance to their closest center or centroid, from a set of input centers. 
    
    This diverges from the CentroidTree by also performing Rule Clustering 
    on its leaf nodes after every split, and updating its centers as it goes.
    
    Args:
        splits (str): May take values 'axis' or 'oblique' that decide on how to compute leaf splits.
        
        max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
            Defaults to None.
            
        max_depth (int, optional): Optional constraint for maximum depth. 
            Defaults to None.
            
        min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
            within a single leaf. Defaults to 1.
            
        norm (int, optional): Takes values 1 or 2. If norm = 1, compute distances using the 
            1 norm. If 2, compute distances with the squared two norm. Defaults to 2.
            
        center_init (str, optional): Center initialization method. Included options are 
            'k-means' which runs a k-means algorithm and uses the output centers,
            'random++' which uses a randomized k-means++ initialization, or 
            'manual' which assumes an input array of centers (in the next parameter). 
            Defaults to None in which case no centers are initialized.
            
        centers (np.ndarray, optional): Input list of reference centers to calculate cost with. 
            Defaults to None.
        
        clusterer (RuleClustering, optional): Clustering object used to cluster leaves 
            and find new centers. Defaults to None, not used if center_updates is False.
            
        random_seed (int, optional): Random seed. In the Tree object randomness is only ever 
            used for breaking ties between nodes, or if you are using a RandomTree!
            
        feature_labels (List[str]): Iterable object with strings representing feature names. 
            
            
    Attributes:
        X (np.ndarray): Size (n x m) dataset passed as input in the fitting process.
        
        root (Node): Root node of the tree.
    
        heap (heapq list): Maintains the heap structure of the tree.
        
        leaf_count (int): Number of leaves in the tree.
        
        node_count (int): Number of nodes in the tree.
        
        n_centers (int, optional): The number of centers to use for computation of cost 
            or center updates.
        
        center_dists (np.ndarray): If being fitted to an (n x m) dataset, computes a
            n x k array of distances (measured either with squared 2 norm or 1 norm) 
            from every point to every center.
            
        cost (float): Clustering cost from latest split.
                
    """
    
    def __init__(self, splits = 'axis', max_leaf_nodes=None, max_depth=None, min_points_leaf = 1, 
                 norm = 2, center_init = 'k-means', centers = None, clusterer = None,
                 random_seed = None, feature_labels = None, start_centers = None):
        
        if center_init is None:
            raise ValueError('Must provide some method of initializing centers.')
        
        if clusterer is None:
            raise ValueError('Must provide some Rule Clustering object to cluster leaves.')
        
        super().__init__(splits = splits, max_leaf_nodes = max_leaf_nodes, max_depth = max_depth, 
                         min_points_leaf = min_points_leaf, norm = norm, center_init = center_init,
                         centers = centers, cluster_leaves = True, clusterer = clusterer, 
                         random_seed = random_seed, feature_labels = feature_labels)
        
        #self.start_centers = start_centers
        #self.centers = start_centers
        #self.n_centers = len(start_centers)
        self.clustering_cost = None
        
    
    def _update_center_dists(self):
        """
        Updates the center_dists array, which tracks distances from all points in self.X
        to all centers. This helps with computational efficiency, since we won't 
        need to recompute this when searching through splits.
        """
        if self.centers is not None:
            if self.norm == 2:
                diffs = self.X[np.newaxis, :, :] - self.centers[:, np.newaxis, :]
                distances = np.sum((diffs)**2, axis=-1)
                self.center_dists = distances.T
                
            elif self.norm == 1:
                diffs = self.X[np.newaxis, :, :] - self.centers[:, np.newaxis, :]
                distances = np.sum(np.abs(diffs), axis=-1)
                self.center_dists = distances.T
        else:
            pass
            
        
    def _cost(self, indices):
        """
        Given a set of indices defining a data subset X_ --
        which may be thought of as a subset of points reaching a given node in the tree --
        compute a cost for X_.
        
        In a centroid tree this amounts to find the sum of distances to 
        centers (using squared 2 norm or 1 norm respectively).
        
        If self.centers is None and no input reference centers have been given, 
        calculate the cost of a data subset X_ by finding the sum of 
        squared distances to mean(X_).
        
        Otherwise, 

        Args:
            indices (np.ndarray[int]): Indices of data points to compute cost with. 
            
        Returns:
            (float): Cost of the given data. 
        """
        
        if len(indices) == 0:
            return np.inf
        
        #elif self.centers is not None:
        else:
            dists_ = self.center_dists[indices,:]
            sum_array = np.sum(dists_, axis = 0)
            return np.min(sum_array)
        '''
        else:
            if self.norm == 2:
                X_ = self.X[indices,:]
                mu = np.mean(X_, axis = 0)
                cost = np.sum(np.linalg.norm(X_ - mu, axis = 1)**2)
                
            elif self.norm == 1:
                X_ = self.X[indices, :]
                eta = np.median(X_, axis = 0)
                cost = np.sum(np.abs(X_ - eta))
                
            return cost
        '''
            


####################################################################################################


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

class ConvertExKMC(Tree):
    """
    Transforms an ExKMC tree to a Tree object as defined here (For visualization purposes). 
    The ExKMC tree is based around work 
    from work from [Frost, Moshkovitz, Rashtchian '20] in their 
    paper titled 'ExKMC: Expanding Explainable k-Means Clustering.'
    The following works by examining the tree created in their implementation, 
    which may be found at: https://github.com/navefr/ExKMC.
    
    NOTE: The fit() method of the parent class will not apply. Any fitting must be done 
    with the ExKMC code. This class simply builds a Tree object out of the already 
    fitted tree which output from their code. 
    """
    
    def __init__(self, ExKMC_root, X, feature_labels = None):
        """
        Args:
            ExKMC_root (ExKMC.Tree.Node): Root node of an ExKMC tree.
            
            feature_labels(List[str], optional): List of strings corresponding to feature names 
                in the data. Useful for explaining results in post. 
        """

        super().__init__()
        self.exkmc_root = ExKMC_root
        self.X = X
        
        # Set feature labels:
        if feature_labels is None:
            self.feature_labels = [None]*X.shape[1]
        else:
            if not len(feature_labels) == X.shape[1]:
                raise ValueError('Labels must match the shape of the data.')
            self.feature_labels = feature_labels
        
        
        self.root = Node()
        self._build_tree(self.exkmc_root, self.root, np.array(range(len(X))), 0)
        self.node_count += 1
        
        
    def _cost(self, indices):
        """
        Assigns cost that rewards data with points all close to their mean
        (i.e. small variance).

        Args:
            indices (np.ndarray[int]): Indices of data points to compute cost with. 
            
        Returns:
            (float): Cost of the given data. 
        """
        
        #if len(X_) == 0:
        #    return np.inf
        #else:
        #    mu = np.mean(X_, axis = 0)
        #    cost = np.sum(np.linalg.norm(X_ - mu, axis = 1)**2) #/len(X_)
        #    return cost
        return 0
        
    # The following methods become irrelevant for this class.
    def fit(self, X):
        self.X = X
        pass
    
    def fit_step(self):
        pass
    
    def split_leaf(self):
        pass
        
    def _build_tree(self, exkmc_node, node_obj: Node, indices, depth):
        """
        Builds the decision tree by traversing the ExKMC tree.
        
        Args:
            exkmc_node (ExKMC.Tree.Node): Node of an ExKMC tree.
            
            node_obj (Node): Corresponding Node object to copy conditions into. 
            
            indices (np.ndarray[int]): Subset of data indices to build node with. 
            
            depth (int): current depth of the tree
        """
        X_ = self.X[indices, :]
        if depth > self.depth:
            self.depth = depth
        
        if exkmc_node.is_leaf():
            class_label = self.leaf_count 
            self.leaf_count += 1
            node_obj.leaf_node(label = class_label, indices = indices, cost = self._cost(indices))
        else:
            feature, threshold = exkmc_node.feature, exkmc_node.value
            left_mask = X_[:, feature] <= threshold
            right_mask = ~left_mask
            
            left_node = Node()
            right_node = Node()
            
            node_obj.tree_node([feature], [1], threshold, left_node, right_node, 
                               indices, self._cost(indices),
                               feature_labels = [self.feature_labels[feature]])
            
            self._build_tree(exkmc_node.left, left_node, indices[left_mask], depth + 1)
            self._build_tree(exkmc_node.right, right_node, indices[right_mask], depth + 1)
            
            self.node_count += 2
            
####################################################################################################

class SklearnTree(Tree):
    """
    Class designed to interface with an Sklearn Tree object. 
    For more information about Sklearn trees, please visit 
    (https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    #sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py)
    
    
    """
    
    def __init__(
        self,
        data_labels = None,
        max_leaf_nodes=None,
        max_depth=None,
        min_points_leaf=1,
        random_seed=None,
        feature_labels=None 
    ):
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
            X (np.ndarray): Size (n x m) dataset passed as input in the fitting process.
            
            root (Node): Root node of the tree.
        
            heap (heapq list): Maintains the heap structure of the tree.
            
            leaf_count (int): Number of leaves in the tree.
            
            node_count (int): Number of nodes in the tree.
            
            n_centers (int): The number of centers to use for computation of cost 
                or center updates.
                
            depth (int): The maximum depth of the tree.
        """

        super().__init__(
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_points_leaf=min_points_leaf,
            random_seed=random_seed,
            feature_labels=feature_labels
        )
        
        self.data_labels = data_labels
        
        
        
    def _cost(self, indices):
        """
        Assigns cost that rewards data with points all close to their mean
        (i.e. small variance).

        Args:
            indices (np.ndarray[int]): Indices of data points to compute cost with. 
            
        Returns:
            (float): Cost of the given data. 
        """
        return 0
        
        
    def fit(self, X):
        self.X = X
        self.label = np.random.choice(self.data_labels)
        y = (self.data_labels == self.label).astype(int)
        
        # Set feature labels if not set already:
        if self.feature_labels is None:
            self.feature_labels = [None]*X.shape[1]
        else:
            if not len(self.feature_labels) == X.shape[1]:
                raise ValueError('Feature labels must match the shape of the data.')
        
        #not_clusterable = False
        #while not not_clusterable:
        self.SklearnT = DecisionTreeClassifier(
            criterion = 'entropy',
            max_depth = self.max_depth,
            min_samples_leaf = self.min_points_leaf,
            max_leaf_nodes = self.max_leaf_nodes
        )
        self.SklearnT.fit(X, y)
        self.sklearn_tree = self.SklearnT.tree_
        #    not_clusterable = True
        #else:
        #    print('Tree not clusterable, retrying...')
        
        self.root = Node()
        self._convert_tree(0, self.root, np.array(range(len(X))), 0)
        self.node_count += 1
        
    def fit_step(self):
        pass
    
    def split_leaf(self):
        pass
        
    def _convert_tree(self, node, node_obj, indices, depth):
        """
        Builds the decision tree by traversing the Sklearn tree.
        
        Args:
            node (int): Node of a new Sklearn tree.
            
            node_obj (Node): Corresponding Node object to copy conditions into. 
            
            indices (np.ndarray[int]): Subset of data indices to build node with. 
        """
        X_ = self.X[indices, :]
        y_ = self.data_labels[indices]
        
        if depth > self.depth:
            self.depth = depth
        
        if (self.sklearn_tree.children_left[node] < 0 and 
            self.sklearn_tree.children_right[node] < 0):
            class_label = self.leaf_count # This may be important...might need to change...
            self.leaf_count += 1
            node_obj.leaf_node(label = class_label, indices = indices, cost = self._cost(indices))
            
            if np.sum(y_ == self.label)/len(y_) < 0.5:
                node_obj.type = 'bad-leaf'
        else:
            feature, threshold = self.sklearn_tree.feature[node], self.sklearn_tree.threshold[node]
            left_mask = X_[:, feature] <= threshold
            right_mask = ~left_mask
            
            left_node = Node()
            right_node = Node()
            
            node_obj.tree_node([feature], [1], threshold, left_node, right_node, 
                               indices, self._cost(indices), 
                               feature_labels = [self.feature_labels[feature]])
            
            self._convert_tree(self.sklearn_tree.children_left[node], left_node, indices[left_mask], depth + 1)
            self._convert_tree(self.sklearn_tree.children_right[node], right_node, indices[right_mask], depth + 1)
            
            self.node_count += 2
            
####################################################################################################

# Older code:

'''
# Some extra attempt to optimize an unsupervised tree, will maybe add in later.
#For means / squared 2 norm:

def _find_best_split(self, X_):
        """
        Chooses the axis aligned split (feature, threshold) with smallest 
        sum of costs between the two branches it creates.
        
        Args:
            X_ (np.ndarray): Input dataset.
            
        Returns:
            Tuple(int, float): A (feature, threshold) split pair. 
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

        return best_split_val, best_split
'''

'''
#For medians / 1 norm:
for i in range(X_.shape[1]):
    eta2 = np.median(X_, axis = 0)
    split_val = np.sum(np.abs(X_ - eta2))
    order = np.argsort(X_[:, i])
    
    for j, idx in enumerate(order[:-1]):
        threshold = X_[idx, i]
        eta1 = np.median(X_[:j+1, :], axis = 0)
        split_val = (split_val + np.sum(np.abs(X_[j,:] - eta1)) 
                        - np.sum(np.abs(X_[j,:] - eta2)))
        eta2 = np.median(X_[j+1:, :], axis = 0)
        
        if split_val < best_split_val:
            best_split_val = split_val
            best_split = (i, threshold)
'''

'''
class KMeansObliqueTree1(Tree):
    """
    Inherits from the Tree class to implement a decision tree in which split criterion chosen so
    that points in any leaf node are close in distance to their mean. 
    
    Args:
        max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
            Defaults to None.
            
        max_depth (int, optional): Optional constraint for maximum depth. 
            Defaults to None.
            
        min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
            within a single leaf. Defaults to 1.
            
        random_seed (int, optional): Random seed. In the kMeansTree object randomness is only ever 
            used for breaking ties between nodes which have the same cost.
            
        feature_labels (List[str]): Iterable object with strings representing feature names. 
        
    Attributes:
        heap (heapq list): Maintains the heap structure of the tree.
        leaf_count (int): Number of leaves in the tree.
        node_count (int): Number of nodes in the tree.
    """
    def __init__(self, max_leaf_nodes=None, max_depth=None, min_points_leaf = 1, random_seed = None,
                 feature_labels = None, centers = None):
        super().__init__(max_leaf_nodes, max_depth, min_points_leaf, random_seed)
        self.centers = centers
        
    def _cost(self, X_):
        """
        Assigns cost that rewards data with points all close to their mean
        (i.e. small variance).

        Args:
            X_ (np.ndarray): Data subset. 
            
        Returns:
            (float): Cost of the given data. 
        """
        
        if len(X_) == 0:
            return np.inf
        else:
            if self.centers is None:
                mu = np.mean(X_, axis = 0)
                cost = np.sum(np.linalg.norm(X_ - mu, axis = 1)**2) #/len(X_)
                return cost
            else:
                diffs = X_[np.newaxis, :, :] - self.centers[:, np.newaxis, :]
                distances = np.sum((diffs)**2, axis=-1)
                sum_array = np.sum(distances, axis=1)
                return np.min(sum_array)


    def _find_best_split(self, X_):
            """
            Chooses the axis aligned split (feature, threshold) with smallest 
            sum of costs between the two branches it creates.
            
            Args:
                X_ (np.ndarray): Input dataset.
                
            Returns:
                Tuple(int, float): A (feature, threshold) split pair. 
            """
            
            n, d = X_.shape
            
            # For every combination of two features:
            feature_pairs = list(itertools.combinations(list(range(d)), 2))
            
            best_cost = np.inf
            best_pair = None
            best_centers = None
            for pair in feature_pairs:
                X_pair = X_[:, pair]

                # Find a split to 2 clusters
                kmeans_cluster = KMeans(n_clusters=2, random_state=self.random_seed).fit(X_pair)
                
                C = kmeans_cluster.cluster_centers_
                weights_ = C[1,:] - C[0,:]
                threshold_ = (C[1,0]**2 - C[0,0]**2 + C[1,1]**2 - C[0,1]**2)/2
                left_mask = np.dot(X_pair, weights_) <= threshold_
                right_mask = ~left_mask
                cost_split = self._cost(X_[left_mask, :]) + self._cost(X_[right_mask,:])

                
                # Record clustering with best cost:
                if cost_split < best_cost:
                    best_cost = cost_split
                    best_pair = pair
                    best_centers = kmeans_cluster.cluster_centers_
                    
            C = best_centers
            weights = C[1,:] - C[0,:]
            threshold = (C[1,0]**2 - C[0,0]**2 + C[1,1]**2 - C[0,1]**2)/2
                        
            return best_cost, (list(best_pair), weights, threshold)
        
        
####################################################################################################

class KMeansObliqueTree2(Tree):
    """
    Inherits from the Tree class to implement a decision tree in which split criterion chosen so
    that points in any leaf node are close in distance to their mean. 
    
    Args:
        max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
            Defaults to None.
            
        max_depth (int, optional): Optional constraint for maximum depth. 
            Defaults to None.
            
        min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
            within a single leaf. Defaults to 1.
            
        random_seed (int, optional): Random seed. In the kMeansTree object randomness is only ever 
            used for breaking ties between nodes which have the same cost.
            
        feature_labels (List[str]): Iterable object with strings representing feature names. 
        
    Attributes:
        heap (heapq list): Maintains the heap structure of the tree.
        leaf_count (int): Number of leaves in the tree.
        node_count (int): Number of nodes in the tree.
    """
    def __init__(self, max_leaf_nodes=None, max_depth=None, min_points_leaf = 1, random_seed = None,
                 feature_labels = None, centers = None):
        super().__init__(max_leaf_nodes, max_depth, min_points_leaf, random_seed)
        self.centers = centers
        
    def _cost(self, X_):
        """
        Assigns cost that rewards data with points all close to their mean
        (i.e. small variance).

        Args:
            X_ (np.ndarray): Data subset. 
            
        Returns:
            (float): Cost of the given data. 
        """
        
        if len(X_) == 0:
            return np.inf
        else:
            if self.centers is None:
                mu = np.mean(X_, axis = 0)
                cost = np.sum(np.linalg.norm(X_ - mu, axis = 1)**2) #/len(X_)
                return cost
            else:
                diffs = X_[np.newaxis, :, :] - self.centers[:, np.newaxis, :]
                distances = np.sum((diffs)**2, axis=-1)
                sum_array = np.sum(distances, axis=1)
                return np.min(sum_array)


    def _find_best_split(self, X_):
        """
        Chooses the axis aligned split (feature, threshold) with smallest 
        sum of costs between the two branches it creates.
        
        Args:
            X_ (np.ndarray): Input dataset.
            
        Returns:
            Tuple(int, float): A (feature, threshold) split pair. 
        """
        
        n, d = X_.shape
        
        # cluster into two:
        kmeans_cluster = KMeans(n_clusters=2, random_state=self.random_seed).fit(X_)
        C = kmeans_cluster.cluster_centers_
        all_weights = C[1,:] - C[0,:]
        midpoint = np.mean(C, axis = 0)
        #threshold = np.dot(all_weights, midpoint)
        
        # For every combination of two features:
        feature_pairs = list(itertools.combinations(list(range(d)), 2))
        
        best_cost = np.inf
        best_pair = None
        best_weights = None
        best_threshold = None
        for pair in feature_pairs:
            X_pair = X_[:, pair]
            
            weights_ = all_weights[list(pair)]
            threshold_ = np.dot(weights_, midpoint[list(pair)])
            left_mask = np.dot(X_pair, weights_) <= threshold_
            right_mask = ~left_mask
            cost_split = self._cost(X_[left_mask, :]) + self._cost(X_[right_mask,:])
            #print(cost_split)
            
            # Record clustering with best cost:
            if cost_split < best_cost:
                best_cost = cost_split
                best_pair = pair
                best_centers = kmeans_cluster.cluster_centers_
                best_weights = weights_
                best_threshold = threshold_
        
        weights = best_weights
        threshold = best_threshold
                    
        return best_cost, (list(best_pair), weights, threshold)
        
'''


'''
class ConvertSklearn(Tree):
    """
    Transforms an Sklearn tree to a Tree object as defined here (For visualization purposes). 
    For more information about Sklearn trees, please visit 
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    
    NOTE: The fit() method of the parent class will not apply. Any fitting must be done 
    with the ExKMC code. This class simply builds a Tree object out of the already 
    fitted tree which output from their code. 
    """
    
    def __init__(self, sklearn_tree, X, feature_labels = None):
        """
        Args:
            sklearn_tree (sklearn.tree._tree.Tree): Sklearn Tree object. Ideally already 
                fit to some dataset!
            
            feature_labels(List[str], optional): List of strings corresponding to feature names 
                in the data. Useful for explaining results in post. 
        """

        super().__init__()
        self.sklearn_tree = sklearn_tree
        self.X = X
        
        # Set feature labels:
        if feature_labels is None:
            self.feature_labels = [None]*X.shape[1]
        else:
            if not len(feature_labels) == X.shape[1]:
                raise ValueError('Labels must match the shape of the data.')
            self.feature_labels = feature_labels
        
        # Sklearn Trees start with root 0
        self.root = Node()
        self._build_tree(0, self.root, np.array(range(len(X))), 0)
        self.node_count += 1
        
        
    def _cost(self, indices):
        """
        Assigns cost that rewards data with points all close to their mean
        (i.e. small variance).

        Args:
            indices (np.ndarray[int]): Indices of data points to compute cost with. 
            
        Returns:
            (float): Cost of the given data. 
        """
        
        #if len(X_) == 0:
        #    return np.inf
        #else:
        #    mu = np.mean(X_, axis = 0)
        #    cost = np.sum(np.linalg.norm(X_ - mu, axis = 1)**2) #/len(X_)
        #    return cost
        return 0
        
        
    def fit(self, X):
        self.X = X
        pass
    
    def fit_step(self):
        pass
    
    def split_leaf(self):
        pass
        
    def _build_tree(self, node, node_obj, indices, depth):
        """
        Builds the decision tree by traversing the ExKMC tree.
        
        Args:
            node (int): Node of an ExKMC tree.
            
            node_obj (Node): Corresponding Node object to copy conditions into. 
            
            indices (np.ndarray[int]): Subset of data indices to build node with. 
        """
        X_ = self.X[indices, :]
        if depth > self.depth:
            self.depth = depth
        
        if (self.sklearn_tree.children_left[node] < 0 and 
            self.sklearn_tree.children_right[node] < 0):
            class_label = self.leaf_count 
            self.leaf_count += 1
            node_obj.leaf_node(label = class_label, indices = indices, cost = self._cost(indices))
        else:
            feature, threshold = self.sklearn_tree.feature[node], self.sklearn_tree.threshold[node]
            left_mask = X_[:, feature] <= threshold
            right_mask = ~left_mask
            
            left_node = Node()
            right_node = Node()
            
            node_obj.tree_node([feature], [1], threshold, left_node, right_node, 
                               indices, self._cost(indices), 
                               feature_labels = [self.feature_labels[feature]])
            
            self._build_tree(self.sklearn_tree.children_left[node], left_node, indices[left_mask], depth + 1)
            self._build_tree(self.sklearn_tree.children_right[node], right_node, indices[right_mask], depth + 1)
            
            self.node_count += 2
            
####################################################################################################
'''