import numpy as np
import copy
import heapq
from sklearn.cluster import KMeans
from ._node import Node

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
        split = leaf_obj[2]
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
            features, weights, threshold = split
            
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