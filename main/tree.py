import numpy as np
import heapq
import itertools
from sklearn.cluster import KMeans
from utils import *


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
            
        satisfied_indices (np.ndarray): The indices of data points belonging to this node.
            
        points (int): The total number of points belonging to this node.  
        
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
        self.satisfied_indices = None
        self.points = None
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
    
    def tree_node(self, features, weights, threshold, left_child, right_child, cost, points, 
                  feature_labels):
        """
        Initializes this as a normal node in the tree.

        Args:
            feature (int): The axis aligned feature to split on. 
            threshold (float): The value of feature to split on
            left_child (Node): Pointer to the left branch of the current node.
            right_child (Node): Pointer to the right branch of the 
                current node. 
            points (np.ndarray): The set of data points associated with points belonging 
                to this node.  
            cost (float): The cost associated with points belonging to this node. 
            feature_label (str): Name for the given feature (for printing and display). 
        """
        self.type = 'node'
        self.features = features
        self.weights = weights
        self.threshold = threshold
        self.left_child = left_child 
        self.right_child = right_child
        self.cost = cost
        self.points = points
        self.feature_labels = feature_labels
        
    def leaf_node(self, label, cost, points):
        """
        Initializes this to be a leaf node in the tree.

        Args:
            label (int): Prediction label to be associated with this node.
            points (np.ndarray): The set of data points associated with points belonging 
                to this node.  
            cost (float): The cost associated with points belonging to this node. 
        """
        self.type = 'leaf'
        self.label = label
        self.cost = cost
        self.points = points
        
####################################################################################################
        
class Tree():
    """
    Decision Tree object with simple axis aligned splitting conditions. 
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
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_points_leaf = min_points_leaf
        self.feature_labels = feature_labels
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.random_seed = random_seed
        
        self.heap = []
        self.leaf_count = 0
        self.node_count = 0
        
    
    def _cost(self,X_):
        """
        Calculates the cost of a subset of data points.

        Args:
            X_ (np.ndarray): Data subset. 
            
        Returns:
            (float): Cost of the given data. 
        """
        return


    def _find_best_split(self, X_):
        """
        Chooses features, weights and a threshold to split upon given some input dataset X.

        Args:
            X_ (np.ndarray): Input dataset.
            
        Returns:
                Tuple(List[int], List[float], float): A (features, weights, threshold) split pair. 
        """
        return
    
    
    def _count_leaves(self):
        """
        Returns:
            (int): Number of leaf nodes in the tree.
        """
        return len(self.heap) + self.leaf_count
    

    def fit(self, X, feature_labels = None):
        """
        Initiates and builds a decision tree around a given dataset. 

        Args:
            X (np.ndarray): Input dataset.
            feature_labels(List[str], optional): List of strings corresponding to feature names 
                in the data. Useful for explaining results in post. 
        """
        # Reset
        self.heap = []
        self.leaf_count = 0
        self.node_count = 0
        
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
        # It prioritizes splitting the leaves with the largest cost.
        # NOTE: I store -1*scores because heapq pops items with minimum cost by default.
        # Heap leaf object: (-1*score, Node object, data subset, depth, split)
        # Each split (features, weights, thresholds) is pre-calculated,
        # and the gain in cost performance from splitting is then given as the score.
        self.root = Node()
        split_cost, split = self._find_best_split(X)
        score = (self._cost(X) - split_cost)
        heapq.heappush(self.heap, (-1*score, self.root, X, 0, split))
        self.node_count += 1
        
        while len(self.heap) > 0:
            self._build_tree()
        

    def _build_tree(self):
        """
        Builds the decision tree by iteratively selecting conditions and splitting leaf nodes.
        """
        
        # pop an object from the heap
        leaf_obj = heapq.heappop(self.heap)
        node_obj = leaf_obj[1]
        X_ = leaf_obj[2]
        depth = leaf_obj[3]
        split = leaf_obj[4]
        
        
        # If we've reached the max depth or max number of leaf nodes -- stop recursing 
        if depth >= self.max_depth or self._count_leaves() >= (self.max_leaf_nodes - 1) or len(X_) <= 1:
            class_label = self.leaf_count 
            self.leaf_count += 1
            node_obj.leaf_node(class_label, self._cost(X_), len(X_))


        # Otherwise, find the best split and recurse into left and right branches
        else:
            #features, weights, threshold = self._find_best_split(X_)
            features, weights, threshold = split
            left_mask = np.dot(X_[:, features], weights) <= threshold
            right_mask = ~left_mask
            
            left_node = Node()
            right_node = Node()
            
            node_obj.tree_node(features, weights, threshold, left_node, right_node, 
                               self._cost(X_), len(X_), 
                               feature_labels = [self.feature_labels[f] for f in features])
            
            
            if len(X_[left_mask]) > 1:
                left_split_cost, left_split = self._find_best_split(X_[left_mask])
            else:
                left_split_cost = 0
                left_split = None
                
            left_score = self._cost(X_[left_mask]) - left_split_cost
            left_obj = (-1*left_score, left_node, X_[left_mask], depth + 1, left_split)
            
            if len(X_[right_mask]) > 1:
                right_split_cost, right_split = self._find_best_split(X_[right_mask])
            else:
                right_split_cost = 0
                right_split = None
                
            right_score = self._cost(X_[right_mask]) - right_split_cost
            right_obj = (-1*right_score, right_node, X_[right_mask], depth + 1, right_split)
            
            
            heapq.heappush(self.heap, left_obj)
            heapq.heappush(self.heap, right_obj)
            self.node_count += 2


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

class KMeansTree(Tree):
    """
    Inherits from the Tree class to implement a decision tree in which 
    axis aligned split criterion are chosen so that points in any leaf node are close
    in distance to their mean. 
    
    The optimized splitting algorithm is attributed to 
    work by [Dasgupata, Frost, Moshkovitz, Rashtchian '20]
    in their paper titled 'Explainable k-means and k-medians clustering.
    
    Args:
        max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
            Defaults to None.
            
        max_depth (int, optional): Optional constraint for maximum depth. 
            Defaults to None.
            
        min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
            within a single leaf. Defaults to 1.
            
        random_seed (int, optional): Random seed. In the kMeansTree object randomness is only ever 
            used for breaking ties between nodes which have the same cost.
            
        feature_labels (List[str], optional): Iterable object with strings 
            representing feature names. 
        
        centers (np.ndarray, optional): Input list of reference centers to calculate cost with. 
            See _cost() for more information. Defaults to None.
        
    Attributes:
        centers (np.ndarray): Input list of reference centers.
        
        heap (heapq list): Maintains the heap structure of the tree.
        
        leaf_count (int): Number of leaves in the tree.
        
        node_count (int): Number of nodes in the tree.
    """
    def __init__(self, max_leaf_nodes=None, max_depth=None, min_points_leaf = 1, random_seed = None,
                 feature_labels = None, centers = None):
        
        super().__init__(max_leaf_nodes, max_depth, min_points_leaf, random_seed, feature_labels)
        
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

    '''
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
    def _find_best_split(self, X_):
        best_split = None
        best_split_val = np.inf
        
        for feature in range(X_.shape[1]):
            unique_vals = np.unique(X_[:,feature])
            for threshold in unique_vals:
                split = ([feature], [1], threshold)

                left_mask = X_[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_points_leaf or np.sum(right_mask) < self.min_points_leaf:
                    split_val = np.inf
                else:
                    X1 = X_[left_mask, :]
                    X1_cost = self._cost(X1)
                    
                    X2 = X_[right_mask, :]
                    X2_cost = self._cost(X2)
                    
                    split_val = X1_cost + X2_cost
                
                if split_val < best_split_val:
                    best_split_val = split_val
                    best_split = split

        return best_split_val, best_split
        
####################################################################################################

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
        
        
####################################################################################################

class KMeansObliqueTree3(Tree):
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
        feature_pairs = list(itertools.combinations(list(range(d)), 2))
        slopes = np.array([[0,1],
                  [1,2],
                  [1,1],
                  [2,1],
                  [1,0],
                  [2,-1],
                  [1,-1],
                  [1,-2]])
        
        best_split = None
        best_split_val = np.inf
        for pair in feature_pairs:
            X_pair = X_[:, pair]
            for i,x in enumerate(X_pair):
                for j, slope in enumerate(slopes):
                    weights = np.array([slope[1], -slope[0]])
                    threshold = slope[1]*x[0] - slope[0]*x[1]
                    split = (list(pair), weights, threshold)

                    left_mask = np.dot(X_pair, weights) <= threshold
                    right_mask = ~left_mask
                    
                    if np.sum(left_mask) < self.min_points_leaf or np.sum(right_mask) < self.min_points_leaf:
                        split_val = np.inf
                    else:
                        X1 = X_[left_mask]
                        X1_cost = self._cost(X1)
                        
                        X2 = X_[right_mask]
                        X2_cost = self._cost(X2)
                        
                        split_val = X1_cost + X2_cost
                    
                    if split_val < best_split_val:
                        best_split_val = split_val
                        best_split = split

        return best_split_val, best_split
        
        
####################################################################################################

class KMediansTree(Tree):
    """
    Inherits from the Tree class to implement a decision tree in which split criterion chosen so
    that points in any leaf node are close in distance to their median. 
    
    NOTE: This is yet to be optimized and so will run quite slowly. Coming soon. 
    
    Args:
        max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
            Defaults to None.
            
        max_depth (int, optional): Optional constraint for maximum depth. 
            Defaults to None.
            
        min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
            within a single leaf. Defaults to 1.
            
        random_seed (int, optional): Random seed. In the kMediansTree object randomness is only ever 
            used for breaking ties between nodes which have the same cost.
            
        feature_labels (List[str]): Iterable object with strings representing feature names. 
        
    Attributes:
        heap (heapq list): Maintains the heap structure of the tree.
        leaf_count (int): Number of leaves in the tree.
        node_count (int): Number of nodes in the tree.
    """
    def __init__(self, max_leaf_nodes=None, max_depth=None, min_points_leaf = 1, random_seed = None,
                 feature_labels = None):
        super().__init__(max_leaf_nodes, max_depth, min_points_leaf, random_seed, feature_labels)
        
        
    def _cost(self, X_):
        """
        Assigns cost that rewards data with points all close to their median.

        Args:
            X_ (np.ndarray): Data subset. 
            
        Returns:
            (float): Cost of the given data. 
        """
        
        if len(X_) == 0:
            return np.inf
        else:
            eta = np.median(X_, axis = 0)
            cost = np.sum(np.abs(X_ - eta)) #/len(X_)
            return cost


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
            best_split = None
            best_split_val = np.inf
            
            '''
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
            
            for feature in range(X_.shape[1]):
                unique_vals = np.unique(X_[:,feature])
                for threshold in unique_vals:
                    split = ([feature], [1], threshold)

                    left_mask = X_[:, feature] <= threshold
                    right_mask = ~left_mask
                    
                    if np.sum(left_mask) < self.min_points_leaf or np.sum(right_mask) < self.min_points_leaf:
                        split_val = np.inf
                    else:
                        X1 = X_[left_mask]
                        X1_cost = self._cost(X1)
                        
                        X2 = X_[right_mask]
                        X2_cost = self._cost(X2)
                        
                        split_val = X1_cost + X2_cost
                    
                    if split_val < best_split_val:
                        best_split_val = split_val
                        best_split = split

            return best_split

####################################################################################################

class RandomTree(Tree):
    """
    Inherits from the Tree class to implement a decision tree in which split criterion are randomly 
    chosen from the input space.
    
    This model is a product of work by [Galmath, Jia, Polak, Svensson 2021] in their paper 
    titled "Nearly-Tight and Oblivious Algorithms for Explainable Clustering."
    
    NOTE: In order to imitate results from [Galmath, Jia, Polak, Svensson 2021], fit this model on 
        a dataset of centers or representative points.
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
        super().__init__(max_leaf_nodes, max_depth, min_points_leaf, random_seed, feature_labels)
        
    def _cost(self, X_):
        """
        Assigns a random cost of a subset of data points.

        Args:
            X_ (np.ndarray): Data subset. 
            
        Returns:
            (float): Cost of the given data. 
        """
        
        # Costs are given by a uniform random sample, meaning nodes will be chosen randomly 
        # to split from 
        return np.random.uniform()


    def _find_best_split(self, X_):
            """
            Randomly chooses an axis aligned threshold value to split upon 
            given some input dataset X. The randomly sampled split must separate at least one
            pair of points from X_.

            Args:
                X_ (np.ndarray): Input dataset.
                
            Returns:
                Tuple(int, float): A (feature, threshold) split pair. 
            """
            
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
                if len(left_branch[0]) > 0 and len(right_branch[0]) > 0:
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
        
        # Set feature labels:
        if feature_labels is None:
            self.feature_labels = [None]*X.shape[1]
        else:
            if not len(feature_labels) == X.shape[1]:
                raise ValueError('Labels must match the shape of the data.')
            self.feature_labels = feature_labels
        
        
        self.root = Node()
        self._build_tree(self.exkmc_root, self.root, X)
        self.node_count += 1
        
        
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
            mu = np.mean(X_, axis = 0)
            cost = np.sum(np.linalg.norm(X_ - mu, axis = 1)**2) #/len(X_)
            return cost
        
        
    def fit(self, X):
        pass
        
    def _build_tree(self, exkmc_node, node_obj: Node, X_):
        """
        Builds the decision tree by traversing the ExKMC tree.
        
        Args:
            exkmc_node (ExKMC.Tree.Node): Node of an ExKMC tree.
            node_obj (Node): Corresponding Node object to copy conditions into. 
            X_ (np.ndarray): Data subset associated with the current node. 
        """
        
        if exkmc_node.is_leaf():
            class_label = self.leaf_count 
            self.leaf_count += 1
            node_obj.leaf_node(label = class_label, cost = self._cost(X_), points = len(X_))
        else:
            feature, threshold = exkmc_node.feature, exkmc_node.value
            left_mask = X_[:, feature] <= threshold
            right_mask = ~left_mask
            
            left_node = Node()
            right_node = Node()
            
            node_obj.tree_node([feature], [1], threshold, left_node, right_node, 
                               self._cost(X_), len(X_), 
                               feature_labels = [self.feature_labels[feature]])
            
            self._build_tree(exkmc_node.left, left_node, X_[left_mask])
            self._build_tree(exkmc_node.right, right_node, X_[right_mask])
            
            self.node_count += 2
            
####################################################################################################

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
        
        # Set feature labels:
        if feature_labels is None:
            self.feature_labels = [None]*X.shape[1]
        else:
            if not len(feature_labels) == X.shape[1]:
                raise ValueError('Labels must match the shape of the data.')
            self.feature_labels = feature_labels
        
        # Sklearn Trees start with root 0
        self.root = Node()
        self._build_tree(0, self.root, X)
        self.node_count += 1
        
        
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
            mu = np.mean(X_, axis = 0)
            cost = np.sum(np.linalg.norm(X_ - mu, axis = 1)**2) #/len(X_)
            return cost
        
        
    def fit(self, X):
        pass
    
        
    def _build_tree(self, node, node_obj, X_):
        """
        Builds the decision tree by traversing the ExKMC tree.
        
        Args:
            node (int): Node of an ExKMC tree.
            node_obj (Node): Corresponding Node object to copy conditions into. 
            X_ (np.ndarray): Data subset associated with the current node. 
        """
        
        if (self.sklearn_tree.children_left[node] < 0 and 
            self.sklearn_tree.children_right[node] < 0):
            class_label = self.leaf_count 
            self.leaf_count += 1
            node_obj.leaf_node(label = class_label, cost = self._cost(X_), points = len(X_))
        else:
            feature, threshold = self.sklearn_tree.feature[node], self.sklearn_tree.threshold[node]
            left_mask = X_[:, feature] <= threshold
            right_mask = ~left_mask
            
            left_node = Node()
            right_node = Node()
            
            node_obj.tree_node([feature], [1], threshold, left_node, right_node, 
                               self._cost(X_), len(X_), 
                               feature_labels = [self.feature_labels[feature]])
            
            self._build_tree(self.sklearn_tree.children_left[node], left_node, X_[left_mask])
            self._build_tree(self.sklearn_tree.children_right[node], right_node, X_[right_mask])
            
            self.node_count += 2
            
####################################################################################################