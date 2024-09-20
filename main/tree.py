import numpy as np
import heapq

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
        feature (int): The axis aligned feature to split on. 
        
        feature_label (str): Name for the given feature (for printing and display). 
        
        threshold (float): The value of feature to split on.
        
        left_child (Node): (For non-leaf nodes only) Pointer to the left branch of the current node.
        
        right_child (Node): (For non-leaf nodes only) Pointer to the right branch of the 
            current node. 
            
        points (np.ndarray): The set of data points associated with points belonging to this node.  
        
        cost (float): The cost associated with points belonging to this node. 
        
    """
    def __init__(self):
        self.random_val = np.random.uniform()
        self.type = None
        self.label = None
        self.feature = None
        self.threshold = None
        self.left_child = None
        self.right_child = None
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
    
    def tree_node(self, feature, threshold, left_child, right_child, cost, points, feature_label):
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
        self.feature = feature 
        self.threshold = threshold
        self.left_child = left_child 
        self.right_child = right_child
        self.cost = cost
        self.points = points
        self.feature_label = feature_label
        
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
    def __init__(self, max_leaf_nodes=None, max_depth=None, min_points_leaf = 1, random_seed = None):
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
            
        Attributes:
            heap (heapq list): Maintains the heap structure of the tree.
            leaf_count (int): Number of leaves in the tree.
            node_count (int): Number of nodes in the tree.
        """
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_points_leaf = min_points_leaf
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.feature_labels = None
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
        Chooses an axis aligned threshold value to split upon given some input dataset X.

        Args:
            X_ (np.ndarray): Input dataset.
            
        Returns:
                Tuple(int, float): A (feature, threshold) split pair. 
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
        # Set feature labels:
        if feature_labels is None:
            self.feature_labels = [None]*X.shape[1]
        else:
            if not len(feature_labels) == X.shape[1]:
                raise ValueError('Labels must match the shape of the data.')
            self.feature_labels = feature_labels
        
        # if stopping criteria weren't provided, set to the maximum possible
        if self.max_leaf_nodes is None:
            self.max_leaf_nodes = len(X)
        
        if self.max_depth is None:
            self.max_depth = len(X) - 1
        
        # Keeping a heap queue for the current leaf nodes in the tree.
        # It prioritizes splitting the leaves with the largest cost.
        # NOTE: I store -1*cost because heapq pops items with minimum cost by default.
        # Heap leaf object: (-1*cost, Node object, data subset, depth)
        self.root = Node()
        heapq.heappush(self.heap, (-1*self._cost(X), self.root, X, 0))
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
        
        
        # If we've reached the max depth or max number of leaf nodes -- stop recursing 
        if depth >= self.max_depth or self._count_leaves() >= (self.max_leaf_nodes - 1) or len(X_) <= 1:
            class_label = self.leaf_count 
            self.leaf_count += 1
            node_obj.leaf_node(class_label, self._cost(X_), len(X_))


        # Otherwise, find the best split and recurse into left and right branches
        else:
            feature, threshold = self._find_best_split(X_)
            left_mask = X_[:, feature] <= threshold
            right_mask = ~left_mask
            
            left_node = Node()
            right_node = Node()
            
            node_obj.tree_node(feature, threshold, left_node, right_node, 
                               self._cost(X_), len(X_), feature_label = self.feature_labels[feature])
            
            left_obj = (-1*self._cost(X_[left_mask]), left_node, X_[left_mask], depth + 1)
            right_obj = (-1*self._cost(X_[right_mask]), right_node, X_[right_mask], depth + 1)
            
            heapq.heappush(self.heap, left_obj)
            heapq.heappush(self.heap, right_obj)
            self.node_count += 2


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
        
        if sample[node.feature] <= node.threshold:
            return self._predict_sample(sample, node.left_child)
        else:
            return self._predict_sample(sample, node.right_child)
        
        
####################################################################################################


class RandomTree(Tree):
    """
    Inherits from the Tree class to implement a decision tree in which split criterion are randomly 
    chosen from the input space.
    
    NOTE: In order to imitate results from [Galmath, Jia, Polak, Svensson 2021], fit this model on 
        a dataset of centers or representative points. 
        
    Args:
        max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
            Defaults to None.
            
        max_depth (int, optional): Optional constraint for maximum depth. 
            Defaults to None.
            
        min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
            within a single leaf. Defaults to 1.
            
        random_seed (int, optional): Random seed. In the Tree object randomness is only ever 
            used for breaking ties between nodes, or if you are using a RandomTree!
        
    Attributes:
        heap (heapq list): Maintains the heap structure of the tree.
        leaf_count (int): Number of leaves in the tree.
        node_count (int): Number of nodes in the tree.
    
    """
    def __init__(self, max_leaf_nodes=None, max_depth=None, min_points_leaf = 1, random_seed = None):
        super().__init__(max_leaf_nodes, max_depth, min_points_leaf, random_seed)
        
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
            while not found:
                # Randomly sample a feature and a value for an axis aligned split:
                rand_feature = np.random.choice(range(X_.shape[1]))
                interval = (np.min(X_[:, rand_feature]), np.max(X_[:, rand_feature]))
                rand_cut = np.random.uniform(interval[0], interval[1])
                left_mask = X_[:, rand_feature] <= rand_cut
                right_mask = ~left_mask
                
                # If sampled split separates at least two centers, then accept:
                left_branch = X_[left_mask]
                right_branch = X_[right_mask]
                if len(left_branch[0]) > 0 and len(right_branch[0]) > 0:
                    found = True
                    split = (rand_feature, rand_cut)

            return split
        
        
####################################################################################################
            

class KMeansTree(Tree):
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
        
    Attributes:
        heap (heapq list): Maintains the heap structure of the tree.
        leaf_count (int): Number of leaves in the tree.
        node_count (int): Number of nodes in the tree.
    """
    def __init__(self, max_leaf_nodes=None, max_depth=None, min_points_leaf = 1, random_seed = None):
        super().__init__(max_leaf_nodes, max_depth, min_points_leaf, random_seed)
        
        
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


    def _find_best_split(self, X_):
            """
            Chooses the axis aligned split (feature, threshold) with smallest 
            sum of costs between the two branches it creates.
            
            Args:
                X_ (np.ndarray): Input dataset.
                
            Returns:
                Tuple(int, float): A (feature, threshold) split pair. 
            """
            
            best_split = None
            best_split_val = np.inf
            
            for feature in range(X_.shape[1]):
                unique_vals = np.unique(X_[:,feature])
                for threshold in unique_vals:
                    split = (feature, threshold)

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

class KMediansTree(Tree):
    """
    Inherits from the Tree class to implement a decision tree in which split criterion chosen so
    that points in any leaf node are close in distance to their median. 
    
    Args:
        max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
            Defaults to None.
            
        max_depth (int, optional): Optional constraint for maximum depth. 
            Defaults to None.
            
        min_points_leaf (int, optional): Optional constraint for the minimum number of points. 
            within a single leaf. Defaults to 1.
            
        random_seed (int, optional): Random seed. In the kMediansTree object randomness is only ever 
            used for breaking ties between nodes which have the same cost.
        
    Attributes:
        heap (heapq list): Maintains the heap structure of the tree.
        leaf_count (int): Number of leaves in the tree.
        node_count (int): Number of nodes in the tree.
    """
    def __init__(self, max_leaf_nodes=None, max_depth=None, min_points_leaf = 1, random_seed = None):
        super().__init__(max_leaf_nodes, max_depth, min_points_leaf, random_seed)
        
        
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
            
            best_split = None
            best_split_val = np.inf
            
            for feature in range(X_.shape[1]):
                unique_vals = np.unique(X_[:,feature])
                for threshold in unique_vals:
                    split = (feature, threshold)

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