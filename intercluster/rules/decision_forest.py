from typing import List, Dict, Any, Tuple
from ._decision_sets import DecisionSet
from ..utils import *

class DecisionForest(DecisionSet):
    """
    Class for a decision set which trains multiple decision trees according to a 
    random forest model, and returns the set of all leaf nodes as rules.
    """
    def __init__(
        self,
        tree_model : Any,
        tree_params : Dict[str, Any],
        num_trees : int,
        #max_depth : int = None,
        max_features : int = None,
        feature_pairings : List[List[int]] = None,
        train_size : float = 1.0,
        #norm : int = 2,
        #center_init : str = None,
        #centers : NDArray = None,
        random_seed : int = None,
        feature_labels : List[str] = None
    ):
        """
        Args:
            tree_model (Tree Object): Decision tree model to be used in the 
                creation of decision sets.
                
            tree_params (dict[str: any]): Dictionary of parameters to be passed as input 
                to the tree_model.
                
            num_trees (int): Number of trees to be trained.
            
            num_features (int, optional): Number of features to be used in each tree. 
                If None, all features are used. Defaults to None.
                
            feature_pairings (List[List[int]]): List of feature indices representing sets 
                features which can be used together in a decision tree. 

        """
        #super().__init__(norm, center_init, centers, random_seed, feature_labels)
        super().__init__(random_seed, feature_labels)
        self.tree_model = tree_model
        self.tree_params = tree_params
        self.num_trees = num_trees
        #self.max_depth = max_depth
        self.max_features = max_features
        self.train_size = train_size
        
        for pairing in feature_pairings:
            if not all(isinstance(i, int) for i in pairing):
                raise ValueError('Feature pairings must be a list of lists of integers.')
            
            #if len(pairing) < max_features:
            #    raise ValueError('Number of features must be less than or equal to the number of features in each pairing.')
            
        self.feature_pairings = feature_pairings
        
        self.covers = {}
        self.costs = {}
        
        
        
    def _fitting(self, X):
        """
        Fits a decision set by training a forest of decision trees, 
        and using collecting their leaf nodes as rules.
        
        Args:
            X (np.ndarray): Input dataset.
            
        returns:
            rules (List[Rule]): List of rules.
        """
        n,d = X.shape
        if self.feature_pairings is None:
            self.feature_pairings = [list(range(d))]
            
        rules = []
        tree = None
        for _ in range(self.num_trees):
            '''
            #rand_depth = np.random.choice(self.max_depth) + 1
            rand_depth = self.max_depth
            pairing = self.feature_pairings[_ % len(self.feature_pairings)]
            #num_features = np.random.choice(min(self.max_features, len(pairing))) + 1
            num_features = self.max_features
            rand_features = np.random.choice(pairing, num_features, replace = False)
            
            rand_feature_labels = None
            if self.feature_labels is not None:
                rand_feature_labels = [self.feature_labels[i] for i in rand_features]
                
            if self.center_init is None:
                tree = self.tree_model(norm = self.norm,
                                    max_depth = rand_depth,
                                    feature_labels = rand_feature_labels,
                                    **self.tree_params)
                
            else:
                tree = self.tree_model(norm = self.norm,
                                    max_depth = rand_depth,
                                    center_init = self.center_init, 
                                    centers = self.centers[:,rand_features],
                                    feature_labels = rand_feature_labels,
                                    **self.tree_params)
            '''
            rand_pairing = np.random.randint(len(self.feature_pairings))
            pairing = self.feature_pairings[rand_pairing]
            rand_features = np.random.choice(
                pairing, 
                size = min(self.max_features, len(pairing)),
                replace = False
            )
            rand_samples = np.random.choice(
                n, 
                size = int(self.train_size * n),
                replace = False
            )
            train_data = X[rand_samples, :]
            train_data = train_data[:, rand_features]
            
            rand_feature_labels = None
            if self.feature_labels is not None:
                rand_feature_labels = [self.feature_labels[i] for i in rand_features]
            
            
            if self.tree_model.__name__ == 'SklearnTree':
                tree = self.tree_model(
                    feature_labels = rand_feature_labels,
                    **dict(self.tree_params, 
                           data_labels = self.tree_params['data_labels'][rand_samples]
                        )
                )
            
            else:
                tree = self.tree_model(
                    feature_labels = rand_feature_labels,
                    **self.tree_params
                )
            
                '''
                rand_depth = np.random.randint(2, self.tree_params['max_depth'] + 1)
                tree = self.tree_model(
                    feature_labels = rand_feature_labels,
                    **dict(self.tree_params, max_depth=rand_depth)
                )
                '''
            
            tree.fit(train_data)
            
            leaves = find_leaves(tree.root)
            
            # NOTE: I don't love this part... this may be another argument for 
            # having feature subsets be input to the tree object.
            nodes_seen = []
            for _, leaf_path in leaves.items():
                # The last node in the path (leaf) is irrelevant for the rule:
                for decision in leaf_path[:-1]:
                    node_obj  = decision[0]
                    if node_obj not in nodes_seen:
                        # transform features back to original indices:
                        node_obj.features = [rand_features[f] for f in node_obj.features]
                        nodes_seen.append(node_obj)
                    
                rules.append(leaf_path)
                
        return rules
        
        
    def rule_covers(self, rule, X):
        """
        Finds data points of X covered by a rule.
        
        Args:
            rule (Rule): Rule object.
            
            X (List[int]): Input dataset.
            
        Returns:
            cover (List[int]): Array of indices of X covered by the rule.
        """
        # NOTE: This is highly specialized for the tree structure.
        # Maybe this should also be a method of the tree object?
        # Or somewhere else like utils?
        cover = []
        for i,x in enumerate(X):
            satisfied = True
            for r in rule[:-1]:
                node = r[0]
                direction = r[1]
                
                if direction == 'left':
                    if np.dot(x[node.features], node.weights) > node.threshold:
                        satisfied = False
                        
                elif direction == 'right':
                    if np.dot(x[node.features], node.weights) <= node.threshold:
                        satisfied = False
            
            if satisfied:
                cover.append(i)
                
        return cover
    
    
    def get_covers(self, X):
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            covers (dict[int, List[int]]): Dictionary with rules labels 
            as keys and arrays of indices of X as values.
        """
        covers = {}
        for i, rule in enumerate(self.decision_set):
            covers[i] = self.rule_covers(rule, X)
        return covers
    
    '''
    # NOTE: Not sure how necessary this part is...
    def cost(self, X_):
        """
        Calculates the cost for a subset of the data X_.
        If self.centers is None, the cost is the sum of the squared distances of each point to
        the mean of the subset. Otherwise, cost is taken as the sum of squared distances to the 
        the center closest to the subset.
        
        Args:
            X_ (np.ndarray): Input dataset.
            
        Returns:
            cost (float): Cost of the subset.
        """
        
        if self.centers is None:
            mu = np.mean(X_, axis = 0)
            cost = np.sum(np.linalg.norm(X_ - mu, axis = 1)**2)
            
        else:
            cost = np.inf
            for center in self.centers:
                cd = np.sum(np.linalg.norm(X_ - center, axis = 1)**2)
                if cd < cost:
                    cost = cd
                
        return cost
        
    
    def get_costs(self, X):
        """
        Finds the cost associated with each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            covers (dict[int, float]): Dictionary with rules labels as keys and costs as values
        """
        if self.covers is None:
            self.covers = self.get_covers(X)
            
        costs = {}
        for i, cover in self.covers.items():
            if len(cover) != 0:
                costs[i] = self.cost(X[cover, :])
            else:
                costs[i] = np.inf
            
        return costs
    '''