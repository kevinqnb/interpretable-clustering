from tree import *
from utils import *
from rule_pruning import *


class DecisionSet:
    """
    Base class for a decision set.
    """
    def __init__(self, norm = 2,
                center_init = None, centers = None,
                random_seed = None, feature_labels = None):
        
        """
        Args:   
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
        """
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
            self.n_centers = None
            
        self.feature_labels = feature_labels
        
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
            
        #self.X = None
        self.all_rules = None
        self.decision_set = None
        
        
    def _fit(self, X):
        """
        Fits a decision set to an input dataset. 
        
        Args:
            X (np.ndarray): Input dataset.
        """
        raise NotImplementedError('Method not implemented.')
        
        
    def fit(self, X):
        """
        Fits a decision set to an input dataset. 
        
        Args:
            X (np.ndarray): Input dataset.
        """
        #self.X = X
        self.all_rules = self._fit(X)
        self.decision_set = self.all_rules
        #self.decision_set, self.tree = self._fit(X)
        
        # Track coverage and costs of all the rules:
        self.covers = self.get_covers(X)
        self.costs = self.get_costs(X)
        
        
    def get_covers(self, X):
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            covers (dict[rule, np.ndarray]): Dictionary with rules as keys and arrays of indices
                of X as values.
        """
        raise NotImplementedError('Method not implemented.')
    
    def get_costs(self, X):
        """
        Finds the cost associated with each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            covers (dict[rule, float]): Dictionary with rules as keys and costs as values
        """
        raise NotImplementedError('Method not implemented.')
    
    
    def prune(self, X, q):
        """
        Prunes the decision set to q rules.
        
        Args:
            q (int): Number of rules to keep.
        """
        raise NotImplementedError('Method not implemented.')
    
    
    def predict(self, X):
        """
        Predicts the label(s) of each data point in X.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            labels (List[List[int]]): 2d list of predicted labels, with the internal list 
                at index i representing the group of decision rules which satisfy X[i,:].
        """
        
        set_covers = self.get_covers(X)
        
        labels = [[] for _ in range(len(X))]
        
        # THIS MAY NEED TO BE CHANGED based on the output of get_covers()
        for i,r_covers in set_covers.items():
            for j in r_covers:
                labels[j].append(i)
                
        # Replace empty lists with np.nan
        labels = [l if l else np.nan for l in labels]
        return labels
    
        
    
class DecisionForest(DecisionSet):
    """
    Class for a decision set which trains multiple decision trees according to a 
    random forest model, and returns the set of all leaf nodes as rules.
    """
    def __init__(self, tree_model, tree_params,
                 num_trees, max_depth = None, max_features = None, feature_pairings = None,
                 norm = 2, center_init = None, centers = None, random_seed = None,
                 feature_labels = None):
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
        super().__init__(norm, center_init, centers, random_seed, feature_labels)
        self.tree_model = tree_model
        self.tree_params = tree_params
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.max_features = max_features
        
        for pairing in feature_pairings:
            if not all(isinstance(i, int) for i in pairing):
                raise ValueError('Feature pairings must be a list of lists of integers.')
            
            #if len(pairing) < max_features:
            #    raise ValueError('Number of features must be less than or equal to the number of features in each pairing.')
            
        self.feature_pairings = feature_pairings
        
        self.covers = {}
        self.costs = {}
        
        
        
    def _fit(self, X):
        """
        Fits a decision set by training a forest of decision trees, 
        and using collecting their leaf nodes as rules.
        
        Args:
            X (np.ndarray): Input dataset.
            
        returns:
            rules (List[Rule]): List of rules.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        if self.feature_pairings is None:
            self.feature_pairings = [list(range(X.shape[1]))]
        
        n,d = X.shape
        rules = []
        tree = None
        
        
        for _ in range(self.num_trees):
            rand_depth = np.random.choice(self.max_depth) + 1
            pairing = self.feature_pairings[_ % len(self.feature_pairings)]
            num_features = np.random.choice(min(self.max_features, len(pairing))) + 1
            rand_features = np.random.choice(pairing, num_features, replace = False)
            
            rand_feature_labels = None
            if self.feature_labels is not None:
                rand_feature_labels = [self.feature_labels[i] for i in rand_features]
                
            if self.center_init is None:
                tree = self.tree_model(norm = self.norm,
                                    max_depth = rand_depth,
                                    #random_seed = self.random_seed, 
                                    feature_labels = rand_feature_labels,
                                    **self.tree_params)
                
            else:
                tree = self.tree_model(norm = self.norm,
                                    max_depth = rand_depth,
                                    #random_seed = self.random_seed, 
                                    center_init = self.center_init, 
                                    centers = self.centers[:,rand_features],
                                    feature_labels = rand_feature_labels,
                                    **self.tree_params)

            tree.fit(X[:, rand_features])
            
            leaves = find_leaves(tree.root)
            
            nodes_seen = []
            for i, leaf in leaves.items():
                # transform features back to original indices:
                for l in leaf[:-1]:
                    if l[0] not in nodes_seen:
                        l[0].features = [rand_features[f] for f in l[0].features]
                        nodes_seen.append(l[0])
                    
                rules.append(leaf)
                
        return rules #, tree
        
        
    def rule_covers(self, rule, X):
        """
        Finds data points of X covered by a rule.
        
        Args:
            rule (Rule): Rule object.
            
            X (List[int]): Input dataset.
            
        Returns:
            cover (np.ndarray): Array of indices of X covered by the rule.
        """
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
    def prune(self, X, q):
        """
        Prunes the decision set to q rules.
        
        Args:
            q (int): Number of rules to keep.
        """
        n,d = X.shape        
        search_range = np.linspace(0,10,1000)
        coverage_threshold = 0.95*n
        selected_rules = grid_search(q, self.covers, self.costs,
                                    search_range, coverage_threshold)
        #selected_rules = greedy_set_cover(n, self.covers, self.costs)
        self.decision_set = [self.all_rules[i] for i in selected_rules]
    '''
        
        
    def prune(self, q, data_labels, rule_labels):
        """
        Given cluster labels, prunes the decision set to at most 
        q rules for every cluster.
        """
        selected_rules = distorted_greedy2(q, data_labels, rule_labels, self.covers)
        return selected_rules
    
    
    def prune2(self, q, data_labels, rule_labels):
        selected_rules = greedy2(q, data_labels, rule_labels, self.covers)
        return selected_rules
    
    '''
    def prune3(self, q, data_labels, rule_labels):
        selected_rules = greedy3(q, data_labels, rule_labels, self.covers)
        return selected_rules
    '''
        


class MultiTree(DecisionSet):
    """
        Class for a decision set which iteratively trains a decision tree, removes the data points 
        from a single leaf (taking that leaf as a rule), and repeats the process until the desired
        number of rules is reached.
    """
    def __init__(self, num_decisions, tree_model, tree_params, max_depth,
                 norm = 2, center_init = None, centers = None, random_seed = None,
                 feature_labels = None):
        """
            Args:
                n_rules (int): Number of rules to be trained.
                
                tree_model (Tree Object): Decision tree model to be used in the 
                    creation of decision sets.
                    
                tree_params (dict[str: any]): Dictionary of parameters to be passed as input 
                    to the tree_model.
                    
                max_depth (int): Maximum depth of the decision tree.
        """
        super().__init__(norm, center_init, centers, random_seed, feature_labels)
        self.num_decisions = num_decisions
        self.tree_model = tree_model
        self.tree_params = tree_params
        self.max_depth = max_depth
        
        self.covers = {}
        self.costs = {}
        
        
    def cost(self, rule, X):
        """
        Calculates the cost of a rule.
        
        Args:
            rule (Rule): Rule object.
            
            X (np.ndarray): Input dataset.
            
        Returns:
            cost (float): Cost of the rule.
        """
        r_covers = self.rule_covers(rule, X)
        
        if len(r_covers) == 0:
            return np.inf
        else:
            if self.norm == 2:
                X_ = X[r_covers,:]
                mu = np.mean(X_, axis = 0)
                cost = np.sum(np.linalg.norm(X_ - mu, axis = 1)**2) #/len(X_)
                
            elif self.norm == 1:
                X_ = X[r_covers, :]
                eta = np.median(X_, axis = 0)
                cost = np.sum(np.abs(X_ - eta))
                
            return cost

    '''
    def coverage_gain(self, rule, X):
        """
        Calculates the coverage gain of a rule.
        
        Args:
            rule (Rule): Rule object.
            
        Returns:
            coverage (int): Number of data points covered by the rule.
        """
        
        r_covers = set(self.rule_covers(rule, X))
        for i, cover in self.covers.items():
            r_covers.union(set(cover))
        
        return len(r_covers)
    
    
    def cost_gain(self, rule, X):
        """
        Calculates the cost gain of a rule.
        
        Args:
            rule (Rule): Rule object.
            
        Returns:
            cost (float): Cost of the rule.
        """
        n,d = X.shape
        cost = n*d - self.cost(rule, X)
        
        #for i, c in self.costs.items():
        #    cost += n*d - c
            
        return cost
    '''
    
    def _fit(self, X):
        """
        Fits a decision set to an input dataset. 
        
        Args:
            X (np.ndarray): Input dataset.
        """
        rules = []
        n = len(X)
        
        X_ = copy.deepcopy(X)
        for _ in range(self.num_decisions):
            '''
            tree = self.tree_model(max_depth = self.q,
                                   norm = self.norm, center_init = self.center_init,
                                   centers = self.centers, random_seed = self.random_seed, 
                                   feature_labels = self.feature_labels, **self.tree_params)
            '''
            tree = self.tree_model(max_depth = self.max_depth,
                                   norm = self.norm, random_seed = self.random_seed, 
                                   feature_labels = self.feature_labels, **self.tree_params)
            
            tree.fit(X_)
            
            leaves = find_leaves(tree.root)
            
            min_leaf = None
            min_cost = np.inf
            for i, leaf in leaves.items():
                leaf_cost = self.cost(leaf, X)
                if leaf_cost < min_cost:
                    min_cost = leaf_cost
                    min_leaf = leaf
                    
            rules.append(min_leaf)
            self.costs[len(rules)-1] = self.cost(min_leaf, X)
            self.covers[len(rules)-1] = self.rule_covers(min_leaf, X)
            
            leaf_indices = self.rule_covers(min_leaf, X_)
            X_ = np.delete(X_, leaf_indices, axis = 0)
            
            '''
            max_leaf = None
            max_score = 0
            for i, leaf in leaves.items():
                leaf_cost = self.cost_gain(leaf, X)
                leaf_coverage = self.coverage_gain(leaf, X)
                if leaf_cost + leaf_coverage > max_score:
                    max_score = leaf_cost + leaf_coverage
                    max_leaf = leaf
                    
            rules.append(max_leaf)
            self.costs[len(rules)-1] = self.cost(max_leaf, X)
            self.covers[len(rules)-1] = self.rule_covers(max_leaf, X)
            leaf_indices = self.rule_covers(max_leaf, X_)
            X_ = np.delete(X_, leaf_indices, axis = 0)
            '''
            
        return rules
    
    def rule_covers(self, rule, X):
        """
        Finds data points of X covered by a rule.
        
        Args:
            rule (Rule): Rule object.
            
            X (List[int]): Input dataset.
            
        Returns:
            cover (np.ndarray): Array of indices of X covered by the rule.
        """
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