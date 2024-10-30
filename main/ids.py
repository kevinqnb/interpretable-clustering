from math import comb
import itertools
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from mlxtend.frequent_patterns import fpgrowth
from utils import *
from rule_clustering import *

####################################################################################################

def f1(itemsets, rule_itemsets):
    size_diff = len(itemsets) - len(rule_itemsets)
    return size_diff

def f2(itemsets, rule_itemsets):
    wmax = max([len(i) for i in itemsets])
    wsum = np.sum([wmax - len(j) for j in rule_itemsets])
    return wsum
      
def f3(n, rule_item_covers):
    '''
    overlap_score = 0
    for r1, r2 in itertools.combinations(range(len(rule_item_covers)),2):
        overlaps = rule_item_covers[r1].intersection(rule_item_covers[r2])
        overlap_score += n - len(overlaps)
        
    return overlap_score
    '''
    N = n * comb(len(rule_item_covers), 2)
    
    ruleset_labels = [[] for _ in range(n)]
    for i,r in enumerate(rule_item_covers):
        for j in r:
            ruleset_labels[j].append(i)
            
    overlaps = 0
    for labs in ruleset_labels:
        overlaps += comb(len(labs), 2)
    
    return N - overlaps
    

def f(n, itemsets, rule_itemsets, rule_item_covers, lambdas):
    score = 0
    score += lambdas[0]*f1(itemsets, rule_itemsets)
    score += lambdas[1]*f2(itemsets, rule_itemsets)
    score += lambdas[2]*f3(n, rule_item_covers)
    return score
    
####################################################################################################
    
def get_covers(df, rule_itemsets):
    rule_item_covers = []
    for r in rule_itemsets:
        columns = list(r)
        row_indices = df[columns].all(axis=1)
        r_covers = set(df.index[row_indices])
        rule_item_covers.append(r_covers)
        
    return rule_item_covers



def sample(itemsets, A, delta):
    probabilities = [(1 + delta)/2 if i in A else (1 - delta)/2 for i in itemsets]
    sampled_items = [item for item, prob in zip(itemsets, probabilities)
                     if np.random.uniform() < prob]
    
    return sampled_items

####################################################################################################


def rule_inclusion_estimate(rule, n, itemsets, itemsets_covers_dict, A, lambdas, delta, opt):
    X2 = len(itemsets)**2
    SE = np.inf
    est_add = []
    est_remove = []
    iteration = 0

    # Until 95% confidence within the error range:
    chunk_size = 10
    while 1.96 * SE > opt/X2 :
        iteration += chunk_size

        for _ in range(chunk_size):
            sample1 = sample(itemsets, A, delta)
            if rule not in sample1:
                sample1.append(rule)

            sample1_covers = [itemsets_covers_dict[i] for i in sample1]
            sample1_score = f(n, itemsets, sample1, sample1_covers, lambdas)

            sample2 = sample(itemsets, A, delta)
            if rule in sample2:
                sample2.remove(rule)

            sample2_covers = [itemsets_covers_dict[i] for i in sample2]
            sample2_score = f(n, itemsets, sample2, sample2_covers, lambdas)


            est_add.append(sample1_score)
            est_remove.append(sample2_score)

        SE = np.sqrt((np.std(est_add)**2 + np.std(est_remove)**2)/iteration)

    omega = np.mean(est_add) - np.mean(est_remove)
    return omega
    

def search_estimates(n, itemsets, itemsets_covers_dict, A, lambdas, delta, opt):
    def compute_omega(i, x):
        return rule_inclusion_estimate(x, n, itemsets, itemsets_covers_dict, A, lambdas, delta, opt)
    
    omega_vals = Parallel(n_jobs=-1)(delayed(compute_omega)(i, x) for i, x in enumerate(itemsets))
    return omega_vals
            

def smooth_local_search(df, itemsets, lambdas, delta, delta_):
    n = len(df)
    X2 = len(itemsets)**2
    itemsets_covers = get_covers(df, itemsets)
    itemsets_covers_dict = {x: itemsets_covers[i] for i,x in enumerate(itemsets)}
    
    A = set()
    sampled = sample(itemsets, A, 0)
    sampled_covers = [itemsets_covers_dict[i] for i in sampled]
    opt = f(n, itemsets, sampled, sampled_covers, lambdas)
    
    mod = True
    while mod:
        omegas = search_estimates(n, itemsets, itemsets_covers_dict, A, lambdas, delta, opt)
        
        added = False
        for i,x in enumerate(itemsets):
            if x not in A and omegas[i] > 2*opt/X2:
                A.add(x)
                added = True
                
        removed = False
        if not added:
            for i,x in enumerate(itemsets):
                if x in A and omegas[i] < -2*opt/X2:
                    A.remove(x)
                    removed = True
                    
        if (not added) and (not removed):
            mod = False
            
    return sample(itemsets, A, delta_)

####################################################################################################

class frequent_itemsets:
    def __init__(self, n_bins, freq):
        # NEED TO ADD DOCSTRINGS
        # NEED a way to allow interpretation of itemsets and rulesets.
        
        self.n_bins = n_bins
        self.freq = freq
        self.itemsets = None
        
    def binning(self, X):
        hist, hist_edges = np.histogramdd(X, bins=self.n_bins)

        binned = np.zeros(X.shape, dtype = int)
        for d in range(X.shape[1]):
            binned[:, d] = np.digitize(X[:, d], bins=hist_edges[d])
            
        encoder_dict = {}
        for i in range(X.shape[1]):
            for j in range(self.n_bins + 2):
                if j == 0:
                    encoder_dict[i*(self.n_bins + 2) + j] = (i, [-np.inf, hist_edges[i][j]])
                elif j == self.n_bins + 1:
                    encoder_dict[i*(self.n_bins + 2) + j] = (i, [hist_edges[i][j-1], np.inf])
                else:
                    encoder_dict[i*(self.n_bins + 2) + j] = (i, [hist_edges[i][j-1],
                                                                 hist_edges[i][j]])
                    
        enc = OneHotEncoder(categories = [list(range(self.n_bins + 2)) 
                                          for _ in range(X.shape[1])])
        one_hot = enc.fit_transform(binned)
        df_sparse = pd.DataFrame.sparse.from_spmatrix(one_hot)
        df_sparse = df_sparse.astype(pd.SparseDtype(int, fill_value=0))
        df_sparse = df_sparse.astype(pd.SparseDtype(bool, fill_value=False))
        
        return df_sparse
    
    def fit(self, X):
        df = self.binning(X)
        fsets = fpgrowth(df, min_support=self.freq)
        self.itemsets = list(fsets.itemsets)
        
        
    def predict(self, X):
        df = self.binning(X)
        itemsets_covers = get_covers(df, self.itemsets)
        
        itemset_labels = [[] for _ in range(len(df))]
        for i,r in enumerate(itemsets_covers):
            for j in r:
                itemset_labels[j].append(i)
                
        # Replace empty lists with np.nan
        itemset_labels = [labels if labels else np.nan for labels in itemset_labels]
        return itemset_labels


####################################################################################################

class IDS(frequent_itemsets):
    def __init__(self, n_bins, freq, lambdas):
        super().__init__(n_bins, freq)
        self.lambdas = lambdas
        self.delta = 1/3
        self.delta_ = 1/3
        self.rulesets = None
        
    def fit(self, X):
        df = self.binning(X)    
        fsets = fpgrowth(df, min_support=self.freq)
        self.itemsets = list(fsets.itemsets)
        self.rulesets = smooth_local_search(df, self.itemsets, 
                                            self.lambdas, self.delta, self.delta_)
        
    def predict(self, X):
        df = self.binning(X)
        rulesets_covers = get_covers(df, self.rulesets)
        
        ruleset_labels = [[] for _ in range(len(df))]
        for i,r in enumerate(rulesets_covers):
            for j in r:
                ruleset_labels[j].append(i)
                
        # Replace empty lists with np.nan
        ruleset_labels = [labels if labels else np.nan for labels in ruleset_labels]
        return ruleset_labels
    
    
####################################################################################################


def g1(rule_covers_dict):
    """
    Computes a submodular coverage objective of the given rule set.
    
    Args:
        rule_covers_dict (dict[set: set]): A dictionary where keys are itemsets (rules) and 
            values are the sets of data point indices covered by the rule.
    
    Returns:
        int: The number of unique data points covered by the rule set.
    """
    S = set()
    for r in rule_covers_dict.values():
        S.update(r)
    return len(S)

def g2(wmax, rulesets):
    """
    Computes a submodular width objective for a given rule set. 
    The width of a rule is defined as the number of items in the rule's itemset.
    
    Args:
        wmax (int): The maximum width of any rule in the itemset.
        rulesets (list[set]): A sub-list of of selected rule itemsets.
        
    Returns:
        int: The width objective of the given rule set.
    """
    S = np.sum([wmax - len(j) for j in rulesets])
    return S

def g3(n, d, rule_cost_dict):
    """
    Computes a submodular cost objective for a fiven rule set. The cost of each rule is defined as 
    the sum of distances between its covered points and their mean. 
    
    NOTE: This currently assumes that each feature is scaled to a [0,1] range.
    NOTE: This currently assumes distances are measured via the squared L2 norm. 
    
    Args:
        X (np.ndarray): The associated n x d numpy array dataset.
        
        rule_covers_dict (dict[set: set]): A dictionary where keys are itemsets (rules) and
            values are the sets of data point indices covered by the rule.
            
    Returns:
        float: The cost objective of the given rule set.
    """
    S = np.sum([n*d - c for c in rule_cost_dict.values()])
    #for r in rule_covers_dict.values():
    #    mu_r = np.mean(X[list(r),:], axis = 0)
    #    S += n*d - np.sum((X[list(r),:] - mu_r)**2)
    return S


def g(rulesets, rule_covers_dict, rule_cost_dict, lambdas, n, d, wmax):
    """
    Submodular objective function. 
    
    Args:
        X (np.ndarray): The associated n x d numpy array dataset.
        
        itemsets (list[set]): A complete list of all itemsets.
        
        rule_itemsets (list[set]): A sub-list of selected rule itemsets.
        
        rule_covers_dict (dict[set: set]): A dictionary where keys are itemsets (rules) and
            values are the sets of data point indices covered by the rule.
            
        lambdas (list[float]): A list of weights for each submodular objective.
        
    Returns:
        float: The submodular objective score of the given rulesets.
    """
    score = 0
    score += lambdas[0]*g1(rule_covers_dict)
    score += lambdas[1]*g2(wmax, rulesets)
    score += lambdas[2]*g3(n, d, rule_cost_dict)
    return score


def greedy_select(k, itemsets, itemsets_covers_dict, 
                  itemsets_cost_dict, lambdas, n ,d, wmax): 
        A = copy.deepcopy(itemsets)
        rulesets = []
        rule_covers_dict = {}
        rule_cost_dict = {}
        score = 0
        while len(rulesets) < k:
            best_index = None
            best_rule = None
            best_score = -np.inf
            
            for i,iset in enumerate(A):
                rulesets.append(iset)
                rule_covers_dict[i] = itemsets_covers_dict[iset]
                rule_cost_dict[i] = itemsets_cost_dict[iset]
                
                score = g(rulesets, rule_covers_dict, rule_cost_dict,
                                       lambdas, n, d, wmax)
                
                rulesets.pop()
                del rule_covers_dict[i]
                del rule_cost_dict[i]

                if score > best_score:
                    best_index = i
                    best_rule = iset
                    best_score = score

            rulesets.append(best_rule)
            rule_covers_dict[best_rule] = itemsets_covers_dict[best_rule]
            rule_cost_dict[best_rule] = itemsets_cost_dict[best_rule]
            #score = sum(rule_cost_dict.values())
            del A[best_index]

        return rulesets #, score



class DecisionSets(frequent_itemsets):
    def __init__(self, n_bins, freq, k, lambdas = None):
        super().__init__(n_bins, freq)
        self.k = k
        
        self.find_lambdas = False
        if lambdas is None:
            self.find_lambdas = True
        self.lambdas = lambdas
        
        self.itemsets = None
        self.itemsets_covers_dict = None
        self.itemsets_cost_dict = None
        self.wmax = None
        self.df = None
        self.rulesets = None
        
    def search(self, X):
        n, d = X.shape
        kmeans = KMeans(n_clusters=3, n_init="auto").fit(X)
        
        # Search for lambda values
        search_range = np.linspace(0,100,1000)
        
        lambdas_prev = np.random.uniform(size = 3)
        lambdas = np.random.uniform(size = 3)
        
        def evaluate_lambda(i, s):
            lambdas_copy = np.copy(lambdas)
            lambdas_copy[i] = s
            rulesets = greedy_select(self.k, self.itemsets, self.itemsets_covers_dict,
                                     self.itemsets_dists_dict, lambdas_copy, n, d, self.wmax)
            score = sum([self.itemsets_dists_dict[r]/len(self.itemsets_covers_dict[r])
                         for r in rulesets])
            return score, s
        
        while not np.array_equal(lambdas_prev, lambdas):
            lambdas_prev = np.copy(lambdas)
            for i in range(3):                
                search_results = Parallel(n_jobs=-1)(delayed(evaluate_lambda)(i, s) 
                                                     for s in search_range)
                #print(search_results)
                #breakpoint()
                best_score, best_val = min(search_results, key=lambda x: x[0])
                lambdas[i] = best_val
                
        return lambdas
        
        
    def fit(self, X):
        n, d = X.shape
        
        # Bin and find frequent itemsets
        self.df = self.binning(X)    
        fsets = fpgrowth(self.df, min_support=self.freq)
        self.itemsets = list(fsets.itemsets)
        
        # Prepare itemsets covers and costs
        if len(self.itemsets) < self.k:
            raise ValueError("Number of itemsets is less than k, raise frequency or reduce k.")
        
        itemsets_covers = get_covers(self.df, self.itemsets)
        self.itemsets_covers_dict = {iset: itemsets_covers[i] 
                                     for i,iset in enumerate(self.itemsets)}
        
        self.itemsets_cost_dict = {}
        for i,iset in enumerate(self.itemsets):
            Xi = X[list(self.itemsets_covers_dict[iset]),:]
            mui = np.mean(Xi, axis = 0)
            self.itemsets_cost_dict[iset] = np.sum((Xi - mui)**2)
            
        self.wmax = max([len(i) for i in self.itemsets])
        
        kmeans = KMeans(n_clusters=3, n_init="auto").fit(X)
        self.itemsets_dists_dict = {}
        for i,iset in enumerate(self.itemsets):
            Xi = X[list(self.itemsets_covers_dict[iset]),:]
            diffs = Xi[np.newaxis, :, :] - kmeans.cluster_centers_[:, np.newaxis, :]
            distances = np.sum(diffs ** 2, axis=-1)
            sum_array = np.sum(distances, axis=1)
            closest_dist = np.min(sum_array)
            self.itemsets_dists_dict[iset] = closest_dist
        
        # Search for lambda values
        if self.find_lambdas:
            self.lambdas = self.search(X)
        
        # Greedily select from the submodular objective
        self.rulesets = greedy_select(self.k, self.itemsets, self.itemsets_covers_dict, 
                  self.itemsets_cost_dict, self.lambdas, n ,d, self.wmax)
        
        
    def predict(self, X):
        df = self.binning(X)
        rulesets_covers = get_covers(df, self.rulesets)
        
        ruleset_labels = [[] for _ in range(len(df))]
        for i,r in enumerate(rulesets_covers):
            for j in r:
                ruleset_labels[j].append(i)
                
        # Replace empty lists with np.nan
        ruleset_labels = [labels if labels else np.nan for labels in ruleset_labels]
        return ruleset_labels