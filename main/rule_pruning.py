import numpy as np
import copy
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

def g1(rule_covers_dict):
    """
    Computes a submodular coverage objective of the given rule set.
    
    Args:
        rule_covers_dict (dict[int: set]): A dictionary where keys are integers (rule labels) and 
            values are the sets of data point indices covered by the rule.
    
    Returns:
        int: The number of unique data points covered by the rule set.
    """
    S = set()
    for r in rule_covers_dict.values():
        S.update(r)
    return len(S)

'''
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
'''

def g3(n, d, rule_cost_dict):
    """
    Computes a submodular cost objective for a fiven rule set. 
    
    NOTE: This currently assumes that each feature is scaled to a [0,1] range.
    
    Args:
        X (np.ndarray): The associated n x d numpy array dataset.
        
        rule_covers_dict (dict[int: set]): A dictionary where keys are integers (rule labels) and
            values are the sets of data point indices covered by the rule.
            
    Returns:
        float: The cost objective of the given rule set.
    """
    S = np.sum([n*d - c for c in rule_cost_dict.values()])
    return S


def g(rule_covers_dict, rule_cost_dict, lambdas, n, d):
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
    #score += lambdas[1]*g2(wmax, rulesets)
    score += lambdas[1]*g3(n, d, rule_cost_dict)
    return score


def greedy_select(q, rules, rule_covers_dict, 
                  rule_cost_dict, n ,d, lambdas): 
        R = rules
        S_idx = []
        S = []
        S_covers_dict = {}
        S_cost_dict = {}
        score = 0

        while len(S) < q:
            best_index = None
            best_rule = None
            best_score = -np.inf
            
            for i,r in enumerate(R):
                if i not in S_idx:
                    S.append(r)
                    S_covers_dict[i] = rule_covers_dict[i]
                    S_cost_dict[i] = rule_cost_dict[i]
                    
                    score = g(rule_covers_dict, rule_cost_dict,
                                        lambdas, n, d)
                    
                    S.pop()

                    if score > best_score:
                        best_index = i
                        best_rule = r
                        best_score = score

            if best_index is None or best_rule is None:
                break
            else:
                S.append(best_rule)
                S_idx.append(best_index)
                S_covers_dict[best_index] = rule_covers_dict[best_index]
                S_cost_dict[best_index] = rule_cost_dict[best_index]

        return S, S_covers_dict, S_cost_dict
    
    
def greedy_search(X, q, rules, rule_covers_dict, 
                  rule_cost_dict): 
    n, d = X.shape
    kmeans = KMeans(n_clusters=7, n_init="auto").fit(X)
    
    # Search for lambda values
    search_range = np.linspace(0,100,1000)
    
    lambdas_prev = np.random.uniform(size = 2)
    lambdas = np.random.uniform(size = 2)
    
    def evaluate_lambda(i, s):
        lambdas_copy = np.copy(lambdas)
        lambdas_copy[i] = s
        rulesets, rulesets_covers, rulesets_costs = greedy_select(q, rules, rule_covers_dict,
                                                                rule_cost_dict, n, d, lambdas_copy)
        
        clustering_score = 0
        for ridx, covers in rulesets_covers.items():
            for j in covers:
                min_distance = np.min([np.linalg.norm(X[j] - center) 
                                        for center in kmeans.cluster_centers_])
                clustering_score += min_distance

        #clustering_score /= g1(rulesets_covers)
        return clustering_score, s
    
    while not np.array_equal(lambdas_prev, lambdas):
        lambdas_prev = np.copy(lambdas)
        for i in range(2):                
            search_results = Parallel(n_jobs=-1)(delayed(evaluate_lambda)(i, s) 
                                                    for s in search_range)
            #search_results = [evaluate_lambda(i, s) for s in search_range]
            #print(search_results)
            #breakpoint()
            best_score, best_val = min(search_results, key=lambda x: x[0])
            lambdas[i] = best_val
            
    return greedy_select(q, rules, rule_covers_dict, rule_cost_dict, n, d, lambdas)



def distorted_greedy(q, rule_covers_dict, rule_cost_dict, lambda_val):
    """
    Implements a distorted greedy algorithm for rule selection. Uses an 
    objective function which combines coverage of datapoints, and the cost 
    of a rule in terms of clustering distances. 
    
    This is attributed to [Harshaw et al. 2019] in their paper titled
    "Submodular Maximization Beyond Non-negativity: Guarantees, Fast Algorithms, and Applications"
    
    Please see their work for more information on distorted greedy and the combination of 
    submodular and linear objectives.
    
    Args:
        q (int): The number of rules to select.
        
        rule_covers_dict (dict[int: set]): A dictionary where keys are integers (rule labels) and 
            values are the sets of data point indices covered by the rule.
        
        rule_cost_dict (dict[int: float]): A dictionary where keys are integers (rule labels) and 
            values are the cost of the rule.
        
        n (int): The number of data points in the dataset.
        
        d (int): The number of features in the dataset.
    """
    
    S = []
    S_covers = set()
    rule_list = list(rule_covers_dict.keys())
     
    for i in range(q):
        best_rule = None
        best_obj = -np.inf
        for r in rule_list:
            if r not in S:
                g = (len(S_covers.union(rule_covers_dict[r])) - len(S_covers))
                c = rule_cost_dict[r]
                
                obj = (1 - 1/q)**(q - (i + 1)) * g - lambda_val * c
                
                if obj > best_obj:
                    best_obj = obj
                    best_rule = r
                    
        if best_obj > 0:
            S.append(best_rule)
            S_covers = S_covers.union(rule_covers_dict[best_rule])
            
    return S


def grid_search(q, rule_covers_dict, rule_cost_dict, search_range, coverage_threshold):
    """
    Performs a grid search over normalization values for
    the distorted greedy objective.  
    
    Args:
        q (int): The number of rules to select.
        
        rule_covers_dict (dict[int: set]): A dictionary where keys are integers (rule labels) and 
            values are the sets of data point indices covered by the rule.
        
        rule_cost_dict (dict[int: float]): A dictionary where keys are integers (rule labels) and 
            values are the cost of the rule.
        
        search_range (np.ndarray): A range of lambda values to search over.
        
        coverage_threshold (float): The minimum number of data points that must
            be covered by the selected rules.
    """
    
    def evaluate_lambda(lambda_val):
        selected = distorted_greedy(q, rule_covers_dict, rule_cost_dict, lambda_val)
        
        covered = set()
        for r in selected:
            covered = covered.union(rule_covers_dict[r])
            
        if len(covered) < coverage_threshold:
            return np.inf, lambda_val
        
        else:
            return np.sum([rule_cost_dict[r] for r in selected])/len(covered), lambda_val
               
    search_results = Parallel(n_jobs=-1)(delayed(evaluate_lambda)(s) 
                                            for s in search_range)
    best_score, best_val = min(search_results, key=lambda x: x[0])
            
    return distorted_greedy(q, rule_covers_dict, rule_cost_dict, best_val)



def greedy_set_cover(n, rule_covers_dict, rule_cost_dict):
    """
    Implements a greedy set cover selection 
    """
    updated_covers  ={i: set(cover) for i, cover in rule_covers_dict.items()}
    S = []
    S_covers = set()
    
    while len(S_covers) < n:
        best_rule = None
        best_score = np.inf
        
        for i, covers in updated_covers.items():
            if len(covers) != 0:
                score = rule_cost_dict[i]/len(covers)
                if score < best_score:
                    best_score = score
                    best_rule = i
                
        S.append(best_rule)
        S_covers = S_covers.union(updated_covers[best_rule])
        
        for i, covers in updated_covers.items():
            updated_covers[i] = covers.difference(S_covers)
        
        
    return S



def distorted_greedy2(q, data_labels, rule_labels, rule_covers_dict):
    unique_labels = np.unique(rule_labels)
    points_to_cover = {l: set(np.where(data_labels == l)[0]) for l in unique_labels}
    covered_so_far = {l: set() for l in unique_labels}
    
    S = []
    rule_list = list(rule_covers_dict.keys())
    for i in range(q):
        best_rule = None
        best_rule_label = None
        best_score = -np.inf
        
        for r in rule_list:
            if r not in S:
                rlabel = rule_labels[r]
                label_covered = covered_so_far[rlabel]
                to_cover = points_to_cover[rlabel]
                r_covers = to_cover.intersection(rule_covers_dict[r])
                
                g = len(label_covered.union(r_covers)) - len(label_covered)
                c = len(rule_covers_dict[r]) - len(r_covers)
                
                score = (1 - 1/q)**(q - (i + 1)) * g - c
                
                if score > best_score:
                    best_rule = r
                    best_rule_label = rlabel
                    best_score = score
                    
        if best_score > 0:
            S.append(best_rule)
            covered_so_far[best_rule_label] = covered_so_far[best_rule_label].union(rule_covers_dict[best_rule])
            
    return S


def greedy2(q, data_labels, rule_labels, rule_covers_dict):
    unique_labels = np.unique(rule_labels)
    points_to_cover = {l: set(np.where(data_labels == l)[0]) for l in unique_labels}
    covered_so_far = {l: set() for l in unique_labels}
    
    S = []
    rule_list = list(rule_covers_dict.keys())
    for i in range(q):
        best_rule = None
        best_rule_label = None
        best_score = -np.inf
        
        for r in rule_list:
            if r not in S:
                rlabel = rule_labels[r]
                label_covered = covered_so_far[rlabel]
                to_cover = points_to_cover[rlabel]
                
                r_covers = to_cover.intersection(rule_covers_dict[r])
                
                g = len(label_covered.union(r_covers)) - len(label_covered)
                
                score = g
                
                if score > best_score:
                    best_rule = r
                    best_rule_label = rlabel
                    best_score = score
                    
        S.append(best_rule)
        covered_so_far[best_rule_label] = covered_so_far[best_rule_label].union(rule_covers_dict[best_rule])
        
    return S

'''
def greedy3(q, data_labels, rule_labels, rule_covers_dict):
    unique_labels = np.unique(rule_labels)
    points_to_cover = {l: set(np.where(data_labels == l)[0]) for l in unique_labels}
    covered_so_far = {l: set() for l in unique_labels}
    
    S = []
    rule_list = list(rule_covers_dict.keys())
    for i in range(q):
        best_rule = None
        best_rule_label = None
        best_score = -np.inf
        
        for r in rule_list:
            if r not in S:
                rlabel = rule_labels[r]
                label_covered = covered_so_far[rlabel]
                to_cover = points_to_cover[rlabel]
                
                r_covers = to_cover.intersection(rule_covers_dict[r])
                
                g = len(label_covered.union(r_covers)) - len(label_covered)
                c = len(rule_covers_dict[r]) - len(r_covers)
                
                score = g - c
                
                if score > best_score:
                    best_rule = r
                    best_rule_label = rlabel
                    best_score = score
                    
        S.append(best_rule)
        covered_so_far[best_rule_label] = covered_so_far[best_rule_label].union(rule_covers_dict[best_rule])
        
    return S
'''