import copy
from dataclasses import replace
import numpy as np
import pandas as pd
from typing import Any, List, Dict, Set, Tuple, Iterator
from numpy.typing import NDArray
from pyarc.data_structures import Consequent, Item, Antecedent, ClassAssocationRule
from .node import Node
from .rules import Condition, Rule, Decision, LinearCondition

from mdlp.discretization import MDLP


####################################################################################################


def tiebreak(scores : NDArray, proxy : NDArray = None) -> NDArray:
    """
    Breaks ties in a length m array of scores by:
    1) (IF given) Comparing values in a same-sized proxy array.
    2) Otherwise breaking ties randomly. 

    NOTE: By default preferences are taken in ascending order.
    
    Args:
        scores (np.ndarray): Length m array of scores.
        proxy (np.ndarray, optional): Length m array of proxy values to use for tiebreaking.
            Defaults to None.
            
    Returns:
        argsort (np.ndarray): Length m argsort of scores with ties broken.
    """
    m = len(scores)
    random_tiebreakers = np.random.rand(m)
    if proxy is not None:
        return np.lexsort((random_tiebreakers, proxy, scores))
    else:
        return np.lexsort((random_tiebreakers, scores))


####################################################################################################


def divide_with_zeros(x : NDArray, y : NDArray) -> NDArray:
    """
    Given two arrays, divide element-wise with the convention that 0/0 = 1 and 1/0 = infty. 

    Args:
        x (np.ndarray): Numerator array.
        y (np.ndarray): Denominator array.

    Returns:
        (np.ndarray): New array with element-wise divisions. 
    """
    assert x.shape == y.shape, "Input arrays do not match in size."

    ones_mask = np.zeros(x.shape, dtype = bool)
    ones_mask = (x == 0) & (y == 0)

    infty_mask = np.zeros(x.shape, dtype = bool)
    infty_mask = (x != 0) & (y == 0)

    xcopy = copy.deepcopy(x)
    ycopy = copy.deepcopy(y)
    xcopy[ones_mask] = 1
    ycopy[ones_mask] = 1
    xcopy[infty_mask] = np.inf
    ycopy[infty_mask] = 1

    return np.divide(xcopy, ycopy)


####################################################################################################


def covered_mask(assignment : np.ndarray) -> NDArray:
    """
    Finds a boolean array describing data coverage.
    
    Args:
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to class j and False (0) otherwise. 
        
    Returns:
        coverage (np.ndarray): Size n array with index i being true if point i
            is covered by at least one cluster, and false otherwise.
    """
    return np.sum(assignment, axis = 1) > 0


####################################################################################################


def update_centers(X : NDArray, current_centers : NDArray, assignment : NDArray) -> NDArray:
    """
    Given a dataset and a current assignment to cluster centers, update the centers by finding 
    the mean of the points assigned to each original center.
    
    Args:
        X (np.ndarray): Input (n x d) dataset.
        
        current_centers (np.ndarray): Current set of cluster centers represented as a (k x d) array.
        
        assignment (np.ndarray): Boolean assignment matrix of size (n x k). Entry (i,j) is 
            `True` if point i is assigned to cluster j and `False` otherwise.
            
    Returns:
        updated_centers (np.ndarray): Size (k x d) array of updated centers.
    """
    n,d = X.shape
    k,d_ = current_centers.shape
    n_,k_ = assignment.shape
    
    assert d == d_, f"Dimensionality of data {d} and cluster centers {d_} do not match."
    assert n == n_, f"Shape of data {n} does not match shape of shape of assignment {n_}."
    assert k == k_, f"Shape of current centers {k} doesn't match shape of shape of assignment {k_}."

    updated_centers = np.zeros((k,d))
    for i in range(k):
        assigned = np.where(assignment[:,i])[0]
        if len(assigned) > 0:
            new_center = np.mean(X[assigned,:], axis = 0)
        else:
            new_center = current_centers[i,:]
            
        updated_centers[i,:] = new_center
        
    return updated_centers


####################################################################################################


def labels_format(labels : NDArray) -> List[Set[int]]:
    """
    Takes a 1 dimensional array of labels and forms it into a 2d label list, which is the 
    default form used in this library.

    NOTE: By convention, a label of -1 indicates no label (empty set).
    
    Args:
        labels (np.ndarray): Length n array of labels.
    """
    return [{i} if i != -1 else set() for i in labels]


####################################################################################################


def can_flatten(labels : List[Set[int]]) -> bool:
    """
    Determines if a 2d list of labels can be flattened so as to be
    exactly represented by a 1 dimensional array.
    
    Args:
        labels (List[Set[int]]): 2d list of integers where the inner list at index i 
            labels of the item with index i.
            
    Returns:
        (bool): True if the labels can be flattened and False otherwise.
    """
    label_lengths = np.array([len(labs) for labs in labels])
    return np.all(label_lengths == 1)


####################################################################################################


def flatten_labels(labels : List[Set[int]]) -> NDArray:
    """
    Given a 2d labels list, returns a flattened list of labels. 

    NOTE: If an inner list is empty, the flattened label is represented as -1.
    
    Args:
        labels (List[Set[int]]): List with sets of integers where the inner set at index i 
            contains the labels of the item with index i.
            
    Returns:
        flattened (np.array): Flattened list of labels.
    """
    flattened_list = []
    for labs in labels:
        if labs:
            for j in labs:
                flattened_list.append(j)
        else:
            flattened_list.append(-1)

    flattened = np.array(flattened_list, dtype = np.int64)
    return flattened


####################################################################################################


def unique_labels(labels : List[Set[int]]) -> Set[int]:
    """
    Given a 2d labels list, returns a set of unique labels. 
    
    Args:
        labels (List[Set[int]]): 2d list of integers where the inner list at index i 
            labels of the item with index i.
            
    Returns:
        unique_labels (Set[int]): Set of unique labels.
    """
    unique = set()
    for labs in labels:
        unique.update(labs)
    return unique


####################################################################################################


def labels_to_assignment(
        labels : List[Set[int]],
        n_labels : int,
    ) -> NDArray:
    """
    Takes an input list of labels and returns its associated clustering matrix.
    NOTE: By convention, clusters are indexed [0...k-1] and items are indexed [0...n-1].
    This is how they should be labeled in the input label array.
    
    Args:
        labels (List[int] OR List[Set[int]]): List of integers where an entry at index i has value 
            j if the item associated with index i is given label j. Alternatively,
            in a soft labeling where points have multiple labels, labels[i] is a list of 
            cluster labels j.
            
        n_labels (int, optional): Total number of unique labels to create the assignment matrix 
            with. Helfpul for cases where points aren't assigned to any label (empty list).

    Returns:
        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to label j and False otherwise.
    """
    assignment_matrix = np.zeros((len(labels), n_labels), dtype = bool)
    for i,labs in enumerate(labels):
        for j in labs:
            assignment_matrix[i, j] = True
        
    return assignment_matrix


####################################################################################################


def assignment_to_labels(assignment : NDArray) -> List[Set[int]]:
    """
    Takes an input n x k boolean assignment matrix, and outputs a list of labels for the 
    datapoints.
     
    NOTE: By convention, clusters are indexed [0...k-1] and items are indexed [0...n-1].
    This is how they will be represented in the output label array.
    
    Args:
        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to label j and False otherwise.

    Returns:
        labels (List[Set[int]]): List of integers where an entry at index i has value 
            j if the item associated with index i is present within label j. Alternatively,
            in a soft labeling where points have multiple labels, labels[i] is a list of 
            labels j.
    """
    labels = []
    for _, assign in enumerate(assignment):
        l = np.where(assign)[0]
        if l.size > 0:
            labels.append(set(l))
        else:
            labels.append(set())
    return labels


####################################################################################################


def assignment_to_dict(
    assignment_matrix : NDArray
) -> Dict[int, Set[int]]:
    """
    Given a 2d labels list, returns a dictionary where the keys are labels,
    and the values are the sets of indices for the inner lists which contain the unique label.
    
    Args:
        assignment_matrix (np.ndarray): n x k boolean matrix with entry (i,j) being True
            if point i belongs to label j and False otherwise.
    
    Returns:
        assignment_dict (Dict[int, set[int]]): Dictionary where the keys are integers (labels) and 
            values are the sets of data point indices covered by the label.
    """        
    assignment_dict = {l: set() for l in range(assignment_matrix.shape[1])}
    for i in range(assignment_matrix.shape[1]):
        #assignment_dict[i] = set(np.where(assignment_matrix[:,i])[0])
        assignment_dict[i] = set(assignment_matrix[:,i].nonzero()[0]) 
    return assignment_dict


####################################################################################################


def traverse(node : Node, path : List[Node] = None) -> Iterator[List[Node]]:
    """
    Traverses a binary tree in a depth-first manner, yielding paths as as they are discovered.
    The function itself utilizes an iterative, yield from approach. For example, the following 
    command creates an iterator object over the set of all paths. Iterating through it 
    with a loop then prints all paths in a depth first manner:
    
    ```
    for path in traverse(root):
        print(path)
    ```
    
    
    Args:
        node (Node): Root node of the subtree to recurse into.
        
        path (List[Node], optional): List of node objects visited so far. Defaults to None 
            which starts a new traversal.
    
    Yields:
        path (List[Node]): List of node objects visited on the current path.
            If the path followed a left child, the corresponding string is 'left'.
            Otherwise, the string is 'right'.
    """
    if path is None:
        path = []
    
    # Yield the path up to and including the current node.
    path_update = path + [node]
    yield path_update
    
    # Yield paths with children
    if node.left_child is not None:
        left_condition_node = copy.deepcopy(node)
        left_condition_node.condition = replace(left_condition_node.condition, direction = -1)
        left_path = path + [left_condition_node]
        yield from traverse(node.left_child, left_path)
        
    # NOTE: This creates a copy of the nodes added to the path, 
    # and depending on the direction taken in the tree, switches the node's logical condition to 
    # the correct direction. By default tree nodes will have direction -1 (<= condition), which is 
    # intended to move left if True, so switching is especially helpful for paths moving right.
    if node.right_child is not None:
        right_condition_node = copy.deepcopy(node)
        right_condition_node.condition = replace(right_condition_node.condition, direction=1)
        right_path = path + [right_condition_node]
        yield from traverse(node.right_child, right_path)
        

####################################################################################################


def collect_nodes(root : Node) -> List[Node]:
    """
    Given the root of a tree, finds all nodes in the tree.
    
    Args:
        root (Node): Root of the tree.
    
    Returns:
        nodes (List[Node]): List of nodes in the tree. 
    """
    
    nodes = []
    for path in traverse(root):
        last_node = path[-1]
        nodes.append(last_node)
            
    return nodes


####################################################################################################


def collect_node_rules(root : Node) -> List[Rule]:
    """
    Given the root, finds all sub-rules in the tree.
    
    Args:
        root (Node): Root of the tree.
    
    Returns:
        nodes (List[Rule]): List of rules in the tree.
    """
    
    rules = []
    for path in traverse(root):
        last_node = path[-1]
        if last_node.type != 'leaf':
            conditions = [node.condition for node in path if node.type != 'leaf']
            rule = Rule(conditions)
            rules.append(rule)

            # Also include the rule with the last condition flipped
            last_condition_flipped = replace(conditions[-1], direction=-1 * conditions[-1].direction)
            rules.append(Rule(conditions[:-1] + [last_condition_flipped]))

    return rules


####################################################################################################


def collect_leaves(root : Node) -> List[Node]:
    """
    Given the root of a tree, finds all leaf nodes in the tree.
    
    Args:
        root (Node): Root of the tree.
    
    Returns:
        leaves (List[Node]): List of leaf nodes in the tree. 
    """
    
    leaves = []
    for path in traverse(root):
        last_node = path[-1]
        if last_node.type == 'leaf':
            leaves.append(last_node)
            
    return leaves


####################################################################################################


def collect_leaf_rules(root : Node) -> List[Rule]:
    """
    Given the root of a tree, finds all leaf nodes in the tree.
    
    Args:
        root (Node): Root of the tree.
    Returns:
        leaf_rules (List[Rule]): List of leaf rules in the tree. 
    """

    leaf_rules = []
    for path in traverse(root):
        last_node = path[-1]
        if last_node.type == 'leaf':
            conditions = [node.condition for node in path if node.type != 'leaf']
            rule = Rule(conditions)
            leaf_rules.append(rule)

    return leaf_rules


####################################################################################################


def get_decision_paths(
    root : Node
) -> List[List[Node]]:
    """
    Given the root of a tree, finds all decision paths 
    used to reach leaf nodes in the tree. Optionally, this takes an array y of training data labels
    AND an array of specific labels to look for. In that case, only paths with leaf nodes
    which have a majority of a label within the labels array are returned.
    
    Args:
        root (Node): Root of the tree.
    Returns:
        paths (List[List[Node]]): List of decision paths in the tree, where each decision path 
            is represented as a list of Node objects. 
    """
    paths = []
    for path in traverse(root):
        last_node = path[-1]
        if last_node.type == 'leaf':
            paths.append(path)
            
    return paths


####################################################################################################


def get_decision_paths_with_labels(
    root : Node,
    #labels : List[Set[int]],
    select_labels : NDArray,
) -> Tuple[List[List[Node]], List[Set[int]]]:
    """
    Given the root of a tree, finds all decision paths 
    used to reach leaf nodes in the tree. Optionally, this takes an input set of data labels
    AND an array of selected labels to look for. In that case, whenever a leaf 
    node is found, consider the data points associated with it. If there is a majority 
    for a selected label, keep that path. Otherwise discard it.
    
    NOTE: Perhaps this should take the trained node's class label instead...

    Args:
        root (Node): Root of the tree.
    
        labels (List[Set[int]]): Training Data labels.
            
        select_labels (np.ndarray): Labels to filter by.
    Returns:
        paths (List[List[Node]]): List of decision paths in the tree, where each decision path 
            is represented as a list of Node objects. 
        path_labels (List[Set[int]]): List of labels corresponding to each path.
    """
    paths = []
    path_labels = []
    for path in traverse(root):
        last_node = path[-1]
        if last_node.type == 'leaf' and len(last_node.indices) > 0:
            if last_node.label in select_labels:
                paths.append(path)
                path_labels.append({last_node.label})
            
    return paths, path_labels


####################################################################################################


def get_depth(root : Node) -> int:
    """
    Given the root of a tree, finds the maximum depth of the tree.
    
    Args:
        root (Node): Root of the tree.
    
    Returns:
        depth (int): Maximum depth of the tree. 
    """
    depths = []
    for path in traverse(root):
        depths.append(len(path) - 1)    
    return max(depths)


####################################################################################################


def satisfies_rule(X : NDArray, rule : Rule) -> NDArray:
    """
    Given a dataset X and a rule, determines 
    which data indices satisfy the rule.
    
    Args:
        X (np.ndarray): Dataset to evaluate.
        rule (Rule): Rule to evaluate with.

    Returns:
        (np.ndarray): Integer array of data indices satisfying the decision path. 
    """        
    return np.where(rule.evaluate(X))[0]


####################################################################################################
        

def satisfies_path(X : NDArray, path : List) -> NDArray:
    """
    Given a dataset X and a decision path, determines 
    which data indices satisfy the path.
    
    Args:
        X (np.ndarray): Dataset to evaluate.
        
        path (List[(Node, str)]): Decision path to evaluate.
    
    Returns:
        (np.ndarray): Integer array of data indices satisfying the decision path. 
    """
    condition_list = [node.condition for node in path[:-1]]
    rule = Rule(condition_list)
    return satisfies_rule(X, rule)


####################################################################################################


def quantile_bin(
        X : NDArray,
        n_bins : int
    ) -> pd.DataFrame:
    """
    Bins each feature of a real valued dataset into quantile-based buckets.
    
    Args:
        X (np.ndarray): Input (n x d) dataset.
        
        n_bins (int): Number of bins to use for each feature.
        
    Returns:
        binned_X (pd.DataFrame): Binned version of the input dataset, where bins are represented by 
            pandas Interval objects (start, stop].
    """
    df = pd.DataFrame(X)
    bin_df = df.apply(pd.qcut, args = (n_bins,), axis = 0, duplicates = 'drop')
    return bin_df


####################################################################################################


def uniform_bin(
        X : NDArray,
        n_bins : int
    ) -> pd.DataFrame:
    """
    Bins each feature of a real valued dataset into uniform-width buckets.
    
    Args:
        X (np.ndarray): Input (n x d) dataset.
        
        n_bins (int): Number of bins to use for each feature.
        
    Returns:
        binned_X (pd.DataFrame): Binned version of the input dataset, where bins are represented by 
            pandas Interval objects (start, stop].
    """
    df = pd.DataFrame(X)
    bin_df = df.apply(pd.cut, args = (n_bins,), axis = 0, duplicates = 'drop')
    return bin_df


####################################################################################################


def entropy_bin(
        X : NDArray,
        y : List[Set[int]],
        random_state : int = None
    ) -> pd.DataFrame:
    """
    Bins each feature of a real valued dataset to minimize the entropy of the resulting 
    binned dataset.

    This function makes use of the Minimum Description Length Principle (MDLP) 
    python implementation: https://github.com/hlin117/mdlp-discretization?tab=readme-ov-file

    Based upon the following work:
    Fayyad, U. M., & Irani, K. B. (1993). 
    Multi-interval discretization of continuous-valued attributes for classification learning.
    
    Args:
        X (np.ndarray): Input (n x d) dataset.
        
        y (List[Set[int]]): Input list of labels associated with each data point.
            NOTE: Each data point must have exactly one label.

        random_state (int, optional): Seed used by the random number generator.
            Defaults to None.
            
    Returns:
        binned_X (pd.DataFrame): Binned version of the input dataset, where bins are represented by 
            pandas Interval objects (start, stop].
    """
    if not can_flatten(y):
        raise ValueError("Each data point must be assigned to a single label.")
        
    y_ = flatten_labels(y)
    discretizer = MDLP(random_state = random_state)
    data_disc = discretizer.fit_transform(X, y_ + 1)  # MDLP does not accept negative labels
    interval_data = {}
    for i, col in enumerate(data_disc.T):
        cut_points = discretizer.cut_points_[i]
        cut_points = np.concatenate(([-np.inf], cut_points, [np.inf]))
        intervals = pd.IntervalIndex.from_breaks(cut_points)
        
        interval_list = []
        for val in col:
            interval_list.append(intervals[val])
        
        interval_data[i] = interval_list

    bin_df = pd.DataFrame(interval_data)
    return bin_df


####################################################################################################


def interval_to_condition(feature : Any, interval : str) -> Tuple[Condition, Condition]:
    """
    Convert an interval string to a Condition object.

    Args:
        interval (str): A string representing an interval, e.g., '(-3.151, -0.701]'.

    Returns:
        Condition: A Condition object representing the interval.
    """
    interval = interval.split(',')

    # Lower bound:
    lower_type = interval[0][0]
    lower_bound = float(interval[0].strip('()[]'))

    # Upper bound:
    upper_type = interval[1][-1]
    upper_bound = float(interval[1].strip('()[]'))

    if lower_type == '(':
        lower_condition = LinearCondition(
            features = [feature],
            weights = [1.0],
            threshold = lower_bound,
            direction = 1
        )
    else:
        raise ValueError(f"Unsupported lower bound type: {lower_type}")
    
    if upper_type == ']':
        upper_condition = LinearCondition(
            features = [feature],
            weights = [1.0],
            threshold = upper_bound,
            direction = -1
        )
    else:
        raise ValueError(f"Unsupported upper bound type: {upper_type}")
    
    return lower_condition, upper_condition


#####################################################################################################


def decision_set_to_cars(
    X: np.ndarray,
    y: list[set[int]],
    decision_set: list[Decision]
) -> list[ClassAssocationRule]:   
    """
    Convert a decision set into a list of class association rules.
    Class association rules are represented using the `pyarc` library.

    Args:
        X (np.ndarray): Input data array.
        y (list[set[int]]): List of label sets for each instance.
            Each label set should contain only a single label.
        decision_set (list[Decision]): Decision set represented as a list of decisions.

    Returns: 
        cars (list[ClassAssocationRule]): List of class association rules.
    """
    if not can_flatten(y):
        raise ValueError("Each label in y must be a single label set.")

    for decision in decision_set:
        for condition in decision.rule.conditions:
            if len(condition.features) != 1:
                raise ValueError("Each condition must have a single feature.")
            
    cars = []
    for i, decision in enumerate(decision_set):
        consequent = f"class:=:{decision.label}"
        antecedent_dict = {}
        support_bool = decision.rule.evaluate(X)

        for condition in decision.rule.conditions:
            feature = condition.features[0]
            if feature not in antecedent_dict:
                antecedent_dict[feature] = [-np.inf, np.inf]
            
            direction = condition.direction
            threshold = condition.threshold

            if direction == -1:
                antecedent_dict[feature][1] = threshold
            elif direction == 1:
                antecedent_dict[feature][0] = threshold
            else:
                raise ValueError("Condition direction must be -1 or 1.")
            
        antecedent = [
            f"{feature}:=:({interval[0]}, {interval[1]}]"
            for feature, interval in antecedent_dict.items()
        ]
        antecedent = sorted(list(antecedent))
        antecedent_items = [Item(*i.split(":=:")) for i in antecedent]

        support = np.sum(support_bool)/X.shape[0]
        confidence = 0
        for idx in np.where(support_bool)[0]:
            if y[idx] == {decision.label}:
                confidence += 1
        confidence /= np.sum(support_bool)

        car = ClassAssocationRule(
            Antecedent(antecedent_items),
            Consequent(*consequent.split(":=:")),
            support = support,
            confidence = confidence
        )
        cars.append(car)
        
    return cars


#####################################################################################################


def cars_to_decision_set(
    cars: list[ClassAssocationRule]
) -> List[Decision]:
    """
    Convert a list of class association rules into a decision set.

    Args:
        cars (list[ClassAssocationRule]): List of class association rules.
    Returns:
        decision_set (list[Decision]): Decision set represented as a list of decisions.
    """
    decision_set = []
    for car in cars:
        conditions = []
        for item in car.antecedent:
            feature = int(item[0])
            interval = item[1][:-1] + ']'
            condition = interval_to_condition(feature, interval)
            conditions.append(condition)
    
        rule = Rule(conditions)
        label = int(car.consequent.value)
        decision = Decision(rule, label)
        decision_set.append(decision)
    return decision_set


####################################################################################################


def filter_rules(
    rules : List[Rule],
    X : NDArray,
    y : List[Set[int]],
    confidence : float = 0.5
) -> List[Rule]:
    """
    Filters a list of rules to only include those with a minimum 
    level of confidence for the labels of the covered points.

    Args:
        rules (List[Rule]): List of rules to filter.
        
        X (np.ndarray): Input data array.
        
        y (List[Set[int]]): List of label sets for each instance.
            Each label set should contain only a single label.

        confidence (float, optional): Minimum confidence threshold for a rule to be kept.
            Defaults to 0.5, in which case rules must have at least 50% confidence and be 
            simple majority rules. 

    Returns:
        filtered_rules (List[List[Condition]]): Filtered list of rules.
    """
    if not can_flatten(y):
        raise ValueError("Each label in y must be a single label set.")
    y_ = flatten_labels(y)

    filtered_rules = []
    for rule in rules:
        covered_indices = satisfies_rule(X, rule)
        covered_labels = y_[covered_indices]
        labs, counts = np.unique(covered_labels, return_counts=True)
        if len(counts) == 0:
            continue
        if np.max(counts) / len(covered_indices) >= confidence:
            filtered_rules.append(rule)

    return filtered_rules


####################################################################################################


def map_rules_to_decisions(
    decision_set: list[Decision]
) -> dict[Rule, Set[Decision]]:
    """
    Given a decision set, returns a dictionary mapping unique rules to their associated decisions.
    Args:
        decision_set (list[Decision]): Decision set represented as a list of decisions.
    Returns:
        rules_to_decisions_dict (dict[Rule, Set[Decision]]): Dictionary mapping unique rules to their associated decisions.
    """
    rules_to_decisions_dict = {}
    for decision in decision_set:
        rule = decision.rule
        if rule not in rules_to_decisions_dict:
            rules_to_decisions_dict[rule] = set()
        rules_to_decisions_dict[rule].add(decision)
    return rules_to_decisions_dict


####################################################################################################


def compute_elbow(x: np.ndarray, y: np.ndarray, increasing: bool = True) -> int:
    """
    Compute the elbow as the point farthest from the end-to-end line.

    Note:
    The distance used is the perpendicular distance to the *infinite* line
    through (x[0], y[0]) and (x[-1], y[-1]) (not to the segment). This matches
    the typical "elbow" heuristic.

    If the first and last points are identical, distances are computed to that
    point instead (Euclidean distance), and the farthest point is returned.

    Args:
        x, y: 1D arrays of equal length describing the curve points (x[i], y[i]).
        increasing: If True, choose the farthest point among those *above* the
            end-to-end line. If False, choose the farthest point among those
            *below* the line.

    Returns:
        int: The index of the elbow point.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        raise ValueError("Need at least two points to compute an elbow")

    # Exclude invalid points: any index where x or y is NaN must never be returned.
    # (np.isfinite also filters +/-inf, which likewise can't yield a useful distance.)
    valid_mask = np.isfinite(x) & np.isfinite(y)
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size < 2:
        raise ValueError("Need at least two finite (non-NaN/inf) points to compute an elbow")

    # Use the first/last *valid* points as the endpoints of the reference line.
    i1 = int(valid_idx[0])
    i2 = int(valid_idx[-1])

    x1, y1 = x[i1], y[i1]
    x2, y2 = x[i2], y[i2]

    dx = x2 - x1
    dy = y2 - y1

    # Degenerate case: end points are identical.
    if dx == 0.0 and dy == 0.0:
        # Signed distance is arbitrary here; treat all finite points as eligible.
        signed = np.zeros_like(x, dtype=float)
        distances = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    else:
        # Oriented line equation value (numerator of point-to-line distance).
        # Positive/negative indicates which side of the line ("above"/"below")
        # relative to the direction (x1,y1)->(x2,y2).
        signed = dy * x - dx * y + (x2 * y1 - y2 * x1)
        denom = np.sqrt(dx * dx + dy * dy)
        distances = np.abs(signed) / denom

    # Invalidate any non-finite points so they are not considered solutions.
    distances = np.where(valid_mask, distances, -np.inf)

    # Keep only points on the requested side.
    # Note: points exactly on the line (signed==0) are excluded.
    side_mask = (signed > 0) if increasing else (signed < 0)
    mask = valid_mask & side_mask

    if np.any(mask):
        masked = np.where(mask, distances, -np.inf)
        idx = int(np.argmax(masked))
    else:
        # Fallback: if all finite points are on the other side (or collinear),
        # return the unconstrained maximum among finite points.
        idx = int(np.argmax(distances))

    return idx


####################################################################################################


def _pack_bool_matrix(mat: np.ndarray) -> np.ndarray:
    """Pack a 2D boolean matrix along the last axis using np.packbits."""
    if mat.dtype != np.bool_:
        mat = mat.astype(np.bool_, copy=False)
    return np.packbits(mat, axis=-1)


def _unpack_bool_matrix(packed: np.ndarray, n_bits: int) -> np.ndarray:
    """Unpack a packed-bits 2D matrix back to boolean with a known original width."""
    out = np.unpackbits(packed, axis=-1)
    return out[..., :n_bits].astype(np.bool_, copy=False)


####################################################################################################