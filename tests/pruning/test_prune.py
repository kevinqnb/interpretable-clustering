import numpy as np 
from intercluster.pruning import *
from intercluster.utils import assignment_to_labels


def test_distorted_greedy():
    # First: Easy test, should find all the data points
    n_rules = 3
    lambda_val = 0

    data_to_cluster_assignment = np.array([
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [0,1,0],
        [0,1,0],
        [0,1,0],
        [0,0,1],
        [0,0,1],
        [0,0,1]
    ])

    rule_to_cluster_assignment = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [0,1,0]
    ])

    data_to_rules_assignment = np.array([
        [1,0,0,1],
        [1,0,0,0],
        [1,0,0,0],
        [0,1,0,0],
        [0,1,0,1],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,1,1],
        [0,0,1,0]
    ])

    selected = distorted_greedy(
        n_rules = n_rules,
        lambda_val=lambda_val,
        data_to_cluster_assignment=data_to_cluster_assignment,
        rule_to_cluster_assignment=rule_to_cluster_assignment,
        data_to_rules_assignment=data_to_rules_assignment
    )

    assert np.array_equal(selected, np.array([0,1,2]))

    # Next: Make sure lambda penalization is working:
    # introduce some overlap
    data_to_rules_assignment = np.array([
        [1,1,1,1],
        [1,1,0,1],
        [1,0,1,1],
        [0,1,1,1],
        [0,1,0,1],
        [1,1,0,1],
        [1,0,1,1],
        [0,0,1,1],
        [1,1,1,1]
    ])

    # This lambda value is just small enough to where the contribution of the first 3 rules 
    # should still be positive in all iterations of distorted greedy (<2/3 ^ 2 ~= 0.44444)
    lambda_val = 0.43

    selected = distorted_greedy(
        n_rules = n_rules,
        lambda_val=lambda_val,
        data_to_cluster_assignment=data_to_cluster_assignment,
        rule_to_cluster_assignment=rule_to_cluster_assignment,
        data_to_rules_assignment=data_to_rules_assignment
    )

    assert np.array_equal(selected, np.array([0,1,2]))

    # Now only two rules should be selected:
    lambda_val = 0.45

    selected = distorted_greedy(
        n_rules = n_rules,
        lambda_val=lambda_val,
        data_to_cluster_assignment=data_to_cluster_assignment,
        rule_to_cluster_assignment=rule_to_cluster_assignment,
        data_to_rules_assignment=data_to_rules_assignment
    )

    assert np.array_equal(selected, np.array([0,1]))

    # Now only 1:
    lambda_val = 0.67

    selected = distorted_greedy(
        n_rules = n_rules,
        lambda_val=lambda_val,
        data_to_cluster_assignment=data_to_cluster_assignment,
        rule_to_cluster_assignment=rule_to_cluster_assignment,
        data_to_rules_assignment=data_to_rules_assignment
    )

    assert np.array_equal(selected, np.array([0]))

    # Now nothing should be selected!:
    lambda_val = 1

    selected = distorted_greedy(
        n_rules = n_rules,
        lambda_val=lambda_val,
        data_to_cluster_assignment=data_to_cluster_assignment,
        rule_to_cluster_assignment=rule_to_cluster_assignment,
        data_to_rules_assignment=data_to_rules_assignment
    )

    assert np.array_equal(selected, np.array([]))



def test_prune_with_grid_search():
    n_rules = 3

    data = np.array([
        [10,0],
        [10,0],
        [10,0],
        [0,1],
        [0,1],
        [0,1],
        [-10,0],
        [-10,0],
        [-10,0]
    ])

    centers = np.array([
        [5,0],
        [0,0],
        [-5,0]
    ])

    data_to_cluster_assignment = np.array([
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [0,1,0],
        [0,1,0],
        [0,1,0],
        [0,0,1],
        [0,0,1],
        [0,0,1]
    ])

    data_labels = assignment_to_labels(data_to_cluster_assignment)

    rule_to_cluster_assignment = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [0,1,0]
    ])

    rule_labels = assignment_to_labels(rule_to_cluster_assignment)

    data_to_rules_assignment = np.array([
        [1,1,1,1],
        [1,1,0,1],
        [1,0,1,1],
        [0,1,1,1],
        [0,1,0,1],
        [1,1,0,1],
        [1,0,1,1],
        [0,0,1,1],
        [1,1,1,1]
    ])

    lambda_search_range = np.array([0, 0.43, 0.5, 0.67, 1])
    prune_objective = KmeansObjective(
        X = data,
        centers = centers,
        average = False,
        normalize = False
    )

    # NOTE that this should only select one rule, 
    frac_cover = 2/3
    selected, obj_vals, cov_vals = prune_with_grid_search(
        n_rules=n_rules,
        frac_cover=frac_cover,
        n_clusters=3,
        data_labels=data_labels,
        rule_labels=rule_labels,
        data_to_rules_assignment=data_to_rules_assignment,
        objective=prune_objective,
        lambda_search_range=lambda_search_range,
        return_full = True
    )

    assert np.array_equal(selected, np.array([0]))

    frac_cover = 0.67
    selected, obj_vals, cov_vals = prune_with_grid_search(
        n_rules=n_rules,
        frac_cover=frac_cover,
        n_clusters=3,
        data_labels=data_labels,
        rule_labels=rule_labels,
        data_to_rules_assignment=data_to_rules_assignment,
        objective=prune_objective,
        lambda_search_range=lambda_search_range,
        return_full = True
    )

    assert np.array_equal(selected, np.array([0,1]))


    frac_cover = 1
    selected, obj_vals, cov_vals = prune_with_grid_search(
        n_rules=n_rules,
        frac_cover=frac_cover,
        n_clusters=3,
        data_labels=data_labels,
        rule_labels=rule_labels,
        data_to_rules_assignment=data_to_rules_assignment,
        objective=prune_objective,
        lambda_search_range=lambda_search_range,
        return_full = True
    )

    assert np.array_equal(selected, np.array([0,1,2]))


    # Tests a case where the required coverage cannot be met:
    frac_cover = 1
    data_to_rules_assignment = np.array([
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,0],
        [0,0,1,0],
        [1,0,0,0],
        [0,0,0,1],
        [0,1,0,0],
        [0,0,0,1]
    ])

    selected, obj_vals, cov_vals = prune_with_grid_search(
        n_rules=n_rules,
        frac_cover=frac_cover,
        n_clusters=3,
        data_labels=data_labels,
        rule_labels=rule_labels,
        data_to_rules_assignment=data_to_rules_assignment,
        objective=prune_objective,
        lambda_search_range=lambda_search_range,
        return_full = True
    )

    assert selected is None
