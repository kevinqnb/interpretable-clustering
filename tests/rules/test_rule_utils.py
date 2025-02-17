import numpy as np
from intercluster.rules import LinearCondition, Node
from intercluster.rules.utils import *

####################################################################################################
# Custom Tree:
leaf1 = Node()
leaf1.leaf_node(
    label = 1,
    cost = 0,
    indices = np.array([0,1,2]),
    depth = 2
)
leaf2 = Node()
leaf2.leaf_node(
    label = 2,
    cost = 0,
    indices = np.array([3,4,5]),
    depth = 3
)
leaf3 = Node()
leaf3.leaf_node(
    label = 3,
    cost = 0,
    indices = np.array([6,7,8]),
    depth = 3
)
leaf4 = Node()
leaf4.leaf_node(
    label = 4,
    cost = 0,
    indices = np.array([9,10,11]),
    depth = 1
)

condition1 = LinearCondition(
    features = np.array([1]),
    weights = np.array([1]),
    threshold = 2,
    direction = -1
)
cost1 = 1.215
internal_node1 = Node()
internal_node1.tree_node(
    left_child = leaf2,
    right_child = leaf3,
    condition = condition1,
    cost = cost1,
    indices = np.array([]),
    depth = 2,
    feature_labels = []
)

condition2 = LinearCondition(
    features = np.array([0]),
    weights = np.array([1]),
    threshold = 0,
    direction = -1
)
cost2 = 2.326
internal_node2 = Node()
internal_node2.tree_node(
    left_child = leaf1,
    right_child = internal_node1,
    condition = condition2,
    cost = cost2,
    indices = np.array([]),
    depth = 1,
    feature_labels = []
)

root_condition = LinearCondition(
    features = np.array([0]),
    weights = np.array([1]),
    threshold = 1,
    direction = -1
)
root_cost = 3.217
root_node = Node()
root_node.tree_node(
    left_child = internal_node2,
    right_child = leaf4,
    condition = root_condition,
    cost = root_cost,
    indices = np.array([]),
    depth = 0,
    feature_labels = []
)

####################################################################################################


def test_traverse():
    # NOTE: I am using cost as an identifier for some of these nodes, since they are copies 
    #   when returned from the traversal.
    for i,path in enumerate(traverse(root_node, [])):
        if i == 0:
            assert path[0].cost == root_node.cost
        else:
            assert path[0].cost == root_node.cost
            if i == 1:
                assert path[1].cost == internal_node2.cost
            if i == 2:
                assert path[1].cost == internal_node2.cost
                assert path[2] == leaf1
            if i == 3:
                assert path[1].cost == internal_node2.cost
                assert path[2].cost == internal_node1.cost
            if i == 4:
                assert path[1].cost == internal_node2.cost
                assert path[2].cost == internal_node1.cost
                assert path[3] == leaf2
            if i == 5:
                assert path[1].cost == internal_node2.cost
                assert path[2].cost == internal_node1.cost
                assert path[3] == leaf3
            if i == 6:
                assert path[1] == leaf4
                
                
def test_collect_nodes():
    nodes = collect_nodes(root_node)
    assert len(nodes) == 7
    assert root_node in nodes
    assert leaf1 in nodes
    assert leaf2 in nodes
    assert leaf3 in nodes
    assert leaf4 in nodes
    

def test_collect_leaves():
    leaves = collect_leaves(root_node)
    assert len(leaves) == 4
    assert leaf1 in leaves
    assert leaf2 in leaves
    assert leaf3 in leaves
    assert leaf4 in leaves
    
    
def test_get_decision_paths():
    paths = get_decision_paths(root_node)
    assert len(paths) == 4
    
    assert len(paths[0]) == 3
    assert paths[0][0].cost == root_node.cost
    assert paths[0][1].cost == internal_node2.cost
    assert paths[0][2] == leaf1
    
    assert len(paths[1]) == 4
    assert paths[1][0].cost == root_node.cost
    assert paths[1][1].cost == internal_node2.cost
    assert paths[1][2].cost == internal_node1.cost
    assert paths[1][3] == leaf2
    
    assert len(paths[2]) == 4
    assert paths[2][0].cost == root_node.cost
    assert paths[2][1].cost == internal_node2.cost
    assert paths[2][2].cost == internal_node1.cost
    assert paths[2][3] == leaf3
    
    assert len(paths[3]) == 2
    assert paths[3][0].cost == root_node.cost
    assert paths[3][1] == leaf4
    
    
def test_get_decision_paths_with_labels():
    labels = [{0}, {0}, {0}, {1}, {1}, {1}, {2}, {2}, {2}, {3}, {3}, {3}]
    full_paths = get_decision_paths(root_node)
    labeled_paths, path_labels = get_decision_paths_with_labels(
        root_node,
        labels = labels,
        select_labels = [0,3]
    )
    assert len(labeled_paths) == 2
    
    assert len(labeled_paths[0]) == 3
    assert labeled_paths[0][0].cost == root_node.cost
    assert labeled_paths[0][1].cost == internal_node2.cost
    assert labeled_paths[0][2] == leaf1
    
    assert len(labeled_paths[1]) == 2
    assert labeled_paths[1][0].cost == root_node.cost
    assert labeled_paths[1][1] == leaf4
    
    
    labeled_paths, path_labels = get_decision_paths_with_labels(
        root_node,
        labels = labels,
        select_labels = np.array([4])
    )
    assert len(labeled_paths) == 0
    
    
    labels = [{0,1,2}, {0}, {0}, {1}, {1}, set(), {2,1}, {2}, {2}, set(), {3}, {3}]
    labeled_paths, path_labels = get_decision_paths_with_labels(
        root_node,
        labels = labels,
        select_labels = np.array([1,2])
    )
    assert len(labeled_paths) == 2
    assert len(labeled_paths[0]) == 4
    assert labeled_paths[0][0].cost == root_node.cost
    assert labeled_paths[0][1].cost == internal_node2.cost
    assert labeled_paths[0][2].cost == internal_node1.cost
    assert labeled_paths[0][3] == leaf2
    
    assert len(labeled_paths[1]) == 4
    assert labeled_paths[1][0].cost == root_node.cost
    assert labeled_paths[1][1].cost == internal_node2.cost
    assert labeled_paths[1][2].cost == internal_node1.cost
    assert labeled_paths[1][3] == leaf3
    
    
def test_get_depth():
    assert get_depth(root_node) == 3
    
    
def test_satisfies_path(satisfies_dataset):
    X = satisfies_dataset
    paths = get_decision_paths(root_node)
    assert np.array_equal(satisfies_path(X, paths[0]), np.array([0, 1]))
    assert np.array_equal(satisfies_path(X, paths[1]), np.array([2, 3]))
    assert np.array_equal(satisfies_path(X, paths[2]), np.array([4, 5]))
    assert np.array_equal(satisfies_path(X, paths[3]), np.array([6, 7]))
    
    
def test_satisfies_conditions():
    cond1 = LinearCondition(
        features = np.array([1]),
        weights = np.array([1]),
        threshold = 2,
        direction = -1
    )
    cond2 = LinearCondition(
        features = np.array([0,1]),
        weights = np.array([1/2, 3]),
        threshold = 5,
        direction = 1
    )
    cond_list = [cond1, cond2]
    
    X = np.array([
        [0, 3],
        [np.inf, 0],
        [1,1],
        [3,4]
    ])
    
    assert np.array_equal(satisfies_conditions(X, cond_list), np.array([1]))
    
    
    
    