import numpy as np
from intercluster.rules import Node
from intercluster.rules.utils import *

####################################################################################################
# Custom Tree:
leaf1 = Node()
leaf1.leaf_node(
    label = 1,
    cost = 0,
    indices = np.array([]),
    depth = 2
)
leaf2 = Node()
leaf2.leaf_node(
    label = 2,
    cost = 0,
    indices = np.array([]),
    depth = 3
)
leaf3 = Node()
leaf3.leaf_node(
    label = 3,
    cost = 0,
    indices = np.array([]),
    depth = 3
)
leaf4 = Node()
leaf4.leaf_node(
    label = 4,
    cost = 0,
    indices = np.array([]),
    depth = 1
)

internal_node1 = Node()
internal_node1.tree_node(
    left_child = leaf2,
    right_child = leaf3,
    features = np.array([1]),
    weights = np.array([1]),
    threshold = 2,
    cost = 0,
    indices = np.array([]),
    depth = 2,
    feature_labels = []
)

internal_node2 = Node()
internal_node2.tree_node(
    left_child = leaf1,
    right_child = internal_node1,
    features = np.array([0]),
    weights = np.array([1]),
    threshold = 0,
    cost = 0,
    indices = np.array([]),
    depth = 1,
    feature_labels = []
)

root_node = Node()
root_node.tree_node(
    left_child = internal_node2,
    right_child = leaf4,
    features = np.array([0]),
    weights = np.array([1]),
    threshold = 1,
    cost = 0,
    indices = np.array([]),
    depth = 0,
    feature_labels = []
)

####################################################################################################


def test_traverse():
    for i,path in enumerate(traverse(root_node, [])):
        if i == 0:
            assert path[0][0] == root_node
        else:
            assert path[0][0] == root_node
            if i == 1:
                assert path[1][0] == internal_node2
            if i == 2:
                assert path[1][0] == internal_node2
                assert path[2][0] == leaf1
            if i == 3:
                assert path[1][0] == internal_node2
                assert path[2][0] == internal_node1
            if i == 4:
                assert path[1][0] == internal_node2
                assert path[2][0] == internal_node1
                assert path[3][0] == leaf2
            if i == 5:
                assert path[1][0] == internal_node2
                assert path[2][0] == internal_node1
                assert path[3][0] == leaf3
            if i == 6:
                assert path[1][0] == leaf4
                
                
def test_collect_nodes():
    nodes = collect_nodes(root_node)
    assert len(nodes) == 7
    assert root_node in nodes
    assert internal_node1 in nodes
    assert internal_node2 in nodes
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
    assert paths[0] == [(root_node, 'left'), (internal_node2, 'left'), (leaf1, None)]
    assert paths[1] == [
        (root_node, 'left'),
        (internal_node2, 'right'),
        (internal_node1, 'left'),
        (leaf2, None)
    ]
    assert paths[2] == [
        (root_node, 'left'),
        (internal_node2, 'right'),
        (internal_node1, 'right'),
        (leaf3, None)
    ]
    assert paths[3] == [(root_node, 'right'), (leaf4, None)]
    
    
def test_get_depth():
    assert get_depth(root_node) == 3
    
    
def test_satisfies_path(satisfies_dataset):
    X = satisfies_dataset
    paths = get_decision_paths(root_node)
    assert np.array_equal(satisfies_path(X, paths[0]), np.array([0, 1]))
    assert np.array_equal(satisfies_path(X, paths[1]), np.array([2, 3]))
    assert np.array_equal(satisfies_path(X, paths[2]), np.array([4, 5]))
    assert np.array_equal(satisfies_path(X, paths[3]), np.array([6, 7]))
    
    
    