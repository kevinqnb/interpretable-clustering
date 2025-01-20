import numpy as np
from sklearn.tree import DecisionTreeClassifier
from intercluster.rules import ID3Tree, collect_leaves, get_depth


def test_max_leaf_nodes(example_dataset):
    X, y = example_dataset
    tree = ID3Tree(max_leaf_nodes = 10)
    tree.fit(X, y)
    labels = tree.predict(X)
    
    sklearn_tree = DecisionTreeClassifier(
        criterion='entropy',
        max_leaf_nodes = 10
    )
    sklearn_tree.fit(X, y)
    sklearn_labels = sklearn_tree.predict(X)
    
    leaves = collect_leaves(tree.root)
    assert len(leaves) == 10
    assert np.array_equal(labels, sklearn_labels)
    
    
def test_max_depth(example_dataset):
    X, y = example_dataset
    tree = ID3Tree(max_depth = 4)
    tree.fit(X, y)
    labels = tree.predict(X)
    
    sklearn_tree = DecisionTreeClassifier(
        criterion='entropy',
        max_depth = 4
    )
    sklearn_tree.fit(X, y)
    sklearn_labels = sklearn_tree.predict(X)
    
    assert get_depth(tree.root) == 4
    assert np.array_equal(labels, sklearn_labels)
    
    
def test_max_leaf_nodes_and_depth(example_dataset):
    X, y = example_dataset
    tree = ID3Tree(max_leaf_nodes = 10, max_depth = 4)
    tree.fit(X, y)
    labels = tree.predict(X)
    
    sklearn_tree = DecisionTreeClassifier(
        criterion='entropy',
        max_leaf_nodes = 10,
        max_depth = 4
    )
    sklearn_tree.fit(X, y)
    sklearn_labels = sklearn_tree.predict(X)
    
    leaves = collect_leaves(tree.root)
    assert len(leaves) == 10
    assert get_depth(tree.root) == 4
    assert np.array_equal(labels, sklearn_labels)
    