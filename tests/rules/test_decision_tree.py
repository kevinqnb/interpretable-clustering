import numpy as np
from sklearn.tree import DecisionTreeClassifier
from intercluster.rules import (
    ID3Tree,
    SklearnTree,
    ObliqueTree,
    collect_leaves,
    get_depth
)
from intercluster.utils import flatten_labels, labels_format


def test_sklearn_tree_max_leaves():
    samples = 1000
    for i in range(samples):
        n = 100
        d = 10
        data = np.random.uniform(size = (n,d))
        labels = np.random.choice(5, size = n)

        clf = DecisionTreeClassifier(
            criterion = 'entropy',
            max_leaf_nodes = 10,
            max_depth = n - 1, 
            min_samples_leaf = 1,
            random_state = i
        )
        clf.fit(data, labels)
        clf_labels = clf.predict(data)

        d_tree = SklearnTree(criterion = 'entropy', max_leaf_nodes = 10, random_state = i)
        d_tree.fit(data, labels_format(labels))
        d_labels = d_tree.predict(data, leaf_labels = False)
        d_label_array = flatten_labels(d_labels)

        assert np.array_equal(clf_labels, d_label_array)
        assert clf.tree_.node_count == d_tree.node_count
        assert clf.get_n_leaves() == d_tree.leaf_count
        assert clf.get_depth() == d_tree.depth


def test_sklearn_tree_max_depth():
    samples = 1000
    for i in range(samples):
        n = 100
        d = 10
        data = np.random.uniform(size = (n,d))
        labels = np.random.choice(5, size = n)

        clf = DecisionTreeClassifier(
            criterion = 'entropy',
            max_leaf_nodes = n,
            max_depth = 4, 
            min_samples_leaf = 1,
            random_state = i
        )
        clf.fit(data, labels)
        clf_labels = clf.predict(data)

        d_tree = SklearnTree(criterion = 'entropy', max_depth = 4, random_state = i)
        d_tree.fit(data, labels_format(labels))
        d_labels = d_tree.predict(data, leaf_labels = False)
        d_label_array = flatten_labels(d_labels)

        assert np.array_equal(clf_labels, d_label_array)
        assert clf.tree_.node_count == d_tree.node_count
        assert clf.get_n_leaves() == d_tree.leaf_count
        assert clf.get_depth() == d_tree.depth



def test_max_leaf_nodes(example_dataset):
    X, y = example_dataset
    tree = ID3Tree(max_leaf_nodes = 10)
    tree.fit(X, y)
    labels = tree.predict(X)
    
    sklearn_tree = DecisionTreeClassifier(
        criterion='entropy',
        max_leaf_nodes = 10
    )
    sklearn_tree.fit(X, flatten_labels(y))
    sklearn_labels = sklearn_tree.predict(X)
    
    leaves = collect_leaves(tree.root)
    assert len(leaves) == 10
    assert np.array_equal(flatten_labels(labels), sklearn_labels)
    
    
def test_max_depth(example_dataset):
    X, y = example_dataset
    tree = ID3Tree(max_depth = 4)
    tree.fit(X, y)
    labels = tree.predict(X)
    
    sklearn_tree = DecisionTreeClassifier(
        criterion='entropy',
        max_depth = 4
    )
    sklearn_tree.fit(X, flatten_labels(y))
    sklearn_labels = sklearn_tree.predict(X)
    
    assert get_depth(tree.root) == 4
    assert np.array_equal(flatten_labels(labels), sklearn_labels)
    
    
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
    sklearn_tree.fit(X, flatten_labels(y))
    sklearn_labels = sklearn_tree.predict(X)
    
    leaves = collect_leaves(tree.root)
    assert len(leaves) == 10
    assert get_depth(tree.root) == 4
    assert np.array_equal(flatten_labels(labels), sklearn_labels)
    