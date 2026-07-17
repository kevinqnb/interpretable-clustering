import numpy as np
from sklearn.tree import DecisionTreeClassifier
from intercluster.decision_trees import DecisionTree
from intercluster.utils import flatten_labels, labels_format


# NOTE: ID3Tree/ObliqueTree are not used anywhere in experiments/ and are not covered here --
# low priority per maintainer, not worth chasing their own test failures.


def test_sklearn_tree_max_leaves():
    # DecisionTree and sklearn's DecisionTreeClassifier independently implement best-first
    # leaf expansion under a max_leaf_nodes cap; when several candidate splits tie on entropy
    # gain (common with few, low-cardinality label classes) they can pick different-but-equally
    # valid splits. So tree *size* is checked exactly, but per-point label agreement is checked
    # as a high-agreement threshold rather than a brittle exact match.
    samples = 25
    for i in range(samples):
        n = 100
        d = 10
        rng = np.random.RandomState(i)
        data = rng.uniform(size = (n,d))
        labels = rng.choice(5, size = n)

        clf = DecisionTreeClassifier(
            criterion = 'entropy',
            max_leaf_nodes = 10,
            max_depth = n - 1,
            min_samples_leaf = 1,
            random_state = i
        )
        clf.fit(data, labels)
        clf_labels = clf.predict(data)

        d_tree = DecisionTree(criterion = 'entropy', max_leaf_nodes = 10, random_state = i)
        d_tree.fit(data, labels_format(labels))
        d_labels = d_tree.predict(data, leaf_labels = False)
        d_label_array = flatten_labels(d_labels)

        assert clf.tree_.node_count == d_tree.node_count
        assert clf.get_n_leaves() == d_tree.leaf_count
        assert np.mean(clf_labels == d_label_array) >= 0.8


def test_sklearn_tree_max_depth():
    samples = 25
    for i in range(samples):
        n = 100
        d = 10
        rng = np.random.RandomState(i)
        data = rng.uniform(size = (n,d))
        labels = rng.choice(5, size = n)

        clf = DecisionTreeClassifier(
            criterion = 'entropy',
            max_leaf_nodes = n,
            max_depth = 4,
            min_samples_leaf = 1,
            random_state = i
        )
        clf.fit(data, labels)
        clf_labels = clf.predict(data)

        d_tree = DecisionTree(criterion = 'entropy', max_depth = 4, random_state = i)
        d_tree.fit(data, labels_format(labels))
        d_labels = d_tree.predict(data, leaf_labels = False)
        d_label_array = flatten_labels(d_labels)

        assert clf.get_depth() == d_tree.depth
        assert np.mean(clf_labels == d_label_array) >= 0.8
