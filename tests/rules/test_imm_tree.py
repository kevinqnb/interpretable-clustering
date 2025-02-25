import numpy as np
from ExKMC.Tree import Tree as ExTree
from sklearn import cluster
from intercluster.rules import (
    ImmTree,
    ExkmcTree,
    collect_leaves,
    get_depth
)
from intercluster.utils import flatten_labels, labels_format


def test_exkmc_tree():
    samples = 1000
    for i in range(samples):
        n = 100
        d = 10
        k = 5
        data = np.random.uniform(size = (n,d))
        kmeans = cluster.KMeans(n_clusters=k, random_state=i).fit(data)
        labels = kmeans.labels_

        exkmc = ExTree(
            k = k,
            max_leaves = 10,
            base_tree = "IMM"
        )
        exkmc.fit(data, kmeans)
        exkmc_labels = exkmc.predict(data)

        exkmc_tree = ExkmcTree(
            k = k,
            kmeans = kmeans,
            max_leaf_nodes = 10,
            imm = True
        )
        exkmc_tree.fit(data, labels_format(labels))
        exkmc_tree_labels = exkmc_tree.predict(data, leaf_labels = False)
        exkmc_tree_label_array = flatten_labels(exkmc_tree_labels)

        assert np.array_equal(exkmc_labels, exkmc_tree_label_array)
        assert exkmc._size() == exkmc_tree.node_count
        assert exkmc._max_depth() == exkmc_tree.depth