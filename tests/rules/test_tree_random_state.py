import numpy as np
from intercluster.decision_trees import ExplanationTree
from intercluster.utils import labels_format


def symmetric_four_cluster_data():
    """
    4 points, one per quadrant. Splitting on feature 0 or feature 1 first gives
    identical information gain, so the base Tree's heap tie-break (previously
    bare np.random.rand(), now a seeded self._rng.random()) determines which
    feature is chosen at the root.
    """
    X = np.array([
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
    ])
    y = labels_format(np.array([0, 1, 2, 3]))
    return X, y


def test_explanation_tree_reproducible_with_same_random_state():
    """
    Same random_state -> identical root split, even when candidate splits are
    exactly tied on gain -- PROVIDED the global NumPy seed is also fixed before
    each fit. ExplanationTree's `random_state` alone only controls the Tree
    base class's heap tie-break; the compiled Cython splitter backing
    ExplanationTree (`split_cy`) has its own tie-break that reads the global
    RNG state directly (see explanation_tree.py's `random_state` docstring
    caveat), so full reproducibility here also requires np.random.seed(...)
    immediately before fit(), exactly as experiments/climate/*.py do.
    """
    X, y = symmetric_four_cluster_data()

    np.random.seed(123)
    tree1 = ExplanationTree(num_clusters=4, random_state=123)
    tree1.fit(X, y)

    np.random.seed(123)
    tree2 = ExplanationTree(num_clusters=4, random_state=123)
    tree2.fit(X, y)

    assert tree1.root.condition.features.tolist() == tree2.root.condition.features.tolist()
    assert tree1.root.condition.threshold == tree2.root.condition.threshold


def test_explanation_tree_unseeded_tiebreak_actually_varies():
    """Sanity check: without a fixed seed, the tied root split isn't forced to
    always resolve the same way (i.e. the RNG is actually wired into the
    tie-break, not dead code)."""
    X, y = symmetric_four_cluster_data()

    chosen_features = set()
    for _ in range(20):
        tree = ExplanationTree(num_clusters=4)
        tree.fit(X, y)
        chosen_features.add(tuple(tree.root.condition.features.tolist()))

    assert len(chosen_features) > 1, "expected root split choice to vary across unseeded runs"
