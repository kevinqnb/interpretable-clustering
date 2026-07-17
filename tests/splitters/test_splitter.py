import numpy as np
from intercluster.decision_trees.splitters import SimpleSplitter
from intercluster.utils import labels_format


# NOTE: SimpleSplitter/DummySplitter are pure test stand-ins (see dummy.py's docstring) and
# aren't used anywhere in src/ or experiments/. Their custom cost() override is only ever
# exercised via cost()/gain() called directly -- split() (inherited from AxisAlignedSplitter)
# always routes through the shared entropy-based Cython path regardless of subclass, so it
# doesn't exercise SimpleSplitter's own cost function at all. That's already covered by
# test_information_splitter.py; here we test the part that's actually unique to SimpleSplitter.


def test_simple_splitter_cost_is_max(simple_dataset):
    splitter = SimpleSplitter()
    splitter.fit(X = simple_dataset, y = labels_format(np.zeros(len(simple_dataset))))
    assert splitter.cost(np.arange(len(simple_dataset))) == np.max(simple_dataset)
    assert splitter.cost(np.array([0, 1])) == np.max(simple_dataset[[0, 1]])


def test_simple_splitter_gain_uses_custom_cost(simple_dataset):
    splitter = SimpleSplitter()
    splitter.fit(X = simple_dataset, y = labels_format(np.zeros(len(simple_dataset))))
    left_indices = np.array([0, 1])
    right_indices = np.array([2, 3])
    parent_cost = splitter.cost(np.arange(len(simple_dataset)))
    expected_gain = parent_cost - (
        splitter.cost(left_indices) + splitter.cost(right_indices)
    )
    assert splitter.gain(left_indices, right_indices) == expected_gain
    assert expected_gain == 7 - (np.max(simple_dataset[left_indices]) + np.max(simple_dataset[right_indices]))
