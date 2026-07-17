import sys
from pathlib import Path

import numpy as np
import pytest

# `experiments/` is a plain (non-installed) top-level directory, not part of the
# `intercluster` package -- only importable once the repo root is on sys.path.
# Mirrors the same bootstrap used in tests/decision_sets/objectives/test_heap_distorted_greedy.py.
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = next((p for p in _HERE.parents if (p / "data").is_dir()), None)
if _PROJECT_ROOT is not None and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.experiment import Experiment
from experiments.modules import Baseline, aggregate_trials
from intercluster.measurements import TotalCoverage


####################################################################################################
# aggregate_trials
####################################################################################################


def test_aggregate_trials_empty():
    assert aggregate_trials([]) == {}


def test_aggregate_trials_basic_mean_std():
    trials = [
        {'a': 1.0, 'b': 10.0},
        {'a': 2.0, 'b': 20.0},
        {'a': 3.0, 'b': 30.0},
    ]
    result = aggregate_trials(trials)
    assert result['a']['mean'] == 2.0
    assert np.isclose(result['a']['std'], np.std([1.0, 2.0, 3.0]))
    assert result['a']['values'] == [1.0, 2.0, 3.0]
    assert result['b']['mean'] == 20.0
    assert result['b']['values'] == [10.0, 20.0, 30.0]


def test_aggregate_trials_excludes_nan_and_none_from_mean_but_keeps_in_values():
    trials = [
        {'a': 1.0},
        {'a': np.nan},
        {'a': None},
        {'a': 3.0},
    ]
    result = aggregate_trials(trials)
    # mean/std only over the two numeric values (1.0, 3.0); NaN/None are excluded.
    assert result['a']['mean'] == 2.0
    assert np.isclose(result['a']['std'], np.std([1.0, 3.0]))
    # but all four raw values (including the excluded ones) are preserved in order.
    assert len(result['a']['values']) == 4
    assert result['a']['values'][0] == 1.0
    assert np.isnan(result['a']['values'][1])
    assert result['a']['values'][2] is None
    assert result['a']['values'][3] == 3.0


def test_aggregate_trials_all_nan_key_returns_nan_mean():
    trials = [{'a': np.nan}, {'a': np.nan}]
    result = aggregate_trials(trials)
    assert np.isnan(result['a']['mean'])
    assert np.isnan(result['a']['std'])
    assert len(result['a']['values']) == 2


####################################################################################################
# Experiment.run() -- baseline handling and per-(module, param_tuple) broadcast
####################################################################################################


class _FakeBaseline(Baseline):
    """A minimal baseline stand-in: 4 points, first 2 in cluster 0, last 2 in cluster 1."""

    def __init__(self):
        super().__init__(name = 'FakeBaseline')
        self.max_rule_length = np.nan
        self.sum_rule_length = np.nan
        self.weighted_average_rule_length = np.nan
        self.labels = [{0}, {0}, {1}, {1}]

    def assign(self, X):
        n = X.shape[0]
        assignment = np.zeros((n, 2), dtype = bool)
        assignment[:2, 0] = True
        assignment[2:, 1] = True
        return assignment


class _FakeModule:
    """A minimal Module stand-in: every point ends up in cluster 0, fully covered."""

    def __init__(self, name):
        self.name = name
        self.max_rule_length = 3
        self.sum_rule_length = 5
        self.weighted_average_rule_length = 2.5
        self.lambda_val = 0.1
        self.n_available_decisions = 7
        self.fitting_params = None

    def update_fitting_params(self, fitting_params):
        self.fitting_params = fitting_params

    def fit(self, X, y):
        n = X.shape[0]
        data_to_rule_assignment = np.ones((n, 1), dtype = bool)
        rule_to_cluster_assignment = np.array([[True, False]])
        data_to_cluster_assignment = np.zeros((n, 2), dtype = bool)
        data_to_cluster_assignment[:, 0] = True
        return data_to_rule_assignment, rule_to_cluster_assignment, data_to_cluster_assignment


def test_experiment_run_broadcasts_across_param_tuple_and_separates_baseline():
    data = np.zeros((4, 2))
    baseline = _FakeBaseline()
    module = _FakeModule('FakeModule')
    # A single fit (param_tuple = (1, 2)) whose result must be broadcast to both keys 1 and 2,
    # exactly as lambda.py's comparison modules rely on (same fit, multiple recorded labels).
    module_list = [(module, {(1, 2): {'some_param': 'value'}})]
    measurement_fns = [TotalCoverage()]

    exp = Experiment(
        data = data,
        baseline = baseline,
        module_list = module_list,
        measurement_fns = measurement_fns,
        cpu_count = 1,
    )
    result_dict = exp.run()

    assert set(result_dict.keys()) >= {'fixed-parameters', 'baseline', 'modules'}

    # Baseline is computed and recorded independently of the module sweep.
    baseline_result = result_dict['baseline']['FakeBaseline']
    assert baseline_result['total-coverage'] == 4  # all 4 points covered, 2 in each cluster

    # The module's single fit result is broadcast identically to both entries of the
    # (1, 2) param tuple key.
    module_result = result_dict['modules']['FakeModule']
    assert module_result['total-coverage'] == {1: 4, 2: 4}
    assert module_result['lambda'] == {1: 0.1, 2: 0.1}
    assert module_result['lambda_n_rules'] == {1: 7, 2: 7}
    assert module_result['max-rule-length'] == {1: 3, 2: 3}
    assert module_result['sum-rule-length'] == {1: 5, 2: 5}
    assert module_result['weighted-avg-length'] == {1: 2.5, 2: 2.5}

    # The fitting params passed through update_fitting_params reached the module unchanged.
    assert module.fitting_params == {'some_param': 'value'}


def test_experiment_run_multiple_param_tuples_are_independent_fits():
    data = np.zeros((4, 2))
    baseline = _FakeBaseline()
    module = _FakeModule('FakeModule')
    # Two distinct param tuples for the same module -- each is its own fit/task, not broadcast
    # into each other, even though this fake module ignores fitting_params and returns the
    # same assignment either way.
    module_list = [(module, {(1,): {'p': 'a'}, (2,): {'p': 'b'}})]
    measurement_fns = [TotalCoverage()]

    exp = Experiment(
        data = data,
        baseline = baseline,
        module_list = module_list,
        measurement_fns = measurement_fns,
        cpu_count = 1,
    )
    result_dict = exp.run()

    module_result = result_dict['modules']['FakeModule']
    assert set(module_result['total-coverage'].keys()) == {1, 2}
    assert module_result['total-coverage'][1] == 4
    assert module_result['total-coverage'][2] == 4
