import numpy as np
import pickle

from intercluster.rules import LinearCondition, save_rules, load_rules, save_decisions, load_decisions
from intercluster import Rule, Decision


def _make_rule(feature, threshold, direction):
    return Rule([
        LinearCondition(
            features=np.array([feature]),
            weights=np.array([1.0]),
            threshold=threshold,
            direction=direction,
        )
    ])


####################################################################################################
# save_rules / load_rules
####################################################################################################


def test_save_and_load_rules_round_trip(tmp_path):
    rules = [_make_rule(0, 1.0, -1), _make_rule(1, 2.0, 1)]
    path = tmp_path / "rules.pkl"

    save_rules(rules, path)
    loaded = load_rules(path)

    assert loaded == rules
    assert isinstance(loaded, list)


def test_load_rules_creates_parent_directories(tmp_path):
    nested_path = tmp_path / "a" / "b" / "rules.pkl"
    rules = [_make_rule(0, 1.0, -1)]
    save_rules(rules, nested_path)
    assert nested_path.exists()
    assert load_rules(nested_path) == rules


def test_load_rules_wraps_non_list_object_in_a_list(tmp_path):
    """load_rules is documented to be permissive: if the pickled object isn't a list/tuple,
    it wraps it in a single-element list instead of raising or returning it bare."""
    single_rule = _make_rule(0, 1.0, -1)
    path = tmp_path / "single_rule.pkl"
    with open(path, "wb") as f:
        pickle.dump(single_rule, f)

    loaded = load_rules(path)
    assert loaded == [single_rule]


####################################################################################################
# save_decisions / load_decisions (analogous round-trip)
####################################################################################################


def test_save_and_load_decisions_round_trip(tmp_path):
    decisions = [Decision(_make_rule(0, 1.0, -1), 0), Decision(_make_rule(1, 2.0, 1), 1)]
    path = tmp_path / "decisions.pkl"

    save_decisions(decisions, path)
    loaded = load_decisions(path)

    assert loaded == decisions
