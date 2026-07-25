import numpy as np
from typing import List, Set
from numpy.typing import NDArray

from intercluster import Rule, Decision, LinearCondition, flatten_labels
from .decision_set import DecisionSet

from Orange.classification.rules import CN2Learner
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable


class CN2(DecisionSet):
    """
    Standard (ordered) CN2 rule inducer wrapped as a DecisionSet.

    Implements the standard CN2 algorithm (Clark & Niblett, 1989) via the
    orange3 library: a single sequential-covering loop over the whole
    multiclass population. At each iteration it finds the single best rule
    (scored by entropy) over all remaining examples regardless of class,
    assigns it a label by majority vote among the points it covers, then
    removes those points and repeats. Because each iteration draws from the
    full remaining population rather than one class at a time, the resulting
    rule list naturally interleaves classes instead of grouping all of one
    class's rules before the next.

    NOTE: Orange's rule list is technically a *decision list* -- a rule's
    majority-vote label is only exactly valid relative to the residual
    population left after earlier rules have "claimed" their covered points.
    On the other hand, intercluster's DecisionSet is a *decision set* -- each rule
    is applied independently to the full population, and the majority-vote label
    is only valid relative to the full population. 

    Args:
        n_select (int, optional): Maximum number of rules to retain. Takes the
            first n_select rules in iteration order (class-grouped). If None,
            all rules found by CN2 are kept. Defaults to None.
        beam_width (int): Width of the beam search. Larger values explore more
            of the hypothesis space at the cost of runtime. Defaults to 10.
        min_covered_examples (int): Minimum number of examples a rule must cover
            to be considered valid. Defaults to 1.
        max_rule_conditions (int): Maximum number of conditions (selectors) per
            rule. This is the orange3 internal cap per rule, distinct from
            n_select which caps the total number of rules. Defaults to 5.
    """

    def __init__(
        self,
        n_select: int = None,
        beam_width: int = 10,
        min_covered_examples: int = 1,
        max_rule_conditions: int = 5,
        rules=None,
        rule_labels=None,
    ):
        super().__init__(rules=[])

        if n_select is not None:
            if not isinstance(n_select, int) or n_select <= 0:
                raise ValueError("n_select must be a positive integer.")
        self.n_select = n_select

        if not isinstance(beam_width, int) or beam_width <= 0:
            raise ValueError("beam_width must be a positive integer.")
        self.beam_width = beam_width

        if not isinstance(min_covered_examples, int) or min_covered_examples <= 0:
            raise ValueError("min_covered_examples must be a positive integer.")
        self.min_covered_examples = min_covered_examples

        if not isinstance(max_rule_conditions, int) or max_rule_conditions <= 0:
            raise ValueError("max_rule_conditions must be a positive integer.")
        self.max_rule_conditions = max_rule_conditions

        # Populated by induce(): the full, untruncated, non-default rule list
        # and the label mapping needed to convert it to Decisions. n_select
        # only affects finalize()'s truncation of these, not induce() itself.
        self._non_default = None
        self._unique_labels = None


    def _build_orange_table(self, X: NDArray, y_flat: NDArray):
        """
        Build an Orange Table from numpy arrays.

        Returns:
            table (Orange.data.Table): Orange-format dataset.
            unique_labels (list[int]): Sorted list of unique integer labels,
                used to map Orange class indices back to original labels.
        """
        n_features = X.shape[1]
        attrs = [ContinuousVariable(f'x{i}') for i in range(n_features)]

        unique_labels = sorted(set(y_flat.tolist()))
        class_var = DiscreteVariable('class', values=[str(l) for l in unique_labels])
        domain = Domain(attrs, class_var)

        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        y_idx = np.array([label_to_idx[l] for l in y_flat], dtype=float)

        return Table.from_numpy(domain, X, y_idx), unique_labels


    @staticmethod
    def _selector_to_condition(selector) -> LinearCondition:
        """
        Convert an Orange Selector to a LinearCondition.

        LinearCondition only has an inclusive `<=` direction (direction=-1);
        its `direction=1` is strict `>`. Orange's `>=` is inclusive, so
        rather than approximating it with strict `>`, it's represented as
        the negated inclusive condition `-x <= -value` (still direction=-1),
        which is exactly equivalent to `x >= value`.
        """
        if selector.op == '<=':
            weight, threshold = 1.0, float(selector.value)
        else:
            weight, threshold = -1.0, -float(selector.value)
        return LinearCondition(
            features=np.array([selector.column]),
            weights=np.array([weight]),
            threshold=threshold,
            direction=-1,
        )


    def _orange_rule_to_decision(self, orange_rule, unique_labels: list) -> Decision:
        """Convert an Orange Rule to an intercluster Decision."""
        conditions = [self._selector_to_condition(s) for s in orange_rule.selectors]
        rule = Rule(conditions)
        label = unique_labels[orange_rule.prediction]
        return Decision(rule, label)


    def select(self, X: NDArray, y: List[Set[int]] = None):
        raise NotImplementedError(
            "CN2 generates rules internally via fit(); select() is not used."
        )


    def induce(self, X: NDArray, y: List[Set[int]] = None):
        """
        Runs the CN2 beam search once and caches the full, untruncated,
        non-default rule list. This is the expensive part of `fit()`.

        NOTE: This was designed to be idempotent: induction will not run again
        for a second call on an instance that has already induced, 
        so a caller sweeping n_select cheaply against a single induction (see `finalize()`) 
        can call this before each `finalize()` without re-triggering the search.
        Because of that, however, a given CN2 instance should only
        ever be induced against one `(X, y)` pair -- the same "fresh instance
        per fit" convention already used for every DecisionSet subclass in
        this codebase.

        Args:
            X (NDArray): Input dataset of shape (n, d).
            y (List[Set[int]]): Cluster labels. Each inner set must contain
                exactly one label — multi-label points are rejected. Points
                with an empty label set (outliers) are excluded from training.

        Raises:
            ValueError: If y is None, contains multi-label points, or yields
                no labeled training examples after outlier filtering.
        """
        if self._non_default is not None:
            return

        if y is None:
            raise ValueError("CN2 requires cluster labels y.")

        y_flat = flatten_labels(y)
        if len(y_flat) != len(y):
            raise ValueError(
                "CN2 requires exactly one label per point; "
                "multi-label points were detected."
            )

        # Exclude outlier points (empty label sets → -1 after flattening)
        valid_mask = y_flat != -1
        X_train = X[valid_mask]
        y_train = y_flat[valid_mask]

        if len(y_train) == 0:
            raise ValueError("No labeled points found after filtering outliers.")

        table, unique_labels = self._build_orange_table(X_train, y_train)

        learner = CN2Learner()
        learner.rule_finder.search_algorithm.beam_width = self.beam_width
        learner.rule_finder.general_validator.max_rule_length = self.max_rule_conditions
        learner.rule_finder.general_validator.min_covered_examples = self.min_covered_examples

        classifier = learner(table)

        # Exclude the default rule (empty selectors, covers everything)
        self._non_default = [r for r in classifier.rule_list if r.length > 0]
        self._unique_labels = unique_labels


    def finalize(self, n_select: int = None):
        """
        Truncates the cached induction (see `induce()`) to `n_select` rules
        and rebuilds `decision_set`/`max_rule_length`. Must be called after `induce()`.

        Args:
            n_select (int, optional): Maximum number of rules to retain, as
                in `__init__`. Defaults to None (keep all induced rules).

        Raises:
            ValueError: If `induce()` has not been called yet.
        """
        if self._non_default is None:
            raise ValueError("Must call induce() before finalize().")

        self.n_select = n_select
        non_default = self._non_default
        if n_select is not None:
            non_default = non_default[:n_select]

        self.decision_set = {
            self._orange_rule_to_decision(r, self._unique_labels) for r in non_default
        }

        self.decision_set = self.trim()
        self.max_rule_length = (
            max(len(d.rule) for d in self.decision_set) if self.decision_set else 0
        )
        self.decision_set = list(self.decision_set)


    def fit(self, X: NDArray, y: List[Set[int]] = None):
        """
        Args:
            X (NDArray): Input dataset of shape (n, d).
            y (List[Set[int]]): Cluster labels. Each inner set must contain
                exactly one label — multi-label points are rejected. Points
                with an empty label set (outliers) are excluded from training.

        Raises:
            ValueError: If y is None, contains multi-label points, or yields
                no labeled training examples after outlier filtering.
        """
        self.induce(X, y)
        self.finalize(self.n_select)
