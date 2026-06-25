import numpy as np
from numpy.typing import NDArray
from typing import List, Set
from intercluster import (
    Rule,
    Decision,
    unique_labels,
)
from .decision_set import DecisionSet


####################################################################################################


class WRABaseline(DecisionSet):
    """
    Decision set baseline that selects rules by Weighted Relative Accuracy (WRA):

        WRA(R, C) = P(X_R) * (P(X_C | X_R) - P(X_C))

    where P(X_R) is the (weighted) fraction of points satisfying rule R,
    P(X_C) is the (weighted) fraction of points in cluster C, and
    P(X_C | X_R) is the (weighted) fraction of R-covered points in cluster C.

    Each rule is matched to its single best cluster (the C maximizing WRA(R, C)). 
    This is so that each rule is only selected once.
    Rules are then ranked by that best WRA and the top n_select with strictly
    positive WRA are returned.
    """
    def __init__(
        self,
        rules: List[Rule],
        n_select: int,
        weights: NDArray = None,
        rule_labels: List[Set[int]] = None,
    ):
        """
        Args:
            rules (List[Rule]): Candidate rule pool.
            n_select (int): Maximum number of rules to select.
            weights (NDArray, optional): Per-point weights for probability estimates.
                Defaults to None (uniform weights).
            rule_labels (List[Set[int]], optional): Must be None for WRABaseline.
        """
        assert rule_labels is None, 'rule_labels must be None for WRABaseline.'
        super().__init__(rules=rules, rule_labels=None)

        assert isinstance(n_select, int) and n_select > 0, \
            'n_select must be a positive integer.'
        self.n_select = n_select

        assert weights is None or isinstance(weights, np.ndarray), \
            'weights must be a numpy array.'
        if weights is not None:
            assert len(weights.shape) == 1, 'weights must be a 1D array.'
        self.weights = weights

    def select(self, X: NDArray, y: List[Set[int]]) -> set[Decision]:
        """
        For each unique rule, finds the cluster maximizing WRA(R, C), then
        returns the top n_select such (Rule, best-cluster) decisions ranked by
        WRA descending. Only decisions with strictly positive WRA are eligible.

        Args:
            X (NDArray): Input dataset of shape (n, d).
            y (List[Set[int]]): Cluster labels; y[i] is the set of clusters
                that point i belongs to.

        Returns:
            set[Decision]: Selected decisions, at most n_select in size.
        """
        if self.decision_set is None:
            raise ValueError('Decision set has not been initialized yet.')

        n = X.shape[0]
        weights = self.weights if self.weights is not None else np.ones(n, dtype=float)
        W_total = float(weights.sum())

        # Precompute per-cluster membership masks and marginal probabilities P(X_C).
        label_set = unique_labels(y)
        cluster_mask: dict[int, np.ndarray] = {}
        p_cluster: dict[int, float] = {}
        for lbl in label_set:
            mask = np.array([lbl in y[i] for i in range(n)], dtype=bool)
            cluster_mask[lbl] = mask
            p_cluster[lbl] = float(weights[mask].sum()) / W_total if W_total > 0.0 else 0.0

        # Group candidate decisions by their underlying rule so we evaluate each
        # rule's coverage mask exactly once.
        rule_to_decisions: dict[Rule, list[Decision]] = {}
        for decision in self.decision_set:
            rule_to_decisions.setdefault(decision.rule, []).append(decision)

        # For each rule, pick the best-WRA (Rule, Cluster) pair.
        scored: list[tuple[float, Decision]] = []
        for rule, decisions in rule_to_decisions.items():
            rule_mask = rule.evaluate(X)
            w_rule = float(weights[rule_mask].sum())
            if w_rule == 0.0:
                continue

            p_rule = w_rule / W_total if W_total > 0.0 else 0.0

            best_wra = 0.0
            best_decision = None
            for decision in decisions:
                lbl = decision.label
                p_cluster_given_rule = float(weights[rule_mask & cluster_mask[lbl]].sum()) / w_rule
                wra = p_rule * (p_cluster_given_rule - p_cluster[lbl])
                if wra > best_wra:
                    best_wra = wra
                    best_decision = decision

            if best_decision is not None:
                scored.append((best_wra, best_decision))

        scored.sort(key=lambda x: -x[0])
        return {decision for _, decision in scored[: self.n_select]}


####################################################################################################
