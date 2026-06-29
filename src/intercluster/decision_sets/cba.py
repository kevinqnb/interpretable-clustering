import numpy as np
from collections import Counter
from typing import List, Set
from numpy.typing import NDArray
from intercluster import (
    Rule,
    Decision,
    simplified_rule_length,
)
from .decision_set import DecisionSet


####################################################################################################


class CBA(DecisionSet):
    """
    Classification-Based Association (CBA) rule selector — M1 variant.

    Selects a decision set from a candidate rule pool using the M1 algorithm
    from Liu et al. (1998). Rules are sorted by CBA precedence (confidence DESC,
    support DESC, rule length ASC, generation order ASC) and added greedily to
    a classifier until all data points are covered. The final classifier is cut
    at the prefix that minimises total errors (rule errors + default-class errors).

    Multi-label resolution: the base class populates ``self.decision_set`` with
    one ``Decision(rule, label)`` per (rule, unique-label) pair, following the
    library convention. Before M1 begins, each unique rule is mapped to the single
    label that maximises its confidence (ties broken by support → rule length →
    generation order). This guarantees that each rule is assigned to at most one
    cluster.

    A ``default_class`` attribute is stored after fitting — the most common label
    among points not covered by any selected rule at the optimal cut point.
    ``predict()`` is not overridden: uncovered points return empty sets, consistent
    with the rest of the library.

    Args:
        rules (List[Rule]): Candidate rule pool.
        rule_labels (List[Set[int]], optional): Per-rule label assignments.
            If None, each rule is paired with every unique label present in ``y``
            during fitting. Defaults to None.
        n_select (int, optional): Maximum number of rules to retain. Applied after
            M1 error minimisation by taking the top-precedence prefix of the
            resulting classifier. If None, no cap is imposed. Defaults to None.
    """

    def __init__(
        self,
        rules: List[Rule],
        rule_labels: List[Set[int]] = None,
        n_select: int = None,
    ):
        super().__init__(rules=rules, rule_labels=rule_labels)

        if n_select is not None:
            assert isinstance(n_select, int) and n_select > 0, \
                "n_select must be a positive integer."
        self.n_select = n_select
        self.default_class = None


    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def select(self, X: NDArray, y: List[Set[int]]) -> set[Decision]:
        """
        Run the CBA M1 algorithm on the current candidate decision set.

        Args:
            X (NDArray): Input dataset of shape (n, d).
            y (List[Set[int]]): Cluster labels; y[i] is the set of clusters
                point i belongs to (typically a singleton from k-means).

        Returns:
            set[Decision]: The selected decision set.
        """
        if self.decision_set is None:
            raise ValueError("Decision set has not been initialized yet.")

        n = X.shape[0]

        # ----------------------------------------------------------------
        # Stage 1: compute (confidence, support) per Decision
        # ----------------------------------------------------------------
        # Generation order is the index of the rule in self.rules (the original
        # input list), giving a deterministic tiebreaker. Rules not in self.rules
        # (e.g. when rule_labels was supplied and set_labels created a different
        # set) fall back to a large sentinel so they sort last.
        rule_gen_order: dict[Rule, int] = {r: i for i, r in enumerate(self.rules)}
        sentinel = len(self.rules)

        # Each entry: (confidence, support, rule_length, gen_order, decision)
        scored: list[tuple[float, float, int, int, Decision]] = []

        for decision in self.decision_set:
            coverage_mask = decision.rule.evaluate(X)
            n_covered = int(np.sum(coverage_mask))
            if n_covered == 0:
                continue

            covered_indices = np.where(coverage_mask)[0]
            n_correct = sum(1 for i in covered_indices if decision.label in y[i])

            support = n_correct / n
            confidence = n_correct / n_covered
            rule_len = simplified_rule_length(decision.rule)
            gen_order = rule_gen_order.get(decision.rule, sentinel)

            scored.append((confidence, support, rule_len, gen_order, decision))

        # ----------------------------------------------------------------
        # Stage 2: keep one (rule, label) pair per unique rule
        # Multi-label resolution: highest precedence = highest confidence,
        # then support, then shorter rule, then earlier generation order.
        # ----------------------------------------------------------------
        rule_to_best: dict[Rule, tuple] = {}
        for entry in scored:
            rule = entry[4].rule
            if rule not in rule_to_best:
                rule_to_best[rule] = entry
            else:
                existing = rule_to_best[rule]
                if _has_higher_precedence(entry, existing):
                    rule_to_best[rule] = entry

        # ----------------------------------------------------------------
        # Stage 3: sort candidates by precedence, highest first
        # Key: (-confidence, -support, rule_length, gen_order)
        # ----------------------------------------------------------------
        candidates = sorted(
            rule_to_best.values(),
            key=lambda e: (-e[0], -e[1], e[2], e[3]),
        )

        # ----------------------------------------------------------------
        # Stage 4: M1 algorithm
        # ----------------------------------------------------------------
        uncovered = set(range(n))
        classifier: list[Decision] = []
        default_classes: list[int | None] = []
        cumulative_rule_errors = 0
        total_errors: list[int] = []

        for confidence, support, rule_len, gen_order, decision in candidates:
            if not uncovered:
                break

            coverage_mask = decision.rule.evaluate(X)
            covered_idx = set(np.where(coverage_mask)[0])

            # Rule fires only if it correctly classifies ≥1 still-uncovered point
            correct_uncovered = {i for i in covered_idx & uncovered if decision.label in y[i]}
            if not correct_uncovered:
                continue

            classifier.append(decision)

            # Rule errors: covered points (anywhere, not just uncovered) with
            # wrong label.  Standard M1 counts errors over the entire antecedent
            # coverage, not just the uncovered subset.
            rule_misses = sum(1 for i in covered_idx if decision.label not in y[i])
            cumulative_rule_errors += rule_misses

            # Remove ALL antecedent-covered points from the remaining dataset
            uncovered -= covered_idx

            # Default class = most common label among remaining uncovered points
            label_counts: Counter = Counter(
                lbl for i in uncovered for lbl in y[i]
            )
            if label_counts:
                default_class, default_class_count = label_counts.most_common(1)[0]
            else:
                default_class, default_class_count = None, 0

            default_errors = len(uncovered) - default_class_count
            default_classes.append(default_class)
            total_errors.append(cumulative_rule_errors + default_errors)

        # ----------------------------------------------------------------
        # Stage 5: cut at minimum-error prefix
        # ----------------------------------------------------------------
        if total_errors:
            min_err = min(total_errors)
            cut = total_errors.index(min_err)
            final_classifier = classifier[: cut + 1]
            self.default_class = default_classes[cut]
        else:
            final_classifier = []
            # No rule fired — default class is the globally most common label
            label_counts = Counter(lbl for yi in y for lbl in yi)
            self.default_class = label_counts.most_common(1)[0][0] if label_counts else None

        # ----------------------------------------------------------------
        # Stage 6: apply n_select cap (top-precedence prefix)
        # ----------------------------------------------------------------
        if self.n_select is not None:
            final_classifier = final_classifier[: self.n_select]

        return set(final_classifier)


####################################################################################################


def _has_higher_precedence(
    a: tuple[float, float, int, int, Decision],
    b: tuple[float, float, int, int, Decision],
) -> bool:
    """Return True if candidate ``a`` has strictly higher CBA precedence than ``b``."""
    conf_a, sup_a, len_a, ord_a, _ = a
    conf_b, sup_b, len_b, ord_b, _ = b
    if conf_a != conf_b:
        return conf_a > conf_b
    if sup_a != sup_b:
        return sup_a > sup_b
    if len_a != len_b:
        return len_a < len_b
    return ord_a < ord_b


####################################################################################################
