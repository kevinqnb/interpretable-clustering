import numpy as np
from typing import List, Set
from numpy.typing import NDArray
from intercluster import (
    Rule,
    Decision,
    flatten_labels,
)
from .decision_set import DecisionSet


####################################################################################################


class IDSCoverageCache:
    """
    Precomputed per-decision coverage statistics for IDS optimization.

    Building this cache once allows reuse across multiple n_select values or
    confidence-sweep subsets (via subset()).
    """

    def __init__(self):
        self.decisions: List[Decision] = None
        self.antecedent_masks: NDArray = None   # (D, N) bool
        self.correct_masks: NDArray = None      # (D, N) bool
        self.overlap_matrix: NDArray = None     # (D, D) int32
        self.same_class_matrix: NDArray = None  # (D, D) bool
        self.N: int = None
        self.L_max: int = None

    def compute(self, decisions: List[Decision], X: NDArray, y_flat: NDArray) -> None:
        """
        Compute coverage statistics for the given decisions on dataset (X, y_flat).

        Args:
            decisions: Ordered list of Decision objects.
            X:         (N, d) data matrix.
            y_flat:    (N,) integer array of cluster labels.
        """
        D = len(decisions)
        N = len(y_flat)
        self.decisions = list(decisions)
        self.N = N
        self.L_max = max(len(d.rule) for d in decisions) if decisions else 0

        antecedent = np.zeros((D, N), dtype=bool)
        correct = np.zeros((D, N), dtype=bool)
        for i, d in enumerate(decisions):
            antecedent[i] = d.rule.evaluate(X)
            correct[i] = antecedent[i] & (y_flat == d.label)

        self.antecedent_masks = antecedent
        self.correct_masks = correct
        ant_int = antecedent.astype(np.int32)
        self.overlap_matrix = ant_int @ ant_int.T

        labels = np.array([d.label for d in decisions])
        self.same_class_matrix = (labels[:, None] == labels[None, :])

    def subset(self, indices) -> 'IDSCoverageCache':
        """
        Return a new cache restricted to the given decision indices.

        Slices all precomputed arrays — useful for confidence-sweep experiments
        where only a subset of the full rule pool is active at each iteration.
        """
        idx = np.asarray(indices)
        cache = IDSCoverageCache()
        cache.decisions = [self.decisions[i] for i in idx]
        cache.N = self.N
        cache.L_max = max(len(d.rule) for d in cache.decisions) if cache.decisions else 0
        cache.antecedent_masks = self.antecedent_masks[idx]
        cache.correct_masks = self.correct_masks[idx]
        cache.overlap_matrix = self.overlap_matrix[np.ix_(idx, idx)]
        cache.same_class_matrix = self.same_class_matrix[np.ix_(idx, idx)]
        return cache


####################################################################################################


class IDSObjective:
    """
    7-term weighted IDS objective function (Lakkaraju et al., KDD 2016).

    f = lambda · [f0, f1, f2, f3, f4, f5, f6] where:
        f0: M - |S|                    (fewer rules is better)
        f1: L_max·M - sum_length(S)    (shorter rules are better)
        f2: N·M² - intraclass_overlap  (less intra-class antecedent overlap)
        f3: N·M² - interclass_overlap  (less inter-class antecedent overlap)
        f4: |distinct classes in S|    (more class diversity)
        f5: N·M - sum_incorrect(S)     (more correct coverage)
        f6: |union of correct covers|  (more distinct correctly covered points)

    Args:
        lambdas: List of 7 non-negative weights.
        cache:   IDSCoverageCache for the candidate decision pool.
        N:       Number of data points.
        M:       Number of decisions in the candidate pool (= len(cache.decisions)).
    """

    def __init__(self, lambdas: List[float], cache: IDSCoverageCache, N: int, M: int):
        if len(lambdas) != 7:
            raise ValueError("lambdas must have exactly 7 elements.")
        self.lambdas = list(lambdas)
        self.cache = cache
        self.N = N
        self.M = M

    def evaluate(self, solution_indices) -> float:
        """
        Evaluate the objective on a solution set given as indices into cache.decisions.

        Args:
            solution_indices: Iterable of int indices (set, list, etc.).

        Returns:
            Scalar objective value.
        """
        S = sorted(solution_indices)
        if not S:
            terms = [
                self.M,
                self.cache.L_max * self.M,
                self.N * self.M ** 2,
                self.N * self.M ** 2,
                0,
                self.N * self.M,
                0,
            ]
        else:
            sub_overlap = self.cache.overlap_matrix[np.ix_(S, S)]
            sub_same = self.cache.same_class_matrix[np.ix_(S, S)]
            intra = int(np.triu(sub_overlap * sub_same, k=1).sum())
            inter = int(np.triu(sub_overlap * (~sub_same), k=1).sum())
            correct_union = np.logical_or.reduce(self.cache.correct_masks[S])
            terms = [
                self.M - len(S),
                self.cache.L_max * self.M - sum(len(self.cache.decisions[i].rule) for i in S),
                self.N * self.M ** 2 - intra,
                self.N * self.M ** 2 - inter,
                len({self.cache.decisions[i].label for i in S}),
                self.N * self.M - sum(self.N - int(self.cache.correct_masks[i].sum()) for i in S),
                int(correct_union.sum()),
            ]
        return float(np.dot(self.lambdas, terms))


####################################################################################################


class SLSOptimizer:
    """
    Stochastic Local Search for unconstrained submodular maximization.

    Implements the SLS algorithm from Mirzasoleiman et al. adapted for the IDS
    objective. Runs two variants of optimize_delta and returns the better solution.

    Args:
        objective:   IDSObjective instance.
        all_indices: List of integer indices into objective.cache.decisions.
    """

    def __init__(self, objective: IDSObjective, all_indices: List[int]):
        self.objective = objective
        self.all_indices = list(all_indices)
        self.n = len(all_indices)

    def _sample_random_set(self, S: set, delta: float) -> set:
        p_in = (delta + 1) / 2     # inclusion prob for elements in S
        p_out = (1 - delta) / 2    # inclusion prob for elements not in S
        u = np.random.random(self.n)
        result = set()
        for k, i in enumerate(self.all_indices):
            p = p_in if i in S else p_out
            if u[k] < p:
                result.add(i)
        return result

    def _estimate_opt(self, n_trials: int = 5) -> float:
        best = 0.0
        for _ in range(n_trials):
            u = np.random.random(self.n)
            subset = {i for i, ui in zip(self.all_indices, u) if ui < 0.5}
            val = self.objective.evaluate(subset)
            best = max(best, val)
        return best

    def _estimate_omega(
        self,
        rule_idx: int,
        S: set,
        delta: float,
        error_threshold: float = 0.05,
        min_samples: int = 5,
        max_samples: int = 100,
    ) -> float:
        """Estimate E[f(T | {r}) - f(T - {r})] with adaptive sampling."""
        samples = []
        while len(samples) < max_samples:
            T = self._sample_random_set(S, delta)
            omega = (
                self.objective.evaluate(T | {rule_idx})
                - self.objective.evaluate(T - {rule_idx})
            )
            samples.append(omega)
            if len(samples) >= min_samples:
                std = np.std(samples)
                if std == 0 or std / np.sqrt(len(samples)) <= error_threshold:
                    break
        return float(np.mean(samples))

    def _optimize_delta(self, delta: float, delta_prime: float) -> set:
        OPT = self._estimate_opt()
        threshold = 2.0 / max(self.n ** 2, 1) * max(OPT, 0.0)
        S: set = set()
        for r in self.all_indices:
            omega = self._estimate_omega(r, S, delta)
            if omega > threshold:
                S.add(r)
            elif omega < -threshold:
                S.discard(r)
        # delta_prime = -1.0 → sample the complement of S
        return self._sample_random_set(S, delta_prime)

    def _backward_eliminate(self, S: set, n_select: int) -> set:
        S = set(S)
        while len(S) > n_select:
            # Remove the least important element (removal causes smallest decrease)
            to_remove = max(S, key=lambda i: self.objective.evaluate(S - {i}))
            S.remove(to_remove)
        return S

    def optimize(self, n_select: int = None) -> List[int]:
        """
        Run SLS and optionally trim the result to n_select rules.

        Returns:
            List of selected indices into objective.cache.decisions.
        """
        if self.n == 0:
            return []
        S1 = self._optimize_delta(delta=1 / 3, delta_prime=1 / 3)
        S2 = self._optimize_delta(delta=1 / 3, delta_prime=-1.0)
        S = S1 if self.objective.evaluate(S1) >= self.objective.evaluate(S2) else S2
        if n_select is not None and len(S) > n_select:
            S = self._backward_eliminate(S, n_select)
        return list(S)


####################################################################################################


class RandomGreedyOptimizer:
    """
    Randomized greedy for non-monotone submodular maximization (Buchbinder et al. 2014).

    At each of k rounds: compute marginal gains for all unchosen elements, collect the
    top-k candidates (by gain), then add one chosen uniformly at random.  Provides a
    1/e approximation ratio for non-monotone submodular functions with a cardinality
    constraint.

    Args:
        objective:   IDSObjective instance.
        all_indices: List of integer indices into objective.cache.decisions.
    """

    def __init__(self, objective: IDSObjective, all_indices: List[int]):
        self.objective = objective
        self.all_indices = list(all_indices)
        self.n = len(all_indices)

    def optimize(self, n_select: int = None) -> List[int]:
        """
        Run randomized greedy and return selected indices.

        Args:
            n_select: Cardinality budget k. Defaults to len(all_indices).

        Returns:
            List of selected indices into objective.cache.decisions.
        """
        if self.n == 0:
            return []
        k = min(n_select, self.n) if n_select is not None else self.n
        remaining = list(self.all_indices)
        S: set = set()
        current_val = self.objective.evaluate(S)
        for _ in range(k):
            if not remaining:
                break
            gains = [(self.objective.evaluate(S | {u}) - current_val, u) for u in remaining]
            gains.sort(key=lambda x: -x[0])
            top_k = [u for _, u in gains[:k]]
            u_star = top_k[np.random.randint(len(top_k))]
            S.add(u_star)
            remaining.remove(u_star)
            current_val = self.objective.evaluate(S)
        return list(S)


####################################################################################################


class IDSCoordinateAscent:
    """
    Coordinate ascent with ternary search for IDS lambda optimization.

    Optimizes a scoring function over the 7-dimensional lambda space by
    holding 6 lambdas fixed and ternary-searching over the remaining one,
    cycling through all 7 for max_iterations rounds.

    Args:
        func:           Callable(List[float]) -> float. The scoring function.
        ranges:         List of 7 (lo, hi) tuples defining the search space.
        precision:      Ternary search stops when interval width < precision.
        max_iterations: Number of full coordinate-sweep rounds.
    """

    def __init__(
        self,
        func,
        ranges: List[tuple],
        precision: float = 0.001,
        max_iterations: int = 10,
        tol: float = 0.0,
    ):
        self.func = func
        self.ranges = list(ranges)
        self.precision = precision
        self.max_iterations = max_iterations
        self.tol = tol

    @staticmethod
    def _ternary_search(func_1d, lo: float, hi: float, precision: float) -> float:
        while hi - lo > precision:
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            if func_1d(m1) < func_1d(m2):
                lo = m1
            else:
                hi = m2
        return (lo + hi) / 2

    def fit(self) -> List[float]:
        """Run coordinate ascent. Returns list of 7 lambda values."""
        lambdas = [lo + np.random.random() * (hi - lo) for lo, hi in self.ranges]
        best_val = self.func(lambdas)
        best_lambdas = list(lambdas)
        for _ in range(self.max_iterations):
            for j, (lo, hi) in enumerate(self.ranges):
                base = list(lambdas)

                def func_1d(val, j=j, base=base):
                    lam = list(base)
                    lam[j] = val
                    return self.func(lam)

                lambdas[j] = self._ternary_search(func_1d, lo, hi, self.precision)
            val = self.func(lambdas)
            improvement = val - best_val
            if val > best_val:
                best_val = val
                best_lambdas = list(lambdas)
            if improvement < self.tol:
                break
        return best_lambdas


####################################################################################################


class IDS(DecisionSet):
    """
    Interpretable Decision Sets (Lakkaraju et al., KDD 2016), reimplemented
    from scratch using Stochastic Local Search.

    Differences from the pyIDS-backed version:
      - No bin_df required: rules are evaluated directly via Rule.evaluate(X).
      - Accepts any Rule type (decision tree rules, forest rules, CARs, etc.).
      - IDSCoverageCache can be precomputed once and reused across n_select
        values or confidence-sweep subsets via cache.subset(indices).
      - Native n_select cap via backward elimination inside SLS.

    Args:
        rules:                     Candidate rule pool.
        rule_labels:               Optional per-rule cluster labels.
        n_select:                  Maximum number of rules to select. If None,
                                   SLS determines the set size automatically.
        lambdas:                   List of 7 lambda weights. If None, coordinate
                                   ascent is run to find good lambdas.
        lambda_search_dict:        Search space for coordinate ascent. Either a
                                   dict (values are (lo, hi) tuples) or a list
                                   of 7 (lo, hi) tuples. Default: [(0,1)] * 7.
        ternary_search_precision:  Stopping precision for ternary search.
        max_iterations:            Max coordinate-ascent rounds.
        cache:                     Pre-built IDSCoverageCache. Pass this to skip
                                   recomputation when reusing across experiments.
    """

    def __init__(
        self,
        rules: List[Rule] = None,
        rule_labels: List[Set[int]] = None,
        n_select: int = None,
        lambdas: List[float] = None,
        lambda_search_dict=None,
        ternary_search_precision: float = 0.001,
        max_iterations: int = 10,
        cache: IDSCoverageCache = None,
        optimizer: str = 'sls',
    ):
        super().__init__(rules=rules, rule_labels=rule_labels)

        if n_select is not None:
            assert isinstance(n_select, int) and n_select > 0, \
                "n_select must be a positive integer."
        self.n_select = n_select

        if lambdas is not None and len(lambdas) != 7:
            raise ValueError("lambdas must be a list of 7 floats.")
        self.lambdas = list(lambdas) if lambdas is not None else None

        if lambda_search_dict is not None:
            if isinstance(lambda_search_dict, dict):
                self._lambda_ranges = list(lambda_search_dict.values())
            else:
                self._lambda_ranges = list(lambda_search_dict)
        else:
            self._lambda_ranges = [(0.0, 1.0)] * 7

        self.ternary_search_precision = ternary_search_precision
        self.max_iterations = max_iterations
        self.cache = cache

        if optimizer not in ('sls', 'random_greedy'):
            raise ValueError(f"optimizer must be 'sls' or 'random_greedy', got {optimizer!r}.")
        self.optimizer = optimizer

    def _make_optimizer(self, obj: IDSObjective, indices: List[int]):
        if self.optimizer == 'random_greedy':
            return RandomGreedyOptimizer(obj, indices)
        return SLSOptimizer(obj, indices)

    def select(self, X: NDArray, y: List[Set[int]]) -> set:
        y_flat = flatten_labels(y)
        if len(y_flat) != len(y):
            raise ValueError("Each data point must have exactly one label.")

        if self.cache is None:
            cache = IDSCoverageCache()
            cache.compute(list(self.decision_set), X, y_flat)
            self.cache = cache

        cache = self.cache
        N = cache.N

        # Filter to decisions that cover at least one point
        valid_indices = [
            i for i in range(len(cache.decisions))
            if cache.antecedent_masks[i].any()
        ]
        if not valid_indices:
            return set()

        sub_cache = (
            cache.subset(valid_indices)
            if len(valid_indices) < len(cache.decisions)
            else cache
        )
        D = len(sub_cache.decisions)
        M = D

        lambdas = self.lambdas
        if lambdas is None:
            def fmax(lam):
                obj = IDSObjective(lam, sub_cache, N, M)
                opt = self._make_optimizer(obj, list(range(D)))
                selected = opt.optimize(n_select=self.n_select)
                return obj.evaluate(set(selected))

            coord_asc = IDSCoordinateAscent(
                fmax,
                self._lambda_ranges,
                precision=self.ternary_search_precision,
                max_iterations=self.max_iterations,
            )
            lambdas = coord_asc.fit()
            self.lambdas = lambdas

        obj = IDSObjective(lambdas, sub_cache, N, M)
        optimizer = self._make_optimizer(obj, list(range(D)))
        selected_indices = optimizer.optimize(n_select=self.n_select)

        return {sub_cache.decisions[i] for i in selected_indices}

    def get_cache(self) -> IDSCoverageCache:
        """Return the precomputed coverage cache (built during fit)."""
        return self.cache


####################################################################################################
