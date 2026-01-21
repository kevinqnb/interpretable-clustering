import heapq
import numpy as np
from numpy.typing import NDArray
from intercluster import Decision
from intercluster.utils import (
    assignment_to_dict, labels_to_assignment, unique_labels, satisfies_rule, map_rules_to_decisions
)

# Added for simple persistence of decision_info_dict
import pickle
from pathlib import Path
from typing import Any, Union

####################################################################################################


class Objective:
    """
    Base class for a selector, which is used to select rules based on a given objective.

    Args:
        n_select (int): The *maximum* number of rules to select.
        alpha_val (float): A hyperparameter for tuning the size of the selected rules.
            Larger values penalize longer rules more heavily. Defaults to 0.0.
        lambda_val (float): A hyperparameter that controls tradeoff between reward and cost.
            Defaults to None, in which case it may be selected automatically.

    Attrs:
        name (str): Name of the objective.
        data_initialized (bool): Whether the data has been initialized.
        decision_set_initialized (bool): Whether the decision set has been initialized.
    """
    def __init__(
        self,
        n_select : int,
        alpha_val : float = 0.0,
        lambda_val : float = None,
        cluster_centers : NDArray = None,
        weights : NDArray = None,
        selection_algorithm : str = 'distorted-greedy',
    ):
        assert n_select > 0, 'n_select must be positive.'
        self.n_select = n_select

        assert alpha_val >= 0.0, 'alpha_val must be non-negative.'
        self.alpha_val = alpha_val

        assert lambda_val is None or lambda_val >= 0.0, 'lambda_val must be non-negative.'
        self.lambda_val = lambda_val

        assert cluster_centers is None or isinstance(cluster_centers, np.ndarray), \
            'cluster_centers must be a numpy array.'
        if cluster_centers is not None:
            assert len(cluster_centers.shape) == 2, 'cluster_centers must be a 2D array.'
        self.cluster_centers = cluster_centers

        assert weights is None or isinstance(weights, np.ndarray), \
            'weights must be a numpy array.'
        if weights is not None:
            assert len(weights.shape) == 1, 'weights must be a 1D array.'
        self.weights = weights

        assert selection_algorithm in ['distorted-greedy', 'lazy-greedy'], \
            'selection_algorithm must be either "distorted-greedy" or "lazy-greedy".'
        self.selection_algorithm = selection_algorithm

        self.data_initialized = False
        self.decision_set_initialized = False

        self.X = None
        self.y = None
        self.label_set = None
        self.n_labels = 0
        self.cluster_coverage_dict = None
        self.rule_to_decision_dict = None
        self.decision_info_dict = None


    def initialize_data(
        self, 
        X : NDArray,
        y : list[set[int]],
    ):
        """
        Sets the data for the objective.
        """
        assert isinstance(X, np.ndarray), 'X must be a numpy array.'
        assert len(X.shape) == 2, 'X must be a 2D array.'
        assert len(y) == X.shape[0], 'y must have the same number of elements as X has rows.'
        assert all(isinstance(label_set, set) for label_set in y), \
            'Each element of y must be a set of labels.'
        
        if self.weights is None:
            self.weights = np.ones(X.shape[0], dtype = float)
        else:
            assert len(self.weights) == X.shape[0], \
                'weights must have the same length as the number of samples in X.'
            
        self.X = X
        self.y = y
        self.label_set = unique_labels(y)
        self.n_labels = len(self.label_set)
        data_to_cluster_assignment = labels_to_assignment(
            y, n_labels = self.n_labels
        )
        self.cluster_coverage_dict = assignment_to_dict(data_to_cluster_assignment)
        self.data_initialized = True


    def initialize_decision_set(
        self,
        decision_set : set[Decision],
    ):
        """
        Sets the decisions for the objective to select from.
        """
        if self.data_initialized is False:
            raise ValueError('Data must be initialized before decisions.')

        assert isinstance(decision_set, set), 'decision_set must be a set.'
        assert all(isinstance(decision, Decision) for decision in decision_set), \
            'Each element of decision_set must be a Decision.'
        
        decision_labels = unique_labels([{d.label} for d in decision_set])
        if not decision_labels.issubset(self.label_set):
            raise ValueError(
                'Decisions must cover the same labels as the input data.'
            )

        # Isolate unique rules and map them to their decisions.
        self.rule_to_decision_dict = map_rules_to_decisions(decision_set)

        # Track info for each decision.
        if self.decision_info_dict is None:
            self.decision_info_dict = {}
            for decision in decision_set:
                rule_coverage = frozenset(list(satisfies_rule(self.X, decision.rule)))
                coverage_array = np.fromiter(rule_coverage, dtype=np.int64)
                coverage_labels = [self.y[i] for i in rule_coverage]
                rule_cluster_coverage = frozenset(
                    self.cluster_coverage_dict[decision.label].intersection(rule_coverage)
                )
                rule_length = len(decision.rule)

                decision_info = {
                    decision: {
                        'coverage': rule_coverage,
                        'coverage_array': coverage_array,
                        'coverage_labels': coverage_labels,
                        'cluster_coverage': rule_cluster_coverage,
                        'length': rule_length,
                        'label': decision.label
                    }
                }
                rule_cost_alpha_zero = self.cost(decision_info, alpha_val=0.0)
                rule_cost = rule_cost_alpha_zero + self.alpha_val * rule_length

                self.decision_info_dict[decision] = decision_info[decision] | {
                    'cost': rule_cost,
                    'cost_alpha_zero': rule_cost_alpha_zero
                }

        else:
            # While this isn't always necessary, we update costs here in case alpha_val has changed.
            for decision, decision_info in self.decision_info_dict.items():
                rule_cost = decision_info['cost_alpha_zero'] + self.alpha_val * decision_info['length']
                self.decision_info_dict[decision]['cost'] = rule_cost


        self.decision_set_initialized = True
        print(f'Initialized objective with {len(decision_set)} decisions.')


    def set_lambda(self, lambda_val : float = None):
        """
        Sets the lambda value for the objective.

        Args:
            lambda_val (float): The new lambda value.
        """
        if lambda_val is None and not (self.data_initialized and self.decision_set_initialized):
            raise ValueError('Data and decision set must be initialized before setting lambda.')
        elif lambda_val is None:
            lambda_vals = self.compute_lambdas()
            if len(lambda_vals) == 0:
                lambda_val = 0.0
                # No valid ratios found; set lambda to 0.
                print('No valid lambda values found; setting lambda to 0.0 and defaulting to lazy-greedy selection.')
                self.selection_algorithm = 'lazy-greedy'
            elif lambda_vals[0] == np.inf:
                print('All coverage/cost ratios are infinite; setting lambda to 0.0 and defaulting to lazy-greedy selection.')
                lambda_val = 0.0
                self.selection_algorithm = 'lazy-greedy'
            else:
                lambda_val = lambda_vals[0]
    
        self.lambda_val = lambda_val


    def reward(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]],
    ) -> float:
        """
        Computes the reward from the selected rules.

        Args:
            selected_decisions_info (dict[Decision, dict[str, any]]): A dictionary mapping each selected
                decision to its information, including labels, points, coverage, and lengths.
        Returns:
            reward (float): The reward from the selected decisions.
        """
        pass


    def cost(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]],
        alpha_val : float = None,
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_decisions_info (dict[Decision, dict[str, any]]): A dictionary mapping each selected
                decision to its information, including labels, points, coverage, and lengths.
        Returns:
            cost (float): The cost of the selected decisions.
        """
        pass


    def compute_objective(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]],
    ) -> float:
        """
        Computes the objective value for the selected decisions.

        Args:
            selected_decisions_info (dict[Decision, dict[str, any]]): A dictionary mapping each selected
                decision to its information, including labels, points, coverage, and lengths.
        Returns:
            objective (float): The objective value for the selected decisions.
        """
        g = self.reward(selected_decisions_info)
        h = self.cost(selected_decisions_info)
        return g - self.lambda_val * h
    

    def compute_lambdas(self) -> NDArray:
        """
        Computes minimum value of lambda necessary for an approximation algorithm.

        Args:
            data (NDArray): (n x d) Data array.
            data_to_cluster_assignment (np.ndarray): Size (n x k) boolean array where entry (i,j) is 
                `True` if point i is assigned to cluster j and `False` otherwise. Data points may be 
                assigned to multiple clusters. 
            data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is `True` if 
                data point i is assigned to rule j and `False` otherwise.
            rule_lengths (list[int]): A list of lengths for each rule.
                
        Returns:
            lambda_vals (NDArray): A sorted array of lambda values, starting from the minimum 
                most value for which the approximation guarantee holds, and increasing
                until reaching the maximum coverage/cost ratio seen
                for any (rule, cluster) assignment pair. 
        """
        ratios = []
        second_max_ratio = 0.0
        for rule in self.rule_to_decision_dict.keys():
            max_rule_ratio = 0.0
            second_max_rule_ratio = 0.0
            for decision in self.rule_to_decision_dict[rule]:
                decision_info = self.decision_info_dict[decision]

                r_coverage = decision_info['coverage']
                r_cluster_coverage = decision_info['cluster_coverage']
                r_length = decision_info['length']
                d_label = decision_info['label']
                d_cost = decision_info['cost']
                h = d_cost

                if h > 0 and not np.isnan(h):
                    d_info = {
                        decision: {
                            'coverage': r_coverage,
                            'coverage_array': np.fromiter(r_coverage, dtype=np.int64),
                            'cluster_coverage': r_cluster_coverage,
                            'length': r_length,
                            'label': d_label,
                            'cost': d_cost
                        }
                    }
                    g = self.reward(d_info)
                    ratio = g / h
                else:
                    ratio = np.inf

                if ratio > max_rule_ratio:
                    second_max_rule_ratio = max_rule_ratio
                    max_rule_ratio = ratio
                elif ratio > second_max_rule_ratio:
                    second_max_rule_ratio = ratio

            ratios.append(max_rule_ratio)
            if second_max_rule_ratio > second_max_ratio:
                second_max_ratio = second_max_rule_ratio
                    
        ratios = [r for r in ratios if r >= second_max_ratio]
        if second_max_ratio == 0.0:
            return np.sort(ratios)
        return np.sort(ratios + [second_max_ratio])


    def marginal_reward(
        self,
        decision_info: dict[str, any],
        total_coverage : set[int],
        cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal reward from selected decision.

        Args:
            decision_info (dict): A dictionary containing information about the decision being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected decisions.
        
        Returns:
            coverage (float): The coverage of the selected decisions.
        """
        pass


    def distorted_greedy_select(
        self,
    ) -> set[Decision]:
        """
        Selects a subset rules using a distorted greedy algorithm. For more information 
        on the algorithm, see the following paper:
        "Submodular Maximization Beyond Non-Negativity: Guarantees, Fast Algorithms, and Applications"
        by Harshaw el al., ICML 2019.
        Args:
            
        Returns:
            decision_set (Set[Decision]): The selected set of decisions.
        """
        if not (self.data_initialized and self.decision_set_initialized):
            raise ValueError('Data and decisions must be initialized before selection.')

        total_coverage = set()
        total_cluster_coverage = {l: set() for l in range(self.n_labels)}
        selected_decisions = set()
        discarded_decisions = set()
        for i in range(self.n_select):
            best_decision = None
            best_decision_score = 0.0

            # NOTE: Should this iterate over decisions in a sorted order?
            for decision, decision_info in self.decision_info_dict.items():
                if (decision not in selected_decisions) and (decision not in discarded_decisions):
                    g = self.marginal_reward(
                        decision_info,
                        total_coverage,
                        total_cluster_coverage
                    )

                    h = decision_info['cost']

                    # Early discard since the marginal reward will only decrease from here on out, 
                    # and its score coefficient will be at most 1.
                    # Therefore if g - lambda * c <= 0, the score will never be positive, 
                    # and it can never be selected.
                    if g - self.lambda_val * h <= 0:
                        discarded_decisions.add(decision)
                    
                    score = (1 - 1/self.n_select)**(self.n_select - (i + 1)) * g - self.lambda_val * h

                    if score > best_decision_score:
                        best_decision = decision
                        best_decision_score = score

            if best_decision_score > 0:
                selected_decisions.add(best_decision)
                best_decision_label = self.decision_info_dict[best_decision]['label']
                best_decision_coverage = self.decision_info_dict[best_decision]['coverage']
                best_decision_cluster_coverage = self.decision_info_dict[best_decision]['cluster_coverage']
                total_cluster_coverage[best_decision_label] = total_cluster_coverage[
                    best_decision_label
                ].union(
                    best_decision_cluster_coverage
                )
                total_coverage = total_coverage.union(best_decision_coverage)

        # Compute final objective value (pass defensive copies to avoid accidental mutation).
        selected_info = {
            decision: dict(self.decision_info_dict[decision]) for decision in selected_decisions
        }
        self.reward_value = self.reward(selected_info)
        self.cost_value = self.cost(selected_info)
        self.objective_value = self.compute_objective(selected_info)
        return selected_decisions
    

    def lazy_greedy_select(
        self,
    ) -> set[Decision]:
        """
        Selects a subset rules using a lazy greedy algorithm. 

        Args:
            
        Returns:
            decision_set (Set[Decision]): The selected set of decisions.
        """
        if not (self.data_initialized and self.decision_set_initialized):
            raise ValueError('Data and decisions must be initialized before selection.')

        total_coverage = set()
        total_cluster_coverage = {l: set() for l in range(self.n_labels)}
        eligible_decisions = set(self.decision_info_dict.keys())
        selected_decisions = set()
        selected_rules = set()

        # Initialize heap
        heap = []
        counter = 0  # tie-breaker so heap never compares Decision objects
        for decision in eligible_decisions:
            decision_info = self.decision_info_dict[decision]
            g = self.marginal_reward(decision_info, total_coverage, total_cluster_coverage)
            h = decision_info["cost"]

            # Optional: early discard (same logic you already have)
            score = g - 2 * self.lambda_val * h
            if score <= 0:
                continue

            heap.append((-score, counter, decision))
            counter += 1

        heapq.heapify(heap)

        # If everything was filtered out during initialization, exit early.
        if not heap:
            self.reward_value = 0.0
            self.cost_value = 0.0
            self.objective_value = 0.0
            return set()

        while heap and len(eligible_decisions) > 0:
            best_decision = None
            best_decision_score = 0.0
            removals = set()

            heap_best_score, _, heap_best_decision = heapq.heappop(heap)
            second_best_score, _, second_best_decision = heap[0] if len(heap) > 0 else (float('-inf'), None, None)

            # Recompute marginal reward for the top decision
            decision_info = self.decision_info_dict[heap_best_decision]
            g = self.marginal_reward(
                decision_info,
                total_coverage,
                total_cluster_coverage
            )
            h = decision_info['cost']

            score = g - 2 * self.lambda_val * h

            if second_best_decision is None or score >= -second_best_score:
                best_decision = heap_best_decision
                best_decision_score = score

                if best_decision_score > 0:
                    selected_decisions.add(best_decision)
                    selected_rules.add(best_decision.rule)
                    best_decision_label = self.decision_info_dict[best_decision]['label']
                    best_decision_coverage = self.decision_info_dict[best_decision]['coverage']
                    best_decision_cluster_coverage = self.decision_info_dict[best_decision]['cluster_coverage']
                    total_cluster_coverage[best_decision_label] = total_cluster_coverage[
                        best_decision_label
                    ].union(
                        best_decision_cluster_coverage
                    )
                    total_coverage = total_coverage.union(best_decision_coverage)

                    # Update eligible decisions with matroid constraints:
                    if len(selected_decisions) >= self.n_select:
                        removals = eligible_decisions.copy()
                    else:
                        for decision in eligible_decisions.copy():
                            if decision.rule in selected_rules:
                                removals.add(decision)

                else:
                    removals.add(best_decision)

            else:
                # Reinsert with updated score (always use an integer counter to avoid comparing Decisions)
                heapq.heappush(heap, (-score, counter, heap_best_decision))
                counter += 1

            eligible_decisions.difference_update(removals)

        # Compute final objective value (pass defensive copies to avoid accidental mutation).
        selected_info = {
            decision: dict(self.decision_info_dict[decision]) for decision in selected_decisions
        }
        self.reward_value = self.reward(selected_info)
        self.cost_value = self.cost(selected_info)
        self.objective_value = self.compute_objective(selected_info)
        return selected_decisions


    def select(
        self,
    ) -> set[Decision]:
        """
        Selects a subset rules using the specified selection algorithm.

        Args:
            
        Returns:
            decision_set (Set[Decision]): The selected set of decisions.
        """
        if self.selection_algorithm == 'distorted-greedy':
            return self.distorted_greedy_select()
        elif self.selection_algorithm == 'lazy-greedy':
            return self.lazy_greedy_select()
        else:
            raise ValueError(
                f'Unknown selection algorithm: {self.selection_algorithm}'
            )

    def save_decision_info_dict(
        self,
        path: Union[str, Path],
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> None:
        """Save the precomputed ``decision_info_dict`` to disk using pickle.

        Notes:
            - Pickle is Python-specific and **must not** be used with untrusted files.
            - This persists ``Decision`` objects as dict keys. This works because `Decision` is a
              frozen dataclass and (in your codebase) is pickleable.

        Args:
            path: Output file path (e.g. "decision_info.pkl").
            protocol: Pickle protocol; defaults to highest available.
        """
        if not self.decision_set_initialized or self.decision_info_dict is None:
            raise ValueError("Decision set must be initialized before saving decision_info_dict.")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Normalize a few fields so the pickle is smaller/more robust.
        # (Sets -> sorted lists; numpy arrays -> numpy arrays are pickleable but normalize anyway.)
        serializable: dict[Decision, dict[str, Any]] = {}
        for decision, info in self.decision_info_dict.items():
            if not isinstance(decision, Decision):
                raise TypeError("decision_info_dict keys must be Decision objects")
            if not isinstance(info, dict):
                raise TypeError("decision_info_dict values must be dicts")

            out: dict[str, Any] = dict(info)

            # Ensure coverage is saved deterministically.
            if "coverage" in out and isinstance(out["coverage"], set):
                out["coverage"] = sorted(out["coverage"])

            if "cluster_coverage" in out and isinstance(out["cluster_coverage"], set):
                out["cluster_coverage"] = sorted(out["cluster_coverage"])

            # coverage_array is redundant; can be rebuilt from coverage. Keep it if present,
            # but ensure it's a numpy array (pickleable).
            if "coverage_array" in out and out["coverage_array"] is not None:
                out["coverage_array"] = np.asarray(out["coverage_array"], dtype=np.int64)

            # Storing coverage_labels makes the file large; keep only if the user already computed it.
            # (It can be recomputed from self.y if needed.)
            serializable[decision] = out

        with p.open("wb") as f:
            pickle.dump(serializable, f, protocol=protocol)

    def load_decision_info_dict(
        self,
        path: Union[str, Path],
    ) -> dict[Decision, dict[str, Any]]:
        """Load a previously saved ``decision_info_dict`` from disk.

        This is the inverse of :meth:`save_decision_info_dict`. It loads the pickled mapping and
        normalizes a few fields back to the in-memory representation expected by selection code:

        - ``coverage`` and ``cluster_coverage`` are converted to ``set[int]`` if they were saved as lists.
        - ``coverage_array`` is rebuilt from ``coverage`` if missing.

        Notes:
            Only load pickle files you created yourself (or otherwise trust).

        Args:
            path: Input file path (e.g. "decision_info.pkl").

        Returns:
            The loaded decision_info_dict (also stored on ``self.decision_info_dict``).
        """
        p = Path(path)
        with p.open("rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, dict):
            raise TypeError("Loaded object is not a dict; expected a decision_info_dict")

        loaded: dict[Decision, dict[str, Any]] = {}
        for decision, info in obj.items():
            if not isinstance(decision, Decision):
                raise TypeError("Loaded decision_info_dict contains a non-Decision key")
            if not isinstance(info, dict):
                raise TypeError("Loaded decision_info_dict contains a non-dict value")

            out: dict[str, Any] = dict(info)

            # Restore sets as *immutable* frozenset to prevent accidental mutation.
            if "coverage" in out and isinstance(out["coverage"], (list, tuple, set, frozenset)):
                out["coverage"] = frozenset(out["coverage"])
            if "cluster_coverage" in out and isinstance(out["cluster_coverage"], (list, tuple, set, frozenset)):
                out["cluster_coverage"] = frozenset(out["cluster_coverage"])

            # Ensure coverage_array exists and is the expected dtype.
            if out.get("coverage_array") is None:
                if "coverage" in out and isinstance(out["coverage"], frozenset):
                    out["coverage_array"] = np.fromiter(out["coverage"], dtype=np.int64)
            else:
                out["coverage_array"] = np.asarray(out["coverage_array"], dtype=np.int64)

            loaded[decision] = out

        self.decision_info_dict = loaded
        return loaded

####################################################################################################