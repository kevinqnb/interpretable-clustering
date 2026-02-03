import heapq
from itertools import combinations
import numpy as np
from numpy.typing import NDArray
from intercluster import (
    Decision,
    simplified_rule_length
)
from intercluster.utils import (
    labels_to_assignment,
    unique_labels,
    satisfies_rule,
    map_rules_to_decisions,
    _pack_bool_matrix,
    _unpack_bool_matrix,
)

# Added for simple persistence of decision_info_dict
import pickle
from pathlib import Path
from typing import Any, Union

# Added for more memory-efficient persistence
import gzip

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
        cluster_centers (NDArray): (k x d) Array of cluster centers for computing coverage.
        weights (NDArray): (n,) Array of weights for each data point. Defaults to None,
        selection_algorithm (str): The selection algorithm to use. Options are
            'distorted-greedy' and 'lazy-greedy'. Defaults to 'distorted-greedy'.
        precomputed_path (Union[str, Path]): Path to precomputed data for the objective. Defaults to None.
        output_path (Union[str, Path]): Path to save output data. Defaults to None.
        pack_bits (bool): Whether to pack boolean matrices as bit vectors for memory efficiency. Defaults to True.

    Attrs:
        name (str): Name of the objective.
        data_initialized (bool): Whether the data has been initialized.
        decision_set_initialized (bool): Whether the decision set has been initialized.
    """
    def __init__(
        self,
        n_select: int = 0,
        alpha_val: float = 0.0,
        lambda_val: float = None,
        cluster_centers: NDArray = None,
        weights: NDArray = None,
        selection_algorithm: str = 'distorted-greedy',
        precomputed_path: Union[str, Path] = None,
        output_path: Union[str, Path] = None,
        pack_bits: bool = True,
    ):
        assert n_select > 0, 'n_select must be given a positive value as input.'
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

        self.pack_bits = pack_bits

        self.data_initialized = False
        self.decision_set_initialized = False

        self.X = None
        self.y = None
        self.label_set = None
        self.n_labels = 0
        self.rule_to_decision_dict = None

        # New storage:
        # - rule_coverage_packed: (R, ceil(N/8)) uint8
        # - cluster_membership_packed: (k, ceil(N/8)) uint8
        # - decision_info_dict: {Decision: {'coverage_idx': int, 'label': int, 'length': int, ...}}
        self.rule_coverage_packed: np.ndarray | None = None
        self.cluster_membership_packed: np.ndarray | None = None
        self.n_rules: int = 0
        self.n_samples: int = 0
        self.decision_info_dict: dict[Decision, dict[str, Any]] | None = None
        self.data_to_center_distances: NDArray | None = None

        if precomputed_path is not None:
            self.load_precomputed(precomputed_path)
            self.precomputed = True
        else:
            self.precomputed = False
        
        assert output_path is None or isinstance(output_path, (str, Path)), \
            'output_path must be a string or Path.'
        self.output_path = output_path


    def initialize_data(
        self,
        X: NDArray,
        y: list[set[int]],
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
        self.n_samples = X.shape[0]

        self.label_set = unique_labels(y)
        self.n_labels = len(self.label_set)

        if not self.precomputed:
            cluster_membership = labels_to_assignment(
                y, n_labels=self.n_labels
            ).T
            self.cluster_membership_packed = _pack_bool_matrix(cluster_membership) if self.pack_bits else cluster_membership

        self.data_initialized = True


    def _iter_covered_indices_from_rule_idx(self, rule_idx: int):
        """Iterate covered sample indices for a rule (used for label reconstruction).

        This avoids storing per-decision coverage lists. It unpacks only the selected rule.
        """
        if self.rule_coverage_packed is None:
            return iter(())

        if self.pack_bits:
            row = self.rule_coverage_packed[rule_idx:rule_idx + 1]
            bits = _unpack_bool_matrix(row, self.n_samples)[0]
            return np.flatnonzero(bits)

        # Unpacked bool matrix
        return np.flatnonzero(self.rule_coverage_packed[rule_idx])


    def get_coverage_labels(self, decision: Decision):
        """Reconstruct coverage labels for a decision efficiently.

        Returns a list[set[int]] matching the old structure, but computed on-demand.
        """
        if self.y is None or self.decision_info_dict is None:
            raise ValueError(
                'Data and decision_info_dict must be initialized before computing coverage labels.'
            )
        info = self.decision_info_dict[decision]
        ridx = int(info['coverage_idx'])
        idxs = self._iter_covered_indices_from_rule_idx(ridx)
        return [self.y[int(i)] for i in idxs]
    

    def initialize_decision_set(
        self,
        decision_set: set[Decision],
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

        if not self.precomputed:
            # Map unique rules -> decisions
            self.rule_to_decision_dict = map_rules_to_decisions(decision_set)

            # Assign each unique rule a stable index
            rules = list(self.rule_to_decision_dict.keys())
            rule_to_idx = {rule: i for i, rule in enumerate(rules)}
            self.n_rules = len(rules)

            # Precompute rule coverage matrix (R, N) bool (then pack bits if enabled)
            rule_cov_bool = np.zeros((self.n_rules, self.n_samples), dtype=np.bool_)
            for rule, idx in rule_to_idx.items():
                covered = satisfies_rule(self.X, rule)
                # covered may be list/iterable of indices
                covered_arr = np.fromiter(covered, dtype=np.int64)
                if covered_arr.size > 0:
                    rule_cov_bool[idx, covered_arr] = True

            self.rule_coverage_packed = _pack_bool_matrix(rule_cov_bool) if self.pack_bits else rule_cov_bool

            self.decision_info_dict = {}
            for decision in decision_set:
                ridx = int(rule_to_idx[decision.rule])
                #rule_length = len(decision.rule)
                rule_length = simplified_rule_length(decision.rule)

                # Compute cost_alpha_zero using a minimal info dict understandable by subclasses.
                # Subclasses that require coverage should use coverage_idx + rule_coverage_packed.
                minimal_info = {
                    decision: {
                        'coverage_idx': ridx,
                        'label': decision.label,
                        'length': rule_length,
                    }
                }
                rule_cost_alpha_zero = self.cost(minimal_info, alpha_val=0.0)
                rule_cost = rule_cost_alpha_zero + self.alpha_val * rule_length

                self.decision_info_dict[decision] = {
                    'coverage_idx': ridx,
                    'label': decision.label,
                    'length': rule_length,
                    'cost': float(rule_cost),
                    'cost_alpha_zero': float(rule_cost_alpha_zero),
                }
            print(f'Initialized objective with {len(decision_set)} decisions.')
        else:
            # Update costs in case alpha_val changed.
            for decision, info in self.decision_info_dict.items():
                rule_cost = info['cost_alpha_zero'] + self.alpha_val * info['length']
                info['cost'] = float(rule_cost)
            print(f'Updated costs for {len(self.decision_info_dict)} precomputed decisions.')

        self.decision_set_initialized = True

        if self.output_path is not None:
            self.save_precomputed(self.output_path)


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
                print('No valid lambda values found; setting lambda to 0.0 and defaulting to lazy-greedy selection.')
                lambda_val = 0.0
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


    def marginal_reward(
        self,
        decision_info: dict[str, any],
        total_coverage : NDArray,
        cluster_coverage : NDArray
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

                h = decision_info['cost']

                if h > 0 and not np.isnan(h):
                    d_info = {
                        decision: {
                            'coverage_idx': decision_info['coverage_idx'],
                            'label': decision_info['label'],
                            'length': decision_info['length'],
                            'cost': decision_info['cost'],
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

        # Track covered sets as packed bit vectors to speed marginal computations in subclasses.
        if self.pack_bits:
            covered_total = np.zeros((1, self.rule_coverage_packed.shape[1]), dtype=np.uint8)
            covered_by_cluster = np.zeros((self.n_labels, self.cluster_membership_packed.shape[1]), dtype=np.uint8)
        else:
            covered_total = np.zeros((self.n_samples,), dtype=np.bool_)
            covered_by_cluster = np.zeros((self.n_labels, self.n_samples), dtype=np.bool_)

        selected_decisions: set[Decision] = set()
        discarded_decisions: set[Decision] = set()

        for i in range(self.n_select):
            best_decision = None
            best_decision_score = 0.0

            for decision, decision_info in self.decision_info_dict.items():
                if (decision in selected_decisions) or (decision in discarded_decisions):
                    continue

                g = self.marginal_reward(decision_info, covered_total, covered_by_cluster)
                h = decision_info['cost']

                if g - self.lambda_val * h <= 0:
                    discarded_decisions.add(decision)

                score = (1 - 1 / self.n_select) ** (self.n_select - (i + 1)) * g - self.lambda_val * h
                if score > best_decision_score:
                    best_decision = decision
                    best_decision_score = score

            if best_decision_score > 0 and best_decision is not None:
                selected_decisions.add(best_decision)

                info = self.decision_info_dict[best_decision]
                ridx = int(info['coverage_idx'])
                lbl = int(info['label'])

                if self.pack_bits:
                    rule_bits = self.rule_coverage_packed[ridx:ridx + 1]
                    cluster_bits = self.cluster_membership_packed[lbl:lbl + 1]
                    new_cluster_bits = np.bitwise_and(rule_bits, cluster_bits)
                    covered_by_cluster[lbl:lbl + 1] = np.bitwise_or(covered_by_cluster[lbl:lbl + 1], new_cluster_bits)
                    covered_total = np.bitwise_or(covered_total, rule_bits)
                else:
                    rule_mask = self.rule_coverage_packed[ridx]
                    cluster_mask = self.cluster_membership_packed[lbl]
                    covered_by_cluster[lbl] |= (rule_mask & cluster_mask)
                    covered_total |= rule_mask

        selected_info = {d: dict(self.decision_info_dict[d]) for d in selected_decisions}
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

        if self.pack_bits:
            covered_total = np.zeros((1, self.rule_coverage_packed.shape[1]), dtype=np.uint8)
            covered_by_cluster = np.zeros((self.n_labels, self.cluster_membership_packed.shape[1]), dtype=np.uint8)
        else:
            covered_total = np.zeros((self.n_samples,), dtype=np.bool_)
            covered_by_cluster = np.zeros((self.n_labels, self.n_samples), dtype=np.bool_)

        eligible_decisions = set(self.decision_info_dict.keys())
        selected_decisions: set[Decision] = set()
        selected_rule_idxs: set[int] = set()

        heap = []
        counter = 0
        for decision in eligible_decisions:
            info = self.decision_info_dict[decision]
            g = self.marginal_reward(info, covered_total, covered_by_cluster)
            h = info['cost']
            score = g - 2 * self.lambda_val * h
            if score <= 0:
                continue
            heap.append((-score, counter, decision))
            counter += 1

        heapq.heapify(heap)
        if not heap:
            self.reward_value = 0.0
            self.cost_value = 0.0
            self.objective_value = 0.0
            return set()

        while heap and len(eligible_decisions) > 0:
            heap_best_score, _, heap_best_decision = heapq.heappop(heap)
            second_best_score = heap[0][0] if len(heap) > 0 else 0.0

            info = self.decision_info_dict[heap_best_decision]
            g = self.marginal_reward(info, covered_total, covered_by_cluster)
            h = info['cost']
            score = g - 2 * self.lambda_val * h

            if score >= -second_best_score:
                if score > 0:
                    selected_decisions.add(heap_best_decision)
                    ridx = int(info['coverage_idx'])
                    lbl = int(info['label'])
                    selected_rule_idxs.add(ridx)

                    if self.pack_bits:
                        rule_bits = self.rule_coverage_packed[ridx:ridx + 1]
                        cluster_bits = self.cluster_membership_packed[lbl:lbl + 1]
                        new_cluster_bits = np.bitwise_and(rule_bits, cluster_bits)
                        covered_by_cluster[lbl:lbl + 1] = np.bitwise_or(covered_by_cluster[lbl:lbl + 1], new_cluster_bits)
                        covered_total = np.bitwise_or(covered_total, rule_bits)
                    else:
                        rule_mask = self.rule_coverage_packed[ridx]
                        cluster_mask = self.cluster_membership_packed[lbl]
                        covered_by_cluster[lbl] |= (rule_mask & cluster_mask)
                        covered_total |= rule_mask

                    # Matroid-like constraint: at most one decision per rule
                    removals = set()
                    if len(selected_decisions) >= self.n_select:
                        removals = eligible_decisions.copy()
                    else:
                        for d in eligible_decisions:
                            if int(self.decision_info_dict[d]['coverage_idx']) in selected_rule_idxs:
                                removals.add(d)
                    eligible_decisions.difference_update(removals)
                else:
                    eligible_decisions.discard(heap_best_decision)
            else:
                heapq.heappush(heap, (-score, counter, heap_best_decision))
                counter += 1

        selected_info = {d: dict(self.decision_info_dict[d]) for d in selected_decisions}
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

    def save_precomputed(self, path: Union[str, Path], *, compress: bool = True) -> None:
        """Save precomputed arrays and decision_info_dict.

        The output is a single gzip-compressed pickle by default containing:
        - metadata (n_samples, n_labels, pack_bits)
        - rule_coverage (packed or unpacked)
        - cluster_membership (packed or unpacked)
        - rule_to_decision_dict (for lambda computations)
        - decision_info_dict (lightweight)
        """
        if not self.decision_set_initialized:
            raise ValueError('Decision set must be initialized before saving precomputed data.')

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        blob = {
            'n_samples': self.n_samples,
            'n_labels': self.n_labels,
            'pack_bits': self.pack_bits,
            'rule_coverage': self.rule_coverage_packed,
            'cluster_membership': self.cluster_membership_packed,
            'rule_to_decision_dict': self.rule_to_decision_dict,
            'decision_info_dict': self.decision_info_dict,
            'data_to_center_distances': self.data_to_center_distances,
        }

        opener = gzip.open if compress else Path.open
        with opener(p, 'wb') as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)


    def load_precomputed(self, path: Union[str, Path]) -> None:
        """Load previously saved precomputed arrays and decision_info_dict."""
        p = Path(path)

        def _open_for_read(pp: Path):
            with pp.open('rb') as raw:
                head = raw.read(2)
            if head == b"\x1f\x8b":
                return gzip.open(pp, 'rb')
            return pp.open('rb')

        with _open_for_read(p) as f:
            blob = pickle.load(f)

        if not isinstance(blob, dict):
            raise TypeError('Loaded object is not a dict; expected precomputed bundle.')

        self.n_samples = int(blob['n_samples'])
        self.n_labels = int(blob['n_labels'])
        self.pack_bits = bool(blob['pack_bits'])

        self.rule_coverage_packed = blob['rule_coverage']
        self.cluster_membership_packed = blob['cluster_membership']
        self.rule_to_decision_dict = blob.get('rule_to_decision_dict')
        self.decision_info_dict = blob.get('decision_info_dict')
        self.data_to_center_distances = blob.get('data_to_center_distances')

        #self.decision_set_initialized = self.decision_info_dict is not None and self.rule_coverage_packed is not None

####################################################################################################