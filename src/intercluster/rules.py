import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy.typing import NDArray
from typing import Callable, List, Optional, Dict, Tuple
from dataclasses import dataclass

# Added for simple persistence of rules/decisions
import pickle
from pathlib import Path
from typing import Iterable, Sequence, Union, Any


####################################################################################################


class Condition():
    """
    Base class for a rule-based condition
    """
    
    def evaluate(self, X : NDArray) -> NDArray:
        """
        Evaluates the condition upon a given subset of data.
        
        Args:
            X (np.ndarray): Size n x d dataset for evaluation. May be either a single 
                data point or a 2d array.
        
        Returns:
            (np.ndarray): Length n boolean array with entry i being `True` if point i satisfies
                the condition, and `False` otherwise.
        """
        raise NotImplementedError("Subclasses must implement evaluate method")


####################################################################################################


@dataclass(frozen=True)
class Rule():
    """
    Object for a rule, which is a conjunction of conditions.
    
    Args:
        conditions (List[Condition]): List of conditions that make up the rule.
    """
    conditions: List[Condition]
    
    def __post_init__(self):
        """
        Converts conditions to tuple and caches hash for efficient operations.
        """
        # Convert list to tuple for immutability and efficient hashing
        object.__setattr__(self, 'conditions', tuple(self.conditions))
        
        # Cache hash value
        object.__setattr__(self, '_hash', hash(self.conditions))
        
        
    def evaluate(self, X : NDArray) -> NDArray:
        """
        Evaluates the rule upon a given subset of data.
        
        Args:
            X (np.ndarray): Size n x d dataset for evaluation. May be either a single 
                data point or a 2d array.
                
        Returns:
            (np.ndarray): Length n boolean array with entry i being `True` if point i satisfies
                the rule, and `False` otherwise.
        """        
        if len(self.conditions) == 0:
            return np.array([True]*X.shape[0])
        
        evals = self.conditions[0].evaluate(X)
        for cond in self.conditions[1:]:
            evals = np.logical_and(evals, cond.evaluate(X))
            
        return evals
    
    def __len__(self):
        """
        Returns the number of conditions in the rule.
        
        Returns:
            (int): Number of conditions in the rule.
        """
        return len(self.conditions)
    
    def __hash__(self):
        """
        Returns a hash value for the rule based on its conditions.
        Uses cached hash value computed during initialization.
        
        Returns:
            (int): Hash value for the rule.
        """
        return self._hash
    
    def __eq__(self, other):
        """
        Checks equality between two rules.
        
        Args:
            other (Rule): Another rule to compare with.
            
        Returns:
            (bool): True if the rules have the same conditions, False otherwise.
        """
        if not isinstance(other, Rule):
            return False
        # Fast path: if same object, return True
        if self is other:
            return True
        # Fast path: compare cached hashes first (if available)
        if hasattr(self, '_hash') and hasattr(other, '_hash'):
            if self._hash != other._hash:
                return False
            # Hashes match - very likely equal, but verify to handle rare collisions
            # For identical objects this is nearly free since conditions is a tuple
            return self.conditions == other.conditions
        return self.conditions == other.conditions


####################################################################################################


@dataclass(frozen=True)
class Decision():
    """
    Object for a decision, which is a rule along with its predicted class label.

    Args:
        rule (Rule): The rule associated with the decision.
        
        label (int): The predicted class label for the decision.
    """
    rule: Rule
    label: int
    
    def __post_init__(self):
        """
        Caches hash for efficient dictionary/set operations.
        """
        object.__setattr__(self, '_hash', hash((self.rule, self.label)))
    
    def __hash__(self):
        """
        Returns a hash value for the decision based on its rule and label.
        Uses cached hash value computed during initialization.
        
        Returns:
            (int): Hash value for the decision.
        """
        return self._hash
    
    def __eq__(self, other):
        """
        Checks equality between two decisions.
        
        Args:
            other (Decision): Another decision to compare with.
            
        Returns:
            (bool): True if the decisions have the same rule and label, False otherwise.
        """
        if not isinstance(other, Decision):
            return False
        # Fast path: if same object, return True
        if self is other:
            return True
        # Fast path: compare cached hashes first (if available)
        if hasattr(self, '_hash') and hasattr(other, '_hash'):
            if self._hash != other._hash:
                return False
            # Hashes match - check cheap comparison (label) first before expensive rule comparison
            return self.label == other.label and self.rule == other.rule
        # Fallback: check cheap comparison first
        return self.label == other.label and self.rule == other.rule


####################################################################################################


@dataclass(frozen=True)
class LinearCondition(Condition):
    """
    Object for a linear splitting condition, either axis aligned
    or oblique. 
    
    Args:
        features (np.ndarray): The chosen features to split on. 
        
        weights (np.ndarray): Weights for each of the splitting features.
        
        threshold (float): The threshold value to split on.
        
        direction (int): Specifies the direction for the inequality. 
            Use -1 for less than or equal (<=) or 1 for greater (>). 
            Defaults to -1.
    """
    features: NDArray
    weights: NDArray
    threshold: float
    direction: int = -1
    
    def __post_init__(self):
        """
        Validates and converts features and weights to numpy arrays after initialization.
        """
        # Convert to numpy arrays if they aren't already
        object.__setattr__(self, 'features', np.array(self.features))
        object.__setattr__(self, 'weights', np.array(self.weights))
        
        # Validate direction
        if self.direction not in {1, -1}:
            raise ValueError("Invalid inequality direction, must be -1 (<=) or 1 (>).")
        
        # Cache hash value for efficient dictionary/set operations
        hash_val = hash((
            tuple(self.features),  # No need for flatten() on 1D arrays
            tuple(self.weights),
            self.threshold,
            self.direction
        ))
        object.__setattr__(self, '_hash', hash_val)
        

    def __hash__(self):
        """
        Returns a hash value for the condition based on its parameters.
        Uses cached hash value computed during initialization.
        
        Returns:
            (int): Hash value for the condition.
        """
        return self._hash
    
    def __eq__(self, other):
        """
        Checks equality between two linear conditions.
        Custom equality needed because numpy arrays use element-wise comparison.
        
        Args:
            other (LinearCondition): Another condition to compare with.
            
        Returns:
            (bool): True if the conditions have the same parameters, False otherwise.
        """
        if not isinstance(other, LinearCondition):
            return False
        # Fast path: if same object, return True
        if self is other:
            return True
        # Fast path: compare cached hashes first (if available)
        if hasattr(self, '_hash') and hasattr(other, '_hash'):
            if self._hash != other._hash:
                return False
            # Hashes match - check cheap comparisons first before expensive numpy array equality
            if self.threshold != other.threshold or self.direction != other.direction:
                return False
            return (
                np.array_equal(self.features, other.features) and
                np.array_equal(self.weights, other.weights)
            )
        # Fallback: check cheap comparisons first, then expensive ones
        if self.threshold != other.threshold or self.direction != other.direction:
            return False
        return (
            np.array_equal(self.features, other.features) and
            np.array_equal(self.weights, other.weights)
        )
        
    
    def evaluate(self, X : NDArray) -> NDArray:
        """
        Evaluates the linear condition upon a given subset of data.
        
        Args:
            X (np.ndarray): Size n x d dataset for evaluation. May be either a single 
                data point or a 2d array.
                
        Returns:
            (np.ndarray): Length n boolean array with entry i being `True` if point i satisfies
                the condition, and `False` otherwise.
        """        
        features_needed = np.max(self.features)
        if len(X.shape) < 2:
            raise ValueError("Data must be two dimensional.")
            
        if X.shape[1] < features_needed:
            raise ValueError("Shape of data does not match the number of features required.")
        
        evals = np.sign(np.dot(X[:,self.features], self.weights) - self.threshold)
        evals[evals == 0] = -1
        
        if self.direction is not None:
            return evals == self.direction
        
        return evals
    

    def display(
            self,
            feature_labels : List[str] = None,
            scaler : Callable = None,
            newline : bool = True
        ) -> str:
        """
        Displays the condition by returning a string representation.

        Args:
            feature_labels (List[str], optional): List of feature labels used for display.
                The feature at index i should correspond to feature i in the dataset.  
                Defaults to None, in which case conditions will be plotted as is.

            scaler (Callable): Sklearn data scaler, which will be used to convert
                thresholds, weights back to their unscaled versions (better interpretability).
                This current supports the StandardScaler or the MinMaxScaler. Defaults 
                to None which leaves values as is.

            newline (bool): Decides whether to add a line break between each summand in 
                in the condition.

        Returns:
            (str): String representation for the condition. 
        """
        est_max_features = np.max(self.features) + 1
        if feature_labels is None:
            feature_labels = [rf"$x_{i}$" for i in range(est_max_features)]

        elif len(feature_labels) < est_max_features:
            raise ValueError(
                "Input feature labels must have as least as many features as the condition's "
                "maximum feature index."
            )
        
        # Convert back to normal scaling, if applicable:
        features = self.features
        weights = self.weights
        threshold = self.threshold
        direction = self.direction

        if isinstance(scaler, StandardScaler):
            scaled_weights = np.zeros(len(weights))
            for i,feat in enumerate(features):
                w = weights[i]
                mu = scaler.mean_[feat]
                std = scaler.scale_[feat]
                scaled_weights[i] = w/std
                threshold += w * mu / std

            if len(features) == 1:
                threshold /= scaled_weights[0]
                if np.sign(scaled_weights[0]) < 0:
                    direction *= -1
                scaled_weights[0] = 1

            weights = scaled_weights

        elif isinstance(scaler, MinMaxScaler):
            scaled_weights = np.zeros(len(weights))
            for i,feat in enumerate(features):
                w = weights[i]
                scale = scaler.scale_[feat]
                minf = scaler.min_[feat]
                scaled_weights[i] = w*scale
                threshold += -1*(w * minf)

            if len(features) == 1:
                threshold /= scaled_weights[0]
                if np.sign(scaled_weights[0]) < 0:
                    direction *= -1
                scaled_weights[0] = 1

            weights = scaled_weights

        condition_str = ""
        #escape = "\n" if (len(features) > 1 and newline) else " "
        escape = "\n" if (newline) else " "
        for i,feat in enumerate(features):
            w = np.round(weights[i], 3)
            addit = r" $+$" if i < len(features) - 1 else ""
            if w != 1:
                condition_str += str(w) + r"$\cdot$" + feature_labels[feat] + addit + escape
            else:
                condition_str += feature_labels[feat] + addit + escape

    
        if self.direction == -1:
            condition_str += r"$\leq$ "
        else:
            condition_str += r"$>$ "

        condition_str += str(np.round(threshold, 3))

        return condition_str


####################################################################################################
# Redundancy removal:
####################################################################################################

def _is_finite_threshold(cond: "Condition") -> bool:
    """Best-effort check for whether a condition is a degenerate 'no-op'.

    Currently, only `LinearCondition` is supported (it has a `threshold` attribute).
    For unknown condition types, we assume it is meaningful.
    """
    thr = getattr(cond, "threshold", None)
    if thr is None:
        return True
    try:
        return bool(np.isfinite(thr))
    except Exception:
        return True
    

def get_bound_type_and_value(c: LinearCondition) -> Optional[Tuple[str, float]]:
    w = float(c.weights.reshape(-1)[0])
    if w == 0.0:
        return None
    f = int(c.features.reshape(-1)[0])
    t = float(c.threshold)

    # direction == -1: w*x <= t
    if c.direction == -1:
        if w > 0:
            return ("ub", t / w)
        else:
            # w < 0: x >= t/w  (since dividing flips inequality)
            return ("lb", t / w)

    # direction == 1: w*x > t
    if w > 0:
        return ("lb", t / w)
    else:
        return ("ub", t / w)


def simplify_rule(
    rule: Rule,
    *,
    drop_nonfinite_thresholds: bool = True,
    deduplicate: bool = True,
    simplify_axis_aligned_linear: bool = True,
) -> Rule:
    """Return a simplified rule with redundant conditions removed.

    This is intended as a *safe* simplifier:
    - Always preserves semantics for the supported cases.
    - Only attempts redundancy removal for axis-aligned `LinearCondition` constraints.

    Notes on axis-aligned logic:
        We treat `LinearCondition` with `len(features)==1` and `weights` length 1 as a
        single-feature constraint of the form:

            direction == -1:  (w * x_f) <= threshold
            direction ==  1:  (w * x_f) >  threshold

        When w != 0, this can be rewritten as a bound on x_f. We use that bound to
        detect dominated constraints and keep only the tightest constraint in each
        direction.
    """
    conditions: List[Condition] = list(rule.conditions)

    if drop_nonfinite_thresholds:
        conditions = [c for c in conditions if _is_finite_threshold(c)]

    if deduplicate:
        # Preserve order while deduplicating (stable).
        seen = set()
        uniq: List[Condition] = []
        for c in conditions:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        conditions = uniq

    if simplify_axis_aligned_linear:
        axis: List[LinearCondition] = []
        other: List[Condition] = []
        for c in conditions:
            if isinstance(c, LinearCondition):
                try:
                    if c.features.size == 1 and c.weights.size == 1:
                        axis.append(c)
                        continue
                except Exception:
                    pass
            other.append(c)

        # Pick tightest constraints per feature and constraint type.
        # Key includes whether it's an upper or lower bound on x_f.
        best: Dict[Tuple[int, str], LinearCondition] = {}

        # Record the best (tightest) constraint for each (feature, bound-type)
        for c in axis:
            bt = get_bound_type_and_value(c)
            if bt is None:
                # Can't reason about it; keep it verbatim.
                other.append(c)
                continue

            btype, bval = bt
            f = int(c.features.reshape(-1)[0])
            key = (f, btype)
            if key not in best:
                best[key] = c
            else:
                prev = best[key]
                prev_bt = get_bound_type_and_value(prev)
                if prev_bt is None:
                    best[key] = c
                else:
                    _, prev_val = prev_bt
                    # Tightness: upper bound => smaller is tighter; lower bound => larger is tighter
                    if (btype == "ub" and bval < prev_val) or (btype == "lb" and bval > prev_val):
                        best[key] = c

        # Keep the best axis-aligned constraints, plus all unhandled conditions.
        conditions = other + list(best.values())

        if deduplicate:
            seen = set()
            uniq = []
            for c in conditions:
                if c not in seen:
                    uniq.append(c)
                    seen.add(c)
            conditions = uniq

    return Rule(list(conditions))


def simplify_decision(decision: Decision, **kwargs) -> Decision:
    """Return a copy of `decision` with a simplified `rule`."""
    return Decision(simplify_rule(decision.rule, **kwargs), decision.label)


def simplified_rule_length(rule: Rule, **kwargs) -> int:
    """Convenience: length of rule after simplification."""
    return len(simplify_rule(rule, **kwargs))


####################################################################################################
# Saving and loading rules/decisions
####################################################################################################

def save_rules(rules: Sequence[Rule], path: Union[str, Path], protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    """Save a list/sequence of :class:`Rule` objects to disk using pickle.

    Notes:
        Pickle is Python-specific and **must not** be used with untrusted files.

    Args:
        rules: Sequence of rules to save.
        path: Output file path (e.g. "rules.pkl").
        protocol: Pickle protocol; defaults to highest available.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(list(rules), f, protocol=protocol)


def load_rules(path: Union[str, Path]) -> List[Rule]:
    """Load rules previously saved with :func:`save_rules`.

    Notes:
        Only load pickle files you created yourself (or otherwise trust).

    Args:
        path: Input file path.

    Returns:
        A list of loaded :class:`Rule` objects.
    """
    p = Path(path)
    with p.open("rb") as f:
        obj: Any = pickle.load(f)

    # Be permissive: accept list/tuple/etc. but return a list.
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


def save_decisions(decisions: Sequence[Decision], path: Union[str, Path], protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    """Save a list/sequence of :class:`Decision` objects to disk using pickle."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(list(decisions), f, protocol=protocol)


def load_decisions(path: Union[str, Path]) -> List[Decision]:
    """Load decisions previously saved with :func:`save_decisions`."""
    p = Path(path)
    with p.open("rb") as f:
        obj: Any = pickle.load(f)

    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


####################################################################################################