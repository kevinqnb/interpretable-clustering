import numpy as np
import pandas as pd
from typing import List, Set, Tuple, Any
from numpy.typing import NDArray
from intercluster import (
    Condition,
    LinearCondition,
    can_flatten,
    flatten_labels,
    entropy_bin,
    interval_to_condition,
)
from .decision_set import DecisionSet

####################################################################################################

# NOTE: The following code is a private submodule used to interface with the PyIDS package.
# Since PyIDS is not a dependency of Intercluster, it must be installed separately.

from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyids.model_selection.coordinate_ascent import CoordinateAscent
from pyids.data_structures.ids_rule import IDSRule
from pyarc.qcba.data_structures import QuantitativeDataFrame

# The same is true for the MDLP package, which is used for discretization.
from mdlp.discretization import MDLP


####################################################################################################


def ids_to_decision_set(cars : List[IDSRule]) -> List[List[Condition]]:
    """
    Convert a list of rules found with PyIDS to a list of Conditions.
    Args:
        cars (List[IDSRule]): A list of Class Association Rules (CARs).
    Returns:
        list: A list of Rule objects.
    """
    decision_set = []
    decision_set_labels = []
    for car in cars:
        car_dict = car.to_dict()
        car_interval_dict = car_dict['antecedent']
        rule_conditions = []
        for interval_dict in car_interval_dict:
            feature = int(interval_dict['name'])
            interval = interval_dict['value']
            # Convert the interval to two Conditions
            # (one for the lower bound and one for the upper bound)
            lower_condition, upper_condition = interval_to_condition(feature, interval)
            rule_conditions.append(lower_condition)
            rule_conditions.append(upper_condition)
        decision_set.append(rule_conditions)
        decision_set_labels.append({int(car_dict['consequent']['value'])})
    return decision_set, decision_set_labels


####################################################################################################


def fit_ids(
        X : NDArray,
        y : List[Set[int]],
        n_mine : int,
        lambdas : list[float],
        lambda_search_dict : dict[str, tuple[float, float]],
        ternary_search_precision : int,
        max_iterations : int,
):
    """
    Fits a decision set using the PyIDS package.

    Args:
        X (np.ndarray): Input dataset.
        y (List[Set[int]]): Target labels.
        lambdas (list[float], optional): List of 7 lambda values for the submodular objective function.
            If None, a coordinate ascent search will be used to find good lambdas. Defaults to None.
        lambda_search_dict (dict[str, tuple[float, float]], optional): Dictionary specifying the 
            search space for each lambda value when using coordinate ascent. 
            Each key should be a string 'l1' to 'l7', and each value should be a tuple (min, max).
            If None and lambdas is also None, default search spaces will be used. Defaults to None.
        ternary_search_precision (int, optional): Precision for ternary search in coordinate 
            ascent. Defaults to 1. For more information, see the absolute_precision parameter 
            used in the following pseudocode: https://en.wikipedia.org/wiki/Ternary_search
        max_iterations (int, optional): Maximum number of iterations for coordinate ascent. 
            Defaults to 50.

    returns:
        decision_set (List[Condition]): List of rules.
        
        decision_set_labels (List[int]): List of labels corresponding to each rule.
    """
    if not can_flatten(y):
        raise ValueError("Each data point must be assigned to a single label.")
    y_ = flatten_labels(y)

    bin_df = entropy_bin(X, y)
    bin_df.columns = bin_df.columns.astype(str)
    bin_df['class'] = y_
    bin_df = bin_df.astype(str)

    quant_df = QuantitativeDataFrame(bin_df)
    cars = mine_CARs(bin_df, n_mine)

    if lambdas is None:
        def fmax(lambda_dict):
            ids = IDS(algorithm="SLS")
            ids.fit(
                class_association_rules=cars,
                quant_dataframe=quant_df,
                lambda_array=list(lambda_dict.values())
            )

            auc = ids.score_auc(quant_df)
            return auc

        coord_asc = CoordinateAscent(
            func=fmax,
            func_args_ranges=lambda_search_dict,
            ternary_search_precision=ternary_search_precision,
            max_iterations=max_iterations
        )

        lambdas = coord_asc.fit()
    else:
        lambdas = lambdas

    print("Lambdas found:", lambdas)
    ids = IDS(algorithm="SLS")
    ids.fit(class_association_rules=cars, quant_dataframe=quant_df, lambda_array=lambdas)
    decision_set, decision_set_labels = ids_to_decision_set(ids.clf.rules)
    return decision_set, decision_set_labels


####################################################################################################