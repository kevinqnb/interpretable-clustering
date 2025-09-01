import numpy as np
import pandas as pd
from typing import List, Set, Tuple, Any
from numpy.typing import NDArray
from intercluster import (
    Condition,
    interval_to_condition,
)
from intercluster.mining import RuleMiner
from .decision_set import DecisionSet


####################################################################################################

# NOTE: The following code is a private submodule used to interface with the PyIDS package.
# Since PyIDS is not a dependency of Intercluster, it must be installed separately.

from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyids.model_selection.coordinate_ascent import CoordinateAscent
from pyids.data_structures.ids_rule import IDSRule
from pyarc.qcba.data_structures import QuantitativeDataFrame


####################################################################################################


class IdsSet(DecisionSet):
    """
    A Decision Set mined with an apriori search, then pruned using a combination of submodular 
    objective functions.

    This algorithm is based upon the paper:
    "Interpretable Decision Sets: A Joint Framework for Description and Prediction"
    by Lakkaraju et al., KDD 2016.

    We make use of the PyIDS package to implement the algorithm:
    Jiri Filip, Tomas Kliegr. 
    PyIDS - Python Implementation of Interpretable Decision Sets Algorithm by Lakkaraju et al, 2016. 
    RuleML+RR2019@Rule Challenge 2019. http://ceur-ws.org/Vol-2438/paper8.pdf
    Github: https://github.com/jirifilip/pyIDS

    Args:
        lambdas (list[float], optional): List of 7 lambda values for the submodular objective function.
            If None, a coordinate ascent search will be used to find good lambdas. Defaults to None.
        lambda_search_dict (dict[str, tuple[float, float]], optional): Dictionary specifying the 
            search space for each lambda value when using coordinate ascent. 
            Each key should be a string 'l1' to 'l7', and each value should be a tuple (min, max).
            If None and lambdas is also None, default search spaces will be used. Defaults to None.
        ternary_search_precision (float, optional): Precision for ternary search in coordinate 
            ascent. Defaults to 1. For more information, see the absolute_precision parameter 
            used in the following pseudocode: https://en.wikipedia.org/wiki/Ternary_search
        max_iterations (int, optional): Maximum number of iterations for coordinate ascent.
        rule_miner (RuleMiner, optional): Rule mining algorithm used to generate the rules.
            If None, the rules must be provided directly. Defaults to None.
        rules (List[List[Condition]], optional): List of rules to initialize the decision set with.
            If None, the rules will be generated using the rule_miner. Defaults to None.
        rule_labels (List[Set[int]], optional): List of labels corresponding to each rule.
            If None, the labels will be generated using the rule_miner. Defaults to None.

    """
    def __init__(
        self,
        lambdas : list[float] = None,
        lambda_search_dict : dict[str, tuple[float, float]] = None,
        ternary_search_precision : float = 1.0,
        max_iterations : int = 50,
        rule_miner : RuleMiner = None, 
        rules : List[List[Condition]] = None,
        rule_labels : List[Set[int]] = None

    ):
        super().__init__(rule_miner, rules, rule_labels)
        if self.rule_miner is None:
            raise ValueError("A rule_miner must be provided for this decision set.")
        if lambdas is not None:
            if not isinstance(lambdas, list):
                raise ValueError("lambdas must be a list of floats.")
            if len(lambdas) != 7:
                raise ValueError("Lambdas must be a list of length 7.")
        if lambda_search_dict is not None:
            if not isinstance(lambda_search_dict, dict):
                raise ValueError("lambda_search_dict must be a dictionary.")
            if len(lambda_search_dict) != 7:
                raise ValueError("Lambda search dictionary must have 7 entries.")
            if not all(isinstance(v, tuple) and len(v) == 2 for v in lambda_search_dict.values()):
                raise ValueError("Each value in the lambda search dictionary must be a tuple of (min, max).")
        elif lambdas is None:
            # Default search space for each lambda
            lambda_search_dict = {
                'l1': (0, 1000),
                'l2': (0, 1000),
                'l3': (0, 1000),
                'l4': (0, 1000),
                'l5': (0, 1000),
                'l6': (0, 1000),
                'l7': (0, 1000)
            }
        if not isinstance(ternary_search_precision, float) or ternary_search_precision <= 0:
            raise ValueError("ternary_search_precision must be a positive floating point.")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")

        self.lambdas = lambdas
        self.lambda_search_dict = lambda_search_dict
        self.ternary_search_precision = ternary_search_precision
        self.max_iterations = max_iterations

    
    def ids_to_decision_set(self, cars : List[IDSRule]) -> List[List[Condition]]:
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


    def prune(self, X : NDArray, y : List[Set[int]] = None):
        """
        Prunes the decision set using the pruner.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels. Defaults to None.
        """
        if self.decision_set is None or self.decision_set_labels is None:
            raise ValueError('Decision set has not been fitted yet.')
        
        quant_df = QuantitativeDataFrame(self.rule_miner.bin_df)

        self.rule_miner.cars = [car for i,car in enumerate(self.rule_miner.cars) 
                                if int(self.rule_miner.cars[i].consequent[1]) != -1]

        if self.lambdas is None:
            def fmax(lambda_dict):
                ids = IDS(algorithm="SLS")
                ids.fit(
                    class_association_rules=self.rule_miner.cars,
                    quant_dataframe=quant_df,
                    lambda_array=list(lambda_dict.values())
                )

                auc = ids.score_auc(quant_df)
                return auc

            coord_asc = CoordinateAscent(
                func=fmax,
                func_args_ranges=self.lambda_search_dict,
                ternary_search_precision=self.ternary_search_precision,
                max_iterations=self.max_iterations
            )

            lambdas = coord_asc.fit()
        else:
            lambdas = self.lambdas

        print("Lambdas found:", lambdas)
        ids = IDS(algorithm="SLS")
        ids.fit(class_association_rules=self.rule_miner.cars, quant_dataframe=quant_df, lambda_array=lambdas)
        decision_set, decision_set_labels = self.ids_to_decision_set(ids.clf.rules)
        return decision_set, decision_set_labels

    '''
    def _fitting(
        self,
        X : NDArray,
        y : List[Set[int]] = None
    ) -> Tuple[List[List[Condition]], List[Set[int]]]:
        """
        Privately used, custom fitting function.
        Fits a decision set to an input dataset. 
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels. Defaults to None.
            
        returns:
            decision_set (List[Condition]): List of rules.
            
            decision_set_labels (List[int]): List of labels corresponding to each rule.
        """
        try:
            from ._ids import fit_ids
        except ImportError:
            raise ImportError(
                "This class requires proper installation of PyIDS. "
                "Please see the documentation for instructions."
        )
        decision_set, decision_set_labels = fit_ids(
            X=X,
            y=y,
            bins=self.bins,
            n_mine=self.n_mine,
            lambdas=self.lambdas,
            lambda_search_dict=self.lambda_search_dict,
            ternary_search_precision=self.ternary_search_precision,
            max_iterations=self.max_iterations,
            quantiles=self.quantiles
        )
        return decision_set, decision_set_labels
    '''
    

####################################################################################################