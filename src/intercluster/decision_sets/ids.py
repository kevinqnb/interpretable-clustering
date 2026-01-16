import numpy as np
import pandas as pd
from typing import List, Set
from numpy.typing import NDArray
from intercluster import (
    Condition,
    Rule,
    Decision,
    interval_to_condition,
    decision_set_to_cars,
    flatten_labels
)
from .mining import RuleMiner
from .decision_set import DecisionSet


####################################################################################################

# NOTE: The following code is a private submodule used to interface with the PyIDS package.
# Since PyIDS is not a dependency of Intercluster, it must be installed separately.

from pyids.algorithms.ids import IDS as IDS_pyids
from pyids.data_structures.ids_cacher import IDSCacher
from pyids.data_structures.ids_ruleset import IDSRuleSet
from pyids.model_selection.coordinate_ascent import CoordinateAscent
from pyids.data_structures.ids_rule import IDSRule
from pyarc.qcba.data_structures import QuantitativeDataFrame


####################################################################################################


class IDS(DecisionSet):
    """
    A Decision Set mined with an apriori search, then selected using a combination of submodular 
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
        ids_cacher (IDSCacher, optional): An optional IDSCacher object to cache computations
            during IDS fitting. Defaults to None.

    """
    def __init__(
        self,
        rules : List[Rule] = None,
        bin_df : pd.DataFrame = None,
        lambdas : list[float] = None,
        lambda_search_dict : dict[str, tuple[float, float]] = None,
        ternary_search_precision : float = 1.0,
        max_iterations : int = 50,
        ids_cacher:  IDSCacher = None,
        rule_labels : List[Set[int]] = None,
    ):
        assert rule_labels is None, 'rule_labels must be None for IDS.'
        super().__init__(rules = rules, rule_labels = None)

        if bin_df is None:
            raise ValueError("bin_df must be provided to initialize IDS.")
        self.bin_df = bin_df
            
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
        self.ids_cacher = ids_cacher

    
    def ids_to_decision_set(self, cars : List[IDSRule]) -> List[Rule]:
        """
        Convert a list of rules found with PyIDS to a list of Conditions.
        Args:
            cars (List[IDSRule]): A list of Class Association Rules (CARs).
        Returns:
            list: A list of Rule objects.
        """
        decision_set = []
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
            rule = Rule(rule_conditions)
            label = int(car_dict['consequent']['value'])
            decision_set.append(Decision(rule, label))
        return decision_set


    def select(self, X : NDArray, y : List[Set[int]] = None):
        """
        selects the decision set using the selectr.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels. Defaults to None.
        """
        y_ = flatten_labels(y)
        if len(y_) != len(y):
            raise ValueError("Each data point must have exactly one label.")
        if self.decision_set is None:
            raise ValueError('Decision set has not been fitted yet.')
        
        cars = decision_set_to_cars(
            X, y,
            self.decision_set
        )
        valid_cars = [car for i,car in enumerate(cars) if int(cars[i].consequent[1]) != -1]
        if len(valid_cars) == 0:
            raise ValueError("No valid (non-outlier) class association rules found. " \
            "Try increasing the number of mined rules.")
        ids_rules = list(map(IDSRule, valid_cars))
        all_rules = IDSRuleSet(ids_rules)
        bin_df = self.bin_df.assign(**{'class': y_})
        bin_df['class'] = bin_df['class'].astype(str)
        quant_df = QuantitativeDataFrame(bin_df)

        if self.ids_cacher is None:
            import time; start_time = time.time()
            print('Calculating IDS cacher overlaps...')
            self.ids_cacher = IDSCacher()
            self.ids_cacher.calculate_overlap(all_rules, quant_df)
            end = time.time()
            print(f"IDS cacher overlaps calculated in {end - start_time:.2f} seconds.")

        if self.lambdas is None:
            def fmax(lambda_dict):
                ids = IDS_pyids(algorithm="SLS")
                ids.ids_ruleset = all_rules
                ids.cacher = self.ids_cacher
                ids.fit(
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
            #print("Lambdas found:", lambdas)
        else:
            lambdas = self.lambdas


        print('Starting IDS selection...')
        import time; start_time = time.time()
        ids = IDS_pyids(algorithm="DLS")
        ids.ids_ruleset = all_rules
        ids.cacher = self.ids_cacher
        ids.fit(quant_dataframe=quant_df, lambda_array=lambdas)
        end = time.time()
        print(f"IDS selection finished in {end - start_time:.2f} seconds.")
        selected_decision_set = self.ids_to_decision_set(ids.clf.rules)
        return selected_decision_set

    def get_cacher(self) -> IDSCacher:
        """
        Get the IDSCacher used during fitting.
        
        Returns:
            IDSCacher: The IDSCacher object.
        """
        return self.ids_cacher
    

####################################################################################################