import numpy as np
import pandas as pd
from typing import List, Set, Tuple, Any
from numpy.typing import NDArray
from intercluster import (
    Condition,
)
from .decision_set import DecisionSet


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
        bins (int): We assume that the input dataset is real valued. Therefore, before 
            applying the algorithm, each feature must be divided into categorical 'buckets'. 
            This parameter specifies the number of buckets to use for discretization.
            This is done using quantiles so that each bucket contains roughly the same number of 
            data points. This parameter specifies the number of buckets to use for discretization. 
        n_mine (int): Total number of rules to mine with the apriori algorithm.
        lambdas (list[float], optional): List of 7 lambda values for the submodular objective function.
            If None, a coordinate ascent search will be used to find good lambdas. Defaults to None.
        lambda_search_dict (dict[str, tuple[float, float]], optional): Dictionary specifying the 
            search space for each lambda value when using coordinate ascent. 
            Each key should be a string 'l1' to 'l7', and each value should be a tuple (min, max).
            If None and lambdas is also None, default search spaces will be used. Defaults to None.
        ternary_search_precision (int, optional): Precision for ternary search in coordinate 
            ascent. Defaults to 1. For more information, see the absolute_precision parameter 
            used in the following pseudocode: https://en.wikipedia.org/wiki/Ternary_search
        quantiles (bool, optional): If True, uses quantiles for discretization so that each bucket 
            contains roughly the same number of data points. If False, uses equal-width bins.
            Defaults to True.

    """
    def __init__(
        self,
        bins : int,
        n_mine : int,
        lambdas : list[float] = None,
        lambda_search_dict : dict[str, tuple[float, float]] = None,
        ternary_search_precision : int = 1,
        max_iterations : int = 50,
        quantiles : bool = True
    ):
        if not isinstance(bins, int) or bins <= 0:
            raise ValueError("bins must be a positive integer.")
        if not isinstance(n_mine, int) or n_mine <= 0:
            raise ValueError("n_mine must be a positive integer.")
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
        if not isinstance(ternary_search_precision, int) or ternary_search_precision <= 0:
            raise ValueError("ternary_search_precision must be a positive integer.")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        if not isinstance(quantiles, bool):
            raise ValueError("quantiles must be a boolean value.")


        self.bins = bins
        self.n_mine = n_mine
        self.lambdas = lambdas
        self.lambda_search_dict = lambda_search_dict
        self.ternary_search_precision = ternary_search_precision
        self.max_iterations = max_iterations
        self.quantiles = quantiles
        super().__init__()


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
    

####################################################################################################