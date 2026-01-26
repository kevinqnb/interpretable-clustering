import numpy as np
from numpy.typing import NDArray
from typing import List, Set, Tuple
from numpy.typing import NDArray
from intercluster import (
    Rule,
    Decision,
)
from .objectives import (
    Objective, CoverageMistakeObjective, CoverageCostObjective,
    TotalCoverageMistakeObjective, TotalCoverageCostObjective,
    CoveragePairwiseDistanceObjective, TotalCoveragePairwiseDistanceObjective
)
from .decision_set import DecisionSet
from pathlib import Path
from typing import Any, Union


####################################################################################################


class DSCluster(DecisionSet):
    """
    Collection of rules drawn as boxes (rules) around collections of points in the dataset.
    """
    def __init__(
        self,
        rules : List[Rule],
        objective_type : str = "coverage-cost",
        n_select : int = 0,
        alpha_val : float = 0.0,
        lambda_val : float = None,
        cluster_centers : NDArray = None,
        cluster_cost_method : str = "kmeans",
        weights : NDArray = None,
        selection_algorithm : str = "distorted-greedy",
        precomputed_path: Union[str, Path] = None,
        output_path: Union[str, Path] = None,
        pack_bits: bool = True,
        rule_labels : List[Set[int]] = None, # DUMMY ARGUMENT TO MATCH PARENT CLASS, must be None
    ):
        """
        Args:
            rules (List[List[Condition]]): List of rules to initialize the decision set with.
            n_select (int): Number of rules to select.
            alpha_val (float, optional): Alpha value for the objective function. Defaults to 0.0.
            lambda_val (float, optional): Lambda value for the objective function. 
                Defaults to None, in which case it will be set automatically.
            cluster_centers (NDArray, optional): Cluster centers for the objective function. 
                Defaults to None.
            weights (NDArray, optional): Weights for the data points. Defaults to None.
            objective_type (str, optional): Type of objective function to use. 
                Options are "coverage-mistake", "total-coverage-mistake", "coverage-cost", and 
                "total-coverage-cost". Defaults to "coverage-cost".
            cluster_cost_method (str, optional): Method to use for clustering costs. Defaults to "kmeans".
            selection_algorithm (str, optional): Algorithm to use for selection. Defaults to "distorted-greedy".
            precomputed_path (str, optional): Path to precomputed data for the decision set. 
                Defaults to None. If this is given, a decision set will be initialized from it, 
                rather than from the rules provided.
            output_path (str, optional): Path to save output data. Defaults to None.
            pack_bits (bool, optional): Whether to pack bits for rule coverage matrix. Defaults to True.
            rule_labels (List[Set[int]], optional): Labels for each rule. Must be None for DSCluster.
        """
        assert rule_labels is None, 'rule_labels must be None for DSCluster.'
        super().__init__(rules = rules, rule_labels = None)

        if (objective_type == "coverage-cost" or objective_type == "total-coverage-cost") and cluster_centers is None:
            raise ValueError('Cluster centers must be provided for cost-based objectives.')

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

        assert cluster_cost_method in ["kmeans", "kmedians"], \
            'cluster_cost_method must be either "kmeans" or "kmedians".'
        self.cluster_cost_method = cluster_cost_method

        assert selection_algorithm in ['distorted-greedy', 'lazy-greedy'], \
            'selection_algorithm must be either "distorted-greedy" or "lazy-greedy".'
        self.selection_algorithm = selection_algorithm

        assert precomputed_path is None or isinstance(precomputed_path, (str, Path)), \
            'precomputed_path must be a string or Path.'
        self.precomputed_path = precomputed_path

        assert output_path is None or isinstance(output_path, (str, Path)), \
            'output_path must be a string or Path.'
        self.output_path = output_path

        assert isinstance(pack_bits, bool), 'pack_bits must be a boolean.'
        self.pack_bits = pack_bits

        self.objective_type = objective_type
        self.initialize_objective()
        

    def initialize_objective(self) -> None:
        """
        Initializes the objective function based on the specified type.
        """
        if self.objective_type == "coverage-mistake":
            self.objective = CoverageMistakeObjective(
                n_select = self.n_select,
                alpha_val = self.alpha_val,
                lambda_val = self.lambda_val,
                weights = self.weights,
                selection_algorithm = self.selection_algorithm,
                precomputed_path = self.precomputed_path,
                output_path = self.output_path,
                pack_bits = self.pack_bits
            )
        elif self.objective_type == "total-coverage-mistake":
            self.objective = TotalCoverageMistakeObjective(
                n_select = self.n_select,
                alpha_val = self.alpha_val,
                lambda_val = self.lambda_val,
                weights = self.weights,
                selection_algorithm = self.selection_algorithm,
                precomputed_path = self.precomputed_path,
                output_path = self.output_path,
                pack_bits = self.pack_bits
            )
        elif self.objective_type == "coverage-cost":
            self.objective = CoverageCostObjective(
                cluster_centers = self.cluster_centers,
                n_select = self.n_select,
                alpha_val = self.alpha_val,
                lambda_val = self.lambda_val,
                weights = self.weights,
                cluster_cost_method = self.cluster_cost_method,
                selection_algorithm = self.selection_algorithm,
                precomputed_path = self.precomputed_path,
                output_path = self.output_path,
                pack_bits = self.pack_bits
            )
        elif self.objective_type == "total-coverage-cost":
            self.objective = TotalCoverageCostObjective(
                cluster_centers = self.cluster_centers,
                n_select = self.n_select,
                alpha_val = self.alpha_val,
                lambda_val = self.lambda_val,
                weights = self.weights,
                cluster_cost_method = self.cluster_cost_method,
                selection_algorithm = self.selection_algorithm,
                precomputed_path = self.precomputed_path,
                output_path = self.output_path,
                pack_bits = self.pack_bits
            )
        elif self.objective_type == "coverage-pairwise-distance":
            self.objective = CoveragePairwiseDistanceObjective(
                n_select = self.n_select,
                alpha_val = self.alpha_val,
                lambda_val = self.lambda_val,
                weights = self.weights,
                selection_algorithm = self.selection_algorithm,
                precomputed_path = self.precomputed_path,
                output_path = self.output_path,
                pack_bits = self.pack_bits
            )
        elif self.objective_type == "total-coverage-pairwise-distance":
            self.objective = TotalCoveragePairwiseDistanceObjective(
                n_select = self.n_select,
                alpha_val = self.alpha_val,
                lambda_val = self.lambda_val,
                weights = self.weights,
                selection_algorithm = self.selection_algorithm,
                precomputed_path = self.precomputed_path,
                output_path = self.output_path,
                pack_bits = self.pack_bits
            )
        else:
            raise ValueError(f'Unknown objective type: {self.objective_type}')


    def select(
            self,
            X : NDArray,
            y : List[Set[int]]
        ) -> set[Decision]:
        """
        selects the decision set by removing rules that do not cover any points in the dataset.

        Args:
            X (np.ndarray): Input dataset.
            y (List[Set[int]]): Target labels.
        """
        if self.decision_set is None:
            raise ValueError('Decision set has not been initialized yet.')
        
        print('Initializing objective with data...')
        self.objective.initialize_data(X, y)
        print('Data initialized.')

        self.objective.initialize_decision_set(self.decision_set)
        self.objective.set_lambda(self.lambda_val)
        self.lambda_val = self.objective.lambda_val
        selected_decision_set = self.objective.select()
        return selected_decision_set


####################################################################################################