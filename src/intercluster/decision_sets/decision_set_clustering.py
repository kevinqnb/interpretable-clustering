import numpy as np
from numpy.typing import NDArray
from typing import List, Set, Tuple
from numpy.typing import NDArray
from intercluster import (
    Condition,
    Rule,
    Decision,
    satisfies_rule,
    labels_to_assignment,
    unique_labels,
)
from .mining import RuleMiner
from .objectives import (
    Objective, CoverageMistakeObjective, CoverageCostObjective,
    TotalCoverageMistakeObjective, TotalCoverageCostObjective
)
from .decision_set import DecisionSet


####################################################################################################


class DSCluster(DecisionSet):
    """
    Collection of rules drawn as boxes (rules) around collections of points in the dataset.
    """
    def __init__(
        self,
        rules : List[Rule],
        n_select : int,
        alpha_val : float = 0.0,
        lambda_val : float = None,
        cluster_centers : NDArray = None,
        weights : NDArray = None,
        objective_type : str = "coverage-cost",
        cluster_cost_method : str = "kmeans",
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
        """
        assert rule_labels is None, 'rule_labels must be None for DSCluster.'
        super().__init__(rules = rules, rule_labels = None)

        if (objective_type == "coverage-cost" or objective_type == "total-coverage-cost") and cluster_centers is None:
            raise ValueError('Cluster centers must be provided for cost-based objectives.')

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

        assert cluster_cost_method in ["kmeans", "kmedians"], \
            'cluster_cost_method must be either "kmeans" or "kmedians".'
        self.cluster_cost_method = cluster_cost_method

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
                weights = self.weights
            )
        elif self.objective_type == "total-coverage-mistake":
            self.objective = TotalCoverageMistakeObjective(
                n_select = self.n_select,
                alpha_val = self.alpha_val,
                lambda_val = self.lambda_val,
                weights = self.weights
            )
        elif self.objective_type == "coverage-cost":
            self.objective = CoverageCostObjective(
                cluster_centers = self.cluster_centers,
                n_select = self.n_select,
                alpha_val = self.alpha_val,
                lambda_val = self.lambda_val,
                weights = self.weights,
                cluster_cost_method = self.cluster_cost_method
            )
        elif self.objective_type == "total-coverage-cost":
            self.objective = TotalCoverageCostObjective(
                cluster_centers = self.cluster_centers,
                n_select = self.n_select,
                alpha_val = self.alpha_val,
                lambda_val = self.lambda_val,
                weights = self.weights,
                cluster_cost_method = self.cluster_cost_method
            )
        elif self.objective_type == "coverage-pairwise-distance":
            self.objective = CoverageMistakeObjective(
                n_select = self.n_select,
                alpha_val = self.alpha_val,
                lambda_val = self.lambda_val,
                weights = self.weights
            )
        elif self.objective_type == "total-coverage-pairwise-distance":
            self.objective = TotalCoverageMistakeObjective(
                n_select = self.n_select,
                alpha_val = self.alpha_val,
                lambda_val = self.lambda_val,
                weights = self.weights
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
        
        self.objective.initialize_data(X, y)
        self.objective.initialize_decision_set(self.decision_set)
        self.objective.set_lambda(self.lambda_val)
        self.lambda_val = self.objective.lambda_val
        selected_decision_set = self.objective.select()
        return selected_decision_set


####################################################################################################