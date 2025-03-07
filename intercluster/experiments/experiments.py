import os
import pandas as pd
import copy
from joblib import Parallel, delayed, parallel_config
from typing import List, Callable, Any
from numpy.typing import NDArray
from intercluster.rules import *
from intercluster.utils import covered_mask, update_centers
from .modules import *

####################################################################################################


class Experiment:
    """
    Default class for performing suite of experiments on an input dataset which measure
    clustering cost with parameter changes.
    
    Args:
        data (np.ndarray): Input dataset.
        
        labels (np.ndarray): Labels for the input dataset.
        
        baseline_list (List[Baseline]): List of baseline modules to use and record results for. 

        module_list (List[Module]): List of modules to use and record results for.
        
        measurement_fns (List[Callable]): List of MeasurementFunction objects
            used to compute results.
            
        n_samples (int): Number of samples to run the experiment for.
        
        labels (List[List[int]]): Labels for the input dataset (if any).
            Defaults to None in which case data is taken to be unlabeled.
        
        verbose (bool, optional): Allows for printing of status. Defaults to True.
    
    Attrs: 
        result_dict (Dict[Tuple[str, str, int], List[float]): Dictionary with keys 
            as tuples of the form (measurement function name, module name, sample number),
            and values which are lists of measurement results.
    """
    def __init__(
        self, 
        data : NDArray,
        baseline_list : List[Baseline],
        module_list : List[Module],
        measurement_fns : List[Callable],
        n_samples : int,
        labels : List[List[int]] = None,
        cpu_count : int = 1,
        verbose : bool = False
    ):
        self.data = data
        self.labels = labels
        self.baseline_list = baseline_list
        self.module_list = module_list
        self.measurement_fns = measurement_fns
        self.n_samples = n_samples
        self.cpu_count = cpu_count
        self.verbose = verbose
        
        # Initializes the result dictionary
        self.result_dict = {}
                

    def run_baseline(self):
        """
        Runs the baseline modules.
        """
        pass 
    
    def run_module(self):
        """
        Runs the modules.
        """
        pass
    
    def run(self):
        """
        Runs the experiment.
        """
        pass
    
    def save_results(self):
        """
        Saves a dataframe of experiment results.
        """
        pass
    
    
####################################################################################################


class CoverageExperiment(Experiment):
    """
    Perfroms an experiment on an input dataset which measures clustering cost as
    coverage requirements are increased. Every step forward in this experiment should call 
    a .step_coverage() method of its modules, which increases the number of rules used by 1.
    
    Args:
        data (np.ndarray): Input dataset.
        
        labels (np.ndarray): Labels for the input dataset.
        
        baseline_list (List[Baseline]): List of baseline modules to use and record results for. 
        
        module_list (List[Module]): List of modules to use and record results for.
        
        measurement_fns (List[Callable]): List of MeasurementFunction objects
            used to compute results.

        n_samples (int): Number of sample trials to run the experiment for.
        
        labels (np.ndarray): Labels for the input dataset (if any). Defaults to None in which case
            data is taken to be unlabeled.
        
        verbose (bool, optional): Allows for optional printing of status. Defaults to False.
        
    Attrs:
        result_dict (Dict[str, List[float]): Dictionary to store costs for each module and baseline.
    """
    def __init__(
        self, 
        data : NDArray,
        baseline_list : List[Baseline],
        module_list : List[Module],
        measurement_fns : List[Callable],
        n_samples : int,
        labels : List[List[int]] = None,
        cpu_count : int = 1,
        verbose : bool = False
    ):
        super().__init__(
            data = data,
            baseline_list = baseline_list,
            module_list = module_list,
            measurement_fns = measurement_fns,
            n_samples = n_samples,
            labels = labels,
            cpu_count = cpu_count,
            verbose = verbose
        )
            
        
    def run_baselines(self, n_steps : int):
        """
        Runs the baseline modules.
        
        Args:
            n_steps (int): Number of steps to run the experiment for.
        """
        for b in self.baseline_list:
            bassign, bcenters = b.assign(self.data)
            for fn in self.measurement_fns:
                for s in range(self.n_samples):
                    self.result_dict[("max-rule-length", b.name, s)] = [b.max_rule_length]*n_steps
                    self.result_dict[("weighted-average-rule-length", b.name, s)] = [
                        b.weighted_average_rule_length
                    ]*n_steps
                    
                    if fn.name == 'distance-ratio':
                        # Distance ratio measurement uses original centers rather than updated ones.
                        self.result_dict[(fn.name, b.name, s)] = [
                            fn(self.data, bassign, b.original_centers)
                        ] * n_steps
                    else:
                        self.result_dict[(fn.name, b.name, s)] = [
                            fn(self.data, bassign, bcenters)
                        ] * n_steps
                    
            
    def run_modules(
            self,
            module_list : List[Module],
            n_steps : int,
            step_size : float
        ) -> Dict[Tuple[str, str], List[float]]:
        """
        Runs the modules.
        
        Args:
            module_list (List[Module]): List of experiment modules to run the experiment with. 

            n_steps (int): Number of steps to run the experiment for.
            
            step_size (float): Size of coverage to increase by for every step.
            
            sample_number (int): Current sample number (helpful for recording results).

        Returns:
            module_result_dict (Dict[Tuple[str, str], List[float]]): Dictionary of results 
                in the form  {(measurement name, module name): List of measurement results}
        """
        module_result_dict = {}
        for m in self.module_list:
            module_result_dict[("max-rule-length", m.name)] = []
            module_result_dict[("weighted-average-rule-length", m.name)] = []
            for fn in self.measurement_fns:
                module_result_dict[(fn.name, m.name)] = []

        for i in range(n_steps):
            for mod in module_list:
                massign, mcenters = mod.step_coverage(self.data, self.labels, step_size = step_size)
                
                # record rule lengths:
                module_result_dict[("max-rule-length", mod.name)].append(
                    mod.max_rule_length
                )
                module_result_dict[("weighted-average-rule-length", mod.name)].append(
                    mod.weighted_average_rule_length
                )
                
                # record results from measurement functions:
                for fn in self.measurement_fns:
                    if fn.name == 'distance-ratio':
                        module_result_dict[(fn.name, mod.name)].append(
                            fn(self.data, massign, mod.original_centers)
                        )
                    else:
                        module_result_dict[(fn.name, mod.name)].append(
                            fn(self.data, massign, mcenters)
                        )
                        
        return module_result_dict
                    
        
    def run(self, n_steps : int, step_size : float):
        """
        Runs the experiment.
        
        NOTE: This should have an increment parameter. Or otherwise 
            n_rules_list should be the input. This would require changing 
            some of the modules to be non-stepwise. 
            
        Args:
            n_steps (int): Number of steps to run the experiment for.
            
        Returns:
            cost_df (pd.DataFrame): DataFrame of the results.
        """
        self.run_baselines(n_steps)

        module_lists = [copy.deepcopy(self.module_list) for _ in range(self.n_samples)]

        module_results = Parallel(n_jobs=self.cpu_count, backend = 'loky')(
                delayed(self.run_modules)(mod_list, n_steps, step_size)
                for mod_list in module_lists
        )

        for i, module_result_dict in enumerate(module_results):
            for key,value in module_result_dict.items():
                self.result_dict[key + (i,)] = value
            
        return pd.DataFrame(self.result_dict)
    
    def save_results(self, path, identifier = ''):
        """
        Saves the results of the experiment.
        
        Args:
            path (str): File path to save the results to.
            
            identifier (str, optional): Unique identifier for the results. Defaults to blank.
        """
        fname = os.path.join(path, 'coverage_exp' + str(identifier) + '.csv')
        cost_df = pd.DataFrame(self.result_dict)
        cost_df.to_csv(fname)
        
        
####################################################################################################


class RelativeCoverageExperiment(Experiment):
    """
    Perfroms an experiment on an input dataset which measures clustering cost as
    coverage requirements are increased. Every step forward in this experiment should call 
    a .step_coverage() method of its modules, which increases the number of rules used by 1.

    This differs from the previous experiment since baselines are evaluated relative to the 
    set of points that a given module covers. This happens for each module in the module list.

    Args:
        data (np.ndarray): Input dataset.
        
        labels (np.ndarray): Labels for the input dataset.
        
        baseline_list (List[Any]): List of baseline modules to use and record results for. 
        
        module (List[Module]): Modules to use and record results for.
        
        measurement_fns (List[Callable]): List of MeasurementFunction objects
            used to compute results.

        n_samples (int): Number of sample trials to run the experiment for.
        
        labels (np.ndarray): Labels for the input dataset (if any). Defaults to None in which case
            data is taken to be unlabeled.
        
        verbose (bool, optional): Allows for optional printing of status. Defaults to False.
        
    Attrs:
        result_dict (Dict[str, List[float]): Dictionary to store costs for each module and baseline.
    """
    def __init__(
        self, 
        data : NDArray,
        baseline_list : List[Baseline],
        module_list : List[Module],
        measurement_fns : List[Callable],
        n_samples : int,
        labels : List[List[int]] = None,
        cpu_count : int = 1,
        verbose : bool = False,
        thread_count : int = 1,
    ):
        super().__init__(
            data = data,
            baseline_list = baseline_list,
            module_list = module_list,
            measurement_fns = measurement_fns,
            n_samples = n_samples,
            labels = labels,
            cpu_count = cpu_count,
            verbose = verbose
        )
        # NOTE: After testing, this should really be a part of 
        # the main experiment class
        self.thread_count = thread_count
        
    def run_baselines(self, n_steps : int):
        """
        Runs the baseline modules, simply finding their assignment matrices instead of 
        computing results.
        """
        for b in self.baseline_list:
            bassign, bcenters = b.assign(self.data)
            for fn in self.measurement_fns:
                self.result_dict[("max-rule-length", b.name, b.name, 0)] = [b.max_rule_length]*n_steps
                self.result_dict[("weighted-average-rule-length", b.name, b.name, 0)] = [
                    b.weighted_average_rule_length
                ]*n_steps
                
                if fn.name == 'distance-ratio':
                    # Distance ratio measurement uses original centers rather than updated ones.
                    self.result_dict[(fn.name, b.name, b.name, 0)] = [
                        fn(self.data, bassign, b.original_centers)
                    ] * n_steps
                else:
                    self.result_dict[(fn.name, b.name, b.name, 0)] = [
                        fn(self.data, bassign, bcenters)
                    ] * n_steps

            
    def run_modules(
            self,
            module_list : List[Module],
            n_steps : int,
            step_size : float
        ) -> Dict[Tuple[str, str], List[float]]:
        """
        Runs the module, and the baseline alongside it. 
        
        Args:
            module_list (List[Module]): List of experiment modules to run the experiment with. 
                Once again, this should be a list containing a single module.

            n_steps (int): Number of steps to run the experiment for.
            
            step_size (float): Size of coverage to increase by for every step.
            
            sample_number (int): Current sample number (helpful for recording results).

        Returns:
            module_result_dict (Dict[Tuple[str, str], List[float]]): Dictionary of results 
                in the form  {(measurement name, module name): List of measurement results}
        """
        # Initialize result dictionaries
        module_result_dict = {}
        for mod in self.module_list:
            module_result_dict[("max-rule-length", mod.name, mod.name)] = []
            module_result_dict[("weighted-average-rule-length", mod.name, mod.name)] = []
            for fn in self.measurement_fns:
                module_result_dict[(fn.name, mod.name, mod.name)] = []

            for base in self.baseline_list:
                module_result_dict[("max-rule-length", mod.name, base.name)] = []
                module_result_dict[("weighted-average-rule-length", mod.name, base.name)] = []
                for fn in self.measurement_fns:
                    module_result_dict[(fn.name, mod.name, base.name)] = []


        for i in range(n_steps):
            for mod in module_list:
                massign, mcenters = mod.step_coverage(self.data, self.labels, step_size = step_size)
                
                # record rule lengths:
                module_result_dict[("max-rule-length", mod.name, mod.name)].append(
                    mod.max_rule_length
                )
                module_result_dict[("weighted-average-rule-length", mod.name, mod.name)].append(
                    mod.weighted_average_rule_length
                )
                
                # record results from measurement functions:
                for fn in self.measurement_fns:
                    if fn.name == 'distance-ratio':
                        module_result_dict[(fn.name, mod.name, mod.name)].append(
                            fn(self.data, massign, mod.original_centers)
                        )
                    else:
                        module_result_dict[(fn.name, mod.name, mod.name)].append(
                            fn(self.data, massign, mcenters)
                        )

                # Find the subset of covered points and evaluate baselines only upon them.
                covered = None
                if massign is not None:
                    covered = covered_mask(massign)

                for base in self.baseline_list:
                    bassign = None
                    bcenters = None

                    # Adjust for coverage:
                    if covered is not None:
                        bassign = base.assignment
                        bcenters = base.centers
                        bassign = copy.deepcopy(bassign)
                        bassign[~covered] = 0
                        bcenters = update_centers(self.data, bcenters, bassign)

                    # record rule lengths:
                    module_result_dict[("max-rule-length", mod.name, base.name)].append(
                        base.max_rule_length
                    )
                    module_result_dict[("weighted-average-rule-length", mod.name, base.name)].append(
                        base.weighted_average_rule_length
                    )
                    
                    # record results from measurement functions:
                    for fn in self.measurement_fns:
                        if fn.name == 'distance-ratio':
                            module_result_dict[(fn.name, mod.name, base.name)].append(
                                fn(self.data, bassign, base.original_centers)
                            )
                        else:
                            module_result_dict[(fn.name, mod.name, base.name)].append(
                                fn(self.data, bassign, bcenters)
                            )
                        
        return module_result_dict
                    
        
    def run(self, n_steps : int, step_size : float):
        """
        Runs the experiment.
        
        NOTE: This should have an increment parameter. Or otherwise 
            n_rules_list should be the input. This would require changing 
            some of the modules to be non-stepwise. 
            
        Args:
            n_steps (int): Number of steps to run the experiment for.
            
        Returns:
            cost_df (pd.DataFrame): DataFrame of the results.
        """
        self.run_baselines(n_steps)

        module_lists = [copy.deepcopy(self.module_list) for _ in range(self.n_samples)]

        #module_results = Parallel(n_jobs=self.cpu_count, backend = 'loky')(
        #        delayed(self.run_modules)(mod_list, n_steps, step_size)
        #        for mod_list in module_lists
        #)
        module_results = None
        with parallel_config(backend = 'loky', inner_max_num_threads = self.thread_count):
            module_results = Parallel(n_jobs=self.cpu_count)(
                    delayed(self.run_modules)(mod_list, n_steps, step_size)
                    for mod_list in module_lists
            )

        for i, module_result_dict in enumerate(module_results):
            for key,value in module_result_dict.items():
                self.result_dict[key + (i,)] = value
            
        return pd.DataFrame(self.result_dict)
    
    def save_results(self, path, identifier = ''):
        """
        Saves the results of the experiment.
        
        Args:
            path (str): File path to save the results to.
            
            identifier (str, optional): Unique identifier for the results. Defaults to blank.
        """
        fname = os.path.join(path, 'exp' + str(identifier) + '.csv')
        cost_df = pd.DataFrame(self.result_dict)
        cost_df.to_csv(fname)
        
        
####################################################################################################

'''
class RulesExperiment(Experiment):
    """
    Perfroms an experiment on an input dataset which measures clustering cost as the 
    number of rules is increased. Every step forward in this experiment should call 
    a .step_num_rules() method of its modules, which increases the number of rules used by 1.
    
    Args:
        data (np.ndarray): Input dataset.
        
        labels (np.ndarray): Labels for the input dataset.
        
        baseline_list (List[Any]): List of baseline modules to use and record results for. 
        
        module_list (List[Any]): List of modules to use and record results for.
        
        measurement_fns (List[Callable]): List of MeasurementFunction objects
            used to compute results.

        n_samples (int): Number of sample trials to run the experiment for.
        
        labels (np.ndarray): Labels for the input dataset (if any). Defaults to None in which case
            data is taken to be unlabeled.
        
        verbose (bool, optional): Allows for optional printing of status. Defaults to False.
        
    Attrs:
        result_dict (Dict[str, List[float]): Dictionary to store costs for each module and baseline.
    """
    def __init__(
        self, 
        data,
        baseline_list,
        module_list,
        measurement_fns,
        n_samples,
        labels = None,
        verbose = False
    ):
        super().__init__(
            data = data,
            baseline_list = baseline_list,
            module_list = module_list,
            measurement_fns = measurement_fns,
            n_samples = n_samples,
            labels = labels,
            verbose = verbose
        )
            
        
    def run_baselines(self, n_steps : int):
        """
        Runs the baseline modules.
        
        Args:
            n_steps (int): Number of steps to run the experiment for.
        """
        for b in self.baseline_list:
            bassign, bcenters = b.assign(self.data)
            for fn in self.measurement_fns:
                for s in range(self.n_samples):
                    self.result_dict[("max-rule-length", b.name, s)] = [b.max_rule_length]*n_steps
                    
                    self.result_dict[(fn.name, b.name, s)] = [
                        fn(self.data, bassign, bcenters)
                    ] * n_steps
                    
            
    def run_modules(self, n_steps : int, sample_number : int):
        """
        Runs the modules.
        
        Args:
            n_steps (int): Number of steps to run the experiment for.
            
            sample_number (int): Current sample number (helpful for recording results).
        """
        for i in range(n_steps):
            if self.verbose:
                print(f"Running for step {i}.")
            for mod in self.module_list:
                massign, mcenters = mod.step_num_rules(self.data, self.labels)
                
                # record maximum rule length:
                self.result_dict[("rule-length", mod.name, sample_number)].append(
                    mod.max_rule_length
                )
                
                # record results from measurement functions:
                for fn in self.measurement_fns:
                    self.result_dict[(fn.name, mod.name, sample_number)].append(
                        fn(self.data, massign, mcenters)
                    )
        
    def run(self, n_steps):
        """
        Runs the experiment.
        
        NOTE: This should have an increment parameter. Or otherwise 
            n_rules_list should be the input. This would require changing 
            some of the modules to be non-stepwise. 
            
        Args:
            n_steps (int): Number of steps to run the experiment for.
            
        Returns:
            cost_df (pd.DataFrame): DataFrame of the results.
        """
        self.run_baselines(n_steps)
        
        for s in range(self.n_samples):
            if self.verbose:
                print(f"Running for sample {s}.")
                
            self.run_modules(n_steps, s)
            
            # Reset the modules in between samples:
            for m in self.module_list:
                m.reset()
            
            if self.verbose:
                print()
            
        return pd.DataFrame(self.result_dict)
    
    def save_results(self, path, identifier = ''):
        """
        Saves the results of the experiment.
        
        Args:
            path (str): File path to save the results to.
            
            identifier (str, optional): Unique identifier for the results. Defaults to blank.
        """
        fname = os.path.join(path, 'rules_exp' + str(identifier) + '.csv')
        cost_df = pd.DataFrame(self.result_dict)
        cost_df.to_csv(fname)
        
        
####################################################################################################


class RulesExperimentV2(Experiment):
    """
    Perfroms an experiment on an input dataset which measures clustering cost as the 
    number of rules changes. In this experiment, a comparison model is used to compare 
    against all other modules. An assignment from this module determines 
    how the others are measured. 
    
    Args:
        data (np.ndarray): Input dataset.
        
        labels (np.ndarray): Labels for the input dataset.
        
        comparison_module (Any): Module to compare the other modules to.
        
        baseline_list (List[Any]): List of baseline modules to use and record results for. 
        
        module_list (List[Any]): List of modules to use and record results for.
        
        measurement_fns (List[Callable]): List of MeasurementFunction objects
            used to compute results.
        
        random_seed (int, optional): Random seed for experiments. Defaults to None.
        
        verbose (bool, optional): Allows for printing of status. Defaults to True.
        
    Attrs:
        result_dict (Dict[str, List[float]): Dictionary to store costs for each module and baseline.
    """
    def __init__(
        self, 
        data,
        comparison_module,
        baseline_list,
        module_list,
        measurement_fns,
        n_samples,
        labels = None,
        random_seed = None,
        verbose = True
    ):
        self.comparison_module = comparison_module
        super().__init__(
            data = data,
            baseline_list = baseline_list,
            module_list = module_list,
            measurement_fns = measurement_fns,
            n_samples = n_samples,
            labels = labels,
            random_seed = random_seed,
            verbose = verbose
        )
        self.comparison_module = comparison_module
        
        for s in range(n_samples):
                self.result_dict[("depth", self.comparison_module.name, s)] = []
                for fn in measurement_fns:
                    self.result_dict[(fn.name, self.comparison_module.name, s)] = []
        
            
        
    def run_baselines(self, n_rules_list : List[int]):
        """
        Runs the baseline modules.
        
        Args:
            n_rules_list (List[int]): List with varying numbers of rules to run the module for.
        """
        for b in self.baseline_list:
            bassign, bcenters = b.assign(self.data)
            for fn in self.measurement_fns:
                for s in range(self.n_samples):
                    for i in n_rules_list:
                        uncovered = self.uncovered_dict[(s,i)]
                        bassign_coverage = bassign.copy()
                        bassign_coverage[uncovered, :] = 0
                        self.result_dict[(fn.name, b.name, s)].append(
                            fn(self.data, bassign_coverage, bcenters)
                        )
                    
            
    def run_modules(self, n_rules_list : List[int], sample_number : int):
        """
        Runs the modules.
        
        Args:
            n_rules_list (List[int]): List with varying numbers of rules to run the module for.
            sample_number (int): Sample number to run the module for.
        """
        for i in n_rules_list:
            if self.verbose:
                print(f"Running for {i} rules.")
            for m in self.module_list:
                massign, mcenters = m.step_num_rules(self.data, self.labels)
                uncovered = self.uncovered_dict[(sample_number,i)]
                massign_coverage = massign.copy()
                massign_coverage[uncovered, :] = 0
                # record depth:
                self.result_dict[("depth", m.name, sample_number)].append(m.n_depth)
                
                # record results from measurement functions:
                for fn in self.measurement_fns:
                    self.result_dict[(fn.name, m.name, sample_number)].append(
                        fn(self.data, massign_coverage, mcenters)
                    )
                    
    def run_comparison(self, n_rules_list : List[int]):
        """
        Runs the comparison module.
        
        Args:
            n_rules_list (List[int]): List with varying numbers of rules to run the module for.
            sample_number (int): Sample number to run the module for.
        """
        self.uncovered_dict = {}
        for s in range(self.n_samples):
            for i in n_rules_list:
                m = self.comparison_module
                massign, mcenters = m.step_num_rules(self.data, self.labels)
                
                # record which data points are covered:
                self.uncovered_dict[(s, i)] = np.where(np.sum(massign, axis = 1) == 0)[0]
                
                # record depth:
                self.result_dict[("depth", m.name, s)].append(m.n_depth)
                
                # record results from measurement functions:
                for fn in self.measurement_fns:
                    self.result_dict[(fn.name, m.name, s)].append(
                        fn(self.data, massign, mcenters)
                    )
            self.comparison_module.reset()
        
        
    def run(self, min_rules : int, max_rules : int):
        """
        Runs the experiment.
        
        NOTE: This should have an increment parameter. Or otherwise 
            n_rules_list should be the input. This would require changing 
            some of the modules to be non-stepwise. 
            
        Args:
            min_rules (int): Minimum number of rules to fit with.
            
            max_rules (int): Maximum number of rules to fit with.
            
        Returns:
            cost_df (pd.DataFrame): DataFrame of the results.
        """
        n_rules_list = list(range(min_rules, max_rules + 1))
        if self.verbose:
                print(f"Running Comparison Module:")
        self.run_comparison(n_rules_list)
        self.run_baselines(n_rules_list)
        
        for s in range(self.n_samples):
            if self.verbose:
                print(f"Running for Modules sample {s}.")
                
            self.run_modules(n_rules_list, sample_number = s)
            
            # reset the modules:
            for m in self.module_list:
                m.reset()
            
            if self.verbose:
                print()
            
        return pd.DataFrame(self.result_dict)
    
    def save_results(self, path, identifier = ''):
        """
        Saves the results of the experiment.
        
        Args:
            path (str): File path to save the results to.
            
            identifier (str, optional): Unique identifier for the results. Defaults to blank.
        """
        fname = os.path.join(path, 'rules_cost_comparison' + str(identifier) + '.csv')
        cost_df = pd.DataFrame(self.result_dict)
        cost_df.to_csv(fname)
'''