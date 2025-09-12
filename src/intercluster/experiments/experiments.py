import os
import pandas as pd
import copy
from joblib import Parallel, delayed, parallel_config
from typing import List, Callable, Any, Set
from numpy.typing import NDArray
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.utils import covered_mask, update_centers
from intercluster.measurements import coverage, coverage_mistake_score
from .modules import *
import time

####################################################################################################


class Experiment:
    """
    Default class for performing suite of experiments on an input dataset which measure
    clustering cost with parameter changes.
    
    Args:
        data (np.ndarray): Input dataset.
        
        labels (np.ndarray): Labels for the input dataset.
        
        baseline (Baseline): Baseline module to use and record results for. 

        module_list (List[Tuple[Module, List[Dict[str, Any]]]]): List of (module, parameter list) pairs
            to use and record results for. Each item in the parameter list should be a dictionary 
            of parameters to pass to the module.
        
        measurement_fns (List[Callable]): List of MeasurementFunction objects
            used to compute results.
            
        n_samples (int): Number of samples to run the experiment for.
        
        labels (List[List[int]]): Labels for the input dataset (if any).
            Defaults to None in which case data is taken to be unlabeled.
        
        verbose (bool, optional): Allows for printing of status. Defaults to True.
    
    Attrs: 
        result_dict (Dict[Tuple[str, str, int], NDArray]): Dictionary with keys 
            as tuples of the form (measurement function name, module name, sample number),
            and values which are arrays of measurement results.
    """
    def __init__(
        self, 
        data : NDArray,
        baseline : Baseline,
        module_list : List[Tuple[Module, List[Dict[str, Any]]]],
        measurement_fns : List[Callable] = None,
        n_samples : int = None,
        labels : List[List[int]] = None,
        cpu_count : int = 1,
        verbose : bool = False
    ):
        self.data = data
        self.labels = labels
        self.baseline = baseline
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
    
    def run_modules(self):
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


class MaxRulesExperiment(Experiment):
    """
    Perfroms an experiment on an input dataset which measures performance as
    the maximum number of allowed rules is increased. 
    
    Every step forward in this experiment should call 
    a .step_n_rules() method of its modules, which increases the number of rules used by 1.

    Args:
        data (np.ndarray): Input dataset.

        n_rules_list (List[int]): List of maximum number of rules to use in the experiment.
        
        baseline (Baseline): Single baseline model to use and record results for. 
        
        module_list (List[Tuple[Module, Dict[Tuple[int], Dict[str, Any]]]]): List of 
            (module, parameter dictionary) pairs to use and record results for. 
            Each module should be a runnable experiment object, and each parameter dictionary 
            should contain pairs {(i,j,k,..) : {fitting params}} to pass to the module. 
            More specifically, each parameter dictionary key should be a tuple of integers 
            representing items from n_rules_list. Each value should be a dictionary of 
            of fitting parameters to pass to the module before running. The output of the 
            fitting process for those parameters is then associated 
            each of the items in the corresponding key list. 
        
        measurement_fns (List[Callable]): List of MeasurementFunction objects
            used to compute results.

        n_samples (int): Number of sample trials to run the experiment for.
        
        labels (np.ndarray): Labels for the input dataset (if any). Defaults to None in which case
            data is taken to be unlabeled.
        
        verbose (bool, optional): Allows for optional printing of status. Defaults to False.
        
    Attrs:
        result_dict (Dict[Tuple[str, str, int], NDArray]): Dictionary with keys 
            as tuples of the form (measurement function name, module name, sample number),
            and values which are arrays of measurement results.
    """
    def __init__(
        self, 
        data : NDArray,
        n_rules_list : List[int],
        baseline : Baseline,
        module_list : List[Tuple[Module, Dict[Tuple[int], Dict[str, Any]]]],
        measurement_fns : List[Callable],
        n_samples : int,
        labels : List[List[int]] = None,
        cpu_count : int = 1,
        verbose : bool = False,
        thread_count : int = 1,
    ):
        self.n_rules_list = n_rules_list
        super().__init__(
            data = data,
            baseline = baseline,
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
        
    
    def run_baseline(self):
        """
        Runs the baseline modules, simply finding their assignment matrices instead of 
        computing results.
        """
        bassign = self.baseline.assign(self.data)
        self.result_dict[("max-rule-length", self.baseline.name, 0)] = {
            i : self.baseline.max_rule_length for i in self.n_rules_list
        }
        self.result_dict[("weighted-average-rule-length", self.baseline.name, 0)] = {
            i : self.baseline.weighted_average_rule_length for i in self.n_rules_list
        }
        for fn in self.measurement_fns:
            if fn.name == 'coverage-mistake-score':
                # Coverage mistake score uses original assignment.
                cover = coverage(assignment = bassign, percentage = False)
                self.result_dict[(fn.name, self.baseline.name, 0)] = {
                    i : cover for i in self.n_rules_list
                }
            else:
                fn_result = fn(data_to_cluster_assignment = bassign)
                self.result_dict[(fn.name, self.baseline.name, 0)] = {
                    i : fn_result for i in self.n_rules_list
                }

            
    def run_modules(
            self,
            module_list : List[Tuple[Module, Dict[Tuple[int], Dict[str, Any]]]]
        ) -> Dict[Tuple[str, str], Dict[int, float]]:
        """
        Runs the module, and the baseline alongside it. 
        
        Args:
            module_list (List[Tuple[Module, Dict[Tuple[int], Dict[str, Any]]]]): List of experiment 
                (module, module parameter dictionary) pairs to run the experiment with. 
                The module should be a runnable experiment object, and each parameter dictionary 
                should contain pairs {(i,j,k,...) : {fitting params}} to pass to the module.
                More specifically, each parameter dictionary key should be a tuple of integers 
                representing items from n_rules_list. Each value should be a dictionary of 
                of fitting parameters to pass to the module before running. The output of the 
                fitting process for those parameters is then associated 
                each of the items in the corresponding key list. 

            n_steps (int): Number of steps to run the experiment for.
            
            step_size (float): Size of coverage to increase by for every step.
            
            sample_number (int): Current sample number (helpful for recording results).

        Returns:
            module_result_dict (Dict[Tuple[str, str], Dict[int, float]]): Dictionary of results 
                in the form  {(measurement name, module name): {n_rules : measurement result}}
        """
        # Initialize result dictionaries
        module_result_dict = {}
        for mod, param_list in module_list:
            module_result_dict[("max-rule-length", mod.name)] = {}
            module_result_dict[("weighted-average-rule-length", mod.name)] = {}
            for fn in self.measurement_fns:
                module_result_dict[(fn.name, mod.name)] = {}

        for mod, param_dict in module_list:
            mod.reset()
            for n_rules_tuple, fitting_params in param_dict.items():
                print(mod.name + " with params: " + str(fitting_params))
                print()
                mod.update_fitting_params(fitting_params)

                try:
                    start = time.time()
                    (
                        data_to_rule_assignment,
                        rule_to_cluster_assignment,
                        data_to_cluster_assignment
                    ) = mod.fit(self.data, self.baseline.labels)
                    end = time.time()
                    print(mod.name + " fitting time: " + str(end - start))
                except:
                    print("Data: ")
                    print(self.data)
                    print()
                    print("Labels: ")
                    print(self.baseline.labels)
                    print()
                    raise ValueError("Fitting failed.")
                
                # record rule lengths:
                for i in n_rules_tuple:
                    # Record only if the module satisfies the given condition:
                    if mod.n_rules <= i:
                        module_result_dict[("max-rule-length", mod.name)][i] = mod.max_rule_length
                        module_result_dict[("weighted-average-rule-length", mod.name)][i] = mod.weighted_average_rule_length
                        
                        # record results from measurement functions:
                        for fn in self.measurement_fns:
                            module_result_dict[(fn.name, mod.name)][i] = (
                                fn(
                                    data_to_rule_assignment,
                                    rule_to_cluster_assignment,
                                    data_to_cluster_assignment
                                )
                            )
                        
        return module_result_dict
                    
        
    def run(self):
        """
        Runs the experiment.
            
        Args:
            n_steps (int): Number of steps to run the experiment for.
            
        Returns:
            result_df (pd.DataFrame): DataFrame of the results.
        """
        self.run_baseline()

        module_lists = [copy.deepcopy(self.module_list) for _ in range(self.n_samples)]

        module_results = Parallel(n_jobs=self.cpu_count, backend = 'loky')(
                delayed(self.run_modules)(mod_list)
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
        result_df = pd.DataFrame(self.result_dict, index=self.n_rules_list)
        result_df.to_csv(fname)
        
        
####################################################################################################


class LambdaExperiment(Experiment):
    """ 
    Perfroms an experiment on an input dataset which measures performance as
    the maximum number of allowed rules is increased. 
    
    Every step forward in this experiment should call 
    a .step_n_rules() method of its modules, which increases the number of rules used by 1.

    Args:
        data (np.ndarray): Input dataset.

        ground_truth_assignment (np.ndarray): Ground truth assignment for the input dataset.

        lambda_array (np.ndarray): Array of lambda values to use in the experiment.
        
        baseline (Baseline): Single baseline model to use and record results for. 
        
        module_list (List[Tuple[Module, Dict[Tuple[float], Dict[str, Any]]]]): List of 
            (module, parameter dictionary) pairs to use and record results for. 
            Each module should be a runnable experiment object, and each parameter dictionary 
            should contain pairs {(i,j,k,..) : {fitting params}} to pass to the module. 
            More specifically, each parameter dictionary key should be a tuple of integers
            representing indices to items from lambda_array. Each value should be a dictionary of 
            of fitting parameters to pass to the module before running. The output of the 
            fitting process for those parameters is then associated 
            each of the items in the corresponding key list. 
        
        measurement_fns (List[Callable]): List of MeasurementFunction objects
            used to compute results.

        n_samples (int): Number of sample trials to run the experiment for.
        
        labels (np.ndarray): Labels for the input dataset (if any). Defaults to None in which case
            data is taken to be unlabeled.
        
        verbose (bool, optional): Allows for optional printing of status. Defaults to False.
        
    Attrs:
        result_dict (Dict[Tuple[str, str, int], NDArray]): Dictionary with keys 
            as tuples of the form (measurement function name, module name, sample number),
            and values which are arrays of measurement results.
    """
    def __init__(
        self, 
        data : NDArray,
        ground_truth_assignment : NDArray,
        lambda_array : NDArray,
        baseline : Baseline,
        module_list : List[Tuple[Module, Dict[Tuple[float], Dict[str, Any]]]],
        n_samples : int,
        cpu_count : int = 1,
        verbose : bool = False,
    ):
        self.ground_truth_assignment = ground_truth_assignment
        self.lambda_array = lambda_array
        super().__init__(
            data = data,
            baseline = baseline,
            module_list = module_list,
            n_samples = n_samples,
            cpu_count = cpu_count,
            verbose = verbose
        )
        
    
    def run_baseline(self):
        """
        Runs the baseline modules, simply finding their assignment matrices instead of 
        computing results.
        """
        bassign = self.baseline.assign(self.data)
        self.result_dict[("max-rule-length", self.baseline.name, 0)] = {
            i : self.baseline.max_rule_length for i in self.lambda_array
        }
        self.result_dict[("weighted-average-rule-length", self.baseline.name, 0)] = {
            i : self.baseline.weighted_average_rule_length for i in self.lambda_array
        }
        cover = coverage(assignment = bassign, percentage = False)
        self.result_dict[('coverage-mistake-score', self.baseline.name, 0)] = {
            l : cover for l in self.lambda_array
        }

            
    def run_modules(
            self,
            module_list : List[Tuple[Module, Dict[Tuple[float], Dict[str, Any]]]]
        ) -> Dict[Tuple[str, str], Dict[int, float]]:
        """
        Runs the module, and the baseline alongside it. 
        
        Args:
            module_list (List[Tuple[Module, Dict[Tuple[float], Dict[str, Any]]]]): List of 
                (module, parameter dictionary) pairs to use and record results for. 
                Each module should be a runnable experiment object, and each parameter dictionary 
                should contain pairs {(i,j,k,..) : {fitting params}} to pass to the module. 
                More specifically, each parameter dictionary key should be a tuple of integers
                representing indices for items from lambda_array. Each value should be a dictionary of 
                of fitting parameters to pass to the module before running. The output of the 
                fitting process for those parameters is then associated 
                each of the items in the corresponding key list. 

        Returns:
            module_result_dict (Dict[Tuple[str, str], Dict[int, float]]): Dictionary of results 
                in the form  {(measurement name, module name): {lambda_val : measurement result}}
        """
        # Initialize result dictionaries
        module_result_dict = {}
        for mod, param_list in module_list:
            module_result_dict[("max-rule-length", mod.name)] = {}
            module_result_dict[("weighted-average-rule-length", mod.name)] = {}
            module_result_dict[('coverage-mistake-score', mod.name)] = {}

        for mod, param_dict in module_list:
            for n_lambda_tuple, fitting_params in param_dict.items():
                mod.update_fitting_params(fitting_params)
                try:
                    (
                        data_to_rule_assignment,
                        rule_to_cluster_assignment,
                        data_to_cluster_assignment
                    ) = mod.fit(self.data, self.baseline.labels)
                except:
                    print("Data: ")
                    print(self.data)
                    print()
                    print("Labels: ")
                    print(self.baseline.labels)
                    print()
                    raise ValueError("Fitting failed.")
                
                # record measurements:
                for i in n_lambda_tuple:
                    module_result_dict[("max-rule-length", mod.name)][i] = mod.max_rule_length
                    module_result_dict[("weighted-average-rule-length", mod.name)][i] = mod.weighted_average_rule_length
                    module_result_dict[('coverage-mistake-score', mod.name)][i] = coverage_mistake_score(
                        lambda_val = self.lambda_array[i],
                        ground_truth_assignment = self.ground_truth_assignment,
                        data_to_rule_assignment = data_to_rule_assignment,
                        rule_to_cluster_assignment = rule_to_cluster_assignment
                    )
                        
        return module_result_dict
                    
        
    def run(self):
        """
        Runs the experiment.
            
        Args:
            n_steps (int): Number of steps to run the experiment for.
            
        Returns:
            result_df (pd.DataFrame): DataFrame of the results.
        """
        self.run_baseline()

        module_lists = [copy.deepcopy(self.module_list) for _ in range(self.n_samples)]

        module_results = Parallel(n_jobs=self.cpu_count, backend = 'loky')(
                delayed(self.run_modules)(mod_list)
                for mod_list in module_lists
        )

        for i, module_result_dict in enumerate(module_results):
            for key,value in module_result_dict.items():
                self.result_dict[key + (i,)] = value
            
        return pd.DataFrame(self.result_dict, index=self.lambda_array)
    
    
    def save_results(self, path, identifier = ''):
        """
        Saves the results of the experiment.
        
        Args:
            path (str): File path to save the results to.
            
            identifier (str, optional): Unique identifier for the results. Defaults to blank.
        """
        fname = os.path.join(path, 'exp' + str(identifier) + '.csv')
        result_df = pd.DataFrame(self.result_dict, index=self.lambda_array)
        result_df.to_csv(fname)
        


####################################################################################################

'''
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

        # NOTE: Test to see how this works...
        for mod in module_list:
            mod.reset()

        for i in range(n_steps):
            print("Step " + str(i) + ": ")
            for mod in module_list:
                start = time.time()
                massign, mcenters = mod.step_coverage(self.data, self.labels, step_size = step_size)
                end = time.time()
                if i == 0:
                    print(mod.name + " fitting: " + str(end - start))
                else:
                    print(mod.name + " pruning: " + str(end - start))
                
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
            print()
                        
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
        #
        #module_results = None
        #with parallel_config(backend = 'loky', inner_max_num_threads = self.thread_count):
        #    module_results = Parallel(n_jobs=self.cpu_count)(
        #            delayed(self.run_modules)(mod_list, n_steps, step_size)
        #            for mod_list in module_lists
        #    )
        #
        

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