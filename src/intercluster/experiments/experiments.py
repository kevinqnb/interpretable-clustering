import os
import pandas as pd
import copy
from joblib import Parallel, delayed
from typing import List, Callable, Any
from numpy.typing import NDArray
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.utils import assignment_to_labels
from intercluster.measurements import (
    coverage, coverage_mistake_score, clustering_distance
)
from .modules import *
from .measurements import *
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

        n_samples (List[int]): List in which each entry i represents the 
            number of sample trials to run module i for.
        
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
        n_samples : List[int],
        labels : List[List[int]] = None,
        cpu_count : int = 1,
        verbose : bool = False
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
        
    
    def run_baseline(self):
        """
        Runs the baseline modules, simply finding their assignment matrices instead of 
        computing results.
        """
        bassign = self.baseline.assign(self.data)
        self.result_dict[("max-rule-length", self.baseline.name, 0)] = {
            i : self.baseline.max_rule_length for i in self.n_rules_list
        }
        self.result_dict[("weighted-avg-length", self.baseline.name, 0)] = {
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
            module_result_dict[("weighted-avg-length", mod.name)] = {}
            for fn in self.measurement_fns:
                module_result_dict[(fn.name, mod.name)] = {}

        for mod, param_dict in module_list:
            mod.reset()
            for n_rules_tuple, fitting_params in param_dict.items():
                if self.verbose:
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
                    if self.verbose:
                        print(mod.name + " fitting time: " + str(end - start))
                except:
                    if self.verbose:
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
                        module_result_dict[("weighted-avg-length", mod.name)][i] = mod.weighted_average_rule_length
                        
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

        module_lists = [
            [copy.deepcopy(self.module_list[i])] * self.n_samples[i]
            for i in range(len(self.module_list))
        ]
        #module_lists = [copy.deepcopy(self.module_list) for _ in range(self.n_samples)]

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
        self.result_dict[("weighted-avg-length", self.baseline.name, 0)] = {
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
            module_result_dict[("weighted-avg-length", mod.name)] = {}
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
                    module_result_dict[("weighted-avg-length", mod.name)][i] = mod.weighted_average_rule_length
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


class RobustnessExperiment(Experiment):
    """
    Performs an experiment in which the robustness of a clustering is measured 
    by its consistency with respect to small perturbations of the data. Specifically,
    consistency is measured as the fraction of pairs of points which are assigned to the same
    cluster in one clustering and different clusters in the other.

    Args:
        data (np.ndarray): Input dataset.
        
        module_list (List[Tuple[Module, Dict[str, Any]]]): List of 
            (module, parameter dictionary) pairs to use and record results for. 
            Each module should be a runnable experiment object, and each parameter dictionary 
            should contain pairs {(i,j,k,..) : {fitting params}} to pass to the module. 
            More specifically, each parameter dictionary key should be a tuple of integers
            representing indices to items from lambda_array. Each value should be a dictionary of 
            of fitting parameters to pass to the module before running. The output of the 
            fitting process for those parameters is then associated 
            each of the items in the corresponding key list. 

        std_dev (float): Standard deviation of the Gaussian noise to add to the data. Note that 
            sampled noise is added to every data point and every feature independently.

        n_samples (int): Number of sample trials to run the experiment for.
        
    Attrs:
        result_dict (Dict[Tuple[str, str, int], NDArray]): Dictionary with keys 
            as tuples of the form (measurement function name, module name, sample number),
            and values which are arrays of measurement results.
    """
    def __init__(
        self, 
        data : NDArray,
        baseline : Baseline,
        module_list : List[Tuple[Module, Dict[str, Any]]],
        std_dev : float,
        n_samples : int,
        ignore = {-1}
    ):
        self.std_dev = std_dev
        self.ignore = ignore
        super().__init__(
            data = data,
            baseline = baseline,
            module_list = module_list,
            n_samples = n_samples
        )


    def run_baseline(self):
        """
        Runs the baseline modules, simply finding their assignment matrices instead of 
        computing results.
        """
        bassign = self.baseline.assign(self.data)


    def run_modules(
            self,
            module_list : List[Tuple[Module, Dict[str, Any]]]
        ) -> Dict[Tuple[str, str], Dict[float, float]]:
        """
        Runs the module, and the baseline alongside it. 
        
        Args:
            module_list (List[Tuple[Module, Dict[str, Any]]]): List of 
                (module, parameter dictionary) pairs to use and record results for. 
                Each module should be a runnable experiment object, and each parameter dictionary 
                should contain pairs {(i,j,k,..) : {fitting params}} to pass to the module. 
                More specifically, each parameter dictionary key should be a tuple of integers
                representing indices for items from lambda_array. Each value should be a dictionary of 
                of fitting parameters to pass to the module before running. The output of the 
                fitting process for those parameters is then associated 
                each of the items in the corresponding key list.
        Returns:
            module_result_dict (Dict[Tuple[str, str], Dict[float, float]]): Dictionary of results 
                in the form  {(measurement name, module name): {std_dev : measurement result}}
        """
        # Initialize result dictionaries
        module_result_dict = {}
        module_label_dict = {}
        for mod, param_list in module_list:
            module_result_dict[("clustering-distance", mod.name)] = {}
            module_label_dict[mod.name] = {}
            

        for mod, fitting_params in module_list:
            mod.update_fitting_params(fitting_params)
            (
            data_to_rule_assignment,
            rule_to_cluster_assignment,
            data_to_cluster_assignment
            ) = mod.fit(self.data, self.baseline.labels)
            module_label_dict[mod.name] = assignment_to_labels(data_to_cluster_assignment)


        def dist_sample():
            sample_dict = {}
            noisy_data = self.data + np.random.normal(
                loc = 0.0,
                scale = self.std_dev,
                size = self.data.shape
            )
            for mod, fitting_params in module_list:
                noisy_labels = mod.predict(noisy_data)
                dist = clustering_distance(
                    labels1 = module_label_dict[mod.name],
                    labels2 = noisy_labels,
                    percentage = True,
                    ignore = self.ignore
                )
                sample_dict[mod.name] = dist
            return sample_dict
        
        module_results = Parallel(n_jobs=self.cpu_count, backend = 'loky')(
                delayed(dist_sample)
                for i in range(self.n_samples)
        )

        for i in range(self.n_samples):
            for mod, fitting_params in module_list:
                dist = module_results[i][mod.name]
                module_result_dict[("clustering-distance", mod.name)][i] = dist

        '''
        for i in range(self.n_samples):
            noisy_data = self.data + np.random.normal(
                loc = 0.0,
                scale = self.std_dev,
                size = self.data.shape
            )
            for mod, fitting_params in module_list:
                noisy_labels = mod.predict(noisy_data)
                dist = clustering_distance(
                    labels1 = module_label_dict[mod.name],
                    labels2 = noisy_labels,
                    percentage = True,
                    ignore = self.ignore
                )
                module_result_dict[("clustering-distance", mod.name)][i] = dist
        '''
                        
        return module_result_dict
    
    
    def run(self):
        """
        Runs the experiment.
            
        Returns:
            result_df (pd.DataFrame): DataFrame of the results.
        """
        self.run_baseline()
        module_results = self.run_modules(self.module_list)
        self.result_dict = module_results
        return pd.DataFrame(module_results)
    

    def save_results(self, path, identifier = ''):
        """
        Saves the results of the experiment.
        
        Args:
            path (str): File path to save the results to.
            
            identifier (str, optional): Unique identifier for the results. Defaults to blank.
        """
        fname = os.path.join(path, 'exp' + str(identifier) + '.csv')
        result_df = pd.DataFrame(self.result_dict)
        result_df.to_csv(fname)


####################################################################################################