import os
import json
import pandas as pd
import copy
from joblib import Parallel, delayed
from typing import List, Callable, Any
from numpy.typing import NDArray
from .modules import *
import time

####################################################################################################


class Experiment:
    """ 
    Perfroms an experiment comparing baseline and module performance as
    across a suite of given measurement functions.

    Args:
        data (np.ndarray): Input dataset.
        
        baseline (Baseline): Single baseline model to use and record results for. 
        
        module_list (List[Tuple[Module, Dict[Tuple[float], Dict[str, Any]]]]): List of 
            (module, parameter dictionary) pairs to use and record results for. 
            Each module should be a runnable experiment object, and each parameter dictionary 
            should contain pairs {(i,j,k,..) : {fitting params}} to pass to the module. 
            More specifically, each parameter dictionary key should be a tuple of values 
            representing some varying model parameters. Each value should be a 
            corresponding dictionary with of input fitting parameters to pass to the module. 
            The output of the fitting process for those parameters is then associated 
            each of the items in the corresponding key list. 
        
        measurement_fns (List[Callable]): List of MeasurementFunction objects
            used to compute results.

        fixed_parameters (dict[str, any], optional): Dictionary of fixed parameters to 
            use throughout the experiment. Defaults to {}.

        cpu_count (int, optional): Number of CPU cores to use. Defaults to 1.
        
        verbose (bool, optional): Allows for optional printing of status. Defaults to False.
        
    Attrs:
        result_dict (Dict): Dictionary with key, value pairs for 'baseline', 'modules',
            and 'fixed-parameters'. The value for 'baseline' is itself a dictionary of results
            in the form {measurement name : measurement result}. The value for 'modules' is
            a dictionary of results in the form 
            {module name : {measurement name : {lambda value : measurement result}}}.
            The value for 'fixed-parameters' is a dictionary of fixed parameters used
            throughout the experiment.
    """
    def __init__(
        self, 
        data : NDArray,
        baseline : Baseline,
        module_list : List[Tuple[Module, Dict[Tuple[float], Dict[str, Any]]]],
        measurement_fns : List[Callable],
        fixed_parameters : dict[str, any] = {},
        cpu_count : int = 1,
        verbose : bool = False,
    ):
        self.data = data
        self.baseline = baseline
        self.module_list = module_list
        self.measurement_fns = measurement_fns
        self.fixed_parameters = fixed_parameters
        self.cpu_count = cpu_count
        self.verbose = verbose
        self.result_dict = {'fixed-parameters': fixed_parameters}


    def run_baseline(self) -> dict[str, dict[str, any]]:
        """
        Runs the baseline modules, simply finding their assignment matrices instead of 
        computing results.

        Returns: 
            result_dict (dict[str, dict[str, any]]): Dictionary of results
                in the form {baseline name : {measurement name : measurement result}}
        """
        bassign = self.baseline.assign(self.data)

        result_dict = {
            self.baseline.name :
                {
                    'lambda' : None,
                    'max-rule-length' : self.baseline.max_rule_length,
                    'sum-rule-length' : self.baseline.sum_rule_length,
                    'weighted-avg-length' : self.baseline.weighted_average_rule_length
                }
        }

        for fn in self.measurement_fns:
            fn_result = fn(
                data_to_rule_assignment = None,
                rule_to_cluster_assignment = None,
                data_to_cluster_assignment = bassign
            )
            result_dict[self.baseline.name][fn.name] = fn_result    

        if self.verbose:
            print(self.baseline.name + " baseline assignment fitted.")
            print()

        return result_dict

            
    def run_module(
            self,
            module : Module,
            param_dict : dict[tuple[float], dict[str, any]]
        ) -> dict[str, dict[str, dict[float, any]]]:
        """
        Runs the module, and the baseline alongside it. 
        
        Args:
            module (Module): Module to run the experiment with.
            
            param_dict (Dict[Tuple[float], Dict[str, Any]]): Parameter dictionary
                containing pairs {(i,j,k,..) : {fitting params}} to pass to the module. 
                More specifically, each parameter dictionary key should be a tuple of lambda values,
                and each value should be a dictionary of fitting parameters to use 
                for those lambda values. The output of the fitting process for those parameters
                is then associated each of the items in the corresponding key list.

        Returns:
            module_result_dict (dict[str, dict[str, dict[float, float]]]): Dictionary of results
                in the form {module name : {measurement name : {lambda value : measurement result}}}
        """
        if self.verbose:
            print(f"Running module " + module.name + "...")
            start = time.time()

        # Initialize result dictionaries
        module_result_dict = {
            module.name :
                {
                    'lambda' : {},
                    'max-rule-length' : {},
                    'sum-rule-length' : {},
                    'weighted-avg-length' : {}
                } |
                {
                    fn.name : {}
                    for fn in self.measurement_fns
                }
        }

        for param_tuple, fitting_params in param_dict.items():
            # Fit module with given parameters:
            module.update_fitting_params(fitting_params)
            (
                data_to_rule_assignment,
                rule_to_cluster_assignment,
                data_to_cluster_assignment
            ) = module.fit(self.data, self.baseline.labels)

            
            # Record measurements:
            for p in param_tuple:
                module_result_dict[module.name]['lambda'][p] = module.lambda_val if hasattr(module, 'lambda_val') else None
                module_result_dict[module.name]['max-rule-length'][p] = module.max_rule_length
                module_result_dict[module.name]['sum-rule-length'][p] = module.sum_rule_length
                module_result_dict[module.name]['weighted-avg-length'][p] = module.weighted_average_rule_length
                
                for fn in self.measurement_fns:
                    module_result_dict[module.name][fn.name][p] = (
                        fn(
                            data_to_rule_assignment,
                            rule_to_cluster_assignment,
                            data_to_cluster_assignment
                        )
                    )

        if self.verbose:
            end = time.time()
            print(f"Module " + module.name + " complete in " + str(end - start) + "(s).")
            print()
                        
        return module_result_dict
                    
        
    def run(self):
        """
        Runs the experiment.
            
        Args:
            n_steps (int): Number of steps to run the experiment for.
            
        Returns:
            result_df (pd.DataFrame): DataFrame of the results.
        """
        if self.verbose:
            print("Running baseline...")

        baseline_dict = {'baseline': self.run_baseline()}

        if self.verbose:
            print(f"Running modules in parallel with {self.cpu_count} cores...")
        module_results = Parallel(n_jobs=self.cpu_count, backend = 'loky')(
                delayed(self.run_module)(mod, mod_params)
                for mod, mod_params in self.module_list
        )
        module_results_dict = {
            'modules': {}
        }
        for i, mod_dict in enumerate(module_results):
            module_results_dict['modules'] = module_results_dict['modules'] | mod_dict

        # Combine results into single result dictionary:
        self.result_dict = self.result_dict | baseline_dict | module_results_dict
            
        return self.result_dict
    
    
    def save_results(self, path, identifier = ''):
        """
        Saves the results of the experiment as a JSON file.
        
        Args:
            path (str): File path to save the results to.
            
            identifier (str, optional): Unique identifier for the results. Defaults to blank.
        """
        import math
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    # Check for NaN
                    if math.isnan(obj):
                        return None
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
            
        fname = os.path.join(path, 'exp' + str(identifier) + '.json')
        with open(fname, 'w') as f:
            json.dump(self.result_dict, f, indent=4, cls=NumpyEncoder)


####################################################################################################