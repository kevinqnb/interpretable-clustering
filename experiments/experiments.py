import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Callable, Any
from numpy.typing import NDArray
from intercluster.clustering import *
from intercluster.rules import *
from intercluster import *
from modules import *

####################################################################################################

class Experiment:
    """
    Default class for performing suite of experiments on an input dataset which measure
    clustering cost with parameter changes.
    
    Args:
        data (np.ndarray): Input dataset.
        
        labels (np.ndarray): Labels for the input dataset.
        
        baseline_list (List[Any]): List of baseline modules to use and record results for. 
        
        module_list (List[Any]): List of modules to use and record results for.
        
        cost_fns (Dict[str, Callable]): Functions to use to calculate cost. Formatted as 
            a dictionary such as {'cost_function_name': cost_function}.
        
        cost_fn_params (Dict[str, Dict[str, Any]]): Parameters to pass to cost function. Formatted 
            as a dictionary such as {'cost_function_name': {'param1': value1, 'param2': value2}}.
            Names should be the same as those in cost_fns.
        
        random_seed (int, optional): Random seed for experiments. Defaults to None.
        
        verbose (bool, optional): Allows for printing of status. Defaults to True.
    
    Attrs: 
        cost_dict (Dict[str, List[float]): Dictionary to store costs for each module and baseline.
    """
    def __init__(
        self, 
        data : NDArray,
        baseline_list : List[Any],
        module_list : List[Any],
        cost_fns : Dict[str, Callable],
        cost_fn_params : Dict[str, Dict[str, Any]],
        labels : NDArray = None,
        random_seed : int = None,
        verbose : bool = True
    ):
        self.data = data
        self.labels = labels
        self.baseline_list = baseline_list
        self.module_list = module_list
        
        if cost_fns.keys() != cost_fn_params.keys():
            raise ValueError("Cost function names do not match cost function parameters.")
        
        self.cost_fns = cost_fns
        self.cost_fn_params = cost_fn_params
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.verbose = verbose
        
        self.cost_dict = {}
        for k in self.cost_fns.keys():
            for b in self.baseline_list:
                self.cost_dict[(k, b.name)] = []
                
            for m in self.module_list:
                self.cost_dict[(k, m.name)] = []

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


class RulesExperiment(Experiment):
    """
    Perfroms an experiment on an input dataset which measures clustering cost as the 
    number of rules changes.
    
    Args:
        data (np.ndarray): Input dataset.
        
        labels (np.ndarray): Labels for the input dataset.
        
        baseline_list (List[Any]): List of baseline modules to use and record results for. 
        
        module_list (List[Any]): List of modules to use and record results for.
        
        cost_fns (Dict[str, Callable]): Functions to use to calculate cost. Formatted as 
            a dictionary such as {'cost_function_name': cost_function}.
        
        cost_fn_params (Dict[str, Dict[str, Any]]): Parameters to pass to cost function. Formatted 
            as a dictionary such as {'cost_function_name': {'param1': value1, 'param2': value2}}.
            Names should be the same as those in cost_fns.
        
        random_seed (int, optional): Random seed for experiments. Defaults to None.
        
        verbose (bool, optional): Allows for printing of status. Defaults to True.
        
    Attrs:
        cost_dict (Dict[str, List[float]): Dictionary to store costs for each module and baseline.
    """
    def __init__(
        self, 
        data,
        baseline_list,
        module_list,
        cost_fns,
        cost_fn_params,
        labels = None,
        random_seed = None,
        verbose = True
    ):
        super().__init__(
            data = data,
            baseline_list = baseline_list,
            module_list = module_list,
            cost_fns = cost_fns,
            cost_fn_params = cost_fn_params,
            labels = labels,
            random_seed = random_seed,
            verbose = verbose
        )
            
        
    def run_baselines(self, n_rules_list : List[int]):
        """
        Runs the baseline modules.
        
        Args:
            n_rules_list (List[int]): List with varying numbers of rules to run the module for.
        """
        for b in self.baseline_list:
            bassign, bcenters = b.assign(self.data)
            for k in self.cost_fns.keys():
                cost_fn = self.cost_fns[k]
                self.cost_dict[(k, b.name)] = [
                    cost_fn(self.data, bassign, bcenters, **self.cost_fn_params[k])
                ] * len(n_rules_list)
        
    def run_modules(self, n_rules_list : List[int]):
        """
        Runs the modules.
        
        Args:
            n_rules_list (List[int]): List with varying numbers of rules to run the module for.
        """
        for i in n_rules_list:
            if self.verbose:
                print(f"Running for {i} rules.")
            for m in self.module_list:
                massign, mcenters = m.step_num_rules(self.data, self.labels)
                for k in self.cost_fns.keys():
                    cost_fn = self.cost_fns[k]
                    try:
                        self.cost_dict[(k, m.name)].append(
                            cost_fn(self.data, massign, mcenters, **self.cost_fn_params[k])
                        )
                    except:
                        breakpoint()
        
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
        self.run_baselines(n_rules_list)
        self.run_modules(n_rules_list)
        
        # reset the modules:
        for m in self.module_list:
            m.reset()
            
        return pd.DataFrame(self.cost_dict)
    
    def save_results(self, path, identifier = ''):
        """
        Saves the results of the experiment.
        
        Args:
            path (str): File path to save the results to.
            
            identifier (str, optional): Unique identifier for the results. Defaults to blank.
        """
        fname = os.path.join(path, 'rules_cost' + str(identifier) + '.csv')
        cost_df = pd.DataFrame(self.cost_dict)
        cost_df.to_csv(fname)
        
        
####################################################################################################

        
class DepthExperiment(Experiment):
    """
    Performs suite of experiments on an input dataset which measures clustering cost as the 
    depth of the rules changes.
    
    Args:
        data (np.ndarray): Input dataset.
        
        baseline_list (List[Any]): List of baseline modules to use and record results for. 
        
        module_list (List[Any]): List of modules to use and record results for.
        
        cost_fns (Dict[str, Callable]): Functions to use to calculate cost. Formatted as 
            a dictionary such as {'cost_function_name': cost_function}.
        
        cost_fn_params (Dict[str, Dict[str, Any]]): Parameters to pass to cost function. Formatted 
            as a dictionary such as {'cost_function_name': {'param1': value1, 'param2': value2}}.
            Names should be the same as those in cost_fns.
        
        random_seed (int, optional): Random seed for experiments. Defaults to None.
        
        verbose (bool, optional): Allows for printing of status. Defaults to True.
    """
    def __init__(
        self, 
        data,
        baseline_list,
        module_list,
        cost_fns,
        cost_fn_params,
        random_seed = None,
        verbose = True
    ):
        super().__init__(
            data,
            baseline_list,
            module_list,
            cost_fns,
            cost_fn_params,
            random_seed,
            verbose
        )
            
        
    def run_baselines(self, n_depth_list : List[int]):
        """
        Runs the baseline modules.
        
        Args:
            n_depth_list (List[int]): List with varying depths to run the module for.
        """
        for b in self.baseline_list:
            bassign, bcenters = b.assign(self.data)
            for k in self.cost_fns.keys():
                cost_fn = self.cost_fns[k]
                self.cost_dict[(k, b.name)] = [
                    cost_fn(self.data, bassign, bcenters, **self.cost_fn_params[k])
                ] * len(n_depth_list)
        
    def run_modules(self, n_depth_list : List[int]):
        """
        Runs the modules.
        
        Args:
            n_depth_list (List[int]): List with varying depths to run the module for.
        """
        for i in n_depth_list:
            if self.verbose:
                print(f"Running for depth {i}.")
            for m in self.module_list:
                massign, mcenters = m.step_depth(self.data)
                for k in self.cost_fns.keys():
                    cost_fn = self.cost_fns[k]
                    self.cost_dict[(k, m.name)].append(
                        cost_fn(self.data, massign, mcenters, **self.cost_fn_params[k])
                    )
            
    def run(self, min_depth : int, max_depth : int):
        """
        Runs the experiment.
        
        Args:
            min_depth (int): Minimum depth to fit with.
            
            max_depth (int): Maximum depth to fit with.
            
        Returns:
            cost_df (pd.DataFrame): DataFrame of the results.
        """
        n_depth_list = list(range(min_depth, max_depth + 1))
        self.run_baselines(n_depth_list)
        self.run_modules(n_depth_list)
        
        # reset the modules:
        for m in self.module_list:
            m.reset()
            
        return pd.DataFrame(self.cost_dict)
        
    
    def save_results(self, path : str, identifier : str = ''):
        """
        Saves the results of the experiment.
        
        Args:
            path (str): File path to save the results to.
            
            identifier (str, optional): Unique identifier for the results. Defaults to blank.
        """
        fname = os.path.join(path, 'depth_cost' + str(identifier) + '.csv')
        cost_df = pd.DataFrame(self.cost_dict)
        cost_df.to_csv(fname)