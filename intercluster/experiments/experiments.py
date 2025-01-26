import os
import pandas as pd
from typing import List, Callable, Any
from numpy.typing import NDArray
from intercluster.rules import *
from .modules import *

####################################################################################################

class Experiment:
    """
    Default class for performing suite of experiments on an input dataset which measure
    clustering cost with parameter changes.
    
    Args:
        data (np.ndarray): Input dataset.
        
        labels (np.ndarray): Labels for the input dataset.
        
        baseline_list (List[Any]): List of baseline modules to use and record results for. 
        
        measurement_fns (List[Callable]): List of MeasurementFunction objects
            used to compute results.
            
        n_samples (int): Number of samples to run each module for.
        
        random_seed (int, optional): Random seed for experiments. Defaults to None.
        
        verbose (bool, optional): Allows for printing of status. Defaults to True.
    
    Attrs: 
        result_dict (Dict[str, List[float]): Dictionary to store costs for each module and baseline.
    """
    def __init__(
        self, 
        data : NDArray,
        baseline_list : List[Any],
        module_list : List[Any],
        measurement_fns : List[Callable],
        n_samples : int,
        labels : NDArray = None,
        random_seed : int = None,
        verbose : bool = True
    ):
        self.data = data
        self.labels = labels
        self.baseline_list = baseline_list
        self.module_list = module_list
        self.measurement_fns = measurement_fns
        self.n_samples = n_samples
        self.random_seed = random_seed
        # NOTE: Does this need to be here??
        #np.random.seed(random_seed)
        self.verbose = verbose
        
        self.result_dict = {}
        for b in baseline_list:
            for fn in measurement_fns:
                for s in range(n_samples):
                    self.result_dict[(fn.name, b.name, s)] = []
                
        for m in self.module_list:
            for s in range(n_samples):
                self.result_dict[("depth", m.name, s)] = []
                for fn in measurement_fns:
                    self.result_dict[(fn.name, m.name, s)] = []
                

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
        baseline_list,
        module_list,
        measurement_fns,
        n_samples,
        labels = None,
        random_seed = None,
        verbose = True
    ):
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
                    self.result_dict[(fn.name, b.name, s)] = [
                        fn(self.data, bassign, bcenters)
                    ] * len(n_rules_list)
            
    def run_modules(self, n_rules_list : List[int], sample_number : int):
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
                
                # record depth:
                self.result_dict[("depth", m.name, sample_number)].append(m.n_depth)
                
                # record results from measurement functions:
                for fn in self.measurement_fns:
                    self.result_dict[(fn.name, m.name, sample_number)].append(
                        fn(self.data, massign, mcenters)
                    )
        
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
        
        for s in range(self.n_samples):
            if self.verbose:
                print(f"Running for sample {s}.")
                
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
        fname = os.path.join(path, 'rules_cost' + str(identifier) + '.csv')
        cost_df = pd.DataFrame(self.result_dict)
        cost_df.to_csv(fname)
        
        
####################################################################################################