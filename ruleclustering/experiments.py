import os
import numpy as np
import pandas as pd
from .tree import *
from .decision_sets import *
from .rule_clustering import *
from .utils import *
from .modules import *


####################################################################################################

class Experiment:
    """
    Performs suite of experiments on an input dataset which measure clustering cost with parameter 
    changes.
    """
    def __init__(
        self, 
        data,
        baseline_list,
        module_list,
        cost_fn,
        cost_params,
        random_seed = None,
        verbose = True
    ):
        """
        Args:
            data (np.ndarray): Input dataset.
            
            n_clusters (int): Number of clusters to use.
            
            random_seed (int, optional): Random seed for experiments. Defaults to None.
            
            verbose (bool, optional): Allows for printing of status.
            
        Attrs:
            kmeans_centers (np.ndarray): (n x k) Array of reference centers found from a 
                run of kmeans.
                
            
        """
        self.data = data
        self.baseline_list = baseline_list
        self.module_list = module_list
        self.cost_fn = cost_fn
        self.cost_params = cost_params
        self.seed = random_seed
        self.verbose = verbose
        np.random.seed(random_seed)
        
        self.cost_df = None

    def run_baseline(self):
        pass 
    
    def run_module(self):
        pass
    
    def run(self):
        pass
    
    def save_results(self):
        pass
    
    
####################################################################################################


class RulesExperiment(Experiment):
    """
    Performs suite of experiments on an input dataset which measure clustering cost with parameter 
    changes.
    """
    def __init__(
        self, 
        data,
        baseline_list,
        module_list,
        cost_fn,
        cost_params,
        random_seed = None,
        verbose = True
    ):
        super().__init__(
            data,
            baseline_list,
            module_list,
            cost_fn,
            cost_params,
            random_seed,
            verbose
        )
        self.cost_dict = {}
        for b in self.baseline_list:
            self.cost_dict[b.name] = []
            
        for m in self.module_list:
            self.cost_dict[m.name] = []
            
        
    def run_baselines(self, n_rules_list):
        """
        Initializes the experiment.
        """
        for b in self.baseline_list:
            bassign, bcenters = b.assign(self.data)
            self.cost_dict[b.name] = [
                self.cost_fn(self.data, bassign, bcenters, **self.cost_params)
            ] * len(n_rules_list)
        
    def run_modules(self, n_rules_list):
        for i in n_rules_list:
            if self.verbose:
                print(f"Running for {i} rules.")
            for m in self.module_list:
                massign, mcenters = m.step_num_rules(self.data)
                self.cost_dict[m.name].append(
                    self.cost_fn(self.data, massign, mcenters, **self.cost_params)
                )
            
    def run(self, min_rules, max_rules):
        """
        Runs the experiment.
        """
        n_rules_list = list(range(min_rules, max_rules + 1))
        self.run_baselines(n_rules_list)
        self.run_modules(n_rules_list)
        
        # reset:
        for m in self.module_list:
            m.reset()
            
        self.cost_df = pd.DataFrame(self.cost_dict)
    
    def save_results(self, path, identifier = ''):
        """
        Saves the results of the experiment.
        """
        fname = os.path.join(path, 'rules_cost' + str(identifier) + '.csv')
        self.cost_df.to_csv(fname)
        
        
####################################################################################################
        
class DepthExperiment(Experiment):
    """
    Performs suite of experiments on an input dataset which measure clustering cost with parameter 
    changes.
    """
    def __init__(
        self, 
        data,
        baseline_list,
        module_list,
        cost_fn,
        cost_params,
        random_seed = None,
        verbose = True
    ):
        super().__init__(
            data,
            baseline_list,
            module_list,
            cost_fn,
            cost_params,
            random_seed,
            verbose
        )
        self.cost_dict = {}
        for b in self.baseline_list:
            self.cost_dict[b.name] = []
            
        for m in self.module_list:
            self.cost_dict[m.name] = []
            
        
    def run_baselines(self, n_depth_list):
        """
        Initializes the experiment.
        """
        for b in self.baseline_list:
            bassign, bcenters = b.assign(self.data)
            self.cost_dict[b.name] = [
                self.cost_fn(self.data, bassign, bcenters, **self.cost_params)
            ] * len(n_depth_list)
        
    def run_modules(self, n_depth_list):
        for i in n_depth_list:
            if self.verbose:
                print(f"Running for depth {i}.")
            for m in self.module_list:
                massign, mcenters = m.step_depth(self.data)
                self.cost_dict[m.name].append(
                    self.cost_fn(self.data, massign, mcenters, **self.cost_params)
                )
            
    def run(self, min_depth, max_depth):
        """
        Runs the experiment.
        """
        n_depth_list = list(range(min_depth, max_depth + 1))
        self.run_baselines(n_depth_list)
        self.run_modules(n_depth_list)
        
        # reset:
        for m in self.module_list:
            m.reset()
            
        self.cost_df = pd.DataFrame(self.cost_dict)
    
    def save_results(self, path, identifier = ''):
        """
        Saves the results of the experiment.
        """
        fname = os.path.join(path, 'depth_cost' + str(identifier) + '.csv')
        self.cost_df.to_csv(fname)