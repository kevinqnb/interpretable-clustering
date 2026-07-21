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


def _run_fit(
    module : Module,
    param_tuple : tuple,
    fitting_params : dict[str, any],
    data : NDArray,
    labels : list,
    measurement_fns : List[Callable],
    profile : bool = False,
) -> dict[str, any]:
    """
    Fits one module at one parameter setting and measures the result. This is the unit of work
    dispatched to a joblib worker (see Experiment.run).

    Deliberately a module-level function rather than an Experiment method: joblib pickles the
    callable together with its arguments, so dispatching a bound method would ship the whole
    Experiment -- including `module_list`, i.e. every module's rule pool -- to every task. Taking
    the single module it needs keeps the per-task payload small. `data` and the arrays held by
    the measurement functions are ndarrays, which joblib memmaps and de-duplicates by identity
    across tasks rather than re-pickling.

    The measurement suite is computed ONCE here and returned for the caller to broadcast across
    every entry of `param_tuple`. The fitted assignments do not depend on which label in the tuple
    we are recording, so evaluating the suite per label (as this loop used to) just recomputed the
    same numbers -- costly when a module is keyed by a long tuple, as the comparison modules in
    lambda.py are.

    Returns:
        record (dict): The fit's results, keyed for merging by Experiment.run.
    """
    prof = {'fit': [0.0, 0]} | {f'meas:{fn.name}': [0.0, 0] for fn in measurement_fns}

    module.update_fitting_params(fitting_params)

    if profile:
        _t = time.perf_counter()
        assignments = module.fit(data, labels)
        prof['fit'][0] += time.perf_counter() - _t
        prof['fit'][1] += 1
    else:
        assignments = module.fit(data, labels)

    (
        data_to_rule_assignment,
        rule_to_cluster_assignment,
        data_to_cluster_assignment
    ) = assignments

    measurements = {}
    for fn in measurement_fns:
        if profile:
            _t = time.perf_counter()
            measurements[fn.name] = fn(
                data_to_rule_assignment,
                rule_to_cluster_assignment,
                data_to_cluster_assignment
            )
            prof[f'meas:{fn.name}'][0] += time.perf_counter() - _t
            prof[f'meas:{fn.name}'][1] += 1
        else:
            measurements[fn.name] = fn(
                data_to_rule_assignment,
                rule_to_cluster_assignment,
                data_to_cluster_assignment
            )

    return {
        'module': module.name,
        'param_tuple': param_tuple,
        'lambda': module.lambda_val if hasattr(module, 'lambda_val') else None,
        'lambda_n_rules': getattr(module, 'n_available_decisions', np.nan),
        'max-rule-length': module.max_rule_length,
        'sum-rule-length': module.sum_rule_length,
        'weighted-avg-length': module.weighted_average_rule_length,
        'rule-source-counts': getattr(module, 'rule_source_counts', None),
        'measurements': measurements,
        '_profile': prof if profile else None,
    }


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
        profile : bool = False,
    ):
        self.data = data
        self.baseline = baseline
        self.module_list = module_list
        self.measurement_fns = measurement_fns
        self.fixed_parameters = fixed_parameters
        self.cpu_count = cpu_count
        self.verbose = verbose
        # When True, _run_fit records a fit-vs-measurement time breakdown
        # (per measurement fn) and attaches it to the returned dict under
        # '_profile' so it survives loky worker dispatch. Off by default.
        self.profile = profile
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
                    'lambda_n_rules' : np.nan,
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

            
    def run(self):
        """
        Runs the experiment.

        Each (module, parameter setting) pair is dispatched as its own joblib task. Dispatching a
        whole module per task instead -- with its parameter sweep run sequentially inside the
        worker -- makes the tasks atomic and wildly unequal: a module with a single fit and a
        module with dozens of fits are both one task, so once the small ones finish their cores
        sit idle while a few workers grind through the long sweeps. Splitting per parameter gives
        joblib many small, comparable tasks to balance across the pool.

        Returns:
            result_dict (dict): The combined results.
        """
        if self.verbose:
            print("Running baseline...")

        baseline_dict = {'baseline': self.run_baseline()}

        # One task per (module, param_tuple). Built in module_list order, and within each module
        # in param_dict order; joblib returns results in submission order, so rebuilding below in
        # this same order reproduces the key insertion order of the sequential implementation.
        tasks = [
            (mod_idx, mod, param_tuple, fitting_params)
            for mod_idx, (mod, mod_params) in enumerate(self.module_list)
            for param_tuple, fitting_params in mod_params.items()
        ]

        if self.verbose:
            print(
                f"Running {len(tasks)} fits from {len(self.module_list)} modules "
                f"in parallel with {self.cpu_count} cores..."
            )
            start = time.time()

        fit_records = Parallel(n_jobs=self.cpu_count, backend = 'loky')(
            delayed(_run_fit)(
                mod,
                param_tuple,
                fitting_params,
                self.data,
                self.baseline.labels,
                self.measurement_fns,
                self.profile,
            )
            for _, mod, param_tuple, fitting_params in tasks
        )

        if self.verbose:
            end = time.time()
            print(f"All modules complete in " + str(end - start) + "(s).")
            print()

        # Per-module result skeletons, one per module_list entry (by index, not by name: a name
        # may legitimately repeat, and the merge below preserves the original 'last one wins'
        # semantics of overwriting an earlier module of the same name wholesale).
        per_module = [
            {
                'lambda' : {},
                'lambda_n_rules' : {},
                'max-rule-length' : {},
                'sum-rule-length' : {},
                'weighted-avg-length' : {},
                'rule-source-counts' : {}
            } |
            {
                fn.name : {}
                for fn in self.measurement_fns
            }
            for _ in self.module_list
        ]

        # Collected per-module fit/measurement timings (when profiling).
        self.profile_results = {}

        for (mod_idx, mod, param_tuple, _), record in zip(tasks, fit_records):
            result = per_module[mod_idx]

            # One fit, one measurement suite -- broadcast across every label in the key tuple.
            for p in param_tuple:
                result['lambda'][p] = record['lambda']
                result['lambda_n_rules'][p] = record['lambda_n_rules']
                result['max-rule-length'][p] = record['max-rule-length']
                result['sum-rule-length'][p] = record['sum-rule-length']
                result['weighted-avg-length'][p] = record['weighted-avg-length']
                result['rule-source-counts'][p] = record['rule-source-counts']
                for fn_name, value in record['measurements'].items():
                    result[fn_name][p] = value

            if record['_profile'] is not None:
                prof = self.profile_results.setdefault(
                    mod.name,
                    {'fit': [0.0, 0]} | {f'meas:{fn.name}': [0.0, 0] for fn in self.measurement_fns},
                )
                for bucket, (secs, n) in record['_profile'].items():
                    prof[bucket][0] += secs
                    prof[bucket][1] += n

        module_results_dict = {'modules': {}}
        for (mod, _), result in zip(self.module_list, per_module):
            module_results_dict['modules'] = module_results_dict['modules'] | {mod.name: result}

        if self.profile and self.profile_results:
            self._report_profile()

        # Combine results into single result dictionary:
        self.result_dict = self.result_dict | baseline_dict | module_results_dict

        return self.result_dict
    
    
    def _report_profile(self):
        """Prints a ranked fit-vs-measurement breakdown across all modules."""
        # Roll up across modules into global buckets, and also keep per-module fit totals.
        global_buckets = {}
        per_module_fit = {}
        for mod_name, prof in self.profile_results.items():
            for bucket, (secs, n) in prof.items():
                g = global_buckets.setdefault(bucket, [0.0, 0])
                g[0] += secs
                g[1] += n
                if bucket == 'fit':
                    per_module_fit[mod_name] = [secs, n]

        def _print_table(title, rows):
            if not rows:
                return
            rows = sorted(rows.items(), key=lambda kv: kv[1][0], reverse=True)
            total = sum(secs for _, (secs, _) in rows)
            width = max(len(name) for name, _ in rows)
            print()
            print("=" * (width + 34))
            print(f"{title} (total tracked: {total:.2f}s)")
            print("=" * (width + 34))
            print(f"{'bucket'.ljust(width)}   {'seconds':>10}  {'calls':>7}  {'s/call':>9}")
            print("-" * (width + 34))
            for name, (secs, n) in rows:
                per = secs / n if n else 0.0
                print(f"{name.ljust(width)}   {secs:>10.3f}  {n:>7d}  {per:>9.4f}")
            print("=" * (width + 34))
            print()

        _print_table("EXPERIMENT PROFILE: fit vs measurements (all modules)", global_buckets)
        _print_table("EXPERIMENT PROFILE: total fit time per module", per_module_fit)


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
            
        os.makedirs(path, exist_ok=True)
        fname = os.path.join(path, 'exp' + str(identifier) + '.json')
        with open(fname, 'w') as f:
            json.dump(self.result_dict, f, indent=4, cls=NumpyEncoder)


####################################################################################################