####################################################################################################
# Path setup

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in _HERE.parents if (p / "data").is_dir()), None)
if PROJECT_ROOT is None:
    raise ModuleNotFoundError("Could not locate repository root.")
sys.path.insert(0, str(PROJECT_ROOT))

####################################################################################################
# Selects one alpha per objective from the alphas.py sweep, by the elbow method.
#
# Mirrors the `select_alphas` function in examples/experiments.ipynb exactly, so this can run
# standalone as part of the scripted pipeline instead of requiring a manual notebook pass.
# Reads alphas/exp{OUTFILE_REF}.json, writes alphas/selected_alphas{OUTFILE_REF}.json.

import json
import numpy as np
from intercluster.utils import compute_elbow
from experiments.fashion.config import ALPHAS_DIR, OUTFILE_REF

# Which measurement plays the role of `reward` and `cost` for each objective type.
objective_cost_reward_dict = {
    'coverage-mistake': {'reward': 'cluster-coverage', 'cost': 'mistakes'},
    'coverage-cost': {'reward': 'cluster-coverage', 'cost': 'rule-clustering-cost'},
    'coverage-pairwise-distance': {'reward': 'cluster-coverage', 'cost': 'rule-pairwise-distance'},
    'coverage-mistake-weighted': {'reward': 'cluster-coverage', 'cost': 'mistakes'},
    'coverage-cost-weighted': {'reward': 'cluster-coverage', 'cost': 'rule-clustering-cost'},
    'coverage-pairwise-distance-weighted': {'reward': 'cluster-coverage', 'cost': 'rule-pairwise-distance'},
}


def select_alphas(alpha_experiment_dict, outfile=None):
    selected_alpha_dict = {}
    for module in alpha_experiment_dict['modules'].keys():
        try:
            objective_type = module.split(';')[1].strip()
            cost = objective_cost_reward_dict[objective_type]['cost']
            reward = objective_cost_reward_dict[objective_type]['reward']

            z = np.array(
                list(alpha_experiment_dict['modules'][module][cost].keys()), dtype=float
            )
            x = np.array(
                [alpha_experiment_dict['modules'][module]['weighted-avg-length'][str(l)] for l in z]
            )
            rl = np.array(
                [alpha_experiment_dict['modules'][module]['sum-rule-length'][str(l)] for l in z]
            )
            y1 = np.array([alpha_experiment_dict['modules'][module][reward][str(l)] for l in z])
            y2 = np.array(
                [alpha_experiment_dict['modules'][module][cost][str(l)] for l in z]
            ) + z * rl
            lambda_vals = np.array(
                [alpha_experiment_dict['modules'][module]['lambda'][str(l)] for l in z]
            )
            y = y1 - lambda_vals * y2

            best_alpha_idx = compute_elbow(x, y, increasing=True)

            # Prefer a larger alpha when it is within 0.1% of the elbow's objective value.
            for idx in range(best_alpha_idx + 1, len(z)):
                if y[idx] >= y[best_alpha_idx] * 0.999:
                    best_alpha_idx = idx

            selected_alpha_dict[module] = z[best_alpha_idx]

        except Exception as e:
            print(f"Could not select alpha for module {module}: {e}")
            selected_alpha_dict[module] = 0.0

    if outfile is not None:
        with open(outfile, 'w') as f:
            json.dump(selected_alpha_dict, f, indent=4)

    return selected_alpha_dict


####################################################################################################

infile = ALPHAS_DIR + 'exp' + OUTFILE_REF + '.json'
outfile = ALPHAS_DIR + 'selected_alphas' + OUTFILE_REF + '.json'

with open(infile) as f:
    alpha_experiment_dict = json.load(f)

selected = select_alphas(alpha_experiment_dict, outfile)

for module, alpha in selected.items():
    print(f"  {module}: {alpha}")
print(f"\nSaved to {outfile}")

####################################################################################################
