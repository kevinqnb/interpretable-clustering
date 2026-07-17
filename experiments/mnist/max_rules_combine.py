import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in _HERE.parents if (p / "data").is_dir()), None)
if PROJECT_ROOT is None:
    raise ModuleNotFoundError("Could not locate repository root.")
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.mnist.config import OUTFILE_REF

max_rules_dir = "data/experiments/mnist/max_rules/"
main_ref = '_dscluster' + OUTFILE_REF
# max_rules_exp.py (Exp-Tree) used to contribute a third ref here, but Exp-Tree
# never appears in examples/experiments.ipynb's `comparison_modules` -- dropped
# along with max_rules_exp.py itself.
combine_refs = ['_exkmc' + OUTFILE_REF]
out_ref = OUTFILE_REF

# Load main experiment dict
fname = max_rules_dir + "exp" + main_ref + ".json"
with open(fname, 'r') as f:
    main_experiment_dict = json.load(f)

# Load combine experiment dicts and merge
for ref in combine_refs:
    fname = max_rules_dir + "exp" + ref + ".json"
    with open(fname, 'r') as f:
        combine_experiment_dict = json.load(f)

    # Merge combine_experiment_dict into main_experiment_dict
    for key in combine_experiment_dict['modules']:
        if key in main_experiment_dict['modules']:
            pass
        else:
            main_experiment_dict['modules'][key] = combine_experiment_dict['modules'][key]


# Save combined experiment dict
output_fname = max_rules_dir + "exp" + out_ref + ".json"
with open(output_fname, 'w') as f:
    json.dump(main_experiment_dict, f, indent=4)
