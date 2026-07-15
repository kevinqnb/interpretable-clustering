import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in _HERE.parents if (p / "data").is_dir()), None)
if PROJECT_ROOT is None:
    raise ModuleNotFoundError("Could not locate repository root.")
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cli_utils import conf_tag, parse_experiment_args


args = parse_experiment_args(confidence_default=0.75)
tag = conf_tag(args.confidence)

max_rules_dir = "data/experiments/mnist/max_rules/"
main_ref = f"_resub_dscluster_conf_{tag}"
combine_refs = [f"_resub_exkmc_conf_{tag}", f"_resub_exp_conf_{tag}"]
out_ref = f"_resub_conf_{tag}"

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