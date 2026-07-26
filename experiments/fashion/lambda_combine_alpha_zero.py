import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in _HERE.parents if (p / "data").is_dir()), None)
if PROJECT_ROOT is None:
    raise ModuleNotFoundError("Could not locate repository root.")
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fashion.config import OUTFILE_REF

lambda_dir = "data/experiments/fashion/lambda/"

# alpha=0 counterpart of lambda_combine.py, restricted to the two alpha-zero parts that actually
# exist (see lambda_alpha_zero.py / lambda_exkmc_alpha_zero.py) -- there is no alpha-zero
# equivalent of the cba/cn2/dtree/ids/pec per-model splits or the monolithic lambda.py fallback,
# since those were never asked for.
#   - lambda_alpha_zero.py: alpha=0 counterpart of the dscluster split (PEC/CBA/Decision-Tree/IDS,
#     everything lambda.py itself fits) -- the primary part, required.
#   - lambda_exkmc_alpha_zero.py: alpha=0 counterpart of the ExKMC split -- optional add-on, since
#     alpha never affects ExKMC's fit (it has no alpha_val parameter; see that script's comment)
#     and merging it in just fills out the comparison table.
DSCLUSTER_PART = 'dscluster_alpha_zero (lambda_alpha_zero.py)'
part_refs = {
    DSCLUSTER_PART: '_dscluster' + OUTFILE_REF + '_alpha_zero',
    'exkmc_alpha_zero': '_exkmc' + OUTFILE_REF + '_alpha_zero',
}
out_ref = OUTFILE_REF + '_alpha_zero'

combined_dict = {'fixed-parameters': {}, 'baseline': None, 'modules': {}}
module_source = {}  # module name -> part name that supplied it
loaded_parts = []
missing_parts = []

for part_name, ref in part_refs.items():
    fname = os.path.join(lambda_dir, "exp" + ref + ".json")
    if not os.path.exists(fname):
        missing_parts.append(part_name)
        continue

    with open(fname, 'r') as f:
        part_dict = json.load(f)
    loaded_parts.append(part_name)

    if combined_dict['baseline'] is None:
        combined_dict['baseline'] = part_dict.get('baseline')

    # Merge modules: union by module name, first part to contribute a given name wins. In
    # practice the two parts never share a module name (exkmc_alpha_zero contributes only
    # 'ExKMC'), so this is effectively a pure union, not an overwrite.
    for key, value in part_dict.get('modules', {}).items():
        if key not in combined_dict['modules']:
            combined_dict['modules'][key] = value
            module_source[key] = part_name

    for key, value in part_dict.get('fixed-parameters', {}).items():
        combined_dict['fixed-parameters'].setdefault(key, value)

if DSCLUSTER_PART in missing_parts:
    raise FileNotFoundError(
        f"{os.path.join(lambda_dir, 'exp_dscluster' + OUTFILE_REF + '_alpha_zero.json')} "
        f"(lambda_alpha_zero.py's output) was not found. This script only ADDS "
        f"lambda_exkmc_alpha_zero.py's ExKMC results on top of that saved solution -- it "
        f"doesn't recompute anything else. Run lambda_alpha_zero.py first if that file "
        f"doesn't exist yet."
    )

if 'exkmc_alpha_zero' in missing_parts:
    print(
        f"Nothing to merge in from ExKMC: "
        f"{os.path.join(lambda_dir, 'exp_exkmc' + OUTFILE_REF + '_alpha_zero.json')} "
        f"(lambda_exkmc_alpha_zero.py's output) was not found. Run "
        f"lambda_exkmc_alpha_zero.py first if you want ExKMC included. Saving the "
        f"dscluster-only solution as-is."
    )

output_fname = os.path.join(lambda_dir, "exp" + out_ref + ".json")
with open(output_fname, 'w') as f:
    json.dump(combined_dict, f, indent=4)

print(f"Loaded parts: {loaded_parts}" + (f"; skipped (not found): {missing_parts}" if missing_parts else ""))
print(f"Saved combined results to {output_fname}")
