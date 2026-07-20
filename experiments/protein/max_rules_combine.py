import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in _HERE.parents if (p / "data").is_dir()), None)
if PROJECT_ROOT is None:
    raise ModuleNotFoundError("Could not locate repository root.")
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.protein.config import OUTFILE_REF

max_rules_dir = "data/experiments/protein/max_rules/"

# Unlike mnist/fashion, this dataset has no per-model split -- max_rules.py's own output
# (exp{OUTFILE_REF}.json) already IS the saved solution for every model, including PEC's
# lazy-greedy modules if you're running a max_rules.py that already fits them directly (see its
# dscluster_module_list loop). This script exists only to retrofit lazy-greedy results into an
# *already-completed* max_rules.py run from before that loop existed, without redoing the whole
# (expensive) run -- max_rules_pec_lazy.py fits just the lazy-greedy modules on their own, and
# this merges that output into the existing saved solution.
#
# Because max_rules_pec_lazy.py's modules are named '...; lazy-greedy' (see that script), they
# never collide with the distorted-greedy-named modules already in exp{OUTFILE_REF}.json, so this
# is a pure addition, not an overwrite of anything that's already there.
MONOLITHIC_PART = 'monolithic (max_rules.py)'
part_refs = {
    MONOLITHIC_PART: OUTFILE_REF,
    'pec_lazy': '_pec_lazy' + OUTFILE_REF,
}
out_ref = OUTFILE_REF

combined_dict = {'fixed-parameters': {}, 'baseline': None, 'modules': {}}
module_source = {}  # module name -> part name that supplied it
loaded_parts = []
missing_parts = []

for part_name, ref in part_refs.items():
    fname = os.path.join(max_rules_dir, "exp" + ref + ".json")
    if not os.path.exists(fname):
        missing_parts.append(part_name)
        continue

    with open(fname, 'r') as f:
        part_dict = json.load(f)
    loaded_parts.append(part_name)

    if combined_dict['baseline'] is None:
        combined_dict['baseline'] = part_dict.get('baseline')

    # Merge modules: union by module name, first part to contribute a given name wins. In
    # practice the two parts never share a module name (pec_lazy's are all '...; lazy-greedy'),
    # so this is effectively a pure union, not an overwrite.
    for key, value in part_dict.get('modules', {}).items():
        if key not in combined_dict['modules']:
            combined_dict['modules'][key] = value
            module_source[key] = part_name

    for key, value in part_dict.get('fixed-parameters', {}).items():
        combined_dict['fixed-parameters'].setdefault(key, value)

if MONOLITHIC_PART in missing_parts:
    raise FileNotFoundError(
        f"{os.path.join(max_rules_dir, 'exp' + OUTFILE_REF + '.json')} (the existing "
        f"max_rules.py output) was not found. This script only ADDS max_rules_pec_lazy.py's "
        f"lazy-greedy modules on top of that saved solution -- it doesn't recompute anything "
        f"else. Run max_rules.py first if that file doesn't exist yet."
    )

if 'pec_lazy' in missing_parts:
    print(
        f"Nothing to merge: {os.path.join(max_rules_dir, 'exp_pec_lazy' + OUTFILE_REF + '.json')} "
        f"(max_rules_pec_lazy.py's output) was not found, so there are no new lazy-greedy "
        f"modules to add. Run max_rules_pec_lazy.py first. Leaving the existing saved solution "
        f"untouched."
    )
else:
    output_fname = os.path.join(max_rules_dir, "exp" + out_ref + ".json")
    with open(output_fname, 'w') as f:
        json.dump(combined_dict, f, indent=4)
    added = sorted(k for k, v in module_source.items() if v == 'pec_lazy')
    print(f"Merged {len(added)} lazy-greedy module(s) into {output_fname}: {added}")

print(f"Loaded parts: {loaded_parts}" + (f"; skipped (not found): {missing_parts}" if missing_parts else ""))
