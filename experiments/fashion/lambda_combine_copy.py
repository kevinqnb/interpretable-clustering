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

OUTFILE_REF = '_conf_50'

lambda_dir = "data/experiments/fashion/lambda/"

# Every part this combiner knows how to merge, in priority order (first part to contribute a
# given module/fixed-parameter key wins). Each is optional and independently produced:
#   - lambda_{cba,cn2,dtree,ids,pec}.py: the per-model split (see those scripts).
#   - lambda_exkmc.py: ExKMC, split out from the start (mnist/fashion take much longer to fit
#     some algorithms -- see experiments/README.md's step-3 note).
#   - lambda.py itself: the original monolithic script (CBA+CN2+PEC+Decision-Tree+IDS in one
#     job), kept as a fallback source in case you ran that instead of the per-model split -- it's
#     listed last so a fresher per-model result always takes priority over a stale monolithic run.
# This lets you launch each part as its own parallel job and combine whatever has actually
# finished -- missing parts are skipped with a warning instead of raising, so you can re-run this
# script as more parts land rather than waiting for every one of them.
#
# CAUTION: the monolithic fallback means "missing" per-model parts don't necessarily block the
# combine the way you might expect -- if an old lambda.py run's exp_dscluster{OUTFILE_REF}.json
# is sitting on disk, its (possibly stale) modules silently fill in for any per-model script you
# haven't (re)run yet. This script prints exactly which modules came from that fallback, every
# run, so a stale backfill is never silent -- check that list before trusting the combined output.
MONOLITHIC_PART = 'monolithic (lambda.py)'
part_refs = {
    'cba': '_cba' + OUTFILE_REF,
    'cn2': '_cn2' + OUTFILE_REF,
    'dtree': '_dtree' + OUTFILE_REF,
    'ids': '_ids' + OUTFILE_REF,
    'pec': '_pec' + OUTFILE_REF,
    'exkmc': '_exkmc' + OUTFILE_REF,
    MONOLITHIC_PART: '_dscluster' + OUTFILE_REF,
}
out_ref = OUTFILE_REF

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

    # Merge modules: union by module name, first part to contribute a given name wins (matches
    # the original exkmc-merge's semantics).
    for key, value in part_dict.get('modules', {}).items():
        if key not in combined_dict['modules']:
            combined_dict['modules'][key] = value
            module_source[key] = part_name

    # Merge fixed-parameters the same way, so a key only one part computes (e.g. PEC's
    # 'lambda_star'/'lambda_grid') still ends up in the combined output as long as that part is
    # present.
    for key, value in part_dict.get('fixed-parameters', {}).items():
        combined_dict['fixed-parameters'].setdefault(key, value)

if not loaded_parts:
    raise FileNotFoundError(
        f"None of the lambda part files were found in {lambda_dir} for ref '{out_ref}' "
        f"(looked for refs: {list(part_refs.values())}). Run at least one of "
        f"lambda_{{cba,cn2,dtree,ids,pec,exkmc}}.py or lambda.py first."
    )

print(f"Combined parts: {loaded_parts}")
if missing_parts:
    print(f"Skipped (not yet completed): {missing_parts}")

stale_modules = sorted(k for k, v in module_source.items() if v == MONOLITHIC_PART)
if stale_modules:
    print(
        f"WARNING: {len(stale_modules)} module(s) came from the '{MONOLITHIC_PART}' fallback, "
        f"not a per-model script -- these may be stale if you meant to (re)run them "
        f"individually: {stale_modules}"
    )

# Save combined experiment dict
output_fname = os.path.join(lambda_dir, "exp" + out_ref + ".json")
with open(output_fname, 'w') as f:
    json.dump(combined_dict, f, indent=4)

print(f"Saved combined results to {output_fname}")
