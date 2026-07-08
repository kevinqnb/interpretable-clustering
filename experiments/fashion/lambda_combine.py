import json

lambda_dir = "data/experiments/fashion/lambda/"
main_ref = "_resub_dscluster"
combine_refs = ["_resub_exp", "_resub_exkmc"]
out_ref = "_resub"

# Load main experiment dict
fname = lambda_dir + "exp" + main_ref + ".json"
with open(fname, 'r') as f:
    main_experiment_dict = json.load(f)

# Load combine experiment dicts and merge
for ref in combine_refs:
    fname = lambda_dir + "exp" + ref + ".json"
    with open(fname, 'r') as f:
        combine_experiment_dict = json.load(f)

    # Merge combine_experiment_dict into main_experiment_dict
    for key in combine_experiment_dict['modules']:
        if key in main_experiment_dict['modules']:
            pass
        else:
            main_experiment_dict['modules'][key] = combine_experiment_dict['modules'][key]


# Save combined experiment dict
output_fname = lambda_dir + "exp" + out_ref + ".json"
with open(output_fname, 'w') as f:
    json.dump(main_experiment_dict, f, indent=4)
