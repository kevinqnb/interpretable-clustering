import json

max_rules_dir = "data/experiments/fashion/max_rules/"
main_ref = "_pairwise_update_dscluster"
combine_refs = ["_pairwise_update_exkmc"]

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

# Save merged experiment dict
out_ref = "_pairwise_update"
output_fname = max_rules_dir + "exp" + out_ref + "_combined.json"
with open(output_fname, 'w') as f:
    json.dump(main_experiment_dict, f, indent=4)