# Some To Do's and Reminders

1. Rename items in the code to match the paper:
    * `ScaledGreedy` is referred to as lazy greedy in the code. 
    * `PEC` may sometimes be referred to as `DSCluster` in the experiment results 

2. `experiments/aniso/max_rules_ids_alt.py` is missing the objective re-scoring section that
   `max_rules.py` has (`SCORING_OBJECTIVE_NAMES`, `_score_decisions_all_objectives`, and the
   `Experiment.run()`-decisions re-scoring pass) -- its results never get an `'objective'` field,
   unlike every other max_rules* variant. This is okay for now, it just means we have to 
   compute the objective post-hoc when visualizing results, but it would be nice to have. 


