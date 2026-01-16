#!/bin/bash -l

#$ -P mcnet

#$ -l h_rt=12:00:00

#$ -pe omp 28

#$ -l cpu_arch=skylake|cascadelake|icelake

#$ -o out/max_rules_out.txt

#$ -e out/max_rules_error.txt

#$ -m e

#$ -hold_jid 2464018

module load python3/3.12.4

cd /project/mcnet/kevin/intercluster/repo

uv run python experiments/climate/max_rules.py
