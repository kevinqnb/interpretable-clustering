#!/bin/bash -l

#$ -P mcnet

#$ -l h_rt=72:00:00

#$ -pe omp 16

#$ -o out/mine_rules_out.txt

#$ -e out/mine_rules_error.txt

#$ -m e

module load python3/3.12.4

cd /project/mcnet/kevin/intercluster/repo

uv run python experiments/fashion/mine_rules.py
