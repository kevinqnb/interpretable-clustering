#!/bin/bash -l

#$ -P mcnet

#$ -l h_rt=24:00:00

#$ -pe omp 16

#$ -o out/alphas_out.txt

#$ -e out/alphas_error.txt

#$ -m e

module load python3/3.12.4

cd /project/mcnet/kevin/intercluster/repo

uv run python experiments/digits/alphas.py
