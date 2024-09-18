#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem 128g
#SBATCH --time 24:00:00
#SBATCH --array 0-9
#SBATCH --output output_log_%A_%a.txt

conda activate qiime2-2023.5
export TMPDIR=/panfs/$USER
python GutAgePrediction_Testing.py
