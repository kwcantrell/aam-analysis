#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem 128g
#SBATCH --time 24:00:00

conda activate qiime2-2023.5
export TMPDIR=/panfs/$USER
python GutAgePrediction_Visuals.py
