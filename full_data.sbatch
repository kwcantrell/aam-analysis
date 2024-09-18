#!/bin/bash -l
#SBATCH --time=48:00:00
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 25G

module load tensorflow_2.14_3.9
source ../aam/bin/activate
cd ../attention-all-microbes
python cli.py fit-regressor --i-table-path /home/bsathappan/aam-testing/raw.nobloom.minfeat.mindepth.biom --m-metadata-file /home/bsathappan/aam-testing/Cleaned_AmericanGut.txt --m-metadata-column 'host_age_normalized_years' --p-missing-samples 'ignore' --p-output-dir /home/bsathappan/aam-testing/
