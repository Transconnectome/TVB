#!/bin/sh

#SBATCH --job-name=scalar_array
#SBATCH --nodes=1
#SBATCH --nodelist=node1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=10gb
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/connectome/TVB/TVB_RESEARCH/run_scripts/run_scalar/SC_ses_2_run_scalar/output_%A-%a.out
#SBATCH --error=/scratch/connectome/TVB/TVB_RESEARCH/run_scripts/run_scalar/SC_ses_2_run_scalar/error_%A-%a.error
#SBATCH --array=1-30%30


start=0.10
gap=0.01
#below : $(()) doens't work for linux in default (float point arithemetic), so used the thing below
threshold=$(expr ${start}+${SLURM_ARRAY_TASK_ID}*${gap} | bc)

python SC_ses_2_run_scalar.py --threshold $threshold
