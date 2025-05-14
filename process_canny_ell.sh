#!/bin/bash
#
#SBATCH -p cpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH -n 8   # number of cores
#SBATCH --mem 60G # memory pool for all cores
#SBATCH -t 1-20:00 # time (D-HH:MM)
#

module load miniconda
source activate process_pupil

echo "running script"
python ~/gd_analysis/run_process_canny.py "$1" "$2"  --nf "${3:-0}" --ow "${4:-0}"