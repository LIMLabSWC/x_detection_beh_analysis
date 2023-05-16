#! /bin/bash

#SBATCH -p cpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 40G # memory pool for all cores
#SBATCH -t 5-0:0 # time
#SBATCH --mail-user=a.onih@ucl.ac.uk
#SBATCH --mail-type=END,FAIL

echo "loading conda env"
module load miniconda
module load cuda/11.6

source activate process_pupil

echo "running script"
python mousepipeline.py "$1" "${2:0}"
