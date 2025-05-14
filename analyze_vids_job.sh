#! /bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH --exclude=gpu-sr670-20
#SBATCH -N 1   # number of nodes
#SBATCH --mem 40G # memory pool for all cores
#SBATCH --gpus-per-node=1
#SBATCH -n 1
#SBATCH -t 2:0:0 # time
#SBATCH --mail-type=FAIL

echo "loading conda env"
module load miniconda
module load cuda/11.6

source activate DEEPLABCUT
export DLClight=True

echo "running script"
/nfs/nhome/live/aonih/.conda/envs/DEEPLABCUT/bin/python ~/dlc/analyze_vids.py "$1" --findvids "${2:-1}" --fileflag "${3:-eye0.mp4}" --dirstyle "${4:-Y_m_d\it}"