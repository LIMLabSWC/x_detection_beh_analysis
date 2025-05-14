#!/bin/bash

#SBATCH -p cpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH -n 16   # number of cores
#SBATCH --mem 64G # memory pool for all cores
#SBATCH -t 0-2:00:00 # time (D-HH:MM)
#

#!/bin/bash

# Default values
CONFIG_FILE=""
SESS_TOP_QUERY=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config_file) CONFIG_FILE="$2"; shift ;;
        --sess_top_query) SESS_TOP_QUERY="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if required arguments are provided
if [[ -z "$CONFIG_FILE" || -z "$SESS_TOP_QUERY" ]]; then
    echo "Usage: $0 --config_file <path> --sess_top_query <query>"
    exit 1
fi

# Run environment and Python script
module load miniconda
source activate process_pupil

/nfs/nhome/live/aonih/.conda/envs/process_pupil/bin/python mousepipeline.py \
    --config_file "$CONFIG_FILE" \
    --sess_top_query "$SESS_TOP_QUERY"
