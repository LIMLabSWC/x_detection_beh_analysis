#!/bin/bash

#SBATCH -N 1
#SBATCH --mem 16G
#SBATCH -t 0-4:00:00 # time (D-HH:MM)
#SBATCH -p gpu
#SBATCH --gpus-per-node=1


# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --video_file) video_file="$2"; shift ;;
        --pupilsense_config_file) config_file="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check required args
if [[ -z "$video_file" || -z "$config_file" ]]; then
    echo "Usage: sbatch run_pupil_extraction.sh --video_file <video_path> --pupilsense_config_file <config_path>"
    exit 1
fi

# Load modules
echo "Loading conda env..."
source /etc/profile.d/modules.sh
module load miniconda
module load cuda/12

echo "GPU status (nvidia-smi):"
nvidia-smi

source activate process_pupil

# Display info
echo "Running pupil extraction..."
echo "Video file: $video_file"
echo "Config file: $config_file"

# Run pupil extraction script
python test.py
/nfs/nhome/live/aonih/.conda/envs/process_pupil/bin/python PupilProcessing/extract_pupil_pupil_sense_skv.py "$video_file" --pupilsense_config_file "$config_file"