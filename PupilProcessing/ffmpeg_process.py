from pathlib import Path
import glob
import subprocess
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('viddir',type=str)
args = parser.parse_args()

viddir = args.viddir
for path in Path(viddir).rglob('*.avi'):
    # path = path.absolute()
    ffmpeg_command = f'ffmpeg -i {path} -c:v copy -c:a copy -y {str(path).replace(".avi",".mp4")}'
    if not Path(str(path).replace(".avi",".mp4")).exists():
        subprocess.run(ffmpeg_command)