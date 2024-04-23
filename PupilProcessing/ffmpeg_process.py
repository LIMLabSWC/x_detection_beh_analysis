from pathlib import Path, PureWindowsPath, PurePosixPath 
import glob
import subprocess
import numpy as np
import os
import argparse


def posix_from_win(path:str) -> Path:
    """
    Convert a Windows path to a Posix path.

    Args:
        path (str): The input Windows path.

    Returns:
        Path: The converted Posix path.
    """
    if ':\\' in path:
        path_bits = PureWindowsPath(path).parts
        path_bits = [bit for bit in path_bits if '\\' not in bit]
        return Path(PurePosixPath(*path_bits))
    else:
        return Path(path)


parser = argparse.ArgumentParser()
parser.add_argument('viddir',type=str)
args = parser.parse_args()

viddir = args.viddir
ceph_dir = Path(r'/ceph/akrami')
print(f'{ceph_dir/posix_from_win(viddir) =}')
for path in (ceph_dir/posix_from_win(viddir)).rglob('*.avi'):
    # path = path.absolute()
    if 'eye' not in str(path):
        continue
    ffmpeg_command = f'ffmpeg -i {path} -c:v copy -c:a copy -y {str(path).replace(".avi",".mp4")}'
    if not Path(str(path).replace(".avi",".mp4")).exists():
        subprocess.run(ffmpeg_command,shell=True)