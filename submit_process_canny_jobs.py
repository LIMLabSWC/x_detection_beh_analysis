import yaml
from pathlib import Path
import subprocess
import argparse
import numpy as np
import platform


parser = argparse.ArgumentParser()
parser.add_argument('config_file',default=Path('config','mouse_fam_Sept23_cohort_conf_unix.yaml'))
parser.add_argument('--os',default='hpc')
parser.add_argument('--name',default='')
parser.add_argument('--date',default='')
parser.add_argument('--ow',default=0)
args = parser.parse_args()

with open(args.config_file, 'r') as file:
    config = yaml.safe_load(file)
os = platform.system().lower()
dates2process = config['dates2process']
animals2process = config['animals2process']
pdatadir = Path(config[f'pdatadir_{os}'])
list_aligned = list(pdatadir.iterdir())

if args.os == 'hpc':
    script_base = rf'sbatch process_canny_ell.sh <pd> <n> --ow {args.ow}'
elif args.os == 'local':
    script_base = rf'python H:\gd_analysis\run_process_canny.py <pd> <n> --ow {args.ow}'
else:
    print('invalid os args')
    exit()
for a in animals2process:
    if args.name:
        if a not in args.name:
            continue
    for d in dates2process:
        if args.date:
            if d not in args.date:
                continue
        name = '_'.join([a,d])
        sess_pdata_paths = pdatadir.glob(f'{name}*')
        for path in sess_pdata_paths:
            path=path.absolute()
            if (path/f'{a}_{d}_canny_ellipses.csv').is_file() and not args.ow:
                print(f'{path} exists. Skipping')
                continue
            # print(path)
            process = subprocess.run(script_base.replace('<pd>', str(path)).replace('<n>', name).split(' '))

