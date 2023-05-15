import glob
import argparse
import os
import subprocess
from loguru import logger
from rich.logging import RichHandler
from pyinspect import install_traceback


install_traceback()

logger.configure(
    handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
)
parser = argparse.ArgumentParser()
parser.add_argument('cf')
script_base = r'sbatch PupilProcessing/process_sess.sh <cf> <d>'

dates2process = [
                    # '221111','221114','221115','221116','221117','221118',
                    '221214','221215','230116','230117', '230119',
                    '230214','230216','230217','230221',
                    '230222','230223','230224','230228','230301','230303'
                    ]  # fam dates

args = parser.parse_args()

for d in dates2process:
    # logger.info(script_base.replace('<cf>',args.cf).replace('<d>',d).split(' '))
    process = subprocess.Popen(script_base.replace('<cf>',args.cf).replace('<d>',d).split(' '))

