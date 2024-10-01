import os

import pandas as pd

from psychophysicsUtils import *
import analysis_utils as utils
from datetime import datetime, time, timedelta
from matplotlib import pyplot as plt
matplotlib.use('Agg')
import numpy as np
from scipy.stats import zscore
import scipy.signal
import time
from PupilProcessing.pupilpipeline import Main as Main
from pathlib import Path, PureWindowsPath, PurePosixPath
import yaml
from loguru import logger
from rich.logging import RichHandler
from pyinspect import install_traceback
import argparse
import platform


def posix_from_win(path: str, ceph_linux_dir='/ceph/akrami') -> Path:
    """
    Convert a Windows path to a Posix path.

    Args:
        path (str): The input Windows path.
        :param ceph_linux_dir:

    Returns:
        Path: The converted Posix path.
    """
    if ':\\' in path:
        path_bits = PureWindowsPath(path).parts
        path_bits = [bit for bit in path_bits if '\\' not in bit]
        return Path(PurePosixPath(*path_bits))
    else:
        assert ceph_linux_dir
        return Path(path).relative_to(ceph_linux_dir)


if __name__ == "__main__":
    install_traceback()

    logger.configure(
        handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
    )
    logger.info('started and loading config')
    today_str = datetime.today().date().strftime('%y%m%d')
    logger_path = Path.cwd()/f'log'/f'log_{today_str}.txt'
    logger_path = utils.unique_file_path(logger_path)
    logger.add(logger_path,level='INFO')

    install_traceback()
    logger.configure(
        handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=Path('config','mouse_fam_old_conf_unix.yaml'))
    parser.add_argument('-date', default=None)
    parser.add_argument('--sess_top_query', default=None)
    args = parser.parse_args()
    os = platform.system().lower()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    logger.info('loaded config')

    tdatadir = Path(config[f'tdatadir_{os}'])
    pdatadir = Path(config[f'pdatadir_{os}'])
    assert tdatadir.is_dir() and pdatadir.is_dir()

    # pdatadir = r'W:\mouse_pupillometry\mousenormdev\aligned_mousenormdev_stage5'
    # pdatadir = r'W:\mouse_pupillometry\mousenormdev_swap\aligned_mousenormdev_swap'

    # pdatadir = r'W:\mouse_pupillometry\mousenormdev_hf\aligned_mousenormdev_hf'
    # pdatadir = r'W:\mouse_pupillometry\mousefam\aligned_mousefam'
    # pdatadir = r'W:\mouse_pupillometry\mousefam_post\aligned_mousefam_post'
    # pdatadir = r'W:\mouse_pupillometry\mouseprobreward\aligned_mouseprobreward'
    # pdatadir = r'W:\mouse_pupillometry\mouseprobreward_hf'
    # pdatadir =r'W:\mouse_pupillometry\mouse_hf'
    # pdatadir = r'W:\mouse_pupillometry\mouse_hf_normdev'

    pkl_prefix = pdatadir.parts[-1]
    # dirstyle = 'N_D_it'
    dirstyle = config['dirstyle']

    if dirstyle == 'N_D_it':
        list_aligned = list(pdatadir.iterdir())
        # list_aligned = list(pdatadir.rglob('*.h5'))
        aligneddir = ''
    else:
        list_aligned = list(pdatadir.iterdir())
        aligneddir = ''

    # if 'harpbins' in list_aligned:
    #     list_aligned.remove('harpbins')
    splitdir = np.unique(np.array([f.name.split('_')[:-1]
                                  for i,f in enumerate(list_aligned) if 'harpbins' not in f.name]),
                         axis=0)
    # splitdir = np.vstack([[np.array(path.parts) for path in list_aligned]])
    dir_animals, dir_animaldates = splitdir[:, 0], splitdir[:, 1]

    if args.date:
        dates2process=[args.date]
    else:
        dates2process = config['dates2process']
    animals2process = config['animals2process']
    spec_dates,spec_animals = len(dates2process) > 0, len(animals2process) > 0
    for i,(e,d) in enumerate(zip(dir_animals, dir_animaldates)):
        if e in dir_animals:  #['DO48']:
            if spec_animals:
                if e not in animals2process:
                    continue
            else:
                if spec_dates:
                    if d not in dates2process:
                        continue
                dates2process.append(dir_animaldates[i])
            if spec_dates:
                if d in dates2process:
                    animals2process.append(e)

    # get animals and dates from session topology
    ceph_dir = Path(config[f'ceph_dir_{os}'])
    if config.get('session_topology_path'):
        use_session_topology = True
    else:
        use_session_topology = False
    
    if use_session_topology:
        sess_top_path = ceph_dir/posix_from_win(config['session_topology_path'])
        session_topology = pd.read_csv(sess_top_path)
        session_topology['videos_dir'] = session_topology['videos_dir'].apply(lambda x: ceph_dir/posix_from_win(x))
        animals2process = session_topology['name'].unique().tolist()
        dates2process = session_topology['date'].unique().astype(str).tolist()

    else:
        session_topology = None
    # animals2process=['DO80']
    # dates2process = ['240419']
    do_zscore = config['do_zscore']
    bandpass_met = config['bandpass_met']
    han_size = config['han_size']
    if han_size:
        lowtype = 'hanning'
    else:
        lowtype = 'filter'
    fs = config['fs']
    pdata_topic = config['pdata_topic']
    han_size_str = f'hanning{str(han_size).replace(".","")}'*bool(han_size)

    pklname = f'{pkl_prefix}_{config["pklname_suffix"]}_{config["protocol"]}_{pdata_topic.split("_")[1]}_{int(fs)}Hz_hpass{str(bandpass_met[0]).replace(".", "")}_lpass{str(bandpass_met[1]).replace(".", "")}' \
              f'{han_size_str}_TOM{"_rawsize" * (not do_zscore)}.pkl'
    # pklname = r'mouse_fm_fam_2d_90Hz_hpass00_hanning025_detrend.pkl'
    preprocess_pkl =f'{pkl_prefix}_{config["pklname_suffix"]}_{config["protocol"]}_w_LR_noTTL.pkl'
    to_redo = config.get('sess_to_redo',[])
    session_topology['date_str'] = session_topology['date'].astype(str)
    if to_redo:
        _to_redo = [[e] if len(e.split('_')) == 2  else
                    [f'{ee}_{e}' for ee in session_topology.query(f'date=="{e}"')['name']] if e.isnumeric() else
                    [f'{e}_{ee}' for ee in session_topology.query(f'name=="{e}"')['date']] for e in to_redo]
        to_redo = sum(_to_redo, [])
        # [e if len(e.split('_')==2) e if e.isnumeric() else [f'{e}_{ee}' for ee in dates2process] for e in to_redo]

    if args.sess_top_query:
        session_topology = session_topology.query(args.sess_top_query)
        animals2process = session_topology['name'].unique().tolist()
        dates2process = session_topology['date'].unique().astype(str).tolist()

    run = Main(animals2process, dates2process,(Path(config[f'pkl_dir_{os}'])/ pklname), tdatadir,
               pdatadir, pdata_topic, fs, han_size=han_size, passband=bandpass_met, aligneddir=aligneddir,
               subjecttype='mouse', dlc_snapshot=[2450000, 1300000], overwrite= config.get('ow_flag',False), do_zscore=do_zscore,
               lowtype=lowtype, dirstyle=dirstyle, dlc_filtflag=True, redo=to_redo,
               preprocess_pklname=Path(config[f'pkl_dir_{os}'])/preprocess_pkl,use_ttl=config['use_TTL'],
               protocol=config['protocol'],use_canny_ell=config['use_canny_ell'],
               session_topology=session_topology)
    logger.info('Main class initialised')
    run.load_pdata()