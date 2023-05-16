import os

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
from pathlib import Path
import yaml
from loguru import logger
from rich.logging import RichHandler
from pyinspect import install_traceback
import argparse


if __name__ == "__main__":
    install_traceback()

    logger.configure(
        handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
    )

    logger_path = Path.cwd()/'log'/'log.txt'
    logger_path = utils.unique_file_path(logger_path)
    logger.add(logger_path,level='INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('date', default=None)
    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    tdatadir = config['tdatadir']
    pdatadir = config['pdatadir']

    # pdatadir = r'W:\mouse_pupillometry\mousenormdev\aligned_mousenormdev_stage5'
    # pdatadir = r'W:\mouse_pupillometry\mousenormdev_swap\aligned_mousenormdev_swap'

    # pdatadir = r'W:\mouse_pupillometry\mousenormdev_hf\aligned_mousenormdev_hf'
    # pdatadir = r'W:\mouse_pupillometry\mousefam\aligned_mousefam'
    # pdatadir = r'W:\mouse_pupillometry\mousefam_post\aligned_mousefam_post'
    # pdatadir = r'W:\mouse_pupillometry\mouseprobreward\aligned_mouseprobreward'
    # pdatadir = r'W:\mouse_pupillometry\mouseprobreward_hf'
    # pdatadir =r'W:\mouse_pupillometry\mouse_hf'
    # pdatadir = r'W:\mouse_pupillometry\mouse_hf_normdev'

    pkl_prefix = pdatadir.split('\\')[2]
    # dirstyle = 'N_D_it'
    dirstyle = config['dirstyle']

    if dirstyle == 'N_D_it':
        list_aligned = os.listdir(pdatadir)
    else:
        list_aligned = os.listdir(pdatadir)
    if 'harpbins' in list_aligned:
        list_aligned.remove('harpbins')
    aligneddir = os.path.split(pdatadir)[-1]
    splitdir = np.vstack([[np.array(path.split('_')) for path in list_aligned]])
    dir_animals, dir_animaldates = splitdir[:, 0], splitdir[:, 1]
    # animals2process = ['DO54','DO55','DO56','DO57']
    # animals2process = ['DO58','DO59','DO60','DO62']
    # animals2process = ['DO54','DO55','DO56','DO57','DO58','DO59','DO60','DO62']
    # animals2process= ['DO57']
    # dates2process = [
    #                 # '221111','221114','221115','221116','221117','221118',
    #                 '221214','221215','230116','230117', '230119',
    #                 '230214','230216','230217','230221',
    #                 '230222','230223','230224','230228','230301','230303'
    #                 ]  # fam dates
    # dates2process = ['230216','230217','230221',
    #                 '230222','230223','230224','230228','230301', '230302','230303']  # new animal fam dates
    # dates2process = ['230116','230117','230119']*4  # normdev dates
    # dates2process = ['230126','230127','230206','230207']
    # dates2process = ['230201','230202','230203','230206','230207','230208','230211'] # probreward dates
    # dates2process = ['230306','230307','230308','230310']  # new normdev 0.1 rate
    # dates2process = ['230317']
    # # old
    # aligneddir = os.path.split(pdatadir)[-1]
    # splitdir = np.array([path.split('_') for path in os.listdir(pdatadir)])
    # animals, animaldates = splitdir[:,0], splitdir[:,1]
    # spec_animal = []
    # spec_animal_dates = []

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
                animals2process.append(e)
                dates2process.append(dir_animaldates[i])
            if spec_dates:
                if d in dates2process:
                    animals2process.append(e)
    # spec_animal = []
    spec_animal_dates = []

    # for i,e in enumerate(animals):
    #     if e in animals:  #['DO48']:
    #         spec_animal.append(e)
    #         spec_animal_dates.append(animaldates[i])

    # han_size = 1
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

    pklname = f'{pkl_prefix}_fam_{pdata_topic.split("_")[1]}_{int(fs)}Hz_lpass{str(bandpass_met[1]).replace(".", "")}' \
              f'{han_size_str}_TOM{"_rawsize" * (not do_zscore)}noTTL_.pkl'
    # pklname = r'mouse_fm_fam_2d_90Hz_hpass00_hanning025_detrend.pkl'
    # preprocess_pkl =f'{pkl_prefix}_fam_w_LR_noTTL.pkl'

    run = Main(animals2process, dates2process,Path(config['pkl_dir'], pklname), tdatadir,
               pdatadir, pdata_topic, fs, han_size=han_size, passband=bandpass_met, aligneddir=aligneddir,
               subjecttype='mouse', dlc_snapshot=[2450000, 1300000], overwrite=False, do_zscore=do_zscore,
               lowtype=lowtype, dirstyle=dirstyle, dlc_filtflag=True, redo=None,
               preprocess_pklname=Path(config['pkl_dir'], config['preprocess_pkl']),use_ttl=config['use_TTL'],
               protocol='probreward')
    run.load_pdata()