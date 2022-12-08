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


if __name__ == "__main__":
    tdatadir = r'C:\bonsai\data\Dammy'

    # pdatadir = r'W:\mouse_pupillometry\mousenormdev\aligned_mousenormdev_stage5'
    # pdatadir = r'W:\mouse_pupillometry\mousenormdev_swap\aligned_mousenormdev_swap'

    # pdatadir = r'W:\mouse_pupillometry\mousenormdev_hf\aligned_mousenormdev_hf'
    # pdatadir = r'W:\mouse_pupillometry\mousefam\aligned_mousefam'
    # pdatadir = r'W:\mouse_pupillometry\mousefam_post\aligned_mousefam_post'
    # pdatadir = r'W:\mouse_pupillometry\mouseprobreward\aligned_mouseprobreward'
    # pdatadir =r'W:\mouse_pupillometry\mouse_hf'
    pdatadir = r'W:\mouse_pupillometry\mouse_hf_normdev'

    pkl_prefix = pdatadir.split('\\')[2]
    dirstyle = 'N_D_it'
    # dirstyle = r'Y_m_d\it'

    if dirstyle == 'N_D_it':
        list_aligned = os.listdir(pdatadir)
    else:
        list_aligned = os.listdir(pdatadir)
    if 'harpbins' in list_aligned:
        list_aligned.remove('harpbins')
    aligneddir = os.path.split(pdatadir)[-1]
    splitdir = np.vstack([[np.array(path.split('_')) for path in list_aligned]])
    animals, animaldates = splitdir[:,0], splitdir[:,1]
    # spec_animal = ['DO54','DO55','DO56','DO57']
    spec_animal= []
    # spec_animal_dates = ['221111','221114','221115','221116','221117']*4
    spec_animal_dates = []

    # # old
    # aligneddir = os.path.split(pdatadir)[-1]
    # splitdir = np.array([path.split('_') for path in os.listdir(pdatadir)])
    # animals, animaldates = splitdir[:,0], splitdir[:,1]
    # spec_animal = []
    # spec_animal_dates = []

    spec_dates = len(spec_animal_dates) > 0
    for i,(e,d) in enumerate(zip(animals,animaldates)):
        if e in animals:  #['DO48']:
            if spec_dates:
                if d in spec_animal_dates:
                    spec_animal.append(e)
            else:
                spec_animal.append(e)
                spec_animal_dates.append(animaldates[i])

    # spec_animal = []
    # spec_animal_dates = []
    #
    # for i,e in enumerate(animals):
    #     if e in animals:  #['DO48']:
    #         spec_animal.append(e)
    #         spec_animal_dates.append(animaldates[i])

    # han_size = 1
    do_zscore = True
    run = Main(spec_animal,spec_animal_dates,rf'pickles\{pkl_prefix}_2d_90Hz_6lpass_025hpass_wdlc_TOM_interpol_all_int02s{"_rawsize"*(not do_zscore)}_221125.pkl',tdatadir,
               pdatadir,
               'pupildata_2d',90.0,han_size=0.25,passband=(0.25, 6),aligneddir=aligneddir,
               subjecttype='mouse',dlc_snapshot=[2450000,2250000],overwrite=False, do_zscore=do_zscore,
               lowtype='filter',dirstyle=dirstyle)
    run.load_pdata()
