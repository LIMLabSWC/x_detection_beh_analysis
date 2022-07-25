import os

from psychophysicsUtils import *
import analysis_utils as utils
from datetime import datetime, time, timedelta
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import zscore
import scipy.signal
import time
from pupilpipeline import Main as Main


if __name__ == "__main__":
    tdatadir = r'C:\bonsai\data\Dammy'

    pdatadir = r'W:\mouse_pupillometry\mousenormdev\aligned_mousenormdev'
    # pdatadir = r'W:\mouse_pupillometry\mousefam\aligned_mousefam'

    aligneddir = os.path.split(pdatadir)[-1]
    splitdir = np.array([path.split('_') for path in os.listdir(pdatadir)])
    animals, animaldates = splitdir[:,0], splitdir[:,1]
    spec_animal = []
    spec_animal_dates = []

    for i,e in enumerate(animals):
        if e in animals:  #['DO48']:
            spec_animal.append(e)
            spec_animal_dates.append(animaldates[i])

    # han_size = 1
    run = Main(spec_animal,spec_animal_dates,r'pickles\mouse_normdev_2d_200Hz_025Shan_driftcorr_hpass04_wdlc.pkl',tdatadir,
               pdatadir,
               'pupildata_2d',200.0,han_size=.25,hpass=0.25,aligneddir=aligneddir,
               subjecttype='mouse',dlc_snapshot=1300000,overwrite=False)
    run.load_pdata()
