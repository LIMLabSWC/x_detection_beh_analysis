import os

from psychophysicsUtils import *
import analysis_utils as utils
from datetime import datetime, time, timedelta
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import zscore
import scipy.signal
import time
from humanpipeline import Main as Main

if __name__ == "__main__":
    tdatadir = r'C:\bonsai\data\Dammy'

    pdatadir = r'W:\mouse_pupillometry\mousefam\aligned_mousefam'
    splitdir = np.array([path.split('_') for path in os.listdir(pdatadir)])
    animals, animaldates = splitdir[:,0], splitdir[:,1]

    # han_size = 1
    run = Main(animals,animaldates,r'pickles\mice_fam_2d_200Hz_015Shan_driftcorr_hpass01.pkl',tdatadir,
               r'W:\mouse_pupillometry\mousefam',
               'pupildata_2d',200.0,han_size=.15,hpass=0.1,aligneddir='aligned_mousefam')
    run.load_pdata()