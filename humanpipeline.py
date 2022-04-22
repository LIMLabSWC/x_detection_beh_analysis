import timeit

import pandas as pd

from psychophysicsUtils import *
import analysis_utils as utils
from analysis_utils import align_wrapper, plot_eventaligned
from datetime import datetime, time, timedelta
from matplotlib import pyplot as plt
from copy import deepcopy as copy
import numpy as np
from scipy.stats import zscore
import scipy.signal
import time
from sklearn.linear_model import LinearRegression


# script for building trial data and pupil data dict
# will generate pickle of dict

# for loading pkl

class Main:
    def __init__(self,names, date_list, pkl_filename, tdatadir, pupil_dir,
                 pupil_file_tag, pupil_samplerate=60.0,outlier_params=(4, 4),
                 han_size=0.2,hpass=0.1,aligneddir='aligned2'):

        # load trial data
        daterange = [sorted(date_list)[0], sorted(date_list)[-1]]
        self.trial_data = utils.merge_sessions(tdatadir,names,'TrialData',daterange)
        self.trial_data = pd.concat(self.trial_data,sort=False,axis=0)

        # format trial data df
        try:
            self.trial_data = self.trial_data.drop(columns=['RewardCross_Time','WhiteCross_Time'])
        except KeyError:
            pass
        # add datetime cols
        for col in self.trial_data.keys():
            if 'Time' in col or 'Start' in col or 'End' in col:
                if 'Wait' not in col and 'dt' not in col:
                    utils.add_datetimecol(self.trial_data,col)
        # self.trial_data['Reaction_time'] = self.trial_data['Trial_End_dt']-(self.trial_data['Trial_Start_dt'] +
        #                                                                     self.trial_data['Stim1_Duration'])
        self.animals = names
        self.anon_animals = [f'Subject {i}' for i in range(len(self.animals))]  # give animals anon labels
        self.dates = date_list

        # init pupil loading vars
        self.samplerate = round(1/pupil_samplerate,3)
        self.pdir = pupil_dir
        self.pupil_file_tag = pupil_file_tag
        self.aligneddir = aligneddir
        self.pklname = pkl_filename
        self.outlier_params = outlier_params
        self.data = {}
        self.han_size = han_size
        self.hpass = hpass

    def get_outliers(self,rawx,rawy,rawsize,rawdiameter,confidence=None) -> (np.ndarray,np.ndarray):

        # outlier xy
        x_pos, x_isout = removeouts(rawx,n_speed=10000,n_size=5)
        y_pos, y_isout = removeouts(rawy,n_speed=10000,n_size=5)

        # outlier speed/size
        size, size_isout = removeouts(rawsize,n_speed=10000,n_size=5)
        diameter, diameter_isout = removeouts(rawdiameter,n_speed=10000,n_size=5)

        outliers_list = [x_isout,y_isout,size_isout,diameter_isout]
        # outlier confidence
        if confidence is not None:
            outliers_list.append((confidence < 0.6).astype(int))

        outs_arr = np.array(outliers_list)
        return outs_arr.any(axis=0).astype(int),outs_arr[-1,:],outs_arr

    def load_pdata(self):
        print('started')
        today = datetime.strftime(datetime.now(),'%y%m%d')
        figdir = os.path.join(os.getcwd(),'figures',today)
        if not os.path.isdir(figdir):
            os.mkdir(figdir)
        if self.pklname is not None or os.path.exists(self.pklname) is False:
            for animal, date in zip(self.animals, self.dates):
                name = f'{animal}_{date}'
                try:
                    # Load pupil date for animal as pandas dataframe
                    animal_pupil = pd.read_csv(os.path.join(self.pdir,self.aligneddir,
                                                            f'{animal}_{date}_{self.pupil_file_tag}.csv'))
                    print(f'loaded {animal}_{date}_{self.pupil_file_tag}.csv')
                    self.data[name] = pupilDataClass(animal)

                    animal_pupil['scalar'] = [scalarTime(t) for t in animal_pupil['frametime']]
                    animal_pupil.index = utils.format_timestr(animal_pupil['frametime'])[1]

                    print(f'Uniformly sampling to {1/self.samplerate} Hz')
                    pupil_uni = animal_pupil.resample(f'{self.samplerate}S').backfill().copy() # resample to samplerate
                    #
                    # self.data[name].rawPupilDiams = np.array(pupil_uni['diameter_2d'])
                    # self.data[name].rawTimes = np.array(pupil_uni['scalar'])

                    # get pupil area
                    for col in ['2d_radii']:
                        col_radius = [e[1:-1].split(',') for e in pupil_uni[col]]
                        for i,coord in enumerate(['a','b']):
                            pupil_uni[f'{col}_{coord}'] = [float(e[i]) for e in col_radius]
                    pupil_uni['rawarea'] = np.array(pupil_uni['2d_radii_a']*pupil_uni['2d_radii_b']*np.pi)

                    # get 2d/3d xy
                    for col in ['2d_centre']:
                        col_xy = [e[1:-1].split(',') for e in pupil_uni[col]]
                        for i,coord in enumerate(['x','y']):
                            pupil_uni[f'{col}_{coord}'] = [float(e[i]) for e in col_xy]
                    pupil_uni['anyisout'],pupil_uni['confisout'], self.data[name].allisout = self.get_outliers(pupil_uni['2d_centre_x'],
                                                                                                               pupil_uni['2d_centre_y'],
                                                                                                               pupil_uni['rawarea'],
                                                                                                               pupil_uni.get('diameter_3d',np.zeros_like(pupil_uni.index)),
                                                                                                               pupil_uni['confidence'])
                    # filt params
                    # hanning window
                    # pupil_empty = np.full_like(pupil_uni['anyisout'],0.0)
                    # for col2norm in ['2d_centre_x','2d_centre_y','rawarea','diameter_3d']:
                    b, a = scipy.signal.butter(3, 0.025)
                    for col2norm in ['rawarea','diameter_3d']:

                        print(f'interpolating {col2norm}')
                        # col_noouts= pupil_uni[col2norm].where(pupil_uni['anyisout']==0,np.nan)
                        col_noouts= pupil_uni[col2norm].where(pupil_uni['anyisout']==0,np.nan)
                        before=time.time()
                        pupil_uni[f'{col2norm}_noouts'] = col_noouts
                        method, order = 'spline', 1

                        try:pupil_uni[f'{col2norm}_interpol'] = interpolatepupil(pupil_uni[f'{col2norm}_noouts'])
                        except TypeError: print('booboo')

                        print(f'interpolate {method},ord{order} took {time.time()-before} s')
                        print(f'smoothing  {col2norm}: hanning window size {self.han_size} ms')

                        pupil_uni[f'{col2norm}_ffilt'] = utils.smooth(np.array(pupil_uni[f'{col2norm}_interpol'].interpolate('linear')),
                                                                      int(self.han_size/self.samplerate))
                        if self.hpass:
                            pupil_uni[f'{col2norm}_ffilt'] = utils.butter_highpass_filter(pupil_uni[f'{col2norm}_ffilt'],
                                                                                      self.hpass,1/self.samplerate)
                        # pupil_uni[f'{col2norm}_ffilt'] = scipy.signal.filtfilt(b, a, pupil_uni[f'{col2norm}_interpol'].interpolate('linear', imit_direction='both'))
                        print(f'z-scoring {col2norm}')
                        pupil_uni[f'{col2norm}_zscored'] = zscore(pupil_uni[f'{col2norm}_ffilt'])

                    self.data[name].pupildf = pupil_uni
                    session_TD = self.trial_data.loc[animal, date].copy()

                    for col in session_TD.keys():
                        if 'Time' in col or 'Start' in col or 'End' in col:
                            if 'Wait' not in col and 'dt' not in col:
                                session_TD[f'{col}_scalar'] = [scalarTime(t) for t in session_TD[col]]
                    self.data[name].trialData = session_TD  # add session trialdata
                except FileNotFoundError:
                    print('nothing done')
                    pass
            if self.pklname is not None:
                with open(self.pklname,'wb') as pklfile:
                    pickle.dump(self.data,pklfile)


if __name__ == "__main__":
    humans = [f'Human{i}' for i in range(20,26)]
    tdatadir = r'C:\bonsai\data\Hilde'
    humandates = ['220311','220316','220405','220407','220407','220408']
    # han_size = 1
    run = Main(humans,humandates,r'pickles\human_familiarity_3d_200Hz_015Shan_driftcorr_hpass01.pkl',tdatadir,r'W:\humanpsychophysics\HumanXDetection\Data',
               'pupildata_3d',200.0,han_size=.15,hpass=0.1,aligneddir='aligned3')
    run.load_pdata()
    plt.plot(run.data['Human21_220316'].pupildf['rawarea_zscored'])
    # plt.plot(run.data['Human25_220408'].pupildf['rawarea_zscored'])


