import math
import os.path
from contextlib import suppress

import pandas as pd

from psychophysicsUtils import *
import analysis_utils as utils
from datetime import datetime, time, timedelta
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import zscore
import scipy.signal
import time


# script for building trial data and pupil data dict
# will generate pickle of dict

# for loading pkl

class Main:
    def __init__(self,names, date_list, pkl_filename, tdatadir, pupil_dir,
                 pupil_file_tag, pupil_samplerate=60.0,outlier_params=(4, 4), overwrite=False,
                 han_size=0.2,hpass=0.1,aligneddir='aligned2',subjecttype='humans', dlc_snapshot=None):

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
        self.overwrite = overwrite
        self.subjecttype = subjecttype

        if self.subjecttype == 'mouse':
            self.paireddirs = utils.pair_dir2sess(os.path.split(self.pdir)[0],self.animals)
        else:
            self.paireddirs = utils.pair_dir2sess(self.pdir,self.animals,subject=self.subjecttype)
        self.dlc_snapshot = dlc_snapshot

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
        today = datetime.strftime(datetime.now(),'%y%m%d')
        figdir = os.path.join(os.getcwd(),'figures',today)
        if not os.path.isdir(figdir):
            os.mkdir(figdir)
        if os.path.exists(self.pklname) and self.overwrite is False:
            self.data = dict()
            with open(self.pklname,'rb') as pklfile:
                print('Loading existing data')
                while True:
                    try:
                        y = (pickle.load(pklfile))
                        z = {**self.data, **y}
                        self.data = z
                    except EOFError:
                        print(f'end of file {self.data.keys()}')
                        break
        elif os.path.exists(self.pklname) is False or self.overwrite is True:
            self.data = dict()
        for animal, date in zip(self.animals, self.dates):
            name = f'{animal}_{date}'
            if name not in list(self.data.keys()):
                pupil_filepath = os.path.join(self.pdir,f'{animal}_{date}_{self.pupil_file_tag}a.csv')
                if not os.path.isfile(pupil_filepath):
                    pupil_filepath = os.path.join(self.pdir,f'{animal}_{date}_{self.pupil_file_tag}.csv')
                    if not os.path.isfile(pupil_filepath):
                        pupil_filepath = os.path.join(os.path.join(self.pdir,self.aligneddir),
                                                      f'{animal}_{date}_{self.pupil_file_tag}a.csv')
                if os.path.isfile(pupil_filepath):
                    # Load pupil date for animal as pandas dataframe
                    animal_pupil = pd.read_csv(pupil_filepath)

                    print(f'loaded {pupil_filepath}')
                    self.data[name] = pupilDataClass(animal)

                    # load dlc if mouse
                    if self.subjecttype in ['mouse','human']:
                        sess_recdir = self.paireddirs[f'{animal}_{date}']
                        if isinstance(sess_recdir,list):
                            _dlc_list = []
                            for rec in sess_recdir:
                                dlc_pathfile = os.path.join(rec,f'eye0DLC_resnet50_mice_pupilJul4shuffle1_{self.dlc_snapshot}.h5')
                                if not os.path.isfile(dlc_pathfile):
                                    dlc_pathfile= os.path.join(rec,f'eye0DLC_resnet50_mice_pupilJul4shuffle1_{700000}.h5')
                                if not os.path.isfile(dlc_pathfile):
                                    dlc_pathfile = None
                                if dlc_pathfile is not None:
                                    try:
                                        _dlc_df = pd.read_hdf(dlc_pathfile)
                                        _dlc_list.append(_dlc_df)
                                    except pd.errors.ParserError:
                                        print(dlc_pathfile.upper())

                            if len(_dlc_list) == 0:
                                print('missing dlc')
                                continue
                            else:
                                dlc_df = pd.concat(_dlc_list)
                        else:
                            try:dlc_pathfile = os.path.join(sess_recdir,f'eye0DLC_resnet50_mice_pupilJul4shuffle1_{self.dlc_snapshot}.h5')
                            except TypeError: (print(sess_recdir))
                            if not os.path.isfile(dlc_pathfile):
                                dlc_pathfile= os.path.join(sess_recdir,f'eye0DLC_resnet50_mice_pupilJul4shuffle1_{700000}.h5')
                            if not os.path.isfile(dlc_pathfile):
                                print('missing dlc')
                                continue
                            dlc_df = pd.read_hdf(dlc_pathfile)
                        dlc_ell = utils.get_dlc_diams(dlc_df,animal_pupil.shape[0])
                        dlc_colnames = ['dlc_radii_a','dlc_radii_b','dlc_centre_x','dlc_centre_x']
                        for colname,coldata in zip(dlc_colnames,dlc_ell):
                            animal_pupil[colname] = coldata
                        animal_pupil['dlc_area'] = (animal_pupil['dlc_radii_a']*animal_pupil['dlc_radii_b']).apply(lambda x: x*math.pi)
                        animal_pupil['dlc_radii_ab'] = (animal_pupil['dlc_radii_a']+animal_pupil['dlc_radii_b'])/2

                    # animal_pupil['scalar'] = [scalarTime(t) for t in animal_pupil['frametime']]
                    animal_pupil.index = utils.format_timestr(animal_pupil['frametime'])[1]

                    date_dt = datetime.strptime(date,'%y%m%d')
                    pupil_df_ix = animal_pupil.index
                    merged_ix = [e.replace(year=date_dt.year,month=date_dt.month,day=date_dt.day) for e in pupil_df_ix]
                    animal_pupil.index = merged_ix

                    print(f'Uniformly sampling to {1/self.samplerate} Hz')
                    pupil_uni = animal_pupil.resample(f'{self.samplerate}S').backfill().copy()  # resample to samplerate

                    # get pupil area
                    for col in ['2d_radii']:
                        col_radius = [e[1:-1].split(',') for e in pupil_uni[col]]
                        for i,coord in enumerate(['a','b']):
                            pupil_uni[f'{col}_{coord}'] = [float(e[i]) for e in col_radius]
                    pupil_uni['rawarea'] = np.array(pupil_uni['2d_radii_a']*pupil_uni['2d_radii_b']*np.pi)

                    # get 2d/3d xy

                    _, pupil_uni['dlc_isout'] = removeouts(pupil_uni['dlc_area'],n_speed=10000,n_size=3.5)

                    for col in ['2d_centre']:
                        col_xy = [e[1:-1].split(',') for e in pupil_uni[col]]
                        for i,coord in enumerate(['x','y']):
                            pupil_uni[f'{col}_{coord}'] = [float(e[i]) for e in col_xy]
                    if self.pupil_file_tag == 'pupildata_3d':
                        pupil_uni['anyisout'],pupil_uni['confisout'], self.data[name].allisout = self.get_outliers(pupil_uni['2d_centre_x'],
                                                                                                                   pupil_uni['2d_centre_y'],
                                                                                                                   pupil_uni['rawarea'],
                                                                                                                   pupil_uni.get('diameter_3d',np.zeros_like(pupil_uni.index)),
                                                                                                                   pupil_uni['confidence'])
                    elif self.pupil_file_tag == 'pupildata_2d':
                        pupil_uni['anyisout'],pupil_uni['confisout'], self.data[name].allisout = self.get_outliers(pupil_uni['2d_centre_x'],
                                                                                                                   pupil_uni['2d_centre_y'],
                                                                                                                   pupil_uni['rawarea'],
                                                                                                                   pupil_uni.get('diameter_2d',np.zeros_like(pupil_uni.index)),
                                                                                                                   pupil_uni['confidence'])


                    b, a = scipy.signal.butter(3, 0.025)
                    for col2norm in ['rawarea','diameter_3d','dlc_area','dlc_radii_a','dlc_radii_b','dlc_radii_ab']:

                        print(f'interpolating {col2norm}')
                        # col_noouts= pupil_uni[col2norm].where(pupil_uni['anyisout']==0,np.nan)
                        if 'dlc' in col2norm:
                            col_noouts= pupil_uni[col2norm].where(pupil_uni['dlc_isout']==0,np.nan)
                        else:
                            col_noouts= pupil_uni[col2norm].where(pupil_uni['confisout']==0,np.nan)
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

                    try:
                        df_cols = pupil_uni.columns
                        cols2use_ix = ['timestamp' in e or 'zscored' in e or 'out' in e for e in df_cols]
                        cols2use =df_cols[cols2use_ix]
                        session_TD = self.trial_data.loc[animal, date].copy()
                        self.data[name].pupildf = pupil_uni[cols2use]
                    except KeyError:
                        print(f'KeyError for session {animal,date}')
                        continue

                    for col in session_TD.keys():
                        if 'Time' in col or 'Start' in col or 'End' in col:
                            if 'Wait' not in col and 'dt' not in col:
                                session_TD[f'{col}_scalar'] = [scalarTime(t) for t in session_TD[col]]
                    self.data[name].trialData = session_TD  # add session trialdata

        if self.pklname is not None:
            with open(self.pklname,'wb') as pklfile:
                pickle.dump(self.data,pklfile)


if __name__ == "__main__":
    tdatadir = r'C:\bonsai\data\Hilde'
    # fam task
    # humans = [f'Human{i}' for i in range(16,28)]
    # humandates = ['220208','220209','220210','220215',
    #               '220311','220316','220405','220407','220407','220408','220422','220425']

    # humans dev norm task
    humans = [f'Human{i}' for i in range(28,33)]
    humandates = ['220518', '220523', '220524','220530','220627']

    # with suppress(ValueError):
    #     humans.remove('Human29')
    #     humandates.remove('220523')

    # han_size = 1
    run = Main(humans,humandates,r'pickles\human_class1_3d_200Hz_015Shan_driftcorr_hpass01.pkl',tdatadir,r'W:\humanpsychophysics\HumanXDetection\Data',
               'pupildata_3d',200.0,han_size=.15,hpass=0.1,aligneddir='aligned_class1',subjecttype='human',
               overwrite=False,dlc_snapshot=1300000)
    run.load_pdata()
    # plt.plot(run.data['Human21_220316'].pupildf['rawarea_zscored'])
    # plt.plot(run.data['Human25_220408'].pupildf['rawarea_zscored'])


