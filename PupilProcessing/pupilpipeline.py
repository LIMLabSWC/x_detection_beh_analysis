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
from copy import deepcopy as copy
import glob

# script for building trial data and pupil data dict
# will generate pickle of dict

# for loading pkl


def remove_missed_ttls(ts: np.ndarray) -> np.ndarray:
    # get average dt
    ts = pd.Series(ts)
    ts_diff = ts.diff()
    mean_dt = ts_diff.mean()
    good_idx = np.where(ts_diff.round('0.01S') > mean_dt.round('0.01S'))[0]
    return ts[good_idx]


class Main:
    def __init__(self,names, date_list, pkl_filename, tdatadir, pupil_dir,
                 pupil_file_tag, pupil_samplerate=60.0,outlier_params=(4, 4), overwrite=False, do_zscore=True,
                 han_size=0.2,passband=(0.1,3),aligneddir='aligned2',subjecttype='humans', dlc_snapshot=None,
                 lowtype='filter',dirstyle=r'Y_m_d\it'):

        # load trial data
        daterange = [sorted(date_list)[0], sorted(date_list)[-1]]
        self.trial_data = utils.merge_sessions(tdatadir,names,'TrialData',daterange)
        self.trial_data = pd.concat(self.trial_data,sort=False,axis=0)
        for col in self.trial_data.keys():
            if 'Time' in col or 'Start' in col or 'End' in col:
                if 'Wait' not in col and 'dt' not in col and col.find('Harp') == -1 and col.find(
                        'Bonsai') == -1 and 'Lick' not in col:
                    self.trial_data[f'{col}_scalar'] = [scalarTime(t) for t in self.trial_data[col]]
        for col in self.trial_data.keys():
            if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1:
                if col.find('Wait') == -1 and col.find('dt') == -1 and col.find('Lick_Times') == -1:
                    utils.add_datetimecol(self.trial_data, col)
        # format trial data df
        try:
            self.trial_data = self.trial_data.drop(columns=['RewardCross_Time','WhiteCross_Time'])
        except KeyError:
            pass
        # add datetime cols
        for col in self.trial_data.keys():
            if 'Time' in col or 'Start' in col or 'End' in col:
                if 'Wait' not in col and 'dt' not in col and col.find('Harp') == -1 and col.find('Bonsai') == -1:
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
        self.passband = passband
        self.overwrite = overwrite
        self.subjecttype = subjecttype
        self.lowtype = lowtype
        self.zscore= do_zscore

        year_lim = datetime.strptime(daterange[0],'%y%m%d').year
        if self.subjecttype == 'mouse':
            if dirstyle == 'N_D_it':
                self.paireddirs = utils.pair_dir2sess(self.pdir, self.animals,
                                                      year_limit=year_lim, dirstyle=dirstyle,
                                                      spec_dates=self.dates, spec_dates_op='=')
            else:
                self.paireddirs = utils.pair_dir2sess(os.path.split(self.pdir)[0],self.animals,
                                                  year_limit=year_lim,dirstyle=dirstyle)
        else:
            self.paireddirs = utils.pair_dir2sess(self.pdir,self.animals,subject=self.subjecttype,
                                                  year_limit=year_lim, dirstyle=dirstyle)
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

    def process_pupil(self,pclass,name,pdf,pdf_colname,timeidx):
        print(f'<< {pdf_colname.upper()} >>')
        if 'dlc' in pdf_colname:
            print('')
        pclass.rawPupilDiams = np.array(pdf[pdf_colname])
        # pclass[name].rawTimes = np.array(pdf.index.asi8)

        pclass.uniformSample(self.samplerate)
        pclass.removeOutliers(n_speed=4, n_size=4)
        pclass.interpolate(gapExtension=0.1)

        # filter blocks of nan dat
        nan_ix = copy(np.isnan(pclass.pupilDiams))
        nan_ix = np.pad(nan_ix,1)
        # nonnan_ix = np.logical_not(nan_ix)
        nan_ix_diff = np.diff(np.nonzero(nan_ix))
        nan_ix_start_end = nan_ix_diff[1:-1].reshape(-1,2)

        # for seq_start_end in nan_ix_start_end:
        #     seq = pclass.pupilDiams[seq_start_end[0]:seq_start_end[1]]
        #     # seq_smoothed = utils.smooth(seq,int(self.han_size/self.samplerate))
        #     seq_smoothed = utils.butter_highpass_filter(seq, 4, 1 / self.samplerate, filtype='low')
        #     seq_filtered = utils.butter_highpass_filter(seq_smoothed,
        #                                                 self.hpass,1/self.samplerate)
        #     pclass.pupilDiams[seq_start_end[0]:seq_start_end[1]] = seq_filtered

        # pclass.pupilDiams = np.nan_to_num(pclass.pupilDiams)
        # pclass.frequencyFilter()
        print(f'nans before interpolation:{np.isnan(pclass.pupilDiams).sum()}')
        pclass.pupilDiams = pd.Series(pclass.pupilDiams).interpolate(limit_direction='both').to_numpy()  # interpolate over nans


        # pclass.pupilDiams = utils.smooth(pclass.pupilDiams,int(self.han_size/self.samplerate))
        if self.zscore:
            if self.lowtype == 'filter':
                pclass.pupilDiams = utils.butter_highpass_filter(pclass.pupilDiams, self.passband,1 / self.samplerate,filtype='band')
            elif self.lowtype == 'hanning':
                pclass.pupilDiams = utils.butter_highpass_filter(utils.smooth(pclass.pupilDiams.copy(),int(self.han_size/self.samplerate)),
                                                                 self.passband[1],1/self.samplerate, filtype='low')
            pclass.zScore()
        else:
            pclass.pupilDiams = utils.butter_highpass_filter(pclass.pupilDiams, self.passband[1], 1 / self.samplerate,
                                                             filtype='low')
        pclass.plot(saveName=f'{name}_{pdf_colname}')

        return pclass.pupilDiams, pclass.isOutlier

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
            session_TD = self.trial_data.loc[animal, date].copy()
            session_TD.set_index('Time_dt', append=True, inplace=True)
            name = f'{animal}_{date}'
            print(f'checking {name}')
            # if name != 'DO57_221130':
            #     continue
            scorer = f'DLC_resnet50_mice_pupilJul4shuffle1_{self.dlc_snapshot[0]}'

            if name in list(self.data.keys()):  # delete empty objects
                if not hasattr(self.data[name],'pupildf'):
                    self.data.pop(name)

            if name not in list(self.data.keys()) :
                pupil_filepath = os.path.join(self.pdir,f'{animal}_{date}_{self.pupil_file_tag}a.csv')
                if not os.path.isfile(pupil_filepath):
                    pupil_filepath = os.path.join(self.pdir,f'{animal}_{date}_{self.pupil_file_tag}.csv')
                    if not os.path.isfile(pupil_filepath):
                        pupil_filepath = os.path.join(os.path.join(self.pdir,self.aligneddir),
                                                      f'{animal}_{date}_{self.pupil_file_tag}a.csv')
                if os.path.isfile(pupil_filepath):
                    # Load pupil date for animal as pandas dataframe
                    animal_pupil = pd.read_csv(pupil_filepath).dropna()

                    print(f'loaded {pupil_filepath}')
                    self.data[name] = pupilDataClass(animal)
                    plabs = True

                else:
                    animal_pupil = pd.DataFrame()
                    sess_recdir = self.paireddirs[f'{animal}_{date}']
                    if isinstance(sess_recdir,(list,tuple,np.ndarray)):
                        recs = pd.concat([pd.read_csv(os.path.join(rec,f'{name}_eye0_timestamps.csv')).iloc[:-1,:]
                                          for rec in list(sess_recdir)],axis=0)

                    else:
                        recs = pd.read_csv(os.path.join(sess_recdir,f'{name}_eye0_timestamps.csv')).iloc[:-1,:]

                    recs['Date'] = np.full_like(recs['Timestamp'],date).astype(str)
                    recs.index = recs['Date']
                    utils.add_datetimecol(recs, 'Bonsai_Time')
                    bonsai0 = recs['Bonsai_Time_dt'][0]  # [int(recs.shape[0]/2.0)]
                    recs['Timestamp_adj'] = recs['Timestamp']-recs['Timestamp'][0]
                    animal_pupil['frametime'] = recs['Timestamp_adj'].apply(lambda e:
                                                                            bonsai0+timedelta(seconds=float(e)/1e9))

                    event92_files = glob.glob(
                        os.path.join(self.pdir, 'harpbins', f'{animal}_HitData_{date}*_event_data_92.csv'))
                    if len(event92_files) > 0:
                        event92_df = pd.concat([pd.read_csv(event_file) for event_file in event92_files], axis=0)
                        cam_ttls = event92_df['Timestamp']
                        cam_ttls_dt = np.full_like(cam_ttls, np.nan, dtype=object)
                        for ei, e in enumerate(cam_ttls.values):
                            harp_dis_min = (session_TD['Harp_Time'] - e).abs().idxmin()
                            try:cam_ttls_dt[ei] = (session_TD['Bonsai_time_dt'][harp_dis_min] - timedelta(
                                hours=float(session_TD['Offset'][harp_dis_min]))
                                               + timedelta(seconds=e - session_TD['Harp_Time'][harp_dis_min]))
                            except KeyError: print(session_TD.columns)
                            except TypeError: print('float error')
                        # ttls2removed = remove_missed_ttls(animal_pupil.index.to_numpy())
                        animal_pupil['frametime'] = cam_ttls_dt[:animal_pupil.shape[0]]

                    self.data[name] = pupilDataClass(animal)
                    plabs = False

                # load dlc if mouse
                if self.subjecttype in ['mouse','human']:
                    sess_recdir = self.paireddirs[f'{animal}_{date}']
                    non_plabs_str = f'{name}_'*np.invert(plabs)
                    if isinstance(sess_recdir,list):
                        _dlc_list = []
                        for rec in sess_recdir:
                            dlc_pathfile = os.path.join(rec,f'{non_plabs_str}eye0DLC_resnet50_mice_pupilJul4shuffle1_{self.dlc_snapshot[0]}.h5')
                            if not os.path.isfile(dlc_pathfile):
                                dlc_pathfile= os.path.join(rec,f'{non_plabs_str}eye0DLC_resnet50_mice_pupilJul4shuffle1_{self.dlc_snapshot[1]}.h5')
                                scorer = f'DLC_resnet50_mice_pupilJul4shuffle1_{self.dlc_snapshot[1]}'
                            if not os.path.isfile(dlc_pathfile):
                                continue
                            if dlc_pathfile is not None:
                                try:
                                    _dlc_df = pd.read_hdf(dlc_pathfile)
                                    _dlc_list.append(_dlc_df)
                                except pd.errors.ParserError:
                                    print(dlc_pathfile.upper())

                        if len(_dlc_list) == 0:
                            print(f'missing dlc for {name}')
                            continue
                        else:
                            dlc_df = pd.concat(_dlc_list)
                    else:
                        try:dlc_pathfile = os.path.join(sess_recdir,f'{non_plabs_str}eye0DLC_resnet50_mice_pupilJul4shuffle1_{self.dlc_snapshot[0]}.h5')
                        except TypeError:
                            (print(sess_recdir))
                            continue
                        if not os.path.isfile(dlc_pathfile):
                            dlc_pathfile= os.path.join(sess_recdir,f'{non_plabs_str}eye0DLC_resnet50_mice_pupilJul4shuffle1_{self.dlc_snapshot[1]}.h5')
                            scorer = f'DLC_resnet50_mice_pupilJul4shuffle1_{self.dlc_snapshot[1]}'
                        if not os.path.isfile(dlc_pathfile):
                            print(f'missing dlc for {name}')
                            continue
                        dlc_df = pd.read_hdf(dlc_pathfile)
                    print(f'loaded dlc for {name}')
                    dlc_ell = utils.get_dlc_diams(dlc_df,animal_pupil.shape[0],scorer)
                    dlc_colnames = ['dlc_radii_a','dlc_radii_b','dlc_centre_x','dlc_centre_x','dlc_EW']
                    for colname,coldata in zip(dlc_colnames,dlc_ell):
                        try:animal_pupil[colname] = coldata
                        except ValueError: print(f'uneven data arrays: {name}')
                    animal_pupil['dlc_area'] = (animal_pupil['dlc_radii_a']*animal_pupil['dlc_radii_b']).apply(lambda x: x*math.pi)
                    animal_pupil['dlc_radii_ab'] = (animal_pupil['dlc_radii_a']+animal_pupil['dlc_radii_b'])/2

                    # animal_pupil['scalar'] = [scalarTime(t) for t in animal_pupil['frametime']]
                    if not isinstance(animal_pupil['frametime'].iloc[0], (pd.Timestamp,datetime)):
                        animal_pupil.index = utils.format_timestr(animal_pupil['frametime'])
                        date_dt = datetime.strptime(date, '%y%m%d')
                        pupil_df_ix = animal_pupil.index
                        merged_ix = [e.replace(year=date_dt.year, month=date_dt.month, day=date_dt.day) for e in
                                     pupil_df_ix]
                        animal_pupil.index = merged_ix
                    else:
                        animal_pupil.index = animal_pupil['frametime']

                    # find large tstamp jumps
                    large_jumps = np.where(animal_pupil.index.to_series().diff() > timedelta(seconds=3600))[0]
                    if len(large_jumps) > 1:
                        print(f'bad pupil timestamping for session: {name}, not including')
                        continue
                    elif len(large_jumps) == 1:
                        print(f'ommitting first {large_jumps[0]} frames for session {name}')
                        animal_pupil = animal_pupil.tail(-large_jumps[0])  # drop rows preceding large jump
                        bonsai0 = recs['Bonsai_Time_dt'][large_jumps[0]]  # [int(recs.shape[0]/2.0)]
                        recs['Timestamp_adj'] = recs['Timestamp'] - recs['Timestamp'][large_jumps[0]]
                        # animal_pupil.reset_index()
                        animal_pupil['frametime'] = recs['Timestamp_adj'].apply(lambda e: bonsai0 + timedelta(
                            seconds=float(e) / 1e9)).tail(-large_jumps[0]).values
                        # animal_pupil.index = animal_pupil['frametime']

                    # get pupil area
                    if plabs:
                        for col in ['2d_radii']:
                            col_radius = [e[1:-1].split(',') for e in animal_pupil[col]]
                            for i,coord in enumerate(['a','b']):
                                animal_pupil[f'{col}_{coord}'] = [float(e[i]) for e in col_radius]
                        animal_pupil['rawarea'] = np.array(animal_pupil['2d_radii_a']*animal_pupil['2d_radii_b']*np.pi)
                        for col in ['2d_centre']:
                            col_xy = [e[1:-1].split(',') for e in animal_pupil[col]]
                            for i,coord in enumerate(['x','y']):
                                animal_pupil[f'{col}_{coord}'] = [float(e[i]) for e in col_xy]
                        animal_pupil_subset = animal_pupil[['confidence', '2d_radii_a', '2d_radii_b', 'rawarea',
                                                            '2d_centre_x', '2d_centre_y',
                                                            'diameter_2d', 'diameter_3d',
                                                            'dlc_radii_a', 'dlc_radii_b', 'dlc_radii_ab',
                                                            'dlc_area', 'dlc_EW']]
                    else:
                        animal_pupil_subset = animal_pupil


                    # Start of Tom's pipeline
                    animal_pupil_subset = animal_pupil_subset.dropna()
                    pupilclass = pupilDataClass(f'{name}')
                    pupilclass.rawTimes = np.array([e.timestamp() for e in animal_pupil_subset.index])
                    unitimes = uniformSample(pupilclass.rawTimes,pupilclass.rawTimes,new_dt=self.samplerate)[1]
                    unitime_ind = [datetime.fromtimestamp(e) for e in unitimes]
                    pupil_uni = pd.DataFrame([],index=unitime_ind)
                    if self.pupil_file_tag == 'pupildata_3d':
                        diam_col = 'diameter_3d'
                    elif self.pupil_file_tag == 'pupildata_2d':
                        diam_col = 'diameter_2d'
                    else:
                        break
                    outs_list = []
                    cols2process = ['dlc_area','dlc_radii_a','dlc_radii_b','dlc_radii_ab','dlc_EW']
                    if plabs:
                        cols2process = cols2process+['rawarea',diam_col,]
                    for col2norm in cols2process:
                        pupil_processed = copy(self.process_pupil(pupilclass,f'{name}_{date}',
                                                                  animal_pupil_subset, col2norm,unitime_ind))
                        pupil_uni[f'{col2norm}_zscored'] = pupil_processed[0]
                        outs_list.append(pupil_processed[1])
                    pupil_uni['isout'],pupil_uni['isout_EW'] = outs_list[3],outs_list[-1]

                    try:
                        df_cols = pupil_uni.columns
                        cols2use_ix = ['timestamp' in e or 'zscored' in e or 'out' in e for e in df_cols]
                        cols2use = df_cols[cols2use_ix]
                        self.data[name].pupildf = pupil_uni[cols2use]
                    except KeyError:
                        print(f'KeyError for session {animal,date}')
                        continue

                    self.data[name].trialData = self.trial_data.loc[animal, date].copy()  # add session trialdata

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
    run = Main(humans,humandates,r'pickles\human_class1_3d_200Hz_015Shan_driftcorr_hpass01_TOM_norm1st_.pkl',tdatadir,r'W:\humanpsychophysics\HumanXDetection\Data',
               'pupildata_3d',200.0,han_size=.15,passband=[0.1,],aligneddir='aligned_class1',subjecttype='human',
               overwrite=False,dlc_snapshot=[1750000,1300000],lowtype='hanning')
    run.load_pdata()
    # plt.plot(run.data['Human21_220316'].pupildf['rawarea_zscored'])
    # plt.plot(run.data['Human25_220408'].pupildf['rawarea_zscored'])


