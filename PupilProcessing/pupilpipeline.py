import logging
import math
import os.path

import pandas as pd

from psychophysicsUtils import *
import analysis_utils as utils
from datetime import datetime, time, timedelta
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import time
from copy import deepcopy as copy
import glob
import pathlib
from pathlib import Path
import yaml
from loguru import logger
from rich.logging import RichHandler
from pyinspect import install_traceback
import psutil
import argparse
import multiprocessing
import platform
import  tqdm
import sys

# script for building trial data and pupil data dict
# will generate pickle of dict


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



def remove_missed_ttls(ts: np.ndarray) -> np.ndarray:
    # get average dt
    ts = pd.Series(ts)
    ts_diff = ts.diff()
    mean_dt = ts_diff.mean()
    good_idx = np.where(ts_diff.round('0.01S') > mean_dt.round('0.01S'))[0]
    return ts[good_idx]


def has_handle(fpath):
    for proc in psutil.process_iter():
        try:
            for item in proc.open_files():
                if fpath == item.path:
                    return True
        except Exception:
            pass

    return False


def get_dlc_est_path(recdir, filt_flag, non_plabs_str,name):
    dlc_estimates_files = []
    if filt_flag:
        dlc_estimates_files_gen = Path(recdir).glob(f'{non_plabs_str}'
                                                 f'eye0DLC_resnet50_mice_pupilJul4shuffle1_'
                                                 f'*_filtered.h5')
        dlc_estimates_files = list(dlc_estimates_files_gen)
    if not len(dlc_estimates_files):
        dlc_estimates_files_gen = Path(recdir).glob(f'{non_plabs_str}'
                                                 f'eye0DLC_resnet50_mice_pupilJul4shuffle1_'
                                                 f'*.h5')
        dlc_estimates_files = list(dlc_estimates_files_gen)
    # dlc_estimates_files = list(dlc_estimates_files_gen)
    dlc_snapshot_nos = [int(str(estimate_file).replace('_filtered', '').split('_')[-1].split('.')[0])
                        for estimate_file in dlc_estimates_files]
    if not len(dlc_estimates_files):
        logger.warning(f'missing dlc for {name}')
        return None,None
    else:
        snapshot2use_idx = np.argmax(dlc_snapshot_nos)
        dlc_pathfile = dlc_estimates_files[snapshot2use_idx]
        dlc_snapshot = dlc_snapshot_nos[snapshot2use_idx]
        return dlc_pathfile,dlc_snapshot


class Main:
    def __init__(self,names, date_list, pkl_filename, tdatadir, pupil_dir,
                 pupil_file_tag, pupil_samplerate=60.0,outlier_params=(4, 4), overwrite=False, do_zscore=True,
                 han_size=0.2,passband=(0.1,3),aligneddir='aligned2',subjecttype='humans', dlc_snapshot=None,
                 lowtype='filter',dirstyle=r'Y_m_d\it',preprocess_pklname='',dlc_filtflag=True,redo=None,
                 protocol='default',use_ttl=False,use_canny_ell=False,session_topology=None):

        # load trial data
        self.existing_sessions = None
        self.pool_results = None
        daterange = [sorted(date_list)[0], sorted(date_list)[-1]]
        self.trial_data = utils.merge_sessions(tdatadir,names,'TrialData',daterange)
        self.trial_data = pd.concat(self.trial_data,sort=False,axis=0)
        # for col in self.trial_data.keys():
        #     if 'Time' in col or 'Start' in col or 'End' in col:
        #         if 'Wait' not in col and 'dt' not in col and col.find('Harp') == -1 and col.find(
        #                 'Bonsai') == -1 and 'Lick' not in col:
        #             self.trial_data[f'{col}_scalar'] = [scalarTime(t) for t in self.trial_data[col]]

        # format trial data df
        try:
            self.trial_data = self.trial_data.drop(columns=['RewardCross_Time','WhiteCross_Time'])
        except KeyError:
            pass
        # add datetime cols
        # self.trial_data['Reaction_time'] = self.trial_data['Trial_End_dt']-(self.trial_data['Trial_Start_dt'] +
        #                                                                     self.trial_data['Stim1_Duration'])
        self.animals = names
        self.anon_animals = [f'Subject {i}' for i in range(len(self.animals))]  # give animals anon labels
        self.dates = date_list

        # init pupil loading vars
        self.samplerate = round(1/pupil_samplerate,3)
        self.pdir = Path(pupil_dir)
        self.pupil_file_tag = pupil_file_tag
        self.aligneddir = Path(aligneddir)
        self.pklname = Path(pkl_filename)
        self.outlier_params = outlier_params
        self.data = {}
        self.preprocessed_pklname = preprocess_pklname
        if self.preprocessed_pklname == '':
            self.preprocessed_pklname = r'pickles\generic_name_plschange.pkl'
        self.preprocessed_pklname = Path(self.preprocessed_pklname)
        self.preprocessed = self.load_pre_processed(self.preprocessed_pklname)
        self.han_size = han_size
        self.passband = passband
        self.overwrite = overwrite
        self.subjecttype = subjecttype
        self.lowtype = lowtype
        self.zscore= do_zscore
        self.dlc_filtflag = dlc_filtflag
        self.redo=redo
        self.protocol = protocol
        self.use_ttl = use_ttl
        self.sessions = {}
        self.use_canny_ell =use_canny_ell
        logger.debug(dirstyle)
        year_lim = datetime.strptime(daterange[0],'%y%m%d').year

        if isinstance(session_topology,pd.DataFrame):
            # assert isinstance(session_topology,pd.DataFrame)
            all_sessnames = set(list(zip(session_topology['name'],session_topology['date'])))
            self.paireddirs = {f'{sessname[0]}_{sessname[1]}':
                                   session_topology.query('name == @sessname[0] and date == @sessname[1]')['videos_dir'].tolist()
                               for sessname in all_sessnames}
        elif self.subjecttype == 'mouse':
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

        today = datetime.strftime(datetime.now(),'%y%m%d')
        self.figdir = Path(r'figures',today)

    def load_pre_processed(self,pre_pklname:pathlib.Path):
        if self.preprocessed_pklname.exists():
            with open(pre_pklname,'rb') as pklfile:
                preprocessed_data = pickle.load(pklfile)
        else:
            'print pre processed pkl not found'
            preprocessed_data = {}
        return preprocessed_data

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

    # @logger.catch
    def process_pupil(self,pclass,name,pdf,pdf_colname,filt_params=None):
        if filt_params == None:
            filt_params = self.passband
        logger.info(f'<< {pdf_colname.upper()} >>')

        pclass.rawPupilDiams = np.array(pdf[pdf_colname])
        # pclass[name].rawTimes = np.array(pdf.index.asi8)

        pclass.uniformSample(self.samplerate)
        # remove linear trend
        sess_mean = pclass.pupilDiams.mean()
        try:pclass.pupilDiams = signal.detrend(np.ma.masked_invalid(pclass.pupilDiams))+sess_mean
        except ValueError: pass

        with HiddenPrints():
            pclass.removeOutliers(n_speed=4, n_size=5)
            pclass.interpolate(gapExtension=0.1)
            pclass.interpolate(gapExtension=0.1)

            # filter blocks of nan dat
            nan_ix = copy(np.isnan(pclass.pupilDiams))
            nan_ix = np.pad(nan_ix,1)
            # nonnan_ix = np.logical_not(nan_ix)
            nan_ix_diff = np.diff(np.nonzero(nan_ix))
            nan_ix_start_end = nan_ix_diff[1:-1].reshape(-1,2)

            logger.info(f'nans before interpolation:{np.isnan(pclass.pupilDiams).sum()}')
            pclass.pupilDiams = pd.Series(pclass.pupilDiams).interpolate(limit_direction='both').to_numpy()  # interpolate over nans

            pupil_diams_nozscore = copy(pclass.pupilDiams)

            # pclass.pupilDiams = utils.smooth(pclass.pupilDiams,int(self.han_size/self.samplerate))
            if self.lowtype == 'filter':
                if filt_params[0] > 0 :
                    pclass.pupilDiams = utils.butter_filter(pclass.pupilDiams, filt_params, 1 / self.samplerate, filtype='band',)
                else:
                    pclass.pupilDiams = utils.butter_filter(pclass.pupilDiams, filt_params[1], 1 / self.samplerate, filtype='low')
            elif self.lowtype == 'hanning':
                if filt_params[0] > 0:
                    pclass.pupilDiams = utils.butter_filter(utils.smooth(pclass.pupilDiams.copy(), int(self.han_size / self.samplerate)),
                                                            filt_params[0], 1 / self.samplerate, filtype='high')
                else:
                    pclass.pupilDiams = utils.smooth(pclass.pupilDiams.copy(), int(self.han_size / self.samplerate))

            if self.zscore:
                pclass.zScore()

        # pclass.plot(self.figdir,saveName=f'{name}_{pdf_colname}',)

        return pclass.pupilDiams, pclass.isOutlier, pupil_diams_nozscore

    # @logger.catch
    def load_pdata(self):
        if not self.figdir.is_dir():
            self.figdir.mkdir()
        if self.pklname.exists() and self.overwrite is False:
            self.data = dict()
            with open(self.pklname,'rb') as pklfile:
                logger.info('Loading existing data')
                while True:
                    try:
                        y = (pickle.load(pklfile))
                        z = {**self.data, **y}
                        self.data = z
                    except EOFError:
                        print(f'end of file {self.data.keys()}')
                        break
            existing_sessions = list(self.data.keys())
            for name in existing_sessions:  # delete empty objects
                if not hasattr(self.data[name],'pupildf'):
                    self.data.pop(name)
                elif not isinstance(self.data[name].pupildf, pd.DataFrame):  # check if None
                    self.data.pop(name)
            existing_sessions = list(self.data.keys())
            self.data = dict()

        # elif self.pklname.exists() is False or self.overwrite is True:
        else:
            self.data = dict()
            existing_sessions = []
        self.existing_sessions = existing_sessions

        if self.redo:
            for sessname in self.redo:
                if sessname in existing_sessions:
                    existing_sessions.remove(sessname)
                if sessname in self.preprocessed.keys():
                    self.preprocessed.pop(sessname)

        for animal in np.unique(self.animals):
            if animal == 'Human19':
                continue
            for date in np.unique(self.dates):
                name = f'{animal}_{date}'
                if name in existing_sessions:
                    continue
                if date not in self.trial_data.loc[animal].index.get_level_values('date'):
                    continue
                session_TD = self.trial_data.loc[animal, date].copy().dropna(axis=1)
                if 'RewardProb' in session_TD.columns and 'prob' not in self.protocol:
                    if session_TD['RewardProb'].sum() > 0:
                        continue
                    if session_TD['Stage'][-1]< 3:
                        continue
                utils.add_dt_cols(session_TD)
                if 'Time_dt' in session_TD.columns:
                    session_TD.set_index('Time_dt', append=True, inplace=True)
                else:
                    session_TD.set_index('Trial_Start_dt', append=True, inplace=True)
                self.sessions[name] = session_TD
                self.data[name] = pupilDataClass(animal)
        # manager = multiprocessing.Manager()
        # self.data = manager.dict(self.data)
        with multiprocessing.Pool() as pool:
            # self.pool_results = list(tqdm(pool.imap(self.read_and_proccess,self.sessions.keys()),
            #                               total=len(self.sessions)))

            sess2run= [sess for sess in self.sessions if sess not in existing_sessions]
            self.pool_results = pool.map(self.read_and_proccess,sess2run)
            # for session in self.sessions:
            # self.read_and_proccess(session,self.sessions[session])
        for sess_name, result in zip(self.sessions, self.pool_results):
            if result[0] is None:
                continue
            self.data[sess_name].pupildf = result[0]
            self.preprocessed[sess_name] = result[1]
            self.data[sess_name].trialData = self.sessions[sess_name]

        # logger.debug(f' data keys{self.data.keys()}')
        # logger.debug(f'pdf  = {self.data[list(self.data.keys())[0]].pupildf.shape}')
        with open(self.pklname, 'ab') as pklfile:
            logger.info(f'Saving {self.pklname}')
            pickle.dump(self.data, pklfile)
        with open(self.preprocessed_pklname, 'wb') as pklfile:
            pickle.dump(self.preprocessed, pklfile)

    def read_and_proccess(self,name:str):
        logger.debug(f'RAM {psutil.virtual_memory().total/(1024**3)} GB')
        logger.debug(f'RAM {psutil.virtual_memory().available/psutil.virtual_memory().total}')

        session_TD = self.sessions[name]
        animal,date = name.split('_')
        logger.info(f'checking {name}')

        if name == 'DO57_221215':
            return None,None
        scorer = f'DLC_resnet50_mice_pupilJul4shuffle1_{self.dlc_snapshot[0]}'


        # if name not in self.existing_sessions:  # list(self.data.keys())
        do_preprocess_steps = not self.preprocessed.get(name,None)
        self.data[name] = pupilDataClass(animal)  # needs to move
        animal_pupil_processed_dfs = []

        if do_preprocess_steps:
            plabs = False
            animal_pupil_dfs = []
            animal_pupil_subset_dfs = []
            animal_pupil_processed_dfs = []

            # for recdir in sess_recdir:
            pupil_filepath = self.pdir / f'{animal}_{date}_{self.pupil_file_tag}a.csv'
            if not pupil_filepath.is_file():
                pupil_filepath =self.pdir/f'{animal}_{date}_{self.pupil_file_tag}.csv'
                if not pupil_filepath.is_file():
                    pupil_filepath = self.pdir/self.aligneddir/f'{animal}_{date}_{self.pupil_file_tag}a.csv'
            if pupil_filepath.is_file():
                # Load pupil date for animal as pandas dataframe
                animal_pupil_dfs.append(pd.read_csv(pupil_filepath).dropna())

                logger.info(f'loaded {pupil_filepath}')
                self.data[name] = pupilDataClass(animal)
                plabs = True

            if plabs is False:
                try:
                    sess_recdir = self.paireddirs[f'{animal}_{date}']
                except KeyError:
                    logger.error(f'no recdir for {animal}_{date}')
                    return None,None
                if isinstance(sess_recdir,str):
                    sess_recdir = [sess_recdir]
                harp_bin_dir = self.pdir.parent.parent/'harpbins'
                event92_files = sorted(harp_bin_dir.glob(f'{animal}_HitData_{date}*_event_data_92.csv'))
                if len(list(event92_files)) > 0 and self.use_ttl:  # code to realign pupil frames and ttls in case of crash
                    event92_df_list = [pd.read_csv(event_file) for event_file in event92_files]
                else:
                    event92_df_list = None

                if isinstance(sess_recdir,(list,tuple,np.ndarray)):
                    # check files
                    if not Path(sess_recdir[0],f'{name}_eye0_timestamps.csv').is_file():
                        return None,None
                    try:
                        recs_list = [pd.read_csv(Path(rec,f'{name}_eye0_timestamps.csv'))
                                        for rec in list(sess_recdir)]
                    except pd.errors.EmptyDataError:
                        logger.error(f'issue with {" ".join(sess_recdir)}')
                        return None,None

                    # recs = pd.concat(recs_list,axis=0)
                    if len(list(event92_files)) > 0 and self.use_ttl:
                        list_match_ttl_pupil = []
                        try:
                            for event_df, rec_df in zip(event92_df_list,recs_list):
                                list_match_ttl_pupil.append(rec_df.head(event_df.shape[0]))
                        except TypeError:
                            pass
                        recs = list_match_ttl_pupil
                    else:
                        recs = recs_list
                else:
                    logger.debug('sess_recdir not list')
                    return None,None

                for ri,rec in enumerate(recs):
                    if rec.empty:
                        logger.warning(f'Recording for {name} empty')
                        continue
                    animal_pupil = pd.DataFrame()  # init df
                    rec.rename(index=str,columns={'timestamp': 'Timestamp'},inplace=True)
                    if self.subjecttype == 'mouse':
                        rec['date'] = np.full_like(rec['Timestamp'],date).astype(str).copy()
                        rec.index = rec['date']
                        try:utils.add_datetimecol(rec, 'Bonsai_Time')
                        except: logger.error(f'add_datetime col failed for Bonsai_Time for pupil_df {name}')
                        try: bonsai0 = rec['Bonsai_Time_dt'].iloc[0]  # [int(recs.shape[0]/2.0)]
                        except KeyError:
                            print(name)
                            continue
                        rec['Timestamp_adj'] = rec['Timestamp']-rec['Timestamp'].iloc[0].copy()
                        animal_pupil['frametime'] = rec['Timestamp_adj'].apply(lambda e:
                                                                                bonsai0+timedelta(seconds=float(e)/1e9))
                        if len(list(event92_files)) > 0 and self.use_ttl:  #  change back to >0
                            use_cam_timestamp = True
                            event92_df = event92_df_list[ri]
                            cam_ttls = event92_df['Timestamp']
                            zeroed_frameids = rec['FrameID']-rec['FrameID'].iloc[0]
                            print(f'{len(cam_ttls)} ttl frames for {name} {ri} \n'
                                    f'{len(rec)} frames in rec {name} {ri}')
                            try:
                                if len(cam_ttls) == len(rec):
                                    matched_ttls_times = cam_ttls
                                else:
                                    print(f'ttl mismatch for {name} {ri}')
                                    matched_ttls_times = cam_ttls.iloc[zeroed_frameids[zeroed_frameids<len(cam_ttls)]]
                            except IndexError:
                                logger.error(f'ttl mismatch for {name} {ri}')
                                continue

                            if use_cam_timestamp:
                                harp_sync_ttl_offest_secs = cam_ttls[0] - session_TD['Harp_time'].iloc[0]
                                new_times = rec['Timestamp_adj'].apply(lambda e:
                                                                                session_TD['Bonsai_time_dt'].iloc[0]
                                                                                +timedelta(seconds=float(e)/1e9+harp_sync_ttl_offest_secs))
                                animal_pupil['frametime'] = new_times
                                animal_pupil['timestamp'] = matched_ttls_times.values

                            else:
                                cam_ttls_dt = np.full_like(cam_ttls, np.nan, dtype=object)
                                ttl_harp_times = np.array(cam_ttls.values)
                                mega_matrix = np.abs(np.array(np.matrix(session_TD['Harp_time'])).T-ttl_harp_times)
                                mega_matrix_mins_idx = np.argmin(mega_matrix, axis=0)
                                matrix_bonsai_times = np.array(session_TD['Bonsai_time_dt'][mega_matrix_mins_idx]
                                                                ,dtype='datetime64[us]')
                                matrix_offset_times = np.array(session_TD['Offset'][mega_matrix_mins_idx],dtype=float)
                                matrix_harp_times = np.array(session_TD['Harp_time'][mega_matrix_mins_idx]
                                                                ,dtype='datetime64[us]')
                                matrix_d_harp_times = ttl_harp_times - \
                                                        session_TD['Harp_time'][mega_matrix_mins_idx]
                                tdelta_arr = np.array((matrix_d_harp_times*1e6).astype(int) +
                                                        (matrix_offset_times*3600*1e6).astype(int),
                                                        dtype='timedelta64[us]')
                                ttl_bonsai_time = matrix_bonsai_times + tdelta_arr
                                # animal_pupil = pd.DataFrame(ttl_bonsai_time,columns=['frametime'])
                                # animal_pupil.index=animal_pupil['frametime']
                                try: animal_pupil['frametime'] = ttl_bonsai_time[:animal_pupil.shape[0]]
                                except: logger.warning(f'ttl times not used for: {name}')

                            # for ei, e in enumerate(cam_ttls.values):
                            #     harp_dis_min = (session_TD['Harp_time'] - e).abs().idxmin()
                            #     try:cam_ttls_dt[ei] = (session_TD['Bonsai_time_dt'][harp_dis_min] - timedelta(
                            #         hours=float(session_TD['Offset'][harp_dis_min]))
                            #                        + timedelta(seconds=e - session_TD['Harp_time'][harp_dis_min]))
                            #     except KeyError: logger.debug(session_TD.columns)
                            #     except TypeError: logger.warning('float error')
                            # # ttls2removed = remove_missed_ttls(animal_pupil.index.to_numpy())
                            # try:animal_pupil['frametime'] = cam_ttls_dt[-animal_pupil.shape[0]:]
                            # except:
                            #     logger.warning(f'ttl times not used for: {name}')
                    animal_pupil_dfs.append(animal_pupil.copy())

            if self.subjecttype in ['mouse','human']:
                logger.info('loading dlc')
                try:sess_recdir = self.paireddirs[f'{animal}_{date}']
                except KeyError: return None,None
                if sess_recdir is None:
                    return None,None
                non_plabs_str = f'{name}_'*np.invert(plabs)
                if not isinstance(sess_recdir,list):
                    sess_recdir = [sess_recdir]
                _dlc_list = []
                for rec_ix, rec in enumerate(sess_recdir):
                    # dlc_estimates_files = []
                    # if self.dlc_filtflag:
                    #     dlc_estimates_files_gen = Path(rec).glob(f'{non_plabs_str}'
                    #                                          f'eye0DLC_resnet50_mice_pupilJul4shuffle1_'
                    #                                          f'*_filtered.h5')
                    #     dlc_estimates_files = list(dlc_estimates_files_gen)
                    # if not len(dlc_estimates_files):
                    #     dlc_estimates_files_gen = Path(rec).glob(f'{non_plabs_str}'
                    #                                          f'eye0DLC_resnet50_mice_pupilJul4shuffle1_'
                    #                                          f'*.h5')
                    #     dlc_estimates_files = list(dlc_estimates_files_gen)
                    # # dlc_estimates_files = list(dlc_estimates_files_gen)
                    # dlc_snapshot_nos = [int(str(estimate_file).replace('_filtered','').split('_')[-1].split('.')[0])
                    #                     for estimate_file in dlc_estimates_files]
                    # if not len(dlc_estimates_files):
                    #     logger.warning(f'missing dlc for {name}')
                    #     continue
                    # else:
                    #     snapshot2use_idx = np.argmax(dlc_snapshot_nos)
                    #     dlc_pathfile = dlc_estimates_files[snapshot2use_idx]
                    dlc_pathfile, snapshot2use = get_dlc_est_path(rec,self.dlc_filtflag,non_plabs_str,name)
                    scorer = f'DLC_resnet50_mice_pupilJul4shuffle1_{snapshot2use}'
                    if dlc_pathfile is not None:
                        try:
                            _dlc_df = pd.read_hdf(dlc_pathfile)
                            if not plabs:
                                _dlc_list.append(_dlc_df.head(animal_pupil_dfs[rec_ix].shape[0]))
                            else:
                                _dlc_list.append(_dlc_df.head(animal_pupil_dfs[0].shape[0]))

                        except pd.errors.ParserError:
                            logger.error(str(dlc_pathfile).upper())
                        except IndexError: continue

                if len(_dlc_list) == 0:
                    logger.warning(f'missing dlc for {name}')
                    return None,None
                else:
                    dlc_dfs = _dlc_list

                logger.info(f'loaded dlc for {name}')
                if plabs:
                    dlc_dfs = [pd.concat(dlc_dfs,axis=0)]
                assert len(dlc_dfs) == len(animal_pupil_dfs)
                for ri,dlc_df in enumerate(dlc_dfs):
                    animal_pupil = animal_pupil_dfs[ri]
                    animal_pupil = animal_pupil.dropna()
                    if dlc_df.shape[0] != animal_pupil.shape[0]:
                        if dlc_df.shape[0] - animal_pupil.shape[0] < 5:
                            animal_pupil = animal_pupil.iloc[:dlc_df.shape[0],:]
                        else:
                            logger.warning(f'many missing frames. skipping {name}')
                            continue
                    scorer = dlc_df.columns.get_level_values(0)[0]
                    dlc_ell = utils.get_dlc_diams(dlc_df,animal_pupil.shape[0],scorer)
                    dlc_colnames = ['dlc_radii_a','dlc_radii_b','dlc_centre_x',
                                    'dlc_centre_y','dlc_EW','dlc_LR']
                    for colname,coldata in zip(dlc_colnames,dlc_ell):
                        try:animal_pupil[colname] = coldata
                        except ValueError:
                            logger.warning(f'uneven data arrays: {name}')
                            continue
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
                        animal_pupil_subset_df = animal_pupil[
                            ['timestamp','confidence', '2d_radii_a', '2d_radii_b', 'rawarea',
                                '2d_centre_x', '2d_centre_y',
                                'diameter_2d', 'diameter_3d',
                                'dlc_radii_a', 'dlc_radii_b', 'dlc_radii_ab',
                                'dlc_area', 'dlc_EW', 'dlc_LR']]

                    else:
                        animal_pupil_subset_df = animal_pupil
                    if self.use_canny_ell:
                        try:rec_canny_ell_df = pd.read_csv(Path(rf'{sess_recdir[ri]}',f'{name}_canny_ellipses.csv'))
                        except FileNotFoundError:
                            logger.error(f'no canny for {name}. not processing')
                            continue
                        if len(animal_pupil.index) != len(rec_canny_ell_df):
                            logger.error(f'incomplete canny file for {name}')
                            continue
                        rec_canny_ell_df.index = animal_pupil.index
                        animal_pupil_subset_df = pd.concat([animal_pupil_subset_df, rec_canny_ell_df], axis=1)
                    animal_pupil_subset_dfs.append(animal_pupil_subset_df)
                self.preprocessed[name] = animal_pupil_subset_dfs

            # Start of Tom's pipeline
        else:
            animal_pupil_subset_dfs = self.preprocessed[name]
        if not animal_pupil_subset_dfs:
            print(name)
        for animal_pupil_subset in animal_pupil_subset_dfs:  # process sessions on day separately
            # animal_pupil_subset = animal_pupil_subset.dropna()
            pupilclass = pupilDataClass(f'{name}')
            # pupilclass.rawTimes = np.array([e.timestamp() for e in animal_pupil_subset.index])
            pupilclass.rawTimes = animal_pupil_subset['timestamp'].values
            with HiddenPrints():
                unitimes = uniformSample(pupilclass.rawTimes,pupilclass.rawTimes,new_dt=self.samplerate)[1]
                # uni_timestamps = uniformSample(animal_pupil_subset['timestamp'].values,
                #                                animal_pupil_subset['timestamp'].values,new_dt=self.samplerate)[1]
            unitime_ind = [datetime.fromtimestamp(e) for e in unitimes]
            # pupil_uni = pd.DataFrame([],index=unitime_ind)
            pupil_uni = pd.DataFrame([],index=unitimes)
            # pupil_uni['timestamp'] = uni_timestamps
            if self.pupil_file_tag == 'pupildata_3d':
                diam_col = 'diameter_3d'
            elif self.pupil_file_tag == 'pupildata_2d':
                diam_col = 'diameter_2d'
            else:
                break
            outs_list = []
            cols2process = ['dlc_radii_a','dlc_radii_b','dlc_radii_ab','dlc_EW','dlc_LR',]
            if self.use_canny_ell:
                cols2process += ['canny_centre_x','canny_centre_y','canny_raddi_a','canny_raddi_b',]
            if 'diameter_2d' in animal_pupil_subset.columns:
                cols2process = cols2process+['rawarea',diam_col,]
            for col2norm in cols2process:
                pupil_processed = self.process_pupil(pupilclass,f'{name}_{date}',
                                                        animal_pupil_subset, col2norm)
                pupil_uni[f'{col2norm}_zscored'] = pupil_processed[0][:pupil_uni.shape[0]]
                pupil_uni[f'{col2norm}_processed'] = pupil_processed[2][:pupil_uni.shape[0]]
                outs_list.append(pupil_processed[1])
            pupil_uni['isout'],pupil_uni['isout_EW'] = outs_list[3],outs_list[3]
            pupil_uni['dlc_EW_normed'] = pupil_uni['dlc_EW_processed']/pupil_uni['dlc_LR_processed']

            try:
                df_cols = pupil_uni.columns
                print(f'{df_cols = }')
                cols2use_ix = ['timestamp' in e or 'zscored' in e or 'out' in e or 'processed' in e
                                or 'normed' in e for e in df_cols]
                cols2use = df_cols[cols2use_ix]
                animal_pupil_processed_dfs.append(pupil_uni[cols2use])
            except KeyError:
                logger.warning(f'KeyError for session {animal,date}')
                continue
        if len(animal_pupil_processed_dfs) == 0:
            logger.critical(f'no dfs for {name}')
        try:self.data[name].pupildf = pd.concat(animal_pupil_processed_dfs,axis=0)
        except:
            logger.critical(f'<NO DFs FOR {name}')
            return None,None
        self.data[name].trialData = self.trial_data.loc[animal, date]  #.copy()  # add session trialdata
        # if 'Stage' not in self.data[name].trialData.columns:
        #     if 'fam' in self.pklname:
        #         self.data[name].trialData['Stage'] = np.full_like(self.data[name].trialData.index.to_series(), 3)
        #     else:
        #         self.data[name].trialData['Stage'] = np.full_like(self.data[name].trialData.index.to_series(), 4)
        return self.data[name].pupildf, self.preprocessed[name]  #, self.preprocessed[name] df.copy()


if __name__ == "__main__":
    logger_path = Path.cwd()/'log'/'logfile.txt'
    logger_path = utils.unique_file_path(logger_path)
    logger.add(str(logger_path),level='TRACE')

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('date',default=None)
    args = parser.parse_args()

    with open(args.config_file,'r') as file:
        config = yaml.safe_load(file)

    tdatadir = r'C:\bonsai\data\Hilde'
    # fam task
    humans = [f'Human{i}' for i in range(20,28)]
    humandates = [#'220209','220210','220215',
                  '220311','220316','220405','220407','220407','220408','220422','220425']
    task = 'fam'

    # # humans dev norm task
    # humans = [f'Human{i}' for i in range(28,33)]
    # humandates = ['220518', '220523', '220524','220530','220627']
    # task = 'normdev'

    # with suppress(ValueError):
    #     humans.remove('Human29')
    #     humandates.remove('220523')

    # han_size = 1
    bandpass_met = (0.125, 2)
    han_size = 0.15
    if han_size: lowtype = 'hanning'
    else: lowtype = 'filter'
    do_zscore = True
    fs = 90.0
    pdata_topic = 'pupildata_3d'
    han_size_str = f'hanning{str(han_size).replace(".","")}'*bool(han_size)
    pklname = f'human_{task}_{pdata_topic.split("_")[1]}_{int(fs)}Hz_driftcorr_lpass_detrend{str(bandpass_met[1]).replace(".","")}' \
              f'_hpass{str(bandpass_met[0]).replace(".","")}_flipped_TOM_{han_size_str}{"_rawsize" * (not do_zscore)}.pkl'
    aligned_dir = f'aligned_{task}'
    run = Main(humans,humandates,os.path.join(r'c:\bonsai\gd_analysis\pickles',pklname),tdatadir,r'W:\humanpsychophysics\HumanXDetection\Data',
               pdata_topic,fs,han_size=1,passband=bandpass_met,aligneddir=aligned_dir,subjecttype='human',
               overwrite=True,dlc_snapshot=[1750000,1300000],lowtype=lowtype,do_zscore=do_zscore,redo=config['sess_to_redo'],
               preprocess_pklname=os.path.join(r'c:\bonsai\gd_analysis\pickles', f'human_fam.pkl'))
    run.load_pdata()
    # plt.plot(run.data['Human21_220316'].pupildf['rawarea_zscored'])
    # plt.plot(run.data['Human25_220408'].pupildf['rawarea_zscored'])


