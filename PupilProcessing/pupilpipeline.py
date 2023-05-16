import math
import os.path
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
import psutil
import argparse

# script for building trial data and pupil data dict
# will generate pickle of dict


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


class Main:
    def __init__(self,names, date_list, pkl_filename, tdatadir, pupil_dir,
                 pupil_file_tag, pupil_samplerate=60.0,outlier_params=(4, 4), overwrite=False, do_zscore=True,
                 han_size=0.2,passband=(0.1,3),aligneddir='aligned2',subjecttype='humans', dlc_snapshot=None,
                 lowtype='filter',dirstyle=r'Y_m_d\it',preprocess_pklname='',dlc_filtflag=True,redo=None,
                 protocol='default',use_ttl=False):

        # load trial data
        daterange = [sorted(date_list)[0], sorted(date_list)[-1]]
        self.trial_data = utils.merge_sessions(tdatadir,names,'TrialData',daterange)
        self.trial_data = pd.concat(self.trial_data,sort=False,axis=0)
        # for col in self.trial_data.keys():
        #     if 'Time' in col or 'Start' in col or 'End' in col:
        #         if 'Wait' not in col and 'dt' not in col and col.find('Harp') == -1 and col.find(
        #                 'Bonsai') == -1 and 'Lick' not in col:
        #             self.trial_data[f'{col}_scalar'] = [scalarTime(t) for t in self.trial_data[col]]
        for col in self.trial_data.keys():
            if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1 :
                if col.find('Wait') == -1 and col.find('dt') == -1 and col.find('Lick_Times') == -1 and col.find('Cross') ==-1:
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

        today = datetime.strftime(datetime.now(),'%y%m%d')
        self.figdir = Path(r'c:\bonsai\gd_analysis','figures',today)

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

    @logger.catch
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

        pclass.removeOutliers(n_speed=4, n_size=5)
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
        if self.zscore:
            if self.lowtype == 'filter':
                if filt_params[0] > 0 :
                    pclass.pupilDiams = utils.butter_filter(pclass.pupilDiams, filt_params, 1 / self.samplerate, filtype='band')
                else:
                    pclass.pupilDiams = utils.butter_filter(pclass.pupilDiams, filt_params[1], 1 / self.samplerate, filtype='low')
            elif self.lowtype == 'hanning':
                if filt_params[0] > 0:
                    pclass.pupilDiams = utils.butter_filter(utils.smooth(pclass.pupilDiams.copy(), int(self.han_size / self.samplerate)),
                                                            filt_params[0], 1 / self.samplerate, filtype='high')
                else:
                    pclass.pupilDiams = utils.smooth(pclass.pupilDiams.copy(), int(self.han_size / self.samplerate))
            pclass.zScore()
        else:
            pclass.pupilDiams = utils.butter_filter(pclass.pupilDiams, self.passband[1], 1 / self.samplerate,
                                                    filtype='low')
        pclass.plot(self.figdir,saveName=f'{name}_{pdf_colname}',)

        return pclass.pupilDiams, pclass.isOutlier, pupil_diams_nozscore

    @logger.catch
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
        elif self.pklname.exists() is False or self.overwrite is True:
            self.data = dict()

        if self.redo:
            for sessname in self.redo:
                if sessname in self.data.keys():
                    self.data.pop(sessname)
                if sessname in self.preprocessed.keys():
                    self.preprocessed.pop(sessname)

        for animal in np.unique(self.animals):
            if animal == 'Human19':
                continue
            for date in np.unique(self.dates):
                if date not in self.trial_data.loc[animal].index.get_level_values('Date'):
                    continue
                session_TD = self.trial_data.loc[animal, date].copy()
                if 'RewardProb' in session_TD.columns and 'prob' not in self.protocol:
                    if session_TD['RewardProb'].sum() > 0:
                        continue
                    if session_TD['Stage'][-1]< 3:
                        continue
                if 'Time_dt' in session_TD.columns:
                    session_TD.set_index('Time_dt', append=True, inplace=True)
                else:
                    session_TD.set_index('Trial_Start_dt', append=True, inplace=True)
                name = f'{animal}_{date}'
                logger.info(f'checking {name}')

                if name == 'DO57_221215':
                    continue
                scorer = f'DLC_resnet50_mice_pupilJul4shuffle1_{self.dlc_snapshot[0]}'

                if name in list(self.data.keys()):  # delete empty objects
                    if not hasattr(self.data[name],'pupildf'):
                        self.data.pop(name)

                if name not in list(self.data.keys()):
                    do_preprocess_steps = name not in self.preprocessed
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
                            try:sess_recdir = self.paireddirs[f'{animal}_{date}']
                            except KeyError: continue
                            if isinstance(sess_recdir,str):
                                sess_recdir = [sess_recdir]
                            event92_files = sorted((self.pdir/'harpbins').glob(f'{animal}_HitData_{date}*_event_data_92.csv'))
                            if len(event92_files) > 0 and self.use_ttl:  # code to realign pupil frames and ttls in case of crash
                                event92_df_list = [pd.read_csv(event_file) for event_file in event92_files]
                            else:
                                event92_df_list = None

                            if isinstance(sess_recdir,(list,tuple,np.ndarray)):
                                # check files
                                if not Path(sess_recdir[0],f'{name}_eye0_timestamps.csv').is_file():
                                    continue
                                recs_list = [pd.read_csv(Path(rec,f'{name}_eye0_timestamps.csv')).iloc[:-1,:]
                                             for rec in list(sess_recdir)]
                                # recs = pd.concat(recs_list,axis=0)
                                if len(event92_files) > 0 and self.use_ttl:
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
                                continue

                            for ri,rec in enumerate(recs):
                                animal_pupil = pd.DataFrame()  # init df
                                rec.rename(index=str,columns={'timestamp': 'Timestamp'},inplace=True)
                                if self.subjecttype == 'mouse':
                                    rec['Date'] = np.full_like(rec['Timestamp'],date).astype(str).copy()
                                    rec.index = rec['Date']
                                    utils.add_datetimecol(rec, 'Bonsai_Time')
                                    bonsai0 = rec['Bonsai_Time_dt'][0]  # [int(recs.shape[0]/2.0)]
                                    rec['Timestamp_adj'] = rec['Timestamp']-rec['Timestamp'][0].copy()
                                    animal_pupil['frametime'] = rec['Timestamp_adj'].apply(lambda e:
                                                                                           bonsai0+timedelta(seconds=float(e)/1e9))
                                    if len(event92_files) > 0 and self.use_ttl:  #  change back to >0
                                        event92_df = event92_df_list[ri]
                                        cam_ttls = event92_df['Timestamp']
                                        cam_ttls_dt = np.full_like(cam_ttls, np.nan, dtype=object)
                                        for ei, e in enumerate(cam_ttls.values):
                                            harp_dis_min = (session_TD['Harp_time'] - e).abs().idxmin()
                                            try:cam_ttls_dt[ei] = (session_TD['Bonsai_time_dt'][harp_dis_min] - timedelta(
                                                hours=float(session_TD['Offset'][harp_dis_min]))
                                                               + timedelta(seconds=e - session_TD['Harp_time'][harp_dis_min]))
                                            except KeyError: logger.debug(session_TD.columns)
                                            except TypeError: logger.warning('float error')
                                        # ttls2removed = remove_missed_ttls(animal_pupil.index.to_numpy())
                                        try:animal_pupil['frametime'] = cam_ttls_dt[-animal_pupil.shape[0]:]
                                        except:
                                            logger.warning(f'ttl times not used for: {name}')
                                animal_pupil_dfs.append(animal_pupil.copy())

                        if self.subjecttype in ['mouse','human']:
                            logger.info('loading dlc')
                            try:sess_recdir = self.paireddirs[f'{animal}_{date}']
                            except KeyError: continue
                            if sess_recdir is None:
                                continue
                            non_plabs_str = f'{name}_'*np.invert(plabs)
                            if not isinstance(sess_recdir,list):
                                sess_recdir = [sess_recdir]
                            _dlc_list = []
                            for rec_ix, rec in enumerate(sess_recdir):
                                dlc_estimates_files = []
                                if self.dlc_filtflag:
                                    dlc_estimates_files = Path(rec).glob(f'{non_plabs_str}'
                                                                         f'eye0DLC_resnet50_mice_pupilJul4shuffle1_'
                                                                         f'*_filtered.h5')
                                if not len(dlc_estimates_files):
                                    dlc_estimates_files = Path(rec).glob(f'{non_plabs_str}'
                                                                         f'eye0DLC_resnet50_mice_pupilJul4shuffle1_'
                                                                         f'*.h5')
                                if len(dlc_estimates_files):
                                    dlc_snapshot_nos = [int(str(estimate_file).replace('_filtered','').split('_')[-1].split('.')[0])
                                                        for estimate_file in dlc_estimates_files]
                                    snapshot2use_idx = np.argmax(dlc_snapshot_nos)
                                    dlc_pathfile = dlc_estimates_files[snapshot2use_idx]
                                    scorer = f'DLC_resnet50_mice_pupilJul4shuffle1_{dlc_snapshot_nos[snapshot2use_idx]}'
                                else:
                                    logger.warning(f'missing dlc for {name}')
                                    continue
                                if dlc_pathfile is not None:
                                    try:
                                        _dlc_df = pd.read_hdf(dlc_pathfile)
                                        if not plabs:
                                            _dlc_list.append(_dlc_df.head(animal_pupil_dfs[rec_ix].shape[0]))
                                        else:
                                            _dlc_list.append(_dlc_df.head(animal_pupil_dfs[0].shape[0]))

                                    except pd.errors.ParserError:
                                        print(dlc_pathfile.upper())
                                    except IndexError: continue

                            if len(_dlc_list) == 0:
                                logger.warning(f'missing dlc for {name}')
                                continue
                            else:
                                dlc_dfs = _dlc_list

                            logger.info(f'loaded dlc for {name}')
                            if plabs:
                                dlc_dfs = [pd.concat(dlc_dfs,axis=0)]
                            assert len(dlc_dfs) == len(animal_pupil_dfs)
                            for ri,dlc_df in enumerate(dlc_dfs):
                                animal_pupil = animal_pupil_dfs[ri]
                                dlc_ell = utils.get_dlc_diams(dlc_df,animal_pupil.shape[0],scorer)
                                dlc_colnames = ['dlc_radii_a','dlc_radii_b','dlc_centre_x',
                                                'dlc_centre_y','dlc_EW','dlc_LR']
                                for colname,coldata in zip(dlc_colnames,dlc_ell):
                                    try:animal_pupil[colname] = coldata
                                    except ValueError: logger.warning(f'uneven data arrays: {name}')
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
                                    animal_pupil_subset_dfs.append(animal_pupil[['confidence', '2d_radii_a', '2d_radii_b', 'rawarea',
                                                                        '2d_centre_x', '2d_centre_y',
                                                                        'diameter_2d', 'diameter_3d',
                                                                        'dlc_radii_a', 'dlc_radii_b', 'dlc_radii_ab',
                                                                        'dlc_area', 'dlc_EW','dlc_LR']])
                                else:
                                    animal_pupil_subset_dfs.append(animal_pupil)
                            self.preprocessed[name] = animal_pupil_subset_dfs

                        # Start of Tom's pipeline
                    else:
                        animal_pupil_subset_dfs = self.preprocessed[name]
                    for animal_pupil_subset in animal_pupil_subset_dfs:  # process sessions on day separately
                        # animal_pupil_subset = animal_pupil_subset.dropna()
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
                        cols2process = ['dlc_area','dlc_radii_a','dlc_radii_b','dlc_radii_ab','dlc_EW','dlc_LR']
                        if plabs:
                            cols2process = cols2process+['rawarea',diam_col,]
                        for col2norm in cols2process:
                            pupil_processed = copy(self.process_pupil(pupilclass,f'{name}_{date}',
                                                                      animal_pupil_subset, col2norm))
                            pupil_uni[f'{col2norm}_zscored'] = pupil_processed[0][:pupil_uni.shape[0]]
                            pupil_uni[f'{col2norm}_processed'] = pupil_processed[2][:pupil_uni.shape[0]]
                            outs_list.append(pupil_processed[1])
                        pupil_uni['isout'],pupil_uni['isout_EW'] = outs_list[3],outs_list[-1]
                        pupil_uni['dlc_EW_normed'] = pupil_uni['dlc_EW_processed']/pupil_uni['dlc_LR_processed']

                        try:
                            df_cols = pupil_uni.columns
                            cols2use_ix = ['timestamp' in e or 'zscored' in e or 'out' in e or 'processed' in e
                                           or 'normed' in e for e in df_cols]
                            cols2use = df_cols[cols2use_ix]
                            animal_pupil_processed_dfs.append(pupil_uni[cols2use])
                        except KeyError:
                            logger.warning(f'KeyError for session {animal,date}')
                            continue

                    self.data[name].pupildf = pd.concat(animal_pupil_processed_dfs,axis=0)
                    self.data[name].trialData = self.trial_data.loc[animal, date].copy()  # add session trialdata
                    if 'Stage' not in self.data[name].trialData.columns:
                        if 'fam' in self.pklname:
                            self.data[name].trialData['Stage'] = np.full_like(self.data[name].trialData.index.to_series(), 3)
                        else:
                            self.data[name].trialData['Stage'] = np.full_like(self.data[name].trialData.index.to_series(), 4)

                    while has_handle(self.pklname):
                        time.sleep(0.01)
                    if self.pklname is not None:
                        with open(self.pklname,'wb') as pklfile:
                            pickle.dump(self.data,pklfile)
                    while has_handle(self.preprocessed_pklname):
                        time.sleep(0.01)
                    with open(self.preprocessed_pklname,'wb') as pklfile:
                        pickle.dump(self.preprocessed,pklfile)


if __name__ == "__main__":
    logger_path = Path.cwd()/'log'/'logfile.txt'
    logger_path = utils.unique_file_path(logger_path)
    logger.add(logger_path,level='INFO')

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
               overwrite=False,dlc_snapshot=[1750000,1300000],lowtype=lowtype,do_zscore=do_zscore,
               preprocess_pklname=os.path.join(r'c:\bonsai\gd_analysis\pickles', f'human_fam.pkl'))
    run.load_pdata()
    # plt.plot(run.data['Human21_220316'].pupildf['rawarea_zscored'])
    # plt.plot(run.data['Human25_220408'].pupildf['rawarea_zscored'])


