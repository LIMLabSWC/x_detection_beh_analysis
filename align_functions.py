import glob
import multiprocessing
import os
import warnings
from copy import deepcopy as copy, deepcopy
from datetime import timedelta, datetime
from functools import partial
from pathlib import Path

import numpy
import numpy as np
import pandas
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from tqdm import tqdm
from psychophysicsUtils import pupilDataClass


def float_ceil(a, precision=0):
    return np.round(a + 0.5 * 10.0**(-precision), precision)


def float_floor(a, precision=0):
    return np.round(a - 0.5 * 10.0**(-precision), precision)


def round_dt_seconds(a:datetime, precision=2, method=float_floor,):
    if not isinstance(a,datetime):
        a = pd.to_datetime(a)
    seconds = float(f'{a.time().second}.{a.time().microsecond}')
    assert method in (float_ceil,float_floor,np.round)
    rounded = method(seconds,precision=precision)
    frac,whole = np.modf(rounded)
    return a.replace(second=int(whole),microsecond=int(frac))


def align2events(df, timeseries_data, pupiltimes, pupiloutliers, beh, t_window, filters=('4pupil', 'b1', 'c1'), baseline=False, eventshift=0,
                 outlierthresh=0.5, stdthresh=20, subset=None) -> tuple[pd.DataFrame, np.ndarray]:
    """

    :param df:
    :param timeseries_data:
    :param pupiltimes:
    :param pupiloutliers:
    :param beh:
    :param t_window:
    :param filters:
    :param baseline:
    :param eventshift:
    :param outlierthresh:
    :param stdthresh:
    :param subset:
    :return:
    """

    dataseries = pd.Series(timeseries_data, index=pupiltimes)
    outlierstrace = pd.Series(pupiloutliers,index=pupiltimes)

    filtered_df = filter_df(df,filters)
    t_window = np.array(t_window)
    dt = pupiltimes.diff().abs().min().total_seconds()

    # dataseries.index = pd.to_datetime(dataseries.index.to_series()).dt.floor('0.01S')
    # eventtimes = pd.to_datetime(filtered_df[beh]).dt.floor('0.01S')

    dataseries.index = dataseries.index.round('0.01S')
    eventtimes = filtered_df[beh].round('0.01S')

    # eventtimez = filtered_df[beh]
    if eventtimes.empty:
        return pd.DataFrame(np.array([])), np.nan
    eventsess = filtered_df.index
    eventdatez = [e[1] for e in eventsess]
    eventnamez = [e[0] for e in eventsess]
    trialtimez = filtered_df.get('Trial_Start_dt')

    t_window = t_window + eventshift

    event_tdeltas = np.round(np.arange(t_window[0],t_window[1],dt),2)
    # aligned_epochs_df = pd.DataFrame(eventpupil_arr,columns=event_tdeltas,
    #                                  index=eventtimes)
    aligned_epochs = [[dataseries.get(eventtime+timedelta(0,tt),np.nan) for tt in event_tdeltas]
                      for eventtime in eventtimes]
    aligned_epochs_df = pd.DataFrame(aligned_epochs,columns=event_tdeltas,
                                     index=pd.MultiIndex.from_tuples(list(zip(trialtimez,eventnamez,eventdatez)),
                                                                     names=['time','name','date']))

    aligned_outs = [[outlierstrace.get(eventtime+timedelta(0,tt)) for tt in event_tdeltas]
                      for eventtime in eventtimes]
    aligned_outs_df = pd.DataFrame(aligned_outs,columns=event_tdeltas,
                                   index=pd.MultiIndex.from_tuples(list(zip(trialtimez,eventnamez,eventdatez)),
                                                                   names=['time','name','date']))
    # for col in aligned_epochs_df.columns:
    #     aligned_epochs_df[col] = pd.to_numeric(aligned_epochs_df[col],errors='coerce')
    baseline_dur = 1.0
    epoch_baselines = aligned_epochs_df.loc[:,-baseline_dur:0.0]
    epoch_outs_baselines = aligned_outs_df.loc[:,-baseline_dur:0.0]
    if baseline:
        aligned_epochs_df = aligned_epochs_df.sub(epoch_baselines.mean(axis=1),axis=0)
    bad_epochs = aligned_outs_df.sum(axis=1)>aligned_epochs_df.shape[1]*0.5
    bad_baselines = epoch_outs_baselines.sum(axis=1)>epoch_baselines.shape[1]*0.5
    aligned_epochs_df = aligned_epochs_df[~np.any([bad_epochs,bad_baselines],axis=0)]
    aligned_epochs_df = aligned_epochs_df[aligned_epochs_df.isna().sum(axis=1)<0.25*aligned_epochs_df.shape[1]]
    aligned_epochs_df = aligned_epochs_df.ffill(axis=1,)

    return aligned_epochs_df, np.any([bad_epochs,bad_baselines], axis=0)  # np.where(np.any([bad_epochs,bad_baselines],axis=0))[0]

    # for i, eventtime in enumerate(filtered_df[beh]):
    #     eventtime = eventtime + timedelta(hours=float(df['Offset'].iloc[i]))
    #     # eventtime = eventtime + timedelta(hours=float(1))
    #     eventpupil = copy(dataseries.loc[eventtime + timedelta(0, t_window[0]): eventtime + timedelta(0, t_window[1])])
    #     eventoutliers = copy(outlierstrace.loc[eventtime + timedelta(0, t_window[0]): eventtime + timedelta(0, t_window[1])])
    #     # print((eventoutliers == 0.0).sum(),float(len(eventpupil)))
    #     if len(eventpupil):
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore")
    #             if eventoutliers.sum()/float(eventpupil.shape[0]) > outlierthresh:  # change back outlierthresh
    #                 print(f'pupil trace for trial {i} incompatible',(eventoutliers == 1).sum())
    #                 outliers += 1
    #                 continue
    #             elif eventpupil.abs().max() > stdthresh*20:
    #                 print(f'pupil trace for trial {i} incompatible')
    #                 varied += 1
    #
    #             else:
    #                 # print('diff',eventpupil.loc[eventtime - 1-eventshift]-eventpupil.loc[eventtime + 1-eventshift])
    #                 if baseline:
    #                     baseline_dur = 1.0
    #                     with warnings.catch_warnings():
    #                         warnings.simplefilter("ignore")
    #                         baseline_period = eventpupil.loc[eventtime - timedelta(0,baseline_dur-eventshift):
    #                                                    eventtime + timedelta(0,eventshift)]
    #                         baseline_outs = eventoutliers.loc[eventtime - timedelta(0,baseline_dur-eventshift):
    #                                                    eventtime + timedelta(0,eventshift)]
    #                         if baseline_outs.mean() > .75:  # change back!
    #                             outliers += 1
    #                             print(outliers)
    #                             continue
    #                         else:
    #                             # baseline_mean = np.nanmean(baseline_period[np.invert(baseline_outs)])
    #                             baseline_mean = np.nanmean(baseline_period)
    #                             eventpupil = (eventpupil - baseline_mean)
    #
    #                             # epoch_linregr = sklearn.linear_model.LinearRegression().fit(np.full_like(eventpupil,baseline_mean).reshape(-1,1),eventpupil)
    #                             # eventpupil = (eventpupil-epoch_linregr.intercept_) /epoch_linregr.coef_[0]
    #
    #                 zeropadded = np.full_like(eventpupil_arr[0],0.0)
    #                 try:
    #                     zeropadded[:len(eventpupil)] = eventpupil
    #                     if len(np.unique(zeropadded)) < 10:
    #                         print('weird', filtered_df.index.tolist()[0])
    #                         continue
    #                     eventpupil_arr[i] = zeropadded
    #                 except ValueError:print('bad shape')
    #     else:
    #         print(filtered_df.index[0])
    #         print(f'no event pupil found: eventime = {eventtime}, pupil range = {timeseries_data.index[[0,-1]]}')
    #         continue


    # #print(f'Outlier Trials:{outliers}\n Too high varinace trials:{varied}')
    # # print(eventpupil_arr.shape)
    # if 'Trial_Start_dt' in filtered_df.columns:
    #     index=pd.MultiIndex.from_tuples(list(zip(filtered_df['Trial_Start_dt'],eventnamez,eventdatez)),names=['time','name','date'])
    # else:
    #     index=pd.MultiIndex.from_tuples(list(zip(filtered_df['Trial_Start_dt'],eventnamez,eventdatez)),names=['time','name','date'])
    #
    # eventpupil_df = pd.DataFrame(eventpupil_arr)
    # eventpupil_df.index = index
    # nonans_eventpuil = eventpupil_df[~np.isnan(eventpupil_arr).any(axis=1)]
    #
    # if subset is not None:
    #     midpnt = nonans_eventpuil.shape[0]/2.0
    #     firsts = nonans_eventpuil[:subset,:]
    #     middles = nonans_eventpuil[int(midpnt-subset/2.0):int(midpnt+subset/2.0)]
    #     lasts = nonans_eventpuil[-subset:,:]
    #     # print(firsts.shape,middles.shape,lasts.shape)
    #     return [firsts,middles,lasts],outliers
    # else:
    #     if nonans_eventpuil.size < 10:
    #         pass
    #     return nonans_eventpuil.iloc[:,:-1],outliers


def getpatterntraces(data, patterntypes,beh, dur=None, eventshifts=None,baseline=True,subset=None,regressed=False,
                     dev_subsetdf=None,coord=None, pupilmetricname='rawarea_zscored',sep_cond_cntrl_flag=False,kwargs=dict()):
    dur = kwargs.get('dur',dur)
    if not dur:
        raise Warning('No duration given')

    baseline = kwargs.get('baseline',baseline)
    subset = kwargs.get('subset',subset)
    coord = kwargs.get('coord',coord)
    pupilmetricname = kwargs.get('pupilmetricname', pupilmetricname)
    sep_cond_cntrl_flag = kwargs.get('sep_cond_cntrl_flag', sep_cond_cntrl_flag)

    list_eventaligned = []
    list_n_outliers = []
    if eventshifts is None:
        eventshifts = np.zeros(len(patterntypes))

    trial_align_points = ['Pretone_end_dt' if any(e in patternfilter for e in ['c1','e=0','none']) else beh
                          for patternfilter in patterntypes]
    # for i, patternfilter in tqdm(enumerate(patterntypes),total=len(patterntypes)):  # maybe multiprocess
    #     if beh == 'ToneTime_dt':
    #         if 'e=0' in patternfilter or 'c1' in patternfilter or 'none' in patternfilter:
    #             beh = 'Pretone_end_dt'
    #             print(f'getpatterntraces none {patternfilter}')
    _pattern_tonealigned = []
    if subset is not None:
        firsts, mids, lasts = [], [], []
    if isinstance(data,pupilDataClass):
        if regressed:
            pupil2use = data.pupilRegressed
        elif coord == 'x':
            pupil2use = data.xc
        elif coord == 'y':
            pupil2use = data.yc
        else:
            pupil2use = data.pupildf[pupilmetricname]
        if dev_subsetdf is None:
            td2use = data.trialData
        else: return None,None
        times2use = pd.Series(data.pupildf.index)
        try:outs2use = data.pupildf['isout']
        except KeyError: outs2use = data.pupildf['confisout']
    elif isinstance(data,dict):
        for name in data.keys():
            if regressed:
                pupil2use = data[name].pupilRegressed
            else:
                pupil2use = data[name].pupildf[pupilmetricname]

            td2use = data[name].trialData
            times2use = pd.Series(data[name].pupildf.index)
            if 'dlc' in pupilmetricname:
                outs2use = data[name].pupildf['isout']
            else:
                outs2use = data[name].pupildf['isout']
    else:
        print('Incorrect data structure')
        return None, None
    for i, (patternfilter, align_point) in enumerate(zip(patterntypes, trial_align_points)):
        tone_aligned_pattern = align2events(td2use, pupil2use, times2use,
                                            outs2use, align_point,
                                            dur, patternfilter,
                                            outlierthresh=0.5, stdthresh=2,
                                            eventshift=eventshifts[i], baseline=baseline, subset=subset)
        if sep_cond_cntrl_flag:  # subtract session controls from aligned epoch traces
            if 'e!0' in patternfilter:
                cntrl_patternfilter = copy(list(map(lambda x: x.replace('e!0', 'e=0'), patternfilter)))
                cntrl_patternfilter = list(map(lambda x: x.replace('tones4', 'e=0'), cntrl_patternfilter))
                cond_controls = align2events(td2use, pupil2use, times2use,
                                             outs2use,'Pretone_end_dt',
                                             dur, cntrl_patternfilter,
                                             outlierthresh=0.5, stdthresh=2,
                                             eventshift=eventshifts[i], baseline=baseline, subset=subset)
                if cond_controls[0].shape[0] > 3:
                    _tone_aligned_pattern = copy(tone_aligned_pattern[0]-cond_controls[0].mean())
                    tone_aligned_pattern = (_tone_aligned_pattern,tone_aligned_pattern[1])

        n_outliers = tone_aligned_pattern[1]
        # if n_outliers:
        #     print(f'# Outlier trials = {n_outliers} for {patternfilter}')
        tone_aligned_pattern = tone_aligned_pattern[0]
        # _pattern_tonealigned.append(tone_aligned_pattern)

        # list_eventaligned.append(pd.concat(_pattern_tonealigned, axis=0))
        list_eventaligned.append(tone_aligned_pattern)
        list_n_outliers.append(n_outliers)
    return list_eventaligned,list_n_outliers


def align_nonpuil(eventzz: pd.Series, timeseries: np.ndarray, window: list,
                  offsetseries: pd.Series, name,date, eventshift=0.0,) -> pd.DataFrame:
    window = np.array(window)
    event_dict = {}
    # eventzz = eventzz - offsetseries.apply(lambda e: timedelta(hours=float(e))) + timedelta(seconds=eventshift)
    eventzz_windows = [[e + timedelta(seconds=window[0]),e + timedelta(seconds=window[1])] for e in eventzz]
    names,dates = [name]*len(eventzz), [date]*len(eventzz)
    epoch_tseries = np.round(np.arange(window[0],window[1],0.01),2)
    if len(epoch_tseries)>400:
        print('wtf!')
    all_epoch_event_arr = np.zeros((eventzz.shape[0],epoch_tseries.shape[0]))
    all_epoch_event_df = pd.DataFrame(all_epoch_event_arr,index=pd.MultiIndex.from_tuples(list(zip(eventzz,names,dates)),
                                                                                          names=['time','name','date']),
                                      columns=epoch_tseries)
    for i, (eventtime,eventzz_window) in enumerate(zip(eventzz,eventzz_windows)):
        min_time = eventtime + timedelta(seconds=window[0])
        max_time = eventtime + timedelta(seconds=window[1])
        # idx_bool=np.argwhere(np.all((timeseries>=min_time,timeseries<max_time)))
        try:idx_bool = (timeseries<= max_time) * (timeseries>=min_time)
        except TypeError: print('skipping')
        col_idx = float_floor((timeseries[idx_bool] - np.full_like(timeseries[idx_bool], eventtime))
                              .astype(float) / 1e6, 2)
        if col_idx.size > 0:
            all_epoch_event_df.loc[eventtime,np.unique(col_idx)] = 1
        # print(f'{eventtime}, {i}')
        # event_dict[eventtime] = float_floor((timeseries[idx_bool]-np.full_like(timeseries[idx_bool],eventtime))
        #                                     .astype(float) / 1e6,2)
    # assert len(event_dict) == eventzz.shape[0]
    # return event_dict
    return all_epoch_event_df


def filter_df(data_df, filters) -> pd.DataFrame:
    """
    Create df subset by recursive filtering of dataframe
    :param data_df: pd.Dataframe data frame to filter
    :param filters: str[] list of fildict keys.
            a(123) trialoutcome, b(01) warmup, c(01) tone poistion, d(0123!0) pattern type, e(0!0) tone played,
            see filtdict for full description
    :return: final filtered dataframe
    """
    if 'Pattern_Type' in data_df.columns:
        dev_nonorder = []
        dev_repeat = []
        unique_devs = sorted(data_df[data_df['Pattern_Type']>0]['PatternID'].unique())
        for dev in unique_devs:
            devarray = np.array(dev.split(';')).astype(int)
            if devarray[1] == devarray[2] or devarray[2] == devarray[3]:
                dev_repeat.append(dev)
            else:
                dev_nonorder.append(dev)

        dev_desc = []
        dev_assc = []
        for dev in unique_devs:
            _parray = np.array(dev.split(';')).astype(int)
            diff_parray = np.diff(_parray)

            if all(diff_parray >= 0):
                dev_assc.append(dev)
            else:
                dev_desc.append(dev)

        # subset normals
        unique_norms = sorted(data_df[data_df['Pattern_Type']==0]['PatternID'].unique())
        normtrain_arr = [np.array([15,17,19,20]),np.array([20,22,24,25])]

        normtrain = []
        normtest = []
        for pat in unique_norms:
            _parray = np.array(pat.split(';')).astype(int)
            if np.all(normtrain_arr != _parray):
                normtest.append(pat)
            else:
                normtrain.append(pat)
    else:
        dev_repeat,dev_assc,dev_desc,dev_nonorder,normtrain,normtest = None,None,None,None,None,None

    # get all time=00:00:00
    try:all_dates = data_df.index.to_frame()['date'].unique()
    except KeyError: pass
    all_dates_t0 = [datetime.strptime(e,'%y%m%d') for e in all_dates]

    fildict = {
        'a0': ['Trial_Outcome', 0], # miss
        'a1': ['Trial_Outcome', 1], # correct
        'a2': ['Trial_Outcome', -1], # early
        'a3': ['Trial_Outcome', -1, '!='], # fail, miss or early
        'b0': ['WarmUp', True],
        'b1': ['WarmUp', False],
        'c0': ['Tone_Position', 0],  # Tone coming before X
        'c1': ['Tone_Position', 1],  # Tone coming after X
        'd0': ['Pattern_Type', 0],  # Normal pattern
        'd!0': ['Pattern_Type', 0, '>'],  # deviant pattern
        'd-1': ['Pattern_Type', -1], #
        'd1': ['Pattern_Type', 1], #
        'd2': ['Pattern_Type', 2], #
        'd3': ['Pattern_Type', 3], #
        'd4': ['Pattern_Type', [1,3]], #
        'e!0': ['Tone_Position', 0],  # There was a tone, if 'c0'
        'e=0': ['Tone_Position', 1], # There was no tone
        '4pupil': ['na'],
        'devrep':['PatternID',dev_repeat,'isin'], #
        'devord': ['PatternID', dev_nonorder, 'isin'],
        'devassc': ['PatternID', dev_assc, 'isin'],
        'devdesc': ['PatternID', dev_desc, 'isin'],
        'normtrain': ['PatternID', normtrain, 'isin'],
        'normtest':['PatternID', normtest, 'isin'],
        'start13':['PatternID', '13;15;17;18'],
        'start17': ['PatternID', '13;15;17;18'],
        'start19': ['PatternID', '19;21;23;24'],
        'start15': ['PatternID', '15;17;19;20'],
        'pnone': ['PatternPresentation_Rate',1.0],
        'plow': ['PatternPresentation_Rate',[0.8,0.9]],
        'pmed': ['PatternPresentation_Rate',0.6],
        'phigh': ['PatternPresentation_Rate',0.1],
        'ppost': ['PatternPresentation_Rate',0.4],
        'p0.5': ['PatternPresentation_Rate',0.5],
        'p0': ['PatternPresentation_Rate',0.0],
        'tones4': ['N_TonesPlayed',4],  # how many tones of the pattern was presented
        'tones3': ['N_TonesPlayed',3],
        'tones2': ['N_TonesPlayed',2],
        'tones1': ['N_TonesPlayed',1],
        'tones0': ['N_TonesPlayed',0],
        's-1': ['Session_Block',-1],
        's0': ['Session_Block',0],
        's1': ['Session_Block',1],
        's01': ['Session_Block',[0,1]],
        's2': ['Session_Block',2],
        's3': ['Session_Block',3], # session block in which deviants are presented
        'sess_a': ['Session', 'a'],
        'sess_b': ['Session', 'b'],  # if there are more than one session in one day per animal
        'stage0': ['Stage', 0],
        'stage1': ['Stage', 1],
        'stage2': ['Stage', 2],  # training
        'stage3': ['Stage',3],  # familiarity
        'stage4': ['Stage', 4],  # norm-dev
        'stage5': ['Stage', 5],  # continous norm-dev
        'stage6': ['Stage', 6],  # continous switch
        'DO45': ['animal', 'DO45'],
        'DO46': ['animal', 'DO46'],
        'DO47': ['animal', 'DO47'],
        'DO48': ['animal', 'DO48'],
        'rig1':['animal', ['DO45','DO47']],
        'rig2': ['animal', ['DO46', 'DO48']],
        '0.5_0': ['0.5_order', 0],
        '0.5_1': ['0.5_order', 1],
        '0.5_2': ['0.5_order', 2],
        '0.5_3': ['0.5_order', 3],
        '0.5_-1': ['0.5_order', -1],
        '-1tone': ['Tone_Position_diff',1.0],
        '-1rew': ['Trial_Outcome_diff',-1.0],
        '-1norew': ['Trial_Outcome_diff',1.0],
        '-1same': ['Trial_Outcome_diff',0.0],
        'prew<1': ['RewardProb',1.0,'<'],
        'prew=1': ['RewardProb',1.0],
        'dearly': ['PreTone_Duration',[1.0,2.0]],
        'dlate': ['PreTone_Duration', [4.0, 5.0]],
        'dmid': ['PreTone_Duration', [3.0]],
        'noplicks': ['Lick_in_window',False],
        'd_C2B': ['C_tone_diff', -2],
        'd_C2A': ['C_tone_diff', -4],
        'd_C2D': ['C_tone_diff', 1],
        'local_rate_1.0': ['Tone_Position_roll', [0.0, 0.1]],
        # 'local_rate_1.0': ['Tone_Position_roll', [0.0, 0.3]],
        'local_rate_0.8': ['Tone_Position_roll', [0.2, 0.3]],
        'local_rate_0.6': ['Tone_Position_roll', [0.4, 0.5]],
        'local_rate_0.4': ['Tone_Position_roll', [0.6, 0.7]],
        'local_rate_0.2': ['Tone_Position_roll', [0.8, 0.9]],
        # 'local_rate_0.2': ['Tone_Position_roll', [0.6, 0.9]],

    }
    for e in np.linspace(0,1,11):
        fildict[f'bin_prate_{e}'] = ['PatternPresentation_Rate_roll', e]

    df2filter = data_df
    # df2filter.loc[:,'animal'] = df2filter.index.get_level_values(0)
    for fil in filters:
        if fil == 'none':
            fil = 'e=0'
        column = fildict[fil][0]
        if fil == '4pupil':
            _df = filt4pupil(df2filter)
        elif len(fildict[fil]) == 2:
            cond = fildict[fil][1]
            if isinstance(cond,list):
                _df = copy(df2filter[df2filter[column].isin(cond)])
            else:
                _df = copy(df2filter[df2filter[column] == cond])

        elif len(fildict[fil]) == 3:
            cond = fildict[fil][1]
            if fildict[fil][2] == '>':
                _df = copy(df2filter[df2filter[column] > cond])
            elif fildict[fil][2] == '<':
                _df = copy(df2filter[df2filter[column] < cond])
            elif fildict[fil][2] == '!=':
                _df = copy(df2filter[df2filter[column] != cond])
            elif fildict[fil][2] == 'isin':
                _df = copy(df2filter[df2filter[column].isin(fildict[fil][1])])
            elif fildict[fil][2] == 'notin':
                _df = copy(df2filter[np.invert(df2filter[column].isin(fildict[fil][1]))])
            else:
                print('incorrect format used, filter skipped')
                _df = df2filter

        else:
            print('Incorrect filter config used. Filter skipped')
            _df = df2filter
        df2filter = _df
    return df2filter


def filt4pupil(data_df):
    gd_df = filter_df(data_df, ['a3','c0'])
    viol_df = filter_df(data_df, ['a2','c0'])
    if len(data_df)>0:
        _df = data_df[(data_df['Trial_End_dt']-data_df['Pretone_end_dt']).apply(lambda t: t.total_seconds()) >= 3]
    else:
        _df = data_df
    #     # _df = viol_df[(viol_df['Trial_End_scalar']-viol_df['ToneTime_scalar']) >= 2]
    #     # _df = data_df[(data_df['Trial_End_dt']-data_df['ToneTime_dt']) >= 3]
    # return pd.concat([gd_df,_df])
    return _df
    # return data_df


def align_wrapper(datadict:dict ,filters,align_beh, duration, alignshifts=None, plotsess=False, plotlabels=None,
                  plottitle=None, xlabel=None,animal_labels=None,plotsave=False,coord=None,baseline=True,
                  pupilmetricname='rawarea_zscored',sep_cond_cntrl_flag=False, parallel=True):
    aligned_dict = dict()
    aligned_list = []
    sess_trials = {}
    sess_excluded = {}

    list_sess_data = [datadict[sess] for sess in datadict]
    _kwargs = {'baseline': True,
               'eventshift': alignshifts,
               'coord': coord,
               'pupilmetricname': pupilmetricname,
               'sep_cond_cntrl_flag': sep_cond_cntrl_flag,
               }

    if parallel:
        with multiprocessing.Pool() as pool:
            _aligned = list(tqdm(pool.imap(partial(getpatterntraces,patterntypes=filters,beh=align_beh,
                                                   dur=duration,baseline=baseline,eventshifts=alignshifts,
                                                   coord=coord,pupilmetricname=pupilmetricname,
                                                   sep_cond_cntrl_flag=sep_cond_cntrl_flag),
                                           list_sess_data), total=len(list_sess_data)))
    else:
        _aligned = []
        for sessix, sess in tqdm(enumerate(datadict.keys()), total=len(datadict)):
            _aligned.append(getpatterntraces(datadict[sess], filters, align_beh, duration,kwargs=_kwargs))

    for sessix, sess in tqdm(enumerate(datadict.keys()), total=len(datadict)):
        aligned_dict[sess] = _aligned[sessix][0]
        sess_excluded[sess] = _aligned[sessix][1]
        sess_trials[sess] = [s.shape[0] for s in aligned_dict[sess]]

    aligned_df = pd.DataFrame.from_dict(aligned_dict,orient='index')
    for ptype in aligned_df.columns:
        for i, sess in enumerate(aligned_df.index):
            if i == 0:
                _array = copy(aligned_df.loc[sess][ptype])
            else:
                _array = pd.concat([_array, copy(aligned_df.loc[sess][ptype])], axis=0)
        aligned_list.append(_array)
    return aligned_list,aligned_df,sess_trials,sess_excluded


def get_event_matrix(data_obj: object, data_dict: dict, harpbin_path: str,
                     harp_filename_pattern="*HitData*32.csv",
                     clock_offest=0, events2parse=np.arange(8)) -> dict:

    if isinstance(harpbin_path,str):
        harpbin_path = Path(harpbin_path)
    assert isinstance(harpbin_path,Path)

    if 'HitData' in harp_filename_pattern:
        event_name_prefix = 'HitData'
    elif 'SoundData' in harp_filename_pattern:
        event_name_prefix = 'SoundData'
    else:
        raise Warning('Harp file pattern not recognised')
    all_bins = np.array(list(harpbin_path.glob(harp_filename_pattern)))
    harpmatrices = {}
    plt.ioff()
    for sess in tqdm(data_dict,total=len(data_dict),desc='Parsing events from session:'):
        sess_bins = all_bins[[sess.split('_')[0] in str(e) and sess.split('_')[1] in str(e) for e in all_bins]]
        # sort out clocks
        sess_idx = sess.split('_')
        if sess_idx[1] not in data_obj.trialData.index.get_level_values('date'):
            print('Date not in trialdata, skipping')
            continue
        sess_td = data_obj.trialData.loc[(sess_idx[0],sess_idx[1])].copy()
        # sess_td.index=sess_td['Trial_Start_Time']
        if 'Bonsai_time_dt' in sess_td.columns:
            bonsai0 = sess_td['Bonsai_time_dt'][0]
        else:
            sess_td['Bonsai_time_dt'] = sess_td['Time_dt']
            bonsai0 = sess_td['Bonsai_time_dt'][0]
        bonsai0 = bonsai0-timedelta(hours=float(sess_td['Offset'][0]))
        harp0 = sess_td['Harp_time'][0]

        plot = False
        if plot:
            fig, ax = plt.subplots()
            bon_diff = sess_td['Bonsai_time_dt'].diff().apply(lambda e: e.total_seconds())
            harp_diff = sess_td['Harp_Time'].diff()
            ax.plot((bon_diff-harp_diff).to_numpy())
            ax.set_ylabel('Difference in elapsed time: Bonsai-Harp (s)')
            ax.set_xlabel('Trial Counter')
            ax.set_title(f'Diff in diff bon - harp for {sess}')
            fig.canvas.manager.set_window_title(f'sync_diffs{sess}')
            fig.savefig(rf'W:\mouse_pupillometry\figures\syncplots_221109\sync_diffs{sess}.svg',format='svg')
            plt.close(fig)
        plt.ion()

        # bonsai0 = datetime.strptime(r'221031 12:25:24.68594','%y%m%d %H:%M:%S.%f')
        # harp0 = 594023.721024

        bin_dict = {}

        assert isinstance(events2parse,(np.ndarray,pd.Series,list,tuple))
        for i in events2parse:
            bin_dict[f'{event_name_prefix}_{i}'] = []

        for bin_path in sess_bins:
            bin_df = pd.read_csv(bin_path)
            for event in bin_df['Payload'].unique():
                event_name = f'{event_name_prefix}_{event}'
                try:bin_dict[event_name].extend(bin_df[bin_df['Payload'] == event]['Timestamp'].to_list())
                except KeyError: print('harp key error')
        for event_name in bin_dict:
            if bin_dict[event_name]:
                aligned_events = align_to_clock(bin_dict[event_name],sess_td['Harp_time'],sess_td['Bonsai_time_dt'])
                bin_dict[event_name] = aligned_events

            # for e in bin_dict[event]:
            #     harp_dis_min = (sess_td['Harp_time']-e).abs().idxmin()
            #     event_times.append(sess_td['Bonsai_time_dt'][harp_dis_min] - timedelta(hours=float(sess_td['Offset'][harp_dis_min]))
            #                        + timedelta(seconds=e - sess_td['Harp_time'][harp_dis_min]))
            # bin_dict[event] = np.array([bonsai0 + timedelta(seconds=e - harp0) for e in bin_dict[event]])
            # bin_dict[event] = np.array(event_times)
        harpmatrices[sess] = bin_dict
        # print(bin_dict[0][0], data_obj.data[sess].harpmatrices[0][0])

    return harpmatrices


def align_to_clock(event_times, sync1_times, sync2_times):
    mega_matrix = np.abs(np.array(np.matrix(sync1_times)).T - event_times)
    mega_matrix_mins_idx = np.argmin(mega_matrix, axis=0)
    matrix_bonsai_times = np.array(sync2_times[mega_matrix_mins_idx], dtype='datetime64[us]')
    matrix_d_harp_times = event_times - sync1_times[mega_matrix_mins_idx]
    tdelta_us_arr = np.array((matrix_d_harp_times * 1e6).astype(int), dtype='timedelta64[us]')
    events_aligned = matrix_bonsai_times + tdelta_us_arr

    return events_aligned


def get_aligned_events(trialData, harpmatrices, eventname, harp_event, window, timeshift=0.0,
                       harp_event_name='Lick',
                       animals_omit=(None,), dates_omit=(None,), plot=True, lfilt=None,
                       byoutcome_flag=False, outcome2filt=None, extra_filts=None, plotcol=None):



    non_name_date_levels = [i for i,idx_name in enumerate(trialData.index.names) if
                            idx_name.lower() not in ['name','date']]
    sess_names = trialData.index.droplevel(non_name_date_levels).unique()
    sess_names = [(name, date) for name,date in sess_names if all((name not in animals_omit,date not in dates_omit))]
    animals,dates = np.unique(np.array(sess_names)[:,0]),np.unique(np.array(sess_names)[:,1])

    # all_sess_mats = [copy(align_nonpuil(trialData.loc[sess_name][eventname],harpmatrices['_'.join(sess_name)][harp_event], window,
    #                                   trialData.loc[sess_name]['Offset'],timeshift)) for sess_name in sess_names]
    if outcome2filt:
        if extra_filts:
            outcome2filt = outcome2filt + extra_filts

    all_sess_mats = []
    for sess_name in tqdm(sess_names,total=len(sess_names),desc='Getting aligned matrices for session'):
        td2use = trialData.loc[sess_name]
        if outcome2filt:
            td2use = filter_df(td2use, outcome2filt)
        try:
            sess_mat = copy(align_nonpuil(td2use[eventname],harpmatrices['_'.join(sess_name)][harp_event], window,
                                      td2use['Offset'],name=sess_name[0],date=sess_name[1]))
            all_sess_mats.append(sess_mat)
        except KeyError:
            print(f'sess {sess_name} not in harp matrices')
    if all_sess_mats:
        all_sess_mats = pd.concat(all_sess_mats,axis=0)

    return all_sess_mats