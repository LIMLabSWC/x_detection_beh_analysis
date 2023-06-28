from operator import itemgetter
from psychophysicsUtils import pupilDataClass
import pandas as pd
import numpy as np
import os
from copy import deepcopy as copy
import time
from datetime import datetime, timedelta
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
# import scipy
import circle_fit as cf
from math import pi
import warnings
from scipy import stats as st
from confidence_intervals import bootstrap_ci
import glob
import scipy.stats
import pathlib
from pathlib import Path
# import cupy as cp
from joblib import Parallel, delayed
import pickle
import itertools
# import jax
# import jax.numpy as jnp
# import jax.lax as lax

def merge_sessions(datadir,animal_list,filestr_cond, date_range, datestr_format='%y%m%d') -> list:
    """
    Function to merge csv files for given conditon
    :param datadir: str. starting point for datafiles
    :param animal_list: list (str) of animals to incluse
    :param filestr_cond: str specifying data filetype. Trial/SummaryData
    :param datestr_format: list len 2. start and end date in %d/%m/%Y format
    :return: concatenated df for animal_animal list over date range
    """

    file_df = []
    if date_range[1] == 'now':
        date_range[1] = datetime.strftime(datetime.now(),datestr_format)
    if date_range[0].find('/') == -1:
        _dates = []
        for d in date_range:
            _dates.append(datetime.strftime(datetime.strptime(d,'%y%m%d'),'%d/%m/%Y'))
        date_range=_dates

    for root, folder, files in os.walk(datadir):
        if filestr_cond == 'SummaryData' or filestr_cond == 'params':
            for file in files:
                if filestr_cond in file:
                    filename_parts = file.split('_')
                    animal_name = filename_parts[0]
                    session_date = filename_parts[2][:6]
                    loaded_file = pd.read_csv(os.path.join(root,file), delimiter=',').dropna()
                    if loaded_file['Name'][0] in animal_list \
                            and datetime.strptime(date_range[0], '%d/%m/%Y') <= datetime.strptime(session_date,datestr_format)\
                            <= datetime.strptime(date_range[1], '%d/%m/%Y'):
                        loaded_file.set_index(['Name','Date']).sort_index()
                        file_df.append(copy(loaded_file))

        elif filestr_cond == 'TrialData':
            for file in files:
                if filestr_cond in file:
                    filename_parts = file.split('_')
                    animal_name = filename_parts[0]
                    session_date = filename_parts[2][:6]
                    if animal_name in animal_list \
                            and datetime.strptime(date_range[0], '%d/%m/%Y') <= datetime.strptime(session_date,datestr_format)\
                            <= datetime.strptime(date_range[1], '%d/%m/%Y'):
                        try:
                            loaded_file = pd.read_csv(os.path.join(root,file), delimiter=',')
                            loaded_file = loaded_file.dropna()
                            if len(loaded_file) >0:
                                name_series = [animal_name] * loaded_file.shape[0]
                                date_series = [session_date] * loaded_file.shape[0]
                                loaded_file['Name'] = name_series
                                loaded_file['Date'] = date_series
                                sess_part = file.split('.')[0][-1]
                                loaded_file['Session'] = np.full_like(name_series,sess_part)
                                loaded_file = loaded_file.set_index(['Name','Date']).sort_index()

                                # Process harp and bonsai time column variables to avoid mismatch due to name
                                # Keep <Time> as time of day and <time> as ticks or seconds
                                if 'Harp_Time' in loaded_file.columns:
                                    loaded_file.rename(index=str,columns={'Harp_Time': 'Harp_time'},inplace=True)
                                if 'Bonsai_Time' in loaded_file.columns:
                                    loaded_file.rename(index=str,columns={'Bonsai_Time': 'Bonsai_time'},inplace=True)
                                if 'Bonsai_time' in loaded_file.columns:
                                    if str(loaded_file['Bonsai_time'].iloc[0]).isnumeric():
                                        time_conv = [datetime(1, 1, 1) + timedelta(microseconds=e / 10)
                                                     for e in loaded_file['Bonsai_time']].copy()
                                        loaded_file['Bonsai_time_dt'] = time_conv
                                    else:
                                        pass
                                        add_datetimecol(loaded_file,'Bonsai_time')
                                if 'Trial_Start_Time' in loaded_file.columns:
                                    loaded_file.rename(index=str, columns={'Trial_Start_Time': 'Trial_Start'}, inplace=True)

                                # Add offset column for daylight savings
                                daylightsavings = np.array([[200329,201025],[210328,211031],[220327,221030],[220326,221029]])  # daylight saving period
                                _dst_arr = daylightsavings - int(session_date)
                                if all(_dst_arr.prod(axis=1) > 0):
                                     offset_series = np.full_like(name_series,0.0)
                                else:
                                    offset_series = np.full_like(name_series, 1.0)

                                loaded_file['Offset'] = offset_series

                                file_df.append(loaded_file.dropna())

                        except pd.errors.EmptyDataError:
                            print('Empty data frame')

        else:
            print('File string condition is not valid')
            return [None]

    return file_df


def get_fractioncorrect(data_df, stimlen_range, animal_list, df_filters=('a3','b1')) -> list:
    performance = []
    ntrial_list = []
    for animal in animal_list:
        stim_performance = []
        animal_df = data_df.loc[animal]
        ntrial_list.append(animal_df.shape[0])
        for stim in stimlen_range:
            stim_df = animal_df[animal_df['Stim1_Duration'] == stim]
            stim_df01 = stim_df  # filter_df(stim_df, df_filters)  # remove violations and warm up trials
            n_correct = (stim_df01['Trial_Outcome'] == 1).sum()
            try: fraction_correct = float(n_correct)/stim_df01.shape[0]
            except ZeroDivisionError:
                fraction_correct = 0
            stim_performance.append(fraction_correct)
        performance.append(stim_performance)
    return performance, ntrial_list


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
    all_dates = data_df.index.to_frame()['Date'].unique()
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

    }
    for e in np.linspace(0,1,11):
        fildict[f'bin_prate_{e}'] = ['PatternPresentation_Rate_roll', e]

    df2filter = data_df
    df2filter['animal'] = df2filter.index.get_level_values(0)
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


def plot_performance(data_df, stims, animal_list, date_range, marker_colors):

    if date_range[1] == 'now':
        date_range[1] = datetime.strftime(datetime.now(),'%d/%m/%Y')

    perfomance_plot, perfomance_ax = plt.subplots(1, 1)

    fractioncorrect = get_fractioncorrect(data_df,stims,animal_list)

    for i, animal in enumerate(animal_list):
        perfomance_ax.plot(stims,fractioncorrect[0][i],label=f'{animal},{fractioncorrect[1][i]} Trials',
                           color=marker_colors[i])
    perfomance_ax.set_ylim((0,1.1))
    perfomance_ax.set_xlim((stims.min(), stims.max()))
    perfomance_ax.set_ylabel('Fraction Correct')
    perfomance_ax.set_xlabel('Stimulus Duration')
    perfomance_ax.set_xticks(stims)
    perfomance_ax.legend()
    perfomance_ax.set_title(f'Peformance for all trials {date_range[0]} to {date_range[1]}')

    return perfomance_plot,perfomance_ax,fractioncorrect


def plot_metric_v_stimdur(data_df, feature,value, animal_list, date_range, marker_colors, df_filters=None,
                          plot_title=None, ytitle=None, legend_labels = None, plottype=None):

    """
    Function to plot a metric(y axis) for each stimulus length (x axis)
    :param data_df: pd.Dataframe trial data frame
    :param feature: str column name of metric in df
    :param value: any value that row in column should be e.g. Trial_outcome == 1
    :param animal_list: str[]
    :param date_range: dd/mm/yyyy[] only need for defualt axis title
    :param marker_colors: float[] list of plot colours for each animal
    :param df_filters: str[] optional extra df filters. see filter_df()
    :param plot_title: str
    :param ytitle: str
    :param legend_labels: str[]
    :param plottype: str  default = line graph. optional 'scatter' option will perform scatter plot'
    :return:
    """

    if date_range[1] == 'now':
        date_range[1] = datetime.strftime(datetime.now(),'%d/%m/%Y')

    perfomance_plot, perfomance_ax = plt.subplots(1, 1)
    performance = []
    ntrial_list = []

    stims = filter_df(data_df, ['b1'])['Stim1_Duration'].unique()
    for animal in animal_list:
        stim_performance = []
        animal_df = data_df.loc[animal]
        if df_filters is not None:
            animal_df = filter_df(animal_df, df_filters)  # pre filter
        ntrial_list.append(animal_df.shape[0])
        for stim in stims:
            # print(stim,animal_df['Stim1_Duration'].unique())
            stim_df = animal_df[animal_df['Stim1_Duration'] == stim]
            n_metric = (stim_df[feature] == value).sum()
            # print(f'n metric {n_metric}')
            # print(f'total trials {stim_df.shape[0]}')
            try:
                fraction_metric = float(n_metric)/stim_df.shape[0]
            except ZeroDivisionError:
                fraction_metric = 0
            stim_performance.append(fraction_metric)
        performance.append(stim_performance)
    if legend_labels is not None:
        animal_list = legend_labels
    for i, animal in enumerate(animal_list):
        if plottype is None:
            perfomance_ax.plot(stims,performance[i],label=f'{animal},{ntrial_list[i]} Trials',
                           color=marker_colors[i])
        elif plottype == 'scatter':
            perfomance_ax.scatter(stims, performance[i], label=f'{animal},{ntrial_list[i]} Trials',
                               color=marker_colors[i])
        else:
            return None

    perfomance_ax.set_ylim((0,1.1))
    perfomance_ax.set_xlim((stims.min(), stims.max()))
    perfomance_ax.set_xlabel('Stimulus Duration')
    perfomance_ax.set_xticks(stims)
    perfomance_ax.legend()
    if plot_title is None:
        perfomance_ax.set_title(f'{feature} for all trials {date_range[0]} to {date_range[1]}')
    else:
        perfomance_ax.set_title(f'{plot_title}: {date_range[0]} to {date_range[1]}')
    if ytitle is None:
        perfomance_ax.set_ylabel(f'{feature} = {value}')
    else:
        perfomance_ax.set_ylabel(f'{ytitle}')

    return perfomance_plot, perfomance_ax, performance


def plot_metricrate_trialnun(data_df, feature, value,
                             filters=('b1',), plot_title=None, ytitle=None, regressionline =False):
    # init plots
    trialnum_vs_featurerate_fig, trialnum_vs_featurerate_ax = plt.subplots(1)

    # filter df
    filtered_df = filter_df(data_df, filters)
    # add trialnumber column to filtered df
    list_trialnums = []
    for session_ix in filtered_df.index.unique():
        list_trialnums.extend(list(range(filtered_df.loc[session_ix].shape[0])))
    filtered_df['Trial#'] = list_trialnums
    xy = []
    # plot trial number vs metric
    for trialnum in np.unique(filtered_df['Trial#']):
        feature_trialnum = filtered_df[filtered_df['Trial#'] == trialnum][feature] == value
        featurerate_trialnum = feature_trialnum.sum() / len(feature_trialnum)
        trialnum_vs_featurerate_ax.scatter(trialnum, featurerate_trialnum, color='lightsteelblue')
        xy.append([trialnum, featurerate_trialnum])
    trialnum_vs_featurerate_ax.set_xlabel('Trial Number')
    # format plot
    if ytitle is None:
        trialnum_vs_featurerate_ax.set_ylabel(f'{feature} = {value}')
    else:
        trialnum_vs_featurerate_ax.set_ylabel(f'{ytitle}')

    if plot_title is None:
        trialnum_vs_featurerate_ax.set_title(f'{feature}')
    else:
        trialnum_vs_featurerate_ax.set_title(f'{plot_title}')

    xy = np.array(xy)
    if regressionline:
        reg = LinearRegression().fit(xy[:, 0].reshape(-1, 1), xy[:, 1])
        regline = [x * reg.coef_ + reg.intercept_ for x in
                   np.arange(xy[:, 0].min(), xy[:, 0].max() + 1)]
        trialnum_vs_featurerate_ax.plot(xy[:,0], regline)
    return trialnum_vs_featurerate_fig, trialnum_vs_featurerate_ax, xy


def plot_frametimes(datfile):

    timestamp_df = pd.read_csv(datfile, delimiter='\t')
    reltime = []
    for i in range(timestamp_df.shape[0]):
        if i == 0:
            reltime.append(0)
        else:
            reltime.append(timestamp_df['sysClock'][i]-timestamp_df['sysClock'][i-1])
    timestamp_df['rel_time'] = reltime
    # frametime_fig, frametime_ax = copy(plt.subplots(2))
    toplot = copy(timestamp_df[timestamp_df['rel_time']<100])
    # print(toplot['rel_time'].max())
    # frametime_ax[0] = plt.hist(toplot['rel_time'], bins=toplot['rel_time'].max())
    # frametime_ax[0].set_xlim(toplot['rel_time'].max())
    # frametime_ax[1] = plt.plot(toplot['frameNum'], toplot['rel_time'])
    # frametime_ax[1].set_ylim(toplot['rel_time'].max())
    return toplot


# def plotvar(data,plot,timeseries):
#     #ci95 = 1*np.std(data,axis=0)/np.sqrt(data.shape[0])
#     #print(ci95.shape)
#     #plot[1].fill_between(timeseries, data.mean(axis=0)+ci95,data.mean(axis=0)-ci95,alpha=0.1)
#     #plot[1].fill_between(data.mean(axis=0)+ci95,data.mean(axis=0)-ci95,alpha=0.1)
#     low, high = bootstrap_ci(data)
#     plot[1].fill_between(timeseries, high, low, alpha=0.1)
#     plot[1].fill_between(high, low, alpha=0.1)

def add_datetimecol(df, colname, timefmt='%H:%M:%S.%f'):
    def vec_dt_replace(series, year=None, month=None, day=None,
                       hour=None, minute= None, second=None, microsecond=None,nanosecond=None):
        return pd.to_datetime(
            {'year': series.dt.year if year is None else year,
             'month': series.dt.month if month is None else month,
             'day': series.dt.day if day is None else day,
             'hour': series.dt.hour if hour is None else hour,
             'minute': series.dt.minute if minute is None else minute,
             'second': series.dt.second if second is None else second,
             'microsecond': series.dt.microsecond if microsecond is None else microsecond,
             'nanosecond': series.dt.nanosecond if nanosecond is None else nanosecond,
             })
    start = time.time()
    # datetime_arr = []
    date_array = df.index.to_frame()['Date']
    date_array_dt = pd.to_datetime(date_array,format='%y%m%d').to_list()   # [datetime.strptime(d,'%y%m%d') for d in date_array]
    date_array_dt_ser = pd.Series(date_array_dt)
    # for i,t in enumerate(df[colname]):
    #     if isinstance(t,str):
    #         t = t.split('+')[0]
    #         t_split = t.split('.')
    #         t_hms = t_split[0]
    #         if len(t_split) == 2:
    #             t_ms = t.split('.')[1]
    #         else:
    #             t_ms = 0
    #         try:t_hms_dt = datetime.strptime(t_hms,'%H:%M:%S')
    #         except ValueError: print(t_hms)
    #         t_ms_micros = round(float(f'0.{t_ms}'),6)*1e6
    #         t_dt = t_hms_dt.replace(microsecond=int(t_ms_micros))
    #         datetime_arr.append(t_dt)
    #
    #     else:
    #         datetime_arr.append(np.nan)
    #         # print(t,df.index[i])
    s = df[colname]
    try:s_split = pd.DataFrame(s.str.split('.').to_list(),columns=['time_hms','time_decimal'])
    except TypeError: print('typeerror')
    s_dt = pd.to_datetime(s_split['time_hms'],format='%H:%M:%S')
    try:s_dt = vec_dt_replace(s_dt,year=date_array_dt_ser.dt.year,month=date_array_dt_ser.dt.month,
                          day=date_array_dt_ser.dt.day, nanosecond=pd.to_numeric(s_split['time_decimal'].str.ljust(9,'0')))
    except:print('error')
    df[f'{colname}_dt'] = s_dt.to_numpy()
    # datetime_arr = pd.to_datetime(df[colname], format='%H:%M:%S.%f')
    # _dt_df = pd.DataFrame(list(zip(date_array_dt,datetime_arr)))
    # try:merged_date_array = [e.replace(year=d.year,month=d.month,day=d.day) for idx,(d,e) in _dt_df.iterrows()]
    # except AttributeError:
    #     print('Attribute error, add dt col')
    #     return None
    # try:df[f'{colname}_dt'] = np.array(merged_date_array)
    # except ValueError: print('Value error, add dt col ')

def align2eventScalar(df, timeseries_data, pupiltimes, pupiloutliers, beh, dur, filters=('4pupil', 'b1', 'c1'), baseline=False, eventshift=0,
                      outlierthresh=0.5, stdthresh=20, subset=None) -> pd.DataFrame:
    """

    :param df:
    :param timeseries_data:
    :param pupiltimes:
    :param pupiloutliers:
    :param beh:
    :param dur:
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
    if 'c1' in filters:
        print(f'align2.. :{filters}')
    filtered_df = filter_df(df,filters)
    dur = np.array(dur)
    # print(t_len)
    dt=pupiltimes.diff().abs().min().total_seconds()
    t_len = int(((dur[1]+dt)-dur[0])/dt)
    eventpupil_arr = np.full((filtered_df.shape[0],t_len),np.nan)  # int((np.abs(dur).sum()+rate) * 1/rate))

    eventtimez = filtered_df[beh]
    if len(eventtimez)==0:
        return pd.DataFrame(np.array([])),np.nan
    eventsess = filtered_df.index
    eventdatez = [e[1] for e in eventsess]
    eventnamez = [e[0] for e in eventsess]

    outliers = 0
    varied = 0
    # if eventshift != 0:
    dur = dur + eventshift
        # print(dur)
    for i, eventtime in enumerate(filtered_df[beh]):
        # print(dataseries)
        # print(eventtime + timedelta(seconds=float(dur[0])),eventtime + timedelta(seconds=float(dur[1])))
        # eventtime = eventtime + timedelta(hours=float(df['Offset'].iloc[i]))
        eventtime = eventtime + timedelta(hours=float(1))
        eventpupil = copy(dataseries.loc[eventtime + timedelta(0,dur[0]): eventtime + timedelta(0,dur[1])])
        eventoutliers = copy(outlierstrace.loc[eventtime + timedelta(0,dur[0]): eventtime + timedelta(0,dur[1])])
        # print((eventoutliers == 0.0).sum(),float(len(eventpupil)))
        if len(eventpupil):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if eventoutliers.sum()/float(eventpupil.shape[0]) > outlierthresh:  # change back outlierthresh
                    print(f'pupil trace for trial {i} incompatible',(eventoutliers == 1).sum())
                    outliers += 1
                    continue
                elif eventpupil.abs().max() > stdthresh*20:
                    print(f'pupil trace for trial {i} incompatible')
                    varied += 1

                else:
                    # print('diff',eventpupil.loc[eventtime - 1-eventshift]-eventpupil.loc[eventtime + 1-eventshift])
                    if baseline:
                        baseline_dur = 1.0
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            baseline_period = eventpupil.loc[eventtime - timedelta(0,baseline_dur-eventshift):
                                                       eventtime + timedelta(0,eventshift)]
                            baseline_outs = eventoutliers.loc[eventtime - timedelta(0,baseline_dur-eventshift):
                                                       eventtime + timedelta(0,eventshift)]
                            if baseline_outs.mean() > .75:  # change back!
                                outliers += 1
                                print(outliers)
                                continue
                            else:
                                # baseline_mean = np.nanmean(baseline_period[np.invert(baseline_outs)])
                                baseline_mean = np.nanmean(baseline_period)
                                eventpupil = (eventpupil - baseline_mean)

                                # epoch_linregr = sklearn.linear_model.LinearRegression().fit(np.full_like(eventpupil,baseline_mean).reshape(-1,1),eventpupil)
                                # eventpupil = (eventpupil-epoch_linregr.intercept_) /epoch_linregr.coef_[0]

                    zeropadded = np.full_like(eventpupil_arr[0],0.0)
                    try:
                        zeropadded[:len(eventpupil)] = eventpupil
                        if len(np.unique(zeropadded)) < 10:
                            print('weird', filtered_df.index.tolist()[0])
                            continue
                        eventpupil_arr[i] = zeropadded
                    except ValueError:print('bad shape')
        else:
            print(filtered_df.index[0])
            print(f'no event pupil found: eventime = {eventtime}, pupil range = {timeseries_data.index[[0,-1]]}')
            continue


    #print(f'Outlier Trials:{outliers}\n Too high varinace trials:{varied}')
    # print(eventpupil_arr.shape)
    if 'Trial_Start_dt' in filtered_df.columns:
        index=pd.MultiIndex.from_tuples(list(zip(filtered_df['Trial_Start_dt'],eventnamez,eventdatez)),names=['time','name','date'])
    else:
        index=pd.MultiIndex.from_tuples(list(zip(filtered_df['Trial_Start_dt'],eventnamez,eventdatez)),names=['time','name','date'])

    eventpupil_df = pd.DataFrame(eventpupil_arr)
    eventpupil_df.index = index
    nonans_eventpuil = eventpupil_df[~np.isnan(eventpupil_arr).any(axis=1)]

    if subset is not None:
        midpnt = nonans_eventpuil.shape[0]/2.0
        firsts = nonans_eventpuil[:subset,:]
        middles = nonans_eventpuil[int(midpnt-subset/2.0):int(midpnt+subset/2.0)]
        lasts = nonans_eventpuil[-subset:,:]
        # print(firsts.shape,middles.shape,lasts.shape)
        return [firsts,middles,lasts],outliers
    else:
        if nonans_eventpuil.size < 10:
            pass
        return nonans_eventpuil.iloc[:,:-1],outliers


def getpatterntraces(data, patterntypes,beh,dur, eventshifts=None,baseline=True,subset=None,regressed=False,
                     dev_subsetdf=None,coord=None, pupilmetricname='rawarea_zscored',sep_cond_cntrl_flag=False):

    list_eventaligned = []
    if eventshifts is None:
        eventshifts = np.zeros(len(patterntypes))
    for i, patternfilter in enumerate(patterntypes):
        if beh == 'ToneTime_dt':
            if 'e=0' in patternfilter or 'c1' in patternfilter or 'none' in patternfilter:
                beh = 'Pretone_end_dt'
                print(f'getpatterntraces none {patternfilter}')
        _pattern_tonealigned = []
        if subset is not None:
            firsts, mids, lasts = [], [], []
        if type(data) == pupilDataClass:
            if regressed:
                pupil2use = data.pupilRegressed
            elif coord == 'x':
                pupil2use = data.xc
            elif coord == 'y':
                pupil2use = data.yc
            else:
                try:pupil2use = data.pupildf[pupilmetricname]
                except AttributeError: pass

            if dev_subsetdf is None:
                td2use = data.trialData
            else: return None
            times2use = pd.Series(data.pupildf.index)
            try:outs2use = data.pupildf['isout']
            except KeyError: outs2use = data.pupildf['confisout']
        elif type(data) == dict:
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
            name = None
        if isinstance(patternfilter,str):
            patternfilter = [patternfilter]
        tone_aligned_pattern = align2eventScalar(td2use,pupil2use,times2use,
                                                 outs2use,beh,
                                                 dur,patternfilter,
                                                 outlierthresh=0.5,stdthresh=2,
                                                 eventshift=eventshifts[i],baseline=baseline,subset=subset)

        if sep_cond_cntrl_flag:  # subtract session controls from aligned epoch traces
            if 'e!0' in patternfilter:
                cntrl_patternfilter = copy(list(map(lambda x: x.replace('e!0', 'e=0'), patternfilter)))
                cntrl_patternfilter = list(map(lambda x: x.replace('tones4', 'e=0'), cntrl_patternfilter))
                cond_controls = align2eventScalar(td2use,pupil2use,times2use,
                                                 outs2use,'Pretone_end_dt',
                                                 dur,cntrl_patternfilter,
                                                 outlierthresh=0.5,stdthresh=2,
                                                 eventshift=eventshifts[i],baseline=baseline,subset=subset)
                if cond_controls[0].shape[0] > 3:
                    _tone_aligned_pattern = copy(tone_aligned_pattern[0]-cond_controls[0].mean())
                    tone_aligned_pattern = (_tone_aligned_pattern,tone_aligned_pattern[1])

        n_outliers = tone_aligned_pattern[1]
        # print(f'# Outlier trials = {n_outliers} for {patternfilter}')
        tone_aligned_pattern = tone_aligned_pattern[0]
        if subset is not None:
            firsts.append(tone_aligned_pattern[0])
            mids.append(tone_aligned_pattern[1])
            lasts.append(tone_aligned_pattern[2])

        else:
            _pattern_tonealigned.append(tone_aligned_pattern)
        if subset is not None:
            print('len subsests=',len(firsts),len(mids),len(lasts))
            _pattern_tonealigned.append(pd.concat(firsts,axis=0))
            _pattern_tonealigned.append(pd.concat(mids,axis=0))
            _pattern_tonealigned.append(pd.concat(lasts,axis=0))

        list_eventaligned.append(pd.concat(_pattern_tonealigned, axis=0))
    return list_eventaligned,n_outliers


def plot_eventaligned(eventdata_list, eventnames, dur, beh, plotax=None, pltsize=(12, 9), plotcols=None, shift=(0.0,),
                      plottype_flag='ts', binflag=False,pdelta_wind=(0.0,1.0), pltargs=(None,None), ctrl_idx=0):

    plt_ls,plt_lw, = [arg or def_arg for arg,def_arg in zip(pltargs,['-', 1])]

    if plotax is None:
        event_fig, event_ax = plt.subplots(1)
    else:
        event_fig, event_ax = plotax
    if plotcols is None:
        plotcols = [f'C{i}' for i in range(len(eventdata_list))]
    print(f'length input lists {len(eventdata_list)}')
    tseries = np.linspace(dur[0], dur[1], eventdata_list[0].shape[1])
    plt_dataset = []
    # tseries = np.floor(tseries*2.0)/2.0
    for i, traces in enumerate(eventdata_list):
        tseries_td_idx = pd.TimedeltaIndex([timedelta(seconds=e) for e in tseries])
        ts2idx = tseries_td_idx.to_series(index=tseries)
        # traces.columns = tseries_td_idx
        try:traces.columns = tseries
        except:print('tseries problem')
        if binflag:
            pass
            # binsize = 90
            # traces = traces.rolling(binsize,axis=1).mean().iloc[:,binsize - 1::binsize]
        if plottype_flag =='ts':
            mean = np.mean
            if eventnames[i] is not 'control' and 'none' not in eventnames[i]:
                event_ax.plot(traces.columns,mean(traces,axis=0), color=plotcols[i],
                              label= f'{eventnames[i]}, {traces.shape[0]} Trials',ls=plt_ls,lw=plt_lw)
            elif eventnames[i] is not 'control' and 'none' in eventnames[i]:
                event_ax.plot(traces.columns, mean(traces, axis=0), color=plotcols[i-1],
                              label=f'{eventnames[i]}, {traces.shape[0]} Trials',ls='--')
            else:
                control_traces = traces.iloc[:,:]
                event_ax.plot(traces.columns,mean(control_traces,axis=0), color='k',
                                          label= f'{eventnames[i]}, {control_traces.shape[0]} Trials')
            plotvar(traces,(event_fig,event_ax),traces.columns,plotcols[i])
            event_ax.legend(loc=1)
        elif plottype_flag == 'boxplot':  # add proper flags for flexibility
            # max_pdelta_series = traces.loc[:,ts2idx[0.5]:ts2idx[2.0]].max(axis=1)
            max_pdelta_series = traces.loc[:,pdelta_wind[0]:pdelta_wind[1]].max(axis=1)
            # event_ax.bar(i,max_pdelta_series.mean(),label=f'{eventnames[i]}, {traces.shape[0]} Trials')
            plt_dataset.append(max_pdelta_series)
            if len(plt_dataset) == len(eventdata_list):
                event_ax.boxplot(plt_dataset,range(len(plt_dataset)),showmeans=False,bootstrap=1000)
                # event_ax.violinplot(plt_dataset,range(len(plt_dataset)),showmeans=False)
                event_ax.set_xticks(np.arange(len(plt_dataset))+1)
                event_ax.set_xticklabels(eventnames)
        elif plottype_flag == 'pdelta_trend':
            if i is not ctrl_idx:
                trace_sub_ctrl  = traces-eventdata_list[ctrl_idx].mean(axis=0)
                print(f'{trace_sub_ctrl.shape}')
                max_pdelta_series = trace_sub_ctrl.loc[:,pdelta_wind[0]:pdelta_wind[1]].max(axis=1)

                plotcols = [f'C{x}' for x in range(len(max_pdelta_series.index.get_level_values('name').unique()))]
                lines = ["--", "-", ":", "-."]
                for ai, animal in enumerate(max_pdelta_series.index.get_level_values('name').unique()):
                    data2plot = max_pdelta_series.loc[:,animal,:].head(15)
                    event_ax.plot(np.arange(data2plot.shape[0]),data2plot,ls=lines[i],
                                  label=f'{animal}: {eventnames[i]}',color=plotcols[ai])
                    event_ax.set_xlabel('Trial Number')
        elif plottype_flag == 'local_baseline':
            trace_sub_local = np.zeros_like(traces)
            for trial_i, (idx, trial) in enumerate(traces.iterrows()):
                if idx > eventdata_list[0].loc[:,[idx[1]],[idx[2]]].index[0]:
                    _d_idx = eventdata_list[0].loc[:,[idx[1]],[idx[2]]].index.get_level_values('time') - idx[0]
                    try:prev_idx = eventdata_list[0].loc[:,[idx[1]],[idx[2]]].index[(np.argwhere(_d_idx == _d_idx[_d_idx < timedelta(0)].max())[0])]
                    except:
                        print('oh no')
                        continue
                    try:trace_sub_local[trial_i, :] = trial.to_numpy() - eventdata_list[0].loc[prev_idx].to_numpy()
                    except:print('oh no')

            if eventnames[i] is not 'control' and 'none' not in eventnames[i]:
                event_ax.plot(traces.columns,np.nanmean(trace_sub_local,axis=0), color=plotcols[i],
                              label= f'{eventnames[i]}, {trace_sub_local.shape[0]} Trials')
            elif eventnames[i] is not 'control' and 'none' in eventnames[i]:
                event_ax.plot(traces.columns, np.nanmean(trace_sub_local, axis=0), color=plotcols[i-1],
                              label=f'{eventnames[i]}, {trace_sub_local.shape[0]} Trials',ls='--')
            else:
                control_traces = trace_sub_local
                event_ax.plot(traces.columns,np.nanmean(control_traces,axis=0), color='k',
                                          label= f'{eventnames[i]}, {control_traces.shape[0]} Trials')
            plotvar(trace_sub_local, (event_fig, event_ax), traces.columns)
            event_ax.legend(loc=1)

        else:
            print('No valid plot type given')
            return None



    if 'ToneTime' in beh and not eventdata_list[0].empty:
        try:plt_date = eventdata_list[0].index.get_level_values('date')[0]
        except KeyError or IndexError: print('boo')
        if 230313 >= int(plt_date) <= 230307:
            event_ax.axvspan(0, 0+0.125, edgecolor='k', facecolor='k', alpha=0.1)
            event_ax.axvspan(0.25, 0.25+0.125, edgecolor='k', facecolor='k', alpha=0.1)
            event_ax.axvspan(0.5, 0.50+0.125, edgecolor='k', facecolor='k', alpha=0.1)
            event_ax.axvspan(0.75,0.75+0.125, edgecolor='k', facecolor='k', alpha=0.1)
        else:
            event_ax.axvspan(0, 0 + 0.125, edgecolor='k', facecolor='k', alpha=0.1)
            event_ax.axvspan(0.3, 0.3 + 0.125, edgecolor='k', facecolor='k', alpha=0.1)
            event_ax.axvspan(0.6, 0.6 + 0.125, edgecolor='k', facecolor='k', alpha=0.1)
            event_ax.axvspan(0.9, 0.9 + 0.125, edgecolor='k', facecolor='k', alpha=0.1)
        event_ax.axvline(0, c='k', alpha=0.5)


    if 'Violation' in beh:
        s = shift[0]
        rect1 = matplotlib.patches.Rectangle(((0-s), -10), 0.125, 20, linewidth=0, edgecolor='k', facecolor='k', alpha=0.1)
        rect2 = matplotlib.patches.Rectangle(((0.25-s), -10), 0.125, 20, linewidth=0, edgecolor='k', facecolor='k',
                                             alpha=0.1)
        rect3 = matplotlib.patches.Rectangle(((0.5-s), -10), 0.125, 20, linewidth=0, edgecolor='k', facecolor='k',
                                             alpha=0.1)
        rect4 = matplotlib.patches.Rectangle(((0.75-s), -10), 0.125, 20, linewidth=0, edgecolor='k', facecolor='k',
                                             alpha=0.1)
        event_ax.axvline(0, c='k', alpha=0.5)
        event_ax.add_patch(rect1)
        event_ax.add_patch(rect2)
        event_ax.add_patch(rect3)
        event_ax.add_patch(rect4)

    if plotax is None:
        event_ax.set_xlabel('Time from event (s)',fontsize=14)
        event_ax.set_title(f'Pupil size aligned to {beh}',fontsize=14)

    return event_fig,event_ax


def plotvar(data,plot,timeseries=None,col_str=None,n_samples=500):

    rand_npdample = [data.to_numpy()[np.random.choice(data.shape[0], data.shape[0], replace=True), :].mean(axis=0)
                     for i in range(n_samples)]
    # except: pass
    rand_npdample = np.array(rand_npdample)
    ci = np.apply_along_axis(mean_confidence_interval, axis=0, arr=rand_npdample)
    # ci = np.apply_along_axis(manual_confidence_interval, axis=0, arr=rand_npdample)
    # plot[1].plot(ci[0, :])
    if col_str:
        plot[1].fill_between(data.columns, ci[1, :], ci[2, :], alpha=0.1,facecolor=col_str)
    else:
        plot[1].fill_between(data.columns, ci[1, :], ci[2, :], alpha=0.1)


# def ts_permutation_test(ts_matrices, n_permutations, conf_interval,cnt_idx=0,  pltax=(None,None),ts_window=None,):
#     """
#
#     :param ts_matrices: list of ts data where cols are times points and rows are epochs
#     :param n_permutations:
#     :param conf_interval:
#     :return:
#     """
#     t0 = time.time()
#     observed_diff = [np.abs(matrix.mean(axis=0) - ts_matrices[cnt_idx].mean(axis=0)) for matrix in ts_matrices]
#     print(f'perm subtract time ={time.time()-t0}')
#     # observed_diff.pop(cnt_idx)
#     epochs_per_matrix = [matrix.shape[0] for matrix in ts_matrices]
#     epochs_start_ix = np.pad(np.cumsum(epochs_per_matrix),[1,0])
#     # mega_matrix = np.vstack(ts_matrices).copy()
#
#     rng = np.random.default_rng()
#     # simulated_diffs = np.zeros((n_permutations,len(ts_matrices),ts_matrices[cnt_idx].shape[1]))
#     t0 = time.time()
#     for cond_idx, cond_matrix in enumerate(ts_matrices):
#         mega_matrix = np.vstack([cond_matrix,ts_matrices[cnt_idx]]).copy()
#         shuffled_indices = [np.random.choice(mega_matrix.shape[0], mega_matrix.shape[0], replace=False) for n in
#                             range(n_permutations)]
#         shuffled_timeseries_data = [mega_matrix[idxs, :] for idxs in shuffled_indices]
#         simulated_diffs = np.array([shuffled[:cond_matrix.shape[0]].mean(axis=0) -
#                                     shuffled[cond_matrix.shape[0]:].mean(axis=0)
#                                     for shuffled in shuffled_timeseries_data])
#         shuffled_subsets = []
#         # for shuffle_i in range(n_permutations):
#         #     mega_matrix_shuffled = rng.permutation(mega_matrix,axis=0)
#         #     simulated_diffs[shuffle_i,cond_idx,:] = mega_matrix_shuffled[:cond_matrix.shape[0]].mean(axis=0) - \
#         #                                             mega_matrix_shuffled[cond_matrix.shape[0]:].mean(axis=0)
#             # shuffled_subsets.append(mega_matrix_shuffled.mean(axis=0))
#
#         # _sim_diffs = [np.abs(matrix - shuffled_subsets[cnt_idx]) for matrix in shuffled_subsets]
#         # if cond_idx is not cnt_idx:
#         #     simulated_diffs[:,cond_idx,:] = _sim_diffs
#     # simulated_diffs = np.delete(simulated_diffs,cnt_idx,axis=1)
#     print(f'shuffle time = {time.time()-t0}')
#     simulated_greater_observed = np.greater(simulated_diffs,observed_diff)
#     portion_above_observed = simulated_greater_observed.mean(axis=0)
#     sig_time_points = portion_above_observed < ((1 - conf_interval)/2)
#
#     if pltax is not None and ts_window is not None:
#         plt_ts = np.linspace(ts_window[0],ts_window[1],sig_time_points.shape[1])
#         ylim0 = pltax[1].get_ylim()[0]
#         trace_is = list(range(len(ts_matrices)))
#         trace_is.pop(cnt_idx)
#         print(trace_is)
#         for cond_i, cond_ts in enumerate(sig_time_points[trace_is]):
#             sig_x_series = plt_ts[np.where(cond_ts==True)]
#             sig_y_series = np.full_like(sig_x_series,ylim0*(1+0.1*cond_i))
#             pltax[1].scatter(sig_x_series,sig_y_series, marker='o', c=f'C{trace_is[cond_i]}', s=2)
#
#     return sig_time_points

def ts_permutation_test(ts_matrices, n_permutations, conf_interval, cnt_idx=0, pltax=(None, None), ts_window=None, n_jobs=1):


    # def calculate_permutation(cond_idx):
    #     cond_matrix = cp.asarray(ts_matrices[cond_idx])
    #     perm_diff = cp.zeros(ts_matrices[cnt_idx].shape[1])
    #     rng = cp.random.default_rng()
    #     for shuffle_i in range(n_permutations):
    #         shuffled_indices = rng.choice(cond_matrix.shape[0], cond_matrix.shape[0], replace=False)
    #         shuffled_cond_matrix = cond_matrix[shuffled_indices]
    #         perm_diff += cp.abs(shuffled_cond_matrix.mean(axis=0) - cp.asarray(ts_matrices[cnt_idx]).mean(axis=0))
    #     return perm_diff
    #
    def calculate_permutation(cond_idx):
        cond_matrix = np.asarray(ts_matrices[cond_idx])
        perm_diff = np.zeros(ts_matrices[cnt_idx].shape[1])
        rng = np.random.default_rng()
        for shuffle_i in range(n_permutations):
            shuffled_indices = rng.choice(cond_matrix.shape[0], cond_matrix.shape[0], replace=False)
            shuffled_cond_matrix = cond_matrix[shuffled_indices]
            perm_diff += np.abs(shuffled_cond_matrix.mean(axis=0) - np.asarray(ts_matrices[cnt_idx]).mean(axis=0))
        return perm_diff
    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_permutation)(cond_idx) for cond_idx in range(len(ts_matrices))
    )

    sig_time_points = np.zeros((len(ts_matrices), ts_matrices[cnt_idx].shape[1]), dtype=bool)
    for cond_idx, perm_diff in enumerate(parallel_results):
        perm_diff /= n_permutations
        sig_time_points[cond_idx] = perm_diff > ((1 - conf_interval) / 2)

    if pltax is not None and ts_window is not None:
        plt_ts = np.linspace(ts_window[0], ts_window[1], sig_time_points.shape[1])
        ylim0 = pltax[1].get_ylim()[0]
        trace_is = list(range(len(ts_matrices)))
        trace_is.pop(cnt_idx)
        print(trace_is)
        for cond_i, cond_ts in enumerate(sig_time_points[trace_is]):
            sig_x_series = plt_ts[np.where(cond_ts == True)]
            sig_y_series = np.full_like(sig_x_series, ylim0 * (1 + 0.1 * cond_i))
            pltax[1].scatter(sig_x_series, sig_y_series, marker='x', c=f'C{trace_is[cond_i]}', s=2)

    return sig_time_points


def ts_two_tailed_ht(ts_matrices, conf_interval, cnt_idx=0, pltax=(None, None), ts_window=None,):
    def two_tailed_ht(sample1, sample2):
        sample1, sample2 = np.array(sample1), np.array(sample2)
        t_stat, p_val = scipy.stats.ttest_ind(sample1,sample2, equal_var=False)
        f,a = plt.subplots()
        if p_val<0.01:
            a.hist(sample1.T,density=False)
            a.hist(sample2.T,density=False)
        return p_val

    pval_ts_matrix = np.zeros((len(ts_matrices)-1, ts_matrices[0].shape[1]))
    for ti, ts_matrix in enumerate(ts_matrices):
        if ti != cnt_idx:
            pval_ts = [two_tailed_ht(time_point_sample1, time_point_sample2)
                       for time_point_sample1, time_point_sample2 in
                       zip(ts_matrices[cnt_idx].to_numpy().T, ts_matrix.to_numpy().T)]
            pval_ts_matrix[ti,:] = pval_ts
    pval_ts_matrix = pval_ts_matrix<(1-conf_interval)/2
    if pltax is not None and ts_window is not None:
        plt_ts = np.linspace(ts_window[0], ts_window[1], pval_ts_matrix.shape[1])
        ylim0 = pltax[1].get_ylim()[0]
        trace_is = list(range(len(ts_matrices)))
        trace_is.pop(cnt_idx)
        for cond_i, cond_ts in enumerate(pval_ts_matrix[trace_is]):
            sig_x_series = plt_ts[np.where(cond_ts == True)]
            sig_y_series = np.full_like(sig_x_series, ylim0 * (1 + 0.1 * cond_i))
            pltax[1].scatter(sig_x_series, sig_y_series, marker='o', c=f'C{trace_is[cond_i]}', s=2)



def align_wrapper(datadict,filters,align_beh, duration, alignshifts=None, plotsess=False, plotlabels=None,
                  plottitle=None, xlabel=None,animal_labels=None,plotsave=False,coord=None,baseline=True,
                  pupilmetricname='rawarea_zscored',sep_cond_cntrl_flag=False):
    aligned_dict = dict()
    aligned_list = []
    sess_trials = {}
    sess_excluded = {}
    if plotsess:
        if all([plotlabels,plottitle,xlabel,animal_labels]):
            pass
        else:
            print('No plot labels or plot title given for plot. Aborting') #ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            return None
    for sessix, sess in enumerate(datadict.keys()):
        _aligned = getpatterntraces(datadict[sess],filters,align_beh,duration,
                                              baseline=baseline, eventshifts=alignshifts,coord=coord,
                                              pupilmetricname=pupilmetricname,sep_cond_cntrl_flag=sep_cond_cntrl_flag)

        aligned_dict[sess] = _aligned[0]
        sess_excluded[sess] = _aligned[1]
        sess_trials[sess] = [s.shape[0] for s in aligned_dict[sess]]
        if plotsess:
            sess_plot = plot_eventaligned(aligned_dict[sess],plotlabels,duration,plottitle)
            sess_plot[1].set_title(f'{sess}: Pupil response to Pattern',size=8)
            sess_plot[1].set_xlabel(xlabel,size=6)
            sess_plot[1].set_ylabel('Normalised pupil diameter',size=6)
            sess_plot[1].set_ylim((-1,2))
            sess_plot[0].set_size_inches(4,3, forward=True)
            if plotsave:
                sess_plot[0].savefig(f'{animal_labels[sessix]}_{align_beh}.png', bbox_inches='tight')
    aligned_df = pd.DataFrame.from_dict(aligned_dict,orient='index')
    for ptype in aligned_df.columns:
        for i, sess in enumerate(aligned_df.index):
            if i == 0:
                _array = copy(aligned_df.loc[sess][ptype])
            else:
                _array = pd.concat([_array, copy(aligned_df.loc[sess][ptype])], axis=0)
        aligned_list.append(_array)
    return aligned_list,aligned_df,sess_trials,sess_excluded

def findfiles(startdir,filetype,datadict,animals=None,dates=None):
    """
    Adds file paths strings to given dictionary. Will walk through subfolders and add fullpaths of matching files
    :param startdir: str dir to start os.walk()
    :param filetype: str to match
    :param datadict: dict for adding full path to
    :param animals: str[] optional list animals
    :param dates: str[] optional list dates
    :return: nothing
    """
    for root, folder, files in os.walk(startdir):
        for file in files:
            if filetype in file:
                splitstr = file.split('_')
                _animal = splitstr[0]
                _date = splitstr[1]
                if dates is None:
                    if _date not in datadict[_animal].keys():
                        datadict[_animal][_date] = dict()
                    datadict[_animal][_date][f'{filetype}file'] = os.path.join(root,file)
                elif dates is not None and animals is not None:
                    if _date in dates and _animal in animals:
                        if _date not in datadict[_animal].keys():
                            datadict[_animal][_date] = dict()
                        datadict[_animal][_date][f'{filetype}file'] = os.path.join(root, file)


def get_diff_traces(arr, basetrace, window, metric='max'):
    diff_arr = arr - basetrace
    diff_arr_window = diff_arr[window[0]:window[1]]
    max_diffs_arr = np.apply_along_axis(lambda r: np.where(r == r.max())[0][0], 1, diff_arr_window)

    if metric == 'max':
        return diff_arr_window.max(), max_diffs_arr
    elif metric == 'integral':
        return diff_arr_window.abs().sum(), max_diffs_arr
    else:
        print('Invalid metric used')
        return None, None


def add_date_ticks(plotax, date_list):
    dates_unique = sorted(pd.Series(date_list).unique())
    plotax.set_xticks(np.arange(len(dates_unique)))
    plotax.set_xticklabels(list(dates_unique), rotation=40, ha='center', size=8)


def format_timestr(timestr_series,date=None) -> (pd.Series, pd.Series):
    """
    function to add decimal to time strings. also returns datetime series
    :param timestr_series:
    :return:
    """
    s=timestr_series
    s_split = pd.DataFrame(s.str.split('.').to_list())
    s_dt = pd.to_datetime(s_split[0],format='%H:%M:%S').replace(microsecond=pd.to_numeric(s_split[1]))
    datetime_arr = []
    for t in s:
        if isinstance(t, str):
            t_split = t.split('.')
            t_hms = t_split[0]
            if len(t_split) == 2:
                t_ms = t.split('.')[1]
            else:
                t_ms = 0
            t_hms_dt = datetime.strptime(t_hms, '%H:%M:%S')
            t_ms_micros = round(float(f'0.{t_ms}'), 6) * 1e6
            t_dt = t_hms_dt.replace(microsecond=int(t_ms_micros))
            if date:
                t_dt = t_dt.replace(date[0],date[1],date[2])
            datetime_arr.append(t_dt)

        else:
            datetime_arr.append(np.nan)
    return datetime_arr

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """



    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        # w=eval('np.'+window+'(window_len)')
        pass
    w=np.hanning(window_len)

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)]


def butter_highpass(cutoff, fs, order=5,filtype='high'):
    nyq = 0.5 * fs
    if filtype == 'band':
        if isinstance(cutoff, (list,tuple)):
            normal_cutoff = [e/nyq for e in cutoff]
        else:
            print('List of filter needed for bandpass. Not filtering')
            return None
    else:
        normal_cutoff = cutoff / nyq
    # print(f'cutoffs:{normal_cutoff}')
    b, a = scipy.signal.butter(order, normal_cutoff, btype=filtype, analog =False)
    return b, a


def butter_filter(data, cutoff, fs, order=3, filtype='high'):
    b, a = butter_highpass(cutoff, fs, order=order,filtype=filtype)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def find_good_sessions(df,stage,n_critereon=100,skip=0):
    sessions = df.index.unique()
    gd_sessions = []
    if skip:
        gd_sessions_names = [sess[0] for sess in sessions]
        gd_sessions_dates = [sess[1] for sess in sessions]
    else:
        for sess_ix in sessions:
            sess_df  = filter_df(df,['e!0',f'stage{stage}',])
            if sess_df is not None:
                if sess_df.shape[0] >= n_critereon:
                    gd_sessions.append(sess_ix)
        gd_sessions_names = [sess[0] for sess in gd_sessions]
        gd_sessions_dates = [sess[1] for sess in gd_sessions]

    return gd_sessions, gd_sessions_names, gd_sessions_dates


def pair_dir2sess(topdir,animals, year_limit=2022, subject='mouse', dirstyle=r'Y_m_d\it',
                  spec_dates=None,spec_dates_op='='):
    paired_dirs = {}
    animals = [e.upper() for e in animals]
    for folder in os.listdir(topdir):
        if os.path.isdir(os.path.join(topdir,folder)):
            abs_folder_path = os.path.join(topdir,folder)
            folder_split = folder.split('_')
            name_vals = []
            roots = []
            root = topdir
            date_in_corstr = None
            if dirstyle == r'Y_m_d\it':
                if folder_split[0].isnumeric():
                    if int(folder_split[0]) >= year_limit:
                        for root,folder, file in os.walk(abs_folder_path):
                            if root.split(os.sep)[-1].isnumeric():
                                directory = root.split(os.sep)
                                date = directory[-2]
                                try:date_in_dt = datetime.strptime(date, '%Y_%m_%d')
                                except ValueError:continue
                                date_in_corstr = datetime.strftime(date_in_dt, '%y%m%d')
                                # first make sure I am doing this within the recordings directory
                                try:content = pd.read_csv(os.path.join(root, 'user_info.csv'), index_col=0)
                                except pd.errors.ParserError or FileNotFoundError:
                                    continue
                                name_vals.append(content['value']['name'])
                                roots.append(root)

            elif dirstyle == 'N_D_it':
                if '_' in folder:
                    filestr = folder_split
                    date_in_dt = datetime.strptime(filestr[1], '%y%m%d')
                    date_in_corstr = datetime.strftime(date_in_dt, '%y%m%d')
                    if spec_dates:
                        spec_dates_int = [int(e) for e in list(spec_dates)]
                        if spec_dates_op == '=':
                            if date_in_corstr not in spec_dates:
                                continue
                        elif spec_dates_op == '<=':
                            if int(date_in_corstr) > spec_dates_int[0]:
                                continue
                        elif spec_dates_op == '>=':
                            if int(date_in_corstr) < spec_dates_int[0]:
                                continue
                        elif spec_dates_op == '<=<':
                            if int(date_in_corstr) > spec_dates_int[1] or int(date_in_corstr) < spec_dates_int[0]:
                                continue
                    name_vals.append(filestr[0])
                    roots.append(os.path.join(topdir,folder))
            else:
                print('invalid dir style given')
                continue
            for name_val,root in zip(name_vals,roots):
                if type(name_val) == str:
                    name_val = name_val.upper()
                    if name_val[:2].upper() == 'D0':
                        name_val = f'DO{name_val[2:]}'
                    if name_val.upper().lstrip() in animals:
                        animal = name_val.upper().lstrip()
                        if subject == 'human':
                            animal=animal.capitalize()
                        sess = f'{animal}_{date_in_corstr}'
                        if paired_dirs.get(sess,None) is None:
                            paired_dirs[sess] = root
                        else:
                            old_item = paired_dirs[sess]
                            if isinstance(old_item,list):
                                paired_dirs[sess] = old_item.append(root)
                            else:
                                paired_dirs[sess] = [old_item,root]

    return paired_dirs


def iterate_fit_ellipse(xy_1d_array):

    ellipse_estimate = (fit_elipse(xy_1d_array.reshape((int(xy_1d_array.shape[0]/2),2),order='F')))
    return ellipse_estimate[1],ellipse_estimate[2],ellipse_estimate[0][0],ellipse_estimate[0][0]


def fit_elipse(point_array):
    xc, yc, r1, r2 = cf.hyper_fit(point_array)

    return (xc,yc), r1, r2


def get_dlc_diams(df: pd.DataFrame,n_frames: int,scorer: str,):
    if n_frames == 0:
        n_frames = df.shape[0]
    # body_points = np.array(df)
    # radii1_ = np.full(n_frames,np.nan)
    # radii2_ = np.full(n_frames,np.nan)
    # centersx_ = np.full(n_frames,np.nan)
    # centersy_ = np.full(n_frames,np.nan)
    diams_EW = np.full(n_frames,np.nan)
    edge_EW = np.full(n_frames,np.nan)

    body_points_names = np.unique(df.columns.get_level_values('bodyparts').to_list())
    for body_point in body_points_names:
        body_point_df = df[scorer,body_point]
        bad_body_points = df[scorer,body_point,'likelihood']<.3
        df.loc[bad_body_points, (scorer,body_point,'x')] = np.nan
        df.loc[bad_body_points, (scorer,body_point,'y')] = np.nan
    pupil_points_only_df = df.drop(['edgeE','edgeW'],axis=1,level=1)
    bad_frames = pupil_points_only_df.isna().sum(axis=1) > 5*2
    df.loc[bad_frames] = np.nan

    xy_df = pupil_points_only_df.loc[:, pd.IndexSlice[:, :, ('x','y')]].values
    # y_df = pupil_points_only_df.loc[:, pd.IndexSlice[:, :, 'y']].values
    xy_arr = np.array(xy_df)
    ellispe_estimates = np.array([iterate_fit_ellipse(r) for r in xy_arr])
    radii1_, radii2_, centersx_, centersy_ = [array.flatten() for array in np.hsplit(ellispe_estimates,4)]
    # xy_df = pd.concat([x_columns,y_columns],axis=1)

    # bodypoints =  body_points.reshape((n_frames/8,8,3))
    # for i,row in enumerate(bodypoints[:n_frames,:]):
    #     reshaped = row[0:24].reshape([8,3])
    #     goodpoints = reshaped[reshaped[:,2]>.3].astype(float)
    #     if goodpoints.shape[0] < 3:
    #         radii1_[i] = np.nan
    #         radii2_[i] = np.nan
    #         centersx_[i] = np.nan
    #         centersy_[i] = np.nan
    #
    #     else:
    #         frame_elipse = fit_elipse(goodpoints[:,[0,1]])
    #         radii1_[i] = frame_elipse[1]
    #         radii2_[i] = frame_elipse[2]
    #         centersx_[i] = frame_elipse[0][0]
    #         centersy_[i] = frame_elipse[0][1]

    eyeEW_arr = np.array((df[scorer, 'eyeW'] - df[scorer, 'eyeE'])[['x', 'y']])
    eyeLR_arr = np.array((df[scorer, 'edgeE'] - df[scorer, 'edgeW'])[['x', 'y']])
    if len(eyeEW_arr) < n_frames:
        eyeEW_arr = np.pad(eyeEW_arr,[(0,n_frames-len(eyeEW_arr)),(0,0)],constant_values=np.nan)
        eyeLR_arr = np.pad(eyeLR_arr, [(0, n_frames - len(eyeLR_arr)), (0, 0)], constant_values=np.nan)
    diams_EW[:n_frames] = np.linalg.norm(eyeEW_arr,axis=1)[:n_frames]
    edge_EW[:n_frames] = np.linalg.norm(eyeLR_arr,axis=1)[:n_frames]

    return radii1_, radii2_, centersx_, centersy_, diams_EW,edge_EW


def get_event_matrix(data_obj: object, data_dict: dict, harpbin_path: str, clock_offest=0) -> dict:
    all_bins = np.array(glob.glob(f'{os.path.join(harpbin_path,"*HitData*32.csv" )}'))
    harpmatrices = {}
    plt.ioff()
    for sess in data_dict:
        sess_bins = all_bins[[sess.split('_')[0] in e and sess.split('_')[1] in e for e in all_bins]]
        bin_dict = {}
        # sort out clocks
        sess_idx = sess.split('_')
        if sess_idx[1] not in data_obj.trialData.index.get_level_values('Date'):
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
        for i in range(8):
            bin_dict[i] = []

        for bin_path in sess_bins:
            bin_df = pd.read_csv(bin_path)
            for event in bin_df['Payload'].unique():
                try:bin_dict[event].extend(bin_df[bin_df['Payload'] == event]['Timestamp'].to_list().copy())
                except KeyError: print('harp key error')
        for event in bin_dict:
            event_times =[]
            for e in bin_dict[event]:
                harp_dis_min = (sess_td['Harp_time']-e).abs().idxmin()
                event_times.append(sess_td['Bonsai_time_dt'][harp_dis_min] - timedelta(hours=float(sess_td['Offset'][harp_dis_min]))
                                   + timedelta(seconds=e - sess_td['Harp_time'][harp_dis_min]))
            # bin_dict[event] = np.array([bonsai0 + timedelta(seconds=e - harp0) for e in bin_dict[event]])
            bin_dict[event] = np.array(event_times)
        harpmatrices[sess] = bin_dict
        # print(bin_dict[0][0], data_obj.data[sess].harpmatrices[0][0])

    return harpmatrices


def align_nonpuil(eventzz: pd.Series, timeseries: np.ndarray, window: list,
                  offsetseries: pd.Series, eventshift=0.0) -> dict:
    window = np.array(window)
    event_dict = {}
    eventzz = eventzz - offsetseries.apply(lambda e: timedelta(hours=float(e))) + timedelta(seconds=eventshift)
    eventzz_windows = [[e + timedelta(seconds=window[0]),e + timedelta(seconds=window[1])] for e in eventzz]

    for i, (eventtime,eventzz_window) in enumerate(zip(eventzz,eventzz_windows)):
        min_time = eventtime + timedelta(seconds=window[0])
        max_time = eventtime + timedelta(seconds=window[1])
        try:idx_bool = (timeseries<= max_time) * (timeseries>=min_time)
        except TypeError: print('skipping')
        # print(f'{eventtime}, {i}')
        event_dict[eventtime] = timeseries[idx_bool]-np.full_like(timeseries[idx_bool],eventtime)
        event_dict[eventtime] = np.array([e.total_seconds() for e in event_dict[eventtime]])
    assert len(event_dict) == eventzz.shape[0]
    return event_dict


def unique_legend(plotfig:(plt.figure().figure,list,tuple),loc=1,fontsize=11):
    if isinstance(plotfig,(tuple,list)):
        if isinstance(plotfig[1],np.ndarray):
            plotaxes2use = plotfig[1].flatten()
        elif isinstance(plotfig[1], dict):
            plotaxes2use = plotfig[1].values()
        else:
            print('wrong figure used, returning none')
            plotaxes2use = None
    elif isinstance(plotfig,np.ndarray):
        plotaxes2use = plotfig.flatten()
    elif isinstance(plotfig[1],dict):
        plotaxes2use = plotfig[1].values()
    else:
        plotaxes2use = None
        print('wrong figure used, returning none')
    for axis in plotaxes2use:
        handle, label = axis.get_legend_handles_labels()
        axis.legend(pd.Series(handle).unique(), pd.Series(label).unique(),loc=loc,fontsize=fontsize)


def in_time_window(t2eval,t,window=(-1,2)):
    in_window = all([t2eval >= t+timedelta(seconds=window[0]), t2eval <= t+timedelta(seconds=window[1])])
    return in_window

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a),a.std()  # a.std()
    h = se * scipy.stats.t.ppf((1 - confidence) / 2., n-1)
    return m, m-h, m+h


def manual_confidence_interval(data, confidence=0.95):
    ordered_data = sorted(data)
    m = np.mean(data)
    m_low = ordered_data[int(data.shape[0]*(1-confidence/2))]
    m_high = ordered_data[int(data.shape[0]*confidence/2)]

    return m, m_low, m_high


def unique_file_path(path, suffix='_a'):
    if not isinstance(path, (pathlib.WindowsPath, pathlib.PosixPath)):
        path = Path(path)
    if suffix:
        path = path.with_stem(f'{path.stem}{suffix}')
    while path.exists():
        new_stem = f'{path.stem[:-1]}{chr(ord(path.stem[-1])+1)}'
        path = path.with_stem(new_stem)
    return path


def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return True

class PupilEventConditions:

    def __init__(self):
        fam_filts = {
            'p_rate': [[['plow'], ['p0.5'], ['phigh'], ['none']],
                       ['0.1', '0.5', '0.9', 'control']],
            'p_rate_ctrl': [[['plow'], ['plow', 'none'], ['p0.5'], ['p0.5', 'none'], ['phigh'], ['phigh', 'none']],
                            ['0.1', '0.1 cntrl', '0.5', '0.5 cntrl', '0.9', '0.9 cntrl', 'control']],
            'p_onset': [[['dearly', 'p0.5'], ['dlate', 'p0.5'], ['dmid', 'p0.5']],
                        ['Early Pattern', 'Late Pattern', 'Middle Presentation']],
            # 'p0.5_block': [[['0.5_0','p0.5'], ['0.5_1','p0.5'], ['0.5_2','p0.5'], ['none']],
            #                ['0.5 Block (0.0)', '0.5 Block 1 (0.1)', '0.5 Block 2 (0.9)', 'Control']],
            'alt_rand': [[['s0', 'p0.5'], ['s1', 'p0.5'], ['none', 'p0.5']],
                         ['0.5 Random', '0.5 Alternating', 'Control']],
            'alt_rand_ctrl': [
                [['s0', 'p0.5'], ['s0', 'none'], ['s1', 'p0.5'], ['s1', 'p0.5', 'none'], ['none', 'p0.5']],
                ['0.5 Random', '0.5 Random ctrl', '0.5 Alternating', '0.5 Alternating ctrl', 'Control']],
            # 'ntones': [[['p0.5','tones4'],['p0.5','tones3'],['p0.5','tones2'],['p0.5','tones1']],['ABCD', 'ABC','AB','A']],
            # 'pat_nonpatt': [[['e!0'],['e=0']],['Pattern Sequence Trials','No Pattern Sequence Trials']],
            'pat_nonpatt_2X': [[['e!0'], ['none']], ['Pattern Sequence Trials', 'No Pattern Sequence Trials']],
            'p_rate_fm': [[['plow'], ['pmed'], ['phigh'], ['ppost'], ['none']],
                          ['0.1', '0.5', '0.9', '0.6', 'control']],

        }

        normdev_filts = {
            'normdev': [[['d0'], ['d!0']], ['Normal', 'Deviant']],
            'normdev_newnorms': [[['d0'], ['d!0'], ['d-1']], ['Normal', 'Deviant', 'New Normal']],
            'pat_nonpatt_2X': [[['e!0'], ['none']], ['Pattern Sequence Trials', 'No Pattern Sequence Trials']],
        }
        self.all_filts = {**fam_filts, **normdev_filts}

    def get_condition_dict(self,dataclass,condition_keys,stages,pmetric2use='dlc_radii_a_zscored',
                           do_baseline=True,extra_filts=()):
        from pupil_analysis_func import batch_analysis

        def get_mean_subtracted_traces(dataclass):
            for key in ['p_rate_ctrl', 'alt_rand_ctrl']:
                if key not in dataclass.aligned.keys():
                    continue
                dataclass.aligned[f'{key}_sub'] = copy(dataclass.aligned[key])
                for ti, tone_df in enumerate(dataclass.aligned[key][2]):
                    if (ti % 2 == 0 or ti == 0) and ti < len(dataclass.aligned[key][2]) - 1:
                        print(ti)
                        control_tone_df = dataclass.aligned[key][2][ti + 1].copy()
                        for sess_idx in tone_df.index.droplevel('time').unique():
                            sess_ctrl_mean = control_tone_df.loc[:, [sess_idx[0]], [sess_idx[1]]].mean(axis=0)
                            tone_df.loc[:, sess_idx[0], sess_idx[1]] = tone_df.loc[:, [sess_idx[0]],
                                                                       [sess_idx[1]]] - sess_ctrl_mean
                        # run.aligned[f'{key}_sub'][2][ti] = copy(tone_df)-run.aligned[key][2][ti+1].mean(axis=0)
                        dataclass.aligned[f'{key}_sub'][2][ti] = copy(tone_df)
                for idx in [1, 2]:
                    if idx < (len(dataclass.aligned[key][2])):
                        dataclass.aligned[f'{key}_sub'][2].pop(idx)


        align_pnts = ['ToneTime', 'Reward', 'Gap_Time']

        if not hasattr(dataclass,'allinged'):
            dataclass.aligned = {}

        for cond_key in condition_keys:
            cond_filts = self.all_filts.get(cond_key,None)
            if cond_filts == None:
                print(f'{cond_key} not in {self.all_filts.keys()}. Skipping')
                continue
            if cond_key in dataclass.aligned.keys():
                print(f'{cond_key} exists. Skipping')
                continue
            if '2X' in cond_key:
                cond_align_point = align_pnts[2]
            else:
                cond_align_point = align_pnts[0]
            batch_analysis(dataclass, dataclass.aligned, stages, f'{cond_align_point}_dt', [[0, f'{cond_align_point}'], ],
                           cond_filts[0], cond_filts[1], pmetric=pmetric2use,
                           filter_df=True, plot=True, sep_cond_cntrl_flag=False, cond_name=cond_key,
                           use4pupil=True, baseline=do_baseline, pdr=False, extra_filts=extra_filts)
        get_mean_subtracted_traces(dataclass)
