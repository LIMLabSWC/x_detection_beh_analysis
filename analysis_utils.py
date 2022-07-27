from psychophysicsUtils import pupilDataClass
import pandas as pd
import numpy as np
import os
from copy import copy
import time
from datetime import datetime, timedelta
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy
import circle_fit as cf
from math import pi
import warnings

def merge_sessions(datadir,animal_list,filestr_cond, date_range, datestr_format='%y%m%d') -> list:
    """
    Function to merge csv files for given conditon
    :param datadir: str. starting point for datafiles
    :param animal_list: list (str) of animals to incluse
    :param filestr_cond: str specifying data filetype. Trial/SummaryData
    :param datestr_format: list len 2. start and end date in %d/%m/%Y format
    :return: concatenated df for aniimal_animal list over date range
    """

    file_df = []
    if date_range[1] == 'now':
        date_range[1] = datetime.strftime(datetime.now(),'%d/%m/%Y')
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
                    loaded_file = pd.read_csv(os.path.join(root,file), delimiter=',')
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
                            if len(loaded_file) >0:
                                name_series = [animal_name] * loaded_file.shape[0]
                                date_series = [session_date] * loaded_file.shape[0]
                                loaded_file['Name'] = name_series
                                loaded_file['Date'] = date_series
                                sess_part = file.split('.')[0][-1]
                                loaded_file['Session'] = np.full_like(name_series,sess_part)
                                loaded_file = loaded_file.set_index(['Name','Date']).sort_index()
                                file_df.append(loaded_file.dropna())
                        except pd.errors.EmptyDataError:
                            print('Empty data frame')

        else:
            print('File string condition is not valid')
            return None

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

    # get all time=00:00:00
    all_dates = data_df.index.to_frame()['Date'].unique()
    all_dates_t0 = [datetime.strptime(e,'%y%m%d') for e in all_dates]

    fildict = {
        'a0': ['Trial_Outcome', 0],
        'a1': ['Trial_Outcome', 1],
        'a2': ['Trial_Outcome', -1],
        'a3': ['Trial_Outcome', -1, '!='],
        'b0': ['WarmUp', True],
        'b1': ['WarmUp', False],
        'c0': ['Tone_Position', 0],
        'c1': ['Tone_Position', 1],
        'd0': ['Pattern_Type', 0],
        'd!0': ['Pattern_Type', 0, '>'],
        'd-1': ['Pattern_Type', -1],
        'd1': ['Pattern_Type', 1],
        'd2': ['Pattern_Type', 2],
        'd3': ['Pattern_Type', 3],
        'd4': ['Pattern_Type', [1,3]],
        'e!0': ['ToneTime_dt',all_dates_t0,'notin'],
        'e=0': ['ToneTime_dt',all_dates_t0,'isin'],
        '4pupil': ['na'],
        'devrep':['PatternID',dev_repeat,'isin'],
        'devord': ['PatternID', dev_nonorder, 'isin'],
        'devassc': ['PatternID', dev_assc, 'isin'],
        'devdesc': ['PatternID', dev_desc, 'isin'],
        'normtrain': ['PatternID', normtrain, 'isin'],
        'normtest':['PatternID', normtest, 'isin'],
        'pnone': ['PatternPresentation_Rate',1.0],
        'plow': ['PatternPresentation_Rate',[0.8,0.9]],
        'pmed': ['PatternPresentation_Rate',0.6],
        'phigh': ['PatternPresentation_Rate',0.1],
        'ppost': ['PatternPresentation_Rate',0.4],
        'tones4': ['N_TonesPlayed',4],
        'tones3': ['N_TonesPlayed',3],
        'tones2': ['N_TonesPlayed',2],
        'tones1': ['N_TonesPlayed',1],
        's-1': ['Session_Block',-1],
        's0': ['Session_Block',0],
        's1': ['Session_Block',1],
        's01': ['Session_Block',[0,1]],
        's2': ['Session_Block',2],
        's3': ['Session_Block',3],
        'sess_a': ['Session', 'a'],
        'sess_b': ['Session', 'b'],
        'stage0': ['Stage', 0],
        'stage1': ['Stage', 1],
        'stage2': ['Stage', 2],
        'stage3': ['Stage',3],
        'stage4': ['Stage', 4],
        'stage5': ['Stage', 5]
    }

    df2filter = data_df
    for fil in filters:
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
    # _df = viol_df[(viol_df['Trial_End_dt']-viol_df['ToneTime_dt']).apply(lambda t: t.total_seconds()) >= 2]
    # _df = viol_df[(viol_df['Trial_End_scalar']-viol_df['ToneTime_scalar']) >= 2]
    _df = data_df[(data_df['Trial_End_scalar']-data_df['ToneTime_scalar']) >= 3]
    # return pd.concat([gd_df,_df])
    return _df


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
#     ci95 = 1*np.std(data,axis=0)/np.sqrt(data.shape[0])
#     print(ci95.shape)
#     plot[1].fill_between(timeseries, data.mean(axis=0)+ci95,data.mean(axis=0)-ci95,alpha=0.1)
#     # plot[1].fill_between(data.mean(axis=0)+ci95,data.mean(axis=0)-ci95,alpha=0.1)

def add_datetimecol(df, colname, timefmt='%H:%M:%S.%f'):

    datetime_arr = []
    date_array = df.index.to_frame()['Date']
    date_array_dt = [datetime.strptime(d,'%y%m%d') for d in date_array]
    for i,t in enumerate(df[colname]):
        if isinstance(t,str):
            if len(t) > 15:
                datetime_arr.append((datetime.strptime(t[:-1], timefmt)))
            else:
                try:datetime_arr.append((datetime.strptime(t,'%H:%M:%S')))
                except ValueError or TypeError: print(t,df.index[i])
        else:
            datetime_arr.append(np.nan)
            print(t,df.index[i])
    merged_date_array = [e.replace(year=d.year,month=d.month,day=d.day) for d,e in zip(date_array_dt,datetime_arr)]
    try:df[f'{colname}_dt'] = np.array(merged_date_array)
    except:ValueError

def align2eventScalar(df,pupilsize,pupiltimes, pupiloutliers,beh, dur, filters=('4pupil','b1','c1'), baseline=False,eventshift=0,
                      outlierthresh=0.9,stdthresh=3,subset=None) -> pd.DataFrame:

    pupiltrace = pd.Series(pupilsize,index=pupiltimes)
    outlierstrace = pd.Series(pupiloutliers,index=pupiltimes)
    filtered_df = filter_df(df,filters)
    dur = np.array(dur)
    # print(t_len)
    dt=pupiltimes.diff().abs().mean().total_seconds()
    t_len = int(((dur[1]+dt)-dur[0])/dt)
    eventpupil_arr = np.full((filtered_df.shape[0],t_len),np.nan)  # int((np.abs(dur).sum()+rate) * 1/rate))

    eventtimez = filtered_df[beh]
    if len(eventtimez)==0:
        return pd.DataFrame(np.array([]))
    eventsess = filtered_df.index
    eventdatez = [e[1] for e in eventsess]
    eventnamez = [e[0] for e in eventsess]

    outliers = 0
    varied = 0
    # if eventshift != 0:
    dur = dur + eventshift
        # print(dur)
    for i, eventtime in enumerate(filtered_df[beh]):
        # print(pupiltrace)
        # print(eventtime + timedelta(seconds=float(dur[0])),eventtime + timedelta(seconds=float(dur[1])))

        eventpupil = copy(pupiltrace.loc[eventtime + timedelta(0,dur[0]): eventtime + timedelta(0,dur[1])])
        eventoutliers = copy(outlierstrace.loc[eventtime + timedelta(0,dur[0]): eventtime + timedelta(0,dur[1])])
        # print((eventoutliers == 0.0).sum(),float(len(eventpupil)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if (eventoutliers == 1.0).sum()/float(len(eventpupil)) > outlierthresh:
                # print(f'pupil trace for trial {i} incompatible',(eventoutliers == 1).sum())
                outliers += 1
            elif eventpupil.abs().max() > stdthresh:
                # print(f'pupil trace for trial {i} incompatible')
                varied += 1

            else:
                # print('diff',eventpupil.loc[eventtime - 1-eventshift]-eventpupil.loc[eventtime + 1-eventshift])
                if baseline:
                    baseline_dur = 1.0
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        baseline_mean = np.nanmean(eventpupil.loc[eventtime - timedelta(0,baseline_dur-eventshift):
                                                   eventtime + timedelta(0,eventshift)])
                        eventpupil = eventpupil - baseline_mean
                    # eventpupil = (eventpupil-eventpupil.mean())/eventpupil.std()
            if len(eventpupil)>0:
                zeropadded = np.full_like(eventpupil_arr[0],0.0)
                try: zeropadded[:len(eventpupil)] = eventpupil
                except ValueError:print('bad shape')
                eventpupil_arr[i] = zeropadded
    # print(f'Outlier Trials:{outliers}\n Too high varinace trials:{varied}')
    # print(eventpupil_arr.shape)
    index=pd.MultiIndex.from_tuples(list(zip(eventtimez,eventnamez,eventdatez)),names=['time','name','date'])
    eventpupil_df = pd.DataFrame(eventpupil_arr)
    eventpupil_df.index = index
    nonans_eventpuil = eventpupil_df[~np.isnan(eventpupil_arr).any(axis=1)]

    if subset is not None:
        midpnt = nonans_eventpuil.shape[0]/2.0
        firsts = nonans_eventpuil[:subset,:]
        middles = nonans_eventpuil[int(midpnt-subset/2.0):int(midpnt+subset/2.0)]
        lasts = nonans_eventpuil[-subset:,:]
        # print(firsts.shape,middles.shape,lasts.shape)
        return [firsts,middles,lasts]
    else:
        if nonans_eventpuil.size < 10:
            pass
        return nonans_eventpuil.iloc[:,:-1]


def getpatterntraces(data, patterntypes,beh,dur, eventshifts=None,baseline=True,subset=None,regressed=False,
                     dev_subsetdf=None,coord=None, pupilmetricname='rawarea_zscored'):

    list_eventaligned = []
    if eventshifts is None:
        eventshifts = np.zeros(len(patterntypes))
    for i, patternfilter in enumerate(patterntypes):
        if 'e=0' in patternfilter:
            beh = 'Pretone_end_dt'
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
                # pupil2use = data.pupildf['diameter_3d_ffilt']
                pupil2use = data.pupildf[pupilmetricname]

            if dev_subsetdf is None:
                td2use = data.trialData
            else: return None
            times2use = pd.Series(data.pupildf.index)
            outs2use = data.pupildf['confisout']
        elif type(data) == dict:
            for name in data.keys():
                if regressed:
                    pupil2use = data[name].pupilRegressed
                else:
                    # pupil2use = data[name].pupildf['diameter_3d_ffilt']
                    pupil2use = data[name].pupildf[pupilmetricname]

                td2use = data[name].trialData
                times2use = pd.Series(data[name].pupildf.index)
                if 'dlc' in pupilmetricname:
                    outs2use = data[name].pupildf['dlc_isout']
                else:
                    outs2use = data[name].pupildf['confisout']
        else:
            print('Incorrect data structure')
            name = None
        if isinstance(patternfilter,str):
            patternfilter = [patternfilter]
        try:tone_aligned_pattern = align2eventScalar(td2use,pupil2use,times2use,
                                                 outs2use,beh,
                                                 dur,patternfilter,
                                                 outlierthresh=0.5,stdthresh=4,
                                                 eventshift=eventshifts[i],baseline=baseline,subset=subset)
        except IndexError: print('broken')
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
    return list_eventaligned


def plot_eventaligned(eventdata_list, eventnames, dur, beh, plotax=None, pltsize=(12, 9), plotcols=None):
    if plotax is None:
        event_fig, event_ax = plt.subplots(1)
    else:
        event_fig, event_ax = plotax
    if plotcols is None:
        plotcols = [f'C{i}' for i in range(len(eventdata_list))]
    print(f'length input lists {len(eventdata_list)}')
    for i, traces in enumerate(eventdata_list):
        tseries = np.linspace(dur[0], dur[1], eventdata_list[i].shape[1])
        if eventnames[i] is not 'control':
            event_ax.plot(tseries,np.nanmean(traces,axis=0), color=plotcols[i],
                          label= f'{eventnames[i]}, {traces.shape[0]} Trials')
        else:
            control_traces = traces.iloc[:,:]
            event_ax.plot(tseries,np.nanmean(control_traces,axis=0), color='k',
                                      label= f'{eventnames[i]}, {control_traces.shape[0]} Trials')

    # event_ax.axvline(0, c='k', linestyle='--')
    # event_ax.axvline(1, c='k', linestyle='--')

        plotvar(traces,(event_fig,event_ax),tseries)

    if 'ToneTime' in beh:
        rect1 = matplotlib.patches.Rectangle((0, -10), 0.125, 20, linewidth=0, edgecolor='k', facecolor='k', alpha=0.1)
        rect2 = matplotlib.patches.Rectangle((0.25, -10), 0.125, 20, linewidth=0, edgecolor='k', facecolor='k',
                                             alpha=0.1)
        rect3 = matplotlib.patches.Rectangle((0.5, -10), 0.125, 20, linewidth=0, edgecolor='k', facecolor='k',
                                             alpha=0.1)
        rect4 = matplotlib.patches.Rectangle((0.75, -10), 0.125, 20, linewidth=0, edgecolor='k', facecolor='k',
                                             alpha=0.1)
        event_ax.axvline(0, c='k', alpha=0.5)
        event_ax.add_patch(rect1)
        event_ax.add_patch(rect2)
        event_ax.add_patch(rect3)
        event_ax.add_patch(rect4)
    if plotax is None:
        event_ax.set_xlabel('Time from event (s)')
        event_ax.set_title(f'Pupil size aligned to {beh}')
    event_ax.legend()

    return event_fig,event_ax


def plotvar(data,plot,timeseries):
    ci95 = 1*np.std(data,axis=0)/np.sqrt(data.shape[0])
    print(ci95.shape)
    plot[1].fill_between(timeseries, np.nanmean(data,axis=0)+ci95,np.nanmean(data, axis=0)-ci95,alpha=0.1)


def align_wrapper(datadict,filters,align_beh, duration, alignshifts=None, plotsess=False, plotlabels=None,
                  plottitle=None, xlabel=None,animal_labels=None,plotsave=False,coord=None,
                  pupilmetricname='rawarea_zscored'):
    aligned_dict = dict()
    aligned_list = []
    sess_trials = {}
    if plotsess:
        if all([plotlabels,plottitle,xlabel,animal_labels]):
            pass
        else:
            print('No plot labels or plot title given for plot. Aborting') #ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            return None
    for sessix, sess in enumerate(datadict.keys()):
        aligned_dict[sess] = getpatterntraces(datadict[sess],filters,align_beh,duration,
                                              baseline=True, eventshifts=alignshifts,coord=coord,
                                              pupilmetricname=pupilmetricname)
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
    return aligned_list,aligned_df,sess_trials

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


def format_timestr(timestr_series) -> (pd.Series, pd.Series):
    """
    function to add decimal to time strings. also returns datetime series
    :param timestr_series:
    :return:
    """
    s=timestr_series
    before = time.time()
    try:formatted=s.where(s.apply(lambda e: len(e.split(':')[-1]))>3,s.apply(lambda e: f'{e}.0'))
    except AttributeError: print('bad time')
    dt_series = [datetime.strptime(e,'%H:%M:%S.%f') for e in formatted]
    return formatted, dt_series

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


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype = "high", analog = False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def find_good_sessions(df,stage,n_critereon=100):
    sessions = df.index.unique()
    gd_sessions = []
    for sess_ix in sessions:
        sess_df  = filter_df(df,['e!0',f'stage{stage}',])
        if sess_df is not None:
            if sess_df.shape[0] >= n_critereon:
                gd_sessions.append(sess_ix)
    gd_sessions_names = [sess[0] for sess in gd_sessions]
    gd_sessions_dates = [sess[1] for sess in gd_sessions]

    return gd_sessions, gd_sessions_names, gd_sessions_dates


def pair_dir2sess(topdir,animals, year_limit=2022,subject='mouse'):
    paired_dirs = {}
    animals = [e.upper() for e in animals]
    for folder in os.listdir(topdir):
        if os.path.isdir(os.path.join(topdir,folder)):
            abs_folder_path = os.path.join(topdir,folder)
            folder_split = folder.split('_')
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
                                name_val = content['value']['name']
                                if type(name_val) == str:
                                    name_val = name_val.upper()
                                    if name_val[:2].upper() == 'D0':
                                        name_val = f'DO{name_val[2:]}'
                                    if name_val.upper().lstrip() in animals:
                                        # animal = content.value[0][-2:]#do46
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


def fit_elipse(point_array):
    xc, yc, r1, r2 = cf.hyper_fit(point_array)

    return (xc,yc), r1, r2


def get_dlc_diams(df,n_frames):
    bodypoints = np.array(df)
    radii1_ = np.full(n_frames,np.nan)
    radii2_ = np.full(n_frames,np.nan)
    centersx_ = np.full(n_frames,np.nan)
    centersy_ = np.full(n_frames,np.nan)

    for i,row in enumerate(bodypoints[:n_frames,:]):
        reshaped = row[0:24].reshape([8,3])
        goodpoints = reshaped[reshaped[:,2]>.3].astype(float)
        if goodpoints.shape[0] < 3:
            radii1_[i] = np.nan
            radii2_[i] = np.nan
            centersx_[i] = np.nan
            centersy_[i] = np.nan

        else:
            frame_elipse = fit_elipse(goodpoints[:,[0,1]])
            radii1_[i] = frame_elipse[1]
            radii2_[i] = frame_elipse[2]
            centersx_[i] = frame_elipse[0][0]
            centersy_[i] = frame_elipse[0][1]
    return radii1_, radii2_, centersx_, centersy_
