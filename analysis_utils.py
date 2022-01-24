import pandas as pd
import numpy as np
import os
from copy import copy
import time
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


def merge_sessions(datadir,animal_list,filestr_cond, date_range, datestr_format='%y%m%d'):
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
    for root, folder, files in os.walk(datadir):
        if filestr_cond == 'SummaryData' or filestr_cond == 'params':
            for file in files:
                if file.find(filestr_cond) != -1:
                    session_date = file[-11:-5]
                    loaded_file = pd.read_csv(os.path.join(root,file), delimiter=',')
                    if loaded_file['Name'][0] in animal_list \
                            and datetime.strptime(date_range[0], '%d/%m/%Y') <= datetime.strptime(session_date,datestr_format)\
                            <= datetime.strptime(date_range[1], '%d/%m/%Y'):
                        loaded_file.set_index(['Name','Date']).sort_index()
                        file_df.append(copy(loaded_file))

        elif filestr_cond == 'TrialData':
            for file in files:
                if file.find(filestr_cond) != -1:
                    animal_name = file[0:4]
                    session_date = file[-11:-5]
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
                                loaded_file = loaded_file.set_index(['Name','Date']).sort_index()
                                file_df.append(loaded_file)
                        except pd.errors.EmptyDataError:
                            print('Empty data frame')

        else:
            print('File string condition is not valid')
            return None

    return file_df


def get_fractioncorrect(data_df, stimlen_range, animal_list, df_filters=('a3','c1')):
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


def filter_df(data_df, filters):

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
        'd!0': ['Pattern_Type', 0, '!='],
        'd1': ['Pattern_Type', 1],
        'd2': ['Pattern_Type', 2],
        'd3': ['Pattern_Type', 3],
        'e!0': ['ToneTime_dt',datetime.strptime('00:00:00','%H:%M:%S'),'!='],
        'e=0': ['ToneTime_dt',datetime.strptime('00:00:00','%H:%M:%S')],
        '4pupil': ['na'],
        'devrep':['PatternID',dev_repeat,'isin'],
        'devord': ['PatternID', dev_nonorder, 'isin'],
        'devassc': ['PatternID', dev_assc, 'isin'],
        'devdesc': ['PatternID', dev_desc, 'isin']

    }

    df2filter = data_df
    for fil in filters:
        column = fildict[fil][0]
        if fil == '4pupil':
            _df = filt4pupil(df2filter)
        elif len(fildict[fil]) == 2:
            cond = fildict[fil][1]
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
    _df = viol_df[(viol_df['Trial_End_scalar']-viol_df['ToneTime_scalar']) >= 2]
    return pd.concat([gd_df,viol_df])


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


def plot_metric_v_stimdur(data_df, stims, feature,value, animal_list, date_range, marker_colors, df_filters=None,
                          plot_title=None, ytitle=None, legend_labels = None, plottype=None):

    if date_range[1] == 'now':
        date_range[1] = datetime.strftime(datetime.now(),'%d/%m/%Y')

    perfomance_plot, perfomance_ax = plt.subplots(1, 1)
    performance = []
    ntrial_list = []

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
            print('something weird happened')
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


def plotvar(data,plot,timeseries):
    ci95 = 1*np.std(data,axis=0)/np.sqrt(data.shape[0])
    print(ci95.shape)
    plot[1].fill_between(timeseries, data.mean(axis=0)+ci95,data.mean(axis=0)-ci95,alpha=0.1)

def add_datetimecol(df, colname, timefmt='%H:%M:%S.%f'):

    datetime_arr = []
    for t in df[colname]:
        if len(t) > 8:
            datetime_arr.append((datetime.strptime(t[:-1], timefmt)))
        else:
            datetime_arr.append((datetime.strptime(t,'%H:%M:%S')))
    df[f'{colname}_dt'] = np.array(datetime_arr)

