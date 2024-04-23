from align_functions import filter_df
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
import warnings
import scipy.stats
import pathlib
from pathlib import Path
# import cupy as cp
from joblib import Parallel, delayed


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
                    if loaded_file['name'][0] in animal_list \
                            and datetime.strptime(date_range[0], '%d/%m/%Y') <= datetime.strptime(session_date,datestr_format)\
                            <= datetime.strptime(date_range[1], '%d/%m/%Y'):
                        loaded_file.set_index(['name','date']).sort_index()
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
                                loaded_file['name'] = name_series
                                loaded_file['date'] = date_series
                                sess_part = file.split('.')[0][-1]
                                loaded_file['Session'] = np.full_like(name_series,sess_part)
                                loaded_file = loaded_file.set_index(['name','date']).sort_index()

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


def add_dt_cols(df):
    for col in df:
        if any([e in col for e in ['Time', 'Start', 'End']]):
            if all([e not in col for e in ['Wait', 'dt', 'Harp', 'Bonsai', 'Times', 'Offset']]):
                add_datetimecol(df, col)


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
             }) # utc=True
    start = time.time()
    # datetime_arr = []
    date_array = df.index.to_frame()['date']
    date_array_dt = pd.to_datetime(date_array,format='%y%m%d').to_list()   # [datetime.strptime(d,'%y%m%d') for d in date_array]
    date_array_dt_ser = pd.Series(date_array_dt)

    s = df[colname]
    s_nans = s.isnull()
    s = s.fillna('00:00:00')
    try:s_split = pd.DataFrame(s.str.split('.').to_list())
    except TypeError: print('typeerror')
    if len(s_split.columns) == 1:
        s_split[1] = np.full_like(s_split[0],'0')
    s_split.columns = ['time_hms','time_decimal']
    s_dt = pd.to_datetime(s_split['time_hms'],format='%H:%M:%S')
    try:s_dt = vec_dt_replace(s_dt,year=date_array_dt_ser.dt.year,month=date_array_dt_ser.dt.month,
                          day=date_array_dt_ser.dt.day, nanosecond=pd.to_numeric(s_split['time_decimal'].str.ljust(9,'0')))
    except:print('error')
    s_dt.iloc[s_nans] = pd.NaT
    df[f'{colname}_dt'] = s_dt.to_numpy()


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
        if plottype_flag == 'ts':
            mean = np.nanmean
            if eventnames[i] is not 'control' and 'none' not in eventnames[i]:
                event_ax.plot(traces.columns,mean(traces,axis=0), color=plotcols[i],
                              label= f'{eventnames[i]}',ls=plt_ls,lw=plt_lw)  # {traces.shape[0]} Trials
            elif eventnames[i] is not 'control' and 'none' in eventnames[i]:
                event_ax.plot(traces.columns, mean(traces, axis=0), color=plotcols[i-1],
                              label=f'{eventnames[i]}',ls='--')  # {traces.shape[0]} Trials
            else:
                control_traces = traces.iloc[:,:]
                event_ax.plot(traces.columns,mean(control_traces,axis=0), color='k',
                                          label= f'{eventnames[i]}')  # , {control_traces.shape[0]} Trials
            if eventnames[i] is not 'control' and 'none' not in eventnames[i]:
                plotvar(traces,(event_fig,event_ax),traces.columns,plotcols[i])
            else:
                plotvar(traces,(event_fig,event_ax),traces.columns,'k')
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
            plotvar(trace_sub_local, (event_fig, event_ax), traces.columns,col_str=plotcols[i])
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

    rand_npdample = [np.nanmean(data.to_numpy()[np.random.choice(data.shape[0], data.shape[0], replace=True), :],axis=0)
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
            sess_df  = filter_df(df, ['e!0', f'stage{stage}', ])
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


def iterate_fit_ellipse(xy_1d_array,fit_function='hyper',plot=None):
    if fit_function == 'hyper':
        ellipse_estimate = (fit_elipse(xy_1d_array.reshape((int(xy_1d_array.shape[0]/2),2),order='F')))
    elif fit_function in ['least_sq','weighted_reps','fns','ransac','renorm']:
        ellipse_estimate = (fit_elipse_extra(xy_1d_array.reshape((int(xy_1d_array.shape[0] / 2), 2), order='F'),
                                             fit_function=fit_function, plot=plot))
    else:
        return None, None, None, None

    return ellipse_estimate[1],ellipse_estimate[2],ellipse_estimate[0][0],ellipse_estimate[0][1]


def fit_elipse(point_array):

    xc, yc, r1, r2 = cf.hyper_fit(point_array)
    # xc, yc, r1, r2 = cf.least_squares_circle(point_array)

    return (xc,yc), r1, r1


def fit_elipse_extra(point_array, fit_function, plot=None):
    import elliptic_fitting.elliptic_fit as ell_fit

    f_0 = 100
    if fit_function == 'weighted_reps':
        A, B, C, D, E, F = ell_fit.elliptic_fitting_by_weighted_repetition(point_array[:, 0], point_array[:, 1], f_0)
    elif fit_function == 'least_sq':
        A, B, C, D, E, F = ell_fit.elliptic_fitting_by_least_squares(point_array[:, 0], point_array[:, 1], f_0)
    elif fit_function == 'renorm':
        A, B, C, D, E, F = ell_fit.elliptic_fitting_by_renormalization(point_array[:, 0], point_array[:, 1], f_0)
    elif fit_function == 'fns':
        A, B, C, D, E, F = ell_fit.elliptic_fitting_by_fns(point_array[:, 0], point_array[:, 1], f_0)
    elif fit_function == 'ransac':
        removed_x, removed_y = ell_fit.remove_outlier_by_ransac(point_array[:, 0], point_array[:, 1], f_0)
        A, B, C, D, E, F = ell_fit.elliptic_fitting_by_fns(removed_x, removed_y, f_0)

    else:
        raise Exception('Invalid fit function given')

    # w_fit_x, w_fit_y = ell_utils.solve_fitting([A,B,C,D,E,F],point_array[:,0],f_0)
    B *= 2
    D, E = [2 * f_0 * val for val in [D, E]]
    F = f_0 ** 2 * F

    r1 = -((np.sqrt(2 * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F) * (
                (A + C) - np.sqrt((A - C) ** 2 + B ** 2)))) / (B ** 2 - 4 * A * C)) #/ f_0
    r2 = -((np.sqrt(2 * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F) * (
                (A + C) + np.sqrt((A - C) ** 2 + B ** 2)))) / (B ** 2 - 4 * A * C)) #/ f_0
    xc = ((2 * C * D - B * E) / (B ** 2 - 4 * A * C))# / f_0
    yc = ((2 * A * E - B * D) / (B ** 2 - 4 * A * C)) #/ f_0
    # if plot:
    #     print(w_fit_x)
    #     plot.scatter(w_fit_x,w_fit_y,marker=7,s=5,c='magenta',label=f'{fit_function} pnts')

    # K = D ** 2 / (4 * A) + E ** 2 / (4 * C) - F
    # denominator = B ** 2 - 4 * A * C
    # xc = (2 * C * D - B * E) / denominator
    # yc = (2 * A * E - B * D) / denominator
    #
    # # K = - np.linalg.det(Q[:3, :3]) / np.linalg.det(Q[:2, :2])
    # root = math.sqrt(((A - C) ** 2 + B ** 2))
    # r1 = math.sqrt(2 * K / (A + C - root))
    # r2 = math.sqrt(2 * K / (A + C + root))
    # xc,r1 = np.mean(w_fit_x),  (np.max(w_fit_x)-np.min(w_fit_x))/2
    # yc,r2 = np.mean(w_fit_y),  (np.max(w_fit_y)-np.min(w_fit_y))/2

    return (xc, yc), r1, r2


def get_dlc_diams(df: pd.DataFrame,n_frames: int,scorer: str,):
    if n_frames == 0:
        n_frames = df.shape[0]

    diams_EW = np.full(n_frames,np.nan)
    edge_EW = np.full(n_frames,np.nan)

    body_points_names = np.unique(df.columns.get_level_values('bodyparts').to_list())
    for body_point in body_points_names:
        body_point_df = df[scorer,body_point]
        bad_body_points = df[scorer,body_point,'likelihood']<.5
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

    eyeEW_arr = np.array((df[scorer, 'eyeW'] - df[scorer, 'eyeE'])[['x', 'y']])
    eyeLR_arr = np.array((df[scorer, 'edgeE'] - df[scorer, 'edgeW'])[['x', 'y']])
    if len(eyeEW_arr) < n_frames:
        eyeEW_arr = np.pad(eyeEW_arr,[(0,n_frames-len(eyeEW_arr)),(0,0)],constant_values=np.nan)
        eyeLR_arr = np.pad(eyeLR_arr, [(0, n_frames - len(eyeLR_arr)), (0, 0)], constant_values=np.nan)
    diams_EW[:n_frames] = np.linalg.norm(eyeEW_arr,axis=1)[:n_frames]
    edge_EW[:n_frames] = np.linalg.norm(eyeLR_arr,axis=1)[:n_frames]

    return radii1_, radii2_, centersx_, centersy_, diams_EW,edge_EW


def unique_legend(plotfig:(plt.figure().figure,list,tuple),loc=1,fontsize=11,ncols=1):
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
        axis.legend(pd.Series(handle).unique(), pd.Series(label).unique(),loc=loc,fontsize=fontsize,ncol=ncols)


def in_time_window(t2eval,t,window=(-1,2)):
    in_window = all([t2eval >= t+timedelta(seconds=window[0]), t2eval <= t+timedelta(seconds=window[1])])
    return in_window

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.nanmean(a),a.std()  # a.std()
    h = se * scipy.stats.t.ppf((1 - confidence) / 2., n-1)
    return m, m-h, m+h


def manual_confidence_interval(data, confidence=0.95):
    ordered_data = sorted(data)
    m = np.nanmean(data)
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


def run_ransac(ii):
    from skimage.measure import EllipseModel, ransac
    # edges_yy, edges_xx = np.where(ii==1)
    edges_y, edges_x = np.where(ii[0::1,0::1]==1)
    # print(f'all points = {len(edges_xx)}, subset = {len(edges_x)}')
    # rand_idx = np.random.choice(edges_x.shape,12,replace=False)
    skip_n = 1
    while len(edges_x)/skip_n < 2:
        skip_n-=1
        if skip_n <= 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ransac(np.column_stack([edges_x*1,edges_y*1]),  # [0::skip_n,:],
                                     EllipseModel,max_trials=25,min_samples=10,residual_threshold=1)
    except ValueError:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    except TypeError:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    try:
        return model[0].params
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan

def iterp_grid(array):
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    # mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    GD1 = scipy.interpolate.griddata((x1, y1), newarr.ravel(),
                               (xx, yy),
                               method='nearest')
    return GD1


if __name__ == "__main__":
    pass