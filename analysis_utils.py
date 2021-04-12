import pandas as pd
import numpy as np
import os
from copy import copy
import time
from datetime import datetime, timedelta
from matplotlib import pyplot as plt


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
            stim_df01 = filter_df(stim_df, df_filters)  # remove violations and warm up trials
            n_correct = (stim_df01['Trial_Outcome'] == 1).sum()
            print(f'n correct {n_correct}')
            print(f'total trials {stim_df01.shape[0]}')
            fraction_correct = float(n_correct)/stim_df01.shape[0]
            stim_performance.append(fraction_correct)
        performance.append(stim_performance)
    return performance, ntrial_list


def filter_df(data_df, filters):

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
        'd1': ['Pattern_Type', 0, '!='],

    }

    df2filter = data_df
    for fil in filters:
        column = fildict[fil][0]
        cond = fildict[fil][1]
        if len(fildict[fil]) == 2:
            _df = copy(df2filter[df2filter[column] == cond])
        elif len(fildict[fil]) == 3:
            if fildict[fil][2] == '>':
                _df = copy(df2filter[df2filter[column] > cond])
            elif fildict[fil][2] == '<':
                _df = copy(df2filter[df2filter[column] < cond])
            elif fildict[fil][2] == '!=':
                _df = copy(df2filter[df2filter[column] != cond])
            else:
                print('incorrect format used, filter skipped')
                _df = df2filter
        else:
            print('Incorrect filter config used. Filter skipped')
            _df = df2filter
        df2filter = _df
    return df2filter


def plot_performance(data_df, stims, animal_list, date_range, marker_colors):
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


def plot_metric_v_stimdur(data_df, stims, feature,value, animal_list, date_range, marker_colors, df_filters=None):

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
            stim_df = animal_df[animal_df['Stim1_Duration'] == stim]
            n_metric = (stim_df[feature] == value).sum()
            print(f'n metric {n_metric}')
            print(f'total trials {stim_df.shape[0]}')
            fraction_metric = float(n_metric)/stim_df.shape[0]
            stim_performance.append(fraction_metric)
        performance.append(stim_performance)
    for i, animal in enumerate(animal_list):
        perfomance_ax.plot(stims,performance[i],label=f'{animal},{ntrial_list[i]} Trials',
                           color=marker_colors[i])
    perfomance_ax.set_ylim((0,1.1))
    perfomance_ax.set_xlim((stims.min(), stims.max()))
    perfomance_ax.set_ylabel(f'{feature} = {value}')
    perfomance_ax.set_xlabel('Stimulus Duration')
    perfomance_ax.set_xticks(stims)
    perfomance_ax.legend()
    perfomance_ax.set_title(f'{feature} for all trials {date_range[0]} to {date_range[1]}')

    return perfomance_plot, perfomance_ax, performance
