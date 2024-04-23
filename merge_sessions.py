import pandas as pd
import numpy as np
import os
from copy import copy
import time
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import plot_sessions

def merge_sessions(datadir,animal_list,filestr_cond, datestr_format='%y%m%d'):
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
                    filename_parts = file.split('_')
                    animal_name = filename_parts[0]
                    session_date = filename_parts[2]
                    loaded_file = pd.read_csv(os.path.join(root,file), delimiter=',')
                    if loaded_file['name'][0] in animal_list \
                            and datetime.strptime(date_range[0], '%d/%m/%Y') <= datetime.strptime(session_date,datestr_format)\
                            <= datetime.strptime(date_range[1], '%d/%m/%Y'):
                        loaded_file.set_index(['name','Date']).sort_index()
                        file_df.append(copy(loaded_file))

        elif filestr_cond == 'TrialData':
            for file in files:
                if file.find(filestr_cond) != -1:
                    filename_parts = file.split('_')
                    animal_name = filename_parts[0]
                    session_date = filename_parts[2]
                    if animal_name in animal_list \
                            and datetime.strptime(date_range[0], '%d/%m/%Y') <= datetime.strptime(session_date,datestr_format)\
                            <= datetime.strptime(date_range[1], '%d/%m/%Y'):
                        try:
                            loaded_file = pd.read_csv(os.path.join(root,file), delimiter=',')
                            if len(loaded_file) >0:
                                name_series = [animal_name] * loaded_file.shape[0]
                                date_series = [session_date] * loaded_file.shape[0]
                                loaded_file['name'] = name_series
                                loaded_file['date'] = date_series
                                loaded_file = loaded_file.set_index(['name','date']).sort_index()
                                file_df.append(loaded_file)
                        except pd.errors.EmptyDataError:
                            print('Empty data frame')

        else:
            print('File string condition is not valid')
            return None

    return file_df


# def create_subset_df(data_dict, fields, date_period=('01/01/1000','now'), rats=None):
#     """
#
#     :param data_dict:
#     :param fields:
#     :param date_period:
#     :param rats:
#     :return:
#     """
#     # print('starting')
#     t0 = time.time()
#     if rats is None or len(rats) < 1 or not isinstance(rats[0], str):
#         rats = list(data_dict.keys())
#     # print(type(data_dict),date_period,rats)
#     dates_dict = subset_dates(data_dict,date_period,rats)
#     #     # list_rat_dfs = []
#     rsf = []
#     for r in rats:  # r is rat name i.e keys for dictionary
#         for s in data_dict[r]:
#             sess_date = extract_val(s['SavingSection_SaveTime']).split()
#             if sess_date[0] in dates_dict[r][1]:
#                 use = 1
#             else:
#                 use = 0
#             if use == 1:
#                 val = []
#                 for f in fields:
#                     val.append(extract_val(s[f]))
#                 rsf.append([r]+val)
#
#     df = pd.DataFrame(rsf)
#     t1 = time.time()
#     # print('Time taken ',t1-t0)
#     return df
def get_fractioncorrect(data_df, stimlen_range,animal_list):

    performance = []
    ntrial_list = []
    for animal in animal_list:
        stim_perfomance = []
        animal_df = data_df.loc[animal]
        ntrial_list.append(animal_df.shape[0])
        for stim in range(stimlen_range):
            stim_df = animal_df[animal_df['Stim1_Duration'] == stim]['Trial_Outcome']
            stim_correct = stim_df == 1
            stim_perfomance.append(stim_correct.mean())
        performance.append(stim_perfomance)
    return performance, ntrial_list


def filter_df(data_df, filters, fildict):
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


animals = [
            # 'DO27',
            # 'DO28',
            'DO39',
]




datadir = r'C:\bonsai\data\Dammy'
date_range =['24/09/2021', '24/09/2021']


# params = merge_sessions(datadir,animals,'params')
# summary_data = merge_sessions(datadir,animals,'params')
# print(f'len animals={len(animals)}, len params = {len(params)}')

# current_params = [animal.tail(1) for animal in params]
# current_params = pd.concat(current_params)
#
# summary_data = pd.concat(summary_data)
marker_colors = ['b','r','c','m','y','g']

trial_data = merge_sessions(datadir,animals,'TrialData')
trial_data = pd.concat(trial_data, sort=False, axis=0)

fractioncorrect = get_fractioncorrect(trial_data, [2,7], trial_data.keys().unique()[0])

def plot_performance(animal_list,stimlen_range):
    perfomance_plot, perfomance_ax = plt.subplots(1, 1)

    for i, animal in enumerate(animal_list):
        perfomance_ax.plot(np.arange(2,7),fractioncorrect[0][i],label=f'{animal},{fractioncorrect[1][i]} Trials',color=marker_colors[i])
    perfomance_ax.set_ylim((0,1))
    perfomance_ax.set_ylabel('Fraction Correct')
    perfomance_ax.set_xlabel('Stimulus Duration')
    perfomance_ax.set_xticks(range(stimlen_range))
    perfomance_ax.legend()
    perfomance_ax.set_title(f'Peformance for all trials {date_range[0]} to {date_range[1]}')

reaction_plot, reaction_ax = plt.subplots()
reaction_times = []

for animal in animals:
    animal_df = trial_data.loc[animal]
    animal_correct_trials = animal_df[animal_df['Trial_Outcome'] == 1]
    trial_start_series = np.array([datetime.strptime(trial_start[:-1],'%H:%M:%S.%f') for trial_start in animal_correct_trials['Trial_Start']])
    trial_end_series = np.array([datetime.strptime(trial_end[:-1],'%H:%M:%S.%f') for trial_end in animal_correct_trials['Trial_End']])
    stimdur_series = np.array([timedelta(seconds=pre+post+0.9) for pre, post in
                               zip(animal_correct_trials['PreTone_Duration'],animal_correct_trials['PostTone_Duration'])])  # use sum of pre/post/tone durs
    reaction_series = trial_end_series-trial_start_series-stimdur_series
    animal_correct_trials['Reaction_Time'] = copy(reaction_series)
    reaction_times.append(reaction_series)

# for i, animal in enumerate(animals):
#     x_axis = np.full(len(reaction_times[i]),i)
#     reaction_seconds = np.array([t.total_seconds() for t in reaction_times[i]])
#     reaction_ax.scatter(x_axis, reaction_seconds,label=animal,s=20, facecolors='none',edgecolor=marker_colors[i])
#     reaction_ax.scatter(i,reaction_seconds.mean(),marker='x',color='k',s=50)

# plot pretone dur vs reaction time, violation rate
pre_vs_reaction_fig, pre_vs_reaction_ax = plt.subplots()
pre_vs_viol_fig, pre_vs_viol_ax = plt.subplots()
# get reaction time for full trial data df
trial_start_series = np.array([datetime.strptime(trial_start[:-1],'%H:%M:%S.%f') for trial_start in trial_data['Trial_Start']])
trial_end_series = np.array([datetime.strptime(trial_end[:-1],'%H:%M:%S.%f') for trial_end in trial_data['Trial_End']])
stimdur_series = np.array([timedelta(seconds=pre+post+0.9) for pre, post in
                           zip(trial_data['PreTone_Duration'],trial_data['PostTone_Duration'])])  # use sum of pre/post/tone durs
reaction_series = trial_end_series-trial_start_series-stimdur_series
rt_float = [t.total_seconds() for t in reaction_series]
trial_data['Reaction_Time'] = copy(rt_float)

# reaction time plot
for i, animal in enumerate(animals):
    x_axis = np.full(len(reaction_times[i]),i)
    reaction_seconds = np.array([t.total_seconds() for t in reaction_times[i]])
    reaction_ax.scatter(x_axis, reaction_seconds,label=animal,s=20, facecolors='none',edgecolor=marker_colors[i])
    reaction_ax.scatter(i,reaction_seconds.mean(),marker='x',color='k',s=50)


reaction_ax.set_xticks([])
reaction_ax.set_xlabel('')
reaction_ax.legend(loc=9,ncol=len(animals))
reaction_ax.set_ylabel('Reaction Time (seconds)')
reaction_ax.axhline(0.5,linestyle='--',color='grey',linewidth=0.5)

# stage2 reaction time only
stage2_stimdur_series = np.array([timedelta(seconds=t) for t in trial_data['Stim1_Duration']])
stage2_reactions = trial_end_series-trial_start_series-stage2_stimdur_series
stage2_reaction_series = np.array([t.total_seconds() for t in stage2_reactions])


correct_reactiontimes = trial_data[trial_data['Trial_Outcome'] == 1]
viol_reactiontimes = trial_data[trial_data['Trial_Outcome'] == -1]
# correct_rt_float = [t.total_seconds() for t in correct_reactiontimes]
# viol_rt_float = [t.total_seconds() for t in viol_reactiontimes]
pre_vs_reaction_ax.scatter(correct_reactiontimes['PreTone_Duration'],correct_reactiontimes['Reaction_Time'],
            s=12, facecolors='none',edgecolors='lightsteelblue')
pre_vs_reaction_ax.scatter(viol_reactiontimes['PreTone_Duration'],viol_reactiontimes['Reaction_Time'],
            s=12, facecolors='none', edgecolors='lavender')
pre_rt_mean = []
pre_viol_rate = []
# loop through unique embed times
for pre in np.sort(trial_data['PreTone_Duration'].unique()):
    pre_rt_mean.append([correct_reactiontimes[correct_reactiontimes['PreTone_Duration'] == pre]['Reaction_Time'].mean(),
                       viol_reactiontimes[viol_reactiontimes['PreTone_Duration'] == pre]['Reaction_Time'].mean()])
    pre_viol_rate.append(viol_reactiontimes[viol_reactiontimes['PreTone_Duration'] == pre].shape[0] /
                         trial_data[trial_data['PreTone_Duration'] == pre].shape[0])

pre_vs_viol_ax.plot(np.sort(trial_data['PreTone_Duration'].unique()),np.array(pre_viol_rate).transpose())

pre_vs_reaction_ax.set_xlim([0.25,5.5])
pre_vs_reaction_ax.set_xticks(trial_data['PreTone_Duration'].unique()[1:])
pre_vs_reaction_ax.set_xlabel('Tone Embedd time (s)')
pre_vs_reaction_ax.set_ylim((-6,4))
pre_vs_reaction_ax.set_ylabel('Reaction Time relative to Stim Duration (s)')

pre_vs_viol_ax.set_xlabel('Tone Embedd time (s)')
pre_vs_viol_ax.set_ylabel('Violation Rate')

for i, animal in enumerate(animals):
    animal_pretone_df = trial_data.loc[animal]
    animal_pretone_corr = animal_pretone_df[animal_pretone_df['Trial_Outcome'] == 1]
    animal_pretone_viol = animal_pretone_df[animal_pretone_df['Trial_Outcome'] == -1]

    pre_rt_mean = []
    for pre in np.sort(animal_pretone_df['PreTone_Duration'].unique()):
        pre_rt_mean.append(
            [animal_pretone_corr[animal_pretone_corr['PreTone_Duration'] == pre]['Reaction_Time'].mean(),
             animal_pretone_viol[animal_pretone_viol['PreTone_Duration'] == pre]['Reaction_Time'].mean(),pre])
    pre_vs_reaction_ax.scatter(np.array(pre_rt_mean).transpose()[2],np.array(pre_rt_mean).transpose()[0],marker='x',s=75,color=marker_colors[i])
    pre_vs_reaction_ax.scatter(np.array(pre_rt_mean).transpose()[2],np.array(pre_rt_mean).transpose()[1],marker='x',s=75,color=marker_colors[i])
pre_vs_reaction_ax.plot(np.sort(trial_data['PreTone_Duration'].unique()[1:-1]),np.array(pre_rt_mean).transpose()[0,1:-1],color='k')
pre_vs_reaction_ax.plot(np.sort(trial_data['PreTone_Duration'].unique()[1:-1]),np.array(pre_rt_mean).transpose()[1,1:-1],color='k')

# plot gaptone amp vs viol rate
gaptone_viol_fig, gaptone_viol_ax = plt.subplots()
trial_data['zeroed_gap_amp'] = trial_data['GapTone_Amplitude']
positive_amp_ix = trial_data['GapTone_Amplitude'] < 0
over_amp_ix = trial_data['GapTone_Amplitude'] > 87
trial_data.loc[positive_amp_ix,'zeroed_gap_amp'] = 0
trial_data.loc[over_amp_ix,'zeroed_gap_amp'] = 87

tone_amp_arr = []
for i, amp in enumerate(np.sort(trial_data['zeroed_gap_amp'].unique())):
    subset_amp = trial_data.loc[trial_data['zeroed_gap_amp'] == amp,'Trial_Outcome'] == -1
    # subset_viols = subset_amp[subset_amp['Trial_Outcome'] == -1]
    amp_viol_rate = subset_amp.sum()/subset_amp.shape[0]
    tone_amp_arr.append([amp,amp_viol_rate])
tone_amp_arr = np.array(tone_amp_arr)
gaptone_viol_ax.scatter(tone_amp_arr[2:,0],tone_amp_arr[2:,1])
gaptone_viol_ax.set_xlim((55,90))
gaptone_viol_ax.set_xticks(np.arange(60,95,5))
gaptone_viol_ax.set_xlabel('Embedded Tone Amplitude (db)')
gaptone_viol_ax.set_ylabel('Violation Rate')


# gotone vs reaction time
gotone_reactions = []
trial_data['zeroed_go_amp'] = trial_data['GoTone_Amplitude']
neagtive_amp_ix = trial_data['GoTone_Amplitude'] < 0
over_amp_ix = trial_data['GoTone_Amplitude'] > 87
trial_data.loc[neagtive_amp_ix,'zeroed_go_amp'] = 0
trial_data.loc[over_amp_ix,'zeroed_go_amp'] = 87
trial_data['stage2_reaction'] = stage2_reaction_series

gotone_viol_fig, gotone_viol_ax = plt.subplots()
for i, amp in enumerate(np.sort(trial_data['zeroed_go_amp'].unique())):
    subset_amp_df = trial_data.loc[trial_data['zeroed_go_amp'] == amp]
    subset_outcome_df = subset_amp_df.loc[subset_amp_df['Trial_Outcome'] == 1]
    gotone_viol_ax.scatter(np.full_like(subset_outcome_df['stage2_reaction'],amp),subset_outcome_df['stage2_reaction'],
                           facecolors='none',edgecolor='lightsteelblue')
    gotone_reaction_mean = subset_outcome_df['stage2_reaction'].mean()
    gotone_reactions.append([amp,gotone_reaction_mean])

gotone_reactions = np.array(gotone_reactions)
gotone_viol_ax.scatter(gotone_reactions[2:,0],gotone_reactions[2:,1],marker='x',color='k')
gotone_viol_ax.set_xlim((85,40))
gotone_viol_ax.set_xticks(np.arange(0,95,5))
gotone_viol_ax.set_xlabel('Go Tone Amplitude (db)')
gotone_viol_ax.set_ylabel('Reaction Time (s)')
gotone_viol_ax.set_title('Stage 2 reaction times vs Go Tone Amplitude')



total_valvetime = animal_correct_trials['Reward_Amount'].sum()
# print(total_valvetime*.112)


# session_dfs = plot_sessions.subset_dates(trial_data)
# plot_sessions.plt_sess_features(session_dfs,['Trial_Outcome'])
# plot_sessions.plot_featuresvsdate(session_dfs,[['Time','total'],['Valve_Time','mean']],animals)