import pandas as pd
import numpy as np
import os
from copy import copy as copy
import time
from datetime import datetime, timedelta
from matplotlib import pyplot as plt


def merge_sessions(datadir,animal_list,filestr_cond, datestr_format='%yy%mm%dd'):
    """
    Function to merge csv files for given conditon
    :param datadir:
    :param animal_list:
    :param filestr_cond:
    :param datestr_format:
    :return:
    """

    file_df = []
    for root, folder, files in os.walk(datadir):
        if filestr_cond == 'SummaryData' or filestr_cond == 'params':
            for file in files:
                if file.find(filestr_cond) != -1:
                    loaded_file = pd.read_csv(os.path.join(root,file), delimiter=',')
                    if loaded_file['Name'][0] in animal_list:
                        loaded_file.set_index(['Name','Date']).sort_index()
                        file_df.append(copy(loaded_file))

        elif filestr_cond == 'TrialData':
            for file in files:
                if file.find(filestr_cond) != -1:
                    animal_name = file[0:4]
                    session_date = file[-11:-5]
                    if animal_name in animal_list \
                            and datetime.strptime(date_range[0], '%d/%m/%Y') <= datetime.strptime(session_date,'%y%m%d')\
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



animals = [
            #'DO12',
            #'DO13',
            #'DO14',
            # 'DO15',
            # 'DO16',
            # 'DO17',
            #'DO18',
            'DO19',
            # 'DO20',
            # 'DO23',
            # 'DO24',
            # 'DO25',
            # 'DO26'
]

datadir = r'C:\bonsai\data\Dammy'
date_range =['20/11/2020', '20/11/2020']


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
# do17_stage2 = pd.concat([trial_data.loc['DO17','201012'],trial_data.loc['DO17','201013']])
# do17_stage2_viols = do17_stage2['Trial_Outcome'] == -1
# do17_stage2_cum_mean = do17_stage2_viols.expanding().mean()
# plt.plot(do17_stage2_cum_mean.values)

performance = []
ntrial_list = []
for animal in animals:
    stim_perfomance = []
    animal_df = trial_data.loc[animal]
    ntrial_list.append(animal_df.shape[0])
    for stim in range(2,7):
        stim_df = animal_df[animal_df['Stim1_Duration'] == stim]['Trial_Outcome']
        stim_correct = stim_df == 1
        stim_perfomance.append(stim_correct.mean())
    performance.append(stim_perfomance)



perfomance_plot, perfomance_ax = plt.subplots(1,1)
for i, animal in enumerate(animals):
    perfomance_ax.plot(np.arange(2,7),performance[i],label=f'{animal},{ntrial_list[i]} Trials',color=marker_colors[i])
perfomance_ax.set_ylim((0,1))
perfomance_ax.set_ylabel('Fraction Correct')
perfomance_ax.set_xlabel('Stimulus Duration')
perfomance_ax.set_xticks(range(2,7))
perfomance_ax.legend()
perfomance_ax.set_title(f'Peformance for all trials {date_range[0]} to {date_range[1]}')

reaction_plot, reaction_ax = plt.subplots()
reaction_times = []

for animal in animals:
    animal_df = trial_data.loc[animal]
    animal_correct_trials = animal_df[animal_df['Trial_Outcome'] == 1]
    trial_start_series = np.array([datetime.strptime(trial_start[:-1],'%H:%M:%S.%f') for trial_start in animal_correct_trials['Trial_Start']])
    trial_end_series = np.array([datetime.strptime(trial_end[:-1],'%H:%M:%S.%f') for trial_end in animal_correct_trials['Trial_End']])
    stimdur_series = np.array([timedelta(seconds=t) for t in animal_correct_trials['Stim1_Duration']])
    reaction_series = trial_end_series-trial_start_series-stimdur_series
    reaction_times.append(reaction_series)

for i, animal in enumerate(animals):
    x_axis = np.full(len(reaction_times[i]),i)
    reaction_seconds = np.array([t.total_seconds() for t in reaction_times[i]])
    reaction_ax.scatter(x_axis, reaction_seconds,label=animal,s=20, facecolors='none',edgecolor=marker_colors[i])
    reaction_ax.scatter(i,reaction_seconds.mean(),marker='x',color='k',s=50)

reaction_ax.set_xticks([])
reaction_ax.legend(loc=9,ncol=len(animals))
reaction_ax.set_ylabel('Reaction Time (seconds)')
reaction_ax.axhline(0.5,linestyle='--',color='grey',linewidth=0.5)

total_valvetime = animal_correct_trials['ValveTime'].sum()
print(total_valvetime*.112)

# def cal_step_mean(start,step,iters,thresh):
#     amount = start
#     tally = 0
#     for i in range(iters):
#         tally+=amount
#         if i%thresh ==0 and i>0:
#             amount+=step
#     return tally, tally/iters
