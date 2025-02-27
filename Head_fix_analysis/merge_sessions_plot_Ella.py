# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 17:54:49 2021

@author: Ella Svahn
"""

#%%
from typing import Type
import pandas as pd
import numpy as np
import os
from copy import copy
import time
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
plt.rcParams.update({'font.size':16})
import seaborn as sns
#import plot_sessions


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
                    if loaded_file['name'][0] in animal_list:
                        loaded_file.set_index(['name','date']).sort_index()
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


'''
def create_subset_df(data_dict, fields, date_period=('01/01/1000','now'), rats=None):
    """

     :param data_dict:
     :param fields:
     :param date_period:
     :param rats:
     :return:
    """
     # print('starting')
     t0 = time.time()
     if rats is None or len(rats) < 1 or not isinstance(rats[0], str):
         rats = list(data_dict.keys())
     # print(type(data_dict),date_period,rats)
     dates_dict = subset_dates(data_dict,date_period,rats)
     #     # list_rat_dfs = []
     rsf = []
     for r in rats:  # r is rat name i.e keys for dictionary
         for s in data_dict[r]:
             sess_date = extract_val(s['SavingSection_SaveTime']).split()
             if sess_date[0] in dates_dict[r][1]:
                 use = 1
             else:
                 use = 0
             if use == 1:
                 val = []
                 for f in fields:
                     val.append(extract_val(s[f]))
                 rsf.append([r]+val)

     df = pd.DataFrame(rsf)
     t1 = time.time()
     # print('Time taken ',t1-t0)
     return df
'''

# params = merge_sessions(datadir,animals,'params')
# summary_data = merge_sessions(datadir,animals,'params')
# current_params = [animal.tail(1) for animal in params]
# current_params = pd.concat(current_params)
# summary_data = pd.concat(summary_data)

# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (9, 6),
#          'axes.labelsize': 'medium',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'medium',
#          'ytick.labelsize':'medium'}
# plt.rcParams.update(params)

marker_colors = ['b','r','c','m','y','g']

animals = ['ES01',
            'ES02',
            'ES03'
            
]

datadir = r'C:\Users\Ella Svahn\Git\data\Ella'
date_range =[
             '13/09/2021',
             '28/09/2021',
             ]


trial_data = merge_sessions(datadir,animals,'TrialData')
trial_data = pd.concat(trial_data, sort=False, axis=0)

#%%
trial_df = trial_data
trial_list=[]
for animal in animals:
    animal_df = trial_df.loc[animal]
    trial_list.append(animal_df.shape[0])
    for trial in trial_list:
        if trial_data[trial_data.Trial_Outcome==1][trial]:
            animal_df['Correct'][trial]=1
        elif trial_data[trial_data.Trial_Outcome==-1][trial]:
            animal_df['Correct'][trial]=0    


#%%
correct= []
violation=[]
abort=[]
trialList=[trial_df.shape[0]]
for trial in range(len(trialList)):
    if trial_data[trial][trial_data.Trial_Outcome==1]:
        correct.append(trial)
    elif trial_data[trial_data.Trial_Outcome==-1]:
        violation.append(trial)    
    elif trial_data[trial_data.Trial_Outcome==0]:
        abort.append(trial)
        

        

#%% simple seaborn performance plot: performance per mouse per day, all stim lenths 
# Stage 0

fig =plt.subplot()
sns.pointplot(data=trial_data.groupby(['name','date']).mean().reset_index(),x= 'date',y='Trial_Outcome', hue = 'name', palette = 'mako')
#plt.axhline(y=0.5, c='0.5', linestyle = '--')
sns.despine()
plt.ylabel('Fraction correct')
plt.xlabel('Days')
plt.ylim(0,1)

#plt.xticks([210909,210910],[0,1])
plt.title('Performance Stage 0')
fig.axes.xaxis.set_ticklabels([])


#%%
# Stage 1 short tones
#[trial_data.Stim1_Duration<0.5]
fig = plt.subplot()
sns.pointplot(data=trial_data[trial_data.Stim1_Duration <0.5].groupby(['name','date']).mean().reset_index(),x= 'date',y='Trial_Outcome', hue = 'name', palette = 'mako')
#plt.axhline(y=0.5, c='0.5', linestyle = '--')
plt.ylim(0,1.1)
sns.despine()
plt.ylabel('Fraction correct')
plt.xlabel('Days')
plt.title('Stage 1: long tone durations')
fig.axes.xaxis.set_ticklabels([])
plt.gca().legend_.remove()


#%%
# Stage 1

fig = plt.subplot()
sns.pointplot(data=trial_data[trial_data.Trial_Outcome !=0].groupby(['name','date']).mean().reset_index(),x= 'date',y='Trial_Outcome', hue = 'name', palette = 'mako')
#plt.axhline(y=0.5, c='0.5', linestyle = '--')
sns.despine()
plt.ylabel('Fraction correct')
plt.xlabel('Days')
plt.title('Performance Stage 1: long tone durations ')
fig.axes.xaxis.set_ticklabels([])
fig.axes.yaxis.set_ticklabels([])
plt.gca().legend_.remove()


 #%%
# Stage 1 performance vs stim1 dur
fig = plt.subplot()
 
sns.scatterplot(data=trial_data[trial_data.WarmUp==False].groupby(['name','date']).mean().reset_index(),x= 'Stim1_Duration',y='Trial_Outcome', hue = 'name', palette = 'mako')
#plt.axhline(y=0.5, c='0.5', linestyle = '--')
sns.despine()
plt.ylabel('Fraction correct')
plt.xlabel('Tone duration (s)')
plt.ylim(0,1)
plt.xlim(0,2.1)
#plt.xticks([1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14])
#plt.title('Stimulus duration increases throughout Stage 1')
#fig.axes.xaxis.set_ticklabels([])

#%%
#decrease in reward attempts over time
fig = plt.subplot()
sns.pointplot(data=trial_data[trial_data.Trial_Outcome !=0].groupby(['name','date']).mean().reset_index(),x= 'date',y='Reward_Attempts', hue = 'name', palette = 'mako')
#plt.axhline(y=0.5, c='0.5', linestyle = '--')
sns.despine()
plt.ylabel('Reward attempts')
plt.xlabel('Days')
plt.title('Reward attempts')
fig.axes.xaxis.set_ticklabels([])
#fig.axes.yaxis.set_ticklabels([])


#%% Final performance per side (L vs R) to show that antibias works 
# mean across mice  
fig = plt.subplot()
sns.pointplot(data=trial_data.groupby(['RewardedSide','date']).mean().reset_index(),x= 'date',y='Trial_Outcome', hue = 'RewardedSide', palette = 'Set2')
sns.despine()
plt.title('Mean % correct for L & R')
plt.ylabel('Fraction correct')
plt.xlabel('Days')
plt.ylim(0,1)
fig.axes.xaxis.set_ticklabels([])

#%% choice ratio to show antibias
Rcorr = trial_data.Trial_Outcome[trial_data.RewardedSide==1].groupby(['name','date']).mean()
Lcorr = trial_data.Trial_Outcome[trial_data.RewardedSide==2].groupby(['name','date']).mean()
prob_L = Rcorr/Lcorr

fig = plt.subplot()
sns.pointplot(data=prob_L.groupby(['name','date']).mean().reset_index(),x= 'date',y='Trial_Outcome', hue = 'name', palette = 'mako')
plt.ylabel('Choice side ratio')
sns.despine()
fig.axes.xaxis.set_ticklabels([])
plt.xlabel('Days')


 

#%%
def get_fractioncorrect(data_df, animal_list):
    performance = []
    ntrial_list = []
    for animal in animals:
        stim_perfomance = []
        stim_stdev = []
        animal_df = trial_data.loc[animal]
        ntrial_list.append(animal_df.shape[0])
        for stim in range(0,1):
            stim_df = animal_df[animal_df['Stim1_Duration'] == stim]['Trial_Outcome']
            stim_correct = stim_df == 1
            stim_perfomance.append(stim_correct.mean())
            stim_stdev.append(stim_correct.std())
        performance.append(stim_perfomance)
        stdev.append(stim_stdev)
    
'''   
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
'''
fractioncorrect = get_fractioncorrect(trial_data, trial_data.keys().unique()[0])


#%%
marker_colors = ['b','r','c','m','y','g']

perfomance_plot, perfomance_ax = plt.subplots(1,1)
for i, animal in enumerate(animals):
    perf = np.array(performance[i])
    std = np.array(stdev[i])
    perfomance_ax.plot(np.arange(2,7),performance[i],label=f'Animal {i}',color=marker_colors[i])
    perfomance_ax.plot(np.arange(2,7),perf+std,color=marker_colors[i], linestyle='--')
    # perfomance_ax.plot(np.arange(2,7),perf-std,color=marker_colors[i], linestyle='--')


perfomance_ax.set_ylim((0,1))
perfomance_ax.set_ylabel('Fraction Correct',fontsize=14)
perfomance_ax.set_xlabel('Stimulus Duration',fontsize=14)
perfomance_ax.set_xticks(range(2,7))
# perfomance_ax.set_xticklabels(xticks,fontsize=10)
# perfomance_ax.set_yticklabels(yticks,fontsize=10)

perfomance_ax.legend()
perfomance_ax.set_title(f'Peformance for all trials')
perfomance_plot.set_size_inches((6,6))

#%%
#function to calculate reaction times 
#'animals' arg is a list of animal names to analyse
def get_reactiontime(trial_data, animals):
    #if wanting to choose which trial_type to analyse: 'correct', 'violation', or 'all, then add 'trial_outcome' as function arg 
    reaction_times = []

    for animal in animals:
        '''
        if trial_outcome == 'violation':
            animal_df = trial_data.loc[animal['Trial_Outcome'] == 1]
        elif trial_outcome == 'correct':
            animal_df = trial_data.loc[animal['Trial_Outcome'] == 1]
        elif trial_outcome == 'all':
            animal_df = trial_data.loc[animal]
        else: 
        ''' 
        animal_df = trial_data.loc[animal] #default is all trial types 

        trial_start_series = np.array([datetime.strptime(trial_start[:-1],'%H:%M:%S.%f') for trial_start in animal_df['Trial_Start']])
        trial_end_series = np.array([datetime.strptime(trial_end[:-1],'%H:%M:%S.%f') for trial_end in animal_df['Trial_End']])
        stimdur_series = np.array([timedelta(seconds=pre+post+stim) for pre, post,stim in # use sum of pre/post/tone durs
                                zip(animal_df['PreTone_Duration'],animal_df['PostTone_Duration'],animal_df['Stim1_Duration'])])  
        reaction_series = trial_end_series - trial_start_series - stimdur_series
    reaction_times.append(reaction_series)
    rt_float = [t.total_seconds() for t in reaction_series]
    animal_df['Reaction_Time'] = copy(rt_float)        
    
    return animal_df, reaction_times 
#print(animal_correct_trials)

#%%
#function for doing various reaction time plots 
def plot_reactiontime(animal_df, reaction_times):
    
    correct_reactiontimes = animal_df[animal_df['Trial_Outcome'] == 1]
    viol_reactiontimes = animal_df[animal_df['Trial_Outcome'] == -1]
    
    # correct_rt_float = [t.total_seconds() for t in correct_reactiontimes]
    # viol_rt_float = [t.total_seconds() for t in viol_reactiontimes]  

    #plot recation time scatter with all RTs for each animal 
    reaction_fig, reaction_ax = plt.subplots(1,1)

    for i, animal in enumerate(animals):
        x_axis = np.full(len(reaction_times[i]),i)
        reaction_seconds = np.array([t.total_seconds() for t in reaction_times[i]])
        reaction_ax.scatter(x_axis, reaction_seconds,label=animal,s=20, facecolors='none',edgecolor=marker_colors[i])
        reaction_ax.scatter(i,reaction_seconds.mean(),marker='x',color='k',s=50)
    reaction_ax[0].set_xticks([])
    reaction_ax[0].legend(loc=9,ncol=len(animals))
    reaction_ax[0].set_ylabel('Reaction Time (seconds)')
    reaction_ax[0].axhline(0.5,linestyle='--',color='grey',linewidth=0.5)

    # plot pretone dur vs reaction time, violation rate
    pre_vs_reaction_fig, pre_vs_reaction_ax = plt.subplots()
    pre_vs_reaction_ax.scatter(correct_reactiontimes['PreTone_Duration'],correct_reactiontimes['Reaction_Time'],
                s=12, facecolors='none',edgecolors='b')
    pre_vs_reaction_ax.scatter(viol_reactiontimes['PreTone_Duration'],viol_reactiontimes['Reaction_Time'],
                s=12, facecolors='none', edgecolors='r')
    
    #plot pre-reation time for corrent vs violation trial 
    pre_rt_mean = []
    for pre in np.sort(trial_data['PreTone_Duration'].unique()):
        pre_rt_mean.append([correct_reactiontimes[correct_reactiontimes['PreTone_Duration']==pre]['Reaction_Time'].mean(),
                        viol_reactiontimes[viol_reactiontimes['PreTone_Duration']==pre]['Reaction_Time'].mean()])
    pre_vs_reaction_ax.plot(np.array(pre_rt_mean).transpose()[0,1:-1])
    pre_vs_reaction_ax.plot(np.array(pre_rt_mean).transpose()[1,1:-1])
    pre_vs_reaction_ax.set_xlim([0.25,5.5])
    pre_vs_reaction_ax.set_xticks(trial_data['PreTone_Duration'].unique()[1:])
    pre_vs_reaction_ax.set_ylabel('Reaction Time relative to Stim Duration (s)')
    pre_vs_reaction_ax.set_xlabel('Tone Embedd time (s)')

    # gaptone amp plot?
#%%

#calcuate and plot reaction times
animal_df, reaction_times = get_reactiontime(trial_data, animals)
plot_reactiontime(animal_df, reaction_times)


#%% 
#function to calculate amount of  licking post X tone until reward (predictive licking), menaing the animal has detected the anomaly 
# UNFINISHED this needs more work
def get_detectionLicks(animal_df):
  
    for animal in animals: 
        trial_start_series = np.array([datetime.strptime(trial_start[:-1],'%H:%M:%S.%f') for trial_start in animal_df['Trial_Start']])
        print('start',trial_start_series)
        #calulate tone onset time from (trial_start - stim1_dur) ?
        #tone_start_series = trial_start_series + animal_df['Stim1_Duration]
        print('sti',animal_df['Stim1_Duration'])
        reward_time_series = np.array([datetime.strptime(reward_time[:-1],'%H:%M:%S.%f') for reward_time in animal_df['RewardTone_Time']])
        print ('rew', reward_time_series)
        #definethe detection window between X tone onset to reward 
        #extract lick signal for each trial between trial start to reward 
        animal_df[animal['Detection_Licks']] = animal_df[animal_df['Licks'][detection_window]]
        #also add to trial_data?
    return animal_df

get_detectionLicks(animal_df)


#%%
# embedd time vs withdrawal

viol_rate = []

for animal in animals:
    t_viol = []
    animal_df = trial_data.loc[animal]
    for t in animal_df['PreTone_Duration'].unique():
        t_df = animal_df[animal_df['PreTone_Duration'] == t]['Trial_Outcome']
        t_viol_df = t_df == -1
        t_viol.append(t_viol_df.sum()/t_df.shape[0])
    viol_rate.append(t_viol)

viol_embedd_plot, viol_embedd_ax = plt.subplots(1,1)
embedd_time_axis = trial_data['PreTone_Duration'].unique()
for i, animal in enumerate(animals):
    viol = np.array(viol_rate[i])
    viol_embedd_ax.scatter(embedd_time_axis,viol_rate[i],label=f'Animal {i}',color=marker_colors[i])
    # perfomance_ax.plot(np.arange(2,7),perf+std,color=marker_colors[i], linestyle='--')
    # perfomance_ax.plot(np.arange(2,7),perf-std,color=marker_colors[i], linestyle='--')


viol_embedd_ax.set_ylim((0,1))
viol_embedd_ax.set_xlim((.2,5.5))
viol_embedd_ax.set_ylabel('Early Withdrawal Rate',fontsize=14)
viol_embedd_ax.set_xlabel('Embed Time',fontsize=14)
viol_embedd_ax.set_xticks(np.arange(.5,6,0.5))
# viol_embedd_ax.set_xticklabels(xticks,fontsize=10)
# viol_embedd_ax.set_yticklabels(yticks,fontsize=10)

viol_embedd_ax.legend()
viol_embedd_ax.set_title(f'Early Withdrawl Rate vs Tone Embed Timepoint')
viol_embedd_plot.set_size_inches((9,6))


for i, animal in enumerate(animals):
    x_axis = np.full(len(reaction_times[i]),i)
    reaction_seconds = np.array([t.total_seconds() for t in reaction_times[i]])
    reaction_ax[0].scatter(x_axis, reaction_seconds,label=animal,s=20, facecolors='none',edgecolor=marker_colors[i])
    reaction_ax[0].scatter(i,reaction_seconds.mean(),marker='x',color='k',s=50)


reaction_ax[0].set_xticks([])
reaction_ax[0].legend(loc=9,ncol=len(animals))
reaction_ax[0].set_ylabel('Reaction Time (seconds)')
reaction_ax[0].axhline(0.5,linestyle='--',color='grey',linewidth=0.5)


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

# session_dfs = plot_sessions.subset_dates(trial_data)
# plot_sessions.plt_sess_features(session_dfs,['Trial_Outcome'])
# plot_sessions.plot_featuresvsdate(session_dfs,[['Time','total'],['Valve_Time','mean']],animals)
