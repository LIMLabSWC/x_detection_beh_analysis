import pandas as pd
import numpy as np
import os
from copy import copy
import time
from datetime import datetime, timedelta
from collections import OrderedDict
from matplotlib import pyplot as plt
import analysis_utils as utils
import pylab
from sklearn.linear_model import LinearRegression
from math import floor,ceil


idrange = [30,43]
# animals = [f'DO{i}' for i in range(idrange[0],idrange[1]+1)]
# exclude = ['DO31','DO34','DO36','DO40','DO41']
# [animals.remove(exc) for exc in exclude]
animals = [
            # 'DO27',
#             'DO28',
#             'DO29',
            'DO42',
            'DO43',
            'DO37',
]
anon_animals = [f'Animal {i}' for i in range(len(animals))]
datadir = r'C:\bonsai\data'
dates = ['22/10/2021', '28/10/2021']
# plot_colours = ['b','r','c','m','y','g']
plot_colours = plt.cm.jet(np.linspace(0,1,len(animals)))
trial_data = utils.merge_sessions(datadir,animals,'TrialData',dates)
trial_data = pd.concat(trial_data, sort=False, axis=0)

#add datetime cols
for col in trial_data.keys():
    if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1:
        if col.find('Wait') == -1 and col.find('dt') == -1:
            utils.add_datetimecol(trial_data,col)
stimdurs = np.arange(4,8.5,.5)

# plot day to day performance
stats_dict = dict()
for animal in animals:
    stats_dict[animal] = dict()
    for date in trial_data.loc[animal].index.unique():
        cols = ['Trials Done', 'NonViol Trials','Early Rate', 'Error Rate']
        data_day = utils.filter_df(trial_data.loc[animal,date],['b1'])
        trials_done_day = data_day.shape[0]
        correct_trials  = utils.filter_df(data_day,['a3']).shape[0]
        try:
            early_rate = utils.filter_df(data_day,['a2']).shape[0]/trials_done_day
        except ZeroDivisionError:
            early_rate = 1
        try:
            error_rate = utils.filter_df(data_day, ['a0']).shape[0] / utils.filter_df(data_day, ['a3']).shape[0]
        except ZeroDivisionError:
            error_rate = 1
        stats_day = pd.DataFrame([[trials_done_day,correct_trials,early_rate,error_rate]],columns=cols)
        stats_dict[animal][date] = stats_day
fig,ax = plt.subplots(4,sharex=True)
for i, animal in enumerate(animals):
    for id, d in enumerate(stats_dict[animal].keys()):
        for f, feature in enumerate(stats_dict[animal][d]):
            ax[f].scatter(id, stats_dict[animal][d][feature], marker='o', color=plot_colours[i],label=animal)
            if i == 0:
                ax[f].set_ylabel(f'{feature}')
                # ax[f].set_xlabel('Session Number')

handles, labels = fig.gca().get_legend_handles_labels()
for axis in ax:
    by_label = OrderedDict(zip(labels, handles))
    # axis.legend(by_label.values(), by_label.keys())
fig.legend(by_label.values(), by_label.keys())
dates_unique = []
for animal in animals:
    dates_unique.extend(stats_dict[animal].keys())
dates_unique = pd.Series(dates_unique).unique()
ax[-1].set_xticks(np.arange(len(dates_unique)))
ax[-1].set_xticklabels(list(dates_unique),rotation=40,ha='center',size=9)

plots = utils.plot_performance(trial_data, stimdurs, animals, dates, plot_colours)
plot_early = utils.plot_metric_v_stimdur(trial_data,stimdurs,'Trial_Outcome',-1,animals,dates,
                                         plot_colours,['b1'], ytitle= 'Early rate',
                                         legend_labels = anon_animals)
plot_error_notones = utils.plot_metric_v_stimdur(trial_data,stimdurs,'Trial_Outcome',0,animals,dates,
                                                 plot_colours,['b1','a3','e=0'],'Error rate without Tones', 'Error Rate no tones',
                                                 legend_labels = anon_animals)
plot_error_notones[1].set_title('Miss rate for non pattern played trials')
plot_error_tones = utils.plot_metric_v_stimdur(trial_data,stimdurs,'Trial_Outcome',0,animals,dates,
                                               plot_colours,['b1','a3','e!0'], 'Error rate with Tones', 'Error Rate Tones',
                                               legend_labels = anon_animals)
plot_error_tones[1].set_title('Miss rate for pattern played trials')
plot_error = utils.plot_metric_v_stimdur(trial_data,stimdurs,'Trial_Outcome', 0,animals,dates,
                                         plot_colours,['b1','a3'],'Miss rate all trials', 'Error Rate',
                                         legend_labels = anon_animals)
plot_error[1].set_title('Miss rate all trials')

# plot_early_embedtime = utils.plot_metric_v_stimdur(trial_data,np.arange(),'Trial_Outcome',-1,animals,dates,
#                                          plot_colours,['b1'], ytitle= 'Early rate',
#                                          legend_labels = anon_animals,plottype='scatter')

# early_df = utils.filter_df(trial_data,['b1','a2'])
early_df = utils.filter_df(trial_data,['b1'])

early_df['Trial_End_datetime'] = np.array([datetime.strptime(trial_end[:-1], '%H:%M:%S.%f')
                                           for trial_end in early_df['Trial_End']])
early_df['Trial_Start_datetime'] = np.array([datetime.strptime(trial_end[:-1], '%H:%M:%S.%f')
                                             for trial_end in early_df['Trial_Start']])
early_df['StimEnd_datetime'] = np.array([(starttime+timedelta(seconds=stimdur)*0) for starttime,stimdur
                                         in zip(early_df['Trial_Start_datetime'],early_df['Stim1_Duration'])])

relearly = early_df['Trial_End_datetime'] - early_df['StimEnd_datetime']
early_df['End_vs_Stimdur'] = np.array([t.total_seconds() for t in relearly])
endvsstimdur_ax, endvsstimdur_ax = plt.subplots(1)
for i, animal in enumerate(animals):
    endvsstimdur_ax.hist(early_df.loc[animal]['End_vs_Stimdur'], edgecolor=plot_colours[i],label=f'animal {i}',histtype='step',
                         density=True,bins=np.arange(floor(early_df.loc[animal]['End_vs_Stimdur'].min()),
                                                     ceil(early_df.loc[animal]['End_vs_Stimdur'].max()),1))
endvsstimdur_ax.legend()
endvsstimdur_ax.set_xlabel('Trial end relative to Trial Start', size =12)
endvsstimdur_ax.axvline(0,color='grey',linestyle='--')
endvsstimdur_ax.set_title('Mouse response aligned to Trial Start',size=14)
# endvsstimdur_ax.set_xlim((-8,8))
for i, animal in enumerate(animals):
    endvsstimdur_ax.hist(early_df.loc[animal]['End_vs_Stimdur'], edgecolor=plot_colours[i], label=f'animal {i}',
                         alpha=0.05,
                         density=True, bins=np.arange(floor(early_df.loc[animal]['End_vs_Stimdur'].min()),
                                                      ceil(early_df.loc[animal]['End_vs_Stimdur'].max()), 1))

# plot early rate vs trial number

# for i,animal in enumerate(animals):
#     animal_df = nowarmupdf.loc[animal]
#     print(animal_df['Trial#'].max())
#     for trialnum in np.unique(animal_df['Trial#']):
#         early_trialnum = animal_df[animal_df['Trial#'] == trialnum]['Trial_Outcome'] == -1
#         earlyrate_trialnum = early_trialnum.sum()/len(early_trialnum)
#         trialnum_vs_earlyrate_ax.scatter(trialnum,earlyrate_trialnum, color=plot_colours[i])





# xy = np.array(xy)
# plot lin regression

earlytrialnum_fig,earlytrialnum_ax,earlytrialnum_xy = utils.plot_metricrate_trialnun(trial_data,'Trial_Outcome',-1,
                                                                                       ('b1',),'Early rate over session',
                                                                                       'Early Rate',True)
correcttrialnum_fig,correcttrialnum_ax,correcttrialnum_xy = utils.plot_metricrate_trialnun(trial_data,'Trial_Outcome',1,
                                                                                       ('b1',),'Correct rate over session',
                                                                                       'Correct Rate',True)
# for root, folder, files in os.walk(r'W:\mouse_pupillometry\4_21_2021'):
#     for file in files:
#         if file.find('timestamp.dat') != -1:
#             data = utils.plot_frametimes(os.path.join(root, file))
#             # plt.plot(data['frameNum'],data['rel_time'])
#             plt.hist(data['rel_time'], bins=data['rel_time'].max(),alpha=0.1,density=True)
hist_posttone_fig,hist_posttone_ax = plt.subplots(2,sharex=True)
weights = np.ones_like(utils.filter_df(trial_data,['d0','b1'])['PostTone_Duration']) / len(utils.filter_df(trial_data,['d0','b1'])['PostTone_Duration'])
hist_posttone_ax[0].hist(utils.filter_df(trial_data,['d0','b1'])['PostTone_Duration'],histtype='step',weights=weights)
weights = np.ones_like(utils.filter_df(trial_data,['d!0','b1'])['PostTone_Duration']) / len(utils.filter_df(trial_data,['d!0','b1'])['PostTone_Duration'])
hist_posttone_ax[1].hist(utils.filter_df(trial_data,['d!0','b1'])['PostTone_Duration'],histtype='step',weights=weights)

for col in trial_data.keys():
    if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1:
        if col.find('Wait') == -1 and col.find('dt') == -1:
            utils.add_datetimecol(trial_data,col)

# plot viol time relative to tone time
viol_t_tone_df = utils.filter_df(trial_data,['b1','a2','e!0'])
viol_t_violtime = viol_t_tone_df['Trial_End_dt']
viol_firsttone = viol_t_tone_df['ToneTime_dt']
viol_vs_tone = [t.total_seconds() for t in viol_t_violtime - viol_firsttone]

hist_viol_tones_fig, hist_viol_tones_ax = plt.subplots(1)
hist_viol_tones_ax.hist(viol_vs_tone,density=True,bins=np.arange(0,np.array(viol_vs_tone).max()+.5,.5),alpha=0.1)

first_n = []
last_n = []
n_earlies = 10
for sess_ix in viol_t_tone_df.index.unique():
    sess_viols = viol_t_tone_df.loc[sess_ix]
    first_n.extend([t.total_seconds() for t in sess_viols['Trial_End_dt'] -sess_viols['ToneTime_dt']][:n_earlies])
    last_n.extend([t.total_seconds() for t in sess_viols['Trial_End_dt'] -sess_viols['ToneTime_dt']][60:70])

hist_viol_tones_fig, hist_viol_tones_ax = plt.subplots(1)

for group in zip([first_n,last_n],[f'first {n_earlies}',f'last {n_earlies}']):
    weights = np.ones_like(group[0]) / len(group[0])
    hist_viol_tones_ax.hist(group[0],label=group[1],weights=weights,histtype='step',bins=np.arange(0,8,.5))
hist_viol_tones_ax.legend()
hist_viol_tones_ax.set_ylabel('Density')
hist_viol_tones_ax.set_xlabel('Early Withrawal Time aligned to Pattern Tone "A" (s)',size=12)
hist_viol_tones_ax.set_ylabel('Density')
hist_viol_tones_ax.set_title('Distribution of early withdrawals aligned to Pattern Tone "A"',size=14)

allviols_fig, allviols_ax = plt.subplots()
embed_t_violcount = []
time_violcount = []
non_warmps_df = utils.filter_df(trial_data,['b1'])
embed_t_viol_df = utils.filter_df(trial_data,['b1','e!0'])
for embed_t in sorted(embed_t_viol_df['PreTone_Duration'].unique()):
    viols_df = copy(embed_t_viol_df[embed_t_viol_df['PreTone_Duration'] == embed_t])
    embed_t_violcount.append((viols_df['Trial_Outcome']==-1).sum()/(viols_df['Trial_Outcome']!=-1).sum())
    viols_df_time = copy(non_warmps_df[non_warmps_df['Stim1_Duration'] == embed_t])
    time_violcount.append((viols_df_time['Trial_Outcome']==-1).sum()/(viols_df_time['Trial_Outcome']!=-1).sum())

allviols_ax.scatter(sorted(embed_t_viol_df['PreTone_Duration'].unique()),embed_t_violcount,label='Pattern Time')
allviols_ax.scatter(sorted(embed_t_viol_df['PreTone_Duration'].unique()),time_violcount,label='Stim Duration Time')
allviols_ax.set_xlabel('Time from pattern tone "A" (s)',size=12)
allviols_ax.set_title('Distribution of early withdrawal rate with pattern presentation time')
allviols_ax.legend()

