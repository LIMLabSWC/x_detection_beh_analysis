import pandas as pd
import numpy as np
import os
from copy import copy
import time
from datetime import datetime, timedelta
from collections import OrderedDict
from matplotlib import pyplot as plt
import analysis_utils as utils
from sklearn.linear_model import LinearRegression


animals = [
            'DO27',
            'DO28',
            'DO29',
]

datadir = r'C:\bonsai\data\Dammy'
dates = ['07/04/2021', 'now']
plot_colours = ['b','r','c','m','y','g']

trial_data = utils.merge_sessions(datadir,animals,'TrialData',dates)
trial_data = pd.concat(trial_data, sort=False, axis=0)

# plot day to day performance
stats_dict = dict()
for animal in animals:
    stats_dict[animal] = dict()
    for date in trial_data.loc[animal].index.unique():
        cols = ['Trials Done', 'Early Rate', 'Error Rate']
        data_day = utils.filter_df(trial_data.loc[animal,date],['b1'])
        trials_done_day = data_day.shape[0]
        try:
            early_rate = utils.filter_df(data_day,['a2']).shape[0]/trials_done_day
        except ZeroDivisionError:
            early_rate = 1
        try:
            error_rate = utils.filter_df(data_day, ['a0']).shape[0] / utils.filter_df(data_day, ['a3']).shape[0]
        except ZeroDivisionError:
            error_rate = 1
        stats_day = pd.DataFrame([[trials_done_day,early_rate,error_rate]],columns=cols)
        stats_dict[animal][date] = stats_day
fig,ax = plt.subplots(3)
for i, animal in enumerate(animals):
    for id, d in enumerate(stats_dict[animal].keys()):
        for f, feature in enumerate(stats_dict[animal][d]):
            ax[f].scatter(id, stats_dict[animal][d][feature], color=plot_colours[i],label=animal)
            if i == 0:
                ax[f].set_ylabel(f'{feature}')
                ax[f].set_xlabel('Session Number')

handles, labels = fig.gca().get_legend_handles_labels()
for axis in ax:
    by_label = OrderedDict(zip(labels, handles))
    axis.legend(by_label.values(), by_label.keys())

plots = utils.plot_performance(trial_data, np.arange(2,5.5,.5), animals, dates, plot_colours)
plot_early = utils.plot_metric_v_stimdur(trial_data,np.arange(2,5.5,.5),'Trial_Outcome',-1,animals,dates,
                                         plot_colours,['b1'], ytitle= 'Early rate')
plot_error_notones = utils.plot_metric_v_stimdur(trial_data,np.arange(2,5.5,.5),'Trial_Outcome',0,animals,dates,
                                                 plot_colours,['b1','a3','c1'],'Error rate without Tones', 'Error Rate no tones')
plot_error_tones = utils.plot_metric_v_stimdur(trial_data,np.arange(2,5.5,.5),'Trial_Outcome',0,animals,dates,
                                               plot_colours,['b1','a3','c0'], 'Error rate with Tones', 'Error Rate Tones')
plot_error = utils.plot_metric_v_stimdur(trial_data,np.arange(2,5.5,.5),'Trial_Outcome', 0,animals,dates,
                                         plot_colours,['b1','a3'],'Error rate: Stage 3', 'Error Rate')

# early_df = utils.filter_df(trial_data,['b1','a2'])
early_df = utils.filter_df(trial_data,['b1'])
early_df['Trial_End_datetime'] = np.array([datetime.strptime(trial_end[:-1], '%H:%M:%S.%f')
                                           for trial_end in early_df['Trial_End']])
early_df['Trial_Start_datetime'] = np.array([datetime.strptime(trial_end[:-1], '%H:%M:%S.%f')
                                             for trial_end in early_df['Trial_Start']])
early_df['StimEnd_datetime'] = np.array([(starttime+timedelta(seconds=stimdur)) for starttime,stimdur
                                         in zip(early_df['Trial_Start_datetime'],early_df['Stim1_Duration'])])

relearly = early_df['Trial_End_datetime'] - early_df['StimEnd_datetime']
early_df['End_vs_Stimdur'] = np.array([t.total_seconds() for t in relearly])
endvsstimdur_fig, endvsstimdur_ax = plt.subplots(1)
for i, animal in enumerate(animals):
    endvsstimdur_ax.hist(early_df.loc[animal]['End_vs_Stimdur'], edgecolor=plot_colours[i],label=animal, alpha=0.25,lw=.5)
endvsstimdur_fig.legend()
endvsstimdur_ax.set_xlabel('Trial end relative to Stimulus duration')
endvsstimdur_ax.axvline(1,color='k',linestyle='--')

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
