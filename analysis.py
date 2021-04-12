import pandas as pd
import numpy as np
import os
from copy import copy
import time
from datetime import datetime, timedelta
from collections import OrderedDict
from matplotlib import pyplot as plt
import analysis_utils as utils


animals = [
            'DO27',
            'DO28',
            'DO29',
]

datadir = r'C:\bonsai\data\Dammy'
dates = ['07/04/2021', '12/04/2021']
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
plot_early = utils.plot_metric_v_stimdur(trial_data,np.arange(2,5.5,.5),'Trial_Outcome',-1,animals,dates, plot_colours,['b1'])
plot_early_notones = utils.plot_metric_v_stimdur(trial_data,np.arange(2,5.5,.5),'Trial_Outcome',-1,animals,dates, plot_colours,['b1','c0'])
plot_early_tones = utils.plot_metric_v_stimdur(trial_data,np.arange(2,5.5,.5),'Trial_Outcome',-1,animals,dates, plot_colours,['b1','c1'])
plot_error = utils.plot_metric_v_stimdur(trial_data,np.arange(2,5.5,.5),'Trial_Outcome',0,animals,dates, plot_colours,['b1','a3'])