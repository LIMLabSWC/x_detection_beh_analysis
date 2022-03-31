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


class TDAnalysis:
    """
    Class for holding trialdata functions, dataframes, and plots
    """

    def __init__(self, tdatadir, animal_list, daterange,):
        self.trial_data = utils.merge_sessions(tdatadir,animal_list,'TrialData',daterange)
        self.trial_data = pd.concat(self.trial_data,sort=False,axis=0)
        self.animals = animal_list
        self.anon_animals = [f'Animal {i}' for i in range(len(self.animals))]  # give animals anon labels
        self.dates = list(self.trial_data.loc[self.animals].index.unique())

        # add datetime cols
        for col in self.trial_data.keys():
            if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1:
                if col.find('Wait') == -1 and col.find('dt') == -1:
                    utils.add_datetimecol(self.trial_data,col)

    def beh_daily(self, plot=False, filters=('b1',)) -> (dict, plt.subplots):
        stats_dict = dict()
        for animal in self.animals:
            stats_dict[animal] = dict()  # dictionary to hold daily stats
            for date in self.dates:
                statnames = ['Trials Done', 'NonViol Trials', 'Early Rate', 'Error Rate']
                data_day = utils.filter_df(self.trial_data.loc[animal, date], filters)
                trials_done_day = data_day.shape[0]
                correct_trials = utils.filter_df(data_day, ['a1']).shape[0]
                try:
                    early_rate = utils.filter_df(data_day, ['a2']).shape[0] / trials_done_day
                except ZeroDivisionError:
                    early_rate = 1
                try:
                    error_rate = utils.filter_df(data_day, ['a0']).shape[0] / utils.filter_df(data_day, ['a3']).shape[0]
                except ZeroDivisionError:
                    error_rate = 1
                stats_day = pd.DataFrame([[trials_done_day, correct_trials, early_rate, error_rate]], columns=statnames)
                stats_dict[animal][date] = stats_day

        fig, ax = plt.subplots(len(stats_day.columns), sharex=True)
        if plot:
            for i, animal in enumerate(self.animals):
                for id, d in enumerate(stats_dict[animal].keys()):
                    for f, feature in enumerate(stats_dict[animal][d]):
                        ax[f].scatter(id, stats_dict[animal][d][feature], marker='o', color=plot_colours[i],
                                      label=animal)
                        if i == 0:
                            ax[f].set_ylabel(f'{feature}')
                            # ax[f].set_xlabel('Session Number')
            handles, labels = fig.gca().get_legend_handles_labels()
            # for axis in ax:
            #     by_label = OrderedDict(zip(labels, handles))
            #     # axis.legend(by_label.values(), by_label.keys())

            by_label = OrderedDict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys())
            tick_dates = []
            for animal in self.animals:
                tick_dates.extend(stats_dict[animal].keys())
            utils.add_date_ticks(ax[-1],tick_dates)

        return stats_dict, (fig, ax)


if __name__ == '__main__':
    datadir = r'C:\bonsai\data'
    animals = [
                'DO42',
                'DO43',
                'DO37',
               ]

    dates = ['22/10/2021', '28/10/2021']  # start/end date for analysis
    plot_colours = plt.cm.jet(np.linspace(0,1,len(animals)))  # generate list of col ids for each animal
    td_obj = TDAnalysis(datadir,animals,dates)
    td_obj.day2day = td_obj.beh_daily(True)



