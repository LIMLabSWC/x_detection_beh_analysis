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
        plt.style.use("seaborn-white")
        self.trial_data = utils.merge_sessions(tdatadir,animal_list,'TrialData',daterange)
        self.trial_data = pd.concat(self.trial_data,sort=False,axis=0)
        try:
            self.trial_data = self.trial_data.drop(columns=['RewardCross_Time','WhiteCross_Time'])
        except KeyError:
            pass
        self.animals = animal_list
        self.anon_animals = [f'Animal {i}' for i in range(len(self.animals))]  # give animals anon labels
        self.dates = sorted(self.trial_data.loc[self.animals].index.to_frame()['Date'].unique())

        # add datetime cols
        for col in self.trial_data.keys():
            if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1:
                if col.find('Wait') == -1 and col.find('dt') == -1:
                    utils.add_datetimecol(self.trial_data,col)

        # add reaction time col
        stim1_tdelta = self.trial_data['Stim1_Duration'].apply(lambda e: timedelta(0,e))
        self.trial_data['Reaction_time'] = self.trial_data['Trial_End_dt']-(self.trial_data['Trial_Start_dt'] +
                                                                            stim1_tdelta)

    def beh_daily(self, plot=False, filters=('b1',)) -> (dict, plt.subplots):
        stats_dict = dict()
        for animal in self.animals:
            stats_dict[animal] = dict()  # dictionary to hold daily stats
            for date in self.dates:
                try:
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
                except KeyError:
                    print('date missing')

        fig, ax = plt.subplots(len(stats_day.columns), sharex=True)
        if plot:
            for i, animal in enumerate(self.animals):
                for id, d in enumerate(sorted(stats_dict[animal].keys())):
                    print(d)
                    for f, feature in enumerate(stats_dict[animal][d]):
                        if i is 0:
                            ax[f].scatter(self.dates,np.full_like(self.dates,0),facecolors='none',edgecolor='none',s=15)
                        ax[f].scatter(d, stats_dict[animal][d][feature], marker='o', facecolors=f'C{i}',
                                      edgecolor=f'C{i}',
                                      label=animal, s=30)
                        if i == 0:
                            ax[f].set_ylabel(f'{feature}')
                            # ax[f].set_xlabel('Session Number')
            handles, labels = fig.gca().get_legend_handles_labels()
            # for axis in ax:
            #     by_label = OrderedDict(zip(labels, handles))
            #     # axis.legend(by_label.values(), by_label.keys())

            by_label = OrderedDict(zip(labels, handles))
            ax[0].legend(by_label.values(), by_label.keys(),fontsize='medium',ncol=len(self.animals),
                         bbox_to_anchor=(0.5, 1.01),loc='lower center')
            # fig.legend(by_label.values(), by_label.keys())
            fig.set_tight_layout(True)
            # fig.set_size_inches(18, 12)
            fig.set_size_inches(9, 12)
            for axis in ax:
                xmin, xmax = axis.get_xlim()
                ymin, ymax = axis.get_ylim()
                if ymax > 1:
                    axis.set_yticks(np.arange(0,ymax,50))
                    axis.set_yticklabels(np.arange(0,ymax,50).astype(int))
                elif ymax <= 1:
                    axis.set_yticks(np.arange(0,1,.25))
                    axis.set_yticklabels(np.arange(0,1,.25))
                # axis.set_xlim(xmin - 0.01, xmax+0.01)
                # axis.set_ylim(ymin - 0.1, ymax+ymax*.1)
            tick_dates = []
            for animal in self.animals:
                tick_dates.extend(stats_dict[animal].keys())
            utils.add_date_ticks(ax[-1],tick_dates)

        return stats_dict, (fig, ax)


if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = [8.00, 6.00]
    # plt.rcParams["figure.autolayout"] = True

    datadir = r'C:\bonsai\data'
    animals = [
                'DO45',
                'DO46',
                'DO47',
                'DO48'
               ]

    dates = ['13/06/2022', 'now']  # start/end date for analysis
    td_obj = TDAnalysis(datadir,animals,dates)
    td_obj.day2day = td_obj.beh_daily(True)
