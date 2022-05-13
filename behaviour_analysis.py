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
from scipy.stats import ttest_ind

class TDAnalysis:
    """
    Class for holding trialdata functions, dataframes, and plots
    """

    def __init__(self, tdatadir, animal_list, daterange,):
        self.trial_data = utils.merge_sessions(tdatadir,animal_list,'TrialData',daterange)
        self.trial_data = pd.concat(self.trial_data,sort=False,axis=0)
        try:
            self.trial_data = self.trial_data.drop(columns=['RewardCross_Time','WhiteCross_Time'])
        except KeyError:
            pass
        self.animals = animal_list
        self.anon_animals = [f'Animal {i}' for i in range(len(self.animals))]  # give animals anon labels
        self.dates = self.trial_data.loc[self.animals].index.to_frame()['Date'].unique()

        # add datetime cols
        for col in self.trial_data.keys():
            if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1:
                if col.find('Wait') == -1 and col.find('dt') == -1:
                    utils.add_datetimecol(self.trial_data,col)

    def beh_daily(self, plot=False, filters =('b1',)) -> (dict, plt.subplots):
        stats_dict = dict()
        for animal in self.animals:
            stats_dict[animal] = dict()  # dictionary to hold daily stats
            for date in self.trial_data.loc[animal].index.unique():
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
            fig, ax = plt.subplots(len(stats_day.columns), sharex=True)
            for i, animal in enumerate(self.animals):
                for id, d in enumerate(stats_dict[animal].keys()):
                    for f, feature in enumerate(stats_dict[animal][d]):
                        ax[f].scatter(id, stats_dict[animal][d][feature], marker='o', color=plot_colours[i],
                                      label=animal, s=2)
                        if i == 0:
                            ax[f].set_ylabel(f'{feature}')
                            #ax[f].set_xlabel('Session Number')
            handles, labels = fig.gca().get_legend_handles_labels()
            #for axis in ax:
                # by_label = OrderedDict(zip(labels, handles))
                # axis.legend(by_label.values(), by_label.keys())


            #for axis in ax:
             #   xmin, xmax = axis.get_xlim()
             #   ymin, ymax = axis.get_ylim()
             #   axis.set_xlim(xmin - 0.01, xmax+0.01)
             #   axis.set_ylim(ymin - 0.1, ymax+ymax*.1)
            #Also works?
            for axis in ax:
                axis.margins(0,0.1)


            by_label = OrderedDict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys())
            tick_dates = []
            for animal in self.animals:
                tick_dates.extend(stats_dict[animal].keys())
            utils.add_date_ticks(ax[-1],tick_dates)

        return stats_dict, (fig, ax)



#Attempt to plot performance depending on warm up or not
    def new_version(self, plot=False):
        total_dict = {}
        mean_dict = {}
        data = self.trial_data
        for animal in self.animals:
            animal_data = data.loc[animal]
            warmup = []
            main = []
            for date in animal_data.index.unique():
                try:
                    correcrate_warmup_day = utils.filter_df(animal_data.loc[date],['b0', 'a1']).shape[0] / utils.filter_df(animal_data.loc[date],['b0']).shape[0]
                    warmup.append(correcrate_warmup_day)
                except ZeroDivisionError:
                    print(f'no warm up trials for session {animal}:{date}')
                    warmup.append(np.nan)
                try:
                    correctrate_main_day =  utils.filter_df(animal_data.loc[date],['b1', 'a1']).shape[0] / utils.filter_df(animal_data.loc[date],['b1']).shape[0]
                    main.append(correctrate_main_day)
                except ZeroDivisionError:
                    print(f'no non_warm up trials for session {animal}:{date}')
                    main.append(np.nan)

            total_dict[animal] = {'warmup': warmup, 'main': main}
            mean_wu = sum(warmup) / len(warmup)
            mean_m = sum(main) / len(main)
            mean_dict[animal] = [mean_wu, mean_m]

            # now have a list of correct rate for warmup and main per day for each animal
            # Testng t-testing
        #Plotting:
        daily_fig, daily_ax = plt.subplots(2,1, sharex=True, num = 'Correct rate session type')
        for i, animal in enumerate(total_dict):
            for p, part in enumerate(total_dict[animal]):
                daily_ax[p].plot(total_dict[animal][part], label=animal, marker = '.', color=plot_colours[i])

        handles, labels = daily_fig.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        daily_ax[0].legend(by_label.values(), by_label.keys())
        daily_ax[0].set_ylabel("Warm up phase")
        daily_ax[1].set_ylabel("Main phase")
        for i, axis in enumerate(daily_ax):
            axis.margins(0, 0.1)

        overall_fig, overall_ax = plt.subplots(2,2)
        count = 0
        for i in range(0,2):
            for k in range(0,2):
                animal = self.animals[count]
                print(mean_dict[animal])
                overall_ax[i, k].bar(['Warm up', 'Main'], mean_dict[animal], label=animal,
                             color=['seagreen','aliceblue'])
                #overall_ax[i, k].legend(by_label.values(), by_label.keys())
                overall_ax[i, k].set_ylabel('Correct rate')
                count += 1
        return(daily_fig, daily_ax)



    def performance_if_pattern(self, plot = False):
        data = self.trial_data
        pattern_df = utils.filter_df(data,['e!0'])
        no_pattern_df = utils.filter_df(data,['e=0']) # are thee mixed up?
        rate_dict = {}
        pattern_list = []
        no_pattern_list = []
        for i, animal in enumerate(self.animals):
            if_pattern_rate = utils.filter_df(pattern_df.loc[animal], ['a1']).shape[0]/pattern_df.loc[animal].shape[0]
            no_pattern_rate = utils.filter_df(no_pattern_df.loc[animal], ['a1']).shape[0]/no_pattern_df.loc[animal].shape[0]
            rate_dict[animal] = [if_pattern_rate, no_pattern_rate]
            pattern_list.append(if_pattern_rate)
            no_pattern_list.append(no_pattern_rate)
        mean_pattern = np.mean(pattern_list)
        mean_no_pattern = np.mean(no_pattern_list)
        means =[mean_pattern,mean_no_pattern]
        std_pattern = np.std(pattern_list)
        std_no_pattern = np.std(no_pattern_list)
        error = [std_pattern, std_no_pattern]

        simple_fig, simple_ax = plt.subplots()
        simple_ax.bar(['Pattern', 'No pattern'], means, yerr=error, capsize=10,color= ['aquamarine','seagreen'])

        tStat, pValue = ttest_ind(pattern_list, no_pattern_list)
        text = 'P-Value:{0:.3g}\nT-Statistic:{1:.3g}'.format(pValue, tStat)
        simple_ax.text(0.1,0.9,text)
        plt.ylim(0,1)
        plt.title('Mean correct rate patter vs. no pattern')

        nplotcols = 3
        fig, ax = plt.subplots(ceil(len(self.animals)/nplotcols),nplotcols, sharey=True)
        handles, labels = fig.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))#Figure out why I dont need, undestand what, should be simple
        for i,(axis,animal) in enumerate(zip(ax.flatten(),self.animals)):
            axis.bar(['Pattern', 'No pattern'], rate_dict[animal], label=animal, color=['aquamarine','seagreen'])
            axis.set_title(animal,fontsize=9)
            axis.set_ylabel('Correct rate')
        fig.set_size_inches(10,10)
        return(rate_dict)



if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = [6.00, 4.00]
    # plt.rcParams["figure.autolayout"] = True

    datadir = r'C:\bonsai\data'
    animals = [ 'DO45', 'DO46', 'DO47', 'DO48']
    #animals = [f'Human{i}' for i in range(16,26)]

    dates = ['08/02/2022', 'now']  # start/end date for analysis
    plot_colours = plt.cm.magma(np.linspace(0,1,len(animals)))  # generate list of col ids for each animal
    td_obj = TDAnalysis(datadir,animals,dates)
    td_obj.day2day = td_obj.beh_daily(True)
    td_obj.warmup_vs_main = td_obj.new_version()
    #td_obj.pattern_perf = td_obj.performance_if_pattern()


