import matplotlib.colors
import pandas as pd
import numpy as np
import os
from copy import copy
import time
from datetime import datetime, timedelta
from collections import OrderedDict
from matplotlib import pyplot as plt
import matplotlib as mpl
import analysis_utils as utils
import pylab
from sklearn.linear_model import LinearRegression
from math import floor,ceil, isnan
from statistics import mean, stdev, median
from scipy import stats



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
                        error_rate = utils.filter_df(data_day, ['a0']).shape[0] / \
                                     utils.filter_df(data_day, ['a3']).shape[0]
                    except ZeroDivisionError:
                        error_rate = 1
                    stats_day = pd.DataFrame([[trials_done_day, correct_trials, early_rate, error_rate]],
                                             columns=statnames)
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
                            ax[f].scatter(self.dates, np.full_like(self.dates, 0), facecolors='none', edgecolor='none',
                                          s=15)
                        ax[f].scatter(d, stats_dict[animal][d][feature], marker='o', facecolors='none',
                                      edgecolor=f'C{i}',
                                      label=animal, s=30)
                        if i == 0:
                            ax[f].set_ylabel(f'{feature}')
                            # ax[f].set_xlabel('Session Number')
            handles, labels = fig.gca().get_legend_handles_labels()
            # for axis in ax:
            #     by_label = OrderedDict(zip(labels, handles))
            #     # axis.legend(by_label.values(), by_label.keys())

            for axis in ax:
                xmin, xmax = axis.get_xlim()
                ymin, ymax = axis.get_ylim()
                # axis.set_xlim(xmin - 0.01, xmax+0.01)
                # axis.set_ylim(ymin - 0.1, ymax+ymax*.1)

            by_label = OrderedDict(zip(labels, handles))
            ax[0].legend(by_label.values(), by_label.keys(), loc=1, fontsize='medium', ncol=len(self.animals))
            # fig.legend(by_label.values(), by_label.keys())
            fig.set_tight_layout(True)
            fig.set_size_inches(18, 12)
            tick_dates = []
            for animal in self.animals:
                tick_dates.extend(stats_dict[animal].keys())
            utils.add_date_ticks(ax[-1], tick_dates)

        return stats_dict, (fig, ax)

    def warmup(self):
        total_dict = {}
        dates_dict = {}
        for animal in self.animals:
            animal_dict = {}
            warmup = []
            main = []
            dates = []
            for date in self.dates:
                try:
                    data_day = self.trial_data.loc[animal, date]
                    warm_up_df = utils.filter_df(data_day, ['b0'])
                    main_df = utils.filter_df(data_day, ['b1'])
                    try:
                        correct_rate_wu = utils.filter_df(warm_up_df, ['a1']).shape[0] / warm_up_df.shape[0]
                        correct_rate_main = utils.filter_df(main_df, ['a1']).shape[0] / main_df.shape[0]
                    except ZeroDivisionError:
                        correct_rate_wu = np.nan
                        correct_rate_main = np.nan
                    warmup.append(correct_rate_wu)
                    main.append(correct_rate_main)
                    dates.append(date)
                except KeyError:
                    print(date, '- date missing')
            animal_dict['warm-up'] = warmup
            animal_dict['main'] = main
            dates_dict[animal] = dates
            total_dict[animal] = animal_dict


        fig, ax = plt.subplots(2,1, sharex= True, sharey=True)

        for i, animal in enumerate(self.animals):
            for f, feature in enumerate(total_dict[animal]):
                print(total_dict[animal][feature])
                ax[f].plot(total_dict[animal][feature], label=animal, marker='.')
                ax[f].set_title(feature)

        handles, labels = fig.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys())
        plt.xticks(range(0,len(dates)), dates)
        plt.xticks(rotation=60)

        main = []
        warmup = []

        for i in total_dict:
            main += total_dict[i]['main']
            warmup += total_dict[i]['warm-up']

        main = [i for i in main if not(isnan(i)) == True]
        warmup = [i for i in warmup if not(isnan(i)) == True]
        means = [mean(warmup), mean(main)]
        st_err = [stdev(warmup),stdev(main)]

        fig2, ax2 = plt.subplots()
        ax2.bar(['warmup', 'main'], means, color=['lightgreen', 'aquamarine'], yerr=st_err, align='center', capsize=10) #alpha=0.5, ecolor='black',
        ax2.set_ylabel('Correct rate')

        t_stat, p_value = stats.ttest_ind(main, warmup, equal_var=False)
        print(t_stat, p_value)
        plt.text(0,1,'t-stat: %s. p-value: %s.' % (t_stat, p_value))

        return fig, ax


    def pattern(self, stage, human=False,): #next step: make  applicable for humans
        stat_dict = {}
        plot_data = {}
        fig, ax = plt.subplots(2,2)
        fig2, ax2 = plt.subplots(2,2)
        fig.suptitle("Animal performance depending on pattern presentation")
        ax = ax.flatten()
        ax2 = ax2.flatten()
        c_map = mpl.cm.get_cmap('Paired')
        color = c_map(np.random.choice(range(0, 9), 4, replace=False))
        for n, animal in enumerate(self.animals):
            data = self.trial_data.loc[animal]
            daily_pattern_rate = []
            daily_none_rate = []
            date_list = []
            for date in self.dates:
                if (utils.filter_df(data,['b1'])['ToneTime_dt'] != datetime.strptime('00:00:00','%H:%M:%S')).any():
                    try:
                        pattern_trials = utils.filter_df(data.loc[date], ['e!0',stage]).shape[0]
                        success_pattern = utils.filter_df(data.loc[date], ['a1', 'e!0',stage]).shape[0]
                        none_trials = utils.filter_df(data.loc[date], ['e=0',stage]).shape[0]
                        success_none = utils.filter_df(data.loc[date], ['a1', 'e=0',stage]).shape[0]
                        try:
                            p_success_if = success_pattern / pattern_trials
                            p_success_none = success_none / none_trials
                            daily_pattern_rate.append(p_success_if)
                            daily_none_rate.append(p_success_none)
                            date_list.append(date)
                        except ZeroDivisionError:
                            print('Zero division: ', success_pattern)
                    except KeyError:
                        print('no data for this date: ', date)

            #none_stats = {'median': median(daily_none_rate), 'mean': mean(daily_none_rate), 'stdev': stdev(daily_none_rate)}
            #pattern_stats = {'median': median(daily_pattern_rate), 'mean': mean(daily_pattern_rate), 'stdev': stdev(daily_pattern_rate)}
            #stat_dict[animal]['none'] = none_stats
            #stat_dict[animal]['pattern'] = pattern_stats
            t_stat, p_value = stats.ttest_ind(daily_none_rate, daily_pattern_rate, equal_var= False)
            #stat_dict[animal]['ttest'] = t_stat, p_value
            print('t-stat: %.3f p-value: %.3f' % (t_stat, p_value))

            plot_data[animal] = {'none': daily_none_rate, 'pattern': daily_pattern_rate}
            df = pd.DataFrame(plot_data[animal])

            if len(daily_pattern_rate) > 1:  # animal has only done 1 day
                st_err = [stdev(daily_none_rate), stdev(daily_pattern_rate)]
                #stat_dict[animal]['st_err'] = st_err
            else:
                stats[animal]['st_err'] = 0
            xaxis = ['no pattern', 'pattern']
            #ax[n].title(animal)
            ax[n].boxplot(df, labels=xaxis, positions=range(len(df.columns)))
            ax[n].set_title(f'{animal} correct rate')
            ax[n].set_ylim((0,1))
            ax2[n].set_ylim((0, 1))
            for i,row in df.iterrows():
                y = row.to_list()
                ax[n].scatter(xaxis,y, c=np.array(color[n]))
                ax[n].margins(0.5, 0.1)
                ax[n].set(ylabel='Correct rate')
                ax[n].text(-0.2,0.2, 'p-value: %.3f' % (p_value))
            colors = [np.array(color[n]), 'grey']
            #plt.setp(ax2, xticks=range(0,len(date_list)), xticklabels=date_list )
            for i, column in enumerate(df):
                ax2[n].plot(date_list, df[column], c=colors[i], marker='.', label='column')
                ax2[n].legend(loc=0)
                ax2[n].set(ylabel='Correct rate', title = animal)
                plt.xticks(rotation=60) # why are not all rotated?



        return  plot_data




    def first_vs_second(self):
        dict = {}
        data = self.trial_data
        c_map = mpl.cm.get_cmap('tab20c',100)
        for animal in self.animals:
            first = []
            second = []
            for date in self.dates:
                try:
                    day_data = data.loc[animal, date].copy()
                    day_data['start diff'] = day_data['Trial_Start_dt'].diff()
                    starts_more1hr = np.where(day_data['start diff']>timedelta(0,3600))[0]
                    for ix,newsess in enumerate(starts_more1hr):  # int for index
                        if ix == 0:
                            day_data['Session'][:newsess] = chr(ord('a')+ix)
                            day_data['Session'][newsess:] = chr(ord('a')+ix+1)
                        else:
                            day_data['Session'][starts_more1hr[ix-1]:newsess] = chr(ord('a')+ix)
                            day_data['Session'][newsess:] = chr(ord('a')+ix +1)

                    n_trial_a = utils.filter_df(day_data,['sess_a']).shape[0]
                    n_trial_b = utils.filter_df(day_data, ['sess_b']).shape[0]
                    try:
                        first.append(utils.filter_df(day_data, ['sess_a', 'a1']).shape[0] / n_trial_a) #could add the rate to the data points in the plot by making another list
                        second.append(utils.filter_df(day_data, ['sess_b', 'a1']).shape[0] / n_trial_b)
                        # first.append(utils.filter_df(day_data, ['sess_a', 'a1']).shape[0])
                        # second.append(utils.filter_df(day_data, ['sess_b', 'a1']).shape[0])
                    except ZeroDivisionError or IndexError:
                        first.append(np.nan)
                        second.append(np.nan)
                except KeyError:
                    print(date, animal, '- date missing')
            dict[animal] = {'a': first, 'b': second}
        #plotting
        fig, axes = plt.subplots(2,2)
        x_names = ['a', 'b']
        for i, ax in enumerate(axes.flat):
            animal = self.animals[i]
            ax.set_title(animal)
            for d, date in enumerate(self.dates):
                color = c_map(np.random.choice(range(0, 100), 1))
                try:
                    list = [dict[animal]['a'][d], dict[animal]['b'][d]]
                    # print(list)
                    if list[0] == 0 or list[1] == 0: # or isnan(list[0]) == True
                        ax.scatter(x_names, list, c=c_map)
                    else:
                        ax.plot(x_names, list, marker='o', c= color)
                except IndexError:
                    print('no data for', date)
                #if isnan(dict[animal]['a'][d]) == True:

                #elif isnan(dict[animal]['a'][d]) == True:
                    #pass
        return fig, ax


    def time_since(self):
        # add column to dataframe that involves the time duration between patten presentation and x presentation
        data = utils.filter_df(self.trial_data, ['b1']['e!0']) # only need main-sess trials where pattern is presented
        data["pattern_to_x"] = np.nan # or just = ''? empty column with the time passed from pattern presentation to
        #for i,row in df.iterrows():
            #row[dat]
            # take the datetime of pattern presentation and the datetime of x presentation and subtract
            # make it into a useful time measure
            # replace nan in new column with value
        # then use ulils.filtr_df for filtering into success or failure
        # make two separate dataframes containing success and failure success_df, fail_df where all other data is removed
        # plt.bar([success, failure], [success_df.shape[0], fail_df.shape[0]]) maybe add percentages
        #may need these:
        #from sklearn.linear_model import LogisticRegression
        #from sklearn.model_selection import train_test_split






if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = [6.00, 4.00]
    # plt.rcParams["figure.autolayout"] = True

    datadir = r'C:\bonsai\data'
    #datadir = '/Users/hildelt/SWC_project/data'
    animals = [ 'DO45', 'DO46', 'DO47', 'DO48']
    #animals = [f'Human{i}' for i in range(28,32)]

    dates = ['14/06/2022', 'now']  # start/end date for analysis
    td_obj = TDAnalysis(datadir,animals,dates)
    #td_obj.day2day = td_obj.beh_daily(True)
    #td_obj.warmup_vs_main = td_obj.warmup()
    td_obj.pattern_response = td_obj.pattern(stage ='stage4') #needs a string defining the stage refering to utils.filtr_df() function
    #td_obj.session = td_obj.first_vs_second()
    #td_obj.after_pattern = td_obj.time_since()
    plt.show()

    # top rig:
    # np.random.choice([13,17,19],3, replace = False)
    # array([19, 17, 13])
    # Bottom rig
    # np.random.choice([13,17,19],3, replace = False)
    # array([13, 19, 17])
