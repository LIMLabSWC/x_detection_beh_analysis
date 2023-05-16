import pandas as pd
import numpy as np
import os
from copy import deepcopy as copy
import time
from datetime import datetime, timedelta
from collections import OrderedDict
from matplotlib import pyplot as plt
import analysis_utils as utils
import pylab
from sklearn.linear_model import LinearRegression
from math import floor,ceil
import copy
import scipy.signal
import psychophysicsUtils


class TDAnalysis:
    """
    Class for holding trialdata functions, dataframes, and plots
    """

    def __init__(self, tdatadir, animal_list, daterange,):
        plt.style.use("seaborn-white")
        self.trialData = utils.merge_sessions(tdatadir, animal_list, 'TrialData', daterange)
        self.trialData = pd.concat(self.trialData, sort=False, axis=0)
        try:
            self.trialData = self.trialData.drop(columns=['RewardCross_Time', 'WhiteCross_Time'])
        except KeyError:
            pass
        self.animals = animal_list
        self.anon_animals = [f'Animal {i}' for i in range(len(self.animals))]  # give animals anon labels
        self.dates = sorted(self.trialData.loc[self.animals].index.to_frame()['Date'].unique())

        # add datetime cols
        for col in self.trialData.keys():
            if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1:
                if col.find('Wait') == -1 and col.find('dt') == -1 and col.find('Lick_Times') == -1:
                    utils.add_datetimecol(self.trialData, col)

        # add reaction time col
        # stim1_tdelta = self.trialData['Stim1_Duration'].apply(lambda e: timedelta(0, e))
        # self.trialData['Reaction_time'] = self.trialData['Trial_End_dt'] - (self.trialData['Trial_Start_dt'] +
        #                                                                     stim1_tdelta)

    def beh_daily(self, plot=False, filters=('b1',)) -> (dict, plt.subplots):
        stats_dict = dict()
        for animal in self.animals:
            stats_dict[animal] = dict()  # dictionary to hold daily stats
            for date in self.dates:
                try:
                    statnames = ['Trials Done', 'NonViol Trials', 'Early Rate', 'Error Rate']
                    data_day = utils.filter_df(self.trialData.loc[animal, date], filters)
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

    def add_data_dict(self):
        self.data = {}
        for idx in self.trialData.index.unique():
            sess_name = '_'.join(idx)
            self.data[sess_name] = psychophysicsUtils.pupilDataClass
            self.data[sess_name].trialData = copy.deepcopy(self.trialData.loc[idx])

    def get_aligned_events(self, eventname, harp_event, window, timeshift=0.0, harp_event_name='Lick',
                           animals=None,animals_omit=None,dates=None,dates_omit=None,plot=True,lfilt=None,
                           byoutcome_flag=False,outcome2filt=None,extra_filts=None,plotcol=None):

        if animals is None:
            animals = self.animals
        if dates is None:
            dates = self.dates
        if dates_omit:
            [dates.remove(e) for e in dates_omit]
        if animals_omit:
            [animals.remove(e) for e in animals_omit]

        fig,axes = plt.subplots(1,len(animals),figsize=(20,10))

        if plot:
            all_sess_eventz_list = []
            for i, (ax,animal) in enumerate(zip(axes,animals)):
                animal_cnt = 0
                for d,date in enumerate(dates):
                    sess_name = f'{animal}_{date}'
                    # sess_mat = copy(utils.align_nonpuil(self.data[sess_name].trialData[eventname],
                    #                                self.data[sess_name].harpmatrices[harp_event], window,timeshift))
                    if date not in self.trialData.index.get_level_values('Date'):
                        # print('Date not in trialdata, skipping')
                        continue

                    td2use = self.trialData.loc[animal,date,:]
                    if outcome2filt:
                        if extra_filts:
                            filts2use = outcome2filt+extra_filts
                        else:
                            filts2use = outcome2filt
                        td2use = utils.filter_df(td2use,filts2use)
                    try:sess_mat = copy.deepcopy(utils.align_nonpuil(td2use[eventname],
                                                                 self.harpmatrices[sess_name][harp_event], window,
                                                                 self.trialData.loc[(animal,date)]['Offset'],
                                                                 timeshift))
                    except KeyError:continue
                    sess_mat
                    byoutcome_ser = self.trialData.loc[(animal,date)]['Trial_Outcome']
                    fs = 0.001
                    all_sess_eventz =pd.DataFrame(np.full((len(sess_mat),int((window[1]-window[0])/fs)),0.0))
                    all_sess_eventz.columns = np.linspace(window[0],window[1],all_sess_eventz.shape[1]).round(3)
                    # axes[i].set_axisbelow(True)
                    # axes[i].yaxis.grid(color='gray', linestyle='dashed',which='both')
                    for e, event in enumerate(sess_mat):
                        axes[i].axhline(animal_cnt-e,c='k',linewidth=.25,alpha=0.25,)
                        epoch_events = np.full(int((window[1]-window[0])/fs)+1,0.0)

                        # print(list(sess_mat.values())[0])
                        event=copy.deepcopy(event)
                        eventz = sess_mat[event].round(3)

                        epoch_events[((sess_mat[event]-window[0])/fs).astype(int)] = 1
                        all_sess_eventz.loc[e,eventz] = 1
                        epoch_events = all_sess_eventz.loc[e,:].to_numpy()

                        if lfilt:
                            epoch_events = utils.butter_filter(epoch_events, lfilt, 1 / fs, filtype='low')
                            # b,a = s
                        epoch_events[epoch_events==0] = np.nan
                        if byoutcome_flag:
                            if plotcol is None:
                                plotcol = int(td2use["Trial_Outcome"][e])
                            axes[i].scatter(all_sess_eventz.columns, epoch_events*(animal_cnt-e),
                                            c=f'C{plotcol}', marker='x',s=3,alpha=1,linewidth=.5)
                        else:
                            axes[i].scatter(all_sess_eventz.columns, epoch_events*(animal_cnt-e), c=f'C{d}', marker='.')
                        axes[i].axvline(0, ls='--', c='k')
                        # axes[i].axvline(0.5, ls='--', c='grey')
                    # ax.axhline(animal_cnt+20,ls='-',c='k')
                    animal_cnt -= len(sess_mat)
                    if outcome2filt:
                        condname=outcome2filt[0].replace('a0','Non Rewarded')
                        condname=condname.replace('a1','Rewarded')
                    else:
                        condname = 'all'
                    axes[i].set_title(f'{harp_event_name} aligned to {eventname.replace("dt","").replace("_", " ").replace("Gap","X")}\n'
                                      f'{animal}, {condname} trials',size=10)
                    axes[i].set_yticks([])

                    all_sess_eventz.index = td2use.index
                    all_sess_eventz_list.append(all_sess_eventz)
        return fig,axes,pd.concat(all_sess_eventz_list,axis=0)

if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = [8.00, 6.00]
    # plt.rcParams["figure.autolayout"] = True

    datadir = r'C:\bonsai\data'
    # animals = [
    #             'DO54',
    #             'DO55',
    #             'DO56',
    #             'DO57'
    #            ]
    animals = [f'DO{i}' for i in range(50,54)]
    dates = ['21/01/2022', '31/01/2023']  # start/end date for analysis
    # dates = ['02/11/2022', '08/11/2022']  # start/end date for analysis
    td_obj = TDAnalysis(datadir,animals,dates)


    # event raster
    td_obj.add_data_dict()
    # harpmatrices = utils.get_event_matrix(td_obj,td_obj.data,r'W:\mouse_pupillometry\mouse_hf\harpbins',)
    # td_obj.harpmatrices = utils.get_event_matrix(td_obj,td_obj.data,r'W:\mouse_pupillometry\mouse_hf\harpbins',)
    td_obj.harpmatrices = utils.get_event_matrix(td_obj,td_obj.data,r'W:\mouse_pupillometry\mouseprobreward_hf\harpbins',)


    # for mat_name in harpmatrices:
    #     td_obj.data[mat_name].harpmatrices = copy.deepcopy(dict())
    #     td_obj.data[mat_name].harpmatrices = copy.deepcopy(harpmatrices[mat_name])
    plt.ion()
    td_obj.lickrasters_whitenoise = td_obj.get_aligned_events('Gap_Time_dt',2,(-5.0,5.0),)
    # td_obj.lickrasters_trialstart = td_obj.get_aligned_events('Trial_Start_dt',0,(-1.0,10.0),)

    # dates2plot = ['221103','221104','221107','221108']
    dates2plot = ['230121', '230126', '230131']
    fig,ax = plt.subplots(len(td_obj.animals),len(dates2plot),sharex='all',sharey='all')
    for di,d in enumerate(dates2plot):
        for a,animal in enumerate(td_obj.animals):
            ax[a][di].plot(td_obj.trialData.loc[animal,d]['Early_Licks'].to_numpy(),label=animal,c=f'C{a}')
            ax[a][di].legend(loc=1)
            ax[a][di].set_ylim((0,25))
            ax[a][di].set_yticks([0,10,20])
        ax[a][di].set_xlabel('Trial number',fontsize=11)
    fig.text(0.075,0.5,'Number of early licks',rotation=90,ha='left', va='center',fontsize=11)
    fig.text(0.5,0.95,f'Change in early licks over trials for {" ".join(dates2plot)}',ha='center', va='top',fontsize=11)
    fig.savefig(rf'W:\mouse_pupillometry\figures\hf_licks\n_early_licks_{"_".join(dates2plot)}_new.svg',bbox_inches='tight')
