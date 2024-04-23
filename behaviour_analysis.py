import pandas as pd
import numpy as np
from copy import deepcopy as copy
from collections import OrderedDict
# mpl.use('TkAgg')
from matplotlib import pyplot as plt

import align_functions
import analysis_utils as utils
import psychophysicsUtils
import yaml
import argparse
from pathlib import Path
import platform
from tqdm import tqdm


class TDAnalysis:
    """
    Class for holding trialdata functions, dataframes, and plots
    """

    def __init__(self, tdatadir, animal_list, daterange,):
        self.trialData = utils.merge_sessions(tdatadir, animal_list, 'TrialData', daterange)
        self.trialData = pd.concat(self.trialData, sort=False, axis=0)
        try:
            self.trialData = self.trialData.drop(columns=['RewardCross_Time', 'WhiteCross_Time'])
        except KeyError:
            pass
        self.animals = animal_list
        self.anon_animals = [f'Animal {i}' for i in range(len(self.animals))]  # give animals anon labels
        self.dates = sorted(self.trialData.loc[self.animals].index.to_frame()['date'].unique())

        # add datetime cols
        self.add_dt_cols2sesstd(['Trial_Start','Trial_End','Time','ToneTime','Gap_Time','RewardTone_Time'])
        self.trialData.set_index('Trial_Start_dt', append=True, inplace=True, drop=False)

            #
            # if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1:
            #     if col.find('Wait') == -1 and col.find('dt') == -1 and col.find('Lick_Times') == -1:
            #         utils.add_datetimecol(self.trialData, col)

        # add reaction time col
        self.trialData['Reaction_time'] = (self.trialData['Trial_End_dt'] -
                                           self.trialData['Gap_Time_dt']).dt.total_seconds()
    def add_dt_cols2sesstd(self,column_names):
        # for sess in self.data:
        #     sess_td = self.data[sess].trialData
        for col in column_names:
            utils.add_datetimecol(self.trialData,col)

    def beh_daily(self, plot=False, filters=('b1',)) -> (dict, plt.subplots):
        stats_dict = dict()
        for animal in self.animals:
            stats_dict[animal] = dict()  # dictionary to hold daily stats
            for date in self.dates:
                try:
                    statnames = ['Trials Done', 'Correct Trials', 'Error Rate']  #'Early Rate'
                    data_day = align_functions.filter_df(self.trialData.loc[[animal], [date],:], filters)
                    trials_done_day = data_day.shape[0]
                    correct_trials = align_functions.filter_df(data_day, ['a1']).shape[0]
                    try:
                        early_rate = align_functions.filter_df(data_day, ['a2']).shape[0] / trials_done_day
                    except ZeroDivisionError:
                        early_rate = 1
                    try:
                        error_rate = align_functions.filter_df(data_day, ['a0']).shape[0] / align_functions.filter_df(data_day, ['a3']).shape[0]
                    except ZeroDivisionError:
                        error_rate = 1
                    stats_day = pd.DataFrame([[trials_done_day, correct_trials, error_rate]], columns=statnames)
                    stats_dict[animal][date] = stats_day
                except KeyError:
                    print('date missing')

        fig, ax = plt.subplots(len(stats_day.columns), sharex=True)
        if plot:
            for i, animal in enumerate(self.animals):
                for id, d in enumerate(sorted(stats_dict[animal].keys())):
                    # print(d)
                    for f, feature in enumerate(stats_dict[animal][d]):
                        if i == 0:
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
            self.data[sess_name].trialData = copy(self.trialData.loc[idx])

    def scatter_trial_metric_bysess(self,metric, dates=None, conditions=None, by_animal_flag=True, pointcloud_flag=True):

        if isinstance(dates, str):
            dates = [dates]
        elif not dates:
            dates = self.dates
        metric_series = align_functions.filter_df(self.trialData, ['a1']).get(metric, None)
        # if any(reaction_times):
        #     print('No "Reaction_time" column, Returning None')
        #     return None
        assert len(metric_series.index.names) == 3, 'missing index level'

        trial_metric_plot = plt.subplots()
        plot_markers = ['1','2','3','4','+']
        for date_i, date in enumerate(dates):
            sessions4dates = metric_series.loc[:,[date],:]
            if by_animal_flag:
                animals4date = sorted(sessions4dates.index.get_level_values('name').unique())
                # animals4date = ['DO69']
                metric_data_df = []
                for animal in animals4date:
                    metric_data_df.append(sessions4dates.loc[animal,:].to_numpy())
            else:
                metric_data_df = [sessions4dates.to_numpy()]
            if pointcloud_flag:
                for sess_i, session_metric_series in enumerate(metric_data_df):
                    x_series = np.random.uniform(low=date_i-0.1, high=date_i+0.1,size=len(session_metric_series))
                    trial_metric_plot[1].scatter(x_series,session_metric_series,c=f'C{sess_i}',
                                       alpha=0.025,)
            for sess_i, session_metric_series in enumerate(metric_data_df):
                metric_low,metric_high = np.abs(np.quantile(session_metric_series,[0.05,0.95],method='normal_unbiased')-np.median(session_metric_series))
                trial_metric_plot[1].errorbar(date_i, np.median(session_metric_series), np.vstack([metric_low,metric_high]),c=f'C{sess_i}',
                                    capsize=10.0,fmt='o',mec='k',mfc=f'C{sess_i}',mew=1,lw=1)



        return trial_metric_plot

    def ntrials_since_last(self,column='Tone_Position'):
        td_df = self.trialData
        column_data = td_df[column]



if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = [8.00, 6.00]
    # plt.rcParams["figure.autolayout"] = True

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', default=Path('config', 'mouse_fam_old_conf_unix.yaml'))
    args = parser.parse_args()

    # animals = [
    #             'DO64',
    #             'DO69',
    #             'DO70'
    #            ]
    # animals = [f'DO{i}' for i in range(54,59)]
    # dates = ['07/06/2023', '09/08/2023']  # start/end date for analysis
    # dates = ['02/02/2023', '03/03/2023']  # start/end date for analysis

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    datadir = config[f'tdatadir_{platform.system().lower()}']

    animals = config['animals2process']
    dates = sorted(config['dates2process'])

    td_obj = TDAnalysis(datadir,animals,(dates[0],dates[-1]))

    plots = utils.plot_performance(td_obj.trialData, np.arange(7,13,1), animals, dates,['b','r','y','purple','cyan'])

    # reaction_time_plot = td_obj.scatter_trial_metric_bysess('Reaction_time',dates=['230607','230608','230609',
    #                                                                                '230717','230718','230719',
    #                                                                                '230720','230721','230724','230725'])
    reaction_time_plot = td_obj.scatter_trial_metric_bysess('Reaction_time',dates=dates)
    reaction_time_plot[1].set_ylabel('Reaction time (s)')
    reaction_time_plot[1].set_ylim(0.1, 1)
    reaction_time_plot[1].set_xticks(np.arange(len(dates)))
    reaction_time_plot[1].set_xticklabels(dates,rotation=40)
    reaction_time_plot[0].show()
    reaction_time_plot[1].set_title('Reaction time to X across sessions')
    reaction_time_plot[0].set_constrained_layout('constrained')
    reaction_time_plot[0].savefig(r'W:\mouse_pupillometry\figures\final_figs\muscimol_rt_all_sess.svg')

    early_licks_plot = td_obj.scatter_trial_metric_bysess('Early_Licks')
    early_licks_plot[0].show()

    eg_sess_scatterplot = plt.subplots()
    eg_sess = td_obj.trialData.loc['DO64','230718',:]

    eg_sess_scatterplot[1].scatter(np.arange(eg_sess.shape[0]),eg_sess['Tone_Position']==0,marker='o',facecolor='white',
                                   edgecolor='grey')
    eg_sess_scatterplot[1].plot(np.arange(eg_sess.shape[0]),(eg_sess['Tone_Position']==0).rolling(10).mean(),
                                color='k',ls='--')
    eg_sess_scatterplot[0].set_size_inches(20,9)
    for r_i, rate in enumerate([0.1,0.5,0.9]):
        eg_sess_scatterplot[1].fill_between(np.arange(eg_sess.shape[0]), 0, 1,
                                            where=eg_sess['PatternPresentation_Rate'].to_numpy() == round(1.0-rate,1), color=f'C{r_i}',
                                            alpha=0.1,label=f'rate = {rate}')
    eg_sess_scatterplot[1].set_title('Pattern presentation rate for example session')
    eg_sess_scatterplot[1].set_ylabel('Pattern Presentation Rate (sliding window size 10)')
    eg_sess_scatterplot[1].set_xlabel('Trial Number')
    eg_sess_scatterplot[0].set_constrained_layout('constrained')
    eg_sess_scatterplot[1].legend(loc=0,ncols=3)
    eg_sess_scatterplot[0].show()

    td_obj.trialData['rounded_stim'] = np.full_like(td_obj.trialData.index, np.nan)
    td2use = td_obj.trialData[td_obj.trialData['Tone_Position'] == 0]
    td2use.loc[:,'rounded_stim'] = (td2use['ToneTime_dt']-td2use['Trial_Start_dt']).dt.total_seconds().round()
    stim_dur_perf = [[td2use.loc[sess][td2use.loc[sess,'rounded_stim']==dur]['Trial_Outcome'].mean()
                     for sess in tqdm(td2use.index.droplevel(2).unique())]
                     for dur in sorted(td2use['rounded_stim'].unique()) if dur != np.nan]
    stim_dur_boxplot = plt.subplots(figsize=(2.5,2.5))
    stim_dur_boxplot[1].boxplot(stim_dur_perf[:4],labels=sorted(td2use['rounded_stim'].unique())[:4])
    stim_dur_boxplot[1].tick_params(which='major',axis='both',labelsize=14)
    stim_dur_boxplot[1].set_ylim(0.5,1.1)
    stim_dur_boxplot[0].set_constrained_layout('constrained')
    stim_dur_boxplot[0].show()
    stim_dur_boxplot[0].savefig(r'X:\Dammy\final_figures\stim_dur_boxplot.svg')

    # # event raster
    # td_obj.add_data_dict()
    # # harpmatrices = utils.get_event_matrix(td_obj,td_obj.data,r'W:\mouse_pupillometry\mouse_hf\harpbins',)
    # # td_obj.harpmatrices = utils.get_event_matrix(td_obj,td_obj.data,r'W:\mouse_pupillometry\mouse_hf\harpbins',)
    # # td_obj.harpmatrices = utils.get_event_matrix(td_obj,td_obj.data,r'W:\mouse_pupillometry\mouseprobreward_hf\harpbins',)
    #
    #
    # # for mat_name in harpmatrices:
    # #     td_obj.data[mat_name].harpmatrices = copy.deepcopy(dict())
    # #     td_obj.data[mat_name].harpmatrices = copy.deepcopy(harpmatrices[mat_name])
    # plt.ion()
    # td_obj.lickrasters_whitenoise = td_obj.get_aligned_events('Gap_Time_dt',2,(-5.0,5.0),)
    # # td_obj.lickrasters_trialstart = td_obj.get_aligned_events('Trial_Start_dt',0,(-1.0,10.0),)
    #
    # # dates2plot = ['221103','221104','221107','221108']
    # dates2plot = ['230607', '230608', '230609']
    # fig,ax = plt.subplots(len(td_obj.animals),len(dates2plot),sharex='all',sharey='all')
    # for di,d in enumerate(dates2plot):
    #     for a,animal in enumerate(td_obj.animals):
    #         ax[a][di].plot(td_obj.trialData.loc[animal,d]['Early_Licks'].to_numpy(),label=animal,c=f'C{a}')
    #         ax[a][di].legend(loc=1)
    #         ax[a][di].set_ylim((0,25))
    #         ax[a][di].set_yticks([0,10,20])
    #     ax[a][di].set_xlabel('Trial number',fontsize=11)
    # fig.text(0.075,0.5,'Number of early licks',rotation=90,ha='left', va='center',fontsize=11)
    # fig.text(0.5,0.95,f'Change in early licks over trials for {" ".join(dates2plot)}',ha='center', va='top',fontsize=11)
    # fig.savefig(rf'W:\mouse_pupillometry\figures\hf_licks\n_early_licks_{"_".join(dates2plot)}_new.svg',bbox_inches='tight')
