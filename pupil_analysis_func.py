import matplotlib.pyplot as plt

import align_functions
import analysis_utils as utils
from analysis_utils import plot_eventaligned
from align_functions import align_wrapper
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime, timedelta
from math import ceil
import matplotlib
import matplotlib.colors
import statsmodels.api as sm
import ruptures as rpt
from copy import deepcopy as copy
from itertools import product
from functools import partial
from scipy.signal import find_peaks
from pathlib import Path
from tqdm import tqdm


# code for analysing pupil self.data


def batch_analysis(dataclass,dataclass_dict,stages,column,shifts,events,labels,pmetric='dlc_radii_a_zscored',use4pupil=False,
                   filter_df=True,pdr=False,plot=False,baseline=True,extra_filts=None,sep_cond_cntrl_flag=False,
                   cond_name=None):
    """
    Perform batch analysis of pupil data.

    Args:
    - dataclass: An object containing the pupil data.
    - dataclass_dict: A dictionary to store the results of the analysis.
    - stages: A list of stages to analyze.
    - column: The column to use for aligning the data.
    - shifts: A list of shifts to apply to the data.
    - events: A list of events to filter the data by.
    - labels: A list of labels to use in plots.
    - pmetric: The metric to use for analyzing the pupil data.
    - use4pupil: A flag indicating whether to use 4-pupil data.
    - filter_df: A flag indicating whether to filter the dataframe.
    - pdr: A flag indicating whether to use partial duration regression.
    - plot: A flag indicating whether to generate plots.
    - baseline: A flag indicating whether to use baseline correction.

    Returns:
    - A list of keys for the analyzed conditions.
    """
    cond_keys = []
    if not extra_filts:
        extra_filts = []
    for s in stages:
        for shift in shifts:
            if cond_name is None:
                cond_name = f'stage{s}_{events}_{column}_{shift[0]}'
            cond_keys.append(cond_name)
            event_filters = []

            if filter_df:
                for e in events:
                    if any([ee[0] in ['d','p'] for ee in e]):
                        event_filters.append(['e!0', f'stage{s}']+e+extra_filts)
                    elif 'none' in e:
                        none_filt = (['c1', f'stage{s}']+extra_filts)
                        if 'tones4' in none_filt:
                            none_filt.remove('tones4')
                        if 'e!0' in none_filt:
                            none_filt.remove('e!0')
                        event_filters.append(none_filt)
                    else:
                        event_filters.append(e)
            else:
                event_filters.append(event_filters)
            for e in event_filters:
                if 'none' in e or 'e=0' in e or 'c1' in e:
                    if 'tones4' in e:
                        e.remove('tones4')
                    if 'e!0' in e:
                        e.remove('e!0')
                    print(f'none filts = {e}')
                if 's1' in e:
                    print(e)
            dataclass_dict[cond_name] = dataclass.get_aligned(event_filters,event_shift=[shift[0]] * len(event_filters), align_col=column,
                                                                 event=shift[1], xlabel=f'Time since {shift[1]}', pdr=pdr,
                                                                 plotlabels=labels[:len(event_filters)],
                                                                 use4pupil=use4pupil,pmetric=pmetric,plot=plot, baseline=baseline,
                                                                 sep_cond_cntrl_flag=sep_cond_cntrl_flag)

            if plot:
                dataclass_dict[cond_name][0].canvas.manager.set_window_title(cond_name)
                fig_savename = f'{cond_name}_a.svg'
                fig_path = os.path.join(dataclass.figdir, fig_savename)
                while os.path.exists(fig_path):
                    file_suffix = os.path.splitext(fig_path)[0][-1]
                    fig_path = f'{os.path.splitext(fig_path)[0][:-1]}' \
                               f'{chr(ord(file_suffix) + 1)}{os.path.splitext(fig_path)[1]}'
                if not os.path.exists(fig_path):
                    dataclass_dict[cond_name][0].savefig(fig_path)
                else:
                    print('path exists, not overwriting')
    return cond_keys


def get_subset(dataclass, dataclass_dict, cond_name, filters=(None,), events=None, beh='default',level2filt='date',drop=None,
               ntrials=None,ntrial_start_idx=None, plttype='ts', ylabel='Pupilsize',xlabel=None,plttitle=None,pltaxis=None,
               pltargs=(None,None),plotcols=None,pdr=False,pdelta_wind=(0,1),
               exclude_idx=(None,),ctrl_idx=0, alt_cond_names=None, merge=(None,)):
    """

    :param dataclass:
    :param dataclass_dict:
    :param cond_name:
    :param filters:
    :param events:
    :param beh:
    :param level2filt:
    :param drop:
    :param ntrials:
    :param ntrial_start_idx:
    :param plttype:
    :param ylabel:
    :param xlabel:
    :param plttitle:
    :param pltaxis:
    :param pltargs:
    :param plotcols:
    :param pdr:
    :param pdelta_wind:
    :param exclude_idx:
    :param ctrl_idx:
    :return:
    """

    aligned_tuple = dataclass_dict[cond_name]

    idx_filters = {'time':[],'name':[],'date':[]}
    for idx_filt in filters:
        if idx_filt in idx_filters:
            if isinstance(filters[idx_filt],str):
                filters[idx_filt] = [filters[idx_filt]]
            idx_filters[idx_filt] = filters[idx_filt]
    if len(idx_filters['name']) == 0:
        idx_filters['name'] = dataclass.labels
    if len(idx_filters['date']) == 0:
        idx_filters['date'] = dataclass.dates

    # get all idx combs
    _idx = [idx_filters['name'], idx_filters['date']]
    idx_combs = [p for p in product(*_idx)]

    aligned_tuple_no_empty = list(filter(lambda e: e.shape[0]>1, aligned_tuple[2]))

    for filt in idx_filters[level2filt]:
        aligned_subset = []
        for aligned_df in aligned_tuple_no_empty:
            aligned_subset.append(aligned_df.loc[aligned_df.index[aligned_df.index.droplevel('time').isin(idx_combs)]])
        #
        # if merge:
        #     [pd.concat([aligned_subset[idx] for idx in idx])for idxs in merge]
        #     aligned_subset = [df for df_i, df in enumerate(aligned_subset)
        #                       if df_i not in exclude_idx]
        #
        aligned_subset = [df for df_i, df in enumerate(aligned_subset)
                                        if df_i not in exclude_idx]
        events_subset = [cond_event for ei, cond_event in enumerate(events)
                                        if ei not in exclude_idx]

        if ntrials:
            if isinstance(ntrials,int):
                ntrials = [ntrials]*len(aligned_subset)
            elif isinstance(ntrials,float):
                assert -1 <= ntrials <= 1
                ntrials = [ntrials]*len(aligned_subset)

            # elif isinstance(ntrials,(list,tuple,np.ndarray)):

            for idx, (aligned_df, ntrials_cond) in enumerate(zip(aligned_subset,ntrials)):
                if ntrials_cond:
                    if not ntrial_start_idx:
                        ntrial_start_idx = 0
                    list_ntrials_cond = []
                    for animal in aligned_df.index.get_level_values('name').unique():
                        for date in aligned_df.index.get_level_values('date').unique():
                            sess_df = aligned_df.loc[:,[animal],[date]].copy()
                            if ntrials_cond > 0:
                                if ntrials_cond >= 1:
                                    list_ntrials_cond.append(sess_df.head(ntrials_cond+ntrial_start_idx).copy()
                                                             [ntrial_start_idx:])
                                else:
                                    subset_sess_df = sess_df.iloc[int(ntrial_start_idx*sess_df.shape[0]):
                                                                  int(ntrials_cond*sess_df.shape[0]+int(ntrial_start_idx*sess_df.shape[0])), :]
                                    list_ntrials_cond.append(subset_sess_df.copy())
                            else:
                                if ntrials_cond <= -1:
                                    list_ntrials_cond.append(sess_df.tail(abs(ntrials_cond)+ntrial_start_idx).copy()
                                                             [:ntrial_start_idx])
                                else:
                                    list_ntrials_cond.append(sess_df.tail(abs(int(sess_df.shape[0]*ntrials_cond))).copy())

                    if len(list_ntrials_cond)> 0:
                        aligned_subset[idx] = pd.concat(list_ntrials_cond,axis=0).copy()
                    else:
                        break

        if drop:
            aligned_subset = [aligned_df.drop(drop[1], level=drop[0]) for aligned_df in aligned_subset]
        if events is None:
            events = [f'Event {i}' for i in range(len(aligned_subset))]
        if pdr:
            _pdr_subset = dataclass.get_pdr(copy(aligned_subset),event=beh,smooth=True)[2]
            for df in _pdr_subset:
                assert np.all(df.to_numpy()>=0.0)
            aligned_subset = _pdr_subset
        aligned_subset_fig,aligned_subset_ax = utils.plot_eventaligned(aligned_subset,events_subset,dataclass.duration,
                                                                       beh,plottype_flag=plttype,binflag=True,
                                                                       plotax=pltaxis,pltargs=pltargs,
                                                                       pdelta_wind=pdelta_wind,ctrl_idx=ctrl_idx,
                                                                       plotcols=plotcols)

        if plttype == 'ts':
            aligned_subset_ax.axvline(0,ls='--',c='k')
            aligned_subset_fig.canvas.manager.set_window_title(f'{cond_name}_{filt} N trials={ntrials}')
        aligned_subset_ax.set_ylabel(ylabel)
        if xlabel:
            aligned_subset_ax.set_xlabel(xlabel)
        if plttitle:
            # plttitle.replace('%d')
            aligned_subset_ax.set_title(plttitle)
        fig_savename = f'{cond_name}_{filt}_{plttype}_a.svg'.replace(':','')
        fig_path = os.path.join(dataclass.figdir, fig_savename)
        while os.path.exists(fig_path):
            file_suffix = os.path.splitext(fig_path)[0][-1]
            fig_path = f'{os.path.splitext(fig_path)[0][:-1]}' \
                       f'{chr(ord(file_suffix) + 1)}{os.path.splitext(fig_path)[1]}'
        if pltaxis is None:
            if not os.path.exists(fig_path):
                aligned_subset_fig.savefig(fig_path)
            else:
                print('path exists, not overwriting')

        if not alt_cond_names:
            alt_cond_names = events_subset
        else:
            if len(alt_cond_names) != len(events_subset):
                alt_cond_names = [cond_event for ei, cond_event in enumerate(alt_cond_names)
                                  if ei not in exclude_idx]

        return aligned_subset_fig,aligned_subset_ax, aligned_subset, alt_cond_names


def plot_traces(iter1, iter2, data, dur, fs, control_idx=None, cond_subset=None, cmap_name='RdBu_r', binsize=0, binskip=1,
                cmpap_lbls=('start', 'end'),pltax=None, cmap_flag=True,linealpha=0.5,
                plotformatdict=None):
    lines = ["--", "-.", ":", "-"]

    if iter1 == 'all':
        iter1 = [data[2][0].index.get_level_values('name').unique()]
    if iter2 == 'all':
        iter2 = [data[2][0].index.get_level_values('date').unique()]

    if isinstance(data, (list,tuple)):
        if len(data) >= 2:
            if isinstance(data[2][0], pd.DataFrame):
                working_dfs = data[2]
            else:
                working_dfs = None
        else:
            working_dfs = None
    else:
        working_dfs = None
    if working_dfs is None:
        print('Incorrect format for data')
        return None

    if cond_subset is None:
        cond_subset = list(range(len(working_dfs)))
        cond_subset.pop(control_idx)
    if pltax:
        fig,axes = pltax
    else:
        fig, axes = plt.subplots(len(iter1), len(iter2),sharex='all',sharey='all',squeeze=False)
    x_ts = np.arange(dur[0],dur[1]-fs,fs)
    # get sessions with alternating pattern trials

    for i1, e1 in enumerate(iter1):
        for i2, e2 in enumerate(iter2):
            if (e1, e2) not in list(working_dfs[0].index.droplevel('time').to_series().unique()):
                continue
            sess_conds_dfs = [working_dfs[cond_idx].loc[:,[ e1],[ e2]] for cond_idx in list(cond_subset)]

            for si, sess_df in enumerate(sess_conds_dfs):
                if binsize:
                    sess_df = sess_df.rolling(binsize).mean()[binsize - 1::binskip]
                cmap = plt.get_cmap(cmap_name, sess_df.shape[0])
                for i, (idx, row) in enumerate(sess_df.iterrows()):
                    if cmap:
                        linecol = cmap(i)
                    else:
                        linecol = 'lightgrey'
                    axes[i1][i2].plot(x_ts,row, c=linecol, ls=lines[si % len(lines)],alpha=linealpha,label=f'{e1} {e2}')
            if control_idx != None:
                control_df = working_dfs[control_idx].loc[:, e1, e2]
                axes[i1][i2].plot(x_ts,control_df.mean(axis=0), c='k')
            axes[i1][i2].axvline(0, c='k', ls='--')

    fig.subplots_adjust(left=0.05, bottom=0.08, right=0.85, top=0.95, wspace=0.025, hspace=0.025)
    # set cbar position
    if not cmap_flag:
        fig_cbar = None
    else:
        x0, y0, width, height = [1.025, -.75, 0.075, 3.0]
        Bbox = matplotlib.transforms.Bbox.from_bounds(x0, y0, width, height)
        ax4cmap = axes[int((axes.shape[0]/2))][-1]
        trans = ax4cmap.transAxes + fig.transFigure.inverted()
        l, b, w, h = matplotlib.transforms.TransformedBbox(Bbox, trans).bounds
        cbaxes = fig.add_axes([l, b, w, h])
        cmap = plt.get_cmap(cmap_name, 1000)
        norm = matplotlib.colors.Normalize()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig_cbar = plt.colorbar(sm, cax=cbaxes,  orientation='vertical')
        # fig_cbar = fig.colorbar(sm, ticks=(0, 1), ax=axes[:, -1])
        # cbaxes.tick_params(labelsize=9)

        # label plot
        cbaxes.set_yticks([0,1],cmpap_lbls)
    if plotformatdict:
        for ai, ax in enumerate(axes[:,0]):
            ax.set_ylabel(plotformatdict.get('ylabel'))
            ax.text(-2,0.5,plotformatdict.get('rowtitles')[ai],rotation='vertical')
        for ci, ax in enumerate(axes[-1,:]):
            ax.set_xlabel(plotformatdict.get('xlabel'))
        for ci, ax in enumerate(axes[0, :]):
            ax.set_title(plotformatdict.get('coltitles')[ci], fontsize=10)

        fig.suptitle(plotformatdict.get('figtitle'),fontsize=11)
        fig.canvas.manager.set_window_title(plotformatdict.get('figtitle'))

    return fig, axes, fig_cbar



def glm_from_baseline(traces: pd.DataFrame,dur,bseln_dur,ax):

    traces = traces.copy()
    trace_times = np.linspace(dur[0],dur[1],traces.shape[1])
    traces.columns = trace_times

    baselines = traces.loc[:,dur[0]:(dur[0]+bseln_dur)]
    baseline_means = baselines.mean(axis=1)
    baselines_diff_vars = baselines.diff(axis=1).std(axis=1)

    # response = traces.loc[:,(dur[0]+bseln_dur):]
    response = traces
    response_diff = response.diff(axis=1)
    # response_diff_max = response_diff.values[:,response_diff.abs().idxmax(axis=1).to_list()]
    response_baselined = response.subtract(baseline_means,axis=0)
    response_stds = response.std(axis=1)
    large_changes = response_baselined.diff(axis=1).abs().ge(response_stds, axis='index')
    try:response_diff_max = [response_diff.at[idx,col] for idx, col in response_diff.loc[:,0.0:1.5].abs().idxmax(axis=1).iteritems()]
    except IndexError: pass

    n_breaks = 2
    n = response_baselined.shape[1]
    # model = rpt.Dynp(model="l1")
    change_mdl = response_baselined.apply(lambda r: copy(rpt.Binseg()).fit(r.to_numpy()),axis=1)
    change_mdl_wstd = pd.concat([change_mdl,response_stds],axis=1)
    # change_bkps = change_mdl_wstd.apply(lambda r: r[0].predict(epsilon=0.5 * n *r[1]**2),axis=1)
    change_bkps = change_mdl_wstd.apply(lambda r: r[0].predict(pen=r[1]**2*np.log(n)),axis=1)
    change_bkps_tdlt = change_bkps.apply(lambda r: np.array([response.columns[i-1] for i in r]))
    plot_chg = False
    if plot_chg:
        fig,ax_chgpnts = plt.subplots()
        ax_chgpnts.plot(traces.iloc[-50])
        for i in change_bkps_tdlt[-50]:
            ax_chgpnts.axvline(i, ls='--', c='k')
        ax_chgpnts.set_xlabel('Time from X')
        ax_chgpnts.set_ylabel('Raw pupil diameter (non-baselined)')

    t_1st_change = change_bkps_tdlt.apply(lambda r: r[r>0.0][0])
    diff_w_t1st_change = pd.concat([response_diff,t_1st_change], axis=1)
    diff_1st_change = diff_w_t1st_change.apply(lambda r: r.loc[r.iloc[-1]],axis=1)

    Xtrain = baseline_means
    Xtrain = sm.add_constant(Xtrain)
    ytrain = np.array(diff_1st_change)>0

    glm = sm.RLM(ytrain,Xtrain,M=sm.robust.norms.HuberT()).fit()

    ax.scatter(baseline_means,diff_1st_change,marker='x',s=2)
    baseline_range = np.linspace(baseline_means.min(),baseline_means.max(),1000)
    ax.plot(baseline_range,baseline_range*glm.params[0], c='k', ls='--',label=f'x1={round(glm.params[0],3)}')
    ax.set_xlabel('baseline mean')
    ax.set_ylabel('Rate of first detected pupil size change after event')
    ax.legend(loc=1)

    return glm


class Main:

    def __init__(self,pklfilename, duration_window=(-1,2),extra_steps=True,figdir=None,fig_ow=False):
            self.subsets = {}
            plt.style.use("seaborn-white")
            with open(pklfilename,'rb') as pklfile:
                self.data = pickle.load(pklfile)

            for sess in self.data.copy():
                try:
                    self.data[sess].trialData
                except AttributeError:
                    self.data.pop(sess)
            for sess in self.data.copy():
                if self.data[sess].trialData is None or self.data[sess].pupildf is None:
                    self.data.pop(sess)
            for sess in self.data:
                self.data[sess].trialData.index.set_names(['name','date'],level=[0,1],inplace=True)

            self.add_dt_cols2sesstd(['Trial_Start','Trial_End','Time','ToneTime','Gap_Time','RewardTone_Time'])

            if extra_steps:
                self.add_pretone_dt()
            self.duration = duration_window
            self.labels = list(np.unique([e.split('_')[0] for e in self.data]))
            self.dates = list(np.unique([e.split('_')[1] for e in self.data]))
            self.sessions = list(self.data.keys())

            # self._pupildf()

            today =  datetime.strftime(datetime.now(),'%y%m%d')
            if figdir is None:
                figdir = os.path.join(os.getcwd(),'figures',today)
            self.figdir = figdir
            if not os.path.isdir(self.figdir):
                os.mkdir(self.figdir)
            else:
                if not fig_ow:
                    figdir = f'{figdir}_copy_{today}a'
                    while os.path.isdir(figdir):
                        file_suffix = os.path.splitext(figdir)[0][-1]
                        figdir = f'{os.path.splitext(figdir)[0][:-1]}' \
                                   f'{chr(ord(file_suffix) + 1)}{os.path.splitext(figdir)[1]}'
                    self.figdir = figdir
                    os.mkdir(self.figdir)
            # self.figdir = f''
            self.samplerate = self.data[self.sessions[0]].pupildf.index.to_series().diff().median().total_seconds()

    def add_date_pupildf(self):
        for sess in self.data:
            date = sess.split('_')[1]
            date_dt = datetime.strptime(date,'%y%m%d')
            pupil_df = self.data[sess].pupildf
            pupil_df_ix = pupil_df.index
            merged_ix = [e.replace(year=date_dt.year,month=date_dt.month,day=date_dt.day) for e in pupil_df_ix]
            pupil_df.index = merged_ix

    def add_dt_cols2sesstd(self,column_names):
        for sess in self.data:
            sess_td = self.data[sess].trialData
            for col in column_names:
                utils.add_datetimecol(sess_td,col)

    def add_offset_ser(self):
        for sess in self.data:
            sess_date = sess.split('_')[1]
            sess_td = self.data[sess].trialData
            daylightsavings = np.array(
                [[200329, 201025], [210328, 211031], [220327, 221030], [220326, 221029]])  # daylight saving period
            _dst_arr = daylightsavings - int(sess_date)
            if all(_dst_arr.prod(axis=1) > 0):
                offset_series = np.full_like(sess_td.index, 0.0)
            else:
                offset_series = np.full_like(sess_td.index, 1.0)
            sess_td['Offset'] = offset_series

    def add_session_block(self,block_num, idx=None):
        for sess in self.data:
            sess_td = self.data[sess].trialData
            sess_td['Session_Block'] = np.full_like(sess_td.index,block_num)

    def add_stage(self,stage, idx=None):
        for sess in self.data:
            sess_td = self.data[sess].trialData
            sess_td['Stage'] = np.full_like(sess_td.index,stage)

    def add_pretone_dt(self):
        for sess in self.data:
            td_df = self.data[sess].trialData
            td_df['Pretone_end_dt'] = [tstart+timedelta(0,predur) for tstart, predur in
                                       zip(td_df['Trial_Start_dt'], td_df['PreTone_Duration'])]

    def add_01_1st_flag(self):
        for sess in self.data:
            td_df = self.data[sess].trialData
            if 3 not in td_df['Stage'].values:
                td_df['01_first'] = np.full_like(td_df.index,-1)
            elif 0.1 not in td_df['PatternPresentation_Rate'].values and 0.9 not in td_df['PatternPresentation_Rate'].values:
                td_df['01_first'] = np.full_like(td_df.index,-1)
            else:
                pres_rates = [1 if e == 0.1 else 0 for e in td_df['PatternPresentation_Rate'] if e in [0.1, 0.9]]
                td_df['01_first'] = np.full_like(td_df.index,pres_rates[0])

    def add_x_col(self):
        for sess in self.data:
            td_df = self.data[sess].trialData
            td_df['Pretone_end_dt'] = [tstart+timedelta(0,predur) for tstart, predur in
                                       zip(td_df['Trial_Start_dt'], td_df['PreTone_Duration'])]

    def add_diff_col_dt(self,colname):
        for sess in self.data:
            td_df = self.data[sess].trialData
            td_df[f'{colname}_diff'] = td_df[colname].diff()

    def add_stage3_05_order(self):
        for sess in self.data:
            td_df = self.data[sess].trialData
            onset_05 = (td_df['PatternPresentation_Rate'] == 0.5).astype(int).diff() > 0.0
            order_05 = onset_05.cumsum()
            td_df['0.5_order'] = order_05-1

    def add_rolling_mean(self, colname, windowsize):
        for sess in self.data:
            td_df = self.data[sess].trialData
            td_df[f'{colname}_roll'] = td_df[colname].rolling(windowsize).mean()

    def add_lick_in_window_bool(self,colname):
        for sess in self.data:
            td_df = self.data[sess].trialData
            td_df.set_index('Trial_Start_dt', append=True, inplace=True, drop=False)
            sess_date = sess.split('_')[1]
            y,m,d = int(f'20{sess_date[:2]}'),int(sess_date[2:4]),int(sess_date[4:])
            td_df['Lick_Times_dt'] = td_df['Lick_Times'].apply(lambda e: utils.format_timestr(e.split(';'),(y,m,d)))
            td_df[f'Lick_in_window'] = td_df.apply(lambda e: any(list(map(partial(utils.in_time_window,t=e[colname],
                                                                                  window=(-1,2)),
                                                                      e['Lick_Times_dt']))),axis=1)

    def add_viol_diff(self):
        for sess in self.data:
            td_df = self.data[sess].trialData
            normal_pattern = align_functions.filter_df(td_df, ['e!0', 'd0', 's2'])['PatternID'].unique()[-1]
            normal_pattern = np.array(normal_pattern.split(';'),dtype=int)
            td_df['C_tone_diff'] = td_df['PatternID'].apply(lambda e: (np.array(e.split(';'), dtype=int)
                                                                       - normal_pattern)[2])
            td_df['D_tone_diff'] = td_df['PatternID'].apply(lambda e: (np.array(e.split(';'), dtype=int)
                                                                       - normal_pattern)[3])

    def get_aligned(self, filters, event_shift=(0.5,), align_col='ToneTime_dt', event='ToneTime', plot=False,
                    xlabel='', plotsess=False, plotlabels=('Normal', 'Deviant'), pdr=False, ax=None, plotcols=None,
                    use4pupil=False, animals=None, daterange=None, pmetric='dlc_area_zscored',baseline=True,
                    sep_cond_cntrl_flag=False):

        if len(event_shift) != len(filters):
            if len(event_shift * len(filters)) == len(filters):
                viol_shifts = event_shift * len(filters)
            else:
                print('invalid viol_shift param')
                return None
        else:
            viol_shifts = event_shift
        print(viol_shifts)
        if use4pupil:
            filters = [fil+['4pupil'] for fil in filters]

        if xlabel == '':
            xlabel = f'Time from {event.split("_")[:-1]}'
        ylabel = 'zcscored pupil size'
        tonealigned_viols, tonealigned_viols_df, tonealigned_trials,trials_excluded = align_wrapper(self.data,filters,align_col,
                                                                                    self.duration,alignshifts=viol_shifts,
                                                                                    plotlabels=plotlabels, plottitle='Violation',
                                                                                    xlabel=xlabel, animal_labels=self.labels,
                                                                                    plotsess=plotsess, baseline=baseline,
                                                                                    pupilmetricname=pmetric,
                                                                                    sep_cond_cntrl_flag=sep_cond_cntrl_flag
                                                                                    )
        try:tonealigned_viols_df.columns = plotlabels
        except ValueError: print('value error')
        if pdr:
            tonealigned_viols = self.get_pdr(tonealigned_viols,None, None, plot=False)[2]
            ylabel = 'PDR a.u'
        for aligned in tonealigned_viols:
            aligned.index = pd.MultiIndex.from_tuples(aligned.index,names=['time','name','date'])
        if plot:
            if animals is not None:
                for animal in animals:
                    try:
                        tonealigned_viols_2plot = [ptype.loc[animal,:,:] for ptype in tonealigned_viols]
                    except:
                        tonealigned_viols_2plot = tonealigned_viols
            else:
                tonealigned_viols_2plot = tonealigned_viols

            tonealigned_viols_fig, tonealigned_viols_ax = plot_eventaligned(tonealigned_viols_2plot,plotlabels,
                                                                        self.duration, event,plotax=ax,plotcols=plotcols, shift=event_shift)
            tonealigned_viols_fig.canvas.manager.set_window_title(f'All trials aligned to {event}')
            # tonealigned_viols_ax.set_ylim((-.5,1))
            tonealigned_viols_ax.set_ylabel(ylabel)
            tonealigned_viols_ax.axvline(0,ls='--',color='k')
            tonealigned_viols_ax.set_xlabel(xlabel)
            tonealigned_viols_fig.set_size_inches(8,6)
            # tonealigned_viols_fig.savefig(os.path.join(self.figdir,'violaligned_normdev.png'),bbox_inches='tight')
        else:
            tonealigned_viols_fig, tonealigned_viols_ax = None,None

        return tonealigned_viols_fig,tonealigned_viols_ax,tonealigned_viols,tonealigned_trials,trials_excluded

    def get_firsts(self,aligned_data,n_firsts, plotlabels, event, shuffle=False, pdr=False, plot=True):

        aligned_arr = aligned_data[2]
        aligned_trialnums = aligned_data[3]
        aligned_firsts = []
        for i,ptype in enumerate(aligned_arr):
            sess_start_idx = 0
            list_ptype_firsts = []
            for s in self.sessions:
                sess_ptype = ptype.iloc[sess_start_idx:sess_start_idx+aligned_trialnums[s][i]]
                if shuffle:
                    np.random.shuffle(sess_ptype)
                    list_ptype_firsts.append(sess_ptype)
                else:
                    try: list_ptype_firsts.append(sess_ptype.iloc[0:n_firsts,:])
                    except IndexError: print('out of bounds')
                sess_start_idx += aligned_trialnums[s][i]
            aligned_firsts.append(np.concatenate(list_ptype_firsts))

        ylabel = 'zcscored pupil size'
        fig, ax = None,None
        if pdr:
            aligned_firsts = self.get_pdr(aligned_firsts,None, None, False)[2]
            ylabel = 'PDR a.u'
        if plot:
            fig,ax = plot_eventaligned(aligned_firsts,plotlabels,self.duration,event)
            # ax.set_ylim((-.5,1))
            if shuffle:
                ax.set_title('Shuffled to Tone time')
                fig.canvas.manager.set_window_title('Shuffled to Tone time')
            else:
                fig.canvas.manager.set_window_title(f'First {n_firsts} each session')
            ax.set_ylabel(ylabel)
            ax.axvline(0,ls='--',color='k')
            fig.set_size_inches(8,6)
            fig.savefig(os.path.join(self.figdir,'violaligned_normdev.png'),bbox_inches='tight')

        return fig, ax, aligned_firsts

    def get_lasts(self, aligned_data, n_trials, plotlabels, event, shuffle=False, pdr=False, plot=True):

            aligned_arr = aligned_data[2]
            aligned_trialnums = aligned_data[3]
            aligned_ntrials = []
            for i,ptype in enumerate(aligned_arr):
                sess_end_idx = 0
                list_ptype_ntrials = []
                for s in self.sessions:
                    sess_end_idx += aligned_trialnums[s][i]
                    sess_ptype = ptype.iloc[sess_end_idx-aligned_trialnums[s][i]: sess_end_idx]
                    if shuffle:
                        np.random.shuffle(sess_ptype.iloc[sess_end_idx:sess_end_idx+aligned_trialnums[s][i]])
                        list_ptype_ntrials.append(sess_ptype.iloc[sess_end_idx:sess_end_idx + n_trials, :])
                    else:
                        try: list_ptype_ntrials.append(sess_ptype.iloc[-n_trials:, :])
                        except IndexError: print('out of bounds')
                    sess_end_idx += aligned_trialnums[s][i]
                aligned_ntrials.append(np.concatenate(list_ptype_ntrials))

            ylabel = 'zcscored pupil size'
            fig, ax = None,None
            if pdr:
                aligned_ntrials = self.get_pdr(aligned_ntrials,None, None, False)[2]
                ylabel = 'PDR a.u'
            if plot:
                fig,ax = plot_eventaligned(aligned_ntrials,plotlabels,self.duration,event)
                # ax.set_ylim((-.5,1))
                if shuffle:
                    ax.set_title('Shuffled to Tone time')
                    fig.canvas.manager.set_window_title('Shuffled to Tone time')
                else:
                    fig.canvas.manager.set_window_title(f'Last {n_trials} each session')
                ax.set_ylabel(ylabel)
                ax.axvline(0,ls='--',color='k')
                fig.set_size_inches(8,6)
                fig.savefig(os.path.join(self.figdir,'violaligned_normdev.png'),bbox_inches='tight')

            return fig, ax, aligned_ntrials

    def get_pdr(self, aligned_data,event,plot=False,plotlabels=None,smooth=False,han_size=0.15):

        if isinstance(aligned_data[-1],list):
            aligned_arr = aligned_data[2]
        elif isinstance(aligned_data[-1],(pd.DataFrame,np.ndarray)):
            aligned_arr = aligned_data
        else:
            print('No valid aligned array provided')
            return None

        aligned_pdrs = []

        for i,ptype_df in enumerate(aligned_arr):

            # find start of dilation events
            pdr_arr = copy((np.zeros_like(ptype_df)))
            for ri, (idx,trial) in enumerate(ptype_df.iterrows()):
                peak_idx = find_peaks(trial*-1,width=int(0.3/self.samplerate))[0]
                if peak_idx.size > 0:
                    pdr_arr[ri,peak_idx] = 1
            assert np.all(pdr_arr>=0.0)
            # aligned_deriv = np.diff(ptype_df,axis=1)/self.samplerate
            # pdr_arr = aligned_deriv
            # pdr_arr = (aligned_deriv>0.0).astype(int)
            if smooth:
                pdr_arr = np.array([utils.smooth(x,int(1/self.samplerate)) for x in pdr_arr])
                # pdr_arr =  np.array([utils.butter_filter(x, 2, 1 / self.samplerate, filtype='low') for x in pdr_arr])
            aligned_pdrs.append(pd.DataFrame(pdr_arr,index=ptype_df.index))
            # aligned_pdrs = pd.DataFrame(aligned_pdrs,index=ptype.index)
            # assert np.all(pdr_arr>=0.0)

        fig, ax = None,None
        if plot:
            fig,ax = plot_eventaligned(aligned_pdrs,plotlabels,self.duration,event)
            fig.canvas.manager.set_window_title('PDR by condition')
            ax.set_ylabel('PDR a.u')
            ax.axvline(0,ls='--',color='k')
            ax.set_title('Dilation rate aligned to ToneTime')
            fig.set_size_inches(8,6)
        for df in aligned_pdrs:
            assert np.all(df.to_numpy()>=0.0)
        return fig, ax, aligned_pdrs

    def get_pupil_delta(self, aligned_data,animals,labels,delta_metric='sum',window=(0,1.5),delta=True):

        if isinstance(aligned_data[-1],(list,tuple)):
            if len(aligned_data[-1]) == 1:
                aligned_arr = aligned_data[-1]
            else:
                 aligned_arr = aligned_data[-1][2]
        elif isinstance(aligned_data[-1],(pd.DataFrame,np.ndarray)):
            aligned_arr = aligned_data
        else:
            print('No valid aligned array provided')
            return None

        if type(aligned_arr) != list:
            print('not list')
            return None
        start_idx =int( (window[0]-self.duration[0])/self.samplerate)
        end_idx = int((window[1]-self.duration[0])/self.samplerate)

        if len(animals)<2:
            fig, axes = plt.subplots()
            axes = np.array([axes])
        else:
            fig,axes = plt.subplots(ceil(int(len(animals))))

        for animal, ax in zip(animals,axes.flatten()):
            control_trace = aligned_arr[-1].xs(animal,level='name').mean()
            for i,ptype in enumerate(aligned_arr[:-1]):
                if i ==0:

                    if delta:
                        aligned_vs_cnt = ptype.xs(animal,level='name')-control_trace
                    else:
                        aligned_vs_cnt = ptype.xs(animal, level='name')

                    new_days = np.where(aligned_vs_cnt.index.to_frame()['time'].diff() > timedelta(0,hours=12))[0]
                    if delta_metric == 'sum':
                        aligned_vs_cnt_delta = aligned_vs_cnt.iloc[:,start_idx:end_idx].sum(axis=1)
                    elif delta_metric == 'max':
                        aligned_vs_cnt_delta = aligned_vs_cnt.iloc[:,start_idx:end_idx].max(axis=1)
                    elif delta_metric.isnumeric():
                        int_idx = float(delta_metric)-self.duration/self.samplerate
                        aligned_vs_cnt_delta = aligned_vs_cnt.iloc[:,int(int_idx)]
                    else:
                        print('bad delta metric')
                        return None

                    aligned_vs_cnt_delta = aligned_vs_cnt_delta.sort_index(level='time')
                    if not delta:
                        ax.plot(control_trace.reset_index().index, aligned_vs_cnt_delta, label=animal)
                    ax.plot(aligned_vs_cnt_delta.reset_index().index,aligned_vs_cnt_delta,label=animal)
                    ax.legend()

                    if len(new_days) > 0:
                        for new_day in new_days:
                            ax.axvline(new_day,linestyle='dotted',color='lightgrey')
                    ax.set_ylabel('Pupil Delta')
                    # ax.set_xlabel('Ntrials')
                    ax.set_title(f'Pupil delta over sessions for {animal}')
        if len(animals)>2:
            for ax in axes:
                ax.set_xlabel('Ntrials')
        fig.set_size_inches(8,6)
        fig.set_tight_layout(True)
        fig.canvas.manager.set_window_title('Pupil Delta by animal')
        return fig,axes

    def pupilts_by_session(self, dataclass, dataclass_dict, key2use, animals2plot, dates2plot, eventnames, dateconds,
                           align_point, tsplots_by_animal, tsplots_by_animal_ntrials=None, ntrials=None,plttype='ts'):

        if isinstance(key2use,(list, tuple)):
            key2use = key2use[0]
        for ai,animal in enumerate(animals2plot):
            for di, date2plot in enumerate(dates2plot):
                if date2plot not in pd.concat(dataclass_dict[key2use][2],axis=0).loc[:,animal,:].index.get_level_values('date'):
                    continue
                get_subset(dataclass,dataclass_dict,key2use,{'date':[date2plot],'name':animal},eventnames,
                           f'{align_point} time', plttitle=dateconds[di], level2filt='name',plttype=plttype,
                           pltaxis=(tsplots_by_animal[0],tsplots_by_animal[1][ai,di]))
                tsplots_by_animal[1][ai, di].set_title('')
                if ai == len(animals2plot)-1:
                    tsplots_by_animal[1][ai, di].set_xlabel(f'Time since {align_point[0]} (s)')
                if ai == 0:
                    tsplots_by_animal[1][ai, di].set_title(dateconds[di])
                if tsplots_by_animal_ntrials:
                    get_subset(dataclass, dataclass_dict, key2use, {'date': [date2plot], 'name': animal},
                               eventnames, f'{align_point} time', plttitle=dateconds[di],
                               level2filt='name', ntrials=ntrials,
                               pltaxis=(tsplots_by_animal_ntrials[0], tsplots_by_animal_ntrials[1][ai, di]))
                    tsplots_by_animal_ntrials[1][ai, di].set_title('')

    def dump_trial_pupil_arr(self,subset_save_dir='subset_pupil_arr'):
        subset_save_dir = Path(subset_save_dir)
        if not subset_save_dir.is_dir():
            subset_save_dir.mkdir()

        today_str = datetime.now().strftime('%y%m%d')
        for subset in self.subsets:
            dfs2dump =[]
            filename = utils.unique_file_path(subset_save_dir/f'{subset}_{today_str}.h5')
            subset_dfs = self.subsets[subset][2]
            cond_names = self.subsets[subset][3]
            if not cond_names:
                cond_names = np.arange(len(subset_dfs))
            for cond_i, (df,c_name) in enumerate(zip(subset_dfs,cond_names)):
                # df = cond[2]
                df = df.assign(condition=np.full_like(df.index,c_name)).set_index('condition', append=True)
                df.index.reorder_levels(['condition','time','name','date'])
                dfs2dump.append(df)
            pd.DataFrame.to_hdf(pd.concat(dfs2dump,axis=0),filename,'df')


class PupilEventConditions:

    def __init__(self):
        fam_filts = {
            'p_rate': [[['plow'], ['p0.5'], ['phigh'], ['none']],
                       ['0.1', '0.5', '0.9', 'control']],
            'p_rate_ctrl': [[['plow'], ['plow', 'none'], ['p0.5'], ['p0.5', 'none'], ['phigh'], ['phigh', 'none']],
                            ['0.1', '0.1 cntrl', '0.5', '0.5 cntrl', '0.9', '0.9 cntrl', 'control']],
            'p_onset': [[['dearly', 'p0.5'], ['dlate', 'p0.5'], ['dmid', 'p0.5']],
                        ['Early Pattern', 'Late Pattern', 'Middle Presentation']],
            # 'p0.5_block': [[['0.5_0','p0.5'], ['0.5_1','p0.5'], ['0.5_2','p0.5'], ['none']],
            #                ['0.5 Block (0.0)', '0.5 Block 1 (0.1)', '0.5 Block 2 (0.9)', 'Control']],
            'alt_rand': [[['s0', 'p0.5'], ['s1', 'p0.5'], ['none', 'p0.5']],
                         ['0.5 Random', '0.5 Alternating', 'Control']],
            'alt_rand_ctrl': [
                [['s0', 'p0.5'], ['s0', 'none'], ['s1', 'p0.5'], ['s1', 'p0.5', 'none'], ['none', 'p0.5']],
                ['0.5 Random', '0.5 Random ctrl', '0.5 Alternating', '0.5 Alternating ctrl', 'Control']],
            # 'ntones': [[['p0.5','tones4'],['p0.5','tones3'],['p0.5','tones2'],['p0.5','tones1']],['ABCD', 'ABC','AB','A']],
            # 'pat_nonpatt': [[['e!0'],['e=0']],['Pattern Sequence Trials','No Pattern Sequence Trials']],
            'pat_nonpatt_2X': [[['e!0'], ['none']], ['Pattern Sequence Trials', 'No Pattern Sequence Trials']],
            'p_rate_fm': [[['plow'], ['pmed'], ['phigh'], ['ppost'], ['none']],
                          ['0.1', '0.5', '0.9', '0.6', 'control']],

            'p_rate_local': [[['local_rate_0.2','tones4','c0'],['local_rate_0.4','tones4','c0'],['local_rate_0.6','tones4','c0'],
                              ['local_rate_0.8','tones4','c0'],['local_rate_1.0','tones4','c0'], ['none']],
                             ['1st Q','2nd Q','3rd Q', '4th Q', '5th Q', 'control']],

        }
        normdev_filts = {
            'normdev': [[['d0','tones4'], ['d!0','tones4']], ['Normal', 'Deviant']],
            'normdev_newnorms': [[['d0','tones4'], ['d!0','tones4'], ['d-1','tones4']],
                                 ['Normal', 'Deviant', 'New Normal']],
            'pat_nonpatt_2X': [[['e!0','tones4'], ['none']], ['Pattern Sequence Trials', 'No Pattern Sequence Trials'],
                               'Gap_Time'],
            'normdev_2TS': [[['e!0', 'tones4'], ['none']],
                            ['Pattern Sequence Trials', 'No Pattern Sequence Trials'], 'Trial_Start'],
            'normdev_2X': [[['e!0', 'tones4'], ['none']],
                            ['Pattern Sequence Trials', 'No Pattern Sequence Trials'], 'Gap_Time']
        }
        self.all_filts = {**fam_filts, **normdev_filts}

    def get_condition_dict(self,dataclass,condition_keys,stages,pmetric2use='dlc_radii_a_zscored',
                           do_baseline=True,extra_filts=(),key_suffix=''):

        def get_mean_subtracted_traces(dataclass,suffix=''):
            for key in ['p_rate_ctrl', 'alt_rand_ctrl']:
                if key not in dataclass.aligned.keys():
                    continue
                dataclass.aligned[f'{key}_sub{suffix}'] = copy(dataclass.aligned[key])
                for ti, tone_df in enumerate(dataclass.aligned[key][2]):
                    if (ti % 2 == 0 or ti == 0) and ti < len(dataclass.aligned[key][2]) - 1:
                        print(ti)
                        control_tone_df = dataclass.aligned[key][2][ti + 1].copy()
                        for sess_idx in tone_df.index.droplevel('time').unique():
                            sess_ctrl_mean = control_tone_df.loc[:, [sess_idx[0]], [sess_idx[1]]].mean(axis=0)
                            tone_df.loc[:, sess_idx[0], sess_idx[1]] = tone_df.loc[:, [sess_idx[0]],
                                                                       [sess_idx[1]]] - sess_ctrl_mean
                        # run.aligned[f'{key}_sub'][2][ti] = copy(tone_df)-run.aligned[key][2][ti+1].mean(axis=0)
                        dataclass.aligned[f'{key}_sub{suffix}'][2][ti] = copy(tone_df)
                for idx in [1, 2]:
                    if idx < (len(dataclass.aligned[key][2])):
                        dataclass.aligned[f'{key}_sub{suffix}'][2].pop(idx)


        align_pnts = ['ToneTime', 'Reward', 'Gap_Time']

        if not hasattr(dataclass,'aligned'):
            dataclass.aligned = {}

        # with multiprocessing.Pool() as pool:

        for cond_key in tqdm(condition_keys,desc=f'processing condition key',total=len(condition_keys)):
            cond_filts = self.all_filts.get(cond_key,None)
            if cond_filts == None:
                print(f'{cond_key} not in {self.all_filts.keys()}. Skipping')
                continue
            if f'{cond_key}{key_suffix}' in dataclass.aligned.keys():
                print(f'{cond_key}{key_suffix} exists. Skipping')
                continue
            if len(cond_filts) == 3:
                cond_align_point = cond_filts[2]
            elif '2X' in cond_key:
                cond_align_point = align_pnts[2]
            else:
                cond_align_point = align_pnts[0]
            batch_analysis(dataclass, dataclass.aligned, stages, f'{cond_align_point}_dt', [[0, f'{cond_align_point}'], ],
                           cond_filts[0], cond_filts[1], pmetric=pmetric2use,
                           filter_df=True, plot=False, sep_cond_cntrl_flag=False, cond_name=f'{cond_key}{key_suffix}',
                           use4pupil=True, baseline=do_baseline, pdr=False, extra_filts=extra_filts)
        # get_mean_subtracted_traces(dataclass,suffix=key_suffix)
