import matplotlib.pyplot as plt

import analysis_utils as utils
from analysis_utils import align_wrapper, plot_eventaligned
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime, timedelta
from math import floor, ceil
import matplotlib
import matplotlib.colors
import statsmodels.api as sm
import ruptures as rpt
from copy import deepcopy as copy
import glob
import random

# code for analysing pupil self.data


def batch_analysis(dataclass,dataclass_dict,stages,column,shifts,events,labels,pmetric='dlc_radii_a_zscored',use4pupil=False,
                   filter_df=True,pdr=False,plot=False,baseline=True):
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
    for s in stages:
        for shift in shifts:
            cond_name = f'stage{s}_{events}_{column}_{shift[0]}'
            cond_keys.append(cond_name)
            event_filters = []
            if filter_df:
                for e in events:
                    if e[0] == 'd':
                        event_filters.append(['e!0', 's3', e, f'stage{s}'])
                    elif e == 'none':
                        event_filters.append(['e=0', 's3', f'stage{s}'])
                    else:
                        event_filters.append(e)
            else:
                event_filters.append(event_filters)
            dataclass_dict[cond_name] = dataclass.get_aligned(event_filters,
                                                                 event_shift=[shift[0]] * len(event_filters), align_col=column,
                                                                 event=shift[1], xlabel=f'Time since {shift[1]}', pdr=pdr,
                                                                 plotlabels=labels[:len(event_filters)],
                                                                 use4pupil=use4pupil,pmetric=pmetric,plot=plot, baseline=baseline)
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


def get_subset(dataclass, dataclass_dict, cond_name, filters, events=None, beh='default',extra_filts=None,drop=None,ntrials=None):

    """
       Get a subset of the pupil data.

       Args:
       - dataclass: An object containing the pupil data.
       - dataclass_dict: A dictionary containing the results of the analysis.
       - cond_name: The key for the condition to subset.
       - filters: A dictionary of filters to apply to the data.
       - events: (Optional) A list of events to filter the data by.
       - beh: (Optional) The behavior to filter the data by.
       - extra_filts: (Optional) Additional filters to apply to the data.
       - drop: (Optional) A list of columns to drop from the data.
       - ntrials: (Optional) The number of trials to select.

       Returns:
       - A tuple containing the subsetted data, metadata, and aligned data.
       """
    aligned_tuple = dataclass_dict[cond_name]
    for level in filters:
        for i, filt in enumerate(filters[level]):
            if extra_filts:
                levels = [level,*list(extra_filts.keys())]
                idx_filts = [filt,*list(extra_filts.values())]
            else:
                levels = level
                idx_filts = filt
            # if isinstance(idx_filts,list):
            #     list_dfs= []
            #     for i, idx in enumerate(idx_filts):

            if level == 'name':
                aligned_subset = [aligned_df.loc[:,idx_filts,:] for aligned_df in aligned_tuple[2]]
            elif level == 'date':
                aligned_subset = [aligned_df.loc[:, : idx_filts] for aligned_df in aligned_tuple[2]]
            else:
                continue

            if ntrials:
                if isinstance(ntrials,int):
                    ntrials = [ntrials]*len(aligned_subset)
                for idx, (aligned_df, ntrials_cond) in enumerate(zip(aligned_subset,ntrials)):
                    if ntrials_cond:
                        list_ntrials_cond = []
                        for animal in aligned_df.index.get_level_values('name').unique():
                            for date in aligned_df.index.get_level_values('date').unique():
                                sess_df = aligned_df.loc[:,animal,date]
                                if ntrials_cond > 0:
                                    list_ntrials_cond.append(sess_df.head(ntrials_cond))
                                else:
                                    list_ntrials_cond.append(sess_df.tail(abs(ntrials_cond)))

                        aligned_subset[idx] = copy(pd.concat(list_ntrials_cond,axis=0))

            if drop:
                aligned_subset = [aligned_df.drop(drop[1], level=drop[0]) for aligned_df in aligned_subset]
            if events is None:
                events = [f'Event {i}' for i in range(len(aligned_subset))]
            aligned_subset_fig,aligned_subset_ax = utils.plot_eventaligned(aligned_subset,events,dataclass.duration,beh)

            aligned_subset_ax.axvline(0,ls='--',c='k')
            aligned_subset_ax.set_ylabel('Pupil Size')
            aligned_subset_fig.canvas.manager.set_window_title(f'{cond_name}_{filt} N trials={ntrials}')
            fig_savename = f'{cond_name}_{filt}_a.svg'.replace(':','')
            fig_path = os.path.join(dataclass.figdir, fig_savename)
            while os.path.exists(fig_path):
                file_suffix = os.path.splitext(fig_path)[0][-1]
                fig_path = f'{os.path.splitext(fig_path)[0][:-1]}' \
                           f'{chr(ord(file_suffix) + 1)}{os.path.splitext(fig_path)[1]}'
            if not os.path.exists(fig_path):
                aligned_subset_fig.savefig(fig_path)
            else:
                print('path exists, not overwriting')


def plot_traces(iter1, iter2, data, dur, fs, control_idx=0, cond_subset=None, cmap_name='RdBu_r', binsize=0,
                cmpap_lbls=('start', 'end'),
                plotformatdict=None):
    lines = ["--", "-.", ":", "-"]
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
        cond_subset = list(range(len(working_dfs))).pop(control_idx)
    fig, axes = plt.subplots(len(iter1), len(iter2),sharex='all',sharey='all')
    x_ts = np.arange(dur[0],dur[1]-fs,fs)
    for i1, e1 in enumerate(iter1):
        for i2, e2 in enumerate(iter2):
            sess_conds_dfs = [working_dfs[cond_idx].loc[:, e1, e2] for cond_idx in list(cond_subset)]
            # sess_conds_dfs.pop(control_idx)

            for si, sess_df in enumerate(sess_conds_dfs):
                if binsize:
                    sess_df = sess_df.rolling(binsize).mean()[binsize - 1::binsize]
                cmap = plt.get_cmap(cmap_name, sess_df.shape[0])
                for i, (idx, row) in enumerate(sess_df.iterrows()):
                    axes[i1][i2].plot(x_ts,row, c=cmap(i), ls=lines[si % len(lines)])
            if control_idx:
                control_df = working_dfs[control_idx].loc[:, e1, e2]
                axes[i1][i2].plot(x_ts,control_df.mean(axis=0), c='k')
            axes[i1][i2].axvline(0, c='k', ls='--')
    fig.subplots_adjust(left=0.05, bottom=0.08, right=0.925, top=0.95, wspace=0.025, hspace=0.025)
    # set cbar position
    x0, y0, width, height = [1.05, -.75, 0.075, 3.5]
    Bbox = matplotlib.transforms.Bbox.from_bounds(x0, y0, width, height)
    ax4cmap = axes[int((axes.shape[0]/2))][-1]
    trans = ax4cmap.transAxes + fig.transFigure.inverted()
    l, b, w, h = matplotlib.transforms.TransformedBbox(Bbox, trans).bounds
    cbaxes = fig.add_axes([l, b, w, h])
    cmap = plt.get_cmap(cmap_name, 1000)
    norm = matplotlib.colors.Normalize()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig_cbar = plt.colorbar(sm, cax=cbaxes, ticks=[0, 1], orientation='vertical')
    # fig_cbar = fig.colorbar(sm, ticks=(0, 1), ax=axes[:, -1])
    fig_cbar.ax.tick_params(labelsize=9)

    # label plot
    fig_cbar.ax.set_yticklabels(cmpap_lbls)
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

    def __init__(self,pklfilename, duration_window=(-1,2),extra_steps=True):
            plt.style.use("seaborn-white")
            with open(pklfilename,'rb') as pklfile:
                self.data = pickle.load(pklfile)

            for sess in self.data.copy():
                try: self.data[sess].trialData
                except AttributeError: self.data.pop(sess)
                if self.data[sess].trialData is None or self.data[sess].pupildf is None:
                    self.data.pop(sess)
            if extra_steps:
                self.add_pretone_dt()
            self.duration = duration_window
            self.labels = list(np.unique([e.split('_')[0] for e in self.data]))
            self.dates = list(np.unique([e.split('_')[1] for e in self.data]))
            self.sessions = list(self.data.keys())

            # self._pupildf()

            today =  datetime.strftime(datetime.now(),'%y%m%d')
            self.figdir = os.path.join(os.getcwd(),'figures',today)
            if not os.path.isdir(self.figdir):
                os.mkdir(self.figdir)
            # self.figdir = f''
            self.samplerate = self.data[self.sessions[0]].pupildf.index.to_series().diff().mean().total_seconds()


    def add_date_pupildf(self):
        for sess in self.data:
            date = sess.split('_')[1]
            date_dt = datetime.strptime(date,'%y%m%d')
            pupil_df = self.data[sess].pupildf
            pupil_df_ix = pupil_df.index
            merged_ix = [e.replace(year=date_dt.year,month=date_dt.month,day=date_dt.day) for e in pupil_df_ix]
            pupil_df.index = merged_ix

    def add_pretone_dt(self):
        for sess in self.data:
            td_df = self.data[sess].trialData
            td_df['Pretone_end_dt'] = [tstart+timedelta(0,predur) for tstart, predur in
                                       zip(td_df['Trial_Start_dt'], td_df['PreTone_Duration'])]

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
            td_df[f'{colname}_roll'] = td_df.rolling(windowsize).mean()

    def get_aligned(self, filters, event_shift=[0.5], align_col='ToneTime_dt', event='ToneTime', plot=True,
                    xlabel='', plotsess=False, plotlabels=('Normal', 'Deviant'), pdr=False, ax=None, plotcols=None,
                    use4pupil=False, animals=None, daterange=None, pmetric='dlc_area_zscored',baseline=True):

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
        tonealigned_viols, tonealigned_viols_df, tonealigned_trials = align_wrapper(self.data,filters,align_col,
                                                                                    self.duration,alignshifts=viol_shifts,
                                                                                    plotlabels=plotlabels, plottitle='Violation',
                                                                                    xlabel=xlabel, animal_labels=self.labels,
                                                                                    plotsess=plotsess, baseline=baseline,
                                                                                    pupilmetricname=pmetric,
                                                                                    )
        tonealigned_viols_df.columns = plotlabels
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

        return tonealigned_viols_fig,tonealigned_viols_ax,tonealigned_viols,tonealigned_trials

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

    def get_pdr(self, aligned_data,plotlabels,event,plot=True,smooth=True,han_size=0.15):

        if isinstance(aligned_data[-1],list):
            aligned_arr = aligned_data[2]
        elif isinstance(aligned_data[-1],(pd.DataFrame,np.ndarray)):
            aligned_arr = aligned_data
        else:
            print('No valid aligned array provided')
            return None

        aligned_pdrs = []

        for i,ptype in enumerate(aligned_arr):
            aligned_deriv = np.diff(ptype,axis=1)/self.samplerate
            pdr_arr = (aligned_deriv>0.0).astype(int)
            if smooth:
                # pdr_arr = np.array([utils.smooth(x,int(han_size/self.samplerate)) for x in pdr_arr])
                pdr_arr =  np.array([utils.butter_highpass_filter(x,4,1/self.samplerate,filtype='low') for x in pdr_arr])
            aligned_pdrs.append(pd.DataFrame(pdr_arr,index=ptype.index))
            # aligned_pdrs = pd.DataFrame(aligned_pdrs,index=ptype.index)

        fig, ax = None,None
        if plot:
            fig,ax = plot_eventaligned(aligned_pdrs,plotlabels,self.duration,event)
            fig.canvas.manager.set_window_title('PDR by condition')
            ax.set_ylabel('PDR a.u')
            ax.axvline(0,ls='--',color='k')
            ax.set_title('Dilation rate aligned to ToneTime')
            fig.set_size_inches(8,6)

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


if __name__ == "__main__":
    # pkldir = r'W:\mouse_pupillometry\working_pickles'
    pkldir = r'C:\bonsai\gd_analysis\pickles'

    # pkl2use = r'pickles\human_familiarity_3d_200Hz_015Shan_driftcorr_hpass01.pkl'
    # pkl2use = r'pickles\human_class1_3d_200Hz_015Shan_driftcorr_hpass01_no29.pkl'
    # pkl2use = r'pickles\human_class1_3d_200Hz_015Shan_driftcorr_hpass01.pkl'
    # pkl2use = r'pickles\human_class1_3d_200Hz_015Shan_driftcorr_hpass01.pkl'
    # pkl2use = r'pickles\DO48_fam_2d_200Hz_015Shan_driftcorr_hpass01.pkl'
    # pkl2use = r'pickles\mouse_normdev_2d_200Hz_015Shan_driftcorr_hpass04_wdlc.pkl'
    # pkl2use = r'pickles\mouse_fam_2d_200Hz_015Shan_driftcorr_hpass04_wdlc.pkl'
    # pkl2use = r'/Volumes/akrami/mouse_pupillometry/pickles/DO48_fam_3d_200Hz_015Shan_driftcorr_hpass01_wdlc.pkl'

    # pkl2use = os.path.join(pkldir,'mouse_normdev_2d_90Hz_025Shan_driftcorr_hpass025_wdlc_TOM.pkl')
    pkl2use = os.path.join(pkldir,'mouse_hf_2d_90Hz_6lpass_025hpass_wdlc_TOM_interpol_all_int02s_221121.pkl')


    # pkl2use = r'pickles\mouse_fam_post_2d_90Hz_025Shan_driftcorr_nohpass_wdlc_TOM.pkl'

    run = Main(pkl2use, (-1,3))
    fig_subdir = 'interpol_all'
    if not os.path.isdir(os.path.join(run.figdir, fig_subdir)):
        os.mkdir(os.path.join(run.figdir, fig_subdir))
        run.figdir = os.path.join(run.figdir, fig_subdir)

    # paradigm = ['altvsrand','normdev']
    # paradigm = ['normdev']
    # paradigm = ['familiarity']
    paradigm = ['familiarity','0.5_fam']
    # pmetric2use = 'dlc_radii_a_zscored'
    pmetric2use = 'dlc_EW_zscored'

    # run.whitenoise_hit_miss = run.get_aligned([['a1'], ['a0']],
    #                                           event='WhiteNoise',
    #                                           event_shift=[0.0, 0.0],
    #                                           plotlabels=['Hit', 'Miss'],
    #                                           align_col='Gap_Time_dt', pmetric=pmetric2use, use4pupil=True)

    if 'familiarity' in paradigm:  # analysis to run for familiarity paradigm
        run.familiarity = run.get_aligned([['e!0','plow','tones4','a1'],['e!0','tones4','p0.5','a1'],
                                           ['e!0','tones4','phigh','a1'], ['e=0','a1']],
                                          event_shift=[0.0, 0.0, 0.0, 0.0],
                                          event='ToneTime', xlabel='Time since pattern onset',
                                          plotlabels=['0.1','0.5','0.9','control'], plotsess=False, pdr=False,
                                          use4pupil=False,
                                          pmetric=pmetric2use
                                          )
        run.familiarity = run.get_aligned([['e=0','phigh','a1'], ['e=0','a1']],
                                          event_shift=[0.0,0.0], align_col='Gap_Time_dt',
                                          event='White Noise', xlabel='Time since pattern onset',
                                          plotlabels=['0.9','control'], plotsess=False, pdr=False,
                                          use4pupil=False,
                                          pmetric=pmetric2use
                                          )
        # run.fam_firsts = run.get_firsts(run.familiarity,8,['0.1','0.4','0.9','0.6','control'],'ToneTime')
        shuffle = False
        if shuffle:  # decide whether to shuffle
            for i in range(5):
                run.get_firsts(run.familiarity,8,['0.1','0.4','0.9','0.6','control'],'ToneTime',shuffle=True)

        run.add_stage3_05_order()


        # run.fam_firsts_pdr = run.get_firsts(run.familiarity,8,['0.1','0.4','0.9','0.6','control'],'ToneTime',pdr=True)
        # run.fam_lasts_pdr = run.get_lasts(run.familiarity,8,['0.1','0.4','0.9','0.6','control'],'ToneTime',pdr=True)
        # run.reward = run.get_aligned([['a1'],['a0']],event='Trial End',xlabel='Time since reward tones',
        #                              plotlabels=['correct','incorrect'],align_col='Trial_End_dt',pdr=False)
        # run.reward = run.get_aligned([['a1']],event='RewardTime',xlabel='Time since reward tones', viol_shift=[-0.0],
        #                                  plotlabels=['correct'],align_col='RewardTone_Time_dt',pdr=True)
        #run.fam_delta = run.get_pupil_delta(run.familiarity[2],['DO48'],['0.1','0.4','0.9','0.6','control'],window=[0,1])

        run_ntones_analysis = True
        if run_ntones_analysis:
            fig= plt.figure()
            ax1 = plt.subplot2grid(shape=(2, 3), loc=(0, 0), colspan=3)
            ax2 = plt.subplot2grid(shape=(2, 3), loc=(1, 0), colspan=1)
            ax3 = plt.subplot2grid(shape=(2, 3), loc=(1, 1), colspan=1,sharex=ax2, sharey=ax2)
            ax4 = plt.subplot2grid(shape=(2, 3), loc=(1, 2), colspan=1,sharex=ax2, sharey=ax2)
            run.ntone_ana = run.get_aligned([['e!0','tones4'],['e!0','tones3'],['e!0','tones2'],['e!0','tones1']],
                                            [0.0,0.0,0.0,0.0],
                                            event='Pattern Onset',xlabel='Time since Pattern onset', align_col='ToneTime_dt',
                                            plotlabels=['ABCD','ABC','AB','A'],pdr=False,ax=[fig,ax1],use4pupil=True,
                                            pmetric=pmetric2use)

            for i, (tone_cond,offset,lbl,axis) in enumerate(zip(['tones3','tones2','tones1'],[0.75, 0.5, 0.25],['ABC','AB','A'],
                                            [ax2,ax3,ax4])):
                run.get_aligned([['e!0','tones4'],['e!0',tone_cond]],[0.0], align_col='ToneTime_dt',
                            event=f'ABCD vs {lbl} tones played',xlabel=f'Time since {lbl[-1]} presentation',
                            plotlabels=['ABCD',lbl],plotsess=False,pdr=False,ax=[fig,axis],plotcols=[f'C{0}',f'C{i+1}'],
                            use4pupil=True,pmetric=pmetric2use)
                axis.legend().remove()

            fig.set_size_inches(7,7)
            fig.set_tight_layout(True)

    # pmetric2use = 'rawarea_zscored'
    if 'normdev' in paradigm:

        stages = [4,5]

        column = 'ToneTime_dt'
        shifts = [[0.0,'ToneTime'],[0.5,'Violation']]

        # column = 'Gap_Time_dt'
        # shifts = [[0.0,'Whitenoise']]

        events = ['d0','d!0','none']
        labels = ['Normal', 'Deviant','None']
        # events = ['d0','d1','d2','d3','none']
        # labels = ['Normal', 'AB_D','ABC_','AB__','None']
        keys = []
        pattern_types = []

        run.normdev = {}
        for s in stages:
            for shift in shifts:
                cond_name = f'stage{s}_{events}_{column}_{shift[0]}'
                event_filters = []
                for e in events:
                    if e[0] == 'd':
                        event_filters.append(['e!0', 's3', e, f'stage{s}','a1','tones4'])
                    elif e == 'none':
                        event_filters.append(['e=0', 's3', f'stage{s}','a1'])

                run.normdev[cond_name] = run.get_aligned(event_filters, pmetric=pmetric2use,
                                                         event_shift=[shift[0]]*len(event_filters), align_col=column,
                                                         event=shift[1], xlabel=f'Time since {shift[1]}', pdr=False,
                                                         plotlabels=labels[:len(event_filters)],use4pupil=True)
                keys.append(cond_name)
                run.normdev[cond_name][0].canvas.manager.set_window_title(cond_name)
                fig_savename = f'{cond_name}_a.svg'
                fig_path = os.path.join(run.figdir,fig_savename)
                while os.path.exists(fig_path):
                    file_suffix = os.path.splitext(fig_path)[0][-1]
                    fig_path = f'{os.path.splitext(fig_path)[0][:-1]}' \
                               f'{chr(ord(file_suffix)+1)}{os.path.splitext(fig_path)[1]}'
                if not os.path.exists(fig_path):
                    run.normdev[cond_name][0].savefig(fig_path)
                else:
                    print('path exists, not overwriting')

        dates2plot = run.normdev[keys[0]][2][0].index.get_level_values('date').unique()
        for ki,key in enumerate(keys):
            for date2plot in list(dates2plot):
                # get_subset(run, run.normdev, keys[ki], {'date': [date2plot]},
                #            labels, f'{pmetric2use} time')
                get_subset(run, run.normdev, keys[ki], {'date': [date2plot]},
                           labels, f'{pmetric2use} time',ntrials=[None,10,10])


            # get_subset(run, run.normdev, keys[0],{'name': run.labels},)

        base_plt_title = 'Evolution of pupil response with successive X presentations'
        animals2plot = run.labels
        dates2plot = run.dates
        animal_date_pltform = {'ylabel': 'z-scored pupil size',
                               'xlabel': 'Time since "X"',
                               'figtitle': base_plt_title,
                               'rowtitles': animals2plot,
                               'coltitles': dates2plot,
                               }
        binsize = 1
        for i, cond in enumerate(labels):
            animal_date_pltform['figtitle'] = f"{base_plt_title} binned {binsize} trials: {cond}"
            # indvtraces_binned = plot_traces(animals2plot, dates2plot, run.normdev[keys[0]], run.duration,
            #                                 run.samplerate,
            #                                 plotformatdict=animal_date_pltform, binsize=binsize, cond_subset=[i], )
            # indvtraces_binned[0].savefig(rf'W:\mouse_pupillometry\figures\probrewardplots\evolve{i}.svg',
            #                              bbox_inches='tight')

        # run.normdev_13 = run.get_aligned([['e!0','s3','d0','tones4','stage5'],['e!0','s3','d4','tones4','stage5']],
        #                                  event_shift=[0.0, 0.0], align_col='Pretone_end_dt',
        #                                  event='ToneTime', xlabel='Time since pattern onset', pdr=False,)
        # run.normdev_13[0].canvas.manager.set_window_title('Stage5normal_deviant_Viol')
        #
        # run.normdev_13_stage4 = run.get_aligned([['e!0','s3','d0','tones4','stage4','a1'],['e!0','s3','d4','tones4','stage4','a1']],
        #                                  event_shift=[0.0, 0.0], align_col='Pretone_end_dt',
        #                                  event='ToneTime', xlabel='Time since pattern onset', pdr=False,)
        # run.normdev_13_stage4[0].canvas.manager.set_window_title('Stage5normal_deviant_Viol')
        #
        # run.normdev_13 = run.get_aligned([['e!0','s3','d0','tones4'],['e!0','s3','d4','tones4']],
        #                                  event_shift=[0.5, 0.5],
        #                                  event='Violation', xlabel='Time since pattern onset', pdr=False,
        #                                  plotlabels=['normal','deviant'], plotsess=False,
        #                                  use4pupil=True, pmetric=pmetric2use)
        #
        # run.normdev2 = run.get_aligned([['e!0','s3','d0','tones4','stage5','a1'],['e!0','s3','d2','tones4','stage5','a1']],
        #                                event_shift=[0.0, 0.0], align_col='Pretone_end_dt',
        #                                event='ToneTime', xlabel='Time since pattern onset', pdr=False,
        #                                plotlabels=['normal','deviant'], plotsess=False,
        #                                use4pupil=True, pmetric=pmetric2use)
        #
        # run.newnorms = run.get_aligned([['e!0','s3','d0','tones4','stage5'],['e!0','s3','d-1','tones4','stage5']],
        #                                event_shift=[0.0, 0.0],
        #                                event='ToneTime', xlabel='Time since pattern onset', pdr=False,
        #                                plotlabels=['normal','new normals'], plotsess=False,
        #                                use4pupil=True, pmetric=pmetric2use)
        # run.abcd_control = run.get_aligned([['e!0','s3','d0','tones4','stage5'],['e=0','s3','stage5']],  # line629 in utils
        #                                       event_shift=[0.0, 0.0],
        #                                       event='ToneTime', xlabel='Time since pattern onset', pdr=False,
        #                                       plotlabels=['normal','deviant'], plotsess=False,
        #                                       use4pupil=True, pmetric='dlc_radii_a_zscored')
        #
        # run.normdev_delta = run.get_pupil_delta(run.normdev_13[2],['DO45','DO46','DO47','DO48'],['normal','deviant'])

    if 'altvsrand' in paradigm:
        run.altvsrand = run.get_aligned([['e!0','s0','tones4'], ['e!0','s1','tones4']], pdr=False,
                                        event_shift=[0.0, 0.0],
                                        xlabel='Time since pattern offset', plotsess=False,
                                        plotlabels=['random','alternating'],
                                        use4pupil=True, pmetric=pmetric2use,)

    if '0.5_fam' in paradigm:
        run.add_stage3_05_order()
        run.fam_05 = {}

        column = 'ToneTime_dt'
        shifts = [[0.0, 'ToneTime']]

        stages = [3]
        events = ['0.5_0', '0.5_1','0.5_2', 'none']
        pattern_types = []
        labels = ['0.5 Block (0.0)', '0.5 Block 1 (0.1)','0.5 Block 2 (0.9)', 'Control']
        rate_filter = ['p0.5','p0.5','p0.5',]
        for s in stages:
            for shift in shifts:
                cond_name = f'stage{s}_{events}_{column}_{shift[0]}'
                event_filters = []
                for ei,e in enumerate(events):
                    if e != 'none':
                        event_filters.append(['e!0', 's0', e, rate_filter[ei], 'tones4', f'stage{s}','a1'])
                    else:
                        event_filters.append(['e=0', 's0', f'stage{s}','a1','p0.5'])

                run.fam_05[cond_name] = run.get_aligned(event_filters,
                                                         event_shift=[shift[0]]*len(event_filters), align_col=column,
                                                         event=shift[1], xlabel=f'Time since {shift[1]}', pdr=False,
                                                         plotlabels=labels[:len(event_filters)])
                run.fam_05[cond_name][0].canvas.manager.set_window_title(cond_name)
                fig_savename = f'{cond_name}_a.svg'
                fig_path = os.path.join(run.figdir,fig_savename)
                while os.path.exists(fig_path):
                    file_suffix = os.path.splitext(fig_path)[0][-1]
                    fig_path = f'{os.path.splitext(fig_path)[0][:-1]}' \
                               f'{chr(ord(file_suffix)+1)}{os.path.splitext(fig_path)[1]}'
                if not os.path.exists(fig_path):
                    run.fam_05[cond_name][0].savefig(fig_path)
                else:
                    print('path exists, not overwriting')
