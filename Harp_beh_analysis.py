import time

import matplotlib.colors
import statsmodels.api as sm
from scipy.stats import bootstrap

import align_functions
from align_functions import get_aligned_events
from pupil_analysis_func import Main
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
from plotting_functions import get_fig_mosaic
from matplotlib import cm
import numpy as np
import pandas as pd
import analysis_utils as utils
from copy import deepcopy as copy
from behaviour_analysis import TDAnalysis
import pickle
from datetime import datetime, timedelta
from scipy.signal import find_peaks, find_peaks_cwt
import ruptures as rpt
from pupil_analysis_func import batch_analysis, plot_traces, get_subset, glm_from_baseline
from pathlib import Path


# pkldir = Path(r'X:\Dammy\mouse_pupillometry\pickles')
pkldir = Path(r'D:\bonsai\offline_data')
# pkl2use = pkldir / 'mouse_hf_2309_batch_w_canny_fam_2d_90Hz_hpass01_lpass4hanning015_TOM.pkl'
pkl2use = pkldir / 'mouse_hf_2309_batch_w_canny_DEC_dates_fam_2d_90Hz_hpass01_lpass4hanning025_TOM.pkl'
# run = Main(pkl2use, (-1.0, 3.0), figdir=Path(rf'W:\mouse_pupillometry\figures\mouse_fam_13start_hpass0'),fig_ow=False)
run = Main(pkl2use, (-1.0, 3.0), figdir=Path(rf'W:\mouse_pupillometry\figures\mouse_fam_13start_hpass0'),fig_ow=False)

list_dfs = utils.merge_sessions(r'H:\data\Dammy', run.labels, 'TrialData', [run.dates[0], run.dates[-1]])
run.trialData = pd.concat(list_dfs)
run.trialData.dropna(inplace=True)
utils.add_dt_cols(run.trialData)
run.trialData.set_index('Trial_Start_dt', append=True, inplace=True, drop=False)
run.get_aligned_events = get_aligned_events
run.trialData['Pretone_end_dt'] = [tstart+timedelta(0,predur) for tstart, predur in
                                       zip(run.trialData['Trial_Start_dt'], run.trialData['PreTone_Duration'])]

td_obj = TDAnalysis(r'H:\data\Dammy',run.labels,[run.dates[0], run.dates[-1]])
plots = utils.plot_performance(td_obj.trialData, np.arange(7,13,1), run.labels, run.dates,['b','r','y','purple','cyan'])

reaction_time_plot = td_obj.scatter_trial_metric_bysess('Reaction_time',dates=run.dates)
reaction_time_plot[0].set_size_inches((15,7))
reaction_time_plot[1].set_ylabel('Reaction time (s)')
reaction_time_plot[1].set_ylim(0.1, 1.)
reaction_time_plot[1].set_xticks(np.arange(len(run.dates)))
reaction_time_plot[1].set_xticklabels(run.dates,rotation=40)
reaction_time_plot[0].show()
reaction_time_plot[1].set_title('Reaction time to X across sessions')
reaction_time_plot[0].set_constrained_layout('constrained')

beh_plots = td_obj.beh_daily(plot=True)


print('break now')
time.sleep((10))
harpmatrices_pkl = pkldir / 'mousefam_hf_harps_matrices_allfam_test.pkl'
harp_pkl_ow_flag = True
if harpmatrices_pkl.is_file() and not harp_pkl_ow_flag:
    with open(harpmatrices_pkl, 'rb') as pklfile:
        run.harpmatrices = pickle.load(pklfile)
else:
    run.harpmatrices = align_functions.get_event_matrix(run, run.data, r'X:\Dammy\mouse_pupillometry\mouse_hf\harpbins', )
    with open(harpmatrices_pkl, 'wb') as pklfile:
        pickle.dump(run.harpmatrices, pklfile)
run.lickrasters_firstlick = {}

events2align = ['Gap_Time']

    # run.lickrasters_firstlick[outcome[0]][0].savefig(os.path.join(run.figdir,
    #                                                                f'allsess_licks_{event2align}_{outcome}.svg'),
    #                                                  bbox_inches='tight')

legend_lbl = ['Pattern Trials', 'Non-Pattern Trials']

harp_aligned_pkl = Path(r'pickles\fam_harp_aligned.pkl')
if harp_aligned_pkl.is_file():
    with open(harp_aligned_pkl,'rb') as pklfile:
        run.lickrasters_firstlick = pickle.load(pklfile)

baseline_flag = False
redo_align = False
for ei, event2align in enumerate(events2align):
    for outcome, align_col in zip([['e!0', ], ['e=0']],['ToneTime','Gap_Time']):
        run.animals = run.labels
        if f'{outcome[0]}_{event2align}' not in run.lickrasters_firstlick.keys() or redo_align:
            run.lickrasters_firstlick[f'{outcome[0]}_{event2align}'] = run.get_aligned_events(run.trialData,run.harpmatrices, f'{event2align}_dt', 'HitData_0', (-1.0, 3.0),
                                                                           byoutcome_flag=True, outcome2filt=outcome,
                                                                           extra_filts=None)
            run.lickrasters_firstlick[outcome[0]][0].set_size_inches((12, 9))



lick_ts_fig, lick_ts_ax = plt.subplots(ncols=len(events2align),sharey='all',layout='constrained')
for ei, event2align in enumerate(events2align):
    for oi, outcome in enumerate([['e!0',], ['e=0']]):
        binsize = 500
        prob_lick_mat = run.lickrasters_firstlick[f'{outcome[0]}_{event2align}'][2].fillna(0).rolling(binsize,
                                                                                   axis=1).mean()  # .mean().iloc[:,binsize - 1::binsize]
        if baseline_flag:
            prob_lick_mat = prob_lick_mat.subtract(prob_lick_mat.loc[:, -1.0:0].mean(axis=1),axis=0)
        prob_lick_mean = prob_lick_mat.mean(axis=0)
        lick_ts_ax[ei].plot(prob_lick_mean.index, prob_lick_mean, label=f'{legend_lbl[oi]}')
        c = utils.plotvar(prob_lick_mat, (lick_ts_fig, lick_ts_ax[ei]), prob_lick_mat.columns,f'C{oi}',n_samples=100)
        # lower, upper = prob_lick_mean-prob_lick_mat.std(axis=0)*1,prob_lick_mean+prob_lick_mat.std(axis=0)*1
        # lick_ts_ax[ei].fill_between(prob_lick_mat.columns, lower, upper, alpha=0.1, facecolor=f'C{oi}')

    event_lbl = event2align.replace('Gap','X').replace('_',' ').replace('Pretone end', 'pattern onset').replace('Time','Onset')
    lick_ts_ax[ei].set_xlabel(f'Time from {event_lbl} (s)',fontsize=14)
    lick_ts_ax[ei].set_title(f'Lick rate aligned to {event_lbl}, {binsize/1000}s bin',fontsize=18)
    lick_ts_ax[ei].legend(fontsize=14)
    lick_ts_ax[ei].axvline(0.0, ls='--', c='k', lw=1)
lick_ts_ax[0].set_ylabel('mean lick rate across animals across sessions',fontsize=14)
utils.unique_legend((lick_ts_ax),fontsize=14)
lick_ts_fig.set_size_inches((4.823*2.25, 2.501*2.25))
lick_ts_fig.show()
lick_ts_fig.savefig(run.figdir / f'allsess_HF_lickrate_ts_{"_".join(events2align)}.svg', bbox_inches='tight')

pattern_hist = plt.subplots()
p09_df = align_functions.filter_df(run.trialData, ['phigh', 'e!0']).loc[:, '230221', :]
p05_df = align_functions.filter_df(run.trialData, ['p0.5', 'e!0']).loc[:, '230221', :]
onset_times, onset_counts = np.unique([p05_df.PreTone_Duration.to_list() +
                                       p09_df.PreTone_Duration.to_list()], return_counts=True)
pattern_hist[1].bar(onset_times, onset_counts / np.sum(onset_counts), align='center')
pattern_hist[1].set_xlabel('Pattern embed time from Trial Start (s)')
pattern_hist[1].set_xticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
pattern_hist[1].set_ylabel('Proportion of Trials')
pattern_hist[1].set_title('Distribution of pattern onset times')
pattern_hist[0].savefig(run.figdir / 'pattern_time_dist.svg', bbox_inches='tight')

perf_plot = plt.subplots(figsize=(9,8))
perf_hist = plt.subplots(figsize=(9,8))
for ci, cond in enumerate([['e!0', ], ['e=0']]):
    meanss = []
    for sess in run.data:
        mean_perf = align_functions.filter_df(run.data[sess].trialData, cond)['Trial_Outcome'].mean()
        meanss.append(mean_perf)
    perf_plot[1].bar(legend_lbl[ci], np.mean(meanss),facecolor=f'lightgrey',edgecolor=f'k',linewidth=3)
    perf_plot[1].scatter([legend_lbl[ci]]*len(meanss), meanss,c=f'C{ci}',marker='x')
    print(np.mean(meanss),np.std(meanss))
perf_plot[1].tick_params(labelsize=14)
perf_plot[0].show()
perf_plot[1].set_ylabel('Mean performance',fontsize=14)
perf_plot[1].set_title(f'Mean performance across sessions: {len(meanss)} sessions',fontsize=19)
perf_plot[0].savefig(run.figdir/'mean_performance.svg',bbox_inches='tight')

ntones1_rt = []
ntones2_rt = []
ntones3_rt = []
ntones4_rt = []
ntrones_rt_list = [ntones1_rt,ntones2_rt,ntones3_rt,ntones4_rt]

for sess in run.data:
    sess_td = run.data[sess].trialData
    for i, ntone_list in enumerate(ntrones_rt_list):
        ntones_df = align_functions.filter_df(sess_td, [f'tones{i + 1}'])
        ntones_df_pat_X = ntones_df[(ntones_df['Gap_Time_dt'] - ntones_df['ToneTime_dt']).dt.total_seconds() < 1.0]
        ntones_df_pat_X_rt = (ntones_df_pat_X['Trial_End_dt'] - ntones_df_pat_X['Gap_Time_dt']).dt.total_seconds()

        ntone_list.extend(ntones_df_pat_X_rt.to_list())




#
# batch_analysis(run, run.aligned, stages, f'{align_pnts[2]}_dt', [[0, f'{align_pnts[2]} '], ],
#                                            list_cond_filts['pat_nonpatt'][0],  list_cond_filts['pat_nonpatt'][1], pmetric=pmetric2use[2],
#                                            filter_df=True, plot=True, sep_cond_cntrl_flag=False, cond_name='pat_nonpatt',
#                                            use4pupil=True, baseline=True, pdr=False, extra_filts=['a1'])