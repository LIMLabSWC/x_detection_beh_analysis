from matplotlib import cm,use

import plotting_functions
from align_functions import get_aligned_events

use('TkAgg')

import align_functions
import pupil_analysis_func

import math
import time
from pupil_analysis_func import Main
from plotting_functions import get_fig_mosaic, plot_ts_var
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import analysis_utils as utils
from copy import deepcopy as copy
from behaviour_analysis import TDAnalysis
import math
import pickle
from datetime import datetime, timedelta
from scipy.signal import find_peaks, find_peaks_cwt
from pupil_analysis_func import batch_analysis, plot_traces, get_subset, glm_from_baseline


if __name__ == "__main__":
    plt.ioff()
    # paradigm = ['altvsrand','normdev']
    # paradigm = ['normdev']
    paradigm = ['familiarity']
    # pkldir = r'c:\bonsai\gd_analysis\pickles'
    # pkldir = r'X:\Dammy\mouse_pupillometry\pickles'
    pkldir = r'D:\bonsai\offline_data'
    # pkl2use = os.path.join(pkldir,'mouse_hf_2309_batch_w_canny_fam_2d_90Hz_hpass01_lpass4hanning015_TOM.pkl')
    pkl2use = os.path.join(pkldir,'mouse_hf_2309_batch_w_canny_DEC_dates_fam_2d_90Hz_hpass01_lpass4hanning025_TOM.pkl')
    # pkl2use = os.path.join(pkldir,'mouse_hf_fam_2d_90Hz_hpass00_lpass4hanning015_TOM.pkl')

    run = Main(pkl2use, (-1.0, 3.0), figdir=rf'figures',fig_ow=False)
    # run_oldmice = Main(r'W:\mouse_pupillometry\pickles\mouse_hf_fam3_2d_90Hz_lpass4_hpass00_hanning025_TOM_w_LR_detrend_wTTL_.pkl',
    #                    (-1.0, 3.0), figdir=rf'W:\mouse_pupillometry\figures\mouse_2305mice_fam',fig_ow=False)
    pmetric2use = ['diameter_2d_zscored','dlc_radii_a_zscored','dlc_EW_zscored','dlc_EW_normed']

    do_baseline = True  # 'rawsize' not in pkl2use
    if 'familiarity' in paradigm:
        run.add_pretone_dt()
        run.aligned = {}
        align_pnts = ['ToneTime','Reward','Gap_Time']
        # dates2plot = ['221214','221215','230116','230117', '230119']
        # dates2plot = ['230116','230117', '230119']

        animals2plot = run.labels
        # animals2plot = ['DO54', 'DO55', 'DO56', 'DO57','DO58','DO59','DO60','DO62']
        # animals2plot = ['DO58','DO59','DO60','DO62']
        dates2plot = run.dates
        stages = [3]
        run.add_stage3_05_order()
        run.add_rolling_mean('Tone_Position',10)

        # dateconds = ['80% Rew 5 uL (day 1)','80% Rew 2 uL','50% Rew 5 uL','80% Rew 5 uL (day 2)','95% Rew 5 uL']
        # dateconds = ['Day 1: 0.9 then 0.1', 'Day 2: 0.9 then 0.1',
        #              'Day 3: 0.9 then 0.1', 'Day 4: 0.9 then 0.1', 'Day 5: 0.9 then 0.1',
        #              'Day 6: 0.1 then 0.9', 'Day 7: 0.1 then 0.9', 'Day 8: Alt vs Rand',
        #              'Day 9: Flat Pattern onset dist', 'Day 10', 'Day 11', 'Day 12', 'Day 13']
        dateconds = run.dates
        # dateconds = ['Day 3: 0.9 then 0.1', 'Day 4: 0.9 then 0.1', 'Day 5: 0.9 then 0.1']

        run.add_diff_col_dt('Trial_Outcome')
        eventnames = [
            # ['0.1','0.5','0.9','control'],
            # ['Early Pattern', 'Late Pattern', 'Middle Presentation'],
            # ['0.1', '0.5', '0.9', 'none'],
            # ['0.5 Block (0.0)', '0.5 Block 1 (0.1)', '0.5 Block 2 (0.9)', 'Control'],
            # ['0.5 Random', '0.5 Alternating', 'Control'],
            # ['0.5 Random', '0.5 Alternating', 'Control'],
            # ['0.5 Random', '0.5 Alternating', 'Control'],
            # ['Correct', 'Incorrect', 'Control'],
            # ['Pattern', 'No Pattern'],
            ['Pattern', 'No Pattern'],
                      ]

        keys = []

        condition_keys = ['p_rate', 'p_onset', 'alt_rand',  'pat_nonpatt_2X',
                          'p_rate_local']
        # condition_keys = ['p_rate','p_rate_local']
        condition_keys_canny = [f'{e}_canny' for e in condition_keys]

        # aligned_pklfile = r'pickles\fm_fam_aligned_nohpass.pkl'
        # aligned_pklfile = r'pickles\DO54_62_aligned_notrend.pkl'
        # aligned_pklfile = r'mouse_hf_2305_batch_no_canny_fam_hpass015.pkl'
        # aligned_pklfile = r'mouse_hf_2309_batch_w_canny_fam_hpass01.pkl'
        aligned_pklfile = r'mouse_hf_2309_batch_DEC_dates_w_canny_fam_hpass01_align.pkl'
        # aligned_pklfile = r'C:\bonsai\gd_analysis\pickles\normdev_2305cohort_aligned.pkl'
        aligned_ow = False
        conditions_class = pupil_analysis_func.PupilEventConditions()
        list_cond_filts = conditions_class.all_filts
        for sess in run.data:
            run.data[sess].trialData['Offset'] = run.data[sess].trialData['Offset'].astype(float) + 0.0

        if os.path.isfile(aligned_pklfile) and aligned_ow is False:
            with open(aligned_pklfile,'rb') as pklfile:
                run.aligned = pickle.load(pklfile)

                keys = [[e] for e in run.aligned.keys()]
        else:
            conditions_class.get_condition_dict(run, condition_keys,stages,)  # 'a1'
            conditions_class.get_condition_dict(run, condition_keys, stages,
                                                pmetric2use='canny_raddi_a_zscored', key_suffix='_canny')

        # with open(aligned_pklfile, 'wb') as pklfile:
        #     pickle.dump(run.aligned,pklfile)

        # run.aligned['alt_rand_nocontrol'] = copy(run.aligned['alt_rand'])
        # run.aligned['alt_rand_nocontrol'][2].pop(2)

        plot = False
        fig_form, chunked_fig_form, n_cols,plt_is = get_fig_mosaic(dates2plot)

        if plot:
            for ki, key2use in enumerate(keys):
                pltsize = (6 * n_cols, 6 * len(chunked_fig_form))
                key_suffix = key2use[0].replace('[','').replace(']','').replace("'",'').replace(', ', '_')
                tsplots_by_dates = plt.subplot_mosaic(fig_form, sharex=False, sharey=True, figsize=pltsize)
                boxplots_by_dates = plt.subplot_mosaic(fig_form, sharex=False, sharey=True, figsize=pltsize)
                trendplots_by_dates = plt.subplot_mosaic(fig_form, sharex=False, sharey=True, figsize=pltsize)
                for plottype,pltfig in zip(['ts',],[tsplots_by_dates,boxplots_by_dates]):
                    for di, date2plot in enumerate(dates2plot):
                        get_subset(run, run.aligned, key2use[0], {'date': [date2plot]},
                                   list_cond_filts[key2use[0]][1],f'{align_pnts[0]} time', plttitle=dateconds[di],
                                   ylabel='Mean of max zscored pupil size for epoch',
                                   xlabel=f'{"Time since Pattern start (s)"*(plottype=="ts")}',
                                   plttype=plottype, pltaxis=(pltfig[0], pltfig[1][plt_is[di]]))
                for di, date2plot in enumerate(dates2plot):
                    get_subset(run, run.aligned, key2use[0], {'date': [date2plot]},
                               list_cond_filts[key2use[0]][1],f'{align_pnts[0]} time', plttitle=dateconds[di],
                               ylabel='Max zscored pupil size for epoch',xlabel=f'Time since Pattern start (s)',
                               plttype='pdelta_trend', pltaxis=(trendplots_by_dates[0], trendplots_by_dates[1][plt_is[di]]))
                               # ['rewarded','not rewarded'],f'{align_pnts[0]} time',extra_filts={'date':date2plot})
                # get_subset(run,run.probreward,"stage1_['a1', 'a0']_Lick_Time_dt_0",{'name':{'date':'221005'},},
            #            ['rewarded','not rewarded'],'Lick time')
                tsplots_by_dates[0].set_size_inches(copy(pltsize))
                tsplots_by_dates[0].savefig(os.path.join(run.figdir, f'alldates_HF_tsplots_EW_{key_suffix}.svg'),
                                            bbox_inches='tight')
                boxplots_by_dates[0].savefig(os.path.join(run.figdir, f'alldates_HF_boxplots_EW_{key_suffix}.svg'),
                                             bbox_inches='tight')
                trendplots_by_dates[0].savefig(os.path.join(run.figdir, f'alldates_HF_pdelta_trendplots_EW_{key_suffix}.svg'),
                                               bbox_inches='tight')

                # animals2plot = run.labels
                tsplots_by_animal = plt.subplots(len(animals2plot), len(dates2plot), squeeze=False, sharex='all',
                                                 sharey='all')
                tsplots_by_animal_ntrials = plt.subplots(len(animals2plot), len(dates2plot), squeeze=False, sharex='all',
                                                         sharey='all')
                histplots_reactiontime = plt.subplots(len(animals2plot), squeeze=False, sharex='all', sharey='all')

                run.pupilts_by_session(run, run.aligned, key2use, animals2plot, dates2plot, eventnames[ki], dateconds,
                                       f'{align_pnts[0]} time', tsplots_by_animal)

                # format plot for saving
                pltsize = (9 * len(dates2plot), 6 * len(animals2plot))
                tsplots_by_animal[0].set_size_inches(pltsize)
                utils.unique_legend(tsplots_by_animal)
                tsplots_by_animal[0].savefig(os.path.join(run.figdir, rf'tspupil_byanimal_{key_suffix}.svg'),
                                             bbox_inches='tight')

        # base_plt_title = 'Evolution of pupil response with successive licks'
        #
        # plot_tsdelta = plt.subplots()
        # utils.plot_eventaligned(run.aligned[keys[3][0]][2],eventnames[3],run.duration,'ToneTime',plot_tsdelta)
        # plot_tsdelta[1].set_ylabel('\u0394 zscored pupil size from condition control')
        # plot_tsdelta[1].set_xlabel('Time since Pattern Onset (s)')
        # plot_tsdelta[0].savefig(os.path.join(run.figdir,'deltacontrols_pupilts_patt_rate.svg'),bbox_inches='tight')

        # pattern non pattern analysis
        pattnonpatt_tsplots = plt.subplots()
        get_subset(run, run.aligned, 'pat_nonpatt_2X', list_cond_filts['pat_nonpatt_2X'][1],
                   list_cond_filts['pat_nonpatt_2X'][1], plttitle='Response to X onset across conditions', plttype='ts',
                   ylabel='zscored pupil size', xlabel=f'Time since X onset (s)',
                   pltaxis=pattnonpatt_tsplots
                   )
        pattnonpatt_tsplots[0].set_size_inches(9,7)
        pattnonpatt_tsplots[0].set_constrained_layout('constrained')
        pattnonpatt_tsplots[0].savefig(os.path.join(run.figdir,'patt_nonpatt_ts_allltrials.svg'),
                                       bbox_inches='tight')
        # utils.ts_permutation_test(run.aligned['pat_nonpatt_2X'][2],500,0.95,1,pattnonpatt_tsplots,run.duration)
        pattnonpatt_tsplots[0].show()

        # prate analysis
        # p_rate_dates = ['230214', '230216', '230221', '230222', '230113', ]  # '230223', '230224'
        # p_rate_dates= [
        #     '230531',
        #     '230601',
        #     '230602',
        #     '230605',
        #     '230606',
        #     '230607',
        #     '230608',   # muscimol day (64, 69)
        #     '230609',
        #     '230717',
        #     '230718',
        #     '230719',  # muscimol day 2 0.5 uL dose (64,69,70)
        #     '230720',
        #     '230721',
        #     '230724',
        #     '230725',
        #     '230804',  # muscimol 1 ul/ul
        #     ]
        plt.close('all')
        p_rate_dates=run.dates
        p_rate_tsplots = plt.subplots(figsize=(9,7))

        run.subsets['prate_rare_freq'] = get_subset(run,run.aligned,'p_rate_local',{'date':p_rate_dates}, events=list_cond_filts['p_rate_local'][1],
                                                    beh=f'{align_pnts[0]} onset',
                                                    plttitle='Response to Pattern onset across conditions',
                                                    plttype='ts',
                                                    ylabel='zscored pupil size', xlabel=f'Time since Pattern Onset (s)',
                                                    pltaxis=p_rate_tsplots, exclude_idx=[1, 2, 3], ctrl_idx=3,
                                                    alt_cond_names=['rare', 'frequent', 'none']
                                                    )
        p_rate_tsplots[0].show()
        p_rate_tsplots[0].set_constrained_layout('constrained')

        utils.ts_permutation_test(run.subsets['prate_rare_freq'][2],500,0.95,3,p_rate_tsplots,run.duration)
        utils.ts_two_tailed_ht(run.subsets['prate_rare_freq'][2],0.95,3,p_rate_tsplots,run.duration)
        p_rate_tsplots[0].show()

        p_rate_over_dates_tsplot = plt.subplots(ncols=len(p_rate_dates),squeeze=False,sharey='all')
        for di,date in enumerate(p_rate_dates):
            get_subset(run, run.aligned, 'p_rate_local_canny', {'date': date}, events=list_cond_filts['p_rate_local'][1],
                       beh=f'{align_pnts[0]} onset', plttitle='Response to Pattern onset across conditions',
                       plttype='ts',
                       ylabel='zscored pupil size', xlabel=f'Time since Pattern Onset (s)',
                       pltaxis=(p_rate_over_dates_tsplot[0],p_rate_over_dates_tsplot[1][0,di]),
                       exclude_idx=[None], ctrl_idx=3
                       )
        p_rate_over_dates_tsplot[0].set_size_inches(30,7)
        p_rate_over_dates_tsplot[0].show()
        # example multiple prate plots over dates
        prate_example_dates = p_rate_dates
        ncols = 4
        prate_multiple_dates_plot = plt.subplots(ncols=ncols,nrows=math.ceil(len(prate_example_dates)/ncols),
                                                 figsize=(9*ncols,21),sharey='all',squeeze=False)
        for di,date2plot in enumerate(prate_example_dates):
            prate_aligned = get_subset(run, run.aligned, 'p_rate_local', {'date': date2plot}, events=list_cond_filts['p_rate_local'][1],
                                       beh=f'{align_pnts[0]} onset',
                                       plttitle=f'Response to pattern onset {date2plot}', plttype='ts',
                                       ylabel='zscored pupil size', xlabel=f'Time since pattern onset (s)',
                                       pltaxis=(prate_multiple_dates_plot[0],
                                                prate_multiple_dates_plot[1][int(di/ncols),di%ncols]),
                                       )
        prate_multiple_dates_plot[0].set_constrained_layout('constrained')
        prate_multiple_dates_plot[0].show()

        # muscimol prate analysis
        p_rate_dates=run.dates
        prate_musc_tsplots = plt.subplots(ncols=2,squeeze=False,figsize=(9*3,7),sharey='all')
        # prate_muscimol_dates = ['230608','230719','230804']
        # prate_muscimol_dates = ['230918','230920','230927','230929','231002','231030','231103']
        prate_muscimol_dates = ['230918','230920','230927','230929','231002','231030','231103',
                                '231128','231201','231206']
        # prate_saline_dates = ['230928','231024','231027','231102']
        prate_saline_dates = ['230928','231024','231027','231102','231204']
        prate_control_dates = [d for d in p_rate_dates if d not in prate_muscimol_dates+prate_saline_dates]
        prate_control_dates = [d for d in prate_control_dates if all(int(d) - np.array(prate_muscimol_dates).astype(int) != -1)]
        # prate_control_dates.remove('231031')
        muscimol_analysis_dfs = []
        for subset_ix, (subset_dates,subset_name) in enumerate(zip([prate_muscimol_dates,prate_control_dates,prate_saline_dates],
                                                                   ['muscimol', 'control','saline'])):
            run.subsets[f'{subset_name}_2patt'] = get_subset(run, run.aligned, 'p_rate_local_canny',{'date':subset_dates,'name':['DO71','DO72','DO75']},
                                                             events=list_cond_filts['p_rate_local'][1],
                                                             beh=f'{align_pnts[0]} onset',
                                                             plttitle=f'Response to pattern onset {subset_name}',
                                                             plttype='ts',
                                                             ylabel='zscored pupil size',
                                                             xlabel=f'Time since pattern onset (s)',
                                                             exclude_idx=(1, 2, 3),
                                                             alt_cond_names=['rare','frequent','none']
                             # pltaxis=(prate_musc_tsplots[0],
                             #          prate_musc_tsplots[1][0, subset_ix]),
                             )
            muscimol_analysis_dfs.append(run.subsets[f'{subset_name}_2patt'][2])

        prate_musc_tsplots[0].show()

        # sess_delta_plot = plt.subplots()
        musc_sal_ctrl_tsplot = plt.subplots(figsize=(9,7))
        rare_freq_delta_tsplot = plt.subplots(figsize=(9,7))
        for cond_i, (cond_dfs,cond_name,ls) in enumerate(zip(muscimol_analysis_dfs,['muscimol','control',],['-','--',':'])):
            rare_df, freq_df, none_df = copy(cond_dfs)
            none_df.index = none_df.index.droplevel('time')
            for df_i, (df,df_name) in enumerate(zip([rare_df,freq_df],['rare','frequent'])):
                df.index = df.index.droplevel('time')
                # for u_idx in df.index.unique():
                    # df.loc[u_idx] = df.loc[u_idx] - none_df.loc[u_idx].median(axis=0)
                musc_sal_ctrl_tsplot[1].plot(none_df.columns, df.mean(axis=0),
                                             c=f'C{df_i}',ls=ls,label=f'{cond_name}: {df_name}')
            delta_dfs = [rare_df.loc[u_idx].mean(axis=0)-freq_df.loc[u_idx].mean(axis=0) for u_idx in rare_df.index.unique() if u_idx in freq_df.index]
            # delta_means = [(rare_df.loc[u_idx]-freq_df.loc[u_idx]).mean(axis=0) for u_idx in rare_df.index.unique() if u_idx in freq_df.index]
            rare_freq_delta_tsplot[1].plot(none_df.columns, np.array(delta_dfs).mean(axis=0),
                                           label=f'{cond_name}')
            plot_ts_var(none_df.columns,np.array(delta_dfs),f'C{cond_i}',rare_freq_delta_tsplot[1])

        rare_freq_delta_tsplot[0].show()

        musc_sal_ctrl_tsplot[1].legend()
        musc_sal_ctrl_tsplot[1].set_ylabel('zscored difference in pupil size from none trials')
        musc_sal_ctrl_tsplot[1].set_xlabel('Time since pattern onset (s)')
        musc_sal_ctrl_tsplot[1].axvline(0,c='k',ls='--')
        musc_sal_ctrl_tsplot[0].set_constrained_layout('constrained')
        musc_sal_ctrl_tsplot[0].show()

        prate_musc_tsplots[0].set_size_inches(18, 7)
        prate_musc_tsplots[0].show()
        prate_musc_tsplots[1][0, 1].set_ylabel('')
        prate_musc_tsplots[0].set_constrained_layout('constrained')

        # plot rare/freq musc vs saline
        reordered_trial_types = dict()
        sess_subsets_dfs = [run.subsets[s_key][2] for s_key in
                            ['muscimol_2patt', 'saline_2patt', 'non muscimol_2patt']]
        for cond_i, cond in enumerate(['rare','frequent',]):
            reordered_trial_types[cond] = []
            for sess_type_i, sess_type in enumerate(['muscimol','saline','none']):
                reordered_trial_types[cond].append(sess_subsets_dfs[sess_type_i][cond_i]-sess_subsets_dfs[sess_type_i][-1].mean(axis=0))
            run.subsets[cond+'delta'] = None, None, reordered_trial_types[cond], ['muscimol','saline', 'control']

        run.dump_trial_pupil_arr()


        musc_nonmusc_2x_tsplot = plt.subplots()
        pltargs = [['-',1],['--',1]]
        for subset_ix, (subset_dates,subset_name) in enumerate(zip([prate_muscimol_dates,prate_control_dates],
                                                                   ['muscimol', 'non muscimol'])):
            get_subset(run, run.aligned, 'pat_nonpatt_2X_canny',{'date':subset_dates,'name':['DO71','DO72','DO75']},
                       # events=list_cond_filts['p_rate_local'][1],
                       events= [f'{e} {subset_name}' for e in list_cond_filts['pat_nonpatt_2X'][1]],
                       beh=f'{align_pnts[2]} onset',
                       plttitle=f'Response to X onset {subset_name}', plttype='ts',
                       ylabel='zscored pupil size', xlabel=f'Time since pattern onset (s)',
                       # exclude_idx=(1,2,3),
                       pltargs=pltargs[subset_ix],
                       pltaxis=musc_nonmusc_2x_tsplot,
                       )
        musc_nonmusc_2x_tsplot[0].set_size_inches(9,7)
        musc_nonmusc_2x_tsplot[0].set_constrained_layout('constrained')
        musc_nonmusc_2x_tsplot[0].show()
        # bootstrap for n days:
        n_bootstrap_repeats = 10
        rand_subset_dates = [np.random.choice(prate_control_dates,len(prate_muscimol_dates), replace=True)
                             for n in range(n_bootstrap_repeats)]
        prate_rand_subset_tsplots = plt.subplots()
        ts_traces_randsubsets = []

        for subset_dates in rand_subset_dates:
            subset_data = get_subset(run, run.aligned, 'p_rate_local', {'date': subset_dates},
                       events=list_cond_filts['p_rate_local'][1],
                       beh=f'{align_pnts[0]} onset',
                       plttitle=f'Response to pattern onset (rand 3 day subsets)', plttype='ts',
                       ylabel='zscored pupil size', xlabel=f'Time since pattern onset (s)',)[2]
            subset_mean_traces = np.array([cond_traces.mean(axis=0) for cond_traces in subset_data])
            ts_traces_randsubsets.append(subset_mean_traces)
        all_subset_means = np.array(ts_traces_randsubsets)
        # all_subset_means = all_subset_means.mean(axis=0)

        ci = np.apply_along_axis(utils.mean_confidence_interval, axis=0, arr=all_subset_means)
        # ci = np.apply_along_axis(manual_confidence_interval, axis=0, arr=rand_npdample)
        # plot[1].plot(ci[0, :])
        col_str = [f'C{i}' if ii != 'control' else 'k' for i,ii in enumerate(list_cond_filts['p_rate_local'][1]) ]
        x_axis = run.aligned['p_rate_local'][2][0].columns
        for cond_i,cond_name in enumerate(list_cond_filts['p_rate_local'][1]):
            if cond_i not in [0,4,5,]:
                continue
            prate_rand_subset_tsplots[1].plot(x_axis,all_subset_means[:,cond_i,:].mean(axis=0),c=col_str[cond_i],label=cond_name)
            prate_musc_tsplots[1][0,2].plot(x_axis,all_subset_means[:,cond_i,:].mean(axis=0),c=col_str[cond_i],label=cond_name)
            prate_rand_subset_tsplots[1].fill_between(x_axis, ci[1,cond_i,:], ci[2,cond_i,:], alpha=0.1, facecolor=col_str[cond_i])
            prate_musc_tsplots[1][0, 2].fill_between(x_axis, ci[1,cond_i,:], ci[2,cond_i,:], alpha=0.1, facecolor=col_str[cond_i])
        prate_musc_tsplots[1][0, 2].set_title('Mean response for random 3 session subsets (100 shuffles) ')
        prate_musc_tsplots[1][0, 2].set_xlabel('Time since pattern onset (s)')
        utils.unique_legend((prate_musc_tsplots[0],prate_musc_tsplots[1][0, 2]))

        # prate_rand_subset_tsplots[0].show()
        # prate_musc_tsplots[0].set_constrained_layout('constrained')
        # prate_musc_tsplots[0].show()
        # conditions_class.get_condition_dict(run_oldmice, ['p_rate'], stages, extra_filts=['a1'])
        # old_vs_new_tsplot = plt.subplots(ncols=2,sharey='all')
        # prate_aligned_old = get_subset(run_oldmice, run_oldmice.aligned, 'p_rate',{'date':['230214', '230216', '230221', '230222', '230113', ]},
        #                                events=list_cond_filts['p_rate'][1],
        #                                beh=f'{align_pnts[0]} onset',
        #                                plttitle=f'Response to pattern onset', plttype='ts',
        #                                ylabel='zscored pupil size', xlabel=f'Time since pattern onset (s)',
        #                                pltaxis=(old_vs_new_tsplot[0],
        #                                         old_vs_new_tsplot[1][0]),
        #                                )
        # old_vs_new_tsplot[0].show()

        print('break now if wanted')
        time.sleep(30)


        # alt rand analysis
        alt_rand_sessnames = []
        alt_sesses = []
        # get sessions with alternating pattern trials
        for sess in run.data:
            altsess = align_functions.filter_df(run.data[sess].trialData, ['s1', 'a1', 'e!0'])
            if altsess.shape[0] > 10:
                alt_sesses.append(altsess)
                alt_rand_sessnames.extend(altsess.index.to_series().unique())
        alt_rand_sessnames = np.array(alt_rand_sessnames)
        alt_rand_dates = np.unique(alt_rand_sessnames[:,1])

        altrand_plot = plt.subplots(figsize=(9, 6), squeeze=False)
        altrand_plot_bydates = plt.subplots(ncols=len(alt_rand_dates),figsize=(6*len(alt_rand_dates),6),squeeze=False)
        altrand_plot_pdelta = plt.subplots(ncols=1,figsize=(18, 6), squeeze=False)

        for di, date in enumerate(alt_rand_dates):
            get_subset(run, run.aligned, 'alt_rand', {'date': date,
                                                      'name' : alt_rand_sessnames[alt_rand_sessnames[:,1]==date][:,0]},
                       list_cond_filts['alt_rand'][1], f'{align_pnts[0]} time', plttitle=date,
                       ylabel='zscored pupil size', xlabel=f'Time since Pattern start (s)',
                       plttype='ts', pltaxis=(altrand_plot_bydates[0], altrand_plot_bydates[1][0, di]))

        get_subset(run, run.aligned, 'alt_rand', {'date': alt_rand_dates},
                   list_cond_filts['alt_rand'][1], f'{align_pnts[0]} time', plttitle='Alt vs Rand over days',
                   ylabel='zscored pupil size', xlabel=f'Time since Pattern start (s)',
                   plttype='ts', pltaxis=(altrand_plot[0], altrand_plot[1][0, 0]),)

        get_subset(run, run.aligned, 'alt_rand_ctrl_sub', {'date': ['230217','230303']},
                   list_cond_filts['alt_rand'][1], f'pattern', plttitle='Alternate vs Random across sessions',
                   ylabel='\u0394 zscored pupil response', xlabel=f'Time since pattern onset (s)',
                   plttype='ts', pltaxis=(altrand_plot_pdelta[0], altrand_plot_pdelta[1][0, 0]), )

        for pltax,figname in zip((altrand_plot,altrand_plot_bydates,altrand_plot_pdelta),('alt_rand','alt_rand_bydate','alt_rand_pdelta')):
            utils.unique_legend(pltax)
            pltax[0].savefig(os.path.join(run.figdir, f'{figname}_pupilts.svg'),
                             bbox_inches='tight')


        altrand_plot[0].show()
        altrand_plot_bydates[0].show()

        altrand_plot_pdelta = plt.subplots(ncols=2,figsize=(18, 7), squeeze=False)
        altrand_by_portion = plt.subplots(ncols=2, figsize=(18, 7), squeeze=False)
        alt_dates = ['230217','230303']
        plt_arr = []
        for di, alt_date in enumerate(alt_dates):
            get_subset(run, run.aligned, 'alt_rand', {'date': [alt_date]},
                       list_cond_filts['alt_rand'][1], f'{align_pnts[0]} time', plttitle=f'Alternate vs Random: Example day {di+1}',
                       ylabel='zscored pupil size', xlabel=f'Time since pattern onset (s)',
                       plttype='ts', pltaxis=(altrand_plot_pdelta[0], altrand_plot_pdelta[1][0,di]))
            altrand_plot_pdelta[0].show()

            for portion, portion_ls,portion_name,portion_start in zip([0.33, 0.33, 0.33,], ['-', '--',':','-.'],
                                                                      ['1st third', '2nd third','3rd third','4th quarter',],
                                                                      [0, 0.33, 0.66, 0.75]):
                if portion_start == 0.33:
                    continue
                half_eventname = [f'{e} {portion_name}' for e in list_cond_filts['alt_rand'][1]]
                plt_arr.append(get_subset(run, run.aligned, 'alt_rand_ctrl_sub', {'date': [alt_date]},
                           half_eventname, f'{align_pnts[0]} time', plttitle='Alt vs Rand by third',
                           ylabel='delta zscored pupil size', xlabel=f'Time since Pattern start (s)',
                           ntrials=portion, ntrial_start_idx=portion_start,
                           plttype='ts', pltaxis=(altrand_by_portion[0], altrand_by_portion[1][0, di]),
                           pltargs=(portion_ls, None),exclude_idx=[2])[2])

        utils.unique_legend(altrand_plot_pdelta),utils.unique_legend(altrand_by_portion)
        altrand_by_portion[0].show()
        altrand_plot_pdelta[0].savefig(os.path.join(run.figdir,f'pupilts_altvsrand.svg'),bbox_inches='tight')
        utils.ts_permutation_test([plt_arr[1][0],plt_arr[0][1],],10000,0.95,pltax=(altrand_by_portion[0],altrand_by_portion[1][0,0]),cnt_idx=0,ts_window=run.duration)
        altrand_by_portion[0].savefig(os.path.join(run.figdir, f'byportion_pupilts_altvsrand.svg'),
                                      bbox_inches='tight')

        # run.get_pdr(run.aligned['p_rate'][2], 'ToneTime', smooth=False,plot=True,plotlabels=list_cond_filts['p_rate'][1],)
        altrand_plot_pdr = plt.subplots()
        get_subset(run, run.aligned, 'alt_rand_ctrl_sub', {'date': ['230217','230303']},
                   list_cond_filts['alt_rand'][1], f'Pattern time', plttitle='1 Day, 8 mice',
                   ylabel='zscored pupil size', xlabel=f'Time since Pattern onset (s)', pdr=False,
                   plttype='ts', pltaxis=(altrand_plot_pdr[0], altrand_plot_pdr[1]))
        altrand_plot_pdr[0].show()
        # dates2plot = ['230126']
        animal_date_pltform = {'ylabel': 'z-scored pupil size',
                               'xlabel': 'Time since Pattern Onset',
                               'figtitle':base_plt_title,
                               'rowtitles': animals2plot,
                               'coltitles': dates2plot,
                               }
        indvtraces_nonbinned = plot_traces(animals2plot,dates2plot,run.aligned[keys[1][0]],run.duration,run.samplerate,
                                           plotformatdict=animal_date_pltform,control_idx=2)

        binsize = 5
        key_ix = 1
        for i,cond in enumerate(eventnames[key_ix]):
            animal_date_pltform['figtitle'] = f"{base_plt_title} binned {binsize} trials: {cond}"
            indvtraces_binned = plot_traces(animals2plot,dates2plot,run.aligned[keys[key_ix][0]], run.duration,run.samplerate,
                                            plotformatdict=animal_date_pltform,binsize=binsize,cond_subset=[i],
                                            control_idx=None)
            indvtraces_binned[0].set_size_inches(9 * len(dates2plot), 6 * len(animals2plot))
            indvtraces_binned[0].savefig(os.path.join(run.figdir, f'pupilts_binned_evolution_{cond}.svg'),
                                         bbox_inches='tight')

        # look at baseline over sessions
        all_sess_df = pd.concat(run.aligned['alt_rand'][2])
        all_baselines = all_sess_df.loc[:,-1.0:0.0].mean(axis=1)
        all_baselines_plot = plt.subplots(figsize=(15,6))
        x_pos = 0
        for ai,animal in enumerate(animals2plot):
            for date in dates2plot:
                data2plot = all_baselines.loc[:,animal,date].to_numpy()
                all_baselines_plot[1].plot(np.arange(x_pos,x_pos+len(data2plot)),data2plot,c=f'C{ai}')
                all_baselines_plot[1].axvline(x_pos+data2plot.shape[0],c='grey',ls='--')
                x_pos += data2plot.shape[0]
        all_baselines_plot[1].set_ylabel('relative pupil size (px)')
        all_baselines_plot[1].set_xlabel('Trials')
        all_baselines_plot[1].set_title('Baseline mean across trials for all sessions')
        all_baselines_plot[0].show()
        all_baselines_plot[0].savefig(os.path.join(run.figdir,'baseline_over_sessions.svg'),bbox_inches='tight')
    do_harp_stuff = False
    if do_harp_stuff:
        plt.ioff()
        list_dfs = utils.merge_sessions(r'c:\bonsai\data\Dammy',run.labels,'TrialData',[run.dates[0],run.dates[-1]])
        run.trialData = pd.concat(list_dfs)
        run.trialData = run.trialData.iloc[:,:-6]
        run.trialData.dropna(inplace=True)
        for col in run.trialData.keys():
            if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1:
                if col.find('Wait') == -1 and col.find('dt') == -1 and col.find('Lick_Times') == -1 and col.find(
                        'Cross') == -1:
                    utils.add_datetimecol(run.trialData, col)
        run.get_aligned_events = get_aligned_events
        run.trialData.set_index('Trial_Start_dt',append=True,inplace=True,drop=False)
        run.trialData['Pretone_end_dt'] = [tstart + timedelta(0, predur) for tstart, predur in
                                   zip(run.trialData['Trial_Start_dt'], run.trialData['PreTone_Duration'])]

        harpmatrices_pkl = os.path.join(pkldir,'mousefam_hf_harps_matrices_allfam2.pkl')
        if os.path.isfile(harpmatrices_pkl):
            with open(harpmatrices_pkl, 'rb') as pklfile:
                run.harpmatrices = pickle.load(pklfile)
        else:
            run.harpmatrices = align_functions.get_event_matrix(run, run.data, r'W:\mouse_pupillometry\mouse_hf\harpbins', )
            with open(harpmatrices_pkl, 'wb') as pklfile:
                pickle.dump(run.harpmatrices,pklfile)
        fig,ax = plt.subplots()
        run.lickrasters_firstlick = {}
        for outcome in [['a1'],['a0']]:
            run.animals = run.labels
            run.lickrasters_firstlick[outcome[0]] = run.get_aligned_events(run,'Pretone_end_dt',0,(-1.0,3.0),byoutcome_flag=True,outcome2filt=outcome)
            run.lickrasters_firstlick[outcome[0]][0].set_size_inches((12,9))
            run.lickrasters_firstlick[outcome[0]][0].savefig(rf'W:\mouse_pupillometry\figures\probrewardplots\alldates_HF_lickraster_EW_{outcome}.svg')

        for outcome in [['a1'],['a0']]:
            binsize= 500
            prob_lick_mat = run.lickrasters_firstlick[outcome[0]][2].fillna(0).rolling(binsize,axis=1).mean()  # .mean().iloc[:,binsize - 1::binsize]
            prob_lick_mean = prob_lick_mat.mean(axis=0)
            ax.plot(prob_lick_mean.index,prob_lick_mean,label=outcome[0])
        ax.set_xlabel('seconds from Trial Start')
        ax.set_ylabel('mean lick rate across animals across sessions')
        ax.set_title('Lick rate aligned to Trial Start, 0.1s bin')
        ax.legend()
        ax.axvline(0.0,ls='--',c='k',lw=0.25)
        fig.set_size_inches((15,12))
        fig.savefig(r'W:\mouse_pupillometry\figures\probrewardplots\alldates_HF_lickrate_EW.svg',bbox_inches='tight')

        pattern_hist = plt.subplots()
        p09_df = align_functions.filter_df(run.trialData, ['phigh', 'e!0']).loc[:, '230221', :]
        p05_df = align_functions.filter_df(run.trialData, ['p0.5', 'e!0']).loc[:, '230221', :]
        onset_times,onset_counts = np.unique([p05_df.PreTone_Duration.to_list() +
                                             p09_df.PreTone_Duration.to_list()], return_counts=True)
        pattern_hist[1].bar(onset_times, onset_counts/np.sum(onset_counts), align='center')
        pattern_hist[1].set_xlabel('Pattern embed time from Trial Start (s)')
        pattern_hist[1].set_xticks([1,2,3,4,5],[1,2,3,4,5])
        pattern_hist[1].set_ylabel('Proportion of Trials')
        pattern_hist[1].set_title('Distribution of pattern onset times')
        pattern_hist[0].savefig(os.path.join(run.figdir,'pattern_time_dist.svg'),bbox_inches='tight')

