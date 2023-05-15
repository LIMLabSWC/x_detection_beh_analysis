import time

import matplotlib.colors
import statsmodels.api as sm
from pupil_analysis_func import Main, get_fig_mosaic
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import os
import analysis_utils as utils
from copy import deepcopy as copy
from behaviour_analysis import TDAnalysis
import pickle
from datetime import datetime, timedelta
from scipy.signal import find_peaks, find_peaks_cwt
import ruptures as rpt
from pupil_analysis_func import batch_analysis, plot_traces, get_subset, glm_from_baseline


if __name__ == "__main__":
    plt.ioff()
    # paradigm = ['altvsrand','normdev']
    # paradigm = ['normdev']
    paradigm = ['familiarity']
    pkldir = r'c:\bonsai\gd_analysis\pickles'
    pkl2use = os.path.join(pkldir,'mouse_fm_fam_2d_90Hz_hpass00_hanning025_detrend.pkl')
    # pkl2use = os.path.join(pkldir,r'mouseprobreward_2d_90Hz_6lpass_025hpass_wdlc_TOM_interpol_all_int02s_221028.pkl')

    run = Main(pkl2use, (-1.0, 3.0), figdir=rf'W:\mouse_pupillometry\figures\mouse_fm_fam',fig_ow=False)
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
        list_cond_filts = {
            'p_rate': [[['plow'],['p0.5'],['phigh'],['none']],
                       ['0.1','0.5','0.9','control']],
            'p_rate_ctrl': [[['plow'], ['plow', 'e=0'], ['p0.5'], ['p0.5', 'e=0'], ['phigh'], ['phigh', 'e=0']],
                            ['0.1','0.1 cntrl','0.5','0.5 cntrl','0.9','0.9 cntrl','control']],
            'p_onset': [[['dearly','p0.5'], ['dlate','p0.5'], ['dmid','p0.5']],
                        ['Early Pattern', 'Late Pattern','Middle Presentation']],
            # 'p0.5_block': [[['0.5_0','p0.5'], ['0.5_1','p0.5'], ['0.5_2','p0.5'], ['none']],
            #                ['0.5 Block (0.0)', '0.5 Block 1 (0.1)', '0.5 Block 2 (0.9)', 'Control']],
            'alt_rand': [[['s0', 'p0.5'], ['s1','p0.5'], ['none','p0.5']],
                         ['0.5 Random', '0.5 Alternating', 'Control']],
            'alt_rand_ctrl': [[['s0','p0.5'], ['s0', 'e=0'],  ['s1','p0.5'], ['s1','p0.5', 'e=0'], ['none','p0.5']],
                              ['0.5 Random','0.5 Random ctrl', '0.5 Alternating', '0.5 Alternating ctrl', 'Control']],
            # 'ntones': [[['p0.5','tones4'],['p0.5','tones3'],['p0.5','tones2'],['p0.5','tones1']],['ABCD', 'ABC','AB','A']],
            # 'pat_nonpatt': [[['e!0'],['e=0']],['Pattern Sequence Trials','No Pattern Sequence Trials']],
            'pat_nonpatt_2X': [[['e!0'], ['e=0']], ['Pattern Sequence Trials', 'No Pattern Sequence Trials']],
            'p_rate_fm': [[['plow'], ['pmed'], ['phigh'],['ppost'], ['none']],
                       ['0.1', '0.5', '0.9','0.6', 'control']],

        }

        aligned_pklfile = r'pickles\fm_fam_aligned.pkl'
        # aligned_pklfile = r'pickles\DO54_62_aligned_notrend.pkl'
        aligned_ow = True
        if os.path.isfile(aligned_pklfile) and aligned_ow is False:
            with open(aligned_pklfile,'rb') as pklfile:
                run.aligned = pickle.load(pklfile)

                keys = [[e] for e in run.aligned.keys()]
        else:
            for cond_i, (cond_filts,cond_key) in enumerate(zip(list_cond_filts.values(),list_cond_filts.keys())):
                keys.append(batch_analysis(run, run.aligned, stages, f'{align_pnts[0]}_dt', [[0, f'{align_pnts[0]}'], ],
                                           cond_filts[0], cond_filts[1], pmetric=pmetric2use[2],
                                           filter_df=True, plot=True, sep_cond_cntrl_flag=False, cond_name=cond_key,
                                           use4pupil=True, baseline=do_baseline, pdr=False, extra_filts=['a1']))  #'a1'

            # subtract control traces and plot
            for key in ['p_rate_ctrl','alt_rand_ctrl']:
                run.aligned[f'{key}_sub'] = copy(run.aligned[key])
                keys.append([list(run.aligned.keys())[-1]])
                for ti,tone_df in enumerate(run.aligned[key][2]):
                    if (ti%2 == 0 or ti == 0) and ti < len(run.aligned[key][2])-1:
                        print(ti)
                        control_tone_df = run.aligned[key][2][ti+1].copy()
                        for sess_idx in tone_df.index.droplevel('time').unique():
                            sess_ctrl_mean = control_tone_df.loc[:,[sess_idx[0]],[sess_idx[1]]].mean(axis=0)
                            tone_df.loc[:,sess_idx[0],sess_idx[1]] = tone_df.loc[:,[sess_idx[0]],[sess_idx[1]]]-sess_ctrl_mean
                        # run.aligned[f'{key}_sub'][2][ti] = copy(tone_df)-run.aligned[key][2][ti+1].mean(axis=0)
                        run.aligned[f'{key}_sub'][2][ti] = copy(tone_df)
                for idx in [1,2]:
                    if idx<(len(run.aligned[key][2])):
                        run.aligned[f'{key}_sub'][2].pop(idx)

            with open(aligned_pklfile, 'wb') as pklfile:
                pickle.dump(run.aligned,pklfile)

        # keys.append(batch_analysis(run, run.aligned, stages, f'{align_pnts[0]}_dt', [[0, f'{align_pnts[0]} '], ],
        #                            [['dearly'], ['dlate'], ['dmid']], eventnames[1], pmetric=pmetric2use[2],
        #                            filter_df=True, plot=True,
        #                            use4pupil=True, baseline=do_baseline, pdr=False,
        #                            extra_filts=['tones4', 'p0.5']))
        # keys.append(batch_analysis(run, run.aligned, stages, f'{align_pnts[0]}_dt', [[0, f'{align_pnts[0]} '], ],
        #                            [['plow'],['plow','e=0'],['p0.5'],['p0.5','e=0'],['phigh'],['phigh','e=0'],],
        #                            eventnames[2], pmetric=pmetric2use[3],
        #                            filter_df=True, plot=False,
        #                            use4pupil=True, baseline=do_baseline, pdr=False,
        #                            extra_filts=['tones4']))
        #
        # # # subtract control traces and plot
        # run.aligned[f'{keys[2][0]}_nocontrols'] = copy(run.aligned[keys[2][0]])
        # keys.append([list(run.aligned.keys())[-1]])
        # for ti,tone_df in enumerate(run.aligned[keys[2][0]][2]):
        #     if ti%2 == 0 or ti == 0:
        #         run.aligned[keys[3][0]][2][ti] = copy(tone_df)-run.aligned[keys[2][0]][2][ti+1].mean(axis=0)
        # for idx in [1,2,3]:
        #     run.aligned[keys[3][0]][2].pop(idx)
        #
        # run.add_stage3_05_order()
        # keys.append(batch_analysis(run, run.aligned, stages, f'{align_pnts[0]}_dt', [[0, f'{align_pnts[0]} '], ],
        #                            [['0.5_0'], ['0.5_1'], ['0.5_2'], ['none']],
        #                            eventnames[2], pmetric=pmetric2use[2],
        #                            filter_df=True, plot=True,
        #                            use4pupil=True, baseline=do_baseline, pdr=False,
        #                            extra_filts=['tones4', 'p0.5']))
        # keys.append(batch_analysis(run, run.aligned, stages, f'{align_pnts[0]}_dt', [[0, f'{align_pnts[0]} '], ],
        #                            [['s0'], ['s1'],['none']],
        #                            eventnames[5], pmetric=pmetric2use[2],
        #                            filter_df=True, plot=False,
        #                            use4pupil=True, baseline=do_baseline, pdr=False,
        #                            extra_filts=['tones4',  'p0.5']))
        # keys.append(batch_analysis(run, run.aligned, stages, f'{align_pnts[0]}_dt', [[0, f'{align_pnts[0]} '], ],
        #                            [['s0'],['s0','e=0'], ['s1'],['s1','e=0'],['none']],
        #                            eventnames[5], pmetric=pmetric2use[2],
        #                            filter_df=True, plot=False,
        #                            use4pupil=True, baseline=do_baseline, pdr=False,
        #                            extra_filts=['tones4', 'p0.5']))
        # # subtract control traces and plot
        # run.aligned[f'{keys[6][0]}_nocontrols'] = copy(run.aligned[keys[6][0]])
        # keys.append([list(run.aligned.keys())[-1]])
        # for ti,tone_df in enumerate(run.aligned[keys[6][0]][2][:-1]):
        #     if ti%2 == 0 or ti == 0:
        #         run.aligned[keys[7][0]][2][ti] = copy(tone_df)-run.aligned[keys[6][0]][2][ti+1].mean(axis=0)
        # for idx in [1,2]:
        #     run.aligned[keys[7][0]][2].pop(idx)

        run.aligned['alt_rand_nocontrol'] = copy(run.aligned['alt_rand'])
        run.aligned['alt_rand_nocontrol'][2].pop(2)

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

        base_plt_title = 'Evolution of pupil response with successive licks'

        plot_tsdelta = plt.subplots()
        utils.plot_eventaligned(run.aligned[keys[3][0]][2],eventnames[3],run.duration,'ToneTime',plot_tsdelta)
        plot_tsdelta[1].set_ylabel('\u0394 zscored pupil size from condition control')
        plot_tsdelta[1].set_xlabel('Time since Pattern Onset (s)')
        plot_tsdelta[0].savefig(os.path.join(run.figdir,'deltacontrols_pupilts_patt_rate.svg'),bbox_inches='tight')

        # pattern non pattern ananlysis
        pattnonpatt_tsplots = plt.subplots()
        get_subset(run, run.aligned, 'pat_nonpatt', list_cond_filts['pat_nonpatt'][1],
                   list_cond_filts['pat_nonpatt'][1], plttitle='Response to Pattern onset across conditions', plttype='ts',
                   ylabel='zscored pupil size', xlabel=f'Time since Pattern start (s)',
                   pltaxis=pattnonpatt_tsplots
                   )
        pattnonpatt_tsplots[0].set_size_inches(9,6)
        pattnonpatt_tsplots[0].set_constrained_layout('constrained')
        pattnonpatt_tsplots[0].savefig(os.path.join(run.figdir,'patt_nonpatt_ts_allltrials.svg'),
                                       bbox_inches='tight')
        utils.ts_permutation_test(run.aligned['pat_nonpatt'][2],500,0.95,1,pattnonpatt_tsplots,run.duration)
        pattnonpatt_tsplots[0].show()

        # prate analysis
        p_rate_dates = ['230214', '230216', '230221', '230222', '230113', ]  # '230223', '230224'
        p_rate_tsplots = plt.subplots(figsize=(9,7))

        prate_aligned = get_subset(run,run.aligned,'p_rate',{None:None}, events=list_cond_filts['p_rate'][1],
                   beh=f'{align_pnts[0]} onset', plttitle='Response to Pattern onset across conditions', plttype='pdelta_trend',
                   ylabel='zscored pupil size', xlabel=f'Time since Pattern Onset (s)',
                   pltaxis=p_rate_tsplots,exclude_idx=[None],ctrl_idx=3
                   )
        p_rate_tsplots[0].show()
        utils.ts_permutation_test(prate_aligned[2],50000,0.95,3,p_rate_tsplots,run.duration)
        p_rate_tsplots[0].show()

        # example multiple prate plots over dates
        prate_example_dates = [p_rate_dates[-1],p_rate_dates[0],p_rate_dates[-2]]
        prate_multiple_dates_plot = plt.subplots(ncols=len(prate_example_dates), figsize=(9*len(prate_example_dates),7),sharey='all')
        for di,date2plot in enumerate(prate_example_dates):
            prate_aligned = get_subset(run, run.aligned, 'p_rate', {'date': date2plot}, events=list_cond_filts['p_rate'][1],
                                       beh=f'{align_pnts[0]} onset',
                                       plttitle=f'Response to pattern onset example day {di+1}', plttype='ts',
                                       ylabel='zscored pupil size', xlabel=f'Time since pattern onset (s)',
                                       pltaxis=(prate_multiple_dates_plot[0],prate_multiple_dates_plot[1][di]),
                                       )
        prate_multiple_dates_plot[0].set_constrained_layout('constrained')
        prate_multiple_dates_plot[0].show()

        print('break now if wanted')
        time.sleep(10)


        # alt rand analysis
        alt_rand_sessnames = []
        alt_sesses = []
        # get sessions with alternating pattern trials
        for sess in run.data:
            altsess = utils.filter_df(run.data[sess].trialData, ['s1','a1','e!0'])
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
        run.get_aligned_events = TDAnalysis.get_aligned_events
        run.trialData.set_index('Trial_Start_dt',append=True,inplace=True,drop=False)
        run.trialData['Pretone_end_dt'] = [tstart + timedelta(0, predur) for tstart, predur in
                                   zip(run.trialData['Trial_Start_dt'], run.trialData['PreTone_Duration'])]

        harpmatrices_pkl = os.path.join(pkldir,'mousefam_hf_harps_matrices_allfam2.pkl')
        if os.path.isfile(harpmatrices_pkl):
            with open(harpmatrices_pkl, 'rb') as pklfile:
                run.harpmatrices = pickle.load(pklfile)
        else:
            run.harpmatrices = utils.get_event_matrix(run,run.data,r'W:\mouse_pupillometry\mouse_hf\harpbins',)
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
        p09_df = utils.filter_df(run.trialData, ['phigh', 'e!0']).loc[:,'230221',:]
        p05_df = utils.filter_df(run.trialData, ['p0.5', 'e!0']).loc[:,'230221',:]
        onset_times,onset_counts = np.unique([p05_df.PreTone_Duration.to_list() +
                                             p09_df.PreTone_Duration.to_list()], return_counts=True)
        pattern_hist[1].bar(onset_times, onset_counts/np.sum(onset_counts), align='center')
        pattern_hist[1].set_xlabel('Pattern embed time from Trial Start (s)')
        pattern_hist[1].set_xticks([1,2,3,4,5],[1,2,3,4,5])
        pattern_hist[1].set_ylabel('Proportion of Trials')
        pattern_hist[1].set_title('Distribution of pattern onset times')
        pattern_hist[0].savefig(os.path.join(run.figdir,'pattern_time_dist.svg'),bbox_inches='tight')

