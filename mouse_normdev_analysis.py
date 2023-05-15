import matplotlib.colors
import statsmodels.api as sm
from pupil_analysis_func import Main, get_fig_mosaic
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import os
import analysis_utils as utils
from copy import copy
from behaviour_analysis import TDAnalysis
import pickle
import ruptures as rpt
from pupil_analysis_func import batch_analysis, plot_traces, get_subset, glm_from_baseline


if __name__ == "__main__":
    # matplotlib.use('TKAgg')
    # plt.ioff()
    # plt.rcParams.update({'font.size': 16})

    # paradigm = ['altvsrand','normdev']
    paradigm = ['normdev']
    # paradigm = ['familiarity']
    pkldir = r'c:\bonsai\gd_analysis\pickles'
    pkl2use = os.path.join(pkldir,'mouse_hf_normdev_2d_90Hz_driftcorr_lpass4_hpass00_hanning025_TOM_w_LR_detrend_wTTL_nospdouts.pkl')
    # pkl2use = os.path.join(pkldir,r'mouseprobreward_2d_90Hz_6lpass_025hpass_wdlc_TOM_interpol_all_int02s_221028.pkl')

    run = Main(pkl2use, (-1.0, 5.0), figdir=rf'W:\mouse_pupillometry\figures\mouse_normdev',fig_ow=False)
    pmetric2use = ['diameter_2d_zscored','dlc_radii_a_zscored','dlc_EW_zscored','dlc_radii_a_processed','dlc_EW_processed']

    do_baseline = True  # 'rawsize' not in pkl2use
    if 'normdev' in paradigm:
        run.add_pretone_dt()
        run.add_lick_in_window_bool('ToneTime_dt')
        run.add_viol_diff()
        run.aligned = {}
        align_pnts = ['ToneTime','Reward','Gap_Time','Violation','Trial_Start']
        # dates2plot = ['221005','221014','221021','221028','221104']
        # dates2plot = ['230126','230127','230206','230207']
        # dates2plot = ['230306','230307','230308','230310']  # new normdev 0.1 rate
        # dates2plot = ['230317']
        dates2plot=run.dates
        animals2plot=run.labels
        stages=[4]

        # dateconds = ['80% Rew 5 uL (day 1)','80% Rew 2 uL','50% Rew 5 uL','80% Rew 5 uL (day 2)','95% Rew 5 uL']
        dateconds = ['Day 1: Deviant C tone', 'Day 2: Deviant C+D tone',
                     'Day 3: Deviant C tone (large)', 'Day 4: Deviant C tone']

        run.add_diff_col_dt('Trial_Outcome')
        eventnames = [['Normal', 'Deviant'],
                      ['Normal', 'C to B','C to A','C to D'],
                      ['Normal', 'AB_D','AB__','Shifted Normal'],
                      ['Normal', 'AB_D','AB__','Shifted Normal']]

        keys = []
        keys.append(batch_analysis(run, run.aligned, stages, f'{align_pnts[0]}_dt', [[0.0, f'{align_pnts[0]}'], ],
                                   [['d0'],['d!0'] ], eventnames[0], pmetric=pmetric2use[1], filter_df=True, plot=True,
                                   use4pupil=True, baseline=do_baseline, pdr=False, extra_filts=['a1','tones4','s3','noplicks']))
        # keys.append(batch_analysis(run, run.aligned, stages, f'{align_pnts[0]}_dt', [[0.0, f'{align_pnts[0]}'], ],
        #                            [['d0'],['d!0','d_C2B'],['d!0','d_C2A'],['d!0','d_C2D']], eventnames[1],
        #                            pmetric=pmetric2use[1], filter_df=True, plot=True,
        #                            use4pupil=True, baseline=do_baseline, pdr=False, extra_filts=['a1','tones4','s3','noplicks']))
        # keys.append(batch_analysis(run, run.normdev, stages, f'{align_pnts[0]}_dt', [[0, f'{align_pnts[0]} '], ],
        #                        ['d0','d1','d3','d-1' ],eventnames[1], pmetric=pmetric2use[2], filter_df=True, plot=False,
        #                        use4pupil=True, baseline=do_baseline, pdr=False, extra_filts=['a1','tones4','s3']))
        # keys.append(batch_analysis(run, run.normdev, stages, f'{align_pnts[0]}_dt', [[0, f'{align_pnts[0]} '], ],
        #                        ['d0','d1','d3','d-1' ],eventnames[1], pmetric=pmetric2use[2], filter_df=True, plot=False,
        #                        use4pupil=True, baseline=do_baseline, pdr=False, extra_filts=['a0','tones4','s3']))
        plot = True
        # plots_by_dates = plt.subplots(len(dates2plot))
        fig_form, chunked_fig_form, n_cols,plt_is = get_fig_mosaic(dates2plot)
        pltsize = (9 * n_cols, 6 * len(chunked_fig_form))

        if plot:
            for ki, key2use in enumerate(keys):
                key_suffix = key2use[0].replace('[','').replace(']','').replace("'",'').replace(', ', '_')
                datepltsize = (9 * n_cols, 6 * len(chunked_fig_form))

                tsplots_by_dates = plt.subplot_mosaic(fig_form, sharex=False, sharey=True, figsize=datepltsize)
                boxplots_by_dates = plt.subplot_mosaic(fig_form, sharex=False, sharey=True, figsize=datepltsize)
                trendplots_by_dates = plt.subplot_mosaic(fig_form, sharex=False, sharey=True, figsize=datepltsize)
                for plottype,pltfig in zip(['ts'],[tsplots_by_dates]):
                    for di, date2plot in enumerate(dates2plot):
                        get_subset(run, run.aligned, key2use[0], {'date': [date2plot],},  # 'animal':['DO54','DO55','DO57','DO60']
                                   eventnames[ki],f'{align_pnts[0]} time', plttitle=dateconds[di],
                                   ylabel='Mean of pupil size', xlabel='Time since Pattern Start (s)',
                                   plttype=plottype, pltaxis=[pltfig[0], pltfig[1][str(di)]])
                for di, date2plot in enumerate(dates2plot):
                    get_subset(run, run.aligned, key2use[0], {'date': [date2plot],},
                               eventnames[ki],f'{align_pnts[0]} time', plttitle=dateconds[di], ntrials=[-10,10],
                               ylabel='delta mean pupil size', xlabel='Time since Pattern start (s)',
                               plttype='ts', pltaxis=[trendplots_by_dates[0], trendplots_by_dates[1][str(di)]])

                               # ['rewarded','not rewarded'],f'{align_pnts[0]} time',extra_filts={'date':date2plot})
                # get_subset(run,run.probreward,"stage1_['a1', 'a0']_Lick_Time_dt_0",{'name':{'date':'221005'},},
            #            ['rewarded','not rewarded'],'Lick time')
                utils.unique_legend(tsplots_by_dates)
                tsplots_by_dates[0].savefig(os.path.join(run.figdir, f'alldates_HF_tsplots_EW_{key_suffix}.svg'),
                                            bbox_inches='tight')
                boxplots_by_dates[0].savefig(os.path.join(run.figdir, f'alldates_HF_boxplots_EW_{key_suffix}.svg'),
                                             bbox_inches='tight')
                trendplots_by_dates[0].savefig(os.path.join(run.figdir, f'alldates_HF_lastN_{key_suffix}.svg'),
                                               bbox_inches='tight')

                animals2plot = run.labels
                tsplots_by_animal = plt.subplots(len(animals2plot), len(dates2plot), squeeze=False, sharex='all',
                                                 sharey='all')
                tsplots_by_animal_ntrials = plt.subplots(len(animals2plot), len(dates2plot), squeeze=False, sharex='all',
                                                         sharey='all')
                histplots_reactiontime = plt.subplots(len(animals2plot), squeeze=False, sharex='all', sharey='all')

                run.pupilts_by_session(run, run.aligned, key2use, animals2plot, dates2plot, eventnames[ki], dateconds,
                                       f'{align_pnts[0]} time', tsplots_by_animal,plttype='ts',)

                # format plot for saving
                pltsize = (9 * len(dates2plot), 6 * len(animals2plot))
                tsplots_by_animal[0].set_size_inches(pltsize)
                utils.unique_legend(tsplots_by_animal)
                tsplots_by_animal[0].savefig(os.path.join(run.figdir, rf'tspupil_byanimal_{key_suffix}.svg'),
                                             bbox_inches='tight')

    base_plt_title = 'Evolution of pupil response with successive licks'
    # animals2plot = ['DO54','DO55','DO56','DO57']

    # dates2plot = ['230126']
    animal_date_pltform = {'ylabel': 'z-scored pupil size',
                           'xlabel': 'Time since "X"',
                           'figtitle':base_plt_title,
                           'rowtitles': animals2plot,
                           'coltitles': dates2plot,
                           }
    # indvtraces_nonbinned = plot_traces(animals2plot,dates2plot,run.probreward[keys[0][0]],run.duration,run.samplerate,
    #                                    plotformatdict=animal_date_pltform)

    binsize = 5



    #
    do_harp_stuff = False
    if do_harp_stuff:
        plt.ioff()
        list_dfs = utils.merge_sessions(r'c:\bonsai\data\Dammy',run.labels,'TrialData',['221005',run.dates[-1]])
        run.trialData = pd.concat(list_dfs)
        for col in run.trialData.columns:
            if 'Time' in col:
                utils.add_datetimecol(run.trialData,col)
        run.get_aligned_events = TDAnalysis.get_aligned_events
        run.trialData.set_index('Trial_Start_Time',append=True,inplace=True,drop=False)


        harpmatrices_pkl = os.path.join(pkldir,'probreward_hf_harps_matrices_221221.pkl')
        if os.path.isfile(harpmatrices_pkl):
            with open(harpmatrices_pkl, 'rb') as pklfile:
                run.harpmatrices = pickle.load(pklfile)
        else:
            run.harpmatrices = utils.get_event_matrix(run,run.data,r'W:\mouse_pupillometry\mouseprobreward_hf\harpbins',)
            with open(harpmatrices_pkl, 'wb') as pklfile:
                pickle.dump(run.harpmatrices,pklfile)
        fig,ax = plt.subplots()
        run.lickrasters_firstlick = {}
        for outcome in [['a1'],['a0']]:
            run.animals = run.labels
            run.lickrasters_firstlick[outcome[0]] = run.get_aligned_events(run,'Trial_Start_Time_dt',0,(-1.0,3.0),byoutcome_flag=True,outcome2filt=outcome)
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

