import matplotlib.colors
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn

import align_functions
from align_functions import get_aligned_events
from pupil_analysis_func import Main
from plotting_functions import get_fig_mosaic
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import os
import analysis_utils as utils
from copy import copy
from behaviour_analysis import TDAnalysis
from collections import OrderedDict
import pickle
import ruptures as rpt
from pupil_analysis_func import batch_analysis, plot_traces, get_subset, glm_from_baseline


if __name__ == "__main__":
    plt.ioff()
    # paradigm = ['altvsrand','normdev']
    paradigm = ['probreward']
    # paradigm = ['familiarity']
    pkldir = r'c:\bonsai\gd_analysis\pickles'
    pkl2use = os.path.join(pkldir,'mouseprobreward_hf_fam_2d_90Hz_driftcorr_lpass4_hpass00_hanning025_TOM_w_LR_detrend.pkl')
    # pkl2use = os.path.join(pkldir,r'mouseprobreward_2d_90Hz_6lpass_025hpass_wdlc_TOM_interpol_all_int02s_221028.pkl')

    run = Main(pkl2use, (-1.0, 3.0),False, rf'W:\mouse_pupillometry\figures\probrewardplots_hpass00')
    run.animals = run.labels
    pmetric2use = ['diameter_2d_zscored','dlc_radii_a_zscored','dlc_EW_zscored']
    # pmetric2use = 'dlc_radii_a_zscored'

    do_baseline = True  # 'rawsize' not in pkl2use
    if 'probreward' in paradigm:
        run.aligned = {}
        align_pnts = ['Lick','Reward','Trial_End','Trial_Start']
        align_idx = 0
        # dates2plot = ['221005','221014','221021','221028','221104']
        dates2plot = ['230203','230208','230211']

        # dateconds = ['80% Rew 5 uL (day 1)','80% Rew 2 uL','50% Rew 5 uL','80% Rew 5 uL (day 2)','95% Rew 5 uL']
        dateconds = ['80% Rew 5 uL (day 1)', '50% Rew 5 uL', '80% Rew 5 uL (day 2)']

        run.add_diff_col_dt('Trial_Outcome')
        eventnames = [['rewarded', 'not rewarded',],
                      ['rew then\n rew','nonrew\n rew','rew then\n nonrew','nonrew\n then non rew'],
                      ['rewarded', ]]  # 'rew then\n nonrew','nonrew\n then rew'
        keys = []

        keys.append([batch_analysis(run, run.aligned, [1], f'{align_pnts[align_idx]}_Time_dt', [[0, f'{align_pnts[align_idx]} time'], ],
                                    ['a1', 'a0', ],
                                    eventnames[0], pmetric=pmetric2use[2], filter_df=True, plot=False,
                                    baseline=do_baseline, pdr=False, extra_filts=['prew<1','sess_a'])])

        keys.append([batch_analysis(run, run.aligned, [1], f'{align_pnts[align_idx]}_Time_dt', [[0, f'{align_pnts[align_idx]} time'], ],
                                    [['a1','-1same'], ['a1','-1norew'],['a0','-1rew'], ['a0','-1same'] ],
                                    eventnames[1],
                                    pmetric=pmetric2use[2], filter_df=True, plot=False,
                                    baseline=do_baseline, pdr=False, extra_filts=['prew<1','sess_a'])])

        keys.append([batch_analysis(run, run.aligned, [1], f'{align_pnts[align_idx]}_Time_dt', [[0, f'{align_pnts[align_idx]} time'], ],
                                    ['a1', ],
                                    eventnames[0], pmetric=pmetric2use[2], filter_df=True, plot=False,
                                    baseline=do_baseline, pdr=False, extra_filts=['prew=1','sess_a'])])
        # keys.append([batch_analysis(run, run.probreward, [1], f'{align_pnts[0]}_Time_dt', [[0, f'{align_pnts[1]} time'], ], ['a1', 'a0','-1rew'],
        #                        ['rewarded', 'not rewarded','rew then nonrew'], pmetric=pmetric2use[1], filter_df=True, plot=False)])
        plot = True
        # plots_by_dates = plt.subplots(len(dates2plot))

        fig_form,chunked_fig_form,n_cols,plt_is = get_fig_mosaic(dates2plot)

        pltsize = (9*n_cols, 6*len(chunked_fig_form))
        tsplots_by_dates = plt.subplot_mosaic(fig_form, sharex=False, sharey=True, figsize=pltsize,)
        boxplots_by_dates = plt.subplot_mosaic(fig_form, sharex=False, sharey=True, figsize=pltsize)
        trendplots_by_dates = plt.subplot_mosaic(fig_form,sharex=False,sharey=True,figsize=pltsize)
        for ki,key in enumerate(keys):
            tsplots_by_dates = plt.subplot_mosaic(fig_form, sharex=False, sharey=True, figsize=pltsize)
            boxplots_by_dates = plt.subplot_mosaic(fig_form, sharex=False, sharey=True, figsize=pltsize)
            trendplots_by_dates = plt.subplot_mosaic(fig_form, sharex=False, sharey=True, figsize=pltsize)
            if plot:
                for plottype,pltfig in zip(['ts','boxplot'],[tsplots_by_dates,boxplots_by_dates]):
                    for di, date2plot in enumerate(dates2plot):
                        get_subset(run, run.aligned, key[0][0], {'date':[date2plot]},
                                   eventnames[ki],f'{align_pnts[align_idx]} time', plttitle=dateconds[di],
                                   ylabel='Mean of max zscored pupil size for epoch', xlabel='Time since lick (s)',
                                   plttype=plottype, pltaxis=(pltfig[0], pltfig[1][str(di)]))  # drop=['name','DO50']

            tsplots_by_dates[0].savefig(os.path.join(run.figdir,rf'alldates_{ki}_HF_tsplots_EW.svg'))
            boxplots_by_dates[0].savefig(os.path.join(run.figdir, rf'alldates_{ki}_HF_boxplots_EW.svg'))
            trendplots_by_dates[0].savefig(os.path.join(run.figdir, rf'alldates_{ki}_HF_pdelta_trendplots_EW.svg'))

        plt.close('all')
        plt.ion()
        animals2plot = run.animals
        dates2plot = run.dates
        tsplots_by_animal = plt.subplots(len(animals2plot),len(dates2plot),squeeze=False,sharex='all',sharey='all')
        tsplots_by_animal_ntrials = plt.subplots(len(animals2plot), len(dates2plot), squeeze=False, sharex='all', sharey='all')
        histplots_reactiontime = plt.subplots(len(animals2plot), squeeze=False, sharex='all', sharey='all')
        key2use = 0
        ntrials = 5000
        # plot_traces(animals2plot,dates2plot,run.probreward[keys[key2use][0][0]],run.duration,fs=run.samplerate,
        #             cmap_name='gray',pltax=tsplots_by_animal,linealpha=0.1,cmap_flag=False)
        for ai,animal in enumerate(animals2plot):
            for di, date2plot in enumerate(dates2plot):
                get_subset(run, run.aligned, keys[key2use][0][0], {'date':[date2plot], 'name':animal}, eventnames[key2use],
                           f'{align_pnts[0]} time', plttitle=dateconds[di], level2filt='name',
                           pltaxis=(tsplots_by_animal[0],tsplots_by_animal[1][ai,di]))
                get_subset(run, run.aligned, keys[key2use][0][0], {'date': [date2plot], 'name': animal},
                           eventnames[key2use], f'{align_pnts[align_idx]} time', plttitle=dateconds[di],
                           level2filt='name', ntrials=(-ntrials,-ntrials),
                           pltaxis=(tsplots_by_animal_ntrials[0], tsplots_by_animal_ntrials[1][ai, di]))
                tsplots_by_animal[1][ai, di].set_title('')
                tsplots_by_animal_ntrials[1][ai, di].set_title('')
                if ai == len(animals2plot)-1:
                    tsplots_by_animal[1][ai, di].set_xlabel(f'Time since {align_pnts[0]} (s)')
                if ai == 0:
                    tsplots_by_animal[1][ai, di].set_title(dateconds[di])
                    # tsplots_by_animal[1][ai, di].set_ylim(-2,6)

        for ts_animal_fig in (tsplots_by_animal,tsplots_by_animal_ntrials):
            ts_animal_fig[0].suptitle(f'Pupil response to first lick, last {ntrials} trials',y=0.9)
            ts_animal_fig[0].set_size_inches(12,18)
            utils.unique_legend(ts_animal_fig,fontsize=9)
        tsplots_by_animal[0].savefig(os.path.join(run.figdir, rf'tspupil_byanimal_noindv.svg'),
                                     bbox_inches='tight')
        tsplots_by_animal_ntrials[0].savefig(os.path.join(run.figdir, rf'tspupil_byanimal_noindv_ntrials_both.svg'),
                                     bbox_inches='tight')
        allsess_ntrials_ts_plot = plt.subplots(nrows=2,ncols=len(dates2plot),squeeze=False,sharey='row')
        ntrial_plot_data = []
        for ni, (ntrial, n_name) in enumerate(zip([ntrials, ntrials*-1],['First','Last'])):
            if ntrial <0:
                n_start_idx = ntrial*-1
            else:
                n_start_idx = 0
            for di, date2plot in enumerate(dates2plot):
                ntrial_plot_data.append(get_subset(run, run.aligned, keys[key2use][0][0], {'date': [date2plot], 'name': []},
                                        eventnames[key2use], f'{align_pnts[align_idx]} time', plttitle=f'{dateconds[di]}, {n_name} {abs(ntrials)} trials',
                                        level2filt='name', ntrials=(ntrial, ntrial), plttype='ts', pdelta_wind=[0.5,2.5],
                                        pltaxis=(allsess_ntrials_ts_plot[0], allsess_ntrials_ts_plot[1][ni, di]), ntrial_start_idx=n_start_idx)[2]),
            allsess_ntrials_ts_plot[0].set_size_inches(6*len(dates2plot),12)
            # utils.unique_legend(allsess_ntrials_ts_plot,fontsize=9)
        allsess_ntrials_ts_plot[0].set_constrained_layout('contrained')
        allsess_ntrials_ts_plot[0].show()
        utils.unique_legend(allsess_ntrials_ts_plot)
        allsess_ntrials_ts_plot[0].savefig(os.path.join(run.figdir, rf'allsess_first_{abs(ntrials)}trials_boxplot.svg'), bbox_inches='tight')

    # compare rewarded trials before and after 1st test day
    dates2plot = ['230201','230202','230203','230206']
    date_labels = ['Before p(80) (-2 days)','Before p(80) (-1 days)', 'p(80)', 'After p(80) (1 day)']
    rew_plt_cols = ['dodgerblue','dodgerblue','C0','navy']
    start_idxs = [10,10,0,10]
    # start_idxs = [5,5,5,5]
    rew_plt_ls = ['--','--','-','-.']
    rewarded_across_dates = plt.subplots()
    for di,date in enumerate(dates2plot):
        if di == 1:
            key = keys[0][0][0]
        else:
            key = keys[2][0][0]
        get_subset(run,run.aligned,key,{'date':date},ntrials=1000,ntrial_start_idx=start_idxs[di],pltaxis=rewarded_across_dates,
                   pltargs=('-',None),plotcols=[rew_plt_cols[di]],exclude_idx=[1],events=[date_labels[di]])

    rewarded_across_dates[0].set_size_inches(4.089*1.5, 3.223*1.5)
    rewarded_across_dates[0].set_constrained_layout('constrained')
    rewarded_across_dates[1].set_title('Pupil response to reward around 80% test day: Last 5 trials',fontsize=14)
    rewarded_across_dates[1].set_ylabel('zscored pupil size',fontsize=14)
    rewarded_across_dates[1].set_xlabel('Time since lick (s)',fontsize=14)
    rewarded_across_dates[1].set_ylim(-.5,1.9)

        # tsplots_indvtraces = plot_traces(animals2plot,dates2plot,run.probreward[keys[key2use][0][0]],run.duration,
        #                                  fs=run.samplerate)
        # tsplots_indvtraces[0].set_size_inches(12,1)
        # tsplots_indvtraces[0].supxlabel(f'Time since {align_pnts[align_idx]} (s)')
        # tsplots_indvtraces[0].supylabel(f'pupil size (zscored)')
        # tsplots_indvtraces[0].savefig(os.path.join(run.figdir, rf'tspupil_indv_traces_nonrew.svg'),bbox_inches='tight')

        # for animal in animals2plot: for hist
        #     sess_df = run.trialData.loc[animal,:,:]
        #     sess_corr_trials = sess_df[sess_df['Trial_Outcome']==1]
        #     sess_react_times_ser = (sess_corr_trials['Trial_Start_Time_dt']-sess_corr_trials['Lick_Time_dt']).timedelta.total_seconds()


                # sess_pd_nonrew = working_df[1].loc[:,animal,date2plot]
                # cmap = plt.get_cmap('plasma',sess_pd_rew.shape[0])
                # for i,(idx,row) in enumerate(sess_pd_rew.iterrows()):
                #     axes[a][d].plot(row,c=cmap(i),ls='--')
        # norm = matplotlib.colors.Normalize()
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # fig_cbar = fig.colorbar(sm,ticks=(0,1),ax=axes[:,-1])
        # fig_cbar.ax.set_yticklabels(['start','end'])

    base_plt_title = 'Evolution of pupil response with successive licks'
    # animals2plot = ['DO50','DO51','DO53']
    # dates2plot = ['221005','221014','221021','221028','221104']
    # animals2plot = ['DO58','DO59','DO60','DO61','DO62']
    # dates2plot = ['221221']
    animal_date_pltform = {'ylabel': 'z-scored pupil size',
                           'xlabel': 'Time since "X"',
                           'figtitle':base_plt_title,
                           'rowtitles': animals2plot,
                           'coltitles': dates2plot,
                           }
    # indvtraces_nonbinned = plot_traces(animals2plot,dates2plot,run.probreward[keys[0][0]],run.duration,run.samplerate,
    #                                    plotformatdict=animal_date_pltform)

    # binsize = 3
    # for i,cond in enumerate(['Rewarded','Non-rewarded','rew then\n nonrew','nonrew\n then rew']):
    #     animal_date_pltform['figtitle'] = f"{base_plt_title} binned {binsize} trials: {cond}"
    #     indvtraces_binned = plot_traces(animals2plot,dates2plot,run.probreward[keys[0][0]], run.duration,run.samplerate,
    #                                     plotformatdict=animal_date_pltform,binsize=binsize,cond_subset=[i],)
    #     indvtraces_binned[0].savefig(rf'W:\mouse_pupillometry\figures\probrewardplots\evolve{i}_hf.svg',bbox_inches='tight')

    # indvtraces_binned[0].savefig(r'W:\mouse_pupillometry\figures\probrewardplots\'Evolution of pupil response with presentations of X.svg',bbox_inches='tight')

    # glm = {}
    # df4glm = pd.concat(run.probreward[keys[0][0]][2]).sort_index(level=1).drop_duplicates()60
    # df4glm =  run.probreward[keys[0][0]][2][1].sort_index(level=1).drop_duplicates()
    # n_subplots = df4glm.index.to_series().unique()
    # fig_glm,axes_glm = plt.subplots(df4glm.index.get_level_values('name').unique().shape[0],len(dates2plot),sharex='all')
    # for ai,name in enumerate(df4glm.index.get_level_values('name').unique()):
    #     for di,date in enumerate(dates2plot):
    #         glm[f'{name}_{date}'] = glm_from_baseline(df4glm.loc[:,name,date],run.duration,1,axes_glm[ai][di])
    # fig_glm.savefig(rf'W:\mouse_pupillometry\figures\probrewardplots\sess_baseline_glm_normed.svg',bbox_inches='tight')
    #
    do_harp_stuff = False
    if do_harp_stuff:
        plt.ioff()
        list_dfs = utils.merge_sessions(r'c:\bonsai\data\Dammy',run.labels,'TrialData',[run.dates[0],run.dates[-1]])
        run.trialData = pd.concat(list_dfs)
        for col in run.trialData.columns:
            if 'Time' in col:
                utils.add_datetimecol(run.trialData,col)
        run.get_aligned_events = get_aligned_events
        # run.index = pd.concat({run.trialData['Trial_Start_Time_dt'].to_numpy():run.trialData},names=['time']).index
        run.trialData.set_index('Trial_Start_Time_dt',append=True,inplace=True,drop=False)

        harpmatrices_pkl = os.path.join(pkldir,'probreward_hf_harps_matrices_230203_230208_230211_only.pkl')
        if os.path.isfile(harpmatrices_pkl):
            with open(harpmatrices_pkl, 'rb') as pklfile:
                run.harpmatrices = pickle.load(pklfile)
        else:
            run.harpmatrices = align_functions.get_event_matrix(run, run.data, r'W:\mouse_pupillometry\mouseprobreward_hf\harpbins', )
            with open(harpmatrices_pkl, 'wb') as pklfile:
                pickle.dump(run.harpmatrices,pklfile)
        run.lickrasters_firstlick = {}
        lickraster_align_idx = 0
        for oi, outcome in enumerate([['a1'],['a0']]):
            run.lickrasters_firstlick[outcome[0]] = run.get_aligned_events(run,f'{align_pnts[lickraster_align_idx]}_Time_dt',0,(-5.0,5.0),
                                                                           byoutcome_flag=True,outcome2filt=outcome,
                                                                           extra_filts=None,plotcol=oi)
            run.lickrasters_firstlick[outcome[0]][0].set_size_inches((12,9))
            run.lickrasters_firstlick[outcome[0]][0].savefig(os.path.join(run.figdir, rf'alldates_HF_lickraster_{align_pnts[lickraster_align_idx]}_{outcome}.svg'
                                                                          ),bbox_inches='tight')

        fig,ax = plt.subplots()
        for outcome in [['a1'],['a0']]:
            binsize= 200
            prob_lick_mat = run.lickrasters_firstlick[outcome[0]][2].fillna(0).rolling(binsize,axis=1).mean()  # .mean().iloc[:,binsize - 1::binsize]
            prob_lick_mean = prob_lick_mat.mean(axis=0)
            condname = outcome[0].replace('a0', 'Non Rewarded')
            condname = condname.replace('a1', 'Rewarded')
            ax.plot(prob_lick_mean.index,prob_lick_mean,label=condname)
        ax.set_xlabel(f'seconds from {align_pnts[lickraster_align_idx]} time')
        ax.set_ylabel('mean lick rate across animals across sessions')
        ax.set_title(f'Lick rate aligned to {align_pnts[lickraster_align_idx]} time, {1000.0/binsize}s bin')
        ax.legend()
        ax.axvline(0.0,ls='--',c='k',lw=0.25)
        fig.set_size_inches((15,12))
        fig.savefig(os.path.join(run.figdir, rf'alldates_HF_lickrate_{align_pnts[lickraster_align_idx]}_5sto5s.svg'),bbox_inches='tight')

        lickrateplot_by_animal = plt.subplots(len(animals2plot),squeeze=False, sharex='col',sharey='col')

        for outcome in [['a1'],['a0']]:
            binsize = 50
            prob_lick_mat = run.lickrasters_firstlick[outcome[0]][2].fillna(0).rolling(binsize,axis=1).mean()  # .mean().iloc[:,binsize - 1::binsize]
            for ai, animal in enumerate(animals2plot):
                condname = outcome[0].replace('a0', 'Non Rewarded')
                condname = condname.replace('a1', 'Rewarded')
                animal_lick_df = prob_lick_mat.loc[[animal]].mean(axis=0)
                lickrateplot_by_animal[1][ai][0].plot(animal_lick_df.index,animal_lick_df,label=condname)
                lickrateplot_by_animal[1][ai][0].set_ylabel('mean lick rate\n across animals\n across sessions')
                lickrateplot_by_animal[1][ai][0].legend(loc=1)
                lickrateplot_by_animal[1][ai][0].axvline(-0.1, ls='--', c='k', lw=0.25)
                lickrateplot_by_animal[1][ai][0].axvline(-0.0, ls='--', c='k', lw=0.25)

            lickrateplot_by_animal[1][0][0].set_title(f'Lick rate aligned to {align_pnts[lickraster_align_idx]} time, {1e-3*binsize}s bin')
            lickrateplot_by_animal[1][-1][0].set_xlabel(f'seconds from {align_pnts[lickraster_align_idx]} time')
        lickrateplot_by_animal[0].set_size_inches((15, 12))

        dur = np.argwhere(prob_lick_mat.keys() == run.duration[0]).astype(int)[0],np.argwhere(prob_lick_mat.keys() == run.duration[1])[0]
        for oi,outcome in enumerate([['a1'],['a0']]):
            binsize = 50
            prob_lick_mat = run.lickrasters_firstlick[outcome[0]][2].fillna(0).rolling(binsize,axis=1).mean()  # .mean().iloc[:,binsize - 1::binsize]
            for ai, animal in enumerate(animals2plot):
                for di, date in enumerate(dates2plot):
                    sess_lick_df = prob_lick_mat.loc[animal,date,:].iloc[:,dur[0][0]:dur[1][0]]
                    sess_lick_rate = sess_lick_df.mean(axis=0)
                    twin_axis = tsplots_by_animal[1][ai,di].twinx()
                    twin_axis.set_axisbelow(True)
                    twin_axis.plot(sess_lick_rate.index,sess_lick_rate+0.001,c=f'C{oi}',alpha=0.75,zorder=-10,ls='-')
                    twin_axis.set_ylabel('mean lick rate')
                    twin_axis.set_yticks([])

        tsplots_by_animal[0].savefig(os.path.join(run.figdir, rf'tspupil_byanimal_wlicks.svg'),
                                     bbox_inches='tight')

        lickrateplot_by_animal[0].savefig(os.path.join(run.figdir, rf'lickrate_by_animal_{align_pnts[lickraster_align_idx]}_5sto5s.svg'),
                                          bbox_inches='tight')

        # glm linreg

        # get diff licks
        lick_rast_by_outcome = []
        for outcome_key, outcome in zip(['a1','a0'],[1,0]):
            outcome_lickrast = run.lickrasters_firstlick[outcome_key][2].fillna(0)  # rolling(binsize, axis=1).mean()
            outcome_lickrast_sliced = outcome_lickrast.iloc[:,dur[0][0]:dur[1][0]]
            outcome_lickrast_sliced.index = pd.concat({outcome:outcome_lickrast_sliced},names=['outcome']).index
            # outcome_lickrast_sliced.index.set_names('time',level=-1,inplace=True)
            outcome_lickrast_sliced.index.set_names(['name','date','time'],level=[1,2,-1],inplace=True)
            lick_rast_by_outcome.append(outcome_lickrast_sliced)
        lick_rast_by_outcome = pd.concat(lick_rast_by_outcome,axis=0)

        lick_rast_by_outcome.columns = lick_rast_by_outcome.columns.to_series().apply(lambda e: pd.Timedelta(seconds=e))
        dt_pupil = round(np.diff(run.duration)[0] / run.aligned[keys[key2use][0][0]][2][0].shape[1], 9)
        lick_rast_by_outcome_resampled = lick_rast_by_outcome.resample(f'{dt_pupil}S',axis=1).sum()
        lick_rast_by_outcome_diff = lick_rast_by_outcome_resampled
        lick_rast_by_outcome_diff = lick_rast_by_outcome_diff.reorder_levels(['outcome','time','name','date'])

        tdelta_cols = lick_rast_by_outcome_diff.columns

        # get pupil diff
        pupil_ts_by_outcome = []
        pupil_ts_list = run.aligned[keys[key2use][0][0]][2]
        for outcome, pupil_ts in zip([1,0],pupil_ts_list):
            pupil_ts_diff = pupil_ts.diff(axis=1)
            pupil_ts_diff.index = pd.concat({outcome:pupil_ts_diff},names=['outcome','time','name','date']).index
            pupil_ts_diff.columns = tdelta_cols
            pupil_ts_by_outcome.append(pupil_ts_diff)
        pupil_ts_by_outcome = pd.concat(pupil_ts_by_outcome,axis=0)

        lick_rast_by_outcome_diff = lick_rast_by_outcome_diff.loc[pupil_ts_by_outcome.index]   # only use matching trials

        #  do regression linreg
        ytrain = pupil_ts_by_outcome.fillna(0.0).to_numpy().mean(axis=0)
        xtrain = lick_rast_by_outcome_diff.fillna(0.0).to_numpy().mean(axis=0)
        # xtrain = sm.add_constant(xtrain)

        glm = sm.GLM(ytrain,xtrain).fit()
        glm_scatter_plot = plt.subplots()
        glm_scatter_plot[1].scatter(xtrain,ytrain)

        outcomeasarr = np.full_like(lick_rast_by_outcome_diff,0)
        for ri,(r,outcome) in enumerate(zip(outcomeasarr,lick_rast_by_outcome_diff.index.get_level_values('outcome').to_numpy())):
            outcomeasarr[ri,:] = np.full_like(r,outcome)

        glm = {}
        ytrain2 =pupil_ts_by_outcome.fillna(0.0).to_numpy()
        xtrain2 = [lick_rast_by_outcome_diff.fillna(0.0).to_numpy(),outcomeasarr]
        ytrain2_list_arr = [np.array(trial_list) for trial_list in ytrain2.tolist()]
        xtrain2_list_arr = [np.array(trial_list) for trial_list in xtrain2[0].tolist()]

        glm_dict = {'pupil':ytrain2,'lick':xtrain2_list_arr,'outcome':xtrain2[1][:,0]}
        # glm_df = pd.from_dict([ytrain2+xtrain2[0]+xtrain2[1]],axis=1)
        # glm2 = sm.GLM(ytrain2,xtrain2).fit()
        glm_results = smf.glm(formula='pupil ~ lick + outcome', data=glm_dict).fit()
        # linreg = sklearn.linear_linreg.LinearRegression().fit(xtrain2[0],ytrain2)
        linreg = sklearn.linear_model.LinearRegression().fit(lick_rast_by_outcome_diff.fillna(0.0),pupil_ts_by_outcome.fillna(0.0))
        print(linreg.intercept_, linreg.coef_, linreg.score(lick_rast_by_outcome_diff.fillna(0.0),pupil_ts_by_outcome.fillna(0.0)))

        glm_byanimal = {}
        for animal in animals2plot:
            # if animal != 'DO62':
            #     continue
            ytrain = pupil_ts_by_outcome.fillna(0.0).loc[:,:,animal,:].transpose()
            xtrain = lick_rast_by_outcome_diff.fillna(0.0).loc[:,:,animal,:].transpose()
            glm = sm.GLM(ytrain,xtrain).fit()
            print(animal, glm.summary())

        binarised_pupilts = np.where(pupil_ts_by_outcome.to_numpy()>0.0,1,-1)
        glm_binary = sm.GLM(binarised_pupilts.mean(axis=0),lick_rast_by_outcome_diff.mean(axis=0)).fit()
        onlylicks = lick_rast_by_outcome_diff.copy()
        onlylicks = onlylicks[onlylicks<1.0] == np.nan
