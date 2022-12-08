import matplotlib.colors
import statsmodels.api as sm
from pupil_analysis import Main
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
from pupil_analysis import batch_analysis, plot_traces, get_subset, glm_from_baseline


if __name__ == "__main__":
    # paradigm = ['altvsrand','normdev']
    paradigm = ['probreward']
    # paradigm = ['familiarity']
    pkldir = r'C:\bonsai\gd_analysis\pickles'
    pkl2use = os.path.join(pkldir,'mouseprobreward_2d_90Hz_6lpass_025hpass_wdlc_TOM_interpol_all_int02s_rawsize_221115.pkl')

    run = Main(pkl2use, (-1.0, 3.0),False)
    pmetric2use = ['diameter_2d_zscored','dlc_radii_a_zscored','dlc_EW_zscored']
    # pmetric2use = 'dlc_radii_a_zscored'
    run.data.pop('DO53_221003')

    fig_subdir = 'probreward'
    if not os.path.isdir(os.path.join(run.figdir, fig_subdir)):
        os.mkdir(os.path.join(run.figdir, fig_subdir))
    run.figdir = os.path.join(run.figdir, fig_subdir)

    do_baseline = False  # 'rawsize' not in pkl2use
    if 'probreward' in paradigm:
        run.probreward = {}
        align_pnts = ['Lick','Reward','Trial_End']
        dates2plot = ['221005','221014','221021','221028','221104']
        run.add_diff_col_dt('Trial_Outcome')
        keys = [batch_analysis(run, run.probreward, [1], f'{align_pnts[0]}_Time_dt', [[0, f'{align_pnts[1]} time'], ], ['a1', 'a0','-1rew'],
                               ['rewarded', 'not rewarded','rew then nonrew'], pmetric=pmetric2use[1], filter_df=True, plot=False,
                               baseline=do_baseline)]
        # keys.append([batch_analysis(run, run.probreward, [1], f'{align_pnts[0]}_Time_dt', [[0, f'{align_pnts[1]} time'], ], ['a1', 'a0','-1rew'],
        #                        ['rewarded', 'not rewarded','rew then nonrew'], pmetric=pmetric2use[1], filter_df=True, plot=False)])
        plot = False
        if plot:
            for date2plot in dates2plot:
                get_subset(run,run.probreward,keys[0][0],{'date':[date2plot]},
                           ['rewarded','not rewarded','rew then nonrew'],f'{align_pnts[0]} time')  # drop=['name','DO50']
                # get_subset(run,run.probreward,f"stage1_['a1', 'a0']_{align_pnts[0]}_Time_dt_0",{'name':['DO50','DO51','DO53']},
                #            ['rewarded','not rewarded'],f'{align_pnts[0]} time',extra_filts={'date':date2plot})
            # get_subset(run,run.probreward,"stage1_['a1', 'a0']_Lick_Time_dt_0",{'name':{'date':'221005'},},
        #            ['rewarded','not rewarded'],'Lick time')

        # for a,animal in enumerate(animals2plot):
        #     for d, date2plot in enumerate(dates2plot):
        #         sess_pd_rew = working_df[0].loc[:,animal,date2plot]
        #         axes[a][d].plot(sess_pd_rew.mean(axis=0),c='C0')
        #
        #         sess_pd_nonrew = working_df[1].loc[:,animal,date2plot]
        #         cmap = plt.get_cmap('plasma',sess_pd_rew.shape[0])
        #         for i,(idx,row) in enumerate(sess_pd_rew.iterrows()):
        #             axes[a][d].plot(row,c=cmap(i),ls='--')
        # norm = matplotlib.colors.Normalize()
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # fig_cbar = fig.colorbar(sm,ticks=(0,1),ax=axes[:,-1])
        # fig_cbar.ax.set_yticklabels(['start','end'])

    base_plt_title = 'Evolution of pupil response with successive licks'
    animals2plot = ['DO50','DO51','DO53']
    dates2plot = ['221005','221014','221021','221028','221104']
    animal_date_pltform = {'ylabel': 'z-scored pupil size',
                           'xlabel': 'Time since "X"',
                           'figtitle':base_plt_title,
                           'rowtitles': animals2plot,
                           'coltitles': dates2plot,
                           }
    # indvtraces_nonbinned = plot_traces(animals2plot,dates2plot,run.probreward[keys[0][0]],run.duration,run.samplerate,
    #                                    plotformatdict=animal_date_pltform)

    binsize = 5
    for i,cond in enumerate(['Rewarded','Non-rewarded']):
        animal_date_pltform['figtitle'] = f"{base_plt_title} binned {binsize} trials: {cond}"
        indvtraces_binned = plot_traces(animals2plot,dates2plot,run.probreward[keys[0][0]], run.duration,run.samplerate,
                                        plotformatdict=animal_date_pltform,binsize=binsize,cond_subset=[i],)
        indvtraces_binned[0].savefig(rf'W:\mouse_pupillometry\figures\probrewardplots\evolve{i}.svg',bbox_inches='tight')

    # indvtraces_binned[0].savefig(r'W:\mouse_pupillometry\figures\probrewardplots\'Evolution of pupil response with presentations of X.svg',bbox_inches='tight')

    glm = {}
    # df4glm = pd.concat(run.probreward[keys[0][0]][2]).sort_index(level=1).drop_duplicates()
    df4glm =  run.probreward[keys[0][0]][2][1].sort_index(level=1).drop_duplicates()
    # n_subplots = df4glm.index.to_series().unique()
    fig_glm,axes_glm = plt.subplots(df4glm.index.get_level_values('name').unique().shape[0],len(dates2plot),sharex='all')
    for ai,name in enumerate(df4glm.index.get_level_values('name').unique()):
        for di,date in enumerate(dates2plot):
            glm[f'{name}_{date}'] = glm_from_baseline(df4glm.loc[:,name,date],run.duration,1,axes_glm[ai][di])
    # fig_glm.savefig(rf'W:\mouse_pupillometry\figures\probrewardplots\sess_baseline_glm_normed.svg',bbox_inches='tight')

    list_dfs = utils.merge_sessions(r'c:\bonsai\data\Dammy',run.labels,'TrialData',['221005',run.dates[-1]])
    run.trialData = pd.concat(list_dfs)
    for col in run.trialData.columns:
        if 'Time' in col:
            utils.add_datetimecol(run.trialData,col)
    run.get_aligned_events = TDAnalysis.get_aligned_events


    # harpmatrices_pkl = os.path.join(pkldir,'probreward_harps_matrices_221110.pkl')
    # if os.path.isfile(harpmatrices_pkl):
    #     with open(harpmatrices_pkl, 'rb') as pklfile:
    #         run.harpmatrices = pickle.load(pklfile)
    # else:
    #     run.harpmatrices = utils.get_event_matrix(run,run.data,r'W:\mouse_pupillometry\mouseprobreward\harpbins',)
    #     with open(harpmatrices_pkl, 'wb') as pklfile:
    #         pickle.dump(run.harpmatrices,pklfile)

    run.animals = run.labels
    # run.lickrasters_firstlick = run.get_aligned_events(run,'Trial_Start_Time_dt',4,(-10.0,10.0),lfilt=75)

