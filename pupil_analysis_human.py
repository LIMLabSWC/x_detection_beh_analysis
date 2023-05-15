import time

from pupil_analysis_func import Main, get_subset
import os
from matplotlib import pyplot as plt
import analysis_utils as utils

if __name__ == "__main__":
    # pkldir = r'W:\mouse_pupillometry\working_pickles'
    pkldir = r'c:\bonsai\gd_analysis\pickles'

    # pkl2use = r'pickles\human_familiarity_3d_200Hz_015Shan_driftcorr_hpass01.pkl'
    # pkl2use = r'pickles\human_class1_3d_200Hz_015Shan_driftcorr_hpass01_no29.pkl'
    # pkl2use = r'pickles\human_class1_3d_200Hz_015Shan_driftcorr_hpass01.pkl'
    # pkl2use = r'pickles\human_class1_3d_200Hz_015Shan_driftcorr_hpass01.pkl'
    # pkl2use = r'pickles\DO48_fam_2d_200Hz_015Shan_driftcorr_hpass01.pkl'
    # pkl2use = r'pickles\mouse_normdev_2d_200Hz_015Shan_driftcorr_hpass04_wdlc.pkl'
    # pkl2use = r'pickles\mouse_fam_2d_200Hz_015Shan_driftcorr_hpass04_wdlc.pkl'
    # pkl2use = r'/Volumes/akrami/mouse_pupillometry/pickles/DO48_fam_3d_200Hz_015Shan_driftcorr_hpass01_wdlc.pkl'

    # pkl2use = os.path.join(pkldir,'mouse_normdev_2d_90Hz_025Shan_driftcorr_hpass025_wdlc_TOM.pkl')
    pkl2use = os.path.join(pkldir,r'human_fam_3d_90Hz_driftcorr_lpass_detrend2_hpass01_flipped_TOM_hanning015.pkl')


    # pkl2use = r'pickles\mouse_fam_post_2d_90Hz_025Shan_driftcorr_nohpass_wdlc_TOM.pkl'

    run = Main(pkl2use, (-1,3))

    fig_subdir = r'W:\mouse_pupillometry\figures\human_fam'
    if not os.path.isdir(os.path.join(run.figdir, fig_subdir)):
        os.mkdir(os.path.join(run.figdir, fig_subdir))
        run.figdir = os.path.join(run.figdir, fig_subdir)

    # paradigm = ['altvsrand','normdev']
    # paradigm = ['normdev']
    paradigm = ['familiarity']
    # paradigm = ['familiarity','0.5_fam']
    # pmetric2use = 'dlc_radii_a_zscored'
    pmetrics = ['diameter_3d_zscored','dlc_radii_a_zscored','dlc_EW_zscored','rawarea_zscored']
    pmetric2use = pmetrics[0]

    # run.whitenoise_hit_miss = run.get_aligned([['a1'], ['a0']],
    #                                           event='WhiteNoise',
    #                                           event_shift=[0.0, 0.0],
    #                                           plotlabels=['Hit', 'Miss'],
    #                                           align_col='Gap_Time_dt', pmetric=pmetric2use, use4pupil=True)

    if 'familiarity' in paradigm:  # analysis to run for familiarity paradigm
        run.familiarity = run.get_aligned([['e!0','plow'],['e!0','pmed'],
                                           ['e!0','phigh'],['e!0','ppost'], ['e=0']],
                                          event_shift=[0.0, 0.0, 0.0,0.0, 0.0],
                                          event='ToneTime', xlabel='Time since pattern onset',
                                          plotlabels=['0.1','0.4','0.9','0.6','control'], plotsess=False, pdr=False,
                                          use4pupil=True,
                                          pmetric=pmetric2use
                                          )
        # run.familiarity[0].savefig('human_fam_diameter3d.svg')
        run.familiarity[0].show()
        run.fam_whitenoise = run.get_aligned([['e!0','plow'],['e!0','pmed'],
                                           ['e!0','phigh'],['e!0','ppost'], ['e=0']],
                                          event_shift=[0.0, 0.0, 0.0,0.0, 0.0],
                                          event='ToneTime', xlabel='Time since pattern onset',
                                          plotlabels=['0.1','0.4','0.9','0.6','control'], plotsess=False, pdr=False,
                                          use4pupil=True,
                                          pmetric=pmetric2use
                                          )
        # run.fam_firsts = run.get_firsts(run.familiarity,8,['0.1','0.4','0.9','0.6','control'],'ToneTime')
        # shuffle = False
        # if shuffle:  # decide whether to shuffle
        #     for i in range(5):
        #         run.get_firsts(run.familiarity,8,['0.1','0.4','0.9','0.6','control'],'ToneTime',shuffle=True)
        #
        # run.add_stage3_05_order()


        # run.fam_firsts_pdr = run.get_firsts(run.familiarity,8,['0.1','0.4','0.9','0.6','control'],'ToneTime',pdr=True)
        # run.fam_lasts_pdr = run.get_lasts(run.familiarity,8,['0.1','0.4','0.9','0.6','control'],'ToneTime',pdr=True)
        # run.reward = run.get_aligned([['a1'],['a0']],event='Trial End',xlabel='Time since reward tones',
        #                              plotlabels=['correct','incorrect'],align_col='Trial_End_dt',pdr=False)
        # run.reward = run.get_aligned([['a1']],event='RewardTime',xlabel='Time since reward tones', viol_shift=[-0.0],
        #                                  plotlabels=['correct'],align_col='RewardTone_Time_dt',pdr=True)
        #run.fam_delta = run.get_pupil_delta(run.familiarity[2],['DO48'],['0.1','0.4','0.9','0.6','control'],window=[0,1])

        run_ntones_analysis = False
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

        stages = [4]

        column = 'ToneTime_dt'
        shifts = [[0.0,'ToneTime'],[0.75,'Violation']]

        # column = 'Gap_Time_dt'
        # shifts = [[0.0,'Whitenoise']]

        # events = ['d0','d!0','none']
        # events = ['d0','d1','d2']
        # labels = ['Normal', 'Deviant','None']
        # events = ['d0','d1','d2','d3','none']
        # labels = ['Normal', 'AB_D','ABC_','AB__','None']
        events = ['d0','d1','d-1','none']
        labels = ['Normal', 'AB_D','New Normal','None']
        keys = []
        pattern_types = []

        run.normdev = {}
        for s in stages:
            plt.ion()
            for shift in shifts:
                cond_name = f'stage{s}_{events}_{column}_{shift[0]}'
                event_filters = []
                for e in events:
                    if e[0] == 'd':
                        event_filters.append(['e!0', e,'tones4'])  # 'tones4
                    elif e == 'none':
                        event_filters.append(['e=0'])

                run.normdev[cond_name] = run.get_aligned(event_filters, pmetric=pmetric2use,
                                                         event_shift=[shift[0]]*len(event_filters), align_col=column,
                                                         event=shift[1], xlabel=f'Time since {shift[1]}', pdr=False,
                                                         plotlabels=labels[:len(event_filters)],use4pupil=True,
                                                         plotsess=True)
                keys.append(cond_name)
                run.normdev[cond_name][0].canvas.manager.set_window_title(cond_name)
                fig_savename = f'{cond_name}_a.svg'
                fig_path = os.path.join(run.figdir,fig_savename)
                while os.path.exists(fig_path):
                    file_suffix = os.path.splitext(fig_path)[0][-1]
                    fig_path = f'{os.path.splitext(fig_path)[0][:-1]}' \
                               f'{chr(ord(file_suffix)+1)}{os.path.splitext(fig_path)[1]}'
                    for char in ['[',']',"'"]:
                        fig_path = fig_path.replace(char,'')
                    fig_path = fig_path.replace(', ','_')
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
                           labels, f'{column.split("_")[0]} time',ntrials=[None,10,10])


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
