import pickle
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
from pupil_analysis import Main
from sklearn.utils import resample
from scipy import stats


if __name__ == "__main__":
    #Human 28 to 31 pickle, made 25.07.22
    #pkl2use = r'/Users/hildelt/Documents/Thesis/gd_analysis/pickles/human_class1_3d_200Hz_025Shan_driftcorr_hpass025.pkl'
    #Human familiarity
    #pkl2use = r'/Users/hildelt/Documents/Thesis/gd_analysis/pickles/human_familiarity_3d_200Hz_015Shan_driftcorr_hpass01.pkl'
    #Mouse stage 4 and stgae 5 until 27.07.22
    #old pickle
    #pkl2use = r'/Users/hildelt/Documents/Thesis/gd_analysis/pickles/mouse_normdev_2d_200Hz_025Shan_driftcorr_hpass04_wdlc.pkl'
    # 9. august pickle
    # pkl2use = r'/Users/hildelt/Documents/Thesis/gd_analysis/pickles/mouse_normdev_2d_90Hz_025Shan_driftcorr_hpass025_wdlc_TOM.pkl'
    # 12. august pickle - mouse familiarity, first few (should be last)
    #pkl2use = r'/Users/hildelt/Documents/Thesis/gd_analysis/pickles/mouse_fam_2d_90Hz_025Shan_driftcorr_hpass025_wdlc_TOM_norm1st.pkl'

    pkl2use = r'pickles\mouse_normdev_2d_90Hz_025Shan_driftcorr_hpass025_wdlc_TOM_norm1st.pkl'

    run = Main(pkl2use, (-1,3))
    paradigm = ['normdev'] #'familiarity' 'altvsrand'
    #paradigm = ['familiarity']
    pmetric2use = 'dlc_radii_a_zscored' # 'rawarea_zscored' to use deeplabcut
    species = 'mouse' # 'human'
    paradigm.append('by_mouse')
    mouse = 'DO47'

    if 'familiarity' in paradigm:  # analysis to run for familiarity paradigm
        run.familiarity = run.get_aligned([['e!0','plow','tones4'],['e!0','tones4','pmed'],
                                           ['e!0','tones4','phigh'],['e!0','tones4','ppost'], ['e=0']],
                                          event_shift=[0.0, 0.0, 0.0, 0.0, 0.0],
                                          event='ToneTime', xlabel='Time since pattern onset',
                                          plotlabels=['0.1','0.4','0.9','0.6','control'], plotsess=False, pdr=False,
                                          use4pupil=True,
                                          pmetric=pmetric2use
                                          )

        run.fam_stars = run.get_aligned([['e!0', 'start13', 'tones4'], ['e!0', 'tones4', 'start19'],
                                           ['e!0', 'tones4', 'start17'], ['e!0', 'tones4', 'start15'], ['e=0']],
                                          event_shift=[0.0, 0.0, 0.0, 0.0, 0.0],
                                          event='ToneTime', xlabel='Time since pattern onset',
                                          plotlabels=['13', '19', '17', '15', 'control'], plotsess=False, pdr=False,
                                          use4pupil=True,
                                          pmetric=pmetric2use
                                          )
        run.fam_stars15_19 = run.get_aligned([['e!0', 'tones4', 'start19'], ['e!0', 'tones4', 'start15'], ['e=0']],
                                          event_shift=[0.0, 0.0, 0.0],
                                          event='ToneTime', xlabel='Time since pattern onset',
                                          plotlabels=['19', '15', 'control'], plotsess=False, pdr=False,
                                          use4pupil=True,
                                          pmetric=pmetric2use
                                          )
        if species == 'mouse':
            run.fam_mouse = run.get_aligned([['e!0', 'tones4', 'DO45'], ['e!0', 'tones4', 'DO46'],
                                            ['e!0', 'tones4', 'DO47'], ['e!0', 'tones4', 'DO48'],['e=0']],
                                            event_shift=[0.0, 0.0, 0.0, 0.0, 0.0],
                                            event='ToneTime', xlabel='Time since pattern onset',
                                            plotlabels=['mouse1', 'mouse2', 'mouse3', 'mouse4', 'control'], plotsess=False, pdr=False,
                                            use4pupil=True,
                                            pmetric=pmetric2use
                                            )

    if species == 'mouse' and 'normdev' in paradigm:
        # Stage 4 normal - deviant - none
        run.normdev4 = run.get_aligned([['e!0', 'd0', 'tones4','a1','stage4', mouse], ['e!0', 'd4', 'tones4','a1','stage4', mouse], ['e=0', mouse]],  # line629 in utils
                                      event_shift=[0.0,0.0,0.0], align_col = 'Pretone_end_dt',
                                      event='ToneTime', xlabel='Time since pattern start', pdr=False,
                                      plotlabels=['normal', 'deviant', 'none'], plotsess=False,
                                      use4pupil=True, pmetric='dlc_radii_a_zscored')

        run.normdev4[0].canvas.manager.set_window_title(f'Stage4normal_variableC&D-devs_none_ToneTime_DO48{mouse}')
        norm_trace = run.normdev4[2][0].mean(axis = 0)
        norm_downsampled = resample(norm_trace, random_state=0, n_samples=90, replace=True)
        dev_trace = run.normdev4[2][1].mean(axis = 0)
        dev_downsampled = resample(dev_trace, random_state=0, n_samples=90, replace=True)
        none_trace = run.normdev4[2][2].mean(axis = 0)
        none_downsampled = resample(none_trace, random_state=0, n_samples=90, replace=True)

        stats.ttest_ind(dev_downsampled ,norm_downsampled, equal_var = False)
        stats.ttest_ind(norm_downsampled, none_downsampled, equal_var=False)
        stats.ttest_ind(dev_downsampled, none_downsampled, equal_var=False)

        norm_stdevs = run.normdev4[2][0].std(axis=0)
        norm_mean_stdev = norm_stdevs.mean(axis=0)
        print('mean standard deviation for normal trace: ', norm_mean_stdev)

        dev_stdevs = run.normdev4[2][1].std(axis=0)
        dev_mean_stdev = dev_stdevs.mean(axis=0)
        print('mean standard deviation for deviant trace: ', dev_mean_stdev)

        none_stdevs = run.normdev4[2][2].std(axis=0)
        none_mean_stdev = none_stdevs.mean(axis=0)
        print('mean standard deviation for none trace: ', none_mean_stdev)

        #run.normdev4[0].text(0, 0,
              #f'mean standard deviation for none trace: {none_mean_stdev} \nmean standard deviation for normal trace: {norm_mean_stdev}\nmean standard deviation for deviant trace:  {dev_mean_stdev}')



        #    pattern_df = run.normdev[2][0]
        #    none_df = run.normdev[2][1]
        # Stage 5 normal - deviant - none
        run.normdev5 = run.get_aligned([['e!0', 'd0', 'tones4','a1','stage5', mouse], ['e!0', 'd4', 'tones4','a1','stage5', mouse], ['e=0','a1','stage5', mouse]],  # line629 in utils
                                      event_shift=[0.0,0.0,0.0], align_col = 'Pretone_end_dt',
                                      event='ToneTime', xlabel='Time since pattern start', pdr=False,
                                      plotlabels=['normal', 'deviant', 'none'], plotsess=False,
                                      use4pupil=True, pmetric='dlc_radii_a_zscored')
        run.normdev5[0].canvas.manager.set_window_title(f'Stage 5 normal / deviant / none - ToneTime {mouse}')


        run.normdev_all = run.get_aligned([['e!0', 'd0', 'tones4','a1', mouse], ['e!0', 'd4', 'tones4','a1', mouse], ['e=0','a1', mouse]],
                                      event_shift=[0.0,0.0,0.0], align_col = 'Pretone_end_dt',
                                      event='ToneTime', xlabel='Time since pattern start', pdr=False,
                                      plotlabels=['normal', 'deviant', 'none'], plotsess=False,
                                      use4pupil=True, pmetric='dlc_radii_a_zscored')

        run.normdev_all[0].canvas.manager.set_window_title(f'All normal / deviant / none - ToneTime {mouse}')

       # run.normdev_pdr = run.get_aligned([['e!0', 'd0', 'tones4', 'stage4','a1'], ['e!0', 'd4', 'tones4', 'stage4','a1'],['e=0', 'stage4','a1']],
        #                              event_shift=[0.0,0.0,0.0],
         #                             event='ToneTime', xlabel='Time since pattern start', pdr=True,
          #                            plotlabels=['normal4', 'deviant4', 'none4'], plotsess=False,
           #                           use4pupil=True, pmetric='dlc_radii_a_zscored')


        #run.normdev_pdr[1].set_title('Stage 4 normal / deviant / none - ToneTime PDR')


        # Normal vs deviant - White noise - stage 4
        #run.xdetect4 = run.get_aligned([['e!0', 'd0', 'tones4','a1','stage4'],['e!0', 'd4', 'tones4','a1','stage4'],['e=0','a1','stage4']], #
        #                                        event_shift=[0.0,0.0,0.0], align_col='Gap_Time_dt', #,0.0
        #                                        event='Whitenoise', xlabel='Time since X', pdr=False,
        #                                        plotlabels=['normal','deviant', 'none'], plotsess=False, #'deviant',
        #                                        use4pupil=True, pmetric='dlc_radii_a_zscored')

        #run.xdetect4[0].canvas.manager.set_window_title('Stage 4 normal / deviant / none - white noise')

        # Normal vs deviant - White noise - stage 5
        #run.xdetect5 = run.get_aligned([['e!0', 'd0', 'tones4','a1','stage5'], ['e!0', 'd4', 'tones4','a1','stage5'],['e=0','a1','stage5']],
        #                                        event_shift=[0.0,0.0,0.0], align_col='Gap_Time_dt', #,0.0
        #                                        event='Whitenoise', xlabel='Time since Whitenoise', pdr=False,
        #                                        plotlabels=['normal','deviant','none'], plotsess=False, #
        #                                        use4pupil=True, pmetric='dlc_radii_a_zscored')
        #run.xdetect5[0].canvas.manager.set_window_title('Stage 5 normal  deviant  none - white noise')

        # Normal vs deviant - White noise - all
        #run.xdetect_all = run.get_aligned(
        #    [['e!0', 'd0', 'tones4', 'a1'], ['e!0', 'd4', 'tones4', 'a1'], ['e=0', 'a1']],
        #    event_shift=[0.0, 0.0, 0.0], align_col='Gap_Time_dt',  # ,0.0
        #    event='Whitenoise', xlabel='Time since Whitenoise', pdr=False,
        #    plotlabels=['normal', 'deviant', 'none'], plotsess=False,  #
        #    use4pupil=True, pmetric='dlc_radii_a_zscored')
        #run.xdetect_all[0].canvas.manager.set_window_title('All sessions normal / deviant / none - white noise')

        # correct vs miss - White noise - stage4
        run.x_performance4 = run.get_aligned(
            [['a1', 'stage4',mouse], ['a0','stage4',mouse]],
            event_shift=[0.0, 0.0], align_col='Gap_Time_dt',
            event='Whitenoise', xlabel='Time since Whitenoise', pdr=False,
            plotlabels=['correct', 'miss'], plotsess=False,
            use4pupil=True, pmetric='dlc_radii_a_zscored')

        run.x_performance4[0].canvas.manager.set_window_title(f'Stage 4 sessions: correct and miss  white noise {mouse}')

        # correct vs miss - White noise - stage5
        run.x_performance5 = run.get_aligned(
            [['a1', 'stage5', mouse], ['a0', 'stage5', mouse]],
            event_shift=[0.0, 0.0], align_col='Gap_Time_dt',
            event='Whitenoise', xlabel='Time since Whitenoise', pdr=False,
            plotlabels=['correct', 'miss'], plotsess=False,
            use4pupil=True, pmetric='dlc_radii_a_zscored')

        run.x_performance5[0].canvas.manager.set_window_title(f'Stage 5 sessions: correct and miss - white noise {mouse}')

        if 'by_mouse' in paradigm:

            run.pattern_by_mouse = run.get_aligned([['e!0', 'tones4', 'DO45'], ['e!0', 'tones4', 'DO46'],
                                            ['e!0', 'tones4', 'DO47'], ['e!0', 'tones4', 'DO48'],['e=0']],
                                            event_shift=[0.0, 0.0, 0.0, 0.0, 0.0], align_col = 'Pretone_end_dt',
                                            event='ToneTime', xlabel='Time since pattern start', pdr=False,
                                            plotlabels=['mouse1', 'mouse2', 'mouse3', 'mouse4', 'control'], plotsess=False,
                                            use4pupil=True, pmetric='dlc_radii_a_zscored')

            run.pattern_by_mouse[0].canvas.manager.set_window_title('Mouse pupil response to pattern -all')

            run.whitenoise_mouse = run.get_aligned([['e!0', 'tones4', 'DO45'], ['e!0', 'tones4', 'DO46'],
                                                    ['e!0', 'tones4', 'DO47'], ['e!0', 'tones4', 'DO48'], ['e=0']],
                                                    event_shift=[0.0, 0.0, 0.0, 0.0, 0.0] , align_col='Gap_Time_dt',
                                                    event='Whitenoise', xlabel='Time since X', pdr=False,
                                                    plotlabels=['mouse1', 'mouse2', 'mouse3', 'mouse4', 'control'], plotsess=False,
                                                    use4pupil=True, pmetric='dlc_radii_a_zscored')

            run.whitenoise_mouse[0].canvas.manager.set_window_title('Mouse pupil response to whitenoise -all')


    '''
        # correct vs miss - X - stage5
        run.x_performance_tone = run.get_aligned(
            [['a1', 'stage5'], ['a0', 'stage5']],
            event_shift=[0.0, 0.0], align_col='Pretone_end_dt',
            event='ToneTime', xlabel='Time since X', pdr=False,
            plotlabels=['correct', 'miss'], plotsess=False,
            use4pupil=True, pmetric='dlc_radii_a_zscored')

        run.x_performance_tone[0].canvas.manager.set_window_title('Stage5_correctmiss_ToneTime')

        # 50 50 presentations
        #run.x_performance_tone = run.get_aligned(
         #   [['a1', 'stage5'], ['a0', 'stage5']],# How do I separate between the two 50s
          #  event_shift=[0.0, 0.0], align_col='Pretone_end_dt', #
           # event='ToneTime', xlabel='Time since X', pdr=False,
            #plotlabels=['correct', 'miss'], plotsess=False,
            #use4pupil=True, pmetric='dlc_radii_a_zscored')

        #run.x_performance_tone[1].set_title('Stage5_correctmiss_ToneTime')
    '''
    if species == 'human':
        pmetric2use = 'rawarea_zscored'
        #Normal vs deviant vs none - Tone time
        run.normdev = run.get_aligned([['e!0', 'd0', 'tones4','a1'], ['e!0', 'd4', 'tones4','a1'], ['e=0','a1']],  # line629 in utils
                                      event_shift=[-0.5,-0.5,-0.5], align_col = 'Pretone_end_dt',
                                      event='Violation', xlabel='Time since pattern start', pdr=False,
                                      plotlabels=['normal', 'deviant', 'none'], plotsess=False,
                                      use4pupil=True)
        run.normdev[0].canvas.manager.set_window_title('human_normdev_tonetime')

        #Normal vs deviant - White noise
        run.xdetect_normdev = run.get_aligned(
            [['e!0', 'd0', 'tones4', 'a1'], ['e!0', 'd4', 'tones4', 'a1'], ['e=0', 'a1']],
            event_shift=[0.0, 0.0, 0.0], align_col='Gap_Time_dt',  # ,0.0
            event='Whitenoise', xlabel='Time since X', pdr=False,
            plotlabels=['normal', 'deviant', 'none'], plotsess=False,  #
            use4pupil=True)
        run.xdetect_normdev[0].canvas.manager.set_window_title('human_normdev_whitenoise')
       
        #Normal vs deviant - Start time 'Trial_Start_dt
        run.start_normdev = run.get_aligned(
            [['e!0', 'd0', 'tones4', 'a1'], ['e!0', 'd4', 'tones4', 'a1'], ['e=0', 'a1']],
            event_shift=[0.0, 0.0, 0.0], align_col='Trial_Start_dt',
            event='Trial_Start_dt', xlabel='Time since trial start', pdr=False,
            plotlabels=['normal', 'deviant', 'none'], plotsess=False,
            use4pupil=True)
        run.start_normdev[0].canvas.manager.set_window_title('human_normdev_trialstart')

        #Correct vs miss - White noise
        run.x_performance = run.get_aligned(
            [['a1'], ['a0']],
            event_shift=[0.0, 0.0], align_col='Gap_Time_dt',
            event='Whitenoise', xlabel='Time since X', pdr=False,
            plotlabels=['correct', 'miss'], plotsess=False,
            use4pupil=True)

        run.x_performance[0].canvas.manager.set_window_title('human_miss_correct_whitenoise')

        #Correct vs miss - Tone time
        run.x_performance_tone = run.get_aligned(
            [['a1'], ['a0']],
            event_shift=[0.0, 0.0], align_col='Pretone_end_dt',
            event='ToneTime', xlabel='Time since X', pdr=False,
            plotlabels=['correct', 'miss'], plotsess=False,
            use4pupil=True)

        run.x_performance_tone[0].canvas.manager.set_window_title('human_miss_correct_ToneTime') 
