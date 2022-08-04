import pickle
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
from pupil_analysis import Main


if __name__ == "__main__":
    #Human 28 to 31 pickle, made 25.07.22
    pkl2use = r'/Users/hildelt/Documents/Thesis/gd_analysis/pickles/human_class1_3d_200Hz_025Shan_driftcorr_hpass025.pkl'
    #Mouse stage 4 and stgae 5 until 27.07.22
    #pkl2use = r'/Users/hildelt/Documents/Thesis/gd_analysis/pickles/mouse_normdev_2d_200Hz_025Shan_driftcorr_hpass04_wdlc.pkl'
    run = Main(pkl2use, (-1,3))
    paradigm = ['normdev'] #'familiarity' 'altvsrand'
    pmetric2use = 'rawarea_zscored' # to use deeplabcut
    species = 'dlc_radii_a_zscored' # 'mouse'

    if species == 'mouse':
        # Stage 4 normal - deviant - none
        run.normdev4 = run.get_aligned([['e!0', 'd0', 'tones4','a1','stage4'], ['e!0', 'd4', 'tones4','a1','stage4'], ['e=0','a1','stage4']],  # line629 in utils
                                      event_shift=[0.0,0.0,0.0], align_col = 'Pretone_end_dt',
                                      event='ToneTime', xlabel='Time since pattern start', pdr=False,
                                      plotlabels=['normal', 'deviant', 'none'], plotsess=False,
                                      use4pupil=True, pmetric='dlc_radii_a_zscored')

        run.normdev4[0].canvas.manager.set_window_title('Stage4normal_deviant_none_ToneTime')
        #    pattern_df = run.normdev[2][0]
        #    none_df = run.normdev[2][1]
        # Stage 5 normal - deviant - none
        run.normdev5 = run.get_aligned([['e!0', 'd0', 'tones4','a1','stage5'], ['e!0', 'd4', 'tones4','a1','stage5'], ['e=0','a1','stage5']],  # line629 in utils
                                      event_shift=[0.0,0.0,0.0], align_col = 'Pretone_end_dt',
                                      event='ToneTime', xlabel='Time since pattern start', pdr=False,
                                      plotlabels=['normal5', 'deviant5', 'none5'], plotsess=False,
                                      use4pupil=True, pmetric='dlc_radii_a_zscored')
        run.normdev5[0].canvas.manager.set_window_title('Stage 5 normal / deviant / none - ToneTime')


        run.normdev_all = run.get_aligned([['e!0', 'd0', 'tones4','a1'], ['e!0', 'd4', 'tones4','a1'], ['e=0','a1']],  # line629 in utils
                                      event_shift=[0.0,0.0,0.0], align_col = 'Pretone_end_dt',
                                      event='ToneTime', xlabel='Time since pattern start', pdr=False,
                                      plotlabels=['normal5', 'deviant5', 'none5'], plotsess=False,
                                      use4pupil=True, pmetric='dlc_radii_a_zscored')

        run.normdev_all[1].set_title('All normal / deviant / none - ToneTime')

        run.normdev_pdr = run.get_aligned([['e!0', 'd0', 'tones4', 'stage4','a1'], ['e!0', 'd4', 'tones4', 'stage4','a1'],['e=0', 'stage4','a1']],
                                      event_shift=[0.0,0.0,0.0],
                                      event='ToneTime', xlabel='Time since pattern start', pdr=True,
                                      plotlabels=['normal4', 'deviant4', 'none4'], plotsess=False,
                                      use4pupil=True, pmetric='dlc_radii_a_zscored')


        run.normdev_pdr[1].set_title('Stage 4 normal / deviant / none - ToneTime PDR')


        # Normal vs deviant - White noise - stage 4
        run.xdetect4 = run.get_aligned([['e!0', 'd0', 'tones4','a1','stage4'],['e!0', 'd4', 'tones4','a1','stage4'],['e=0','a1','stage4']], #
                                                event_shift=[0.0,0.0,0.0], align_col='Pretone_end_dt', #,0.0
                                                event='White noise', xlabel='Time since X', pdr=False,
                                                plotlabels=['normal','deviant', 'none'], plotsess=False, #'deviant',
                                                use4pupil=True, pmetric='dlc_radii_a_zscored')

        run.xdetect4[1].set_title('Stage 4 normal / deviant / none - white noise')

        # Normal vs deviant - White noise - stage 5
        run.xdetect5 = run.get_aligned([['e!0', 'd0', 'tones4','a1'], ['e!0', 'd4', 'tones4','a1'],['e=0','a1','stage4']],
                                                event_shift=[0.0,0.0,0.0], align_col='Pretone_end_dt', #,0.0
                                                event='White noise', xlabel='Time since X', pdr=False,
                                                plotlabels=['normal','deviant','none'], plotsess=False, #
                                                use4pupil=True, pmetric='dlc_radii_a_zscored')
        run.xdetect5[1].set_title('Stage 5 normal / deviant / none - white noise')

        # Normal vs deviant - White noise - all
        run.xdetect_all = run.get_aligned(
            [['e!0', 'd0', 'tones4', 'a1'], ['e!0', 'd4', 'tones4', 'a1'], ['e=0', 'a1', 'stage4']],
            event_shift=[0.0, 0.0, 0.0], align_col='Gap_Time_dt',  # ,0.0
            event='White_Noise', xlabel='Time since X', pdr=False,
            plotlabels=['normal', 'deviant', 'none'], plotsess=False,  #
            use4pupil=True, pmetric='dlc_radii_a_zscored')
        run.xdetect_all[1].set_title('All sessions normal / deviant / none - white noise')

        # correct vs miss - White noise - stage4
        run.x_performance4 = run.get_aligned(
            [['a1', 'stage4'], ['a0','stage4']],
            event_shift=[0.0, 0.0], align_col='Gap_Time_dt',
            event='White_Noise', xlabel='Time since X', pdr=False,
            plotlabels=['correct', 'miss'], plotsess=False,
            use4pupil=True, pmetric='dlc_radii_a_zscored')

        run.x_performance4[1].set_title('Stage 4 sessions: correct and miss - white noise')

        # correct vs miss - White noise - stage5
        run.x_performance5 = run.get_aligned(
            [['a1', 'stage5'], ['a0', 'stage5']],
            event_shift=[0.0, 0.0], align_col='Gap_Time_dt',
            event='White_Noise', xlabel='Time since X', pdr=False,
            plotlabels=['correct', 'miss'], plotsess=False,
            use4pupil=True, pmetric='dlc_radii_a_zscored')

        run.x_performance5[0].canvas.manager.set_window_title('Stage 5 sessions: correct and miss - white noise')

        run.x_performance_tone = run.get_aligned(
            [['a1', 'stage5'], ['a0', 'stage5']],
            event_shift=[0.0, 0.0], align_col='Pretone_end_dt',
            event='ToneTime', xlabel='Time since X', pdr=False,
            plotlabels=['correct', 'miss'], plotsess=False,
            use4pupil=True, pmetric='dlc_radii_a_zscored')

        run.x_performance_tone[1].set_title('Stage5_correctmiss_ToneTime')


    if species == 'human':
        pmetric2use = 'rawarea_zscored'
        #Normal vs deviant vs none - Tone time
        run.normdev = run.get_aligned([['e!0', 'd0', 'tones4','a1'], ['e!0', 'd4', 'tones4','a1'], ['e=0','a1']],  # line629 in utils
                                      event_shift=[0.0,0.0,0.0], align_col = 'Pretone_end_dt',
                                      event='ToneTime', xlabel='Time since pattern start', pdr=False,
                                      plotlabels=['normal', 'deviant', 'none'], plotsess=False,
                                      use4pupil=True)
        run.normdev[0].canvas.manager.set_window_title('human_normdev_tonetime')

        #Normal vs deviant - White noise
        run.xdetect_normdev = run.get_aligned(
            [['e!0', 'd0', 'tones4', 'a1'], ['e!0', 'd4', 'tones4', 'a1'], ['e=0', 'a1']],
            event_shift=[0.0, 0.0, 0.0], align_col='Gap_Time_dt',  # ,0.0
            event='White_Noise', xlabel='Time since X', pdr=False,
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
            event='White_Noise', xlabel='Time since X', pdr=False,
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

