import analysis_utils as utils
from analysis_utils import align_wrapper, plot_eventaligned
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime, timedelta
import random

# code for analysing pupil self.data


class Main:

    def __init__(self,pklfilename,tdatapkl, duration_window=(-1,2)):
        with open(pklfilename,'rb') as pklfile:
            self.data = pickle.load(pklfile)
        with open(tdatapkl,'rb') as pklfile:
            self.trial_data = pickle.load(pklfile)
        self.add_pretone_dt()
        self.duration = duration_window
        self.labels = [e.split('_')[0] for e in self.data]
        self.sessions = list(self.data.keys())

        today =  datetime.strftime(datetime.now(),'%y%m%d')
        self.figdir = os.path.join(os.getcwd(),'figures',today)
        if not os.path.isdir(self.figdir):
            os.mkdir(self.figdir)
        self.figdir = f''
        self.samplerate = self.data[self.sessions[0]].pupildf.index.to_series().diff().mean().total_seconds()

    def add_pretone_dt(self):
        for sess in self.data:
            td_df = self.data[sess].trialData
            td_df['Pretone_end_dt'] = [tstart+timedelta(0,predur) for tstart, predur in
                                       zip(td_df['Trial_Start_dt'], td_df['PreTone_Duration'])]

    def get_aligned(self,filters,viol_shift=[0.5], align_col='ToneTime_dt', event='ToneTime',
                        xlabel='',plotsess=False, plotlabels=('Normal', 'Deviant'), aggregate=False):

        if len(viol_shift) != len(filters):
            if len(viol_shift*len(filters)) == len(filters):
                viol_shifts = viol_shift*len(filters)
            else:
                print('invalid viol_shift param')
                return None
        else:
            viol_shifts = viol_shift

        print(viol_shifts)
        if xlabel != '':
            xlabel = f'Time from {event.split("_")[:-1]}'
        tonealigned_viols, tonealigned_viols_df, tonealigned_trials = align_wrapper(self.data,filters,align_col,
                                                                                    self.duration,alignshifts=viol_shifts,
                                                                                    plotlabels=plotlabels, plottitle='Violation',
                                                                                    xlabel=xlabel, animal_labels=self.labels,
                                                                                    plotsess=plotsess)
        all_normals = tonealigned_viols[0]
        tonealigned_viols_df.columns = plotlabels
        tonealigned_viols_fig, tonealigned_viols_ax = plot_eventaligned(tonealigned_viols,plotlabels,
                                                                        self.duration, event)
        tonealigned_viols_fig.canvas.manager.set_window_title(f'All trials aligned to {event}')
        # tonealigned_viols_ax.set_ylim((-.5,1))
        tonealigned_viols_ax.set_ylabel('zcscored pupil size')
        tonealigned_viols_ax.axvline(0,ls='--',color='k')
        tonealigned_viols_fig.set_size_inches(8,6)
        tonealigned_viols_fig.savefig(os.path.join(self.figdir,'violaligned_normdev.png'),bbox_inches='tight')

        return tonealigned_viols_fig,tonealigned_viols_ax,tonealigned_viols,tonealigned_trials

    def get_firsts(self,aligned_data,n_firsts, plotlabels, event,shuffle=False):

        aligned_arr = aligned_data[2]
        aligned_trialnums = aligned_data[3]
        aligned_firsts = []
        for i,ptype in enumerate(aligned_arr):
            sess_start_idx = 0
            list_ptype_firsts = []
            for s in self.sessions:
                if aligned_trialnums[s][i]>=n_firsts:
                    if shuffle:
                        np.random.shuffle(ptype[sess_start_idx:sess_start_idx+aligned_trialnums[s][i]])
                        list_ptype_firsts.append(ptype[sess_start_idx:sess_start_idx+n_firsts,:])
                    else:
                        try: list_ptype_firsts.append(ptype[sess_start_idx:sess_start_idx+n_firsts,:])
                        except IndexError: print('out of bounds')
                sess_start_idx += aligned_trialnums[s][i]
            aligned_firsts.append(np.concatenate(list_ptype_firsts))

        fig,ax = plot_eventaligned(aligned_firsts,plotlabels,self.duration,event)
        # ax.set_ylim((-.5,1))
        if shuffle:
            ax.set_title('Shuffled to Tone time')
            fig.canvas.manager.set_window_title('Shuffled to Tone time')
        else:
            fig.canvas.manager.set_window_title(f'First {n_firsts} each session')
        ax.set_ylabel('zcscored pupil size')
        ax.axvline(0,ls='--',color='k')
        fig.set_size_inches(8,6)
        fig.savefig(os.path.join(self.figdir,'violaligned_normdev.png'),bbox_inches='tight')

        return fig, ax, aligned_firsts

    def get_pdr(self, aligned_data,plotlabels,event):

        aligned_arr = aligned_data[2]
        aligned_trialnums = aligned_data[3]
        aligned_pdrs = []

        for i,ptype in enumerate(aligned_arr):
            aligned_deriv = np.diff(ptype,axis=1)/self.samplerate
            aligned_pdrs.append((aligned_deriv>0.0).astype(int))

        fig,ax = plot_eventaligned(aligned_pdrs,plotlabels,self.duration,event)
        fig.canvas.manager.set_window_title('PDR by condition')
        ax.set_ylabel('PDR a.u')
        ax.axvline(0,ls='--',color='k')
        ax.set_title('Dilation rate aligned to ToneTime')
        fig.set_size_inches(8,6)

        return fig, ax, aligned_pdrs



    # def aggregate_aligned(self):
    #     # combination attempt
    #     _array = None
    #     n_traces = 10
    #     type_N_traces = []
    #     for i,col in enumerate(tonealigned_viols_df.columns):
    #         first_traces = []  # list all first traces for this current pattern type
    #         _type_first = []
    #         last_traces = []
    #         _type_last = []
    #         for session in list(tonealigned_viols_df.index):
    #             session_traces = tonealigned_viols_df.loc[session][col]
    #             first_traces.append(session_traces[:n_traces,:])
    #             _type_first.append([session,col])
    #             last_traces.append(session_traces[-n_traces:, :])
    #             _type_last.append([session,col])
    #         type_N_traces.append(copy([np.concatenate(first_traces,axis=0),np.concatenate(last_traces, axis=0)]))
    #
    #     type0_traces = type_N_traces[0]
    #     type1_traces = type_N_traces[1]
    #
    # first_last_plot = plt.subplots()
    # subset_type = ['firsts','lasts']
    # for ptype, pytype_label in zip([type0_traces,type1_traces],['Normal', 'Deviant']):
    #     if pytype_label == 'Normal':
    #         plot_eventaligned([all_normals],
    #                       [f'All Normals'],duration,samplerate,
    #                       'Normal vs Deviant aligned to Violation', plotax=first_last_plot)
    #
    #         plot_eventaligned([ptype[0],ptype[1]],
    #                       [f'{pytype_label} {subset_type[0]}', f'{pytype_label} {subset_type[1]}'],duration,samplerate,
    #                       'Normal vs Deviant aligned to Violation', plotax=first_last_plot)
    # first_last_plot[1].set_ylim((-.5,.5))
    # first_last_plot[1].axvline(0,ls='--', color='k')
    # first_last_plot[1].set_xlabel('Time from violation (s)')
    # first_last_plot[1].set_ylabel('zscored pupil size')
    # first_last_plot[1].set_title('First vs Last presentation of deviants (Last 5, 9 sessions)')
    # first_last_plot[0].set_size_inches(4,3, forward=True)
    # first_last_plot[0].savefig(os.path.join(self.figdir,'firstlast_normdev_va.png'),bbox_inches='tight')
    #
    # dev_traces = {}
    # dev_traces_list = []
    #
    # devsubset_TA_viols, df_devTA_traces = align_wrapper(self.data,[['devord','d!0'], ['devrep','d!0']],
    #                                                   'ToneTime_scalar',duration,samplerate,alignshifts=[.5,.5,.75,.5])
    # df_devTA_traces.columns = ['Bad Order', 'Repeated']
    # devsubset_fig, devsubset_ax = plot_eventaligned(devsubset_TA_viols,df_devTA_traces.columns,duration,samplerate,
    #                                                 'Violation', plotax=(tonealigned_viols_fig, tonealigned_viols_ax))
    # devsubset_ax.set_ylim((-.5,.5))
    # devsubset_ax.axvline(0,ls='--',color='k')
    # devsubset_ax.set_xlabel('Time from violation (s)')
    # devsubset_fig.set_size_inches(4,3)
    #
    #
    # violaligned_traces,violaligned_df = align_wrapper(self.data,[['d0'], ['d1'],['d2'],['d3']],
    #                                                   'ToneTime_scalar',duration,samplerate,alignshifts=[.5,.5,.75,.5])
    # violaligned_df.columns = ['Normal','AB_D','ABC_','AB__']
    # violaligned_fig,violaligned_ax = plot_eventaligned(violaligned_traces,violaligned_df.columns,duration,samplerate, 'Violation Time by pattern')
    # violaligned_ax.set_ylim((-.5,.5))
    # violaligned_ax.set_xlabel('Time from violation (s)')
    # violaligned_ax.set_ylabel('zscored pupil size')
    # violaligned_ax.axvline(0,ls='--', color='k')
    # violaligned_fig.set_size_inches(4,3)
    # violaligned_fig.savefig(os.path.join(self.figdir,'norm_dev_byclass_va.png'),bbox_inches='tight')
    #
    # all_devsubsetplot = plot_eventaligned(dev_traces_list,['Bad Order', 'Repeated'],duration,samplerate,
    #                                       'Normal vs Deviant aligned to Violation',plotax=[violaligned_fig, violaligned_ax])
    # all_devsubsetplot[1].set_ylim(-.5,.5)
    # all_devsubsetplot[0].set_size_inches(4,3)
    #
    # violaligned_devsubset_traces,violaligned_devsubset_df = align_wrapper(self.data,[['d0'], ['d1']],
    #                                                   'ToneTime_scalar',duration,samplerate,alignshifts=[.5,.5])
    # violaligned_devsubset_df.columns = ['Normal','AB_D']
    # violaligned_devsubset_fig,violaligned_devsubset_ax = plot_eventaligned(violaligned_devsubset_traces,  # [0],violaligned_devsubset_traces[0]+violaligned_devsubset_traces[1]]
    #                                                                              violaligned_devsubset_df.columns,
    #                                                                              duration,samplerate, 'Violation Time by pattern')
    # violaligned_devsubset_ax.set_ylim((-.5,.5))
    # violaligned_ax.set_xlabel('Time from violation (s)')
    # violaligned_devsubset_ax.set_ylabel('zscored pupil size')
    # violaligned_devsubset_ax.axvline(0,ls='--', color='k')
    # violaligned_devsubset_fig.set_size_inches(4,3)
    # violaligned_devsubset_fig.savefig(os.path.join(self.figdir,'norm_dev1_va.png'),bbox_inches='tight')
    #
    # rewardtone_traces, rewardtone_df = align_wrapper(self.data,[['a0'], ['a1']],'Gap_Time_scalar',duration,samplerate)
    # rewardtone_fig,rewardtone_ax = plot_eventaligned(rewardtone_traces,rewardtone_df.columns,duration,samplerate, '"X"')
    # rewardtone_ax.axvline(0,ls='--',color='k')
    # rewardtone_fig.set_size_inches(4,3)
    #
    # # look at descending
    # dev_traj_traces, dev_traj_df = align_wrapper(self.data,[['d0'],['d!0','devassc'], ['d!0','devdesc']]
    #                                              ,'ToneTime_scalar',duration, samplerate,alignshifts=[.5,.5,.5])
    # dev_traj_df.columns = ['Normal','Deviant Assc','Deviant Dessc']
    # dev_traj_fig,dev_traj_ax = plot_eventaligned(dev_traj_traces,dev_traj_df.columns,duration,samplerate,
    #                                              'Ascending and Descending Deviant patterns')
    # dev_traj_ax.axvline(0,ls='--',color='k')
    # devsubset_ax.set_xlabel('Time from violation (s)')
    # dev_traj_ax.set_ylim(-.5,.5)
    # dev_traj_ax.set_ylabel('zscored pupil size')
    # dev_traj_fig.set_size_inches(4,3)
    # dev_traj_fig.savefig(os.path.join(self.figdir,'assc_desc.png'),bbox_inches='tight')
    #
    # # xy position of the eye norm dev
    # # for i in ['x','y']:
    # #     tonealigned_viols_xy, tonealigned_viols_xy_df  = align_wrapper(self.data,[['d0''], ['d!0'']],'ToneTime_scalar',
    # #                                                              duration,samplerate,alignshifts=[.5,.5],coord=i)
    # #     tonealigned_viols_xy_df.columns = ['Normal', 'Deviant']
    # #     tonealigned_viols_xy_fig, tonealigned_viols_xy_ax = plot_eventaligned(tonealigned_viols_xy,['Normal', 'Deviant'],
    # #                                                                     duration,samplerate, 'Violation')
    # #     tonealigned_viols_xy_ax.set_ylim((-.5,.5))
    # #     tonealigned_viols_xy_ax.set_ylabel(f'{i} position')
    # #     tonealigned_viols_xy_ax.axvline(0,ls='--',color='k')
    # #     tonealigned_viols_xy_fig.set_size_inches(4,3)
    #
    # # compare normals
    # normcomp_traces, normcomp_df = align_wrapper(self.data,[['d0'],['normtrain'],['normtest'],['d2']]
    #                                              ,'ToneTime_scalar',duration, samplerate,alignshifts=[.5,.5,.5,.5])
    # normcomp_df.columns = ['All Normal','Normal: Train Phase','Normal: Test Phase','Deviants']
    # normcomp_fig, normcomp_ax = plot_eventaligned(normcomp_traces,normcomp_df.columns, duration,samplerate,
    #                                               'Normal Patterns comparison')
    # normcomp_ax.set_ylim(-.5,.5)
    # normcomp_ax.set_ylabel('zscored pupil size')
    # normcomp_ax.axvline(0,ls='--',color='k')
    # normcomp_fig.set_size_inches(4,3)
    # normcomp_fig.savefig(os.path.join(self.figdir,'normcomp.png'),bbox_inches='tight')
    #
    # # Hilde stuff:
    # # mean_normals = tonealigned_viols[0].mean(axis=0)  # mean of normal trace
    # mean_normals = tonealigned_viols[0][-1]
    # max_diffs_list = []
    # eval_window = np.array([0,1]) + (-duration[0])  # in seconds
    # eval_window_ts = (eval_window/samplerate).astype(int)  # as index number
    #
    # for r,trace in enumerate(tonealigned_viols[0]):
    #     trial_dev_trace = tonealigned_viols[0][r,:]
    #     diff_trace = (mean_normals[eval_window_ts[0]:eval_window_ts[1]]-trial_dev_trace[eval_window_ts[0]:eval_window_ts[1]]) # get diff using a metric
    #     max_diffs_list.append([diff_trace.max(),diff_trace.sum(),np.where(diff_trace==diff_trace.max())[0][0]*samplerate])
    # max_diffs_arr = np.array(max_diffs_list)
    # # plt.plot(max_diffs_arr[:,0])
    # plt.plot(max_diffs_arr[:,1])

if __name__ == "__main__":
    pkl2use = r'pickles\human_familiarity_3d_200Hz_015Shan_driftcorr_hpass01.pkl'
    run = Main(pkl2use, r'pickles\humans_21_25_td.pkl', (-1,3))
    # run2 = Main(pkl2use.replace('5Hz','25Hz'), r'pickles\humans_21_25_td.pkl', (-1,2))

    run.data.pop('Human24_220407')
    run.labels.pop(4)
    run.sessions.pop(4)

    # run2.data.pop('Human24_220407')
    #
    # # run.data.pop('Human23_220407')
    #
    # run2.sessions.pop(3)
    # run2.labels.pop(3)

    # run.sessions.pop(2)
    # #
    # # run.labels.pop(0)
    # run.labels.pop(2)

    # run.normvdev = run.get_aligned([['e!0','tones4','d0'],['e!0','tones4','d1']],event='Violation Time')
    # run.familiarity = run.get_aligned([['plow'],['pmed'],['phigh'],['ppost']],
    # for run in [run,run1shan]:
    # run.data.pop('Human23_220407')

    run.familiarity = run.get_aligned([['e!0','plow','tones4'],['e!0','tones4','pmed'],
                                       ['e!0','tones4','phigh'],['e!0','tones4','ppost'], ['e=0']],
                                      viol_shift=[0.0],
                                      event='ToneTime',xlabel='Time since pattern onset',
                                      plotlabels=['0.2','0.4','0.9','0.6','control'],plotsess=False)
    run.fam_firsts = run.get_firsts(run.familiarity,8,['0.2','0.4','0.9','0.6','control'],'ToneTime')
    # for i in range(5):
    #     run.get_firsts(run.familiarity,8,['0.2','0.4','0.9','0.6'],'ToneTime',shuffle=True)
    run.fam_pdr = run.get_pdr(run.familiarity,['0.2','0.4','0.9','0.6','control'],'ToneTime')
    run.reward = run.get_aligned([['a1'],['a0']],event='Trial End',xlabel='Time since reward tones',
                                 plotlabels=['correct','incorrect'],align_col='Trial_End_dt')
    run.reward = run.get_aligned([['a1']],event='RewardTime',xlabel='Time since reward tones', viol_shift=[-0.0],
                                     plotlabels=['correct'],align_col='RewardTone_Time_dt')

    # run = Main(pkl2use.replace('01S','1S'), r'pickles\humans_21_25_td.pkl', (-1,2))
    #
    # run2.familiarity = run2.get_aligned([['e!0','plow','tones4'],['e!0','tones4','pmed'],
    #                                    ['e!0','tones4','phigh'],['e!0','tones4','ppost']], viol_shift=[0.0],
    #                                   event='ToneTime',xlabel='Time since pattern onset',
    #                                   plotlabels=['0.2','0.4','0.9','0.6'],plotsess=False)
    # run2.fam_firsts = run2.get_firsts(run2.familiarity,5,['0.2','0.4','0.9','0.6'],'ToneTime')
    # run2.fam_rand5 = run2.get_firsts(run2.familiarity,5,['0.2','0.4','0.9','0.6'],'ToneTime',shuffle=True)


