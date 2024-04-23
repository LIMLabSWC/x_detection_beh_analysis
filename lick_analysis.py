from behaviour_analysis import TDAnalysis
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
import align_functions
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import zscore
from scipy.signal import convolve
from scipy.signal.windows import gaussian

def read_lick_times(beh_event_path:Path):
    beh_events = pd.read_csv(beh_event_path)
    lick_times = beh_events.query('Payload == 0')['Timestamp'].values
    return lick_times

def cluster_spike_times(spike_times:np.ndarray, spike_clusters:np.ndarray)->dict:
    assert spike_clusters.shape == spike_times.shape, Warning('spike times/ cluster arrays need to be same shape')
    cluster_spike_times_dict = {}
    for i in tqdm(np.unique(spike_clusters),desc='getting session cluster spikes times',
                  total=len(np.unique(spike_clusters)),disable=True):
        cluster_spike_times_dict[i] = spike_times[spike_clusters == i]
    return cluster_spike_times_dict



def get_spike_times_in_window(event_time:int,spike_time_dict:dict, window:[list | np.ndarray],fs):
    """
    Get spike times in a specified window for a given event.

    Parameters:
    - event_time: int
    - spike_time_dict: dict
    - window: list or np.ndarray
    - fs: unspecified type

    Returns:
    - window_spikes_dict: dict
    """
    window_spikes_dict = {}

    for cluster_id in tqdm(spike_time_dict, desc='getting spike times for event', total=len(spike_time_dict),
                           disable=True):
        all_spikes = (spike_time_dict[cluster_id] - event_time)   # / fs

        window_spikes_dict[cluster_id] = all_spikes[(all_spikes >= window[0]) * (all_spikes <= window[1])]
    return window_spikes_dict


def gen_spike_matrix(spike_time_dict: dict, window, fs):
    precision = np.ceil(np.log10(fs)).astype(int)
    time_cols = np.round(np.arange(window[0],window[1]+1/fs,1/fs),precision)
    spike_matrix = pd.DataFrame(np.zeros((len(spike_time_dict), int((window[1]-window[0])*fs)+1)),
                                index=list(spike_time_dict.keys()),columns=time_cols)
    rounded_spike_dict = {}
    for cluster_id in tqdm(spike_time_dict, desc='rounding spike times for event', total=len(spike_time_dict),
                           disable=True):
        rounded_spike_dict[cluster_id] = np.round(spike_time_dict[cluster_id],precision)
    for cluster_id in tqdm(rounded_spike_dict, desc='assigning spikes to matrix', total=len(spike_time_dict),
                           disable=True):
        spike_matrix.loc[cluster_id][rounded_spike_dict[cluster_id]] = 1
    return spike_matrix


class SessionLicks:
    def __init__(self, spike_times: np.ndarray, sess_start_time: float, fs=1e3, resample_fs=1e2):
        """
        Initialize the SpikeSorter class.

        Parameters:
            spike_times_path (Path|str): The path to the spike times file.
            spike_clusters_path (Path|str): The path to the spike clusters file.
            sess_start_time (float): The start time of the session.
            parent_dir (optional): The parent directory. Defaults to None.
            fs (float): The sampling frequency. Defaults to 30000.0.
            resample_fs (float): The resampled sampling frequency. Defaults to 1000.0.

        Returns:
            None
        """
        self.start_time = sess_start_time
        self.fs = fs
        self.new_fs = resample_fs
        self.spike_times, self.clusters = spike_times,np.zeros_like(spike_times)
        # self.spike_times = self.spike_times/fs + sess_start_time  # units seconds

        self.cluster_spike_times_dict = cluster_spike_times(self.spike_times, self.clusters)
        self.bad_units = set()

        self.event_spike_matrices = dict()
        self.event_cluster_spike_times = dict()

    def get_event_spikes(self,event_times: [list|np.ndarray|pd.Series], event_name: str, window: [list| np.ndarray]):

        for event_time in event_times:
            if f'{event_name}_{event_time}' not in list(self.event_cluster_spike_times.keys()):
                self.event_cluster_spike_times[f'{event_name}_{event_time}'] = get_spike_times_in_window(event_time,self.cluster_spike_times_dict,window,self.new_fs)
            if f'{event_name}_{event_time}' not in list(self.event_spike_matrices.keys()):
                self.event_spike_matrices[f'{event_name}_{event_time}'] = gen_spike_matrix(self.event_cluster_spike_times[f'{event_name}_{event_time}'],
                                                                                           window,self.new_fs)

    def curate_units(self):
        # self.bad_units = set()
        for unit in self.cluster_spike_times_dict:
            d_spike_times = np.diff(self.cluster_spike_times_dict[unit])
            if np.mean(d_spike_times>10)>0.05:
                # self.cluster_spike_times_dict.pop(unit)
                self.bad_units.add(unit)
        print(f'popped units {self.bad_units}, remaining units: {len(self.cluster_spike_times_dict) - len(self.bad_units)}/{len(self.cluster_spike_times_dict)}')
        for unit in self.bad_units:
            self.cluster_spike_times_dict.pop(unit)

    def curate_units_by_rate(self):
        # self.bad_units = set()
        for unit in self.cluster_spike_times_dict:
            d_spike_times = np.diff(self.cluster_spike_times_dict[unit])
            if np.mean(d_spike_times) > (1/0.05):
                # self.cluster_spike_times_dict.pop(unit)
                self.bad_units.add(unit)
        print(f'popped units {self.bad_units}, remaining units: {len(self.cluster_spike_times_dict) - len(self.bad_units)}/{len(self.cluster_spike_times_dict)}')
        for unit in self.bad_units:
            self.cluster_spike_times_dict.pop(unit)


def gen_firing_rate_matrix(spike_matrix: pd.DataFrame, bin_dur=0.01, baseline_dur=0.0,
                           zscore_flag=False, gaus_std=0.04) -> pd.DataFrame:
    # print(f'zscore_flag = {zscore_flag}')
    guas_window = gaussian(int(gaus_std/bin_dur),int(gaus_std/bin_dur))
    spike_matrix.columns = pd.to_timedelta(spike_matrix.columns,'s')
    rate_matrix = spike_matrix.T.resample(f'{bin_dur}S').mean().T/bin_dur
    cols = rate_matrix.columns
    rate_matrix = np.array([convolve(row,guas_window,mode='same') for row in rate_matrix.values])
    assert not all([baseline_dur, zscore_flag])
    rate_matrix = pd.DataFrame(rate_matrix,columns=cols)
    if baseline_dur:
        rate_matrix = pd.DataFrame(rate_matrix, columns=cols)
        rate_matrix = rate_matrix.sub(rate_matrix.loc[:,timedelta(0,-baseline_dur):timedelta(0,0)].median(axis=1),axis=0)
    if zscore_flag:
        # rate_matrix = (rate_matrix.T - rate_matrix.mean(axis=1))/rate_matrix.std(axis=1)
        rate_matrix = zscore(rate_matrix,axis=1,)
    rate_matrix = rate_matrix.fillna(0)
    return rate_matrix


animals = ['DO80','DO81','DO82']
dates = ['240415','240416','240417']
run = TDAnalysis(r'H:\data\Dammy',animals,[min(dates), max(dates)])

freq_trials = run.trialData.query('PatternPresentation_Rate == 0.1 & Tone_Position == 0 & Session_Block == 0')
rare_trials = run.trialData.query('PatternPresentation_Rate == 0.9 & Tone_Position == 0 & Session_Block == 0')
unique_sess = freq_trials.index.droplevel('Trial_Start_dt').unique()

dt_patt_plot = plt.subplots()
dt_patts = []
for si,(subset,s_name) in enumerate(zip([freq_trials, rare_trials],['freq','rare'])):
    dt_patt = np.hstack([subset.loc[sess, 'ToneTime_dt'].iloc[1:-1].diff().dt.total_seconds().values for sess in unique_sess])
    print(f'{s_name} mean: {np.nanmedian(dt_patt)}, std: {np.nanstd(dt_patt)}')
    dt_patts.append(dt_patt[dt_patt<200])
    # dt_patt_plot[1].hist(dt_patt, alpha=0.5,bins=500, label=f'{s_name} block',density=True)
dt_patt_plot[1].boxplot(dt_patts,usermedians=[np.nanmedian(d) for d in dt_patts],labels=['freq','rare'],
                        showfliers=False,bootstrap=10000)

dt_patt_plot[1].set_ylabel('Time between Tone presentations (s)')
# dt_patt_plot[1].set_xlim(0,200)
dt_patt_plot[0].show()
dt_patt_plot[0].savefig('dt_patt.svg',bbox_inches='tight')


lick_times_path = Path(r'X:\Dammy\harpbins\DO79_HitData_240417b_event_data_32.csv')
licks = read_lick_times(lick_times_path)

sound_events_path = Path(r'X:\Dammy\harpbins\DO79_SoundData_240417b_write_indices.csv')
sound_events = pd.read_csv(sound_events_path)
lick_obj = SessionLicks(licks, 0)

# spikes_in_window =
window = [-3,3]
lick_obj.get_event_spikes(sound_events.query('Payload == 3')['Timestamp'].values,'lick_to_X',window)
licks_to_X = pd.concat([e for e in lick_obj.event_spike_matrices.values()])
fig,ax = plt.subplots()
ax.imshow(licks_to_X.values,cmap='binary',extent=[window[0],window[1],licks_to_X.shape[0],0],aspect='auto',
          origin='upper')
ax.axvline(0,c='k',ls='--')
ax.set_ylabel('Trials')
ax.set_xlabel('time since X (s)')
fig.show()
fig.savefig('licks_to_X.svg',bbox_inches='tight')
pkldir = Path(r'X:\Dammy\mouse_pupillometry\pickles')
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

