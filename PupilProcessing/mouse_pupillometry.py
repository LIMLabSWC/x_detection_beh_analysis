from psychophysicsUtils import *
import analysis_utils as utils
from datetime import datetime, time, timedelta
from matplotlib import pyplot as plt
from copy import deepcopy as copy

def add_datetimecol(df, colname, timefmt='%H:%M:%S.%f'):

    datetime_arr = []
    for t in df[colname]:
        if len(t) > 8:
            datetime_arr.append((datetime.strptime(t[:-1], timefmt)))
        else:
            datetime_arr.append((datetime.strptime(t,'%H:%M:%S')))
    df[f'{colname}_dt'] = np.array(datetime_arr)



def align2event(df,pupiltrace, beh, dur, rate, filters=('a3','b1'), baseline=False,eventshift=0):
    filtered_df = utils.filter_df(df,filters)
    dur = np.array(dur)
    eventpupil_arr = np.zeros((filtered_df.shape[0], int(np.abs(dur).sum() * rate)))
    for i, eventtime in enumerate(filtered_df[beh]):
        # print(pupiltrace)
        # print(eventtime + timedelta(seconds=float(dur[0])),eventtime + timedelta(seconds=float(dur[1])))

        if eventshift != 0:
            dur+=float(eventshift)
        eventpupil = copy(pupiltrace.loc[eventtime + timedelta(seconds=float(dur[0])):
                                         eventtime + timedelta(seconds=float(dur[1]))])
        print('diff',eventpupil.loc[eventtime - timedelta(seconds=1-eventshift)]-eventpupil.loc[eventtime + timedelta(seconds=1-eventshift)],baseline_mean)
        if baseline:
            baseline_mean = eventpupil.loc[eventtime - timedelta(seconds=1-eventshift):
                                           eventtime + timedelta(seconds=1-eventshift)].mean()
            eventpupil = eventpupil-baseline_mean
        if len(eventpupil)>0:
            zeropadded = np.zeros_like(eventpupil_arr[i])
            zeropadded[:len(eventpupil)] = eventpupil
            eventpupil_arr[i] = zeropadded
    return eventpupil_arr


def getpatterntraces(animals, pupiltraces, alldata_df, patterntypes,dur, rate, eventshifts=None,baseline=True):

    list_eventaligned = []
    if eventshifts is None:
        eventshifts = np.zeros(len(patterntypes))
    for i, patternfilter in enumerate(patterntypes):
        print(patternfilter)
        _pattern_tonealigned = []
        for animal in animals:
            for date in pupiltraces[animal].keys():
                session_df = alldata_df.loc[animal, date]  # unfiltered session trial data
                tone_aligned_pattern = align2event(session_df, pupildata[animal][date],
                                                   'ToneTime_dt', dur,rate, ('a3', 'b1', 'c0', patternfilter),
                                                   baseline=baseline, eventshift=eventshifts[i])
                _pattern_tonealigned.append(tone_aligned_pattern)
        list_eventaligned.append(np.concatenate(_pattern_tonealigned, axis=0))
    return list_eventaligned


def plot_eventaligned(eventdata_arr,eventnames,dur,rate):
    event_fig, event_ax = plt.subplots(1)
    for i, trace in enumerate(eventdata_arr):
        event_ax.plot(np.arange(dur[0], dur[1],1/rate), trace.mean(axis=0),
                      label= f'{eventnames[i]}, {trace.shape[0]} Trials')

    # event_ax.axvline(0, c='k', linestyle='--')
    # event_ax.axvline(1, c='k', linestyle='--')

    event_ax.set_xlabel('Time from event (s)')
    event_ax.legend()

    return event_fig,event_ax


animals = [
            # 'DO27',
            'DO28',
            # 'DO29',
]

dates = ['210428','210430']

pupildata = {}

for animal in animals:
    if animal not in pupildata.keys():
        pupildata[animal] = {}
    for date in dates:
        # animal_TD = pd.read_csv(f'C:\\bonsai\\data\\Dammy\\{animal}\\TrialData\\{animal}_TrialData_{date}a.csv')
        animal_pupil = pd.read_csv(f'W:\\mouse_pupillometry\\analysed\\{animal}_{date}_pupildata.csv',skiprows=2)
        animal_pupil = animal_pupil.rename(columns={'Unnamed: 25': 'frametime', 'Unnamed: 26': 'diameter' })

        # frametime_dt = [datetime.strptime(t, '%H:%M:%S.%f') for t in animal_pupil['frametime']]
        # animal_pupil['scalartime'] = [t.hour*60 + t.minute + t.milliseconds/1000 for t in animal_pupil['frametime']
        animal_pupil['scalartime'] = [scalarTime(t) for t in animal_pupil['frametime']]
        # animal_uniform = uniformSample(np.array(animal_pupil),np.array(animal_pupil['scalartime']), new_dt=30)
        _timeseries = []
        for t in animal_pupil['frametime']:
            try:
                _timeseries.append((datetime.strptime(t, '%H:%M:%S.%f')))
            except ValueError:
                _timeseries.append((datetime.strptime(t, '%H:%M:%S')))
        animal_pupil['frametime_dt'] = _timeseries

        samplerate = animal_pupil['frametime_dt'].diff().median().total_seconds()*1000
        pupil_uniformsampled = pd.DataFrame(np.array(animal_pupil['diameter']),
                                           index=animal_pupil['frametime_dt']).resample(f'{str(samplerate)}L').mean()
        # pupil_uniformsampled_med = pd.DataFrame(np.array(animal_pupil['diameter']),
        #                                     index=animal_pupil['frametime_dt']).resample(f'{str(samplerate)}L').median()
        absSpeed = pupil_uniformsampled.diff().abs()*samplerate
        size = pupil_uniformsampled[0]
        size_speed = pd.concat([size,absSpeed],axis=1)
        size_speed.columns = [0,1]
        n_size = 2.5
        n_speed = 2.5
        MAD_speed = (absSpeed - absSpeed.median()).abs().median()
        MAD_size = (size - size.median()).abs().median()
        threshold_speed_low = (absSpeed.median() - n_speed * MAD_speed)[0]
        threshold_size_low = size.median() - n_size * MAD_size
        threshold_speed_high = (absSpeed.median() + n_speed * MAD_speed)[0]
        threshold_size_high = size.median() + n_size * MAD_size

        # data = size * (absSpeed < threshold_speed_high) * (absSpeed > threshold_speed_low)
        # print(" (%.2f%%) " % (100 * (
        #             1 - np.sum((absSpeed < threshold_speed_high) * (absSpeed > threshold_speed_low)) / len(data))),
        #       end="")
        #
        # data = data * (size > threshold_size_low) * (size != 0)  # only take away low sizes and zero values
        # print("and size lowliers/zero (%.2f%%) " % (
        #             100 * (1 - np.sum((size > threshold_size_low) * ((size != 0))) / len(data))), end="")

        size_speed = size_speed[threshold_size_low<size_speed[0]]
        size_speed = size_speed[threshold_size_high >size_speed[0]]
        size_speed = size_speed[threshold_speed_low<size_speed[1]]
        size_speed = size_speed[threshold_speed_high >size_speed[1]]

        pupil_size_downsampled = size_speed.resample('33L').mean()[0]

        # interpolate

        interpolated = pupil_size_downsampled.interpolate(method='time')
        filtered = frequencyFilter(interpolated,pd.Series(interpolated.index),10,1,highpass=False, mousecam=True)
        filtered = frequencyFilter(filtered, pd.Series(interpolated.index),0.05, 0.01,highpass=True, mousecam=True)
        zscored = zScore(filtered)

        zscored_fig, zscored_ax = plt.subplots(1)
        zscored_fig.set_size_inches(9,6,forward=True)
        zscored_ax.plot(pd.Series(interpolated.index),zscored)
        zscored_ax.set_xlabel('Time',fontsize=24)
        zscored_ax.tick_params(axis='x', labelsize=18)
        zscored_ax.set_ylabel('Standard deviations',fontsize=24)
        zscored_ax.tick_params(axis='y', labelsize=18)

        datenowstr = datetime.strftime(datetime.now().date(),'%y%m%d')
        plotdir = os.path.join('plots',datenowstr)
        if os.path.isdir(plotdir) is False:
            os.mkdir(plotdir)

        zscored_fig.savefig(os.path.join(plotdir,f'zscoredtrace_{animal}_{date}.png'))
        plt.close(zscored_fig)

        if date not in pupildata[animal].keys():
            pupildata[animal][date] = {}
        pupildata[animal][date] = pd.Series(zscored,index=interpolated.index)


datadir = r'C:\bonsai\data\Dammy'
dates = ['28/04/2021', '30/04/2021']
plot_colours = ['b','r','c','m','y','g']

trial_data = utils.merge_sessions(datadir,animals,'TrialData',dates)
trial_data = pd.concat(trial_data, sort=False, axis=0)

# formatting
# if tones were played
for animal in animals:
    for date in trial_data.loc[animal].index.unique():
        session_df = trial_data.loc[animal,date]
        add_datetimecol(session_df,'ToneTime')
        toneplayed = [t.time() > time() for t in session_df['ToneTime_dt']]
        trial_data.loc[animal, date]['TonePlayed'] = toneplayed

for col in trial_data.keys():
    if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1:
        if col.find('Wait') == -1 and col.find('dt') == -1:
            print(col)
            add_datetimecol(trial_data,col)

# # align to trial start
# for animal in animals:
#     for date in pupildata[animal].keys():
#         session_df = trial_data.loc[animal,date]
#         session_pupil = pupildata[animal][date]
#         session_df = utils.filter_df(session_df,['b1','a1'])
#         dur = np.array([-5,5])
#         trialpupil_arr = np.zeros((session_df.shape[0],np.abs(dur).sum()*10))
#         for i,trialstart in enumerate(session_df['Trial_Start_dt']):
#             trialpupil = session_pupil.loc[trialstart+timedelta(seconds=float(dur[0])):
#                                            trialstart+timedelta(seconds=float(dur[1]))]
#             trialpupil_arr[i] = trialpupil
# plt.plot(np.arange(dur[0],dur[1],0.1),trialpupil_arr.mean(axis=0),c='k')
# plt.axvline(0)

# align to tones
duration = (-3.0, 4.0)
patterns = ['d0','d1','d2','d3']
patternnames = ['ABCD','AB_D','ABC_', 'AB__']
violshift_t = [.5,.5,.75,.5]
# for patternfilter in patterns:
#     print(patternfilter)
#     pattern_tonealigned = []
#     for animal in animals:
#         for date in pupildata[animal].keys():
#             session_df = trial_data.loc[animal,date]  # unfiltered session trial data
#             tone_aligned_pattern = align2event(session_df,pupildata[animal][date],
#                                                'ToneTime_dt', dur,('a3','b1','c0',patternfilter),baseline=True)
#             pattern_tonealigned.append(tone_aligned_pattern)
#     list_tonealigned.append(np.concatenate(pattern_tonealigned,axis=0))
#
# tone_aligned_fig, tone_aligned_ax = plt.subplots(1)

tone_aligned = getpatterntraces(animals,pupildata,trial_data,patterns,duration,samplerate,baseline=False)
tone_aligned_plot = plot_eventaligned(tone_aligned,patternnames, duration,samplerate)

gap_aligned = getpatterntraces(animals,pupildata,trial_data,['a3'],duration,samplerate)
gap_aligned_plot = plot_eventaligned(gap_aligned,[''],duration,samplerate)

viol_aligned = getpatterntraces(animals,pupildata,trial_data,patterns,duration,samplerate,violshift_t,baseline=True)
viol_aligned_plot = plot_eventaligned(viol_aligned,patternnames,duration,samplerate)
viol_aligned_plot[1].axvline(0.25,c='r',ls='--')

# hist_h5py = pd.read_hdf(r'W:\mouse_pupillometry\analysed\DO27_210430_sessionvidDLC_resnet50_pupildiamterApr28shuffle1_900000.h5')
# scorer = 'DLC_resnet50_pupildiamterApr28shuffle1_900000'

# bodyparts = [
# 'eyeN',
# 'eyeNE',
# 'eyeE',
# 'eyeSE',
# 'eyeS',
# 'eyeSW',
# 'eyeW',
# 'eyeNW'
# ]
# colors = cm.plasma(np.linspace(0, 1, len(bodyparts)))
# datenowstr = datetime.strftime(datetime.now().date(),'%y%m%d')
# plotdir = os.path.join('plots',datenowstr)
# if os.path.isdir(plotdir) is False:
#     os.mkdir(plotdir)
#
# for i, eyepart in enumerate(bodyparts):
#     hist_fig, hist_ax = plt.subplots(1)
#     hist_ax.hist(hist_h5py[scorer,eyepart]['likelihood'],color=colors[i],density=True)
#     hist_fig.savefig(os.path.join(plotdir,f'{eyepart}_likelihood.png'))
#     plt.close(hist_fig