from psychophysicsUtils import *
import analysis_utils as utils
from datetime import datetime, time, timedelta
from matplotlib import pyplot as plt
from copy import deepcopy as copy
import numpy as np
from sklearn.linear_model import LinearRegression



def align2eventScalar(df,pupilsize,pupiltimes, pupiloutliers,beh, dur, rate, filters=('a3','b1','c1'), baseline=False,eventshift=0,
                      outlierthresh=0.5,stdthresh=3,subset=None):

    pupiltrace = pd.Series(pupilsize,index=pupiltimes)
    outlierstrace = pd.Series(pupiloutliers,index=pupiltimes)
    filtered_df = utils.filter_df(df,filters)
    t_len = int((abs(dur[0]) + abs(dur[1])) / rate)+1
    dur = np.array(dur)
    # print(t_len)
    eventpupil_arr = np.full((filtered_df.shape[0],t_len),np.nan)  # int((np.abs(dur).sum()+rate) * 1/rate))
    outliers = 0
    varied = 0
    if eventshift != 0:
        dur = dur + eventshift
        # print(dur)
    for i, eventtime in enumerate(filtered_df[beh]):
        # print(pupiltrace)
        # print(eventtime + timedelta(seconds=float(dur[0])),eventtime + timedelta(seconds=float(dur[1])))

        eventpupil = copy(pupiltrace.loc[eventtime + dur[0]: eventtime + dur[1]])
        eventoutliers = copy(outlierstrace.loc[eventtime + dur[0]: eventtime + dur[1]])
        # print((eventoutliers == 0.0).sum(),float(len(eventpupil)))
        if (eventoutliers == 1.0).sum()/float(len(eventpupil)) > outlierthresh:
            # print(f'pupil trace for trial {i} incompatible',(eventoutliers == 1).sum())
            outliers += 1
        elif eventpupil.abs().max() > stdthresh:
            # print(f'pupil trace for trial {i} incompatible')
            varied += 1

        else:
            # print('diff',eventpupil.loc[eventtime - 1-eventshift]-eventpupil.loc[eventtime + 1-eventshift])
            if baseline:
                baseline_dur = .5
                baseline_mean = np.nanmean(eventpupil.loc[eventtime - (baseline_dur-eventshift):
                                               eventtime + eventshift])
                eventpupil = eventpupil - baseline_mean
                # # regress out
                # if np.isnan(baseline_mean) is False:
                #     reg = LinearRegression().fit(np.full_like(pupildiams,baseline_mean).reshape(-1, 1), pupildiams)
                #     eventpupil = eventpupil - (reg.coef_ * eventpupil + reg.intercept_) - baseline_mean
            if len(eventpupil)>0:
                zeropadded = np.zeros_like(eventpupil_arr[i])
                zeropadded[:len(eventpupil)] = eventpupil
                eventpupil_arr[i] = zeropadded
    # print(f'Outlier Trials:{outliers}\n Too high varinace trials:{varied}')
    # print(eventpupil_arr.shape)
    nonans_eventpuil = eventpupil_arr[~np.isnan(eventpupil_arr).any(axis=1)]
    if subset is not None:
        midpnt = nonans_eventpuil.shape[0]/2.0
        firsts = nonans_eventpuil[:subset,:]
        middles = nonans_eventpuil[int(midpnt-subset/2.0):int(midpnt+subset/2.0)]
        lasts = nonans_eventpuil[-subset:,:]
        # print(firsts.shape,middles.shape,lasts.shape)
        return [firsts,middles,lasts]
    else:
        return nonans_eventpuil



def getpatterntraces(data, patterntypes,beh,dur, rate, eventshifts=None,baseline=True,subset=None,regressed=False):

    list_eventaligned = []
    if eventshifts is None:
        eventshifts = np.zeros(len(patterntypes))
    for i, patternfilter in enumerate(patterntypes):
        _pattern_tonealigned = []
        if subset is not None:
            firsts, mids, lasts = [], [], []
        for name in data.keys():
            if regressed:
                pupil2use = data[name].pupilRegressed
            else:
                pupil2use = data[name].pupilDiams
            tone_aligned_pattern = align2eventScalar(data[name].trialData,pupil2use,data[name].times,
                                                     data[name].isOutlier,beh,
                                                     dur,rate,[patternfilter],
                                                     outlierthresh=0.25,stdthresh=5,
                                                     eventshift=eventshifts[i],baseline=baseline,subset=subset)
            if subset is not None:
                firsts.append(tone_aligned_pattern[0])
                mids.append(tone_aligned_pattern[1])
                lasts.append(tone_aligned_pattern[2])

            else:
                _pattern_tonealigned.append(tone_aligned_pattern)
        if subset is not None:
            print('len subsests=',len(firsts),len(mids),len(lasts))
            _pattern_tonealigned.append(np.concatenate(firsts,axis=0))
            _pattern_tonealigned.append(np.concatenate(mids,axis=0))
            _pattern_tonealigned.append(np.concatenate(lasts,axis=0))

        list_eventaligned.append(np.concatenate(_pattern_tonealigned, axis=0))
    return list_eventaligned


def plot_eventaligned(eventdata_arr,eventnames,dur,rate,beh):
    event_fig, event_ax = plt.subplots(1)
    print(f'length input lists {len(eventdata_arr)}')
    for i, trace in enumerate(eventdata_arr):
        tseries = np.linspace(dur[0], dur[1], trace.shape[1])
        event_ax.plot(tseries, trace.mean(axis=0),
                      label= f'{eventnames[i]}, {trace.shape[0]} Trials')

    # event_ax.axvline(0, c='k', linestyle='--')
    # event_ax.axvline(1, c='k', linestyle='--')

        plotvar(trace,(event_fig,event_ax),tseries)

    if beh.find('ToneTime') != -1:
        rect1 = matplotlib.patches.Rectangle((0, -10), 0.125, 20, linewidth=0, edgecolor='k', facecolor='k', alpha=0.1)
        rect2 = matplotlib.patches.Rectangle((0.25, -10), 0.125, 20, linewidth=0, edgecolor='k', facecolor='k',
                                             alpha=0.1)
        rect3 = matplotlib.patches.Rectangle((0.5, -10), 0.125, 20, linewidth=0, edgecolor='k', facecolor='k',
                                             alpha=0.1)
        rect4 = matplotlib.patches.Rectangle((0.75, -10), 0.125, 20, linewidth=0, edgecolor='k', facecolor='k',
                                             alpha=0.1)
        event_ax.axvline(0, c='k', alpha=0.5)
        event_ax.add_patch(rect1)
        event_ax.add_patch(rect2)
        event_ax.add_patch(rect3)
        event_ax.add_patch(rect4)

    event_ax.set_xlabel('Time from event (s)')
    event_ax.set_title(f'Pupil size aligned to {beh}')
    event_ax.legend()

    return event_fig,event_ax


def plotvar(data,plot,timeseries):
    ci95 = 1*np.std(data,axis=0)/np.sqrt(data.shape[0])
    print(ci95.shape)
    plot[1].fill_between(timeseries, data.mean(axis=0)+ci95,data.mean(axis=0)-ci95,alpha=0.1)



# loaded merged trial data
animals = [
            # 'DO27',
            # 'DO28',
            # 'DO29',
            'DO39'
]

datadir = r'C:\bonsai\data\Dammy'
dates = ['10/08/2021', '10/08/2021']
# dates = ['26/05/2021', '26/05/2021']
plot_colours = ['b','r','c','m','y','g']

trial_data = utils.merge_sessions(datadir,animals,'TrialData',dates)
trial_data = pd.concat(trial_data, sort=False, axis=0)

# set up pupil data class for mice. value for each key is object with data 1 session


# dates = ['210427','210428','210430','210510']
# dates = ['210513','210514','210519','210521','210526']
dates = ['210810']
samplerate = round(1/59,3)
data = {}
for animal in animals:
    for date in dates:
        name = f'{animal}_{date}'
        try:
            animal_pupil = pd.read_csv(f'W:\\mouse_pupillometry\\analysed\\{animal}_{date}_pupildata_hypcffit.csv', skiprows=2)
            data[name] = pupilDataClass(animal)
            # Load pupil date for animal as pandas dataframe
            animal_pupil = animal_pupil.rename(columns={'Unnamed: 25': 'frametime', 'Unnamed: 26': 'diameter'})
            animal_pupil['scalar'] = [scalarTime(t) for t in animal_pupil['frametime']]

            data[name].rawPupilDiams = np.array(animal_pupil['diameter'])
            data[name].rawTimes = np.array(animal_pupil['scalar'])
            # # use pd. resample
            # _timeseries = []
            # for t in animal_pupil['frametime']:
            #     try:
            #         _timeseries.append((datetime.strptime(t, '%H:%M:%S.%f')))
            #     except ValueError:
            #         _timeseries.append((datetime.strptime(t, '%H:%M:%S')))
            # animal_pupil['frametime_dt'] = _timeseries
            #
            # pupil_uniformsampled = pd.DataFrame(np.array(animal_pupil['diameter']),
            #                                     index=animal_pupil['frametime_dt']).resample(f'{str(samplerate*1000)}L').mean()
            # data[name].pupilDiams = np.array(pupil_uniformsampled[0])
            # data[name].times = np.array([(t.hour * 60 + t.minute) * 60 + t.second + t.microsecond/1000000.0
            #                     for t in pupil_uniformsampled.index])
            # # end of using pd.resample

            data[name].uniformSample(samplerate)
            data[name].removeOutliers(n_speed=4, n_size=4)
            # data[name].downSample()
            data[name].interpolate(gapExtension=0.)
            data[name].frequencyFilter(lowF=0.1, lowFwidth=0.01, highF=5, highFwidth=1)
            data[name].zScore()

            session_TD = trial_data.loc[animal, date]
            for col in session_TD.keys():
                if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1:
                    if col.find('Wait') == -1 and col.find('dt') == -1:
                        session_TD[f'{col}_scalar'] = [scalarTime(t) for t in session_TD[col]]
            data[name].trialData = session_TD  # add session trialdata
            data[name].plot()
        except FileNotFoundError:
            pass
# align 2 tone trial start:
# name = list(data.keys())[0]
duration = [-1,3]
tonealigned = getpatterntraces(data,['a3'],'ToneTime_scalar',duration,samplerate,baseline=False)[0]

tonealigned = tonealigned[~np.isnan(tonealigned).any(axis=1)]
tone_aligned_plot = plot_eventaligned([tonealigned],['ToneStart'],duration,samplerate,'ToneTime')

patterns = ['d0','d1','d3','d2']
patternnames = ['ABCD','AB_D','AB__','ABC_']
# patterns = ['d0', 'd3']
# patterns = ['d0', 'd!0']
# patternnames = ['Normal','Violations']

# patternnames = ['Normal', 'Deviant']

# Tone aligned by pattern type
tonealigned_viols = getpatterntraces(data,patterns,'ToneTime_scalar',duration,samplerate,baseline=True)
tonealigned_viols_fig,tonealigned_viols_ax = plot_eventaligned(tonealigned_viols,patternnames,duration,samplerate, 'ToneTime by pattern')
tonealigned_viols_ax.set_ylim((-.5,1))
tonealigned_viols_ax.set_ylabel('zscored pupil diameter')

# viol aligned by pattern
violshift_t = [.5,.5,.5,.75]
violaligned = getpatterntraces(data,patterns[:-1],'ToneTime_scalar',duration,samplerate,baseline=True, eventshifts=violshift_t[:-1])
violaligned_fig,violaligned_ax = plot_eventaligned(violaligned,patternnames,duration,samplerate, 'Violation Time by pattern')
# violaligned_fig,violaligned_ax = plot_eventaligned(violaligned[:-1],patterns[:-1],duration,samplerate, 'Violation Time by pattern')
violaligned_ax.set_ylim((-1,1))
violaligned_ax.set_ylabel('zscored pupil diameter')
violaligned_ax.axvline(0,ls='--', color='k')

# normal vs viol
tonealigned_viols = getpatterntraces(data, ['d0', 'd!0'],'ToneTime_scalar',duration,samplerate,baseline=True)
tonealigned_viols_fig,tonealigned_viols_ax = plot_eventaligned(tonealigned_viols,['Normal', 'Deviant'],duration,samplerate, 'ToneTime by pattern')
tonealigned_viols_ax.set_ylim((-.5,1))
tonealigned_viols_ax.set_ylabel('zscored pupil diameter')

violaligned = getpatterntraces(data,['d0', 'd!0'],'ToneTime_scalar',duration,samplerate,baseline=True, eventshifts=[0.5,0.5])
violaligned_fig,violaligned_ax = plot_eventaligned(violaligned,['Normal', 'Deviant'],duration,samplerate, 'Violation Time by pattern')
violaligned_ax.set_ylim((-1,1))
violaligned_ax.set_ylabel('zscored pupil diameter')
violaligned_ax.axvline(0,ls='--', color='k')

# violation aligned by subset
violaligned_subsets = getpatterntraces(data,patterns,'ToneTime_scalar',duration,samplerate,
                                       baseline=True,subset=10)

subset_names = ['Normal Firsts']
# for p in violaligned_subsets:
#     len_subs = int(len(p)/3)
#     firsts= p[:len_subs]
#     mids = p[len_subs:len_subs*2]
#     lasts = p[-len_subs:]
# viol_subsets_2plot = [np.concatenate(firsts,axis=0),np.concatenate(mids,axis=0),np.concatenate(lasts,axis=0)]
# violaligned_subsets_fig,violaligned_subsets_ax = plot_eventaligned(viol_subsets_2plot,patternnames,duration,samplerate, 'ToneTime')
# violaligned_subsets_ax.set_ylim((-0.6,1))
# violaligned_subsets_ax.set_ylabel('zscored pupil diameter')

rewardaligned = getpatterntraces(data,['a1'],'RewardTone_Time_scalar',duration,samplerate,baseline=True)[0]
rewardaligned_fig, rewardaligned_ax = plot_eventaligned([rewardaligned],['RewardTones'],[-1,2],samplerate,'Reward Time')
rewardaligned_ax.axvline(0,ls='--', color='k')

# whitenoise aligned
whitenoise_aligned = getpatterntraces(data,['a3'],'Gap_Time_scalar',duration,samplerate,baseline=True)[0]
whitenoise_aligned_fig, whitenoise_aligned_ax = plot_eventaligned([rewardaligned],['White Noise'],[-1,2],samplerate,'White Noise Time')
whitenoise_aligned_ax.axvline(0,ls='--', color='k')
whitenoise_aligned_ax.axvline(0.25,ls='--', color='grey')


# # load dpixels
# dpx_files = [r'C:\Users\Danny\Desktop\DO28_210513_dpixels_sum.csv',r'C:\Users\Danny\Desktop\DO27_210513_dpixels_sum.csv']
# datakeys = ['DO28_210513','DO27_210513']
# starts = [data['DO28_210513'].trialData['Trial_Start_scalar'][0],
#           data['DO27_210513'].trialData['Trial_Start_scalar'][0]]
# for i, f in enumerate(dpx_files):
#     dpx_df = pd.read_csv(f)['Mean_dpixels']
#     basetime = np.arange(0,len(dpx_df))/24
#     alignedtime = basetime + starts[i]
#     data[datakeys[i]].dpx, data[datakeys[i]].dpxtimes =  uniformSample(dpx_df,alignedtime,samplerate)

# # regress dpx:
# for name in data.keys():
#     pupildiams = data[name].pupilDiams
#     dpx = data[name].dpx
#     reg = LinearRegression().fit(dpx[:len(pupildiams)].reshape(-1, 1), pupildiams)
#     print(f'{name} coef = {reg.coef_} int = {reg.intercept_}')
#     corr_dpx = pupildiams-(reg.coef_*pupildiams+reg.intercept_)
#     data[name].pupilRegressed = corr_dpx
# # plot regressed
#
# violaligned_reg = getpatterntraces(data,patterns,'ToneTime_scalar',duration,samplerate,baseline=True,eventshifts=violshift_t)
# violaligned_reg_fig,violaligned_reg_ax = plot_eventaligned(violaligned_reg,patternnames,duration,samplerate, 'Violation Time (baseline regressed)')
# violaligned_reg_ax.set_ylim((-0.6,1))
# violaligned_reg_ax.set_ylabel('zscored pupil diameter')
# violaligned_reg_ax.axvline(0,ls='--', color='k')
#
# violreg_subset = getpatterntraces(data,patterns,'ToneTime_scalar',duration,samplerate,baseline=True,eventshifts=violshift_t,subset=10)
# fig, ax = plt.subplots(1)
# ax.plot(violreg_subset[3][-10:].mean(axis=0))
# plot_eventaligned(violreg_subset,patternnames,duration,samplerate, 'Violation Time (baseline regressed)')


