from psychophysicsUtils import *
import analysis_utils as utils
from datetime import datetime, time, timedelta
from matplotlib import pyplot as plt
from copy import deepcopy as copy
import numpy as np
from sklearn.linear_model import LinearRegression



def align2eventScalar(df,pupilsize,pupiltimes, pupiloutliers,beh, dur, rate, filters=('4pupil','b1','c1'), baseline=False,eventshift=0,
                      outlierthresh=0.9,stdthresh=3,subset=None):

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
                baseline_dur = 1.0
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


def getpatterntraces(data, patterntypes,beh,dur, rate, eventshifts=None,baseline=True,subset=None,regressed=False,dev_subsetdf=None,coord=None):

    list_eventaligned = []
    if eventshifts is None:
        eventshifts = np.zeros(len(patterntypes))
    for i, patternfilter in enumerate(patterntypes):
        _pattern_tonealigned = []
        if subset is not None:
            firsts, mids, lasts = [], [], []
        if type(data) == pupilDataClass:
            if regressed:
                pupil2use = data.pupilRegressed
            elif coord == 'x':
                pupil2use = data.xc
            elif coord == 'y':
                pupil2use = data.yc
            else:
                pupil2use = data.pupilDiams
            if dev_subsetdf is None:
                td2use = data.trialData
            elif dev_subsetdf == 'repeat':
                td2use = data.dfDevRepeat
            elif dev_subsetdf == 'order':
                td2use = data.DevOrder
            else: return None
            times2use = data.times
            outs2use = data.isOutlier
        elif type(data) == dict:
            for name in data.keys():
                if regressed:
                    pupil2use = data[name].pupilRegressed
                else:
                    pupil2use = data[name].pupilDiams
                td2use = data[name].trialData
                times2use = data[name].times
                outs2use = data[name].isOutlier
        else:
            print('Incorrect data structure')
            name = None
        if isinstance(patternfilter,str):
            patternfilter = [patternfilter]
        tone_aligned_pattern = align2eventScalar(td2use,pupil2use,times2use,
                                                 outs2use,beh,
                                                 dur,rate,patternfilter,
                                                 outlierthresh=0.5,stdthresh=4,
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


def plot_eventaligned(eventdata_arr,eventnames,dur,rate,beh,plotax=None,pltsize=(12,9)):
    if plotax is None:
        event_fig, event_ax = plt.subplots(1)
    else:
        event_fig, event_ax = plotax
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
    if plotax is None:
        event_ax.set_xlabel('Time from event (s)')
        event_ax.set_title(f'Pupil size aligned to {beh}')
    event_ax.legend()

    return event_fig,event_ax


def plotvar(data,plot,timeseries):
    ci95 = 1*np.std(data,axis=0)/np.sqrt(data.shape[0])
    print(ci95.shape)
    plot[1].fill_between(timeseries, data.mean(axis=0)+ci95,data.mean(axis=0)-ci95,alpha=0.1)


def get_traces(df2use,datadict,dict2use,align_col,ptypes,_dur,_rate,labels,filters,plt_trace=None,align_name=None):
    all_normals = []
    # (df, pupilsize, pupiltimes, pupiloutliers, beh, dur, rate, filters
    #  =('4pupil', 'b1', 'c1'), baseline = False, eventshift = 0,
    #                                                      outlierthresh = 0.9, stdthresh = 3, subset = None)
    if align_name is None:
        align_name = f'Aligned to {align_col}'

    for ax_ix, session in enumerate(datadict.keys()):
        sess_ix = session.split('_')
        df2use.loc[sess_ix[0],sess_ix[1]]
        alignedtraces = align2eventScalar(df2use.loc[sess_ix[0],sess_ix[1]],datadict[session].pupilDiams, datadict[session].times,datadict[session].isOutlier,
                                          align_col,_dur,_rate,filters,outlierthresh=0.9,stdthresh=5,baseline=True)
        dict2use[session] = alignedtraces
        all_normals.append(alignedtraces[0])
        plot_eventaligned(alignedtraces, ptypes, _dur,_rate, align_name,
                          (plt_trace[0],plt_trace[1][ax_ix]))
        sess_plot = plot_eventaligned(alignedtraces, ptypes, _dur, _rate, 'ToneTime by pattern')
        sess_plot[1].set_title(f'{labels[ax_ix]}: Pupil response to patterns',size=8)
        sess_plot[1].set_xlabel('Time from pattern presentation (s)',size=6)
        sess_plot[1].set_ylabel('Normalised pupil diameter',size=6)
        sess_plot[1].set_ylim((-1,2))
        sess_plot[0].set_size_inches(4,3, forward=True)
        # sess_plot[0].tight_layout()
        sess_plot[0].savefig(f'{labels[ax_ix]}_{align_name}.png',bbox_inches='tight')
        plt_trace[ax_ix].set_ylim((-1,2))
        # plt_trace[ax_ix].set_ylabel(f' {session} zscored pupil size')
        plt_trace[ax_ix].set_title(f'Pupil Response to Stimulus {session}')
    all_normals = np.concatenate(all_normals)
    return all_normals,alignedtraces,sess_plot


def align_wrapper(datadict,filters,align_beh, duration, samplerate, alignshifts=None, plotsess=False, plotlabels=None,
                  plottitle=None, xlabel=None,animal_labels=None,plotsave=False,coord=None):
    aligned_dict = {}
    aligned_list = []
    if plotsess:
        if all([plotlabels,plottitle,xlabel,animal_labels]):
            pass
        else:
            print('No plot labels or plot title given for plot. Aborting')
            return None
    for sessix, sess in enumerate(datadict.keys()):
        aligned_dict[sess] = getpatterntraces(datadict[sess],filters,align_beh,duration, samplerate,
                                              baseline=True, eventshifts=alignshifts,coord=coord)
        if plotsess:
            sess_plot = plot_eventaligned(aligned_dict[sess],plotlabels,duration,samplerate,plottitle)
            sess_plot[1].set_title(f'{animal_labels[sessix]}: Pupil response to Pattern',size=8)
            sess_plot[1].set_xlabel(xlabel,size=6)
            sess_plot[1].set_ylabel('Normalised pupil diameter',size=6)
            sess_plot[1].set_ylim((-1,2))
            sess_plot[0].set_size_inches(4,3, forward=True)
            if plotsave:
                sess_plot[0].savefig(f'{animal_labels[ax_ix]}_{align_beh}.png', bbox_inches='tight')
    aligned_df = pd.DataFrame.from_dict(aligned_dict,orient='index')
    for ptype in aligned_df.columns:
        for i, sess in enumerate(aligned_df.index):
            if i == 0:
                _array = copy(aligned_df.loc[sess][ptype])
            else:
                _array = np.concatenate([_array, copy(aligned_df.loc[sess][ptype])], axis=0)
        aligned_list.append(_array)
    return aligned_list,aligned_df

# for loading pkl
data_pkl_name = None

# loaded merged trial data
animals = [
            # 'DO27',
            # 'DO28',
            # 'DO29',
            # 'DO37',            # 'DO42',
            # 'DO43',
            # 'ES01',
            # 'ES02',
            'ES03'
]

anon_animals = [f'Animal {i}' for i in range(len(animals))]

datadir = r'C:\bonsai\data'
# dates = ['2/04/2021', '29/10/2021']
dates = ['06/12/2021', '08/12/2021']
plot_colours = ['b','r','c','m','y','g']

today =  datetime.strftime(datetime.now(),'%y%m%d')
figdir = os.path.join(os.getcwd(),'figures',today)
if not os.path.isdir(figdir):
    os.mkdir(figdir)


trial_data = utils.merge_sessions(datadir,animals,'TrialData',dates)
trial_data = pd.concat(trial_data, sort=False, axis=0)
try:
    trial_data = trial_data.drop(columns=['RewardCross_Time','WhiteCross_Time'])
except KeyError:
    print('nothing to drop')

for col in trial_data.keys():
    if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1:
        if col.find('Wait') == -1 and col.find('dt') == -1:
            utils.add_datetimecol(trial_data,col)
# set up pupil data class for mice. value for each key is object with data 1 session


# dates = ['210427','210428','210430','211022','211026','211028']
# dates = ['210513','210514','210519','210521','210526']
# dates = ['211022','211026','211028','211029']
dates = ['211208']
samplerate = round(1/60,3)
data = {}
# start of pipeline
# data_pkl_name = r'pickles\all_withxy.pkl'
plabs = True
if data_pkl_name is None:
    for animal in animals:
        for date in dates:
            name = f'{animal}_{date}'
            try:
                if plabs:
                    animal_pupil = pd.read_csv(f'W:\\mouse_pupillometry\\analysed\\{animal}_{date}_pupildata.csv')
                else:
                    animal_pupil = pd.read_csv(f'W:\\mouse_pupillometry\\analysed\\{animal}_{date}_pupildata_hypcffit.csv', skiprows=2)
                    animal_pupil = animal_pupil.rename(columns={'Unnamed: 25': 'frametime', 'Unnamed: 26': 'diameter','Unnamed: 27':'xcyc'})
                data[name] = pupilDataClass(animal)
                # Load pupil date for animal as pandas dataframe
                animal_pupil['scalar'] = [scalarTime(t) for t in animal_pupil['frametime']]

                data[name].rawPupilDiams = np.array(animal_pupil['diameter'])
                data[name].rawTimes = np.array(animal_pupil['scalar'])

                data[name].uniformSample(samplerate)
                data[name].removeOutliers(n_speed=4, n_size=5)
                data[name].interpolate(gapExtension=0.05)
                data[name].frequencyFilter(lowF=0.1, lowFwidth=0.01, highF=3, highFwidth=0.5,do_highpass=False)

                data[name].zScore()

                try:
                    animal_pupil['xcyc']
                except KeyError:
                    animal_pupil['xcyc'] = np.full_like(animal_pupil['frametime'],'"0, 0"')

                _rawxcyc = pd.DataFrame(animal_pupil['xcyc'].apply(lambda e: e[1:-1].split(',')))
                data[name].rawxcyc = pd.DataFrame(_rawxcyc['xcyc'].tolist(), columns=['x', 'y'])
                # data[name].rawxcyc = pd.DataFrame(data[name].rawxcyc)
                for xycol in data[name].rawxcyc.columns:
                    _uni = \
                    uniformSample(np.array(data[name].rawxcyc[xycol]).astype(float), data[name].rawTimes, samplerate)[0]
                    _xyouts, _xyoutsbool = removeOutliers(_uni, data[name].times, n_speed=4, n_size=5)
                    _inter = interpolateArray(_xyouts, data[name].times, 0.05)[0]
                    if xycol == 'x':
                        data[name].xc = zScore(frequencyFilter(_inter, data[name].times, 0.1, 0.01))
                    else:
                        data[name].yc = zScore(frequencyFilter(_inter, data[name].times, 0.1, 0.01))

                session_TD = trial_data.loc[animal, date]
                for col in session_TD.keys():
                    if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1:
                        if col.find('Wait') == -1 and col.find('dt') == -1:
                            session_TD[f'{col}_scalar'] = [scalarTime(t) for t in session_TD[col]]
                data[name].trialData = session_TD  # add session trialdata
                data[name].plot()
            except FileNotFoundError:
                pass
else:
    with open(data_pkl_name,'rb') as f:
        data = pickle.load(f)
# align 2 tone trial start:
# name = list(data.keys())[0]
duration = [-1,2]

# normal vs viol
anon_sess_labels = ['Animal 1, Session 1',
                    'Animal 1, Session 2',
                    'Animal 1, Session 3',
                    'Animal 2, Session 2',
                    'Animal 2, Session 3',
                    'Animal 3, Session 1',
                    'Animal 3, Session 2',
                    'Animal 3, Session 3',
                    'Animal 4, Session 1',

                    ]

tonealigned_viols, tonealigned_viols_df  = align_wrapper(data,[['d0','4pupil'], ['d!0','4pupil']],'ToneTime_scalar',
                                                         duration,samplerate,alignshifts=[.5,.5])
all_normals = tonealigned_viols[0]
tonealigned_viols_df.columns = ['Normal', 'Deviant']
tonealigned_viols_fig, tonealigned_viols_ax = plot_eventaligned(tonealigned_viols,['Normal', 'Deviant'],
                                                                duration,samplerate, 'Violation')
tonealigned_viols_ax.set_ylim((-.5,.5))
tonealigned_viols_ax.set_ylabel('zcscored pupil size')
tonealigned_viols_ax.axvline(0,ls='--',color='k')
tonealigned_viols_fig.set_size_inches(4,3)
tonealigned_viols_fig.savefig(os.path.join(figdir,'violaligned_normdev.png'),bbox_inches='tight')
# combination attempt
_array = None
n_traces = 10
type_N_traces = []
for i,col in enumerate(tonealigned_viols_df.columns):
    first_traces = []  # list all first traces for this current pattern type
    _type_first = []
    last_traces = []
    _type_last = []
    for session in list(tonealigned_viols_df.index):
        session_traces = tonealigned_viols_df.loc[session][col]
        first_traces.append(session_traces[:n_traces,:])
        _type_first.append([session,col])
        last_traces.append(session_traces[-n_traces:, :])
        _type_last.append([session,col])
    type_N_traces.append(copy([np.concatenate(first_traces,axis=0),np.concatenate(last_traces, axis=0)]))

type0_traces = type_N_traces[0]
type1_traces = type_N_traces[1]

first_last_plot = plt.subplots()
subset_type = ['firsts','lasts']
for ptype, pytype_label in zip([type0_traces,type1_traces],['Normal', 'Deviant']):
    if pytype_label == 'Normal':
        plot_eventaligned([all_normals],
                      [f'All Normals'],duration,samplerate,
                      'Normal vs Deviant aligned to Violation', plotax=first_last_plot)
        # plot_eventaligned([ptype[0], ptype[1]],
        #                   [f'{pytype_label} {subset_type[0]}', f'{pytype_label} {subset_type[1]}'], duration,
        #                   samplerate,
        #                   'Normal vs Deviant aligned to Violation', plotax=first_last_plot)
    else:
        plot_eventaligned([ptype[0],ptype[1]],
                      [f'{pytype_label} {subset_type[0]}', f'{pytype_label} {subset_type[1]}'],duration,samplerate,
                      'Normal vs Deviant aligned to Violation', plotax=first_last_plot)
first_last_plot[1].set_ylim((-.5,.5))
first_last_plot[1].axvline(0,ls='--', color='k')
first_last_plot[1].set_xlabel('Time from violation (s)')
first_last_plot[1].set_ylabel('zscored pupil size')
first_last_plot[1].set_title('First vs Last presentation of deviants (Last 5, 9 sessions)')
first_last_plot[0].set_size_inches(4,3, forward=True)
first_last_plot[0].savefig(os.path.join(figdir,'firstlast_normdev_va.png'),bbox_inches='tight')

dev_traces = {}
dev_traces_list = []

devsubset_TA_viols, df_devTA_traces = align_wrapper(data,[['devord','4pupil','d!0'], ['devrep','4pupil','d!0']],
                                                  'ToneTime_scalar',duration,samplerate,alignshifts=[.5,.5,.75,.5])
df_devTA_traces.columns = ['Bad Order', 'Repeated']
devsubset_fig, devsubset_ax = plot_eventaligned(devsubset_TA_viols,df_devTA_traces.columns,duration,samplerate,
                                                'Violation', plotax=(tonealigned_viols_fig, tonealigned_viols_ax))
devsubset_ax.set_ylim((-.5,.5))
devsubset_ax.axvline(0,ls='--',color='k')
devsubset_ax.set_xlabel('Time from violation (s)')
devsubset_fig.set_size_inches(4,3)

#
# for ptype in df_devTA_traces.columns:
#     for i, session in enumerate(df_devTA_traces.index):
#         if i == 0:
#             _array = copy(df_devTA_traces.loc[session][ptype])
#         else:
#             _array=np.concatenate([_array,copy(df_devTA_traces.loc[session][ptype])],axis=0)
#     dev_traces_list.append(_array)

violaligned_traces,violaligned_df = align_wrapper(data,[['d0','4pupil'], ['d1','4pupil'],['d2','4pupil'],['d3','4pupil']],
                                                  'ToneTime_scalar',duration,samplerate,alignshifts=[.5,.5,.75,.5])
violaligned_df.columns = ['Normal','AB_D','ABC_','AB__']
violaligned_fig,violaligned_ax = plot_eventaligned(violaligned_traces,violaligned_df.columns,duration,samplerate, 'Violation Time by pattern')
violaligned_ax.set_ylim((-.5,.5))
violaligned_ax.set_xlabel('Time from violation (s)')
violaligned_ax.set_ylabel('zscored pupil size')
violaligned_ax.axvline(0,ls='--', color='k')
violaligned_fig.set_size_inches(4,3)
violaligned_fig.savefig(os.path.join(figdir,'norm_dev_byclass_va.png'),bbox_inches='tight')

all_devsubsetplot = plot_eventaligned(dev_traces_list,['Bad Order', 'Repeated'],duration,samplerate,
                                      'Normal vs Deviant aligned to Violation',plotax=[violaligned_fig, violaligned_ax])
all_devsubsetplot[1].set_ylim(-.5,.5)
all_devsubsetplot[0].set_size_inches(4,3)

violaligned_devsubset_traces,violaligned_devsubset_df = align_wrapper(data,[['d0','4pupil'], ['d1','4pupil']],
                                                  'ToneTime_scalar',duration,samplerate,alignshifts=[.5,.5])
violaligned_devsubset_df.columns = ['Normal','AB_D']
violaligned_devsubset_fig,violaligned_devsubset_ax = plot_eventaligned(violaligned_devsubset_traces,  # [0],violaligned_devsubset_traces[0]+violaligned_devsubset_traces[1]]
                                                                             violaligned_devsubset_df.columns,
                                                                             duration,samplerate, 'Violation Time by pattern')
violaligned_devsubset_ax.set_ylim((-.5,.5))
violaligned_ax.set_xlabel('Time from violation (s)')
violaligned_devsubset_ax.set_ylabel('zscored pupil size')
violaligned_devsubset_ax.axvline(0,ls='--', color='k')
violaligned_devsubset_fig.set_size_inches(4,3)
violaligned_devsubset_fig.savefig(os.path.join(figdir,'norm_dev1_va.png'),bbox_inches='tight')

rewardtone_traces, rewardtone_df = align_wrapper(data,[['a0'], ['a1']],'Gap_Time_scalar',duration,samplerate)
rewardtone_fig,rewardtone_ax = plot_eventaligned(rewardtone_traces,rewardtone_df.columns,duration,samplerate, '"X"')
rewardtone_ax.axvline(0,ls='--',color='k')
rewardtone_fig.set_size_inches(4,3)

# look at descending
dev_traj_traces, dev_traj_df = align_wrapper(data,[['d0','4pupil'],['d!0','devassc','4pupil'], ['d!0','devdesc','4pupil']]
                                             ,'ToneTime_scalar',duration, samplerate,alignshifts=[.5,.5,.5])
dev_traj_df.columns = ['Normal','Deviant Assc','Deviant Dessc']
dev_traj_fig,dev_traj_ax = plot_eventaligned(dev_traj_traces,dev_traj_df.columns,duration,samplerate,
                                             'Ascending and Descending Deviant patterns')
dev_traj_ax.axvline(0,ls='--',color='k')
devsubset_ax.set_xlabel('Time from violation (s)')
dev_traj_ax.set_ylim(-.5,.5)
dev_traj_ax.set_ylabel('zscored pupil size')
dev_traj_fig.set_size_inches(4,3)
dev_traj_fig.savefig(os.path.join(figdir,'assc_desc.png'),bbox_inches='tight')

# xy position of the eye norm dev
for i in ['x','y']:
    tonealigned_viols_xy, tonealigned_viols_xy_df  = align_wrapper(data,[['d0','4pupil'], ['d!0','4pupil']],'ToneTime_scalar',
                                                             duration,samplerate,alignshifts=[.5,.5],coord=i)
    tonealigned_viols_xy_df.columns = ['Normal', 'Deviant']
    tonealigned_viols_xy_fig, tonealigned_viols_xy_ax = plot_eventaligned(tonealigned_viols_xy,['Normal', 'Deviant'],
                                                                    duration,samplerate, 'Violation')
    tonealigned_viols_xy_ax.set_ylim((-.5,.5))
    tonealigned_viols_xy_ax.set_ylabel(f'{i} position')
    tonealigned_viols_xy_ax.axvline(0,ls='--',color='k')
    tonealigned_viols_xy_fig.set_size_inches(4,3)