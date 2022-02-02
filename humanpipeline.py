from psychophysicsUtils import *
import analysis_utils as utils
from analysis_utils import align_wrapper, plot_eventaligned
from datetime import datetime, time, timedelta
from matplotlib import pyplot as plt
from copy import deepcopy as copy
import numpy as np
from sklearn.linear_model import LinearRegression


# for loading pkl
data_pkl_name = None
humans = [f'Human{i}' for i in range(2,16)]
humandates = ['220118','220119','220120','220121','220124','220124','220125','220126','220127','220128']
animals = humans

datadir = r'C:\bonsai\data\Hilde'
dates = ['18/01/2022', '28/01/2022']

today =  datetime.strftime(datetime.now(),'%y%m%d')
figdir = os.path.join(os.getcwd(),'figures',today)
if not os.path.isdir(figdir):
    os.mkdir(figdir)

# loaded merged trial data
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
dates = humandates
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
                    animal_pupil = pd.read_csv(f'W:\\humanpsychophysics\\HumanXDetection\\Data\\analysed\\{animal}_{date}_pupildata.csv')
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

duration = [-1,2]

# normal vs viol

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