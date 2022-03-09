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

# ids = [4,5,7,9,10,12,14,15]
ids = [17,18]
humans = [f'Human{i}' for i in ids]
humandates = ['220118','220119','220120','220121','220124','220125','220126','220127','220128','220208','220209','220210']
animals = humans

datadir = r'C:\bonsai\data\Hilde'
dates = ['08/02/2022', '10/02/2022']

today = datetime.strftime(datetime.now(),'%y%m%d')
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
samplerate = round(1/100,3)

# start of pipeline
data_pkl_name = r'pickles\human_lessstims_nofilt.pkl'

plabs = True
data = {}
if data_pkl_name is None or os.path.exists(data_pkl_name) is False:
    for animal in animals:
        for date in dates:
            name = f'{animal}_{date}'
            # if os.path.exists(f'W :\\humanpsychophysics\\HumanXDetection\\Data\\analysed\\{animal}_{date}_pupildata.csv'):
            try:
                if plabs:
                    plabs_dir = r'W:\humanpsychophysics\HumanXDetection\Data\analysed'
                    animal_pupil = pd.read_csv(os.path.join(plabs_dir,f'{animal}_{date}_pupildata.csv'))
                else:
                    animal_pupil = pd.read_csv(f'W:\\mouse_pupillometry\\analysed\\{animal}_{date}_pupildata_hypcffit.csv', skiprows=2)
                    animal_pupil = animal_pupil.rename(columns={'Unnamed: 25': 'frametime', 'Unnamed: 26': 'diameter','Unnamed: 27':'xcyc'})
                data[name] = pupilDataClass(animal)
                # Load pupil date for animal as pandas dataframe
                animal_pupil['scalar'] = [scalarTime(t) for t in animal_pupil['frametime']]

                data[name].rawPupilDiams = np.array(animal_pupil['diameter'])
                data[name].rawTimes = np.array(animal_pupil['scalar'])

                data[name].uniformSample(samplerate)
                data[name].removeOutliers(n_speed=4, n_size=4)
                data[name].interpolate(gapExtension=0.1)
                # data[name].frequencyFilter(lowF=0.1, lowFwidth=0.01, highF=3, highFwidth=0.5,do_highpass=False)

                data[name].zScore()
                try:
                    animal_pupil['xcyc']
                    interpol_xy = True
                except KeyError:
                    animal_pupil['xcyc'] = np.full_like(animal_pupil['frametime'],'"0, 0"')
                    interpol_xy = False

                if interpol_xy:
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
                else:
                    data[name].xc = np.full_like(animal_pupil['frametime'],0)
                    data[name].yc = np.full_like(animal_pupil['frametime'],0)

                session_TD = trial_data.loc[animal, date]
                for col in session_TD.keys():
                    if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1:
                        if col.find('Wait') == -1 and col.find('dt') == -1:
                            session_TD[f'{col}_scalar'] = [scalarTime(t) for t in session_TD[col]]
                data[name].trialData = session_TD  # add session trialdata
                data[name].plot()
            except FileNotFoundError:
                pass
        if data_pkl_name is not None:
            with open(data_pkl_name,'wb') as pklfile:
                pickle.dump(data,pklfile)
else:
    with open(data_pkl_name,'rb') as f:
        data = pickle.load(f)

duration = [-1,2]

# normal vs viol

tonealigned_viols, tonealigned_viols_df = align_wrapper(data,[['d0'], ['d!0']],'RewardTone_Time_scalar',
                                                        duration,samplerate,alignshifts=[0,0],
                                                        plotlabels=['Normal', 'Deviant'], plottitle='Violation',
                                                        xlabel='Time from violation', animal_labels=animals,
                                                        plotsess=True)
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

devsubset_TA_viols, df_devTA_traces = align_wrapper(data,[['devord','d!0'], ['devrep','d!0']],
                                                  'ToneTime_scalar',duration,samplerate,alignshifts=[.5,.5,.75,.5])
df_devTA_traces.columns = ['Bad Order', 'Repeated']
devsubset_fig, devsubset_ax = plot_eventaligned(devsubset_TA_viols,df_devTA_traces.columns,duration,samplerate,
                                                'Violation', plotax=(tonealigned_viols_fig, tonealigned_viols_ax))
devsubset_ax.set_ylim((-.5,.5))
devsubset_ax.axvline(0,ls='--',color='k')
devsubset_ax.set_xlabel('Time from violation (s)')
devsubset_fig.set_size_inches(4,3)


violaligned_traces,violaligned_df = align_wrapper(data,[['d0'], ['d1'],['d2'],['d3']],
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

violaligned_devsubset_traces,violaligned_devsubset_df = align_wrapper(data,[['d0'], ['d1']],
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
dev_traj_traces, dev_traj_df = align_wrapper(data,[['d0'],['d!0','devassc'], ['d!0','devdesc']]
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
# for i in ['x','y']:
#     tonealigned_viols_xy, tonealigned_viols_xy_df  = align_wrapper(data,[['d0''], ['d!0'']],'ToneTime_scalar',
#                                                              duration,samplerate,alignshifts=[.5,.5],coord=i)
#     tonealigned_viols_xy_df.columns = ['Normal', 'Deviant']
#     tonealigned_viols_xy_fig, tonealigned_viols_xy_ax = plot_eventaligned(tonealigned_viols_xy,['Normal', 'Deviant'],
#                                                                     duration,samplerate, 'Violation')
#     tonealigned_viols_xy_ax.set_ylim((-.5,.5))
#     tonealigned_viols_xy_ax.set_ylabel(f'{i} position')
#     tonealigned_viols_xy_ax.axvline(0,ls='--',color='k')
#     tonealigned_viols_xy_fig.set_size_inches(4,3)

# compare normals
normcomp_traces, normcomp_df = align_wrapper(data,[['d0'],['normtrain'],['normtest'],['d2']]
                                             ,'ToneTime_scalar',duration, samplerate,alignshifts=[.5,.5,.5,.5])
normcomp_df.columns = ['All Normal','Normal: Train Phase','Normal: Test Phase','Deviants']
normcomp_fig, normcomp_ax = plot_eventaligned(normcomp_traces,normcomp_df.columns, duration,samplerate,
                                              'Normal Patterns comparison')
normcomp_ax.set_ylim(-.5,.5)
normcomp_ax.set_ylabel('zscored pupil size')
normcomp_ax.axvline(0,ls='--',color='k')
normcomp_fig.set_size_inches(4,3)
normcomp_fig.savefig(os.path.join(figdir,'normcomp.png'),bbox_inches='tight')

# Hilde stuff:
# mean_normals = tonealigned_viols[0].mean(axis=0)  # mean of normal trace
mean_normals = tonealigned_viols[0][-1]
max_diffs_list = []
eval_window = np.array([0,1]) + (-duration[0])  # in seconds
eval_window_ts = (eval_window/samplerate).astype(int)  # as index number

for r,trace in enumerate(tonealigned_viols[0]):
    trial_dev_trace = tonealigned_viols[0][r,:]
    diff_trace = (mean_normals[eval_window_ts[0]:eval_window_ts[1]]-trial_dev_trace[eval_window_ts[0]:eval_window_ts[1]]) # get diff using a metric
    max_diffs_list.append([diff_trace.max(),diff_trace.sum(),np.where(diff_trace==diff_trace.max())[0][0]*samplerate])
max_diffs_arr = np.array(max_diffs_list)
# plt.plot(max_diffs_arr[:,0])
plt.plot(max_diffs_arr[:,1])


