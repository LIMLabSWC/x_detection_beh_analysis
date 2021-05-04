from psychophysicsUtils import *

animals = [
            # 'DO27',
            'DO28',
            # 'DO29',
]

dates = ['210430']

data = {}

for animal in animals:
    for date in dates:
        animal_TD = pd.read_csv(f'C:\\bonsai\\data\\Dammy\\{animal}\\TrialData\\{animal}_TrialData_{date}a.csv')
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
        samplerate = round(1/30.0,2)
        pupil_uniformsampled = pd.DataFrame(np.array(animal_pupil['diameter']),
                                           index=animal_pupil['frametime_dt']).resample(f'{str(samplerate)}S').mean()

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

        size_speed = size_speed[threshold_size_low<size_speed[0]]
        size_speed = size_speed[threshold_size_high >size_speed[0]]
        size_speed = size_speed[threshold_speed_low<size_speed[1]]
        size_speed = size_speed[threshold_speed_high >size_speed[1]]

        pupil_size_downsampled = size_speed.resample('0.1S').mean()[0]

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
#     plt.close(hist_fig)

