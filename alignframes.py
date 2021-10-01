import pandas as pd
import numpy as np
import os 
from datetime import datetime, timedelta
from ellipse import LsqEllipse
from matplotlib import pyplot as plt
import circle_fit as cf


def matchvid2data(animal,date,starttime,viddir):
    
    try:
        timestampfile = f'{animal}_{date}_timestamp.dat'
        timestamp_df = pd.read_csv(os.path.join(viddir, timestampfile), delimiter='\t')
        frametime = [(starttime+timedelta(milliseconds=framereltime)).time() for framereltime in timestamp_df['sysClock']]

    except FileNotFoundError:
        timestampfile = f'{animal}_{date}_timestamp.csv'
        timestamp_df = pd.read_csv(os.path.join(viddir, timestampfile))
        frametime = [(starttime+timedelta(milliseconds=framereltime)).time() for framereltime in timestamp_df['Time Stamp (ms)']]

    timestamp_df['frametime'] = frametime

    return timestamp_df


def alignframes():
    pass

def fit_elipse(point_array):

    # # lsq ellipse method
    # lsqe = LsqEllipse()
    # lsqe.fit(point_array)
    # center, width, height, phi = lsqe.as_parameters()
    # return center, width, height, phi

    # circle fit method
    # xc, yc, r, _ = cf.least_squares_circle(point_array)
    xc, yc, r, _ = cf.hyper_fit(point_array)

    return (xc,yc), r, r


toprocess_dict = dict()

animals = [
            # 'DO27',
            'DO39',
            # 'DO29',
]

for animal in animals:
    toprocess_dict[animal] = dict()

# def find_h5files(h5dir, names):


def findfiles(startdir,filetype,datadict,animals=None,dates=None):
    for root, folder, files in os.walk(startdir):
        for file in files:
            if file.find(filetype) != -1:
                splitstr = file.split('_')
                _animal = splitstr[0]
                _date = splitstr[1]
                if dates is None:
                    if _date not in datadict[_animal].keys():
                        datadict[_animal][_date] = dict()
                    datadict[_animal][_date][f'{filetype}file'] = os.path.join(root,file)
                elif dates is not None and animals is not None:
                    if _date in dates and _animal in animals:
                        if _date not in datadict[_animal].keys():
                            datadict[_animal][_date] = dict()
                        datadict[_animal][_date][f'{filetype}file'] = os.path.join(root, file)


# find h5 
findfiles(r'W:\mouse_pupillometry\analysed','h5',toprocess_dict,animals, ['210810'])
# find camstarts
findfiles(r'C:\bonsai\data\Dammy','camstart',toprocess_dict,animals, ['210810'])

list_h5pupils = []
timestampdir = r'W:\mouse_pupillometry\cropsessionvids'
for animal in toprocess_dict.keys():
    for date in toprocess_dict[animal].keys():
        if 'h5file' in toprocess_dict[animal][date].keys():
            camstart = pd.read_csv(toprocess_dict[animal][date]['camstartfile'],header=None)[0][0]
            h5pupildf = pd.read_hdf(toprocess_dict[animal][date]['h5file'])
            camstart_dt = datetime.strptime(camstart[:-1], '%H:%M:%S.%f')
            frame_timestamps = matchvid2data(animal,date,camstart_dt,timestampdir)

            h5pupildf['frametime'] = frame_timestamps['frametime']
            print('videofile read')

            toprocess_dict[animal][date]['pupildf'] = h5pupildf

scorer = 'DLC_resnet50_pupildiamterApr28shuffle1_900000'
for animal in toprocess_dict.keys():
    for date in toprocess_dict[animal].keys():
        if 'pupildf' in toprocess_dict[animal][date].keys():
            df = toprocess_dict[animal][date]['pupildf']
            bodypoints = np.array(df)
            diams_ = []
            for row in bodypoints:
                reshaped = row[0:24].reshape([8,3])
                goodpoints = reshaped[reshaped[:,2]>.5].astype(float)
                if goodpoints.shape[0] < 3:
                    diams_.append(0.0)
                else:
                    frame_elipse = fit_elipse(goodpoints[:,[0,1]])
                    diams_.append((frame_elipse[1]+frame_elipse[2])/2)
            eyeEW_arr = np.array(df[scorer,'eyeW']-df[scorer,'eyeE'])
            diams = np.linalg.norm(eyeEW_arr,axis=1)
            # toprocess_dict[animal][date]['pupildf']['diameter'] = diams
            toprocess_dict[animal][date]['pupildf']['diameter'] = diams_
            df2save = toprocess_dict[animal][date]['pupildf']
            h5filename = f'{animal}_{date}_pupildata.h5'
            # df2save.to_hdf(os.path.join(r'W:\mouse_pupillometry\analysed',h5filename))
            df2save.to_csv(os.path.join(r'W:\mouse_pupillometry\analysed',h5filename.replace('.h5','_hypcffit.csv')))
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
# colors = cm.rainbow(np.linspace(0, 1, len(bodyparts)))
# for i, eyepart in enumerate(bodyparts):
#     plt.hist(df[scorer,eyepart]['likelihood'],alpha=.25,label=eyepart,edgecolor='None',color=colors[i])
# for i, eyepart in enumerate(bodyparts):
#     plt.hist(df[scorer,eyepart]['likelihood'],alpha=.25,label=eyepart,ls='dashed', lw=3, facecolor="None")
# plt.legend()
