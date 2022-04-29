import pandas as pd
from datetime import datetime
# from analysis_utils import findfiles
import os
from os.path import join
from functools import lru_cache
from copy import copy
import math
import numpy as np


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



def correctdrift(timeseries, syncseries1, syncseries2) -> pd.Series:
    """

    :param timeseries:  timeseries to be corrected
    :param syncseries1:  handshake pupiltime
    :param syncseries2:  handshake bonsaitime
    :return:
    """

    timeseries0 = timeseries.iloc[0]

    syncdiff = syncseries1 - syncseries2  # series with diff between clocks, get total drfit between clocks
    # +ve means bonsai drifting away from plabs
    drift_scalar = (syncdiff.iloc[-1] - syncdiff.iloc[0])/(syncseries1.iloc[-1]-syncseries1.iloc[0])  # total drift/ time elasped plabs
    if math.isnan(drift_scalar):
        drift_scalar = 1
    print(f' first {syncdiff.iloc[0]} last{syncdiff.iloc[-1]}, scalar {drift_scalar}')
    # correct drift dt*scale factor3.3
    corrected_ts = timeseries.apply(lambda t: timeseries0+((t-timeseries0)/(1+drift_scalar)))
    print(f'original endtime: {timeseries.iloc[-1]},correct endtime: {corrected_ts.iloc[-1]}')
    # corrected_ts = timeseries
    return corrected_ts


class Main:
    def __init__(self, animals, dates, datadir, timesyncdir):
        """

        :param animals:
        :param dates:
        """
        self.animals = animals
        self.dates = dates
        self.datadir = datadir
        self.timesyncdir = timesyncdir
        self.toprocess_dict = dict()

    @lru_cache()
    def findfiles(self):
        for animal in self.animals:
            self.toprocess_dict[animal] = dict()
        findfiles(self.timesyncdir, 'timesync', self.toprocess_dict, self.animals, self.dates)
        findfiles(self.datadir, 'extracted_pupils', self.toprocess_dict, self.animals, self.dates)

    @lru_cache()
    def align(self, times_path, pupils_path) -> tuple:
        """

        :param times_path:
        :param pupils_path:
        :return:
        """
        timescsv = pd.read_csv(times_path, header=None)
        pupilcsv = pd.read_csv(pupils_path)

        # split pupilcsv into 2d and 3d
        list_df = [pupilcsv[pupilcsv['topic'] == topic] for topic in sorted(pupilcsv['topic'].unique())]
        pupilcsv_2d, pupilcsv_3d = list_df[0].copy(deep=True), list_df[1].copy(deep=True)
        pupilcsv_2d.index = np.arange(len(pupilcsv_2d.index))
        pupilcsv_3d.index = np.arange(len(pupilcsv_3d.index))

        # format times to timestamps
        timescsv.columns = ['pctime', 'pupiltime', 'bonsaitime']
        timescsv['pctime'] = timescsv['pctime'].apply(lambda e: datetime.strptime(f'010101 {e}',
                                                                                '%d%m%y %H:%M:%S.%f').timestamp())
        timescsv['bonsaitime'] = timescsv['bonsaitime'].apply(lambda e: datetime.strptime(f'010101 {e[:-1]}',
                                                                                        '%d%m%y %H:%M:%S.%f').timestamp())

        # correct for any drift in sync
        driftcorrected = correctdrift(pupilcsv_3d['timestamp'], timescsv['pupiltime'], timescsv['bonsaitime'])
        # driftcorrected = pupilcsv_3d['timestamp']
        # align to bonsaitime
        syncoffest = timescsv['pupiltime'].iloc[0] - timescsv['bonsaitime'].iloc[0]
        alignedtime_secs = driftcorrected - syncoffest

        pupilcsv_3d['frametime'] = alignedtime_secs.apply(lambda t: datetime.fromtimestamp(t).time())
        pupilcsv_2d['frametime'] = alignedtime_secs.apply(lambda t: datetime.fromtimestamp(t).time())
        pupilcsv_3d.columns = ['eye_id', 'timestamp', 'topic', 'confidence', 'diameter_2d',
                               'diameter_3d', '2d_radii', '2d_centre', 'frametime']
        pupilcsv_2d.columns = ['eye_id', 'timestamp', 'topic', 'confidence', 'diameter_2d',
                               'diameter_3d', '2d_radii', '2d_centre', 'frametime']
        return pupilcsv_2d, pupilcsv_3d


    @lru_cache()
    def main_loop(self):
        self.findfiles()
        for animal in self.toprocess_dict:
            for date in self.toprocess_dict[animal]:
                if 'extracted_pupilsfile' in self.toprocess_dict[animal][date].keys():
                    savename = f'{animal}_{date}_pupildata.csv'
                    if os.path.exists(join(self.datadir,'aligned3',savename.replace('.csv','_3d.csv'))):
                        print('path exists not overwriting')
                    else:
                        aligned = self.align(self.toprocess_dict[animal][date]['timesyncfile'],
                                    self.toprocess_dict[animal][date]['extracted_pupilsfile'])
                        aligned[0].to_csv(join(self.datadir,'aligned3',savename.replace('.csv','_2d.csv')),index=False)
                        aligned[1].to_csv(join(self.datadir,'aligned3',savename.replace('.csv','_3d.csv')),index=False)


if __name__ == '__main__':
    humans = [f'Human{i}' for i in range(16,28)]
    humandates = ['220208','220209','220210','220215',
                  '220311','220316','220405','220407','220407','220408','220422','220425']
    run = Main(humans,humandates,r'W:\humanpsychophysics\HumanXDetection\Data',r'C:\bonsai\data\Hilde\Human\timeSyncs',)
    run.main_loop()
