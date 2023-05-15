import pandas as pd
from datetime import datetime, date, timedelta
from analysis_utils import merge_sessions, find_good_sessions, add_datetimecol, format_timestr
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

    syncdiff = syncseries1 - (syncseries2 - (syncseries2 - syncseries1)[
        0])  # series with diff between clocks, get total drfit between clocks
    # +ve means bonsai drifting away from plabs
    drift_scalar = (syncdiff.iloc[-1] - syncdiff.iloc[0]) / (
                syncseries1.iloc[-1] - syncseries1.iloc[0])  # total drift/ time elasped plabs
    if math.isnan(drift_scalar) or abs(drift_scalar) > 0.1:
        if drift_scalar < 0:
            drift_scalar = -0.000125
        else:
            drift_scalar = 0.000125
    print(f' first {syncdiff.iloc[0]} last{syncdiff.iloc[-1]}, scalar {drift_scalar}')
    # correct drift dt*scale factor3.3
    corrected_ts = timeseries.apply(lambda t: t + ((t - timeseries0) * (-drift_scalar)))
    # corrected_ts = timeseries
    print(f'original endtime: {timeseries.iloc[-1]},correct endtime: {corrected_ts.iloc[-1]}')
    return corrected_ts


class Main:
    def __init__(self, animals, dates, datadir, timesyncdir,aligned_dir,overwrite,merge):
        """

        :param animals:
        :param dates:
        """
        self.animals = animals
        self.dates = dates
        self.datadir = datadir
        self.timesyncdir = timesyncdir
        self.toprocess_dict = dict()
        self.aligned_dir = aligned_dir
        self.overwrite = overwrite
        self.merge = merge

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
        pupilcsv = pupilcsv[pupilcsv.eye_id==0]
        # split pupilcsv into 2d and 3d
        list_df = [pupilcsv[pupilcsv['topic'] == topic] for topic in sorted(pupilcsv['topic'].unique())]
        pupilcsv_2d, pupilcsv_3d = list_df[0].copy(deep=True), list_df[1].copy(deep=True)
        pupilcsv_2d.index = np.arange(len(pupilcsv_2d.index))
        pupilcsv_3d.index = np.arange(len(pupilcsv_3d.index))

        # format times to timestamps
        arb_date = date.fromisoformat('2020-01-01')
        timescsv.columns = ['pctime', 'pupiltime', 'bonsaitime']
        timescsv['pctime_dt'] = format_timestr(timescsv['pctime'])
        timescsv['pctime_dt'] = [e.replace(year=arb_date.year) for e in timescsv['pctime_dt']]
        timescsv['pctime'] = [e.timestamp() for i,e in timescsv['pctime_dt'].iteritems()]
        timescsv['bonsaitime_dt'] = format_timestr(timescsv['bonsaitime'])
        timescsv['bonsaitime_dt'] = [e.replace(year=arb_date.year) for e in timescsv['bonsaitime_dt']]
        timescsv['bonsaitime'] = [e.timestamp() for i,e in timescsv['bonsaitime_dt'].iteritems()]

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
                    output_fullpath = f"{join(self.datadir,self.aligned_dir,savename.replace('.csv','_3d.csv'))}"
                    output_fullpath = join(f"{output_fullpath.split('.')[0]}a.csv")
                    if not os.path.isdir(os.path.split(output_fullpath)[0]):
                        os.mkdir(os.path.split(output_fullpath)[0])
                    if os.path.exists(output_fullpath) and not self.overwrite:
                        print('not overwriting')
                        continue
                    if os.path.exists(output_fullpath) and self.merge:  # check
                        old_df3d = pd.read_csv(output_fullpath)
                        old_df2d = pd.read_csv(output_fullpath.replace('_3d','_2d'))
                        print('path exists will merging')
                    else:
                        old_df3d = None
                        old_df2d = None
                    try:
                        aligned = self.align(self.toprocess_dict[animal][date]['timesyncfile'],
                                self.toprocess_dict[animal][date]['extracted_pupilsfile'])
                        aligned2dsave = pd.concat([old_df2d,aligned[0]])
                        aligned2dsave.to_csv(output_fullpath.replace('_3d','_2d') ,index=False)
                        aligned3dsave = pd.concat([old_df3d,aligned[1]])
                        aligned3dsave.to_csv(join(output_fullpath),index=False)
                    except KeyError:
                            print('keyerror')
                else:
                    print('missing extracted pupil file')


if __name__ == '__main__':
    subject_type = 'humans'

    if subject_type in ['mice', 'rats']:
        animals = ['DO45','DO46','DO47','DO48','ES01','ES02','ES03','DO50','DO51','DO53']
        dates = ['01/10/2022', '31/12/2022']
        tdatadir = r'C:\bonsai\data\Dammy'
        protocol_dirname = r'W:\mouse_pupillometry\mouseprobreward'
        protocol_aligneddir = f'aligned_{os.path.split(protocol_dirname)[-1]}'

        td_df = pd.concat(merge_sessions(tdatadir,animals,'TrialData',dates),sort=False,axis=0,)
        for col in td_df.keys():
            if col.find('Time') != -1 or col.find('Start') != -1 or col.find('End') != -1:
                if col.find('Wait') == -1 and col.find('dt') == -1 and col.find('Harp') == -1 and col.find('Bonsai') == -1:
                    # print(col)
                    try:add_datetimecol(td_df,col)
                    except AttributeError: print(col)
        valid_sessions = find_good_sessions(td_df,4,50,skip=1)
        run = Main(valid_sessions[1],valid_sessions[2],protocol_dirname,
                   tdatadir,protocol_aligneddir,overwrite=0,merge=0)
        run.main_loop()

    else:
        task = 'normdev'
        # human fam task
        if task == 'fam':
            humans = [f'Human{i}' for i in range(20,28)]
            humandates = [#'220208','220209','220210','220215',
                          '220311','220316','220405','220407','220407','220408','220422','220425']
        # humans dev norm task
        elif task == 'normdev':
            humans = [f'Human{i}' for i in range(28, 33)]
            humandates = ['220518', '220523', '220524', '220530', '220627']

        else:
            print('invalid task name given')
            exit()
        run = Main(humans,humandates,r'W:\humanpsychophysics\HumanXDetection\Data',r'C:\bonsai\data\Hilde\Human\timeSyncs',
                   f'aligned_{task}',overwrite=True,merge=0)
        run.main_loop()
