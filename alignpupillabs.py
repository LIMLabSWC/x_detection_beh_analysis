import pandas as pd
from datetime import datetime
from alignframes import findfiles
from os.path import join
from functools import lru_cache
from copy import copy


def correctdrift(timeseries, syncseries1, syncseries2) -> pd.Series:
    """

    :param timeseries:
    :param syncseries1:
    :param syncseries2:
    :return:
    """

    timeseries0 = timeseries.iloc[0]
    syncdiff = syncseries1 - syncseries2  # series with diff between clocks
    drift_scalar = (syncdiff.iloc[-1] - syncdiff.iloc[0])/(syncseries1.iloc[-1]-syncseries1.iloc[0])
    print(f' first {syncdiff.iloc[0]} last{syncdiff.iloc[-1]}, scalar {drift_scalar}')
    corrected_ts = timeseries.apply(lambda t: (t-timeseries0)*drift_scalar+t)

    return corrected_ts


class Main:
    def __init__(self,animals,dates):
        """

        :param animals:
        :param dates:
        """
        self.animals = animals
        self.dates = dates
        self.toprocess_dict = dict()

    @lru_cache()
    def findfiles(self):
        for animal in self.animals:
            self.toprocess_dict[animal] = dict()
        findfiles(r'W:\mouse_pupillometry\timeSync_csvs', 'timesync', self.toprocess_dict, self.animals, self.dates)
        findfiles(r'W:\mouse_pupillometry\analysed', 'extracted_pupils', self.toprocess_dict, self.animals, self.dates)

    @lru_cache()
    def align(self, times_path, pupils_path) -> pd.DataFrame:
        """

        :param times_path:
        :param pupils_path:
        :return:
        """

        timescsv = pd.read_csv(times_path, header=None)
        pupilcsv = pd.read_csv(pupils_path)

        # split pupilcsv into 2d and 3d
        list_df = [pupilcsv[pupilcsv['topic'] == topic] for topic in sorted(pupilcsv['topic'].unique())]
        pupilcsv_2d, pupilcsv_3d = list_df[0], list_df[1]

        # format times to timestamps
        timescsv.columns = ['pctime', 'pupiltime', 'bonsaitime']
        timescsv['pctime'] = timescsv['pctime'].apply(lambda e: datetime.strptime(f'010101 {e}',
                                                                                  '%d%m%y %H:%M:%S.%f').timestamp())
        timescsv['bonsaitime'] = timescsv['bonsaitime'].apply(lambda e: datetime.strptime(f'010101 {e[:-1]}',
                                                                                          '%d%m%y %H:%M:%S.%f').timestamp())

        # correct for any drift in sync
        driftcorrected = correctdrift(pupilcsv_3d['timestamp'], timescsv['pupiltime'], timescsv['bonsaitime'])
        # align to bonsaitime
        syncoffest = timescsv['pupiltime'].iloc[0] - timescsv['bonsaitime'].iloc[0]
        alignedtime_secs = driftcorrected - syncoffest

        pupilcsv_3d['frametime'] = alignedtime_secs.apply(lambda t: datetime.fromtimestamp(t).time())
        pupilcsv_3d['diameter'] = pupilcsv_3d['diameter_2d [px]']
        pupilcsv_3d['diameter 2d'] = pupilcsv_2d['diameter_2d [px]']
        return pupilcsv_3d

    @lru_cache
    def main_loop(self):
        self.findfiles()
        for animal in self.toprocess_dict:
            for date in self.toprocess_dict[animal]:
                print(self.toprocess_dict[animal][date].keys())
                if 'extracted_pupilsfile' in self.toprocess_dict[animal][date].keys():
                    print(self.toprocess_dict[animal][date]['timesyncfile'])
                    aligned = self.align(self.toprocess_dict[animal][date]['timesyncfile'],
                                         self.toprocess_dict[animal][date]['extracted_pupilsfile'])
                    savename = f'{animal}_{date}_pupildata.csv'
                    aligned.to_csv(join(r'W:\mouse_pupillometry\analysed',savename),index=False)


if __name__ == '__main__':
    run = Main(['ES01','ES02','ES03'],['211208'])
    run.main_loop()
