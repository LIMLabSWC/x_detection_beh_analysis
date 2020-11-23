from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime


def subset_dates(df):
    # unique dates for each animal
    unique_dates = np.unique(df.index.values)
    unique_sessions = []
    for ind in unique_dates:
        unique_sessions.append(df.loc[ind])

    return unique_sessions


def plt_sess_features(list_df, features):
    """
    function plot feature across one session e.g. cum ave over one session
    df with multiple sessions will be plot on same axis
    :param list_df: list of dataframes where each len(df.index) == 1
    :param features: list of df col_names to plot
    :return: nothing. will save new image for each figure
    """

    # x axis will be trial number
    for feature in features:
        figure, axis = plt.subplots()
        axis.set_xlabel('Trial Number')
        axis.set_ylabel(feature.replace('_', ' '))
        for df in list_df:  # df = session when list unique sessions given
            axis.plot(np.arange(1,df.shape[0]+1),df[feature], label=f'{np.unique(df.index.values)[0][0]} '
                                                                  f'{np.unique(df.index.values)[0][1]}')
        axis.legend()
        figure.savefig(f'{feature}_{datetime.now().strftime("%y%m%d")}.png')


def plot_featuresvsdate(list_df,features,animals):
    """

    :param list_df:
    :param features: list of feature and mean type operation e.g [Ntrials,total]
    :return:
    """
    dates = []
    for df in list_df:
        print(np.unique(df.index.values))
        dates.append(np.unique(df.index.values)[0][1])
    unique_dates = np.unique(dates)

    # dates_axis = [datetime.strptime(date,'%y%m%d') for date in unique_dates]

    for feature in features:
        figure, axis = plt.subplots()
        axis.set_xlabel('Date')
        axis.set_ylabel(feature[0].replace('_', ' '))

        plot_dict = {}
        for animal in animals:
            plot_dict[animal] = []
        if feature[1] == 'mean':
            series_toplot = [[df.index.values[0][0], df.index.values[0][1], df[feature[0]].mean()] for df in list_df]
        elif feature[1] == 'total':
            series_toplot = [[df.index.values[0][0], df.index.values[0][1], df[feature[0]].shape[0]] for df in list_df]
        elif feature[1] == 'max':
            series_toplot = [[df.index.values[0][0], df.index.values[0][1], df[feature[0]].max()] for df in list_df]
        elif feature[1] == 'min':
            series_toplot = [[df.index.values[0][0], df.index.values[0][1], df[feature[0]].min()] for df in list_df]
        elif feature[1] == 'sum':
            series_toplot = [[df.index.values[0][0], df.index.values[0][1], df[feature[0]].sum()] for df in list_df]
        else:
            print(f'Operation {feature[1]} not valid')
            return None
        for val in series_toplot:
            plot_dict[val[0]].append([val[1],val[2]])

        for animal in list(plot_dict.keys()):
            plot_array = np.array(plot_dict[animal])
            print(plot_array)
            axis.plot(datetime.strptime(plot_array[0][0],'%y%m%d'),int(plot_array[0][1]))


def plot_featurevsfeature(list_df,features):
    pass
