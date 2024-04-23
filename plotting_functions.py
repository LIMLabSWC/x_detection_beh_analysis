from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from analysis_utils import mean_confidence_interval, plotvar, unique_file_path
from datetime import datetime
from copy import deepcopy as copy


def scatter_on_frame(ax, frame, bbox, x_ser, y_ser, label, pointstyle, ):
    x_ser,y_ser = np.array(x_ser), np.array(y_ser)
    pnt_c, pnt_s, pnt_m = pointstyle
    ax.imshow(frame, cmap='gray')

    ax.scatter(x_ser + bbox[1], y_ser + bbox[0],
               c=pnt_c, s=pnt_s, marker=pnt_m, label=label)


def plot_bbox(ax, bbox, line_c='g'):
    ax.axhline(bbox[0], c=line_c)
    ax.axhline(bbox[2], c=line_c)
    ax.axvline(bbox[1], c=line_c)
    ax.axvline(bbox[3], c=line_c)


def draw_ellipse_fit(ax, ell_params, plot_label, pointstyle):
    u, v, a, b, = ell_params  # 75,150,30,10
    pnt_c, pnt_s, pnt_m = pointstyle
    t = np.linspace(0, 2 * np.pi, 100)  # Angle values
    Ell = np.array([b * np.cos(t), a * np.sin(t)])
    ax.plot(u + Ell[0, :], v + Ell[1, :],
            c=pnt_c, lw=pnt_s, ls='--', label=plot_label)

    ax.scatter(u, v, marker='8', c='cyan')


def format_figure(fig: plt.Figure, figsize: tuple[int, int], show=True):
    fig.set_size_inches(figsize)
    if show:
        fig.show()


def format_axis(ax: plt.Axes, xlabel:str, ylabel:str, title:str,legend_pos=1):
    ax.legend(loc=legend_pos)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.se


def set_fig_font_size_params(rel_fig_width,scalar):
    ref_axis_label_size = 19
    ref_title_size = 19
    ref_axis_ticks_size = 14
    ref_legend_size = 14
    rel_fig_width = rel_fig_width*scalar

    params = {'legend.fontsize': rel_fig_width*ref_legend_size,
              'axes.labelsize': rel_fig_width*ref_title_size,
              'axes.titlesize': rel_fig_width*ref_axis_label_size,
              'xtick.labelsize': rel_fig_width*ref_axis_ticks_size,
              'ytick.labelsize':  rel_fig_width*ref_axis_ticks_size,
              }
    plt.rcParams.update(params)


def set_line_widths(rel_fig_width,scalar):
    ref_lw = 1.5
    plt.rcParams.update({'lines.linewidth':ref_lw*rel_fig_width*scalar})
    plt.rcParams.update({'patch.linewidth':ref_lw*rel_fig_width*scalar})


def plot_ts_line(x_ser:np.ndarray|pd.Series, y_ser:np.ndarray|pd.Series, plt_ax:plt.Axes, ts_name:str,
                 mean_method=np.mean, kwargs=None):
    if isinstance(x_ser,pd.Series):
        x_ser = x_ser.to_numpy()
    if isinstance(y_ser,pd.Series):
        y_ser = y_ser.to_numpy()

    if x_ser.ndim != 1:
        raise TypeError('x_ser must be 1D')
    if y_ser.ndim > 1:
        if not mean_method:
            raise Warning('mean method needed for 2D y_ser ')
        if y_ser.ndim > 2:
            raise NotImplementedError('time series plotting for ndim > 2 not supported')
        y_ser2plot = mean_method(y_ser,axis=0)
    else:
        y_ser2plot = y_ser

    if not kwargs:
        kwargs = {'label':ts_name}
    if kwargs:
        plt_ax.plot(x_ser,y_ser2plot,**kwargs)
    if y_ser.ndim > 1:
        plot_ts_var(x_ser,y_ser,kwargs['c'],plt_ax)
        # plotvar(y_ser,plt_ax,x_ser,col_str=kwargs['c'])

def plot_ts_var(x_ser:np.ndarray|pd.Series, y_ser:np.ndarray|pd.Series,colour:str,plt_ax:plt.Axes):
    if isinstance(x_ser,pd.Series):
        x_ser = x_ser.to_numpy()
    if isinstance(y_ser,pd.Series):
        y_ser = y_ser.to_numpy()

    rand_npdample = [y_ser[np.random.choice(y_ser.shape[0], y_ser.shape[0], replace=True), :].mean(axis=0)
                     for i in range(500)]
    rand_npsample = np.array(rand_npdample)
    ci = np.apply_along_axis(mean_confidence_interval, axis=0, arr=rand_npsample).astype(float)

    plt_ax.fill_between(x_ser.tolist(),ci[1], ci[2],alpha=0.1,fc=colour)


def plot_sound_vbars(ax:plt.Axes,c='k'):
    t_onsets = np.arange(0,1,.25)
    sound_dur = 0.15
    for onset in t_onsets:
        ax.axvspan(onset,onset+sound_dur, facecolor='k', alpha=0.1)


def set_axis_frame(ax,frame_bool:tuple[bool,bool,bool,bool]):
    top,right,bottom,left = frame_bool
    for pos, pos_bool in zip(['top','right','bottom','left'],frame_bool):
        ax.spines[pos].set_visible(pos_bool)


def set_color_palette(colors=('F72585', '7209B7', '3A0CA3', '4361EE', '4CC9F0')):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)


def save_fig(fig,savepath):
    fig.savefig(savepath,bbox_inches='tight')


def plot_event_raster(ax, event_matrix):
    non_name_date_levels = [i for i, idx_name in enumerate(event_matrix.index.names) if
                            idx_name.lower() not in ['name', 'date']]
    sess_names = event_matrix.index.droplevel(non_name_date_levels).unique()
    animals, dates = np.unique(np.array(sess_names)[:, 0]), np.unique(np.array(sess_names)[:, 1])

    all_sess_eventz_list = []
    for i, (ax, animal) in enumerate(zip(axes, animals)):
        animal_cnt = 0
        for d, date in enumerate(dates):
            sess_name = f'{animal}_{date}'
            # sess_mat = copy(utils.align_nonpuil(self.data[sess_name].trialData[eventname],
            #                                self.data[sess_name].harpmatrices[harp_event], window,timeshift))
            if date not in trialData.index.get_level_values('date'):
                # print('Date not in trialdata, skipping')
                continue

            td2use = trialData.loc[[animal], [date], :]
            if outcome2filt:
                if extra_filts:
                    filts2use = outcome2filt + extra_filts
                else:
                    filts2use = outcome2filt
                td2use = filter_df(td2use, filts2use)
            try:
                sess_mat = copy(align_nonpuil(td2use[eventname],
                                              harpmatrices[sess_name][harp_event], window,
                                              trialData.loc[(animal, date)]['Offset'],
                                              timeshift))
            except KeyError:
                continue
            sess_mat
            byoutcome_ser = trialData.loc[(animal, date)]['Trial_Outcome']
            fs = 0.001
            all_sess_eventz = pd.DataFrame(np.full((len(sess_mat), int((window[1] - window[0]) / fs)), 0.0))
            all_sess_eventz.columns = np.linspace(window[0], window[1], all_sess_eventz.shape[1]).round(3)
            # axes[i].set_axisbelow(True)
            # axes[i].yaxis.grid(color='gray', linestyle='dashed',which='both')
            for e, event in enumerate(sess_mat):
                axes[i].axhline(animal_cnt - e, c='k', linewidth=.25, alpha=0.25, )
                epoch_events = np.full(int((window[1] - window[0]) / fs) + 1, 0.0)

                # print(list(sess_mat.values())[0])
                event = copy(event)
                eventz = sess_mat[event].round(3)

                epoch_events[((sess_mat[event] - window[0]) / fs).astype(int)] = 1
                all_sess_eventz.loc[e, eventz] = 1
                epoch_events = all_sess_eventz.loc[e, :].to_numpy()

                # if lfilt:
                #     epoch_events = utils.butter_filter(epoch_events, lfilt, 1 / fs, filtype='low')
                # b,a = s
                epoch_events[epoch_events == 0] = np.nan
                if byoutcome_flag:
                    if plotcol is None:
                        plotcol = int(td2use["Trial_Outcome"][e])
                    axes[i].scatter(all_sess_eventz.columns, epoch_events * (animal_cnt - e),
                                    c=f'C{plotcol}', marker='x', s=3, alpha=1, linewidth=.5)
                else:
                    axes[i].scatter(all_sess_eventz.columns, epoch_events * (animal_cnt - e), c=f'C{d}', marker='.')
                axes[i].axvline(0, ls='--', c='k')
                # axes[i].axvline(0.5, ls='--', c='grey')
            # ax.axhline(animal_cnt+20,ls='-',c='k')
            animal_cnt -= len(sess_mat)
            if outcome2filt:
                condname = outcome2filt[0].replace('a0', 'Non Rewarded')
                condname = condname.replace('a1', 'Rewarded')
            else:
                condname = 'all'
            axes[i].set_title(
                f'{harp_event_name} aligned to {eventname.replace("dt", "").replace("_", " ").replace("Gap", "X")}\n'
                f'{animal}, {condname} trials', size=10)
            axes[i].set_yticks([])

            all_sess_eventz.index = td2use.index
            all_sess_eventz_list.append(all_sess_eventz)

class FigureObj:
    def __init__(self, figsize=(9,7),font_scalar=1,lw_scalar=1):
        # print('This is a class to store plots and functions for a figure')
        self.plots = {}
        self.plotdata = {}
        self.legend_labels = {}
        self.figsize = figsize
        self.rel_fig_w = round(figsize[0]/9,1)
        self.font_scalar = font_scalar
        self.lw_scalar = lw_scalar

        self.update_plt_size_params()
        plt_params = {
            'savefig.format':'svg',
            # 'axes.labelpad': 0,
            'legend.frameon': False,
            'svg.fonttype': 'none',
            # 'axes.prop_cycle': mpl.cycler(color=['m', 'y', 'c', 'darkorange'])

        }
        plt.rcParams.update(plt_params)
        set_color_palette()
        # plt.rcParams['axes.ylabelpad']= -2

    def load_plotdata(self,plotname,plotdata_path):
        path = Path(plotdata_path)
        # if '.h5' in path:
        #     ftype = 'h5'
        # elif '.csv' in path:
        #     ftype = 'csv'
        # else:
        #     raise Warning('file')
        readers = [pd.read_hdf,pd.read_csv]
        reader2use = [ftype_i for ftype_i,ftype in enumerate(['.h5','.csv'] )if path.suffix == ftype]
        if len(reader2use) != 1:
            raise Warning(f'file extension {path.suffix} is invalid. readers available for .h5, csv  ')
        reader = readers[reader2use[0]]
        self.plotdata[plotname] = data = reader(path)
        self.legend_labels[plotname] = data.index.get_level_values('condition').to_series().unique()

    def update_plt_size_params(self):
        set_fig_font_size_params(self.rel_fig_w,self.font_scalar)
        set_line_widths(self.rel_fig_w,self.lw_scalar)

    def plot_ts(self, plotname, xlabel:str, ylabel:str, title:str, frame_bools=(False,False,True,True),
                exclude=None,**plt_kwargs,):
        fig,ax = plt.subplots()
        cis = plt_kwargs.get("cis", np.arange(len(self.legend_labels[plotname])))
        lss = plt_kwargs.get("lss", ['-']*len(cis))
        for cond_i, cond in enumerate(self.legend_labels[plotname]):
            cond_x_ser,cond_y_ser = self.plotdata[plotname].columns,self.plotdata[plotname].xs(cond, level=3).values
            line_kwargs = {'c': f'C{cis[cond_i]}','label': cond, 'ls':lss[cond_i]}
            if cond.lower() in ['none','control']:
                line_kwargs['c'] = 'k'
            plot_ts_line(cond_x_ser,cond_y_ser,ax,cond, np.mean,kwargs=line_kwargs)

        figsize = plt_kwargs.get('figsize',self.figsize)
        ax.axvline(0,ls='--',c='k')
        plot_sound_vbars(ax)
        set_axis_frame(ax, frame_bools)
        ylim = plt_kwargs.get('ylim',None)

        if ylim:
            ax.set_ylim(ylim[0],ylim[1])

        format_axis(ax, xlabel, ylabel,title)
        format_figure(fig,figsize)
        self.plots[plotname] = fig,ax

    def save_plots(self,savedir):
        savedir = Path(savedir)
        if not savedir.is_dir():
            savedir.mkdir()
        today_str = datetime.now().strftime('%y%m%d')
        for plotname in self.plots:
            savename = unique_file_path(savedir/f'{plotname}_{today_str}')
            save_fig(self.plots[plotname][0],savename)


def get_fig_mosaic(dates2plot):
    fig_form = ''
    n_cols = len(dates2plot)
    izz = '0123456789abcdefghijklmnopqrstuvwxyz'


    if n_cols == 4:
        n_cols = 2
    if n_cols <= 3:
        n_cols = n_cols
    else:
        n_cols = 3

    if len(dates2plot) > 1:
        for di, d in enumerate(dates2plot):
            fig_form += str(izz[di]) * 2
        chunked_fig_form = [fig_form[i:i + n_cols * 2].center(n_cols * 2, '.') for i in
                            range(0, len(fig_form), n_cols * 2)]
        fig_form = '\n'.join(chunked_fig_form)
    else:
        chunked_fig_form = ['00']
        fig_form = '00'

    return fig_form, chunked_fig_form, n_cols, izz

