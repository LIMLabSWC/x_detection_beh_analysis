import skvideo.io
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
from math import ceil
import analysis_utils as utils
from pathlib import Path
import pickle
from PupilProcessing.pupilpipeline import get_dlc_est_path
from tqdm import tqdm
import jax.numpy as jnp
import jax.image
from scipy.ndimage import gaussian_filter
from skimage import feature


def get_canny_edges(frame, bbox,sigma=2.5):
    min_x,min_y,max_x,max_y = bbox
    frame = np.squeeze(frame).copy()
    frame_cropped = frame[min_y:max_y, min_x:max_x].astype(float)
    # smooth out light pixels
    frame_high_thresh = frame_cropped>100
    frame_cropped[frame_high_thresh] = np.nan
    frame_cropped= utils.iterp_grid(frame_cropped)
    frame_cropped = gaussian_filter(frame_cropped,sigma=sigma)
    frame_cropped = jnp.array(frame_cropped)
    edges = feature.canny(frame_cropped,sigma)
    return edges

class FrameAnalysis:
    def __init__(self, viddir, name, subset_frame_pklpath, n_rand_frames=4,):
        self.crop_bbox = None
        self.frame_fig = np.array([])
        dlc_estimate_path, _ = get_dlc_est_path(viddir, True, f'{name}_', '')
        self.dlc_estimates = pd.read_hdf(dlc_estimate_path)
        self.frame_ellipses = {}

        video_path = Path(viddir) / f'{name}_eye0.mp4'
        if subset_frame_pklpath and Path(subset_frame_pklpath).is_file():
            with open(subset_frame_pklpath, 'rb') as pklfile:
                loaded_pkl = pickle.load(pklfile)
                self.subset_frames, self.subset_frame_idx = loaded_pkl['frames'], loaded_pkl['frame_idx']
                # randomsise order of frames
                rand_idxs = np.random.permutation(len(self.subset_frames))
                self.subset_frames = self.subset_frames[rand_idxs]
                self.subset_frame_idx = self.subset_frame_idx[rand_idxs]
        else:
            self.load_subset_of_frames(video_path, n_rand_frames, dtype=float)
            self.pickle_subset_frames(subset_frame_pklpath)

        # self.plot_frames(n_rand_frames)
        # self.get_dlc_diams()

    def load_subset_of_frames(self, video_path, n_frames, dtype=float, rand_frames=True):
        # Read the video using skvideo
        print('Loading video')
        video_data = skvideo.io.vread(str(video_path),as_grey=True,outputdict={"-pix_fmt": "gray"})
        video_data = np.squeeze(video_data)
        print(video_data.shape)

        # Check if the end_frame is greater than the total number of frames
        num_frames = video_data.shape[0]

        if rand_frames:
            frame_numbers = np.random.randint(0,num_frames,n_frames)
        else:
            frame_numbers = np.arange(0,n_frames)

        # Select the subset of frames and convert to the specified data type
        subset_frames = video_data[frame_numbers]  # .astype(dtype)
        self.subset_frames = subset_frames
        self.subset_frame_idx = frame_numbers

    def plot_frames(self,n_frames2plot,ncols=2,crop_bbox=None):
        if n_frames2plot<ncols:
            ncols= n_frames2plot
        self.frame_fig = plt.subplots(ceil(n_frames2plot/ncols),ncols,squeeze=False,sharey='all',sharex='all',
                                      figsize=(10,10))
        self.frame_fig[0].set_constrained_layout('constrained')
        for ax_i, ax in enumerate(self.frame_fig[1].flatten()):
            if not crop_bbox:
                ax.imshow(self.subset_frames[ax_i], cmap='gray')
            else:
                ax.imshow(self.subset_frames[ax_i][crop_bbox[2]:crop_bbox[3],crop_bbox[0]:crop_bbox[1]], cmap='gray')
            ax.set_xticks([]), ax.set_yticks([])
        self.show_fig()
        self.crop_bbox = crop_bbox

    def get_dlc_eye_ellipse(self, conf_thresh=0.5, fit_function='hyper'):

        estimates4dlc = self.dlc_estimates.loc[self.subset_frame_idx, :]
        scorer = estimates4dlc.columns.get_level_values(0)[0]

        body_points_names = np.unique(estimates4dlc.columns.get_level_values('bodyparts').to_list())
        for body_point in body_points_names:
            body_point_df = estimates4dlc[scorer, body_point]
            if fit_function == 'hyper':
                bad_body_points = estimates4dlc[scorer, body_point, 'likelihood'] < conf_thresh
                estimates4dlc.loc[bad_body_points, (scorer, body_point, 'x')] = np.nan
                estimates4dlc.loc[bad_body_points, (scorer, body_point, 'y')] = np.nan
        pupil_points_only_df = estimates4dlc.drop(['edgeE', 'edgeW'], axis=1, level=1)
        bad_frames = pupil_points_only_df.isna().sum(axis=1) > 5 * 2
        estimates4dlc.loc[bad_frames] = np.nan

        xy_df = pupil_points_only_df.loc[:, pd.IndexSlice[:, :, ('x', 'y')]].values
        # y_df = pupil_points_only_df.loc[:, pd.IndexSlice[:, :, 'y']].values
        xy_arr = np.array(xy_df)
        ellispe_estimates = np.array([utils.iterate_fit_ellipse(r,fit_function,plot=self.frame_fig[1].flatten()[i])
                                      for i,(_,r) in enumerate(zip(self.frame_fig[1].flatten(),xy_arr))])
        radii1_, radii2_, centersx_, centersy_ = [array.flatten() for array in np.hsplit(ellispe_estimates,4)]
        if self.crop_bbox:
            centersx_ -= self.crop_bbox[0]
            centersy_ -= self.crop_bbox[2]

        eyeEW_arr = np.array((estimates4dlc[scorer, 'eyeW'] - estimates4dlc[scorer, 'eyeE'])[['x', 'y']])
        eyeLR_arr = np.array((estimates4dlc[scorer, 'edgeE'] - estimates4dlc[scorer, 'edgeW'])[['x', 'y']])

        self.frame_ellipses[fit_function] = np.array([radii1_, radii2_, centersx_, centersy_])



    def get_dlc_diams_slow(self):
        dlc_diams = utils.get_dlc_diams(self.dlc_estimates, self.dlc_estimates.shape[0],
                                        self.dlc_estimates.columns.get_level_values(0)[0])
        _arr = np.array(dlc_diams[:4])
        self.frame_ellipses = _arr[:,self.subset_frame_idx]

    def draw_dlc_diam(self,fit_function='hyper',plot_kwargs=None):
        for ax_i, ax in enumerate(self.frame_fig[1].flatten()):
            frame_ellipse = self.frame_ellipses[fit_function][:,ax_i]
            angle = 0
            a, b, u, v,  = frame_ellipse

            # Generate points for the ellipse
            t = np.linspace(0, 2 * np.pi, 100)  # Angle values
            Ell = np.array([b * np.cos(t), a * np.sin(t)])
            ax.plot(u+Ell[0,:], v+Ell[1,:], label=fit_function, **(plot_kwargs if plot_kwargs else {}))
            # ax.scatter(u,v)

        self.show_fig()

    def plot_dlc_points(self,plot_kwargs=None):
        subset_dlc_points = self.dlc_estimates.loc[self.subset_frame_idx, :]
        subset_dlc_points.columns = subset_dlc_points.columns.reorder_levels(['coords','scorer','bodyparts'])
        body_points_names = np.unique(subset_dlc_points.columns.get_level_values('bodyparts').to_list())
        scorer = subset_dlc_points.columns.get_level_values(0)[0]

        for ax_i, ax in enumerate(self.frame_fig[1].flatten()):
            frame_idx = self.subset_frame_idx[ax_i]
            ax.scatter(subset_dlc_points.loc[frame_idx,['x']].to_numpy() - self.crop_bbox[0] if self.crop_bbox else 0,
                       subset_dlc_points.loc[frame_idx,['y']].to_numpy() - self.crop_bbox[2] if self.crop_bbox else 0,
                       **(plot_kwargs if plot_kwargs else {}))
        self.show_fig()

    def pickle_subset_frames(self,pklpath):
        with open(pklpath,'wb') as pklfile:
            pickle.dump({'frames':self.subset_frames, 'frame_idx':self.subset_frame_idx},pklfile)

    def show_fig(self,ncols=None):
        self.frame_fig[0].set_constrained_layout('constrained')
        self.frame_fig[0].show()
        if ncols:
            pass
            # utils.unique_legend(self.frame_fig,self.ncols)
        else:
            pass
            # utils.unique_legend(self.frame_fig)

    def get_bounding_box(self,pad=7):
        estimates4dlc = self.dlc_estimates
        pupil_points_only_df = estimates4dlc.drop(['edgeE', 'edgeW'], axis=1, level=1)

        x = pupil_points_only_df.loc[:, pd.IndexSlice[:, :, ('x')]].values
        y = pupil_points_only_df.loc[:, pd.IndexSlice[:, :, ('y')]].values
        min_x, min_y, max_x, max_y = np.int_(
            [x.min(axis=1) - pad, y.min(axis=1) - pad, x.max(axis=1) + pad, y.max(axis=1) + pad])
        self.bbox = [np.nanmedian(e).astype(int) for e in [min_x, min_y, max_x, max_y]]

if __name__ == "__main__":

    viddir = r'X:\Dammy\mouse_pupillometry\mouse_hf\DO75_230927_000'
    name = 'DO75_230927'
    dlc_path,_ = get_dlc_est_path(viddir,True,f'{name}_','')
    n_rand_frames = 1
    subset_frame_pklpath = f'subset_frames_{name}.pkl'

    run = FrameAnalysis(viddir, name, subset_frame_pklpath)

    run.plot_frames(n_rand_frames,crop_bbox=None)
    # run.plot_frames(n_rand_frames,crop_bbox=[50,130,70,150])
    run.get_dlc_eye_ellipse()

    # main.get_dlc_diams_slow()
    # run.draw_dlc_diam()
    run.plot_dlc_points(plot_kwargs=dict(c='gold',marker='x',s=100))
    fit_funcs = ['hyper']
    for fit_func in fit_funcs:
        run.get_dlc_eye_ellipse(fit_function=fit_func)
        run.draw_dlc_diam(fit_func,plot_kwargs=dict(lw=3,c='r',ls='--'))
    run.show_fig(0)
    # #
    # run.get_bounding_box()
    # run.all_edges = [get_canny_edges(f,run.bbox) for f in run.subset_frames]
    # all_edgeXY = [np.where(f==1) for f in run.all_edges]
    # # canny_ells = np.array([utils.run_ransac(f) for f in run.all_edges])
    # canny_ells = np.array([utils.fit_elipse_extra(np.column_stack(f),'ransac') for f in run.all_edges])
    # run.frame_ellipses['canny_ransac'] = canny_ells[:,:4]
    # run.draw_dlc_diam('canny_ransac')