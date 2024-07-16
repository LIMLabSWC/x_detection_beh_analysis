import time
import sys
sys.path.append("..")
from pupil_estimate_analysis.manual_frame_evaluation import FrameAnalysis as FrameAnalysis
from skimage import feature
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import yaml
from PupilProcessing.pupilpipeline import get_dlc_est_path
import analysis_utils as utils
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from pathlib import Path
from analysis_utils import run_ransac
import skvideo.io
from tqdm import tqdm
import jax
import jax.numpy as jnp
import jax.image
from scipy.ndimage import gaussian_filter
from jax.scipy.signal import convolve
import scipy
import multiprocessing
import warnings
from skimage.restoration import denoise_bilateral
from copy import deepcopy as copy
from plotting_functions import scatter_on_frame,plot_bbox,draw_ellipse_fit



def sobel_filter(image):
    # Define Sobel filter kernels for x and y directions
    kernel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Perform 2D convolution
    gradient_x = convolve(image, kernel_x, mode='same')
    gradient_y = convolve(image, kernel_y, mode='same')

    return gradient_x, gradient_y


def get_canny_edges(frame, bbox,mask_thresh,sigma=1.5,):
    assert bbox
    if not bbox:  # getting smaller bbox on every frame
        frames4bbox = np.full_like(frame, 255)
        frames4bbox[ 40:-40, 40:-40] = frame[40:-40,40:-40]  # setting boundary  pixels to white for bbox
        black_thresh_frames = (frames4bbox <= np.percentile(frames4bbox, .25))  # remove
        mean_xy = np.mean(np.where(black_thresh_frames == 1), axis=1)
        pad = 10
        bbox = mean_xy[0] - pad, mean_xy[1] - pad, mean_xy[0] + pad, mean_xy[1] + pad

    min_y,min_x,max_y,max_x = bbox
    frame = np.squeeze(frame).copy()
    frame_cropped = frame[min_y:max_y, min_x:max_x].astype(float)
    # smooth out light pixels
    frame_high_thresh = frame_cropped>np.percentile(frame_cropped,90)
    frame_cropped[frame_high_thresh] = np.nan
    frame_cropped= utils.iterp_grid(frame_cropped,)
    # frame_cropped = gaussian_filter(frame_cropped,sigma=0.5)
    # frame_cropped = denoise_bilateral(frame_cropped,sigma_spatial=2)
    # frame_cropped = frame_cropped<=np.percentile(frame_cropped,5)
    # frame_cropped = frame_cropped.astype(int)
    frame_cropped = jnp.array(frame_cropped)
    edges = feature.canny(frame_cropped,sigma,0.7,0.98,use_quantiles=True)  # was 0.7,0.98
    return edges


def canny_edge_detection(image, low_threshold, high_threshold, sigma=1.0):
    # Apply Gaussian smoothing to the image
    image = gaussian_filter(image, sigma)

    def sobel_filter(image):
        # Define Sobel filter kernels for x and y directions
        kernel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Perform 2D convolution
        gradient_x = convolve(image, kernel_x, mode='same')
        gradient_y = convolve(image, kernel_y, mode='same')

        return gradient_x, gradient_y

    # Apply the Sobel filter to the image
    gradient_x, gradient_y = sobel_filter(image)

    # Compute gradient magnitude
    gradient_magnitude = jnp.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Define edges based on gradient magnitude and threshold values
    strong_edges = gradient_magnitude > high_threshold
    weak_edges = (gradient_magnitude >= low_threshold) & (gradient_magnitude <= high_threshold)

    # Create a binary edge map
    edge_map = jnp.where(strong_edges, 1, 0)

    return edge_map



class Main:

    def __init__(self, viddir, name,num_frames=0, vreader=skvideo.io.vreader):
        self.canny_ells = None
        self.frame_fig = np.array([])
        dlc_estimate_path, _ = get_dlc_est_path(viddir, True, f'{name}_', '')
        # self.dlc_estimates = pd.read_hdf(dlc_estimate_path)
        self.viddir = viddir
        self.name = name

        self.video_path = Path(viddir) / f'{name}_eye0.mp4'
        print(f'Loading video')
        video_frame = vreader(str(self.video_path),outputdict={"-pix_fmt": "gray"},as_grey=True,num_frames=num_frames)
        if type(video_frame) == np.ndarray:
            video_frame = np.squeeze(video_frame)
        self.frames = video_frame
        self.get_bounding_box()
        self.num_frames = int(num_frames)

        debug_plot_dir = Path(self.viddir)/'debug_plots'
        if not debug_plot_dir.is_dir():
            debug_plot_dir.mkdir()
        self.debug_plot_dir = debug_plot_dir

    def get_bounding_box(self,pad=25,num_frames=10000):
        # get first frames to get average mean black
        if type(self.frames) == np.ndarray:
            self.frames_arr = self.frames
        else:
            self.frames_arr = skvideo.io.vread(str(self.video_path),outputdict={"-pix_fmt": "gray"}
                                               ,as_grey=True,num_frames=num_frames)
        self.frames_arr = np.squeeze(self.frames_arr)
        frames4bbox = np.full_like(self.frames_arr,255)
        frames4bbox[:,40:-40,40:-40] = self.frames_arr[:,40:-40,40:-40]  # setting boundary  pixels to white for bbox
        self.num_frames_arr = num_frames
        black_thresh_frames = (frames4bbox <= np.percentile(frames4bbox, .25))  # remove
        # print(black_thresh_frames.mean(axis=0).min(),black_thresh_frames.mean(axis=0).max())
        mean_xy = np.mean(np.where(black_thresh_frames.mean(axis=0) > 0.1), axis=1)
        # mean_xy = [135,95]
        bbox = mean_xy[0] - pad, mean_xy[1] - pad, mean_xy[0] + pad, mean_xy[1] + pad
        # print(bbox)

        self.bbox = [int(e) for e in bbox]
        self.black_thresh = np.percentile(self.frames_arr, .3)

    def imshow_sample_frames(self,frame_iter, num_frames=16,name_suffix=('',)):
        rand_idx = np.random.choice(len(frame_iter),num_frames)
        fig,ax = plt.subplots(4,4,figsize=(16,16))
        for frame_i, ai in zip(rand_idx, ax.flatten()):
            ai.imshow(np.squeeze(frame_iter[frame_i]), cmap='gray')
            ai.axhline(self.bbox[0], c='g')
            ai.axhline(self.bbox[2], c='g')
            ai.axvline(self.bbox[1], c='g')
            ai.axvline(self.bbox[3], c='g')
            ai.set_xticks([])
            ai.set_yticks([])
        fig.set_constrained_layout('constrained')
        fig.savefig(str(self.debug_plot_dir/f'{"_".join(name_suffix)}.svg'))

    def scatter_sample_frames(self,frame_iter,x_ser,y_ser,pointstyle:(str,int,str), num_frames=16,name_suffix=('',)):

        rand_idx = np.random.choice(len(frame_iter),num_frames)
        fig,ax = plt.subplots(4,4,figsize=(16,16))
        # pnt_c, pnt_s, pnt_m = pointstyle
        for frame_i, ai in zip(rand_idx, ax.flatten()):
            # ai.imshow(np.squeeze(frame_iter[frame_i]), cmap='gray')
            #
            # ai.scatter(x_ser[frame_i]+self.bbox[1],y_ser[frame_i]+self.bbox[0],
            #            c=pnt_c, s=pnt_s, marker=pnt_m, label="_".join(name_suffix))
            #
            # ai.axhline(self.bbox[0], c='g')
            # ai.axhline(self.bbox[2], c='g')
            # ai.axvline(self.bbox[1], c='g')
            # ai.axvline(self.bbox[3], c='g')

            scatter_on_frame(ai,np.squeeze(frame_iter[frame_i]),self.bbox,x_ser[frame_i],y_ser[frame_i],
                             label="_".join(name_suffix),pointstyle=pointstyle)
            plot_bbox(ai,self.bbox)
            ai.set_xticks([])
            ai.set_yticks([])
            ai.legend()
        fig.set_constrained_layout('constrained')
        fig.savefig(str(self.debug_plot_dir / f'{"_".join(name_suffix)}.svg'))

    def ellipse_sample_frames(self,frame_iter,ell_iter,pointstyle:(str,int,str), num_frames=16,name_suffix=('',)):

        rand_idx = np.random.choice(len(frame_iter),num_frames)
        fig,ax = plt.subplots(4,4,figsize=(16,16))
        # Generate points for the ellipse

        for frame_i, ai in zip(rand_idx, ax.flatten()):
            # ai.imshow(np.squeeze(frame_iter[frame_i]), cmap='gray')

            frame_ellipse = ell_iter[frame_i]
            canny_edge_xs = [np.where(self.all_edges[frame_i] == 1)[1]]
            canny_edge_ys = [np.where(self.all_edges[frame_i] == 1)[0]]

            scatter_on_frame(ai,np.squeeze(frame_iter[frame_i]),self.bbox,canny_edge_xs,canny_edge_ys,
                             label="_".join(name_suffix),pointstyle=(['green',.5,'x']))
            draw_ellipse_fit(ai,frame_ellipse,"_".join(name_suffix),pointstyle)
            plot_bbox(ai,self.bbox)
            ai.set_xticks([])
            ai.set_yticks([])
            ai.legend()
        fig.set_constrained_layout('constrained')
        fig.savefig(str(self.debug_plot_dir / f'{"_".join(name_suffix)}.svg'))

    def process_frames(self):

        print('getting edges')
        if type(self.frames) == np.ndarray:
            with multiprocessing.Pool() as pool:
                self.all_edges = list(tqdm(pool.imap(get_canny_edges, (self.frames,self.bbox,self.black_thresh,3)),
                                           total=1))
        else:
            self.all_edges = [get_canny_edges(frame,self.bbox,self.black_thresh,2.5 )
                              for frame_idx, frame in tqdm(enumerate(self.frames))]
        # for plotting_notebooks debug
        canny_edge_xs = [np.where(frame_edges==1)[1] for frame_edges in self.all_edges]
        canny_edge_ys = [np.where(frame_edges==1)[0] for frame_edges in self.all_edges]

        self.scatter_sample_frames(self.frames_arr[:len(self.all_edges)],canny_edge_xs,canny_edge_ys,('pink',3,'x'),
                                   name_suffix=('canny',))
        print('getting ellipses')
        print(len(self.all_edges[0]))
        with multiprocessing.Pool() as pool:
            # report the number of processes in the pool
            # report the number of active child processes
            chunksize = int(len(self.all_edges))

            canny_ells = list(tqdm(pool.imap(run_ransac, self.all_edges,),
                                   total=len(self.all_edges)))
        # canny_ells = [run_ransac(frame_edges) for frame_edges in tqdm(self.all_edges)]
        canny_res = np.array(canny_ells)[:,:-1]
        canny_res[:,0] += self.bbox[1]
        canny_res[:,1] += self.bbox[0]
        canny_res_df = pd.DataFrame(canny_res,columns=['canny_centre_x','canny_centre_y',
                                                       'canny_raddi_a','canny_raddi_b',])
        self.canny_ells = canny_res_df
        self.ellipse_sample_frames(self.frames_arr[:len(self.all_edges)],self.canny_ells.values,
                                   ('magenta',1.5,'-'),name_suffix=('canny_fit',))

    def save_canny_df(self):
        if self.canny_ells is not None:
            canny_ell_path = Path(self.viddir)/f'{self.name}_canny_ellipses.csv'
            pd.DataFrame.to_csv(self.canny_ells,canny_ell_path,index=False)




if __name__ == "__main__":

    viddir = r'X:\Dammy\mouse_pupillometry\mouse_hf\DO76_231027_000'
    name = 'DO76_231027'
    dlc_path,_ = get_dlc_est_path(viddir,True,f'{name}_','')
    n_rand_frames = 4
    subset_frame_pklpath = rf'H:\gd_analysis\pupil_estimate_analysis\subset_frames_{name}.pkl'

    run = FrameAnalysis(viddir,name,subset_frame_pklpath)
    fig,ax = plt.subplots()
    frame_no = 2
    frame = run.subset_frames[frame_no].copy()  # .astype(float)
    frame_high_thresh = frame>100
    frame[frame_high_thresh] = 255
    mask = frame == 255
    frame[frame_high_thresh] = np.interp(np.flatnonzero(frame_high_thresh),np.flatnonzero(np.invert(frame_high_thresh)),
                                         frame[np.invert(frame_high_thresh)])
    ax.imshow(frame,cmap='gray')
    edges = feature.canny(frame,2.5)
    edges_for_mask = np.invert(edges)
    edges = edges.astype(float)
    edges[edges_for_mask] = np.nan
    ax.imshow(edges*255,cmap='spring')
    fig.show()

    # get dlc for frame
    estimates4dlc = run.dlc_estimates.loc[run.subset_frame_idx, :]
    pupil_points_only_df = estimates4dlc.drop(['edgeE', 'edgeW'],axis=1 , level=1)
    # bad_frames = pupil_points_only_df.isna().sum() > 5 * 2
    # estimates4dlc.loc[bad_frames] = np.nan
    # get bounding box
    x = pupil_points_only_df.loc[:, pd.IndexSlice[:, :, ('x')]].values
    y = pupil_points_only_df.loc[:, pd.IndexSlice[:, :, ('y')]].values
    ax.scatter(x[1,:],y[1,:],c='r')
    # xy_arr = np.array(xy_df)
    fig2,ax2 = plt.subplots()
    pad = 5
    min_x,min_y,max_x,max_y = np.int_([x.min(axis=1)-pad, y.min(axis=1)-pad,x.max(axis=1)+pad,y.max(axis=1)+pad])
    ax2.imshow(frame[min_y[frame_no]:max_y[frame_no],min_x[frame_no]:max_x[frame_no]],cmap='gray')
    ax2.imshow(edges[min_y[frame_no]:max_y[frame_no],min_x[frame_no]:max_x[frame_no]],cmap='spring')
    edge_cropped = edges[min_y[frame_no]:max_y[frame_no],min_x[frame_no]:max_x[frame_no]]
    # ax.scatter(x[1],y[1])
    edge_y,edge_x = np.where(edge_cropped==1)
    ax2.scatter(edge_x, edge_y,c='y')
    fig2.show()

    (xc,yc),r1,r2 = utils.fit_elipse_extra(np.column_stack((edge_x,edge_y)),'ransac')
    run.frame_ellipses['canny_ransac'] = np.array([r1,r2,xc,yc])
    t = np.linspace(0, 2 * np.pi, 100)  # Angle values
    Ell = np.array([r2 * np.cos(t), r1 * np.sin(t)])
    ax2.plot(xc + Ell[0, :], yc + Ell[1, :], label='canny_ransac',c='y')
    fig2.show()

    before = time.time()
    print(skvideo.io.vread(r'X:\Dammy\mouse_pupillometry\mouse_hf\DO75_231002_000\DO75_231002_eye0.mp4',
                     outputdict={"-pix_fmt": "gray"},as_grey=True).shape)
    print(time.time()-before)
