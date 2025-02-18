import circle_fit as cf
import numpy as np
import pandas as pd


def iterate_fit_ellipse(xy_1d_array, fit_function='hyper', plot=None):
    if fit_function == 'hyper':
        ellipse_estimate = (fit_elipse(xy_1d_array.reshape((int(xy_1d_array.shape[0]/2),2),order='F')))
    elif fit_function in ['least_sq','weighted_reps','fns','ransac','renorm']:
        ellipse_estimate = (fit_elipse_extra(xy_1d_array.reshape((int(xy_1d_array.shape[0] / 2), 2), order='F'),
                                             fit_function=fit_function, plot=plot))
    else:
        return None, None, None, None

    return ellipse_estimate[1],ellipse_estimate[2],ellipse_estimate[0][0],ellipse_estimate[0][1]


def fit_elipse(point_array):

    xc, yc, r1, r2 = cf.hyper_fit(point_array)
    # xc, yc, r1, r2 = cf.least_squares_circle(point_array)

    return (xc,yc), r1, r1


def fit_elipse_extra(point_array, fit_function, plot=None):
    import elliptic_fitting.elliptic_fit as ell_fit

    f_0 = 100
    if fit_function == 'weighted_reps':
        A, B, C, D, E, F = ell_fit.elliptic_fitting_by_weighted_repetition(point_array[:, 0], point_array[:, 1], f_0)
    elif fit_function == 'least_sq':
        A, B, C, D, E, F = ell_fit.elliptic_fitting_by_least_squares(point_array[:, 0], point_array[:, 1], f_0)
    elif fit_function == 'renorm':
        A, B, C, D, E, F = ell_fit.elliptic_fitting_by_renormalization(point_array[:, 0], point_array[:, 1], f_0)
    elif fit_function == 'fns':
        A, B, C, D, E, F = ell_fit.elliptic_fitting_by_fns(point_array[:, 0], point_array[:, 1], f_0)
    elif fit_function == 'ransac':
        removed_x, removed_y = ell_fit.remove_outlier_by_ransac(point_array[:, 0], point_array[:, 1], f_0)
        A, B, C, D, E, F = ell_fit.elliptic_fitting_by_fns(removed_x, removed_y, f_0)

    else:
        raise Exception('Invalid fit function given')

    # w_fit_x, w_fit_y = ell_utils.solve_fitting([A,B,C,D,E,F],point_array[:,0],f_0)
    B *= 2
    D, E = [2 * f_0 * val for val in [D, E]]
    F = f_0 ** 2 * F

    r1 = -((np.sqrt(2 * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F) * (
                (A + C) - np.sqrt((A - C) ** 2 + B ** 2)))) / (B ** 2 - 4 * A * C)) #/ f_0
    r2 = -((np.sqrt(2 * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F) * (
                (A + C) + np.sqrt((A - C) ** 2 + B ** 2)))) / (B ** 2 - 4 * A * C)) #/ f_0
    xc = ((2 * C * D - B * E) / (B ** 2 - 4 * A * C))# / f_0
    yc = ((2 * A * E - B * D) / (B ** 2 - 4 * A * C)) #/ f_0
    # if plot:
    #     print(w_fit_x)
    #     plot.scatter(w_fit_x,w_fit_y,marker=7,s=5,c='magenta',label=f'{fit_function} pnts')

    # K = D ** 2 / (4 * A) + E ** 2 / (4 * C) - F
    # denominator = B ** 2 - 4 * A * C
    # xc = (2 * C * D - B * E) / denominator
    # yc = (2 * A * E - B * D) / denominator
    #
    # # K = - np.linalg.det(Q[:3, :3]) / np.linalg.det(Q[:2, :2])
    # root = math.sqrt(((A - C) ** 2 + B ** 2))
    # r1 = math.sqrt(2 * K / (A + C - root))
    # r2 = math.sqrt(2 * K / (A + C + root))
    # xc,r1 = np.mean(w_fit_x),  (np.max(w_fit_x)-np.min(w_fit_x))/2
    # yc,r2 = np.mean(w_fit_y),  (np.max(w_fit_y)-np.min(w_fit_y))/2

    return (xc, yc), r1, r2


def get_dlc_diams(df: pd.DataFrame,n_frames: int,scorer: str,):
    if n_frames == 0:
        n_frames = df.shape[0]

    diams_EW = np.full(n_frames,np.nan)
    edge_EW = np.full(n_frames,np.nan)

    body_points_names = np.unique(df.columns.get_level_values('bodyparts').to_list())
    for body_point in body_points_names:
        body_point_df = df[scorer,body_point]
        bad_body_points = df[scorer,body_point,'likelihood']<.5
        df.loc[bad_body_points, (scorer,body_point,'x')] = np.nan
        df.loc[bad_body_points, (scorer,body_point,'y')] = np.nan
    pupil_points_only_df = df.drop(['edgeE','edgeW'],axis=1,level=1)
    bad_frames = pupil_points_only_df.isna().sum(axis=1) > 5*2
    df.loc[bad_frames] = np.nan

    xy_df = pupil_points_only_df.loc[:, pd.IndexSlice[:, :, ('x','y')]].values
    # y_df = pupil_points_only_df.loc[:, pd.IndexSlice[:, :, 'y']].values
    xy_arr = np.array(xy_df)
    ellispe_estimates = np.array([iterate_fit_ellipse(r) for r in xy_arr])
    radii1_, radii2_, centersx_, centersy_ = [array.flatten() for array in np.hsplit(ellispe_estimates,4)]

    eyeEW_arr = np.array((df[scorer, 'eyeW'] - df[scorer, 'eyeE'])[['x', 'y']])
    eyeLR_arr = np.array((df[scorer, 'edgeE'] - df[scorer, 'edgeW'])[['x', 'y']])
    if len(eyeEW_arr) < n_frames:
        eyeEW_arr = np.pad(eyeEW_arr,[(0,n_frames-len(eyeEW_arr)),(0,0)],constant_values=np.nan)
        eyeLR_arr = np.pad(eyeLR_arr, [(0, n_frames - len(eyeLR_arr)), (0, 0)], constant_values=np.nan)
    diams_EW[:n_frames] = np.linalg.norm(eyeEW_arr,axis=1)[:n_frames]
    edge_EW[:n_frames] = np.linalg.norm(eyeLR_arr,axis=1)[:n_frames]

    return radii1_, radii2_, centersx_, centersy_, diams_EW,edge_EW
