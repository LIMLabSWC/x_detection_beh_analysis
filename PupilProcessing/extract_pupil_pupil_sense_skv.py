import argparse
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import cv2
import numpy as np

import pandas as pd
import skvideo.io
import time

import torch
from tqdm import tqdm
import yaml

print('imported non pupil sense')

from inference_pupil_sense import Inference, get_center_and_radius


def posix_from_win(path: str, ceph_linux_dir='/ceph/akrami') -> Path:
    """
    Convert a Windows path to a Posix path.

    Args:
        path (str): The input Windows path.
        :param ceph_linux_dir:

    Returns:
        Path: The converted Posix path.
    """
    if ':\\' in path:
        path_bits = PureWindowsPath(path).parts
        path_bits = [bit for bit in path_bits if '\\' not in bit]
        return Path(PurePosixPath(*path_bits))
    else:
        assert ceph_linux_dir
        return Path(path).relative_to(ceph_linux_dir)


def main(eye_video_paths,invert_gray_im, **kwargs):
    for eye_video_path in eye_video_paths:
        out_dir = Path(eye_video_path).parent/'sample_detection'
        if not out_dir.exists():
            out_dir.mkdir(parents=False, exist_ok=True)
        
        infer = Inference(str(kwargs.get('config_path')),str(kwargs.get('model_path')),
                          im_out_dir=out_dir)
        

        # load eye video
        eye_video_meta = skvideo.io.ffprobe(eye_video_path)
       
        print('loading')
        # time loading video
        load_start = time.time()
        # load video using skvideo.io.vread
        num_frames_to_load = kwargs.get('num_frames',0) 
        eye_video = skvideo.io.vread(str(eye_video_path), num_frames=num_frames_to_load, outputdict={'-pix_fmt': 'gray'}) 
        load_end = time.time()
        print(f'load time: {round((load_end-load_start)/60,2)} mins')
        # eye_video = eye_video.dtype(np.uint8)
        # read each frame of video and run pupil detectors
        eye_csvname = (eye_video_path.with_stem(f'{eye_video_path.stem}_eye_ellipse')).with_suffix('.csv')
        time_pre = time.time()

        # process subset of frames to find bbox
        eye_frames_for_bbox  =  eye_video[10000::5000] # every 5000 frames
        bboxes = [infer.predictor(frame)['instances'].pred_boxes.tensor.cpu().numpy()  # [x1, y1, x2, y2]
                  for frame in tqdm(eye_frames_for_bbox, desc='Finding bboxes', total=len(eye_frames_for_bbox))]
        bboxes = np.squeeze(np.array(bboxes))
        print(f'bboxes shape: {bboxes.shape}')

        # crop video to bboxes
        pad = 5
        mean_bbox = np.nanmean(bboxes, axis=0).astype(int)
        eye_crop_frames = eye_video[:,mean_bbox[1]-pad:mean_bbox[3]+pad, mean_bbox[0]-pad:mean_bbox[2]+pad]
        eye_crop_frames = eye_crop_frames.astype(np.uint8)

        ellipse_output = []

        for frame_number, (eye_frame, crop_frame) in tqdm(enumerate(zip(eye_video,eye_crop_frames)), total=len(eye_video), 
                                            desc='Processing frames'):

            # run detector on video frame
            output = infer.predictor(eye_frame)
            instances = output["instances"]

            if len(bboxes) <= 0:
                # ellipse_output[frame_number] = {'frame_num': frame_number,'radius': np.nan, 'xc': np.nan, 'yc': np.nan}
                ellipse_output.append([np.nan, np.nan, np.nan, np.nan, np.nan])
                continue


            if frame_number % 10000 == 0:
                infer.infer_image_display(infer.predictor(eye_frame), eye_frame, infer.im_out_dir, f'{frame_number}.png')
                # infer.infer_image_display(infer.predictor(crop_frame), crop_frame, infer.im_out_dir, f'{frame_number}.png',x_offset=mean_bbox[0]-pad, y_offset=mean_bbox[1]-pad)


            boxes = instances.pred_boxes.tensor.cpu().numpy()
            
            classes = instances.pred_classes
            scores = instances.scores

            instances_with_scores = [(i, score) for i, score in enumerate(scores)]
            instances_with_scores.sort(key=lambda x: x[1], reverse=True)

            for index, score in instances_with_scores:
                if classes[index] == 0:  # 0 is Pupil
                    pupil = boxes[index]
                    pupil_info = get_center_and_radius(pupil)
                    radius = int(pupil_info["radius"])
                    xc = int(pupil_info["xCenter"])
                    yc = int(pupil_info["yCenter"])
                    # ellipse_output[frame_number] = {'frame_num': frame_number,'radius': radius, 'xc': xc, 'yc': yc, 'score': score}
                    ellipse_output.append([frame_number, radius, xc, yc, int(score)])
                    break  # Only one prediction per frame

            # write dict as csv using pd.dataframe
        pupil_est_df = pd.DataFrame(np.array(ellipse_output), columns=['frame_num','radius','xc','yc','score'])
        # pupil_est_df.columns = ['frame_num','radius','xc','yc','score']
        pupil_est_df.to_csv(eye_csvname, index=False)
        print(f'total time taken: {round((time.time()-time_pre)/60,2)} mins')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eye_video_paths")
    parser.add_argument("--invert",default=0,type=int)
    parser.add_argument("--pupilsense_config_file",default='configs/pupil_sense.yaml',type=str)

    print('Running pupil detection on eye video')
    print('----------------------------------')

    # model configs
    args = parser.parse_args()
    with open(args.pupilsense_config_file) as f:
        config = yaml.safe_load(f)

    ceph_dir = Path(config[f'ceph_dir_{platform.system().lower()}'])
    config_path = ceph_dir / posix_from_win(config['config_path'])
    model_path = ceph_dir / posix_from_win(config['model_path'])

    print(f'config_path: {config_path}')
    print(f'model_path: {model_path}')
    num_frames = config['num_frames']

    eye_video_paths = [ceph_dir/ posix_from_win(eye_video_path) 
                       for eye_video_path in args.eye_video_paths.split(';')]

    main(eye_video_paths, args.invert,
         config_path=config_path, model_path=model_path, num_frames=num_frames)

