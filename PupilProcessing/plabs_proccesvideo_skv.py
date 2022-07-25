import argparse
import csv
from operator import invert
import os
import cv2
import numpy as np
from pupil_detectors import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
import skvideo.io
import time


def main(eye_video_paths, eye_ts_paths,show_video,invert_gray_im,show_2d):
    for eye_video_path, eye_ts_path in zip(eye_video_paths.split(','), eye_ts_paths.split(',')):
        eye_video_path = f'{eye_video_path}\eye0.mp4'
        # eye_ts_path = f'{eye_ts_path}\eye0'
        if eye_ts_path.find('.npy') != -1:
            eye_ts = np.load(eye_ts_path)
        elif eye_ts_path.find('.csv') != -1:
            eye_ts = np.loadtxt(open(eye_ts_path, "rb"), delimiter=",", skiprows=1)
        else:
            print('no timestamp array found')
            eye_ts = np.arange(1000000)
        # create 2D detector
        detector_2d = Detector2D()
        # create pye3D detector
        camera = CameraModel(focal_length=561.5, resolution=[400, 400])
        detector_3d = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)
        # load eye video
        eye_video_meta = skvideo.io.ffprobe(eye_video_path)
        frames_time = [int(item) for item in eye_video_meta['video']['@avg_frame_rate'].split('/')]
        fps = frames_time[0]/frames_time[1]
        print('loading')
        eye_video = skvideo.io.vreader(eye_video_path) 
        print('loaded')
        # read each frame of video and run pupil detectors
        eye_csvname = f'{eye_video_path.split(".")[0]}_eye_ellipse.csv'
        with open(eye_csvname,'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['timestamp','2d_centre','2d_radii','3d_centre','3d_radii'])
            time_pre = time.time()
            for frame_number, eye_frame in enumerate(eye_video):
                # read video frame as numpy array
                grayscale_array = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
                if invert_gray_im:
                    grayscale_array = cv2.bitwise_not(grayscale_array)
                # run 2D detector on video frame
                result_2d = detector_2d.detect(grayscale_array)
                result_2d["timestamp"] = frame_number / fps
                ellipse_2d = result_2d['ellipse']
                # pass 2D detection result to 3D detector
                result_3d = detector_3d.update_and_detect(result_2d, grayscale_array)
                ellipse_3d = result_3d["ellipse"]

                if frame_number%10000==0 and frame_number>0:
                    
                    print((time.time()-time_pre)/frame_number)
                csvwriter.writerow([round(eye_ts[int(frame_number)],3),[round(float(v), 3) for v in ellipse_2d["center"]],[round(float(v / 2), 3) for v in ellipse_2d["axes"]],\
                                    [round(float(v), 3) for v in ellipse_3d["center"]],[round(float(v / 2), 3) for v in ellipse_3d["axes"]]])
            print(f'total time taken: {round((time.time()-time_pre)/60,2)} mins')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eye_video_paths")
    parser.add_argument("eye_ts_paths")
    parser.add_argument("--show_video",default=1,type=int)
    parser.add_argument("--invert",default=0,type=int)
    parser.add_argument("--show2d",default=0,type=int)
    args = parser.parse_args()
    main(args.eye_video_paths, args.eye_ts_paths,args.show_video,args.invert,args.show2d)

