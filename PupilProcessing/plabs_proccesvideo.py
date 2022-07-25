import argparse
import csv
from operator import invert
import os
import cv2
import numpy as np
from pupil_detectors import Detector2D
# import pandas as pd
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
import skvideo


def main(eye_video_paths, eye_ts_paths,show_video,invert_gray_im,show_2d,rotate):
    for eye_video_path, eye_ts_path in zip(eye_video_paths.split(','), eye_ts_paths.split(',')):
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
        camera = CameraModel(focal_length=3.04, resolution=(244, 244))
        detector_3d = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)
        # load eye video
        eye_video = cv2.VideoCapture(eye_video_path)
        # read each frame of video and run pupil detectors
        eye_csvname = f'{eye_video_path.split(".")[0]}_eye_ellipse.csv'


        out = cv2.VideoWriter(r'C:\bonsai\gd_analysis\videos\mouse_plab3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 100, (256, 256))

        with open(eye_csvname,'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['ts','2d_centre','2d_radii','3d_centre','3d_radii'])
            

            while eye_video.isOpened():
                # eye_video.set(cv2.CAP_PROP_FPS,10000)
                frame_number = eye_video.get(cv2.CAP_PROP_POS_FRAMES)
                fps = eye_video.get(cv2.CAP_PROP_FPS)
                ret, eye_frame = eye_video.read()
                # if rotate:
                #     eye_frame = cv2.rotate(eye_frame,cv2.ROTATE_180)
                if ret:
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
                    # ellipse_3d.to_csv('ellipse.csv',index=False)
                    # draw 3D detection result on eye frame
                    try:
                        cv2.ellipse(
                            eye_frame,
                            tuple(int(v) for v in ellipse_3d["center"]),
                            tuple(int(v / 2) for v in ellipse_3d["axes"]),
                            ellipse_3d["angle"],
                            0,
                            360,  # start/end angle for drawing
                            (0, 255, 0),  # color (BGR): red
                        )
                        if show_2d:
                            cv2.ellipse(
                            grayscale_array,
                            tuple(int(v) for v in ellipse_2d["center"]),
                            tuple(int(v / 2) for v in ellipse_2d["axes"]),
                            ellipse_2d["angle"],
                            0,
                            360,  # start/end angle for drawingpup
                            (0, 0, 255),  # color (BGR): red
                        )
                        # show frame
                        if show_video:
                            cv2.imshow("eye_frame", eye_frame)
                    except:
                        print(f'skipped frame with axes {tuple(int(v / 2) for v in ellipse_3d["axes"])}')
                    # save ellipse
                    # out.write(eye_frame)
                    # csvwriter.writerow([round(eye_ts[int(frame_number)],3),[round(float(v), 3) for v in ellipse_2d["center"]],[round(float(v / 2), 3) for v in ellipse_2d["axes"]],\
                    #                     [round(float(v), 3) for v in ellipse_3d["center"]],[round(float(v / 2), 3) for v in ellipse_3d["axes"]]])
                    # press esc to exit
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                else:
                    break
        eye_video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eye_video_paths")
    parser.add_argument("eye_ts_paths")
    parser.add_argument("--show_video",default=1,type=int)
    parser.add_argument("--invert",default=0,type=int)
    parser.add_argument("--rotate",default=0,type=int)
    parser.add_argument("--show2d",default=0,type=int)
    args = parser.parse_args()
    main(args.eye_video_paths, args.eye_ts_paths,args.show_video,args.invert,args.show2d,args.rotate)

