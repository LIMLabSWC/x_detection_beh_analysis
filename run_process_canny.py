import argparse

import skvideo.io

from elliptic_fitting.pupil_edge_detection import Main
from pathlib import Path
import multiprocessing

if __name__ == "__main__":
    print(multiprocessing.cpu_count())

    parser = argparse.ArgumentParser()
    parser.add_argument('vd')
    parser.add_argument('n')
    parser.add_argument('--nf',default=0)
    parser.add_argument('--ow',default=0)

    args = parser.parse_args()
    print(args.n)

    if not Path(args.vd,f'{args.n}_canny_ellipses.csv').is_file() or args.ow:
        run = Main(args.vd,args.n,num_frames=args.nf,vreader=skvideo.io.vreader)
        run.process_frames()
        run.save_canny_df()

    else:
        print('canny file exists')