import pandas as pd
import numpy as np
import os 
from datetime import datetime, timedelta


def matchvid2data(animal,date,starttime,viddir):
    
    timestampfile = f'{animal}_{date}_timestamp.dat'
    timestamp_df = pd.read_csv(os.path.join(viddir,timestampfile), delimiter='\t')

    frametime = [starttime+timedelta(milliseconds=framereltime) for framereltime in timestamp_df['sysClock']]
    timestamp_df[frametime] = frametime

    return timestamp_df


def alignframes
