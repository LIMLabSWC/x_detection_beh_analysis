# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:44:57 2021

@author: Ella Svahn
"""

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns



#%%
fig = plt.subplot()
#sns.plot(data=trial_data[trial_data.WarmUp==False].groupby(['Name','Date']).mean().reset_index(),x= 'Stim1_Duration',y='Trial_Outcome', hue = 'Name', palette = 'mako')
plt.plot(lick_df)

#%%
#6 is spout 0, 5 is spout 1, 3 is spout 2
x = lick_df.Timestamp
y = lick_df.Payload

#harp_time = harp_df.Times
#port0 = harp_df['PORT0 DO'] 

fig=plt.subplot()
plt.scatter(x,y)
plt.ylim(2,6.5) 
fig.axes.xaxis.set_ticklabels([])
fig.axes.yaxis.set_ticklabels([])
plt.xlabel('Time')
plt.xlim(2756770, 2756820)
plt.axvline(x=2756772, color = 'lightgreen', alpha=0.4, linewidth=18)#LED0
plt.axvline(x=2756774, color = 'orange', alpha=0.2, linewidth=6)#LED1
plt.axvline(x=2756800, color = 'lightblue', alpha=0.5, linewidth=8)#reward



'''
LED0 = np.zeros(2856765)
LED0[2756765:2756774]=3

LED1 = np.zeros(2856765)
LED1[2756774:2756775]=7

plt.fill(LED0, color = 'lightblue', alpha=0.3)
plt.fill(LED1, color = 'lightgreen', alpha=0.3)
#plt.plot(port0, harp_time)
'''


