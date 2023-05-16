import numpy as np
import random
import pandas as pd

def bootstrap_ci(data):

    def bootstrap(trace):
        #takes an aligned pupil trace
        # --> dataframe of bootstrapped medians
        means = [] #empty list to store mean of each bootstrap
        for i in range(100):
                bt_data = pd.DataFrame(columns=trace.columns) #making empty df with pupil traces length
                for k in range(trace.shape[0]): #making the bootstrapped data same size as acutal
                    selected_num = random.choice(range(trace.shape[0])) #for each new row, select random old
                    bt_data = pd.concat([bt_data,trace[selected_num: selected_num + 1]]) #append new row
                try:
                    means.append(np.nanmean(bt_data, axis=0)) #get mean for each corresponding time
                except ZeroDivisionError:
                    print(len(bt_data), 'there was an attempt to divide by zero')

        return pd.DataFrame(means)

    def boots_ci(boot_df, ci_type = 'pointwise',alpha = 0.05):
        #calculating bootstrao confidence bands
        #boot_df df of bootstrapped values. Values as rows, samples
        #alpha = confidence level
        if alpha < 0 or alpha > 1:
            print('Invalid alpha')
        else:
            #pointwiseconfidenceband
            if ci_type == 'pointwise':
                low_list =[]
                high_list = []
                for i in boot_df.columns:
                    low_list.append(np.quantile(boot_df[i], (alpha/2)))
                    high_list.append(np.quantile(boot_df[i],(1-(alpha/2))))

        return low_list, high_list

    boot_data = bootstrap(data)
    low_band, high_band = boots_ci(boot_data)

    return low_band, high_band

def significance(boot_df,low_band, high_band):
    frames_sec = 90
    boot_len = boot_df.shape[1]
    half_sec = boot_len / (frames_sec*2)
    start = 0
    end = half_sec
    half_sec_means = []
    half_sec_upper = []
    half_sec_lower = []
    for i in range(len(frames_sec/2)):
        half_sec_means.append(np.nanmean(boot_df[start:end]))
        half_sec_upper.append(np.nanmean(high_band[start:end]))
        half_sec_lower.append(np.nanmean(low_band[start:end]))
        start = start + half_sec
        end = end + half_sec
