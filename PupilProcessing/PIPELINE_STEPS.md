# Pupillometry Processing

This is a subdirectory containing classes and run code for process pupil data recorded with the pupil labs software
Aim: Get pupil video/output, return uniformly sampled filtered and interpolated pupil dataframe

## 1: Extract formatted pupil data
Step 1 takes either a session folder from pupil labs - "c:\recordings\000"  - or a video with timestamps and outputs formatted dataframe with pupil labs detection results.
Dataframe fields ("eye_id","timestamp","topic","confidence","diameter_2d [px]","diameter_3d [mm]","2d_radii","2d_centre") for each frame. 

### Non-pupil labs videos:

Generate plab equivalent data using `plabs_processvideos_skv,py`. Give video and timestamp path. Outputs extracted pupils csv

### Pupil labs videos

Generate extracted pupils csv using `extract_diameter.py`. Give top directory. Script will find session appropriate folders for analysis

### DeepLabCut

Deeplabcut is used as an alternative pupil detector. Associated analysed videos needed to downstream scirpts
Bodypoint coordiantes (.h5 files) and labelled videos generated using `analyze_vids.py` and `analyze_vids_job.sh` scipts run on hpc

## 2: Align and convert pupil timestamps to behavioural time 

`align_pupils.py`: Reformats extracted pupils outputs to Bonsai time using timesync file. Corrects for drift caused by different clock speaeds  
Give top directory. Script will find **extracted_pupils** tagged csv files in subdirectories. Valid_sessions used given by find_good_session(_**stage**,**trial threshold**)  
Scipt will also merge multiple sessions taken on the same day by same animal
Saves csv file in specified `aligned_dir` tagged **_pupil_data** for 2d and 3d detectors for each session

## 3: Processing aligned session pupil data + session behaviour data
`pupilpipeline.py` contains class Main. `Main.load_pdata()` loads, processes aligned pupil labed detected diameters and deeplabcut pupildiamters.

Give top directory as well as parameters for preproccessing used in `Main.load_pdata()`. 

### Pupil diameter data processing steps

* `uniformSample()`: uniformly samples the pupil data so it has a constant rate
* `removeOutliers()`: identifies points which are speed/size outliers and sets them to zero
* `downSample()`: down samples the data to a lower frame rate to make further processing quicker
* `interpolate()` takes locations of outliers (show as red) and extends the range by a short gap (yellow) then replaces these with linear interpolation between flanking values
* `frequencyFilter()`: applies a high and/or low pass filter to remove drift and/or noise from the data
* `zScore()`: z scores the pupil diameters

### Pickle saving
Dictionary holding pupil dataframe and trialdata for each session saved as pickle file. This is the output file used for analysis


