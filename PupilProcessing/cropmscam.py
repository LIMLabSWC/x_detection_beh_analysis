import os
import glob
import sys
import time
import subprocess
import csv
import re


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


viddir = sys.argv[1]
print(viddir,type(viddir))
if len(sys.argv) < 4:
    cropbox = [250,250]
else:
    cropbox = [int(item) for item in sys.argv[3].split(',')]
if len(sys.argv) < 3:
    vidtype = '.avi'
else:
    vidtype = sys.argv[2]
    print(vidtype)

# get all filenames of vids into text files to interate

list_allfiles = sorted_alphanumeric(glob.glob(f'{viddir}/*{vidtype}'))
allfilesstr = [f"file '{filename}'" for filename in list_allfiles]
with open(os.path.join(viddir,'allfiles.txt'),'w') as vidsfile:
    [vidsfile.write(f'{line} \n') for line in allfilesstr]
    # writer.writerows(allfilesstr)

# subprocess.run(f'ls {viddir} -1v *.avi | while read each; do "file $each" >> allfiles.txt; done'.split())
time.sleep(0)
#get crop params for this recording
with open(os.path.join(viddir,'crop.txt'),'r') as cropfile:
    cropparams = cropfile.readlines()
    cropparams = cropparams[0].split(',')

with open(os.path.join(viddir,'allfiles.txt'),'r') as vidsfile:
    vids = vidsfile.readlines()  # get list of videofiles

savedir = os.path.join(os.path.dirname(viddir),'crop/')
if os.path.exists(savedir):
    print('Crop path already exists. Will overwrite, abort if needed')
    time.sleep(5)
else:
    os.mkdir(savedir)  #create directory for cropped vids

print(savedir)
# with open(os.path.join(viddir,'allscripts.sh'),'w') as scriptfile:
for vid in vids:  # vid is a abs filepath
    vid = vid.split()
    vid = vid[1].replace("'",'')
    savename = os.path.join(savedir,os.path.basename(f'{vid[:-4]}'))
    # print(vid,savedir,savename)
    vidin = os.path.join(viddir,vid.replace("'",''))
    script = f'ffmpeg -i {vidin} -f avi -c:v libx264 -crf 18 -preset veryslow -filter:v crop={cropbox[0]}:{cropbox[1]}:{cropparams[0]}:{cropparams[1]} {savename}_crop.avi'
    # print(script)
    # scriptfile.write(f'{script} \n')
    subprocess.run(script.split())


list_allcrops = sorted_alphanumeric(glob.glob(os.path.join(viddir,f'crop/*{vidtype}'))) 
# print(list_allcrops)
allcropsstr = [f"file '{filename}'" for filename in list_allcrops]
# print(allcropsstr[0])
with open(os.path.join(viddir,'crop','allfiles.txt'),'w') as vidsfile:
    [vidsfile.write(f'{line} \n') for line in allcropsstr]
subprocess.run(f'ffmpeg -i {os.path.join(viddir,"crop","allfiles.txt")} -c copy {os.path.join(viddir,"crop","sessionvid.avi")}'.split()) 
# print(f'ffmpeg -f concat -safe 0 -i {os.path.join(viddir,"crop","allfiles.txt")} -c copy {os.path.join(viddir,"crop","sessionvid.avi")} -c:v ffv1')