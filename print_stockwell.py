# -*- coding: utf-8 -*-

## Header =====================================================================

"""
Generate normalised s-transform arrays for specific types of sensor data.

Important: Adjust settings in the USER TO EDIT section and via the input files.

Reads filenames and assosciated information from a configuration csv file 
'info_fp'. Each file (all of which need to reside in 'data_folder_fp') is then
read and interpreted as one of the three expected file types from which
individual sensor data columns are extracted:
    
'mic' - Microphone data ['mic' extracted]
'acc' - Accelerometer data ['acx','acy','acz' extracted]
'for' - Force gauge data ['frx','fry','frz' extracted]
 
These are then split according to data point snip lengths, where any left
over data points are removed evenly from the start and end of the data column 
before the splitting begins. There are seven individual signals, each one is
treated seperately. The 1D time domain snips are converted into 2D 
time-frequency domain arrays via a Stockwell transform with settings listed in
stock_setts. The arrays are resized into square shaped 2D arrays of size 
imsize x imsize and normalised between 0-1. These are saved in .csv format with
a naming convention as follows:
    
Format - XXXXXXXXXcN..._TTT

XXXXXXXXX - Test data tag
N... - Snip number (starts at 0 and can reach any length)
TTT - Signal type tag

Example: t01p02s01c0_acx

t01p02s01 - Test 1, pass 1, section 1
_acx - Acceleration signal in the x direction
c0 - First (python zero) snip

Additionaly, a brief log file is produced detailing which files were succesfully converted.

Depending on settings used, heatmap plots will also be generated as standalone
images and linearly interpolated labels (wears) will be written.

Warning - Low "decim" and high "n_fft" values cause long runtimes and may lead
to RAM issues for long sections of timeseries data. 

Warning - A 'memory leakage' issue has been identified due to the use of Numpy
Arrays (slicing does not create a new object, instead it's just a view). This 
has not been fixed but can be circumvented by using shorter production runs and
clearing variables inbetween these (check RAM usage and only restart once it
python RAM usage drops as deletion takes a minute).
"""

## USER TO EDIT - START =======================================================

# Flags
array = False # Create normalised stockwell arrays
plot = False # Create normalised stockwell array heatmap images
# Generate wear labels via linear interpolation (only works for ftype 'FOR')
label = True

# Number of sections (predefined by data format)
n_sections = 3

# Number of columns which are not stockwell settings in info_fp
offset = 4

# See types below, insert value of 0 (FOR),1 (ACC) or 2 (MIC)
ftype_selected = 1

# Filepath to the parent folder containing all of the input/raw data files
data_folder_fp = (r"C:\Users\jakub\OneDrive\Desktop\Emergency_Backup"
                  r"\02_Experimental\02_Raw")

# Full filepath of the log file to be created
log_fp = (r"C:\Users\jakub\OneDrive\Desktop\Emergency_Backup\02_Experimental"
           r"\06_Logs\Second_Print.log")

# Full filepath of the configuration/info file
info_fp = (r"C:\Users\jakub\OneDrive\Desktop\Emergency_Backup\02_Experimental"
           r"\04_Additional\Second_Print.txt")

# Filepath to the parent folder containing output bitmap arrays
array_fp = (r"C:\Users\jakub\OneDrive\Desktop\Emergency_Backup\02_Experimental"
           r"\05_Converted")

# Filepath to the parent folder containing output plots
plot_fp = (r"C:\Users\jakub\OneDrive\Desktop\Emergency_Backup\02_Experimental"
           r"\08_Images")

# Full filepath of the output labels (wears) file
label_fp = (r"C:\Users\jakub\OneDrive\Desktop\Emergency_Backup\02_Experimental"
           r"\07_Labels\Second_Print.txt")

# Types of filename variants occuring after the stem (linked to ftype_selected)
ftypes = (".txt","_acc.txt","_reduced_mic.txt")

# Sampling frequency of the base signal (lowest sampling frequency)
sfreq_base = 1000

# Output array size (square shaped array will be generated)
imsize = 128

## USER TO EDIT - END =========================================================

## Libraries ==================================================================

# Python version: 3.9.13
# Spyder version: 5.2.2
    
# Import libraries - internal
import os

# Import libraries - 3rd party
import numpy as np # 1.21.5
from mne.time_frequency import tfr_array_stockwell # 0.24.0
from skimage.transform import resize # 0.19.2
import matplotlib.pyplot as plt # 3.5.2

## Define functions ===========================================================

def split_info_data(data):
    
    '''
    For predefined types of experimental data outlined in
    'How to read the data.pdf' that were loaded in as numpy arrays, remove
    undesired (predefined) columns and return the data types and signal length.
    
    Inputs:
    data (numpy array) - Experimental data of a predifined format (see below).
        
    Outputs:
    sdata (numpy array) - Signal data trimmed of irrelevant columns.
    dtypes (list) - Type of sensor (3 chars).
    dtype_main (string) - Signal labels (3 chars).
    nlength (integer) - Signal length.
    '''

    try:
        nlength, ntype = data.shape # Signal length and number of columns
    except:
        # If .shape fails, assume that the array is 1D (list)
        nlength = len(data)

    # Force and current, only keep force
    if ftype_selected == 0:
        sdata = data[:,:3]
        dtypes = ["frx","fry","frz"]
        dtype_main = "for"
    # Acceleration, only keep Dytran
    elif ftype_selected == 1:
        sdata = data[:,:3]
        dtypes = ["acx","acy","acz"]
        dtype_main = "acc"
    # Sound
    elif ftype_selected == 2:
        sdata = data
        dtypes = ["mic"]
        dtype_main = "mic"
    # Raise error if unexpected number of signals
    else:
        raise Exception("The data file has an unexpected number of signals")
        
    return sdata,dtypes,dtype_main,nlength

def get_signal_frags(sdata,dtype,sniplen,signal_idx):
    
    '''
    Return a list of equal length signal fragments for one sensor, trimming
    any additional values that could not be 'binned' equally from both sides.
    
    Inputs:
    sdata (numpy array) - Signal data trimmed of irrelevant columns.
    dtype (string) - Type of sensor (3 chars).
    sniplen (integer) - Length of signal fragment/snip.
    signal_idx (integer) - Signal to be used column index.
        
    Outputs:
    signal_frags (list) - Signal fragments of length sniplen.
    '''
    
    if dtype == "mic":
        signal = sdata # Handle 1D array
    else:
        signal = sdata[:,signal_idx] # Extract relevant column
    nremove = signal.size % sniplen # Calculate remainder
    # Remove values that cannot be 'binned'
    signal = signal[np.arange(int(nremove/2),int(signal.size - (nremove/2)))]
    rsignal = np.expand_dims(signal, axis=[0,1]) # Add dimension
    signal_frags = np.split(rsignal,len(signal)/sniplen,2) # Split into frags
    
    return signal_frags

def stockwell_low_def(data,datalen,sfreq,width,n_fft,decim,lfreq,hfreq,imsize):
    
    '''
    Return a square shaped 2D array which is the 0-1 range normalised 'heatmap'
    corresponding to a stockwell transform of timeseries data with transform 
    options defined by other paramater values.
    
    Inputs:
    data (list) - Numeric timeseries data.
    datalen (integer) - Length of data.
    sfreq (integer) - Sampling frequency of the data.
    width (float) - Gaussian window width (affects time/freq resolution).
    n_fft (integer) - Length of window for fast fourier transform.
    decim (integer) - Decimation factor on the time axis.
    lfreq (integer) - Lower frequency limit to be included.
    hfreq (integer) - Higher frequency limit to be included.
    imsize (integer) - Size of the output array, both height and width.
        
    Outputs:
    norm_power (array) - 0-1 normalised square shaped 'heatmap' array of the 
    data's stockwell transform resized to imsize x imsize.
    '''
    
    # Stockwell transform
    power,temp,temp = tfr_array_stockwell(data=data,sfreq=sfreq,width=width,
                                          n_fft=n_fft,fmin=lfreq,fmax=hfreq,
                                          decim=decim)
    resized_power = resize(power[0],(imsize,imsize)) # Resize to square array
    norm_power = resized_power / resized_power.max() # Normalise 0-1
    return norm_power

## Main =======================================================================

# Load config csv as numpy array of strings (ignore lines starting with #)
info_array = np.loadtxt(info_fp,'str','#',',')

# Select only one filetype (FOR,ACC,MIC)
ftype = ftypes[ftype_selected]

# Loop through filename stems from config/info file
for stem_idx, stem in enumerate(info_array[:,0]):
    
    n_section = int(stem[-2:]) # Section number from filestem
    
    # Sniplength of the base (FOR) case
    sniplen_base = int(info_array[stem_idx,1])
    # Lower, higher and range of wears for this file
    wear_lower = float(info_array[stem_idx,2])
    wear_higher = float(info_array[stem_idx,3])
    wear_range = wear_higher - wear_lower
        
    # Extract stockwell settings for the filetype selected
    stockwell_setts = info_array[stem_idx,offset+ftype_selected].split(';')
    sfreq = int(stockwell_setts[0])
    width = int(stockwell_setts[1])
    n_fft = int(stockwell_setts[2])
    decim = int(stockwell_setts[3])
    lfreq = int(stockwell_setts[4])
    hfreq = int(stockwell_setts[5])
    
    filename = stem + ftype # Assemble full filename
    data_fp = os.path.join(data_folder_fp,filename) # Assemble data filepath
    data = np.loadtxt(data_fp,'float','#',',') # Load data csv as numpy array
    # Split the data into individual signals and provide info about the data
    sdata,dtypes,dtype_main,nlength = split_info_data(data)
    
    # Calculate factor by which the sniplength needs to be mulitplied by
    # this depends on the sampling frequency of the data type in comparison
    # to the base (FOR) sampling frequency
    snip_mult = sfreq/sfreq_base;
    sniplen = sniplen_base * snip_mult # Apply to sniplength
    
    # Loop through signal types (e.g. FRX, FRY, FRZ)
    for type_idx, dtype in enumerate(dtypes):
        
        # Get signal by type and split it into fragments (clip remainder symm.)
        frags = get_signal_frags(sdata,dtype,sniplen,type_idx)
        
        # Loop through fragments
        for frag_idx, frag in enumerate(frags):
            
            if array or plot:
                # Complete stockwell transform, resize and normalise
                bitmap = stockwell_low_def(frag,nlength,sfreq,width,n_fft,
                                           decim,lfreq,hfreq,imsize)
            else:
                pass
            
            # Name/title of the array
            title = stem + 'c' + str(frag_idx) + '_' + dtype
            
            # If selected, save the bitmap as a csv file
            if array:
                filename = title + ".csv"
                out_fp = os.path.join(array_fp,filename)
                np.savetxt(out_fp, bitmap, fmt='%.18e', delimiter=',')
            else:
                pass
            
            # If selected and only once per stem, save the wear label
            if label and type_idx == 0:
                # Calculate current wear via linear interpolation
                wear_bot = wear_lower + wear_range*((n_section-1)/n_sections)
                wear_top = wear_lower + wear_range*(n_section/n_sections)
                wear = wear_bot + ( (wear_top - wear_bot)*\
                       (2*frag_idx-3)/(2*len(frags)) )
                if wear < 0:
                    # Handle floating point errors near 0
                    wear = 0
                else:
                    pass
                # Append wear labels to new/existing text file
                with open(label_fp,'a+') as f:
                    f.write(title[:-4] + "," + str(wear) + "\n")
            else:
                pass

            # If selected, save the plot (heatmap) as a png file
            if plot:
                plt.imshow(bitmap,cmap='seismic',interpolation='nearest',
                           origin='lower')
                plt.axis('off')
                plt.title(title)
                filename = title + ".png"
                out_fp = os.path.join(plot_fp,filename)
                plt.savefig(out_fp)
                plt.close()
            else:
                pass
            
        # Add completion information to log file
        with open(log_fp,'a+') as f:
            f.write("Completed"+","+stem+"_"+dtype+"\n")