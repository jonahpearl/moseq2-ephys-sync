from datetime import time
import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import subprocess
from glob import glob
import joblib
import argparse
import json

from mlinsights.mlmodel import PiecewiseRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer

import moseq2_extract.io.video as moseq_video

# from moseq2_ephys_sync.video import get_mkv_stream_names, get_mkv_info
# from moseq2_ephys_sync.extract_leds import gen_batch_sequence, get_led_data, get_events
# from moseq2_ephys_sync.plotting import plot_code_chunk, plot_matched_scatter, plot_model_errors, plot_matches_video_time,plot_video_frame

from . import mkv, arduino, ttl, sync, plotting

import pdb

"""
TODO:

-- create workflows for each source and separate from main function

-- remove video, replaced with functions in mkv
"""


def main_function(base_path, first_source, second_source, led_loc=None, led_blink_interval=5, arduino_spec=None):
    """
    Uses 4-bit code sequences to create a piecewise linear model to predict first_source times from second_source times
    ----
    Inputs:
        base_path (str): path to the .mkv and any other files needed
        first_source (str): 'ttl', 'mkv', 'arduino', or 'basler'. Source to be predicted.
            ttl: looks for open ephys data in __ format
            mkv: looks for an MKV file recorded with the k4a recorder
            arduino: looks for a text file with cols specified by arduino_col_type
            basler: looks for an mp4
        second_source (str): same as first_source, but these codes are used to predict first_source.
        led_loc (str): specifiy one of four corners of the movie in which to find the LEDs: topright, bottomright, topleft, bottomleft
        led_blink_interval (int): interval in seconds between LED changes. Typically 5 seconds.

    Outputs:
        -

    Notes:
        - Each workflow checks for already-pre-processed data, so that the script should be pretty easy to debug.
    """

    print(f'Running sync on {base_path} with {first_source} as first source and {second_source} as second source.')


    ## SETUP ##
    # Built-in params (should make dynamic)
    mkv_chunk_size = 2000
    num_leds = 4
    ephys_fs = 3e4  # sampling rate in Hz


    # Set up save path
    save_path = '%s/sync/' % base_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)



    ## INDIVIDUAL DATA STREAM WORKFLOWS ##

    # Deal with first source
    if first_source == 'mkv':
        first_source_led_codes = mkv.mkv_workflow(base_path, save_path, num_leds, led_blink_interval, mkv_chunk_size, led_loc)
    elif first_source == 'arduino':
        first_source_led_codes, ino_average_fs = arduino.arduino_workflow(base_path, save_path, num_leds, led_blink_interval, arduino_spec)
    


    # Deal with second source
    if second_source == 'ttl':
        second_source_led_codes = ttl.ttl_workflow(base_path, save_path, num_leds, led_blink_interval, ephys_fs)
        
        

    # Save the codes for use later
    np.savez('%s/codes.npz' % save_path, first_source_codes=first_source_led_codes, second_source_codes=second_source_led_codes)

    ## visualize a small chunk of the bit codes. do you see a match? 
    # Codes array should have times in seconds by this point
    plotting.plot_code_chunk(first_source_led_codes, second_source_led_codes, save_path)



    ## SYNCING :D ##

    # Returns two columns of matched event times
    matches = np.asarray(sync.match_codes(first_source_led_codes[:,0],  ## all times should be in seconds by here
                                  first_source_led_codes[:,1], 
                                  second_source_led_codes[:,0],
                                  second_source_led_codes[:,1],
                                  minMatch=10,maxErr=0,remove_duplicates=True ))

    
    ## plot the matched codes against each other:
    plotting.plot_matched_scatter(matches, save_path)

    ####################### Make the models! ####################

    # Rename for clarity.
    ground_truth_source1_event_times = matches[:,0]
    ground_truth_source2_event_times = matches[:,1]
    
    # Model first source from second soure, and vice versa.
    # I'm sure there's a cleaner way to do this, but it works for now.
    for i in range(2):
        if i == 0:
            s1 = ground_truth_source1_event_times
            t1 = first_source_led_codes
            n1 = first_source
            s2 = ground_truth_source2_event_times
            t2 = second_source_led_codes
            n2 = second_source
        elif i == 1:
            s1 = ground_truth_source2_event_times
            t1 = second_source_led_codes
            n1 = second_source
            s2 = ground_truth_source1_event_times
            t2 = first_source_led_codes
            n2 = first_source
        
        # Learn to predict s1 from s2. Syntax is fit(X,Y).
        mdl = PiecewiseRegressor(verbose=True,
                                binner=KBinsDiscretizer(n_bins=10))
        mdl.fit(s2.reshape(-1, 1), s1)

        # Verify accuracy of predicted event times
        predicted_event_times = mdl.predict(s2.reshape(-1, 1) )
        time_errors = predicted_event_times - s1 
        plotting.plot_model_errors(time_errors,save_path)

        # Verify accuracy of all predicted times
        all_predicted_times = mdl.predict(t2[:,0].reshape(-1, 1) )
        plotting.plot_matches_video_time(all_predicted_times, t2, t1, save_path)

        # Save
        joblib.dump(mdl, f'{save_path}/{n2}_timebase.p')
        print(f'Saved model that predicts {n1} from {n2}')


    print('Syncing complete. FIN')



if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str)  # path to data
    parser.add_argument('--first_source', type=str)  # ttl, mkv, arduino (txt) 
    parser.add_argument('--second_source', type=str)  # ttl, mkv, arduino 
    parser.add_argument('--led_loc', type=str)
    parser.add_argument('--led_blink_interval', type=int, default=5)  # default blink every 5 seconds
    parser.add_argument('--arduino_spec', type=str)  # specifiy cols in arduino text file

    settings = parser.parse_args(); 

    base_path = settings.path
    first_source = settings.first_source
    second_source = settings.second_source
    led_loc = settings.led_loc
    led_blink_interval = settings.led_blink_interval
    arduino_spec = settings.arduino_spec

    main_function(base_path, first_source, second_source, led_loc, led_blink_interval, arduino_spec)

    
