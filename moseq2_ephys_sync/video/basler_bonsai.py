# updated basler workflow, using bonsai to sync instead of LEDs. Will look much more similar to the arduino workflow.

import pandas as pd
import numpy as np
from glob import glob
import pdb

import moseq2_ephys_sync.sync as sync
import moseq2_ephys_sync.util as util


def basler_bonsai_workflow(base_path, save_path, num_leds, leds_to_use, led_blink_interval, bonsai_spec='header', timestamp_jump_skip_event_threshhold=1):
    """
    Workflow to get codes from bonsai outputted txt file. 

    Bonsai is a "reactive" language, meaning it generates output only when it gets input.
    In this case, the input that it's yoked to is the Basler camera, recording at ~120 Hz (with 1 ms exposures).
    So, there should be as many rows in the text file as there are frames in the movie. If this is true, everything is easy --
    just assign each frame the corresponding timestamp, and sync the timestamps.
    The timestamps are in microseconds by default.
    
    Inputs:
        base_path (str): path to the .txt file
        num_leds (int): expects 4
        leds_to_use (list): eg ['1', '2', '3', '4']. Must contain 4.
        led_blink_interval: sets an upper bound (in seconds) on how fast the sync code can change. Useful for noisy vids.
        bonsai_spec (str): Specifies what the column names should be in the data that gets read in. Current option is "default" (time, img num, led1, led2, led3, led4) or "header" (use header in file)
        timestamp_jump_skip_event_threshhold (int): if there is a jump in timestamps larger than this (in seconds), skip any artifactual "event" that might arise because of it.
    """
    print('Doing bonsai basler workflow...')
    assert num_leds == len(leds_to_use)
    txt_colnames, txt_dtypes = get_col_info(bonsai_spec)
    txt_data = load_txt_data(base_path, txt_colnames, txt_dtypes, file_glob='*.csv')
    bonsai_timestamps = txt_data.time / 1e9  # these are in MICROseconds, convert to seconds

    led_names = ['led1', 'led2', 'led3', 'led4']
    led_list = []
    for idx in leds_to_use:
        led_list.append(txt_data[led_names[int(idx) - 1]])

    bonsai_events = list_to_events(bonsai_timestamps, led_list, tskip=timestamp_jump_skip_event_threshhold)
    bonsai_codes, _ = sync.events_to_codes(bonsai_events, nchannels=num_leds, minCodeTime=led_blink_interval-1) 
    bonsai_codes = np.asarray(bonsai_codes)
    bonsai_codes[:,0] = bonsai_codes[:,0]

    return bonsai_codes, bonsai_timestamps





def get_col_info(spec):
    """
    Given a string specifying the experiment type, return expected list of columns
    """
    if spec == 'default':
        colnames = ['time', 'led1', 'led2', 'led3', 'led4', 'yaw', 'roll', 'pitch', 'accx', 'accy', 'accz', 'therm', 'olfled']
        dtypes = ['int64', 'int64', 'int64', 'int64','int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int32', 'uint8']
    
    elif spec == 'header':  # headers in txt files
        colnames = None
        dtypes = None

    else:
        raise ValueError('Did not recognize requested bonsai txt file spec')

    return colnames, dtypes


def load_txt_data(base_path, colnames, dtypes, file_glob='*.txt'):

    # Define header data types
    # Do not use unsigned integers!! Otherwise np.diff() will not be able to return negatives.
    header_val_dtypes = {
        'time': 'int64',
        'img_num': 'int64',
        'led1': 'int8',
        'led2': 'int8',
        'led3': 'int8',
        'led4': 'int8',
    }

    # Find file
    data_path = util.find_file_through_glob_and_symlink(base_path, file_glob)
        
    # Check if header is present
    with open(data_path, 'r') as f:
        first_row = f.readline().strip('\r\n').split(',')
    if first_row[0] == 'time':
        header = 1
        colnames = first_row
        print('Found header in bonsai txt file, using...')
    else:
        header = 0

    if header:
        dtype_dict = {col: header_val_dtypes[col] for col in colnames}
        data = pd.read_csv(data_path, header=0, dtype=dtype_dict, index_col=False)  # header=0 means first row
    else:
        dtype_dict = {colname: dtype for colname, dtype in zip(colnames, dtypes)}
        try:
            # Try loading the entire thing first. 
            data = pd.read_csv(data_path, header=0, names=colnames, dtype=dtype_dict, index_col=False)
        except ValueError:
            try:
                # If needed, try ignoring the last line. This is slower so we don't use as default.
                data = pd.read_csv(data_path, header=0, names=colnames, dtype=dtype_dict, index_col=False, warn_bad_lines=True, skipfooter=1)
            except:
                raise RuntimeError('Could not load bonsai basler data -- check text file for weirdness. \
                Most common issues text file issues are: \
                -- line that ends with a "-" (minus sign), "." (decima) \
                -- line that begins with a "," (comma) \
                -- usually no more than one issue like this per txt file')
    return data


def list_to_events(time_list, led_states, tskip):
    """
    Transforms list of times and led states into list of led change events.
    ---
    Input: pd.Series from text file
        tskip (int): if there is a jump in timestamps larger than this, skip any artifactual "event" that might arise because of it.
    ---
    Output: 
    events : 2d array
        Array of pixel clock events (single channel transitions) where:
            events[:,0] = times
            events[:,1] = channels (0-indexed)
            events[:,2] = directions (1 or -1)
    """

    # Check for timestamp skips
    time_diffs = np.diff(time_list)
    skip_list = np.asarray(time_diffs >= tskip).nonzero()[0] + 1

    # Get lists of relevant times and events
    times = pd.Series(dtype='int64', name='times')
    channels = pd.Series(dtype='int8', name='channels')
    directions = pd.Series(dtype='int8', name='directions')
    for i in range(len(led_states)):
        states = led_states[i]  # list of 0s and 1s for this given LED
        assert states.shape[0] == time_list.shape[0]
        diffs = np.diff(states)
        events_idx = np.asarray(diffs != 0).nonzero()[0] + 1  # plus 1, because the event should be the first timepoint where it's different
        events_idx = events_idx[~np.isin(events_idx, skip_list)]  # remove any large time skips because they're not guaranteed to be synchronized
        times = times.append(pd.Series(time_list[events_idx], name='times'), ignore_index=True)
        channels = channels.append(pd.Series(np.repeat(i,len(events_idx)), name='channels'), ignore_index=True)
        directions = directions.append(pd.Series(np.sign(diffs[events_idx-1]), name='directions'), ignore_index=True)
    events = pd.concat([times, channels, directions], axis=1)
    sorting = np.argsort(events.loc[:,'times'])
    events = events.loc[sorting, :]
    assert np.all(np.diff(events.times)>=0), 'Event times are not sorted!'
    return np.array(events)