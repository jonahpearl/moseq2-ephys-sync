import numpy as np
import pandas as pd
import os
from os.path import join, exists
from glob import glob
import joblib
from mlinsights.mlmodel import PiecewiseRegressor
from sklearn.preprocessing import KBinsDiscretizer

from moseq2_ephys_sync import sync, viz, util, workflows

import pdb

def get_valid_source_abbrevs():
    return ['oe', 'mkv', 'arduino', 'txt', 'csv', 'basler', 'basler_bonsai', 'avi']
def process_source(source,
                    base_path=None,
                    save_path=None,
                    num_leds=None,
                    leds_to_use=None,
                    led_blink_interval=None, 
                    source_timescale_factor_log10=None,
                    led_loc=None,
                    led_rois_from_file=None,
                    overwrite_extraction=False,
                    arduino_spec=None
                    ):


    #TODO: remove save path from these args, doesn't do anything I dont think!

    if source == 'oe':
        source_led_codes, source_full_timestamps = workflows.oe_workflow(base_path, save_path, num_leds, leds_to_use, led_blink_interval, ephys_fs)

    elif source == 'mkv':
        assert not (led_loc and led_rois_from_file), "User cannot specify both MKV led location (top right, etc) and list of exact MKV LED ROIs!"
        assert '4' in leds_to_use, "LED extraction code expects that last LED is LED 4 (switching every interval)" 
        source_led_codes, source_full_timestamps = workflows.mkv_workflow(base_path, save_path, num_leds, led_blink_interval, mkv_chunk_size, led_loc, led_rois_from_file, overwrite_extraction)

    elif source == 'arduino' or source=='txt':
        source_led_codes, source_full_timestamps = workflows.arduino_workflow(base_path, num_leds, leds_to_use, led_blink_interval, arduino_spec=arduino_spec, source_timescale_factor_log10=source_timescale_factor_log10)

    elif source.endswith('.csv') or source.endswith('.txt'):
        source_led_codes, source_full_timestamps = workflows.arduino_workflow(source, num_leds, leds_to_use, led_blink_interval, arduino_spec=arduino_spec, source_timescale_factor_log10=source_timescale_factor_log10)

    elif source == 'basler':
        assert not (led_loc and led_rois_from_file), "User cannot specify both Basler led location (top right, etc) and list of exact Basler LED ROIs!"
        source_led_codes, source_full_timestamps = vid_workflows.basler_workflow(base_path, num_leds, led_blink_interval, led_loc, basler_chunk_size, led_rois_from_file, overwrite_extraction)

    elif source == 'basler_bonsai':
        source_led_codes, source_full_timestamps = workflows.basler_bonsai_workflow(base_path, num_leds, leds_to_use, led_blink_interval, source_timescale_factor_log10=source_timescale_factor_log10)

    elif source == 'avi':
        assert '4' in leds_to_use, "LED extraction code expects that last LED is LED 4 (switching every interval)" 
        source_led_codes, source_full_timestamps = avi.avi_workflow(base_path, save_path, num_leds=num_leds, led_blink_interval=led_blink_interval, led_loc=led_loc, avi_chunk_size=avi_chunk_size, overwrite_extraction=overwrite_extraction)
    

    return source_led_codes, source_full_timestamps


#TODO: figure out how to get in params like 
# ephys_fs=ephys_fs,
                    # mkv_chunk_size=mkv_chunk_size,
                    # basler_chunk_size=basler_chunk_size,
                    # avi_chunk_size=avi_chunk_size,

def sync_two_sources(matches,
 first_source_led_codes, 
 first_source_name, 
 first_source_full_timestamps, 
 second_source_led_codes, 
 second_source_name, 
 second_source_full_timestamps, 
 save_path,
 sources_to_predict=None,
 overwrite_models=False):
    
    if sources_to_predict is None:
        sources_to_predict = []

    # Rename for clarity.
    ground_truth_source1_event_times = matches[:,0]
    ground_truth_source2_event_times = matches[:,1]
    
    # Model first source from second soure, and vice versa.
    # I'm sure there's a cleaner way to do this, but it works for now.
    # s1 and s2 match in shape, and represent matched timestamps.
    # t1 and t2 don't match in shape, and represent all codes detected in each channel.
    for i in range(2):
        if i == 0:
            s1 = ground_truth_source1_event_times
            t1 = first_source_led_codes
            n1 = first_source_name
            full1 = first_source_full_timestamps
            s2 = ground_truth_source2_event_times
            t2 = second_source_led_codes
            n2 = second_source_name
            full2 = second_source_full_timestamps
    
        elif i == 1:
            s1 = ground_truth_source2_event_times
            t1 = second_source_led_codes
            n1 = second_source_name
            full1 = first_source_full_timestamps
            s2 = ground_truth_source1_event_times
            t2 = first_source_led_codes
            n2 = first_source_name
            full2 = first_source_full_timestamps
    
        # Learn to predict s1 from s2. Syntax is fit(X,Y).
        mdl = PiecewiseRegressor(verbose=True,
                                binner=KBinsDiscretizer(n_bins=4))
        mdl.fit(s2.reshape(-1, 1), s1)
    
        outname = f'{n1}_from_{n2}'
    
        # Verify accuracy of predicted event times
        predicted_event_times = mdl.predict(s2.reshape(-1, 1))
        time_errors = predicted_event_times - s1 
        viz.plot_model_errors(time_errors, save_path, outname)
    
        # Plot all predicted times
        all_predicted_times = mdl.predict(t2[:,0].reshape(-1, 1))  # t1-timebase times of t2 codes (predict t1 from t2)
        viz.plot_matched_times(all_predicted_times, t2, t1, n1, n2, save_path, outname)
    
        # Save
        fname = join(save_path,f'{outname}.p')
        if exists(fname) and not overwrite_models:
            print(f'Model that predicts {n1} from {n2} already exists, not saving...')
        else:
            joblib.dump(mdl, fname)
            print(f'Saved model that predicts {n1} from {n2}')
    
        # Compute and save the full synced timestamps.
        # Eg: if we're predicting timestamps for oe from txt, it will return a list of times of length (num times in txt file), where each entry is the corresponding time in the ephys file
        # I would recommend not predicting timestamps from oe, as it will be ~ 1GB.
        if str(i+1) in sources_to_predict:
            fout = join(save_path,f'{outname}_fullTimes.npy')
            if not exists(fout) or overwrite_models:
                print(f'Computing full synced timestamp list for {n1} from {n2} (this may take a while...)')
                full_predicted_s1 = mdl.predict(full2.reshape(-1,1))
                np.save(fout, full_predicted_s1)
            else:
                print(f'Full synced timestamp list for {n1} from {n2} already exists, continuing...')

    return 


def main_function(base_path,
first_source,
second_source,
output_dir_name='sync',
led_loc=None, 
led_blink_interval=5, 
s1_timescale_factor_log10=None,
s2_timescale_factor_log10=None,
arduino_spec=None, 
s1_led_rois_from_file=False,
s2_led_rois_from_file=False, 
overwrite_models=False,
overwrite_extraction=False,
leds_to_use=[1,2,3,4],
sources_to_predict=None):

    """
    Uses 4-bit code sequences to create a piecewise linear model to predict first_source times from second_source times
    ----
    Inputs:
        base_path (str): path to the .mkv and any other files needed
        output_dir: path to save output models and plots. Default: {base_path}/sync.
        first_source (str): 'oe', 'mkv', 'arduino', or 'basler'. Source to be predicted.
            oe: looks for open ephys data in __ format
            mkv: looks for an MKV file recorded with the k4a recorder
            arduino: looks for a text file with cols specified by arduino_col_type
            basler: looks for an mp4
        second_source (str): same as first_source, but these codes are used to predict first_source.
        led_loc (str): MKV only! specifiy one of four corners of the movie in which to find the LEDs: topright, bottomright, topleft, bottomleft
        led_blink_interval (int): interval in seconds between LED changes. Typically 5 seconds.
        s1_led_rois_from_file, s2_led_rois_from_file (bool): whether to look in base_path for led roi pickle.
        overwrite_mkv_extraction (bool): whether to re-do the MKV LED extraction
    Outputs:
        -

    Notes:
        - Each workflow checks for already-pre-processed data, so that the script should be pretty easy to debug.
        - Basler code expects an mp4 at 120 fps. If you use 60 fps, probably need to change the minCodeTime arg in line 80 of basler.py.
    """

    """
    TODO: 
    -- make basler timebase dynamic (or auto detect??)
    -- allow passing each source's time units, eg seconds vs microseconds
    -- if using a video source, import the video modules!!
    """
    print(f'Running sync on {base_path} with {first_source} as first source and {second_source} as second source.')

    #### SETUP ####

    if sources_to_predict is None:
        sources_to_predict = []

    # Detect num leds
    num_leds = len(leds_to_use)

    # Set up save path
    save_path = join(base_path, output_dir_name)
    if not exists(save_path):
        os.makedirs(save_path)

    # Detect whether the user passed paths to source files, or used abbreviations
    first_source_name, second_source_name = util.verify_sources(first_source, second_source)

    if (first_source_name in ['arduino', 'txt']) and (second_source_name in ['arduino', 'txt']):
        raise ValueError('Cannot pass arduino/txt for both sources, as this is ambiguous. Use explicit paths to both files.')

    # Check if models already exist, only over-write if requested
    model_exists_bool = exists(join(save_path, f'{first_source_name}_from_{second_source_name}.p')) or \
                        exists(join(save_path, f'{second_source_name}_from_{first_source_name}.p'))

    if model_exists_bool and not overwrite_models:
        raise ValueError("One or both models already exist and overwrite_models is false!")

    print('Dealing with first souce...')
    # first_source_led_codes: array of reconstructed pixel clock codes where: codes[:,0] = time, codes[:,1] = code (and codes[:,2] = trigger channel but that's not used in this code)
    # first_source_full_timestamps: full list of timestamps from source 1 (every timestamp, not just event times! For prediction with the model at the end.)
    first_source_led_codes, first_source_full_timestamps = \
    process_source(first_source,
                    base_path=base_path,
                    save_path=save_path, num_leds=num_leds,
                    leds_to_use=leds_to_use,
                    led_blink_interval=led_blink_interval, 
                    source_timescale_factor_log10=s1_timescale_factor_log10,
                    led_loc=led_loc,
                    led_rois_from_file=s1_led_rois_from_file,
                    overwrite_extraction=overwrite_extraction,
                    arduino_spec=arduino_spec
                    )

    print('Dealing with second souce...')
    second_source_led_codes, second_source_full_timestamps = \
    process_source(second_source,
                    base_path=base_path,
                    save_path=save_path, num_leds=num_leds,
                    leds_to_use=leds_to_use,
                    led_blink_interval=led_blink_interval, 
                    source_timescale_factor_log10=s2_timescale_factor_log10,
                    led_loc=led_loc,
                    led_rois_from_file=s2_led_rois_from_file,
                    overwrite_extraction=overwrite_extraction,
                    arduino_spec=arduino_spec
                    )
   


    # Sanity check on timestamps being in seconds
    first_source_full_timestamps = np.array(first_source_full_timestamps)
    assert (first_source_full_timestamps[-1] - first_source_full_timestamps[0]) < 7200, f"Your timestamps for {first_source} appear to span more than two hours...are you sure the timestamps are in seconds?"

    second_source_full_timestamps = np.array(second_source_full_timestamps)
    assert (second_source_full_timestamps[-1] - second_source_full_timestamps[0]) < 7200, f"Your timestamps for {second_source} appear to span more than two hours...are you sure the timestamps are in seconds?"

    # Save the codes for use later
    np.savez(f'{save_path}/codes_{first_source_name}_and_{second_source_name}.npz', first_source_codes=first_source_led_codes, second_source_codes=second_source_led_codes)

    # Visualize a small chunk of the bit codes. do you see a match? 
    # Codes array should have times in seconds by this point
    viz.plot_code_chunk(first_source_led_codes, first_source_name, second_source_led_codes, second_source_name, save_path)


    #### SYNCING :D ####
    print('Syncing the two sources...')
    # Returns two columns of matched event times. All times must be in seconds by here
    matches = np.asarray(sync.match_codes(first_source_led_codes[:,0],  
                                  first_source_led_codes[:,1], 
                                  second_source_led_codes[:,0],
                                  second_source_led_codes[:,1],
                                  minMatch=10,maxErr=0,remove_duplicates=True ))

    assert len(matches) > 0, 'No matches found -- if using a movie, double check LED extractions and correct assignment of LED order'

    ## Plot the matched codes against each other:
    viz.plot_matched_scatter(matches, first_source_name, second_source_name, save_path)

    #### Make the models! ####
    print('Modeling the two sources from each other...')
    sync_two_sources(matches, first_source_led_codes, first_source_name, first_source_full_timestamps, second_source_led_codes, second_source_name, second_source_full_timestamps, save_path, sources_to_predict, overwrite_models)

    print('Syncing complete. FIN')


def load_oe_data(base_path):
    ephys_ttl_path = glob(join(base_path, '**', 'TTL_*/'), recursive = True)[0]
    channels = np.load(join(ephys_ttl_path, 'channel_states.npy'))
    ephys_TTL_timestamps = np.load(join(ephys_ttl_path, 'timestamps.npy'))  # these are in sample number
    return channels, ephys_TTL_timestamps


def oe_workflow(base_path, num_leds, leds_to_use, led_blink_interval, ephys_fs=3e4):
    """
    
    """
    # assert num_leds==4, "TTL code expects 4 LED channels, other nums of channels not yet supported"
    if num_leds != len(leds_to_use):
        raise ValueError('Num leds must match length of leds to use!')

    # Load the TTL data
    channels, ephys_TTL_timestamps = load_oe_data(base_path)

    # Need to subtract the raw traces' starting timestamp from the TTL timestamps
    # (This is a bit of a glitch in open ephys / spike interface, might be able to remove this in future versions)
    continuous_timestamps_path = glob(join(base_path, '**', 'continuous', '**', 'timestamps.npy'), recursive = True)[0] ## load the continuous stream's timestamps
    continuous_timestamps = np.load(continuous_timestamps_path)
    ephys_TTL_timestamps -= continuous_timestamps[0]   # subract the first timestamp from all TTLs; this way continuous ephys can safely start at 0 samples or seconds
    ephys_TTL_timestamps = ephys_TTL_timestamps / ephys_fs
    continuous_timestamps = continuous_timestamps / ephys_fs

    ttl_channels = [int(i)*sign for i in leds_to_use for sign in [-1,1]]

    ttl_bool = np.isin(channels, ttl_channels)
    ephys_events = np.vstack([ephys_TTL_timestamps[ttl_bool], abs(channels[ttl_bool])-1, np.sign(channels[ttl_bool])]).T
    codes, ephys_latencies = sync.events_to_codes(ephys_events, nchannels=num_leds, minCodeTime=(led_blink_interval-1))
    codes = np.asarray(codes)

    return codes, continuous_timestamps



def arduino_workflow(base_path,
num_leds, 
leds_to_use, 
led_blink_interval, 
arduino_spec=None, 
timestamp_jump_skip_event_threshhold=0.1, 
file_glob='*.txt',
source_timescale_factor_log10=None):
    """
    Workflow to get codes from arduino txt file. Note arduino sampling rate is calculated empirically below because it's not stable from datapoint to datapoint.
    
    Inputs:
        base_path (str): path to the .txt file
        num_leds (int): expects 4
        leds_to_use (list): eg ['1', '2', '3', '4']. Must contain 4.
        led_blink_interval: sets an upper bound on how fast the sync code can change. Useful for noisy vids.
        arduino_spec (str): Specifies what the column names should be in the data that gets read in. Current options are "fictive_olfaction" or "odor_on_wheel", which are interpreted below.
        timestamp_jump_skip_event_threshhold (int): if there is a jump in timestamps larger than this, skip any artifactual "event" that might arise because of it (in sec).
    """
    print('Doing arduino workflow...')

    if num_leds != len(leds_to_use):
        raise ValueError('Num leds must match length of leds to use!')

    if source_timescale_factor_log10 is None:
        source_timescale_factor_log10 = 3

    if arduino_spec: 
        arduino_colnames, arduino_dtypes = util.get_col_info(arduino_spec)
        ino_data = load_arduino_data(base_path, arduino_colnames, arduino_dtypes, file_glob=file_glob)
    else:
        ino_data = load_arduino_data(base_path, file_glob=file_glob)
    ino_timestamps = ino_data.time / (10**source_timescale_factor_log10)  # these are in milliseconds, convert to seconds



    # led_names = ['led1', 'led2', 'led3', 'led4']
    led_names = [colname for colname in ino_data.columns if "led" in colname]
    led_names.sort()  # in case user has manually edited order of leds in txt file due to experimental mistake
    led_list = []
    for idx in leds_to_use:
        led_list.append(ino_data[led_names[int(idx) - 1]])

    
    ino_events = list_to_events(ino_timestamps, led_list, tskip=timestamp_jump_skip_event_threshhold)
    ino_codes, _ = sync.events_to_codes(ino_events, nchannels=num_leds, minCodeTime=(led_blink_interval-1))  # I think as long as the column 'timestamps' in events and the minCodeTime are in the same units, it's fine (for ephys, its nsamples, for arudino, it's ms)
    ino_codes = np.asarray(ino_codes)

    return ino_codes, ino_timestamps 


def basler_bonsai_workflow(base_path,
    num_leds,
    leds_to_use,
    led_blink_interval, 
    timestamp_jump_skip_event_threshhold=0.1, 
    file_glob='basler*.csv',
    source_timescale_factor_log10=None):
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
    
    if num_leds != len(leds_to_use):
        raise ValueError('Num leds must match length of leds to use!')
    
    if source_timescale_factor_log10 is None:
        source_timescale_factor_log10 = 9

    txt_data = load_arduino_data(base_path, file_glob=file_glob)
    bonsai_timestamps = txt_data.time / (10**source_timescale_factor_log10)  # these are in NANOseconds, convert to seconds

    led_names = ['led1', 'led2', 'led3', 'led4']
    led_list = []
    for idx in leds_to_use:
        led_list.append(txt_data[led_names[int(idx) - 1]])

    bonsai_events = list_to_events(bonsai_timestamps, led_list, tskip=timestamp_jump_skip_event_threshhold)
    bonsai_codes, _ = sync.events_to_codes(bonsai_events, nchannels=num_leds, minCodeTime=led_blink_interval-1) 
    bonsai_codes = np.asarray(bonsai_codes)

    return bonsai_codes, bonsai_timestamps


def load_arduino_data(base_path, file_glob='*.txt'):

    # Define header data types
    # Do not use unsigned integers!! Otherwise np.diff() will not be able to return negatives.
    colnames_to_use = ['time', 'frame', 'led1', 'led2', 'led3', 'led4']  # only cols needed for syncing
    header_val_dtypes = {
        'time': 'float64',  # all times in initially in whole numbers of ms, but use float so that when convert to ms, you don't lose precision
        'frame': 'int64',
        'led1': 'int8',
        'led2': 'int8',
        'led3': 'int8',
        'led4': 'int8',
    }

    # Find file
    if base_path.endswith('.csv') or base_path.endswith('.txt'):
        arduino_data_path = base_path
    else:
        arduino_data_path = util.find_file_through_glob_and_symlink(base_path, file_glob)
        
    # Check if header is present
    with open(arduino_data_path, 'r') as f:
        first_row = f.readline().strip('\r\n').split(',')
    if first_row[0] == 'time':
        colnames = first_row
        print('Found header in arduino file, using...')
    else:
        raise ValueError('Expected header in csv to begin with "time", or did not find header')

    colnames_to_use = [col for col in colnames_to_use if col in colnames]
    dtype_dict = {col: header_val_dtypes[col] for col in colnames_to_use}
    data = pd.read_csv(arduino_data_path, header=0, dtype=dtype_dict, index_col=False, usecols=colnames_to_use)  # header=0 means first row
    
    return data


def list_to_events(time_list, led_states, tskip):
    """
    Transforms list of times and led states into list of led change events.
    ---
    Input: pd.Series from arduino text file
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
