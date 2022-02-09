from datetime import time
import numpy as np
import pandas as pd
import sys,os
from tqdm import tqdm
from glob import glob
import joblib
import argparse
import moseq2_extract.io.video as moseq_video
import pickle
import decord
from skimage import color
from cv2 import resize

import pdb

import plotting, extract_leds, sync

def basler_workflow(base_path, save_path, num_leds, led_blink_interval, led_loc, basler_chunk_size=1000, led_rois_from_file=False, overwrite_extraction=False):
    """
    Workflow to extract led codes from a Basler mp4 file.

    We know the LEDs only change once every (led_blink_interval), so we can just grab a few frames per interval. 
    
    """

    # Set up 
    basler_path = glob('%s/*.mp4' % base_path )[0]
    vr = decord.VideoReader(basler_path, ctx=decord.cpu(0), num_threads=2)
    num_frames = len(vr)
    timestamps = vr.get_frame_timestamp(np.arange(0,num_frames))  # blazing fast. nframes x 2 (beginning,end), take just beginning times
    print(f'Using video at {basler_path}')
    print('Assuming Basler recorded at 120 fps...')
    # timestamps = timestamps*2  # when basler records at 120 fps, timebase is halved :/

    ############### Cycle through the frame chunks to get all LED events: ###############
    
    # Prepare to load video (use frame batches)
    frame_batches = gen_batch_sequence(num_frames, basler_chunk_size, 0, offset=0)
    num_chunks = len(frame_batches)
    basler_led_events = []
    if led_rois_from_file:
        diff_thresholds, led_roi_list = load_led_rois_from_file(base_path)
    basler_led_events_path = os.path.join(base_path, 'basler_led_events.npz')
    print('num_chunks = ', num_chunks)

    # Do the loading
    if overwrite_extraction or (not os.path.isfile(basler_led_events_path)):

        for i in tqdm(range(num_chunks)[0:]):
            
            print(frame_batches[i])
            
            batch_timestamps = timestamps[frame_batches[i], 0]

            # New batch loading method: take only one channel, downsample with cv2
            frame_data_chunk = vr.get_batch(list(frame_batches[i])).asnumpy()
            frame_data_chunk = frame_data_chunk[:,:,:,0]  # basler records rgb but it's grayscale, all channels are the same. Pick one.
            new_shape = (int(frame_data_chunk.shape[1]/2), int(frame_data_chunk.shape[2]/2))
            new_frames = np.zeros((frame_data_chunk.shape[0], *new_shape), dtype='uint8')
            for iFrame,im in enumerate(np.arange(frame_data_chunk.shape[0])):
                new_frames[iFrame,:,:] = resize(frame_data_chunk[iFrame,:,:], new_shape[::-1])  # cv2 expects WH, decord (and everyone else) uses HW, hence ::-1
            frame_data_chunk = new_frames

            # Plot frame for user if first one
            if i==0:
                plotting.plot_video_frame(frame_data_chunk.std(axis=0), 600, '%s/basler_frame_std.pdf' % save_path)

            # Do the extraction
            if led_rois_from_file:                    
                leds = extract_leds.get_led_data_from_rois(frame_data_chunk=frame_data_chunk, 
                                                        led_roi_list=led_roi_list,
                                                        movie_type='basler',
                                                        diff_thresholds=diff_thresholds,
                                                        save_path=save_path)
            else:
                leds = extract_leds.get_led_data_with_stds( \
                                        frame_data_chunk=frame_data_chunk,
                                        movie_type='basler',
                                        num_leds=num_leds,
                                        chunk_num=i,
                                        led_loc=led_loc,
                                        save_path=save_path)

            # Save the events
            tmp_event = extract_leds.get_events(leds, batch_timestamps)
            if len(tmp_event) == 0:
                print('Warning: no events detected! Check led rois, or np.diff thresholds in extract_leds.get_events. Ignoring for now...')
            basler_led_events.append(tmp_event)

            # actual_led_nums = np.unique(tmp_event[:,1]) ## i.e. what was found in this chunk
            # if np.all(actual_led_nums == range(num_leds)):    
            # else:
            #     print('Found %d LEDs found in chunk %d. Skipping... ' % (len(actual_led_nums),i))

        basler_led_events = np.concatenate(basler_led_events)
        np.savez(basler_led_events_path, led_events=basler_led_events)
    else:
        basler_led_events = np.load(basler_led_events_path)['led_events']

    
    ############### Convert the LED events to bit codes ###############

    # Check if ISIs have all been shortened from 5 sec (happens when basler drops frames)
    isis = np.diff(basler_led_events[:,0])
    isis = isis[isis > 1]
    if np.mean(isis) < 4.5:
        print('Warning: mean inter-LED switch time less than 4.5 sec! Likely have dropped frames.')
        print('Changing minCodeTime param so that syncing will still work.')
        minCodeTime = 3
    else:
        minCodeTime = (led_blink_interval-1)

    basler_led_codes, latencies = sync.events_to_codes(basler_led_events, nchannels=4, minCodeTime=minCodeTime)  
    basler_led_codes = np.asarray(basler_led_codes)

    return basler_led_codes, timestamps[:,0]


### Basler HELPER FUNCTIONS ###

def load_led_rois_from_file(base_path):
    fin = os.path.join(base_path, 'led_rois.pickle')
    with open(fin, 'rb') as f:
        led_roi_dict = pickle.load(f)
    led_roi_list = led_roi_dict['roi_list']
    diff_thresholds = led_roi_dict['thresholds']
    return diff_thresholds, led_roi_list


def gen_batch_sequence(nframes, chunk_size, overlap, offset=0):
    '''
    Generates batches used to chunk videos prior to extraction.

    Parameters
    ----------
    nframes (int): total number of frames
    chunk_size (int): desired chunk size
    overlap (int): number of overlapping frames
    offset (int): frame offset

    Returns
    -------
    Yields list of batches
    '''

    seq = range(offset, nframes)
    out = []
    for i in range(0, len(seq) - overlap, chunk_size - overlap):
        out.append(seq[i:i + chunk_size])
    return out