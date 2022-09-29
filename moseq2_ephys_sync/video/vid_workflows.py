from datetime import time
import numpy as np
import pandas as pd
import sys,os
from tqdm import tqdm
from glob import glob
import joblib
import argparse
import pickle
import decord
import imageio
from skimage import color
from cv2 import resize

from . import vid_util, extract_leds

import moseq2_ephys_sync.viz as viz
# import moseq2_ephys_sync.video.extract_leds as extract_leds
import moseq2_ephys_sync.sync as sync


def avi_workflow(base_path, save_path, num_leds=4, led_blink_interval=5000, led_loc=None, avi_chunk_size=2000, overwrite_extraction=False):

    
    # Set up paths
    ir_path = util.find_file_through_glob_and_symlink(base_path, '*ir.avi')
    timestamp_path = util.find_file_through_glob_and_symlink(base_path, '*device_timestamps.npy')
    
    # Load timestamps
    timestamps = np.load(timestamp_path)

    ############### Cycle through the frame chunks to get all LED events: ###############    
    # Prepare to load video using imageio

    # get frame size (must be better way lol)
    vid = imageio.get_reader(ir_path)
    for frame in vid:
        fsize = frame.shape  # nrows ncols nchannels
        break

    vid = imageio.get_reader(ir_path, pixelformat='gray8', dtype='uint8')
    nframes = vid.count_frames()
    assert timestamps.shape[0] == nframes
    frame_batches = gen_batch_sequence(nframes, avi_chunk_size, overlap=0, offset=0)
    num_chunks = len(frame_batches)
    avi_led_events = []
    print(f'num_chunks = {num_chunks}')

    avi_led_events_path = '%s_led_events.npz' % os.path.splitext(ir_path)[0]

    # If data not already extracted, load and process
    if not os.path.isfile(avi_led_events_path) or overwrite_extraction:
        print('Loading and processing avi frames...')
        for i in tqdm(range(num_chunks)[0:]):

            # Load frames in chunk
            frame_data_chunk = np.zeros((len(frame_batches[i]), fsize[0], fsize[1]))
            
            for j, frame_num in enumerate(frame_batches[i]):
                frame = vid.get_data(frame_num) 
                if j == 0:
                    assert np.all(frame[:,:,0]==frame[:,:,1])
                    assert np.all(frame[:,:,0]==frame[:,:,2])
                frame_data_chunk[j,:,:] = frame[:,:,0]
            
            # Display std for debugging
            if i==0:
                viz.plot_video_frame(frame_data_chunk.std(axis=0), 600, '%s/frame_std.png' % save_path)

            # Find LED ROIs
            leds = extract_leds.get_led_data_with_stds( \
                                        frame_data_chunk=frame_data_chunk,
                                        movie_type='avi',
                                        num_leds=num_leds,
                                        chunk_num=i,
                                        led_loc=led_loc,
                                        save_path=save_path)

            # Extract events and append to event list
            tmp_event = extract_leds.get_events(leds,timestamps[frame_batches[i]])
            actual_led_nums = np.unique(tmp_event[:,1]) ## i.e. what was found in this chunk
            if np.all(actual_led_nums == range(num_leds)):
                avi_led_events.append(tmp_event)
            else:
                print('%d LEDs returned in chunk %d. Skipping... (check ROIs, thresholding)' % (len(actual_led_nums),i)) 
            
        avi_led_events = np.concatenate(avi_led_events)

        ## optional: save the events for further use
        np.savez(avi_led_events_path, led_events=avi_led_events)
        print('Successfullly extracted avi leds, converting to codes...')    

    else:
        avi_led_events = np.load(avi_led_events_path)['led_events']
        print('Using saved led events')
    
    ############### Convert the LED events to bit codes ############### 
    avi_led_events[:,0] = avi_led_events[:, 0] / 1e6  # convert to sec (caleb's timestamps in microseconds!)
    avi_led_codes, latencies = sync.events_to_codes(avi_led_events, nchannels=num_leds, minCodeTime=(led_blink_interval-1))
    avi_led_codes = np.asarray(avi_led_codes)
    print('Converted.')

    return avi_led_codes, timestamps/1e6



def mkv_workflow(base_path, save_path, num_leds, led_blink_interval, mkv_chunk_size=2000, led_loc=None, led_rois_from_file=False, overwrite_mkv_extraction=False):
    """
    Workflow to extract led codes from an MKV file
    
    """

    # Set up paths
    depth_path = glob('%s/*.mkv' % base_path )[0]
    stream_names = get_mkv_stream_names(depth_path) # e.g. {'DEPTH': 0, 'IR': 1}
    info_path = '%s/info.json' % base_path  # make paths for info and timestamps. if they exist, don't recompute:
    timestamp_path = '%s/mkv_timestamps.csv' % base_path 


    # Load timestamps and mkv info if exist, otherwise calculate
    if (os.path.exists(info_path) and os.path.exists(timestamp_path) ):
        print('Loading preexisting mkv timestamps...')

        with open(info_path,'r') as f:
            info = json.load(f)

        timestamps = pd.read_csv(timestamp_path)
        timestamps = timestamps.values[:,1].flatten()

    else:
        print('Loading mkv timestamps de novo...')
        ## get info on the depth file; we'll use this to see how many frames we have
        info,timestamps = get_mkv_info(depth_path,stream=stream_names['DEPTH'])  # timestamps already in seconds

        ## save info and timestamps:
        timestamps = pd.DataFrame(timestamps)
        timestamps.to_csv(timestamp_path) # save the timestamps
        timestamps = timestamps.values.flatten()
        
        with open(info_path, 'w') as f:
            json.dump(info, f)

    # Debugging
    # print('info = ', info)
    # print('timestamps.shape = ', timestamps.shape)

    
    ############### Cycle through the frame chunks to get all LED events: ###############
    
    # Prepare to load video (use frame batches like in moseq2-extract)
    frame_batches = gen_batch_sequence(info['nframes'], mkv_chunk_size,
                                           0, offset=0)
    num_chunks = len(frame_batches)
    mkv_led_events = []
    print('num_chunks = ', num_chunks)

    if led_rois_from_file:
        led_roi_list = load_led_rois_from_file(base_path)

    mkv_led_events_path = '%s/led_events.npz' % save_path

    # Do the loading
    if not os.path.isfile(mkv_led_events_path) or overwrite_mkv_extraction:

        for i in tqdm(range(num_chunks)[0:]):
        # for i in [45]:
            
            frame_data_chunk = moseq_video.load_movie_data(depth_path,  # nframes, nrows, ncols
                                           frames=frame_batches[i],
                                           mapping=stream_names['IR'], movie_dtype=">u2", pixel_format="gray16be",
                                          frame_size=info['dims'],timestamps=timestamps,threads=8,
                                                          finfo=info)

            if i==0:
                viz.plot_video_frame(frame_data_chunk.std(axis=0), 600, '%s/frame_std.pdf' % save_path)

            if led_rois_from_file:
                leds = extract_leds.get_led_data_from_rois(frame_data_chunk=frame_data_chunk, led_roi_list=led_roi_list, save_path=save_path)
            else:
                leds = extract_leds.get_led_data_with_stds( \
                                        frame_data_chunk=frame_data_chunk,
                                        movie_type='mkv',
                                        num_leds=num_leds,
                                        chunk_num=i,
                                        led_loc=led_loc,
                                        save_path=save_path)
            
            # time_offset = frame_batches[i][0] ## how many frames away from first chunk's  #### frame_chunks[0,i]
            
            tmp_event = extract_leds.get_events(leds,timestamps[frame_batches[i]])

            actual_led_nums = np.unique(tmp_event[:,1]) ## i.e. what was found in this chunk


            if np.all(actual_led_nums == range(num_leds)):
                mkv_led_events.append(tmp_event)
            else:
                print('Found %d LEDs found in chunk %d. Skipping... ' % (len(actual_led_nums),i))
                
                
            
        mkv_led_events = np.concatenate(mkv_led_events)

        ## optional: save the events for further use
        np.savez(mkv_led_events_path,led_events=mkv_led_events)
    
    else:
        mkv_led_events = np.load(mkv_led_events_path)['led_events']

    print('Successfullly extracted mkv leds, converting to codes...')    


    ############### Convert the LED events to bit codes ############### 
    mkv_led_codes, latencies = sync.events_to_codes(mkv_led_events, nchannels=num_leds, minCodeTime=(led_blink_interval-1))
    mkv_led_codes = np.asarray(mkv_led_codes)
    print('Converted.')

    return mkv_led_codes, timestamps


def basler_workflow(base_path, save_path, num_leds, led_blink_interval, led_loc, basler_chunk_size=3000, led_rois_from_file=False, overwrite_extraction=False):
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
    frame_batches = vid_util.gen_batch_sequence(num_frames, basler_chunk_size, 0, offset=0)
    num_chunks = len(frame_batches)
    basler_led_events = []
    if led_rois_from_file:
        diff_thresholds, led_roi_list = vid_util.load_led_rois_from_file(base_path)
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
                viz.plot_video_frame(frame_data_chunk.std(axis=0), 600, '%s/basler_frame_std.pdf' % save_path)

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