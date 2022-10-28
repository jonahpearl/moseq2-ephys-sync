import numpy as np
import pandas as pd
from os.path import join, exists
import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from glob import glob
import decord
import imageio
from cv2 import resize
from itertools import repeat

from moseq2_ephys_sync.video import vid_util, extract_leds, vid_io
from moseq2_ephys_sync import viz, sync, util

import pdb

def batch_var_summer(frame_batch, ir_path, reporter_val, output_name, overwrite=False, downsample=None):
    
    # Just load the file if already calculated
    if exists(output_name) and not overwrite:
        variance = np.load(output_name)
        return variance
    
    # Otherwise run the parallel processing
    agg = (0, 0, 0)
    with vid_io.videoReader(ir_path, np.array(frame_batch), reporter_val) as vid:
        for frame in vid:
            agg = vid_util.update_running_var(agg, frame)
    mean, variance = vid_util.finalize_running_var(agg)
    np.save(output_name, variance)

    return variance

def net_frame_std_parallel(ir_path, save_path, frame_chunksize=1000, overwrite_extraction=False):
    net_std_name = join(save_path, 'batch_variances', 'net_std.npy')
    # if exists(net_std_name) and not overwrite_extraction:
        # return np.load(net_std_name)
    if exists(net_std_name):
        return np.load(net_std_name)

    # Prep for parallel proc
    nframes = vid_io.count_frames(ir_path)
    batch_seq = vid_util.gen_batch_sequence(nframes, frame_chunksize, 0)
    out_path = join(save_path, 'batch_variances')
    if not exists(out_path):
        os.makedirs(out_path)
    out_names = [f'batch_var_img_{i}.npy' for i in range(len(batch_seq))]
    parallel_output_names = [join(out_path, out_name) for out_name in out_names]
    # reporter_vals = (i for i in range(len(batch_seq)))  # debugging
    reporter_vals = (None for i in range(len(batch_seq)))

    # Do the parallel proc
    batch_variances = process_map(batch_var_summer, batch_seq, repeat(ir_path), reporter_vals, parallel_output_names, repeat(overwrite_extraction), chunksize=1)
    net_std = np.sqrt(np.mean(batch_variances, axis=0))  # could replace this with proper variance combination but eh
    viz.plot_video_frame(net_std, 600, join(save_path, 'batch_variances', 'net_std.png'))
    np.save(net_std_name, net_std)

    return net_std
    
def extract_led_events_parallel(ir_path, save_path, frame_chunksize, labeled_led_img, led_labels, led_sorting, led_blink_interval, overwrite_extraction=False):
    # Prep for parallel proc
    nframes = vid_io.count_frames(ir_path)
    batch_seq = vid_util.gen_batch_sequence(nframes, frame_chunksize, 0)
    reporter_vals = (None for i in range(len(batch_seq)))
    out_path = join(save_path, 'avi_led_signals.npy')
    if exists(out_path) and not overwrite_extraction:
        led_signals = np.load(out_path)
        return led_signals
    
    # Do the parallel proc
    led_signals = process_map(extract_leds.batch_roi_event_extractor, batch_seq, repeat(ir_path), reporter_vals, repeat(labeled_led_img), repeat(led_labels), repeat(led_sorting), repeat('avi'), repeat(led_blink_interval), chunksize=1)
    led_signals = np.concatenate(led_signals, axis=1)
    np.save(out_path, led_signals)

    return led_signals

def avi_parallel_workflow(base_path, save_path, source, num_leds=4, led_blink_interval=5, led_loc=None, exclude_center=False, manual_reverse=False, avi_chunk_size=1000, source_timescale_factor_log10=None, overwrite_extraction=False):

    if source_timescale_factor_log10 is None:
        source_timescale_factor_log10 = 6  # azure's timestamps in microseconds!

    # Set up paths
    if exists(source):
        ir_path = source
        source_name = os.path.split(source)[1]
        # TODO: Need to figure out how to get timestamps here if it's not CW's code.
    elif source == 'top_ir_avi':
        ir_path = util.find_file_through_glob_and_symlink(base_path, '*top.ir.avi')
        source_name = source
    timestamp_path = util.find_file_through_glob_and_symlink(base_path, '*top.device_timestamps.npy')
    
    print(f'Using file at {ir_path}...')

    # Load timestamps
    timestamps = np.load(timestamp_path)
    timestamps = timestamps / (10**source_timescale_factor_log10)  # convert to seconds

    # Check if already processed
    avi_led_events_path = join(save_path, f'{source_name}_led_events.npy')
    if exists(avi_led_events_path) and not overwrite_extraction:
        events = np.load(avi_led_events_path)
    else:
        # Get the std across all frames
        net_std = net_frame_std_parallel(ir_path, save_path, frame_chunksize=avi_chunk_size, overwrite_extraction=overwrite_extraction)

        # Extract LEDs from the net variance image
        # NB: this strategy will fail if the LEDs move during the session! In that case, need to treat each batch separately.
        num_features, filled_image, labeled_led_img = extract_leds.extract_initial_labeled_image(net_std, 'avi')

        # If too many features, check for location parameter and filter by it
        if (num_features > num_leds) and (led_loc or exclude_center):
            print('Too many features, using provided LED position...')
            labeled_led_img = extract_leds.clean_by_location(filled_image, labeled_led_img, led_loc, exclude_center)

        # Recompute num features (minus 1 for background)
        num_features = len(np.unique(labeled_led_img)) - 1

        # If still too many features, remove small ones
        if (num_features > num_leds):
            print('Oops! Number of features (%d) did not match the number of LEDs (%d)' % (num_features,num_leds))
            labeled_led_img = extract_leds.clean_by_size(labeled_led_img, lower_size_thresh=20, upper_size_thresh=100)
        
        # Recompute num features (minus 1 for background)
        num_features = len(np.unique(labeled_led_img)) - 1

        # Show user a check
        image_to_show = np.copy(labeled_led_img)
        viz.plot_video_frame(image_to_show, 200, join(save_path, 'net_var_led_labels_preEvents.png'))

        led_labels = [label for label in np.unique(labeled_led_img) if label > 0 ]
        assert led_labels == sorted(led_labels)  # note that these labels aren't guaranteed only the correct ROIs yet... but the labels should be strictly sorted at this point.
        print(f'Found {len(led_labels)} LED ROIs after size- and location-based cleaning...')        

        # At this point, sometimes still weird spots, but they're roughly LED sized.
        # So, to distinguish, get event data and then look for things that don't 
        # look like syncing LEDs in that data.

        # We use led_labels to extract events for each ROI.
        # sorting will be a nLEDs-length list, zero-indexed sort based on ROI horizontal or vertical position.
        # leds wil be an np.array of size (nLEDs, nFrames) with values 1 (on) and -1 (off) for events,
            #  and row index is the sort value.
        # So led_labels[sorting[0]] is the label of the ROI the script thinks belongs to LED #1,
            # and leds[sorting[0]] is the sequence of ONs and OFFs for that LED.
        sorting = extract_leds.get_roi_sorting(labeled_led_img, led_labels)

        # This is where the magic happens! Extract LED on/off info for each frame.
        leds = extract_led_events_parallel(ir_path, save_path, avi_chunk_size, labeled_led_img, led_labels, sorting, led_blink_interval, overwrite_extraction=overwrite_extraction)


        # useful debugging example:
        # extract_leds.batch_roi_event_extractor(range(21000, 22000), ir_path, None, labeled_led_img, led_labels, sorting, 'avi', led_blink_interval)

        # In the ideal case, there are 4 ROIs, extract events, double check LED 4 is switching each time, and we're done.
        if leds.shape[0] == num_leds:
            reverse = extract_leds.check_led_order(leds, num_leds)
        else:
            # Sometimes though you get little contaminating blips that look like LEDs.
            # They usually have way too many or events or hardly any 
            while leds.shape[0] > num_leds:

                # Look for an ROI with way more or way fewer events than the expected number
                mean_fps = 1/np.mean(np.diff(timestamps))
                expected_num_events = leds.shape[1]/mean_fps/led_blink_interval
                abs_log_ratios = np.abs(np.log(np.sum(leds != 0, axis=1) / expected_num_events))
                roi_to_remove_idx = np.argmax(abs_log_ratios)

                # Remove it
                row_bool = ~np.isin(np.arange(leds.shape[0]), roi_to_remove_idx)
                print(f'Removing roi #{roi_to_remove_idx} based on being farthest from expected event count...')
                leds = leds[row_bool,:]  # drop row
                labeled_led_img[labeled_led_img==led_labels[sorting[roi_to_remove_idx]]] = 0  # set ROI to bg
                sorting = sorting[row_bool]  # remove from sort
                
            # Figure out which LED is #4
            reverse = extract_leds.check_led_order(leds, num_leds)

        if reverse or manual_reverse:
            print('Reversed detected led order...')
            leds = leds[::-1, :]
            sorting = sorting[::-1]

        # Re-plot labeled led img, with remaining four led labels mapped to their sort order.
        # Use tmp because if you remap, say, 2 --> 3 before looking for 3, then when you look for 3, you'll also find 2.
        image_to_show = np.copy(labeled_led_img)
        tmp_idx_to_update = []
        for i in range(len(sorting)):
            tmp_idx_to_update.append(image_to_show == led_labels[sorting[i]])
        for i in range(len(sorting)):
            image_to_show[tmp_idx_to_update[i]] = (i+1)
        viz.plot_video_frame(image_to_show, 200, join(save_path, f'{source_name}_sort_order_postEvents.png'))

        # Extract events and append to event list
        if leds.shape[1] != len(timestamps):
            raise ValueError('Num frames and num timestamps do not match!')
        events = extract_leds.get_events(leds, timestamps)
        np.save(avi_led_events_path, events)
        print('Successfullly extracted avi leds, converting to codes...')    

    # Convert events to codes
    avi_led_codes, latencies = sync.events_to_codes(events, nchannels=num_leds, minCodeTime=(led_blink_interval-1))
    avi_led_codes = np.asarray(avi_led_codes)
    print('Converted.')

    return avi_led_codes, timestamps


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


def basler_workflow(base_path, save_path, num_leds, led_blink_interval, led_loc, basler_chunk_size=500, led_rois_from_file=False, overwrite_extraction=False):
    """
    Workflow to extract led codes from a Basler mp4 file.

    We know the LEDs only change once every (led_blink_interval), so we can just grab a few frames per interval. 
    
    """

    # Set up 
    basler_path = glob(join(base_path, 'basler_*.mp4'))
    frame_info_path = glob(join(base_path, 'basler_*.csv'))

    if len(basler_path) > 1:
        raise ValueError(f'There is more than one mp4 video in {base_path}')
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

        # old for-loop
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

        # new parallel method
        #TODO: implement parallel method, add global option to indicate whether to use it
        
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