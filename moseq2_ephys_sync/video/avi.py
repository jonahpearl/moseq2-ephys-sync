# A workflow to deal with caleb's PyK4a acquisition system
import numpy as np
from tqdm import tqdm
import pdb
import os
import imageio
import moseq2_ephys_sync.viz as viz
import moseq2_ephys_sync.video.extract_leds as extract_leds
import moseq2_ephys_sync.sync as sync
import moseq2_ephys_sync.util as util



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


def avi_workflow(base_path, save_path, num_leds=4, led_blink_interval=5000, led_loc=None, avi_chunk_size=1000, overwrite_extraction=False):

    
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


