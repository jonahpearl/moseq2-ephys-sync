import numpy as np
import os
import joblib
import argparse
from mlinsights.mlmodel import PiecewiseRegressor
from sklearn.preprocessing import KBinsDiscretizer

import mkv, arduino, ttl, sync, plotting, basler

import pdb


def main_function(base_path,
output_dir_name,
first_source,
second_source,
led_loc=None, 
led_blink_interval=5, 
arduino_spec=None, 
s1_led_rois_from_file=False,
s2_led_rois_from_file=False, 
overwrite_models=False,
overwrite_mkv_extraction=False):
    """
    Uses 4-bit code sequences to create a piecewise linear model to predict first_source times from second_source times
    ----
    Inputs:
        base_path (str): path to the .mkv and any other files needed
        output_dir: path to save output models and plots. Default: {base_path}/sync.
        first_source (str): 'ttl', 'mkv', 'arduino', or 'basler'. Source to be predicted.
            ttl: looks for open ephys data in __ format
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

    """
    print(f'Running sync on {base_path} with {first_source} as first source and {second_source} as second source.')


    #### SETUP ####
    # Built-in params (should make dynamic)
    mkv_chunk_size = 2000
    basler_chunk_size = 2000  
    num_leds = 4
    ephys_fs = 3e4  # sampling rate in Hz

    # Set up save path
    save_path = f'{base_path}/{output_dir_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check if models already exist, only over-write if requested
    model_exists_bool = os.path.exists(f'{save_path}/{first_source}_from_{second_source}.p') or os.path.exists(f'{save_path}/{second_source}_from_{first_source}.p')
    if model_exists_bool and not overwrite_models:
        raise RuntimeError("Models already exist and overwrite_models is false!")

    # Require led rois for basler
    if (first_source == 'basler' and not s1_led_rois_from_file) or (second_source == 'basler' and not s2_led_rois_from_file):
        raise RuntimeError("User must specify LED rois for basler workflow")



    #### INDIVIDUAL DATA STREAM WORKFLOWS ####

    # Deal with first source
    if first_source == 'ttl':
        first_source_led_codes = ttl.ttl_workflow(base_path, save_path, num_leds, led_blink_interval, ephys_fs)
    elif first_source == 'mkv':
        assert not (led_loc and s1_led_rois_from_file), "User cannot specify both MKV led location (top right, etc) and list of exact MKV LED ROIs!"
        first_source_led_codes = mkv.mkv_workflow(base_path, save_path, num_leds, led_blink_interval, mkv_chunk_size, led_loc, s1_led_rois_from_file, overwrite_mkv_extraction)
    elif first_source == 'arduino':
        first_source_led_codes, ino_average_fs = arduino.arduino_workflow(base_path, save_path, num_leds, led_blink_interval, arduino_spec)
    elif first_source == 'basler':
        first_source_led_codes = basler.basler_workflow(base_path, save_path, num_leds, led_blink_interval, basler_chunk_size, s1_led_rois_from_file, overwrite_models)

    # Deal with second source
    if second_source == 'ttl':
        second_source_led_codes = ttl.ttl_workflow(base_path, save_path, num_leds, led_blink_interval, ephys_fs)
    elif second_source == 'mkv':
        assert not (led_loc and s2_led_rois_from_file), "User cannot specify both MKV led location (top right, etc) and list of exact MKV LED ROIs!"
        second_source_led_codes = mkv.mkv_workflow(base_path, save_path, num_leds, led_blink_interval, mkv_chunk_size, led_loc, s1_led_rois_from_file, overwrite_mkv_extraction)
    elif second_source == 'arduino':
        second_source_led_codes, ino_average_fs = arduino.arduino_workflow(base_path, save_path, num_leds, led_blink_interval, arduino_spec)
    elif second_source == 'basler':
        second_source_led_codes = basler.basler_workflow(base_path, save_path, num_leds, led_blink_interval, basler_chunk_size, s1_led_rois_from_file, overwrite_models)

    # Save the codes for use later
    np.savez('%s/codes.npz' % save_path, first_source_codes=first_source_led_codes, second_source_codes=second_source_led_codes)


    # Visualize a small chunk of the bit codes. do you see a match? 
    # Codes array should have times in seconds by this point
    plotting.plot_code_chunk(first_source_led_codes, first_source, second_source_led_codes, second_source, save_path)


    #### SYNCING :D ####

    # Returns two columns of matched event times
    matches = np.asarray(sync.match_codes(first_source_led_codes[:,0],  ## all times should be in seconds by here
                                  first_source_led_codes[:,1], 
                                  second_source_led_codes[:,0],
                                  second_source_led_codes[:,1],
                                  minMatch=10,maxErr=0,remove_duplicates=True ))

    ## Plot the matched codes against each other:
    plotting.plot_matched_scatter(matches, save_path)





    #### Make the models! ####

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
        joblib.dump(mdl, f'{save_path}/{n1}_from_{n2}.p')
        print(f'Saved model that predicts {n1} from {n2}')


    print('Syncing complete. FIN')



if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str)  # path to data
    parser.add_argument('-o', '--output_dir_name', type=str, default='sync')  # name of output folder within path
    parser.add_argument('-s1', '--first_source', type=str)  # ttl, mkv, basler (mp4 with rois), arduino (txt) 
    parser.add_argument('-s2', '--second_source', type=str)   
    parser.add_argument('--led_loc', type=str)
    parser.add_argument('--led_blink_interval', type=int, default=5)  # default blink every 5 seconds
    parser.add_argument('--arduino_spec', type=str, help="Currently supported: fictive_olfaction, odor_on_wheel, basic_thermistor")  # specifiy cols in arduino text file
    parser.add_argument('--s1_led_rois_from_file', action="store_true", help="Flag to look for lists of points for source 1 led rois")  # need to run separate jup notbook first to get this
    parser.add_argument('--s2_led_rois_from_file', action="store_true", help="Flag to look for lists of points for source 2 led rois")  # need to run separate jup notbook first to get this
    parser.add_argument('--overwrite_models', action="store_true")  # overwrites old models if True (1)
    parser.add_argument('--overwrite_mkv_extraction', action="store_true")  # re-does mkv extraction (can take a long time, hence a separate flag)

    settings = parser.parse_args(); 

    main_function(base_path=settings.path,
                output_dir_name=settings.output_dir_name,
                first_source=settings.first_source,
                second_source=settings.second_source,
                led_loc=settings.led_loc,
                led_blink_interval=settings.led_blink_interval,
                arduino_spec=settings.arduino_spec,
                s1_led_rois_from_file=settings.s1_led_rois_from_file,
                s2_led_rois_from_file=settings.s2_led_rois_from_file,
                overwrite_models=settings.overwrite_models,
                overwrite_mkv_extraction=settings.overwrite_mkv_extraction)

    
