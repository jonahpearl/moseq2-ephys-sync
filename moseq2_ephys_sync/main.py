import numpy as np
import os
import joblib
import argparse
from mlinsights.mlmodel import PiecewiseRegressor
from sklearn.preprocessing import KBinsDiscretizer

import mkv, arduino, ttl, sync, plotting, basler, avi, basler_bonsai

import pdb


def process_source(source,
                    base_path=None,
                    save_path=None,
                    num_leds=None,
                    leds_to_use=None,
                    led_blink_interval=None, 
                    ephys_fs=None,
                    mkv_chunk_size=None,
                    basler_chunk_size=None,
                    avi_chunk_size=None,
                    led_loc=None,
                    led_rois_from_file=None,
                    overwrite_extraction=False,
                    arduino_spec=None
                    ):

    if source == 'ttl':
        source_led_codes, source_full_timestamps = ttl.ttl_workflow(base_path, save_path, num_leds, leds_to_use, led_blink_interval, ephys_fs)

    elif source == 'mkv':
        assert not (led_loc and led_rois_from_file), "User cannot specify both MKV led location (top right, etc) and list of exact MKV LED ROIs!"
        assert '4' in leds_to_use, "LED extraction code expects that last LED is LED 4 (switching every interval)" 
        source_led_codes, source_full_timestamps = mkv.mkv_workflow(base_path, save_path, num_leds, led_blink_interval, mkv_chunk_size, led_loc, led_rois_from_file, overwrite_extraction)

    elif source == 'arduino' or source=='txt':
        source_led_codes, source_full_timestamps = arduino.arduino_workflow(base_path, save_path, num_leds, leds_to_use, led_blink_interval, arduino_spec)

    elif source == 'basler':
        assert not (led_loc and led_rois_from_file), "User cannot specify both Basler led location (top right, etc) and list of exact Basler LED ROIs!"
        source_led_codes, source_full_timestamps = basler.basler_workflow(base_path, save_path, num_leds, led_blink_interval, led_loc, basler_chunk_size, led_rois_from_file, overwrite_extraction)

    elif source == 'basler_bonsai':
        # TODO: add in possibility in main code of passing in both arduino spec and bonsai spec (although ideally they just have headers...)
        source_led_codes, source_full_timestamps = basler_bonsai.basler_bonsai_workflow(base_path, save_path, num_leds, leds_to_use, led_blink_interval)

    elif source == 'avi':
        assert '4' in leds_to_use, "LED extraction code expects that last LED is LED 4 (switching every interval)" 
        source_led_codes, source_full_timestamps = avi.avi_workflow(base_path, save_path, num_leds=num_leds, led_blink_interval=led_blink_interval, led_loc=led_loc, avi_chunk_size=avi_chunk_size, overwrite_extraction=overwrite_extraction)

    else:
        raise RuntimeError(f'First source keyword {source} not recognized')

    return source_led_codes, source_full_timestamps


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
overwrite_extraction=False,
leds_to_use=[1,2,3,4],
sources_to_predict=None):

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
    basler_chunk_size = 3000  # need to use high enough so that std of LEDs is high enough to separate them  
    avi_chunk_size = 2000
    ephys_fs = 3e4  # sampling rate in Hz


    # Detect num leds
    num_leds = len(leds_to_use)

    # Set up save path
    save_path = f'{base_path}/{output_dir_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check if models already exist, only over-write if requested
    model_exists_bool = os.path.exists(f'{save_path}/{first_source}_from_{second_source}.p') or \
                        os.path.exists(f'{save_path}/{second_source}_from_{first_source}.p')

    if model_exists_bool and not overwrite_models:
        raise RuntimeError("One or both models already exist and overwrite_models is false!")


    print('Dealing with first souce...')
    # first_source_led_codes: array of reconstructed pixel clock codes where: codes[:,0] = time, codes[:,1] = code (and codes[:,2] = trigger channel but that's not used in this code)
    # first_source_full_timestamps: full list of timestamps from source 1 (every timestamp, not just event times! For prediction with the model at the end.)
    first_source_led_codes, first_source_full_timestamps = \
    process_source(first_source,
                    base_path=base_path,
                    save_path=save_path, num_leds=num_leds,
                    leds_to_use=leds_to_use,
                    led_blink_interval=led_blink_interval, 
                    ephys_fs=ephys_fs,
                    mkv_chunk_size=mkv_chunk_size,
                    basler_chunk_size=basler_chunk_size,
                    avi_chunk_size=avi_chunk_size,
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
                    ephys_fs=ephys_fs,
                    mkv_chunk_size=mkv_chunk_size,
                    basler_chunk_size=basler_chunk_size,
                    avi_chunk_size=avi_chunk_size,
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
    np.savez(f'{save_path}/codes_{first_source}_and_{second_source}.npz', first_source_codes=first_source_led_codes, second_source_codes=second_source_led_codes)


    # Visualize a small chunk of the bit codes. do you see a match? 
    # Codes array should have times in seconds by this point
    plotting.plot_code_chunk(first_source_led_codes, first_source, second_source_led_codes, second_source, save_path)




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
    plotting.plot_matched_scatter(matches, first_source, second_source, save_path)


    #### Make the models! ####
    print('Modeling the two sources from each other...')

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
            n1 = first_source
            full1 = first_source_full_timestamps
            s2 = ground_truth_source2_event_times
            t2 = second_source_led_codes
            n2 = second_source
            full2 = second_source_full_timestamps

        elif i == 1:
            s1 = ground_truth_source2_event_times
            t1 = second_source_led_codes
            n1 = second_source
            full1 = first_source_full_timestamps
            s2 = ground_truth_source1_event_times
            t2 = first_source_led_codes
            n2 = first_source
            full2 = first_source_full_timestamps

        # Learn to predict s1 from s2. Syntax is fit(X,Y).
        mdl = PiecewiseRegressor(verbose=True,
                                binner=KBinsDiscretizer(n_bins=4))
        mdl.fit(s2.reshape(-1, 1), s1)

        outname = f'{n1}_from_{n2}'

        # Verify accuracy of predicted event times
        predicted_event_times = mdl.predict(s2.reshape(-1, 1))
        time_errors = predicted_event_times - s1 
        plotting.plot_model_errors(time_errors, save_path, outname)

        # Plot all predicted times
        all_predicted_times = mdl.predict(t2[:,0].reshape(-1, 1))  # t1-timebase times of t2 codes (predict t1 from t2)
        plotting.plot_matched_times(all_predicted_times, t2, t1, n1, n2, save_path, outname)

        # Save
        joblib.dump(mdl, os.path.join(save_path,f'{outname}.p'))
        print(f'Saved model that predicts {n1} from {n2}')

        # Compute and save the full synced timestamps.
        # Eg: if we're predicting timestamps for ttl from txt, it will return a list of times of length (num times in txt file), where each entry is the corresponding time in the ephys file
        # I would recommend not predicting timestamps from ttl, as it will be ~ 1GB.
        if str(i+1) in sources_to_predict:
            fout = os.path.join(save_path,f'{outname}_fullTimes.npy')
            if not os.path.exists(fout) or overwrite_models:
                print(f'Computing full synced timestamp list for {n1} from {n2} (this may take a while...)')
                full_predicted_s1 = mdl.predict(full2.reshape(-1,1))
                np.save(fout, full_predicted_s1)
            else:
                print(f'Full synced timestamp list for {n1} from {n2} already exists, continuing...')

    print('Syncing complete. FIN')



if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str)  # path to data
    parser.add_argument('-o', '--output_dir_name', type=str, default='sync')  # name of output folder within path
    parser.add_argument('-s1', '--first_source', type=str)  # ttl, mkv, basler (mp4 with rois), arduino (txt), or absolute path to data file with extension
    parser.add_argument('-s2', '--second_source', type=str)   
    parser.add_argument('--led_loc', type=str, help="Location of the syncing LEDs in the video, as seen from plt.imshow()'s point of view. Currenly supported: quadrants (topright, topleft, bottomright, bottomleft), some vertical strips (rightquarter, leftquarter), some horizontal strips (topquarter, topthird, bottomquarter). Add more in extract_leds.py.")
    parser.add_argument('--led_blink_interval', type=int, default=5)  # default blink every 5 seconds
    parser.add_argument('--arduino_spec', type=str, help="Currently supported: fictive_olfaction, odor_on_wheel, basic_thermistor")  # specifiy cols in arduino text file
    parser.add_argument('--s1_led_rois_from_file', action="store_true", help="Flag to look for lists of points for source 1 led rois")  # need to run separate jup notbook first to get this
    parser.add_argument('--s2_led_rois_from_file', action="store_true", help="Flag to look for lists of points for source 2 led rois")  # need to run separate jup notbook first to get this
    parser.add_argument('--overwrite_models', action="store_true")  # overwrites old models if True (1)
    parser.add_argument('--overwrite_extraction', action="store_true")  # re-does mkv or avi extraction (can take a long time, hence a separate flag)
    parser.add_argument('--leds_to_use', nargs='*', default=['1', '2' , '3' , '4'], help='Choose a subset of leds (1-indexed) to use if one was broken (syntax: --leds_to_use 1 2 4 --next_arg...')
    parser.add_argument('--predict_full_timestamps_of_source', nargs='*', default=[], help='Choose which sources (1, 2, or both) to predict full list of times for (syntax: ...of_source 1 2 --next_arg')

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
                overwrite_extraction=settings.overwrite_extraction,
                leds_to_use=settings.leds_to_use,
                sources_to_predict=settings.predict_full_timestamps_of_source)

    
