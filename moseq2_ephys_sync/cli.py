import numpy as np
import os
from os.path import join, exists
import glob
import joblib
# import argparse
import click
from mlinsights.mlmodel import PiecewiseRegressor
from sklearn.preprocessing import KBinsDiscretizer

from moseq2_ephys_sync import workflows, sync, viz

import pdb

VALID_SOURCE_ABBREVS = ['oe', 'mkv', 'arduino', 'txt', 'csv', 'basler', 'basler_bonsai', 'avi']

@click.command()
@click.argument('--path', type=str)
@click.argument('-o', '--output-dir-name', type=str, default='sync')
@click.argument('-s1', '--first-source', type=str)
@click.argument('-s2', '--second-source', type=str)
@click.argument('--led-loc', type=str, help="Location of the syncing LEDs in the video, as seen from plt.imshow()'s point of view. Currenly supported: quadrants (topright, topleft, bottomright, bottomleft), some vertical strips (rightquarter, leftquarter), some horizontal strips (topquarter, topthird, bottomquarter). Add more in extract_leds.py.")
@click.argument('--led-blink-interval', type=int, default=5)
@click.argument('--arduino-spec', type=str, help="DEPRECATED: use header instead! Currently supported: fictive_olfaction, odor_on_wheel, basic_thermistor")
@click.argument('--s1-led-rois-from-file', is_flag=True, help="Flag to look for lists of points for source 1 led rois")
@click.argument('--s2-led-rois-from-file', is_flag=True, help="Flag to look for lists of points for source 2 led rois")
@click.argument('--overwrite-models', is_flag=True)
@click.argument('--overwrite_extraction', is_flag=True)
@click.argument('--leds_to_use', nargs=-1, default=['1', '2', '3', '4'], help='Choose a subset of leds (1-indexed) to use if one was broken (syntax: --leds_to_use 1 2 4 --next_arg...')
@click.argument('--predict_full_timestamps_of_source', nargs=-1, default=None, help='Choose which sources (1, 2, or both) to predict full list of times for (syntax: ...of_source 1 2 --next_arg')
def main(base_path,
    first_source=None,
    second_source=None,
    output_dir_name='sync',
    led_loc=None, 
    led_blink_interval=5, 
    arduino_spec=None, 
    s1_led_rois_from_file=False,
    s2_led_rois_from_file=False, 
    overwrite_models=False,
    overwrite_extraction=False,
    leds_to_use=[1,2,3,4],
    sources_to_predict=None):

    
    if (first_source is None) and (second_source is None):
        # # Try to infer what files the user wants
        # oe_file = glob.glob(join(base_path, '**', 'TT__*'), recursive=True)[0]
        # mkv_files = glob.glob(join(base_path, '.mkv'))[0]
        # arduino_files = glob.glob(join(base_path, '.txt.'))[0]
        # basler_csv_files = [f for f in glob.glob(base_path) if (f.endswith('.csv') and ('Basler' in f))][0]
        # basler_video_files = [f for f in glob.glob(base_path) if (f.endswith('.avi') and ('Basler' in f))][0]  # old; try not to use
        # # tdt_files = 

        # if len(oe_files) == 1:
        #     s1 = oe_
        # elif len(oe_files) > 1:
        #     raise ValueError('Two OE files found!')
        pass

    elif (first_source is not None and second_source is None) or (first_source is None and second_source is not None):
        raise ValueError('Cannot specify one source but not the other!')
    else:
        workflows.main_function(base_path,
            first_source,
            second_source,
            output_dir_name='sync',
            led_loc=led_loc, 
            led_blink_interval=led_blink_interval, 
            arduino_spec=arduino_spec, 
            s1_led_rois_from_file=s1_led_rois_from_file,
            s2_led_rois_from_file=s2_led_rois_from_file, 
            overwrite_models=overwrite_models,
            overwrite_extraction=overwrite_extraction,
            leds_to_use=leds_to_use,
            sources_to_predict=sources_to_predict)



if __name__ == "__main__" :
    main()
