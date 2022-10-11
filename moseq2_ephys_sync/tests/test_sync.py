import os
import numpy as np
import pandas as pd
import pytest

from moseq2_ephys_sync import workflows, sync



def test_arduino_and_oe_sync():

    PATH_TO_TEST_DATA = '/n/groups/datta/Jonah/moseq2-ephys-sync/test_data/20220927_test'
    # Get the event codes from each source
    first_source_led_codes, first_source_full_timestamps = workflows.arduino_workflow(PATH_TO_TEST_DATA, num_leds=4, leds_to_use=[1,2,3,4], led_blink_interval=5)
    second_source_led_codes, second_source_full_timestamps = workflows.oe_workflow(PATH_TO_TEST_DATA, num_leds=4, leds_to_use=[1,2,3,4], led_blink_interval=5, ephys_fs=3e4)

    # Do the syncing.
    # NB leaves off (minMatch) codes from the end!! So if you want 60 minutes of data you need to record 61 minutes!
    matches = np.asarray(sync.match_codes(first_source_led_codes[:,0],  
                                  first_source_led_codes[:,1], 
                                  first_source_led_codes[:,3], 
                                  second_source_led_codes[:,0],
                                  second_source_led_codes[:,1],
                                  second_source_led_codes[:,3],
                                  minMatch=10,maxErr=0,remove_duplicates=True ))

    assert len(matches) > 0
    assert matches.shape == (54, 4)
    np.testing.assert_allclose(matches[0, :2], np.array([355.072, 0.]))
    np.testing.assert_allclose(matches[-1, :2], np.array([620.125 , 267.03943333]))

    
def test_arduino_and_basler_bonsai_sync():
    PATH_TO_TEST_DATA = '/n/groups/datta/Jonah/moseq2-ephys-sync/test_data/ino_basler_test/'

    first_source_led_codes, first_source_full_timestamps = workflows.arduino_workflow(PATH_TO_TEST_DATA, num_leds=4, leds_to_use=[1,2,3,4], led_blink_interval=5)
    second_source_led_codes, second_source_full_timestamps = workflows.basler_bonsai_workflow(PATH_TO_TEST_DATA, num_leds=4, leds_to_use=[1,2,3,4], led_blink_interval=5)
    matches = np.asarray(sync.match_codes(first_source_led_codes[:,0],  
                                  first_source_led_codes[:,1], 
                                  first_source_led_codes[:,3], 
                                  second_source_led_codes[:,0],
                                  second_source_led_codes[:,1],
                                  second_source_led_codes[:,3],
                                  minMatch=10,maxErr=0,remove_duplicates=True ))
    assert len(matches) > 0
    assert matches.shape == (145, 4)
    np.testing.assert_allclose(matches[0, :2], np.array([65.014, 7779.91675594]))
    np.testing.assert_allclose(matches[-1, :2], np.array([785.158, 8500.07755815]))