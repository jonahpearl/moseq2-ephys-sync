import os
import numpy as np
import pandas as pd
import pytest

from moseq2_ephys_sync import workflows, sync

PATH_TO_TEST_DATA = '/n/groups/datta/Jonah/moseq2-ephys-sync/test_data/20220927_test'

def test_arduino_loading():
    data = workflows.load_arduino_data(PATH_TO_TEST_DATA)
    assert type(data) is pd.DataFrame
    assert data.loc[0, 'time'] == 346475.0
    assert data.shape == (323225, 5)

def test_list_to_events_and_codes():
    times = np.arange(10)
    lists = np.array([[0, 0, 1, 1, 0, 0, 1, 1, 1, 1],  # 1
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # 2
                      [1, 1, 1, 0, 1, 1, 0, 0, 1, 1],  # 4
                      [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]) #8  
    output = np.array([[ 1,  3, -1],
       [ 2,  3,  1],
       [ 2,  0,  1],
       [ 3,  2, -1],
       [ 3,  3, -1],
       [ 4,  1,  1],
       [ 4,  2,  1],
       [ 4,  0, -1],
       [ 4,  3,  1],
       [ 5,  3, -1],
       [ 6,  2, -1],
       [ 6,  3,  1],
       [ 6,  0,  1],
       [ 7,  3, -1],
       [ 8,  2,  1],
       [ 8,  3,  1],
       [ 9,  3, -1]])
    assert np.all(workflows.list_to_events(times, lists, 100) == output)

    ino_codes, _ = sync.events_to_codes(output, nchannels=4, minCodeTime=0.5) 
    codes = [y for x,y,z in ino_codes]
    assert codes == [4, 13, 1, 14, 6, 11, 3, 15, 7]  # skips the first code, otherwise is good.


    def test_arduino_workflow():
        ino_codes, ino_timestamps = workflows.arduino_workflow(PATH_TO_TEST_DATA, num_leds=4, leds_to_use=[1,2,3,4], led_blink_interval=5)
        code_times = [x for x,y,z in ino_codes]
        codes_vals = [y for x,y,z in ino_codes]
        assert ino_codes.shape == (64, 3)
        assert codes_vals[0] == 6
        assert codes_vals[1] == 11
        assert codes_vals[2] == 7
        assert codes_vals[-1] == 12
        assert code_times[0] == 350.071
        assert code_times[1] == 355.072
        assert code_times[-1] == 665.134
        assert ino_timestamps[0] == 346.475
        assert ino_timestamps[-1] == 669.761
        assert ino_timestamps.shape == (323225,)