import os
import numpy as np
import pandas as pd
import pytest

from moseq2_ephys_sync import workflows, sync

PATH_TO_TEST_DATA = '/n/groups/datta/Jonah/moseq2-ephys-sync/test_data/20220927_test'

def test_oe_loading():
    ch, ts = workflows.load_oe_data(PATH_TO_TEST_DATA)
    assert ch.shape == (547,)
    assert ts.shape == (547,)
    assert ch[0] == 2
    assert ch[-1] == -4
    assert ts[0] == 0
    assert ts[-1] == 32316169


def test_oe_workflow():
    codes, continuous_timestamps = workflows.oe_workflow(PATH_TO_TEST_DATA, num_leds=4, leds_to_use=[1,2,3,4], led_blink_interval=5, ephys_fs=3e4)
    code_times = [x for x,y,z,i in codes]
    codes_vals = [y for x,y,z,i in codes] 
    codes_idx = [int(i) for x,y,z,i in codes]
    assert codes_vals[0] == 11
    assert codes_vals[1] == 7
    assert codes_vals[2] == 15
    assert codes_vals[-1] == 7
    code_times[0] == 0
    code_times[1] == 6.98563333
    code_times[2] == 11.98663333
    code_times[-1] == 1077.20563333

    assert continuous_timestamps[0] == 0
    assert np.allclose(np.diff(continuous_timestamps), 3.3333333e-05)

    print(codes_idx[0])
    assert np.allclose(continuous_timestamps[codes_idx[0]], code_times[0])