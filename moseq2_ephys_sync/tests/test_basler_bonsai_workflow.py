import os
import numpy as np
import pandas as pd
import pytest

from moseq2_ephys_sync import workflows, sync

def test_load_basler_bonsai():
    PATH_TO_TEST_DATA = '/n/groups/datta/Jonah/moseq2-ephys-sync/test_data/ino_basler_test/'
    txt_data = workflows.load_arduino_data(PATH_TO_TEST_DATA, file_glob='basler*.csv')
    np.testing.assert_allclose(txt_data.loc[0, 'time'], 7777583387622.0)
    assert txt_data.shape == (92384, 6)
    assert all(txt_data.columns == ['time', 'frame', 'led1', 'led2', 'led3', 'led4'])