from os.path import join, exists
import os
from moseq2_ephys_sync import cli
import numpy as np

def test_cli_with_paths_to_files():
    PATH_TO_TEST_DATA = '/n/groups/datta/Jonah/Thermistor_only_recordings/TRIMMED_test/'
    avi_data = join(PATH_TO_TEST_DATA, '20221010_gmou78_TRIM.top.ir.avi')
    ino_data = join(PATH_TO_TEST_DATA, '20221010_gmou78.txt')
    os.system(f'moseq2_ephys_sync -i {PATH_TO_TEST_DATA} -s1 {avi_data} -s2 {ino_data} -s1tf 6 -s2tf 3 --led-loc bottomright --pytesting')

    matches = np.load('./tests/tmp_matches.npy')
    assert len(matches) > 0
    assert matches.shape == (15, 4)
    np.testing.assert_allclose(matches[0, :2], np.array([4.066888, 375.459]))
    np.testing.assert_allclose(matches[-1, :2], np.array([74.033555, 445.471]))

    os.remove('./tests/tmp_matches.npy')