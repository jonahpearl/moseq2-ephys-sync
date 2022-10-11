from os.path import join, exists
import os
from moseq2_ephys_sync import cli
import numpy as np

def test_entrypoint():
    exit_status = os.system('moseq2_ephys_sync --help')
    assert exit_status == 0

def test_cli_with_workflow_names():
    PATH_TO_TEST_DATA = '/n/groups/datta/Jonah/moseq2-ephys-sync/test_data/ino_basler_test/'
    os.system(f'moseq2_ephys_sync -i {PATH_TO_TEST_DATA} -s1 arduino -s2 basler_bonsai --pytesting')

    matches = np.load('./tests/tmp_matches.npy')
    assert len(matches) > 0
    assert matches.shape == (145, 4)
    np.testing.assert_allclose(matches[0, :2], np.array([65.014, 7779.91675594]))
    np.testing.assert_allclose(matches[-1, :2], np.array([785.158, 8500.07755815]))

    os.remove('./tests/tmp_matches.npy')

def test_cli_with_paths_to_files():
    PATH_TO_TEST_DATA = '/n/groups/datta/Jonah/moseq2-ephys-sync/test_data/ino_basler_test/'
    arduino_fname = join(PATH_TO_TEST_DATA, '20220928_sync_test.txt')
    basler_data = join(PATH_TO_TEST_DATA, 'basler_2022-09-28T15_15_49.csv')
    os.system(f'moseq2_ephys_sync -i {PATH_TO_TEST_DATA} -s1 {arduino_fname} -s2 {basler_data} --s2-timescale-factor-log10 9 --pytesting')

    matches = np.load('./tests/tmp_matches.npy')
    assert len(matches) > 0
    assert matches.shape == (145, 4)
    np.testing.assert_allclose(matches[0, :2], np.array([65.014, 7779.91675594]))
    np.testing.assert_allclose(matches[-1, :2], np.array([785.158, 8500.07755815]))

    os.remove('./tests/tmp_matches.npy')

def test_cli_outputs_with_path_to_files():
    PATH_TO_TEST_DATA = '/n/groups/datta/Jonah/moseq2-ephys-sync/test_data/ino_basler_test/'
    out_dir = './tmp'
    arduino_fname = join(PATH_TO_TEST_DATA, '20220928_sync_test.txt')
    basler_data = join(PATH_TO_TEST_DATA, 'basler_2022-09-28T15_15_49.csv')
    os.system(f'moseq2_ephys_sync -i {PATH_TO_TEST_DATA} -s1 {arduino_fname} -s2 {basler_data}  -o {out_dir} --s2-timescale-factor-log10 9')

    assert exists(join(PATH_TO_TEST_DATA, out_dir, 'basler_2022-09-28T15_15_49_from_20220928_sync_test.p'))
    assert exists(join(PATH_TO_TEST_DATA, out_dir, '20220928_sync_test_from_basler_2022-09-28T15_15_49.p'))

    os.system(f'rm -rf {join(PATH_TO_TEST_DATA, out_dir)}')
    