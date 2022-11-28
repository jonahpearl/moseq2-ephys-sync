from os.path import join, exists
import os
from moseq2_ephys_sync import cli
import numpy as np

def test_entrypoint():
    exit_status = os.system('moseq2_ephys_sync --help')
    assert exit_status == 0

def test_cli_with_workflow_names():
    PATH_TO_TEST_DATA = '/n/groups/datta/Jonah/moseq2-ephys-sync/test_data/ino_basler_test/'
    # current_dir = os.getcwd()
    # desired_dir = '/n/groups/datta/Jonah/moseq2-ephys-sync'
    # if current_dir != desired_dir:
    #     raise RuntimeError(f'Please run CLI testing from {desired_dir}')
    os.system(f'moseq2_ephys_sync {PATH_TO_TEST_DATA} -s1 arduino -s2 basler_bonsai --pytesting')

    save_path = join(PATH_TO_TEST_DATA, 'sync')
    matches_path = join(save_path, 'tmp_matches.npy')
    matches = np.load(matches_path)
    assert len(matches) > 0
    assert matches.shape == (145, 4)
    np.testing.assert_allclose(matches[0, :2], np.array([65.014, 7779.91675594]))
    np.testing.assert_allclose(matches[-1, :2], np.array([785.158, 8500.07755815]))

    # os.remove(matches_path)

def test_cli_with_paths_to_files():
    PATH_TO_TEST_DATA = '/n/groups/datta/Jonah/moseq2-ephys-sync/test_data/ino_basler_test/'
    arduino_fname = join(PATH_TO_TEST_DATA, '20220928_sync_test.txt')
    basler_data = join(PATH_TO_TEST_DATA, 'basler_2022-09-28T15_15_49.csv')
    os.system(f'moseq2_ephys_sync {PATH_TO_TEST_DATA} -s1 {arduino_fname} -s2 {basler_data} --s2-timescale-factor-log10 9 --pytesting')

    save_path = join(PATH_TO_TEST_DATA, 'sync')
    matches_path = join(save_path, 'tmp_matches.npy')
    matches = np.load(matches_path)
    assert len(matches) > 0
    assert matches.shape == (145, 4)
    np.testing.assert_allclose(matches[0, :2], np.array([65.014, 7779.91675594]))
    np.testing.assert_allclose(matches[-1, :2], np.array([785.158, 8500.07755815]))

    # os.remove(matches_path)

def test_cli_outputs_with_path_to_files():
    PATH_TO_TEST_DATA = '/n/groups/datta/Jonah/moseq2-ephys-sync/test_data/ino_basler_test/'
    out_dir = './tmp'
    arduino_fname = join(PATH_TO_TEST_DATA, '20220928_sync_test.txt')
    basler_data = join(PATH_TO_TEST_DATA, 'basler_2022-09-28T15_15_49.csv')
    os.system(f'moseq2_ephys_sync {PATH_TO_TEST_DATA} -s1 {arduino_fname} -s2 {basler_data}  -o {out_dir} --s2-timescale-factor-log10 9')

    assert exists(join(PATH_TO_TEST_DATA, out_dir, 'basler_2022-09-28T15_15_49_from_20220928_sync_test.p'))
    assert exists(join(PATH_TO_TEST_DATA, out_dir, '20220928_sync_test_from_basler_2022-09-28T15_15_49.p'))

    os.system(f'rm -rf {join(PATH_TO_TEST_DATA, out_dir)}')
    