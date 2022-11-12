from os.path import join, exists
import os
from moseq2_ephys_sync import cli
import numpy as np

def test_cli_top_ir_avi_without_led4():
    PATH_TO_TEST_DATA = '/n/groups/datta/Jonah/moseq2-ephys-sync/test_data/IA_led123/'
    out_dir = './tmp'
    arduino_name = 'txt'
    vid_name = 'top_ir_avi'
    os.system(f'moseq2_ephys_sync -i {PATH_TO_TEST_DATA}' + \
             f' -s1 {arduino_name} -s2 {vid_name}  --s2-timescale-factor-log10 6' + \
             f' -o {out_dir} --manual-reverse --leds-to-use 123' + \
             ' --exclude-center --led-loc rightquarter')