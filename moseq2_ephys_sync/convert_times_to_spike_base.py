import numpy as np
import os
from os.path import join, exists
import joblib
import argparse
from mlinsights.mlmodel import PiecewiseRegressor
import pynwb as nwb
import warnings
import re
import pandas as pd

def load_sniff_data_from_nwb(nwb_filepath):
    io = nwb.NWBHDF5IO(nwb_filepath, 'r')
    nwb_file = io.read()

    # For debugging
    # raw_times = nwb_file.acquisition['dac'].timestamps[:]
    # raw_therm = nwb_file.acquisition['therm'].data[:]
    # dac = nwb_file.acquisition['dac'].data[:]

    # read in data
    interpd = ts_interface_to_df(nwb_file.processing['behavior']['interpd_500'])
    sniff_peak_times = nwb_file.processing['behavior']['sniff_events']['sniff_times'].timestamps[:]
    sniff_peak_locs = np.where(nwb_file.processing['behavior']['sniff_events']['sniff_idx'].data[:]==1)[0]
    exhale_peak_times = nwb_file.processing['behavior']['sniff_events']['exhale_times'].timestamps[:]    
    exhale_peak_locs = np.where(nwb_file.processing['behavior']['sniff_events']['exhale_idx'].data[:]==1)[0]

    io.close()
    
    return interpd, sniff_peak_times, sniff_peak_locs, exhale_peak_times, exhale_peak_locs

def ts_interface_to_df(interface, timeseries_to_use='all'):
    """Turns a BehavioralTimeSeries with aligned data into a pd dataframe
    
    timeseries_to_use: list of which timeseries to use, or 'all'
    """
    
    # Parse which timeseries to use
    if timeseries_to_use == 'all':
        ts_list = list(interface.time_series.keys())
    else:
        ts_list = list(timeseries_to_use)
    
    # Check that all times match
    times_first = interface[ts_list[0]].timestamps[:]
    for ts in ts_list:
        assert all(interface[ts].timestamps[:] == times_first)
    
    # Pre allocate
    arr = np.zeros((times_first.shape[0], len(ts_list)+1))
    
    # Add times as first column
    arr[:,0] = times_first
    
    # Collect rest of data
    for i, ts in enumerate(ts_list):
#         pdb.set_trace()
        arr[:,i+1] = np.array(interface[ts].data[:])
        
    return pd.DataFrame(arr, columns = ['time']+ts_list)


def main_function(base_path, input_dir_name, output_dir_name, sources=None, nwb_dir=None, overwrite_output=False):
    
    print(f'Generating ephys-base times for {sources} for {base_path}')

    extractor_from_openEphysFolder = re.compile('.*/(?P<mouse>gmou\d*)_(?P<date>\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}_(?P<session_type>\w*)')
    extractor_from_nwb_filename = re.compile('(?P<date>\d{4}-\d{2}-\d{2})_(?P<mouse>gmou\d*)_(?P<session_type>.*).nwb')

    for source in sources:

        if source == 'ino_interpd_nwb' or source == 'ino_txt_raw':
            mdl_name = 'ttl_from_txt.p'
        else:
            mdl_name = f'ttl_from_{source}.p'
        mdl_file = join(base_path, input_dir_name, mdl_name)
        

        # Check if syncing model exists
        if not exists(mdl_file):
            raise ValueError(f'{mdl_name} does not exist in input dir {join(base_path, input_dir_name)}; go back and run syncing model first')

        # Check if output already exists; if so, warn user, and only overwrite if allowed to
        output_name = f'{source}_times_in_ttl_base.npy'  # eventual output name
        out_file = join(base_path, output_dir_name, output_name)
        if exists(out_file) and not overwrite_output:
            warnings.warn('Output file exists! Use flag overwrite_output to overwrite. Quitting...')
            raise ValueError('Output already exists!')
        elif exists(out_file) and overwrite_output:
            warnings.warn('Output file exists, overwriting as requested...')
        
        # Load the syncing model
        mdl = joblib.load(mdl_file)

        # Load desired source's input times
        if source == 'txt':
            raise ValueError('Please specify ino_raw_txt or ino_interpd_nwb')
        elif source == 'ino_raw_txt':
            arduino_data_path = util.find_file_through_glob_and_symlink(base_path, '*.txt')
            raise ValueError('Not implemented yet')

        elif source == 'ino_interpd_nwb':
            if nwb_dir is None: raise ValueError('nwb_dir cannot be none if using ino_interpd_nwb')

            # Get base path meta info
            rexp = extractor_from_openEphysFolder.search(base_path)
            mouse = rexp.group('mouse')
            date = rexp.group('date')
            session_type = rexp.group('session_type')
    
            # Find corresponding nwb
            nwb_filepath = join(nwb_dir, f'{date}_{mouse}_{session_type}.nwb')
            if not exists(nwb_filepath):
                raise ValueError('No NWB file at {nwb_filepath}')
            print(f'Using NWB {nwb_filepath}')

            # Load the interpd times
            interpd, _, _, _, _ = load_sniff_data_from_nwb(nwb_filepath)
            times_to_sync = np.array(interpd.time)

        elif source == 'mkv':
            raise ValueError('Not implemented yet')

        # Sync the times
        print('Begin running the model...')
        synced_times = mdl.predict(times_to_sync.reshape(-1,1))  # takes like 30 min

        # Make out dir if not exists
        if not exists(join(base_path, output_dir_name)):
            os.mkdir(join(base_path, output_dir_name))

        # Save
        with open(out_file, 'wb') as fout:
            np.save(fout, synced_times)
        
        print(f'Done with source {source}')

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str)  # path to data
    parser.add_argument('-i', '--input_dir_name', type=str, default='sync')  # name of input folder within path
    parser.add_argument('-o', '--output_dir_name', type=str, default='times_in_spike_base')  # name of output folder within path
    parser.add_argument('--sources', nargs='*', type=str, help='Which sources to convert into spike time base ("ttl"). A syncing model must already exist.')  # ino_raw_txt, ino_interpd_nwb, mkv, basler (mp4 with rois)
    parser.add_argument('--nwb_dir', type=str, help='Where to look for nwb file') 
    parser.add_argument('--overwrite_output', action="store_true")
    settings = parser.parse_args(); 

    main_function(base_path=settings.path,
                input_dir_name=settings.input_dir_name,
                output_dir_name=settings.output_dir_name,
                sources=settings.sources,
                nwb_dir=settings.nwb_dir,
                overwrite_output=settings.overwrite_output)
                

