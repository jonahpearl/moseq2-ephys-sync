import numpy as np

from glob import glob
import sync
import pdb

def ttl_workflow(base_path, save_path, num_leds, leds_to_use, led_blink_interval, ephys_fs):
    """
    
    """
    # assert num_leds==4, "TTL code expects 4 LED channels, other nums of channels not yet supported"
    assert num_leds == len(leds_to_use)

    # Load the TTL data
    ephys_ttl_path = glob('%s/**/TTL_*/' % base_path,recursive = True)[0]
    channels = np.load('%s/channel_states.npy' % ephys_ttl_path)
    ephys_TTL_timestamps = np.load('%s/timestamps.npy' % ephys_ttl_path)  # these are in sample number

    # Need to subtract the raw traces' starting timestamp from the TTL timestamps
    # (This is a bit of a glitch in open ephys / spike interface, might be able to remove this in future versions)
    continuous_timestamps_path = glob('%s/**/continuous/**/timestamps.npy' % base_path,recursive = True)[0] ## load the continuous stream's timestamps
    continuous_timestamps = np.load(continuous_timestamps_path)
    ephys_TTL_timestamps -= continuous_timestamps[0]   # subract the first timestamp from all TTLs; this way continuous ephys can safely start at 0 samples or seconds
    ephys_TTL_timestamps = ephys_TTL_timestamps / ephys_fs
    continuous_timestamps = continuous_timestamps / ephys_fs


    ttl_channels = [int(i)*sign for i in leds_to_use for sign in [-1,1]]

    ttl_bool = np.isin(channels, ttl_channels)
    ephys_events = np.vstack([ephys_TTL_timestamps[ttl_bool], abs(channels[ttl_bool])-1, np.sign(channels[ttl_bool])]).T
    codes, ephys_latencies = sync.events_to_codes(ephys_events, nchannels=num_leds, minCodeTime=(led_blink_interval-1))
    codes = np.asarray(codes)

    return codes, continuous_timestamps

