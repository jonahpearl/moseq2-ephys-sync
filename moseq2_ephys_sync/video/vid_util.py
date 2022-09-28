from datetime import time
import numpy as np
import pandas as pd
import sys,os
from tqdm import tqdm
from glob import glob
import joblib
import argparse
import pickle
from .video import extract_leds
import decord
from skimage import color
from cv2 import resize

import moseq2_extract.io.video as moseq_video  #TODO: make this a stand-alone module


## General utils
def load_led_rois_from_file(base_path):
    fin = os.path.join(base_path, 'led_rois.pickle')
    with open(fin, 'rb') as f:
        led_roi_dict = pickle.load(f)
    led_roi_list = led_roi_dict['roi_list']
    diff_thresholds = led_roi_dict['thresholds']
    return diff_thresholds, led_roi_list


def gen_batch_sequence(nframes, chunk_size, overlap, offset=0):
    '''
    Generates batches used to chunk videos prior to extraction.

    Parameters
    ----------
    nframes (int): total number of frames
    chunk_size (int): desired chunk size
    overlap (int): number of overlapping frames
    offset (int): frame offset

    Returns
    -------
    Yields list of batches
    '''

    seq = range(offset, nframes)
    out = []
    for i in range(0, len(seq) - overlap, chunk_size - overlap):
        out.append(seq[i:i + chunk_size])
    return out



### MKV HELPER FUNCTIONS ###

def get_mkv_info(fileloc, stream=1):
    stream_features = ["width", "height", "r_frame_rate", "pix_fmt"]

    outs = {}
    for _feature in stream_features:
        command = [
            "ffprobe",
            "-select_streams",
            "v:{}".format(int(stream)),
            "-v",
            "fatal",
            "-show_entries",
            "stream={}".format(_feature),
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            fileloc,
            "-sexagesimal",
        ]
        ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = ffmpeg.communicate()
        if err:
            print(err)
        outs[_feature] = out.decode("utf-8").rstrip("\n")

    # need to get duration and nframes the old fashioned way
    outs["duration"] = get_mkv_duration(fileloc)
    timestamps = get_mkv_timestamps(fileloc,stream)
    outs["nframes"] = len(timestamps)

    return (
        {
            "file": fileloc,
            "dims": (int(outs["width"]), int(outs["height"])),
            "fps": float(outs["r_frame_rate"].split("/")[0])
            / float(outs["r_frame_rate"].split("/")[1]),
            "duration": outs["duration"],
            "pixel_format": outs["pix_fmt"],
            "nframes": outs["nframes"],
        },
        timestamps,
    )

def get_mkv_duration(fileloc, stream=1):
    command = [
        "ffprobe",
        "-v",
        "fatal",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        fileloc,
    ]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()
    if err:
        print(err)
    return float(out.decode("utf-8").rstrip("\n"))


def get_mkv_timestamps(fileloc, stream=1,threads=8):
    command = [
        "ffprobe",
        "-select_streams",
        "v:{}".format(int(stream)),
        "-v",
        "fatal",
        "-threads", str(threads),
        "-show_entries",
        "frame=pkt_pts_time",
        "-of",
        "csv=p=0",
        fileloc,
    ]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()
    if err:
        print(err)
    timestamps = out.decode("utf-8").rstrip("\n").split("\n")
    timestamps = np.array([float(_) for _ in timestamps])
    return timestamps

def get_mkv_stream_names(fileloc):
    stream_tag = "title"

    outs = {}
    command = [
        "ffprobe",
        "-v",
        "fatal",
        "-show_entries",
        "stream_tags={}".format(stream_tag),
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        fileloc,
    ]
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()
    if err:
        print(err)
    out = out.decode("utf-8").rstrip("\n").split("\n")
    
    
    ## !! changed the key/value order here from what JM had: (i.e. so the string name is the key, the stream is the value)
    return dict(list(zip(out,np.arange(len(out)))))


def get_mkv_stream_tag(fileloc, stream=1, tag="K4A_START_OFFSET_NS"):

    command = [
            "ffprobe",
            "-v",
            "fatal",
            "-show_entries",
            "format_tags={}".format(tag),
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            fileloc,
        ]
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()
    if err:
        print(err)
    out = out.decode("utf-8").rstrip("\n")
    return out


    

# def load_led_rois_from_file(base_path):
#     """Old versino of load led rois that was used in mkv workflow
#     """
#     fin = os.path.join(base_path, 'led_rois.pickle')
#     with open(fin, 'rb') as f:
#         led_roi_list = pickle.load(f, pickle.HIGHEST_PROTOCOL)
#     return led_roi_list