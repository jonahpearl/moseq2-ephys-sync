Tools to sync ephys data with video data using IR LEDs, specifically using Open Ephys and Azure Kinect (mkv files) data. The sync signal comes from IR LEDs in view of the camera, whose trigger signals are also routed as TTL inputs to the ephys acquisition system. The LED on/off states are converted to bit codes, and the resulting sequences of bit codes in the video frames and ephys data are matched. The matches are used to build linear models (piecewise linear regression) that translate video time into ephys time and vice versa. 

To get started, try the following in termanal (these instructions assume you're on a compute node on a cluster):

1. `conda create -n sync_test python=3.7`
2. `conda activate sync_test`
3. `cd ~/code/` (or wherever you want the repo to live
4. `git clone https://github.com/jonahpearl/moseq2-ephys-sync.git`
5. `cd ./moseq2-ephys-sync/`
6. `git switch refactor_workflows`
6. `pip install -e .`
9. [if using video modules] `module load ffmpeg`

The script assumes your input folder structure looks like this:
```
/input_directory/
│   
│       
│
└───ephys_folder/ (this might be e.g. Record Node 10X/ or experiment1/ depending on the version of the Open Ephys GUI you're using) 
│   │
│   └───recording1/
│       │
│       └───events/
│           │
│           └───Rhythm_FPGA-10X.0/
│               │
│               └───TTL_1/
│                     channel_states.npy
│                     channels.npy
│                     full_words.npy
│                     timestamps.npy
│                   
│
└───depth.mkv (can be named anything, with an .mkv extension)
```

To run an extraction:
`python main.py -path /input_directory/`

This will extract the IR LED data from the video and ephys files, find matches in the resulting bit codes, plot the results in `/input_directory/sync/` and save two models that can be used for translating between the two timebases: `video_model.p` which takes as inputs video times (in seconds) and translates them into ephys times; and `ephys_model.p` which conversely takes in ephys times (in seconds) and translated them into video times. 

To use the resulting models, be sure to transform all values to be in seconds before inputting to the models, and if using ephys data, be sure to use zero-subtracted data (i.e. the first value should be 0). Try:
1. `import joblib`
2. `ephys_model = joblib.load('input_directory/sync/ephys_timebase.p')`
3. `video_times = ephys_model.predict(ephys_times.reshape(-1,1))` (assuming times are `1D` arrays)
4. `video_model = joblib.load('input_directory/sync/video_timebase.p')`
5. `ephys_times = video_model.predict(video_times.reshape(-1,1))`



Notes on using the models:
Different workflows transform their inputs in certain ways to help with downstream analysis.
1) All workflows convert timestamps into seconds. This does make certain assumptions that are currently hard-coded, specifically, it is assumed that the arduino timestamps are in milliseconds; it is assumed that ephys is sampled at 30 kHz; it is assumed that AVI timestamps (`device_timestamps.npy` from CW's pyk4a script) are in microseconds; and that mkv / basler timestamps are already in seconds. Currently, no options exist to override these defaults, so if you need to, I'd make a new branch and edit them.
2) The TTL workflow has the first time subtracted, such that it begins at 0. This allows it to play nicely with an open ephys glitch.

