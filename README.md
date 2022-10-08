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

Here is a janky copy/paste of the current frequently used CLI options to stand in for better documentation:
* @click.option('--input-path', '-i', type=str)
* @click.option('-s1', '--first-source', type=str)
* @click.option('-s2', '--second-source', type=str)
* @click.option('-o', '--output-dir-name', type=str, default='sync', help='Relative path to output, from input')
* @click.option('--led-blink-interval', type=int, default=5, help='LED change interval, in seconds')
* @click.option('--s1-timescale-factor-log10', type=int, help='If in ms, use 3; us, use 6; etc.')
* @click.option('--s2-timescale-factor-log10', type=int, help='If in ms, use 3; us, use 6; etc.')
* @click.option('--overwrite-models', is_flag=True)
* @click.option('--leds_to_use', cls=PythonLiteralOption, default='["1", "2", "3", "4"]', help='Subset of leds (1-indexed) to use (eg if one was broken) (syntax: --leds_to_use ["1", "2", "3", "4"]')
* @click.option('--predict_full_timestamps_of_source', '-r', multiple=True, default=None, help='Choose which sources (1, 2, or both) to predict full list of times for (syntax: ...of_source 1 2 --next_arg')

Here are the other CLI options that are a bit more niche. These mostly have to do with getting the led info from videos:
* @click.option('--overwrite_extraction', is_flag=True)
* @click.option('--led-loc', type=str, help="Location of the syncing LEDs in the video, as seen from plt.imshow()'s point of view. Currenly supported: quadrants (topright, topleft, bottomright, bottomleft), some vertical strips (rightquarter, leftquarter), some horizontal strips (topquarter, topthird, bottomquarter). Add more in extract_leds.py.")
* @click.option('--s1-led-rois-from-file', is_flag=True, help="Flag to look for lists of points for source 1 led rois")
* @click.option('--s2-led-rois-from-file', is_flag=True, help="Flag to look for lists of points for source 2 led rois")

To run an extraction, for example: (note that we don't pass an arduino timescale because I know the default will work; but you can pass it for every source and it won't hurt)
`moseq2_ephys_sync -i /n/groups/datta/Jonah/moseq2-ephys-sync/test_data/ino_basler_test -s1 arduino -s2 basler_bonsai --s2-timescale-factor-log10 9 -o sync`

This will extract the IR LED data from the video and ephys files, find matches in the resulting bit codes, plot the results in `/input_directory/sync/` and save two models that can be used for translating between the two timebases: `video_model.p` which takes as inputs video times (in seconds) and translates them into ephys times; and `ephys_model.p` which conversely takes in ephys times (in seconds) and translated them into video times. 

To use the resulting models, be sure to transform all values to be in seconds before inputting to the models, and if using ephys data, be sure to use zero-subtracted data (i.e. the first value should be 0). Try:
1. `import joblib`
2. `ephys_model = joblib.load('input_directory/sync/ephys_timebase.p')`
3. `video_times = ephys_model.predict(ephys_times.reshape(-1,1))` (assuming times are `1D` arrays)
4. `video_model = joblib.load('input_directory/sync/video_timebase.p')`
5. `ephys_times = video_model.predict(video_times.reshape(-1,1))`



Notes on using the models:
Different workflows transform their inputs in certain ways to help with downstream analysis.
1) All workflows convert timestamps into seconds. This does make certain assumptions that are currently hard-coded, specifically, it is assumed that the arduino timestamps are in milliseconds; it is assumed that ephys is sampled at 30 kHz; it is assumed that AVI timestamps (`device_timestamps.npy` from CW's pyk4a script) are in microseconds; and that mkv / basler timestamps are already in seconds.
2) The TTL workflow has the first time subtracted, such that it begins at 0. This allows it to play nicely with an open ephys glitch.

