import os
from os.path import join, exists
import click
import ast
import pdb
from moseq2_ephys_sync import workflows

VALID_SOURCE_ABBREVS = ['oe', 'mkv', 'arduino', 'txt', 'csv', 'basler', 'basler_bonsai', 'avi']

class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


# NB: changes to this function must be re-installed via pip!
@click.command()
@click.option('--input-path', '-i', type=str)
@click.option('-s1', '--first-source', type=str)
@click.option('-s2', '--second-source', type=str)
@click.option('-o', '--output-dir-name', type=str, default='sync')
@click.option('--led-loc', type=str, help="Location of the syncing LEDs in the video, as seen from plt.imshow()'s point of view. Currenly supported: quadrants (topright, topleft, bottomright, bottomleft), some vertical strips (rightquarter, leftquarter), some horizontal strips (topquarter, topthird, bottomquarter). Add more in extract_leds.py.")
@click.option('--led-blink-interval', type=int, default=5, help='LED change interval, in seconds')
@click.option('--arduino-spec', type=str, help="DEPRECATED: use header instead! Currently supported: fictive_olfaction, odor_on_wheel, basic_thermistor")
@click.option('--s1-led-rois-from-file', is_flag=True, help="Flag to look for lists of points for source 1 led rois")
@click.option('--s2-led-rois-from-file', is_flag=True, help="Flag to look for lists of points for source 2 led rois")
@click.option('--overwrite-models', is_flag=True)
@click.option('--overwrite_extraction', is_flag=True)
# @click.option('--leds_to_use', nargs=-1, default=['1', '2', '3', '4'], help='Choose a subset of leds (1-indexed) to use if one was broken (syntax: --leds_to_use 1 2 4 --next_arg...')
@click.option('--leds_to_use', cls=PythonLiteralOption, default='["1", "2", "3", "4"]', help='Subset of leds (1-indexed) to use (eg if one was broken) (syntax: --leds_to_use ["1", "2", "3", "4"]')
@click.option('--predict_full_timestamps_of_source', '-r', multiple=True, default=None, help='Choose which sources (1, 2, or both) to predict full list of times for (syntax: ...of_source 1 2 --next_arg')
# @click.option('--pytesting', cls=PythonLiteralOption, default="[]", help='Select kwargs to return for testing (will not run rest of script!)')
def main(input_path=None,
    first_source=None,
    second_source=None,
    output_dir_name='sync',
    led_loc=None, 
    led_blink_interval=5, 
    arduino_spec=None, 
    s1_led_rois_from_file=False,
    s2_led_rois_from_file=False, 
    overwrite_models=False,
    overwrite_extraction=False,
    leds_to_use=["1", "2", "3", "4"],
    predict_full_timestamps_of_source=None,
    pytesting=None):

    if (first_source is not None and second_source is None) or (first_source is None and second_source is not None):
        raise ValueError('Cannot specify one source but not the other!')
    else:
        if input_path is None:
            raise ValueError('Must specify an input path!')
        workflows.main_function(input_path,
            first_source,
            second_source,
            output_dir_name=output_dir_name,
            led_loc=led_loc, 
            led_blink_interval=led_blink_interval, 
            arduino_spec=arduino_spec, 
            s1_led_rois_from_file=s1_led_rois_from_file,
            s2_led_rois_from_file=s2_led_rois_from_file, 
            overwrite_models=overwrite_models,
            overwrite_extraction=overwrite_extraction,
            leds_to_use=leds_to_use,
            sources_to_predict=predict_full_timestamps_of_source)



if __name__ == "__main__" :
    main()
