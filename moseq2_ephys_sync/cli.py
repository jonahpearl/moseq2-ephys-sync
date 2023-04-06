import os
from os.path import join, exists
import click
import ast
import pdb
from moseq2_ephys_sync import workflows

VALID_SOURCE_ABBREVS = ['oe', 'mkv', 'arduino', 'txt', 'csv', 'basler', 'basler_bonsai', 'avi']


# NB: changes to this function must be re-installed via pip!
@click.command()
@click.argument('input-path', type=str)
@click.option('-s1', '--first-source', type=str)
@click.option('-s2', '--second-source', type=str)
@click.option('-o', '--output-dir-name', type=str, default='sync', help='Relative path to output, from input')
@click.option('--led-loc', type=str, help="Location of the syncing LEDs in the video, as seen from plt.imshow()'s point of view. Currenly supported: quadrants (topright, topleft, bottomright, bottomleft), some vertical strips (rightquarter, leftquarter), some horizontal strips (topquarter, topthird, bottomquarter). Add more in extract_leds.py.")
@click.option('--exclude-center', is_flag=True, default=False, help='If true, do not allow LED ROIs in the center 1/9 (ie center of a 3x3 block)')
@click.option('--manual-reverse', is_flag=True, default=False, help='In rare cases: if true, force reversal of led detection order (used to debug sessions where LED4 is missing and hence cannot auto-detect need for reversal)')
@click.option('--led-blink-interval', type=int, default=5, help='LED change interval, in seconds')
@click.option('-s1tf', '--s1-timescale-factor-log10', type=int, help='If in ms, use 3; us, use 6; etc.')
@click.option('-s2tf', '--s2-timescale-factor-log10', type=int, help='If in ms, use 3; us, use 6; etc.')
@click.option('--arduino-spec', type=str, help="DEPRECATED: use header instead! Currently supported: fictive_olfaction, odor_on_wheel, basic_thermistor")
@click.option('--s1-led-rois-from-file', is_flag=True, help="Flag to look for lists of points for source 1 led rois")
@click.option('--s2-led-rois-from-file', is_flag=True, help="Flag to look for lists of points for source 2 led rois")
@click.option('--overwrite-models', is_flag=True)
@click.option('--overwrite-extraction', is_flag=True)
@click.option('--leds-to-use', type=str, default='1234', help='Subset of leds (1-indexed) to use (eg if one was broken) (syntax: --leds_to_use \'1234\'')
@click.option('--exclude-only-off-events', is_flag=True, default=False, help="If true, do not consider events where LEDs only turned off (can be 1 frame delayed in videos due to hardware issues).")
@click.option('--predict-full-timestamps-of-source', '-r', multiple=True, default=None, help='Choose which sources (1, 2, or both) to predict full list of times for (syntax: -r 1 -r 2 --next_arg')
@click.option('--pytesting', is_flag=True, help='If true, return matches and dont actually make the syncing models')
def main(input_path=None,
         first_source=None,
         second_source=None,
         output_dir_name='sync',
         led_loc=None,
         exclude_center=False,
         manual_reverse=False,
         led_blink_interval=5,
         s1_timescale_factor_log10=None,
         s2_timescale_factor_log10=None,
         arduino_spec=None,
         s1_led_rois_from_file=False,
         s2_led_rois_from_file=False,
         overwrite_models=False,
         overwrite_extraction=False,
         leds_to_use='1234',
         exclude_only_off_events=False,
         predict_full_timestamps_of_source=None,
         pytesting=False):

    # Parse leds to use
    if ',' in leds_to_use:
        leds_to_use = leds_to_use.replace(',', '')
    leds_to_use = [str(i) for i in leds_to_use if (len(i)>0 and i!="'")]

    if len(leds_to_use) > 4:
        raise ValueError(f'Expected no more than 4 LEDs but got {len(leds_to_use)} ({leds_to_use})...check that leds_to_use input value is being parsed correctly.')

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
                                exclude_center=exclude_center,
                                manual_reverse=manual_reverse,
                                led_blink_interval=led_blink_interval,
                                s1_timescale_factor_log10=s1_timescale_factor_log10,
                                s2_timescale_factor_log10=s2_timescale_factor_log10,
                                arduino_spec=arduino_spec,
                                s1_led_rois_from_file=s1_led_rois_from_file,
                                s2_led_rois_from_file=s2_led_rois_from_file,
                                overwrite_models=overwrite_models,
                                overwrite_extraction=overwrite_extraction,
                                leds_to_use=leds_to_use,
                                exclude_only_off_events=exclude_only_off_events,
                                sources_to_predict=predict_full_timestamps_of_source,
                                pytesting=pytesting)


if __name__ == "__main__":
    main()
