'''
Tools for extracting LED states from video files
'''

import os
import numpy as np
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.morphology import binary_erosion, binary_dilation, disk
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.measure import regionprops
from skimage.draw import polygon
from moseq2_ephys_sync.plotting import plot_code_chunk, plot_matched_scatter, plot_model_errors, plot_video_frame
import pdb


def get_led_data_from_rois(frame_data_chunk, led_roi_list, movie_type, diff_thresholds=None,  save_path=None):
    """
    Given pre-determined rois for LEDs, return sequences of ons and offs
    Inputs:
        frame_data_chunk: array-like of video data (typically from moseq_video.load_movie_data() but could be any)
        rois (list): ordered list of rois [{specify ROI format, eg x1 x2 y1 y2}] to get LED data from
        led_thresh (int): value above which LEDs are considered on. Default 2e4. In the k4a recorder, LEDs that are on register as 65535 = 6.5e4, off ones are roughly 1000.
        save_path (str): where to save plots for debugging if desired
    Returns:
        leds (np.array): (num leds) x (num frames) array of 0s and 1s, indicating if LED is above or below threshold (ie, on or off)
    """

    leds = []

    # Set up thresholds to detect LED ON/OFF events.
    # This will be used like: np.diff(led_signal) > threshold and vice versa for off.
    if diff_thresholds is None:
        if movie_type == 'mkv':
            led_thresh = 2e4
        elif movie_type == 'avi':
            led_thresh = None  # dynamically determine below with otsu
        elif movie_type == 'basler':  # encoded as uint8 for now
            led_thresh = 250
    elif diff_thresholds is not None:
        assert len(diff_thresholds) == len(led_roi_list)

    for i in range(len(led_roi_list)):

        # Points in the frame that correspond to the LED
        pts = led_roi_list[i]

        # Mean frame-by-frame LED signal for this chunk
        led = frame_data_chunk[:, pts[0], pts[1]].mean(axis=1)  # (slice returns a nframes x npts array, then you get mean of all the pts at each time)

        # Specify different threshold for each LED if necessary
        if diff_thresholds is not None:
            led_thresh = diff_thresholds[i]

        # Detect events
        led_on = np.where(np.diff(led) > led_thresh)[0]   #rise indices
        led_off = np.where(np.diff(led) < -led_thresh)[0]   #fall indices

        # Store data
        led_vec = np.zeros(frame_data_chunk.shape[0])
        led_vec[led_on] = 1
        led_vec[led_off] = -1

        leds.append(led_vec)

    leds = np.vstack(leds) #spiky differenced signals to extract times 
    detected_count = np.sum(leds!=0, axis=1)
    print(f'Num detected events: {detected_count}')

    return leds



def relabel_labeled_leds(labeled_led_img):
    """Relabels arbitrary non-zero labels to be 1,2,3,4... (0 is background)
    """
    vals = np.unique(labeled_led_img)
    for i,val in enumerate(vals):
        if i == 0:
            continue
        else:
            labeled_led_img[labeled_led_img == val] = i
    return labeled_led_img


def extract_initial_labeled_image(frames_uint8, movie_type):
    """Use std (usually) of frames and otsu thresholding to extract LED positions
    movie_type:
    """
    
    std_px = frames_uint8.std(axis=0)    
    mean_px = frames_uint8.mean(axis=0)
    vary_px = std_px if np.std(std_px) < np.std(mean_px) else mean_px # pick the one with the lower variance
    
    # Get threshold for LEDs
    if movie_type == 'mkv':
        thresh = threshold_otsu(vary_px)
    elif movie_type == 'avi' or movie_type == 'basler':
        thresh = threshold_multiotsu(vary_px,5)[-1]  # take highest threshold from multiple
    
    # Get mask
    if movie_type == 'mkv' or movie_type == 'avi':
        thresh_px = np.copy(vary_px)
        thresh_px[thresh_px<thresh] = 0
        
        # Initial regions from mask
        edges = canny(thresh_px/255.) ## find the edges
        filled_image = ndi.binary_fill_holes(edges) ## fill its edges
        labeled_led_img, num_features = ndi.label(filled_image) ## get the clusters
    
    elif movie_type == 'basler':
        thresh_px = np.copy(vary_px)
        thresh_px = (thresh_px > thresh)
        thresh_px = binary_erosion(thresh_px, disk(5))  # separate LEDs
        thresh_px = binary_dilation(thresh_px)  # fix ragged edges

        # Initial regions from mask
        edges = canny(thresh_px) ## find the edges
        filled_image = ndi.binary_fill_holes(edges) ## fill its edges
        labeled_led_img, num_features = ndi.label(filled_image) ## get the clusters


    return num_features, filled_image, labeled_led_img


def clean_by_location(filled_image, labeled_led_img, led_loc):
    """Take labeled led image, and a location, and remove labeled regions not in that loc
    led_loc (str): Location of LEDs in an plt.imshow(labeled_leds)-oriented plot. Options are topright, topleft, bottomleft, or bottomright.
    """
    centers_of_mass = ndi.measurements.center_of_mass(filled_image, labeled_led_img, range(1, np.unique(labeled_led_img)[-1] + 1))  # exclude 0, which is background
    centers_of_mass = [(x/filled_image.shape[0], y/filled_image.shape[1]) for (x,y) in centers_of_mass]  # normalize
    # Imshow orientation: x is the vertical axis of the image and runs top to bottom; y is horizontal and runs left to right. (0,0 is top-left)
    if led_loc == 'topright':
        idx = np.asarray([((x < 0.5) and (y > 0.5)) for (x,y) in centers_of_mass]).nonzero()[0]
    elif led_loc == 'topleft':
        idx = np.asarray([((x < 0.5) and (y < 0.5)) for (x,y) in centers_of_mass]).nonzero()[0]
    elif led_loc == 'bottomleft':
        idx = np.asarray([((x > 0.5) and (y < 0.5)) for (x,y) in centers_of_mass]).nonzero()[0]
    elif led_loc == 'bottomright':
        idx = np.asarray([((x > 0.5) and (y > 0.5)) for (x,y) in centers_of_mass]).nonzero()[0]
    elif led_loc == 'bottomquarter':
        idx = np.asarray([(x > 0.75) for (x,y) in centers_of_mass]).nonzero()[0]
    elif led_loc == 'topquarter':
        idx = np.asarray([(x < 0.25) for (x,y) in centers_of_mass]).nonzero()[0]
    elif led_loc == 'topthird':
        idx = np.asarray([(x < 0.33) for (x,y) in centers_of_mass]).nonzero()[0]
    elif led_loc == 'rightquarter':
        idx = np.asarray([(y > 0.75) for (x,y) in centers_of_mass]).nonzero()[0]
    elif led_loc == 'leftquarter':
        idx = np.asarray([(y < 0.25) for (x,y) in centers_of_mass]).nonzero()[0]
    else:
        RuntimeError('led_loc not recognized')
    
    # Add back one to account for background. Ie, if 3rd center of mass was in correct loc, this corresponds to label 4 in labeled_leds
    idx = idx+1
    
    # Remove non-LED labels
    labeled_led_img[~np.isin(labeled_led_img, idx)] = 0
    
    # Relabel 
    labeled_led_img = relabel_labeled_leds(labeled_led_img)

    return labeled_led_img


def clean_by_size(labeled_led_img, lower_size_thresh, upper_size_thresh):
    """Remove ROIs with num pixels less than lower thresh or greater than upper thresh. Can pass None to either to ignore.
    """

    ## erase small rois:
    if lower_size_thresh:
        labels_to_erase = [label for label in np.unique(labeled_led_img) if (len(np.where(labeled_led_img==label)[0]) < lower_size_thresh and label > 0) ]
        for erase in labels_to_erase:
            print('Erasing extraneous label #%d based on too small size...' % erase)
            labeled_led_img[labeled_led_img==erase] = 0
    
    ## erase large rois
    if upper_size_thresh:
        labels_to_erase = [label for label in np.unique(labeled_led_img) if (len(np.where(labeled_led_img==label)[0]) > upper_size_thresh and label > 0) ]
        for erase in labels_to_erase:
            print('Erasing extraneous label #%d based on too large size...' % erase)
            labeled_led_img[labeled_led_img==erase] = 0

    # Relabel 
    labeled_led_img = relabel_labeled_leds(labeled_led_img)

    return labeled_led_img

def get_roi_sorting(labeled_led_img, led_labels, sort_by):
    ## get LED x and y positions for sorting
    leds_xs = [np.where(labeled_led_img==i)[1].mean() for i in led_labels] 
    leds_ys = [np.where(labeled_led_img==i)[0].mean() for i in led_labels]  
    
    # LEDs are numbered 1-4; figure out how to order them
    # If not specified, sort by where there's most variance    
    if sort_by is None: 
        print('Sorting LEDs by variance...if no matches found, check LED sorting!')
        if np.std(leds_xs) > np.std(leds_ys): # sort leds by the x coord:
            sorting = np.argsort(leds_xs)
        else:
            sorting = np.argsort(leds_ys)
    elif sort_by == 'vertical':
          sorting = np.argsort(leds_ys)
    elif sort_by == 'horizontal':
        sorting = np.argsort(leds_xs)
    else:
        Warning('Argument to sort_by not recognized, using variance')
        if np.std(leds_xs) > np.std(leds_ys): # sort leds by the x coord:
            sorting = np.argsort(leds_xs)
        else:
            sorting = np.argsort(leds_ys)
    
    return sorting

def extract_roi_events(labeled_led_img, led_labels, sorting, frame_data_chunk, movie_type):
    
    # List to hold events by frame
    leds = []
    
    for i in range(len(sorting)):
        led_x = np.where(labeled_led_img==led_labels[sorting[i]])[0]
        led_y = np.where(labeled_led_img==led_labels[sorting[i]])[1]
        led = frame_data_chunk[:,led_x,led_y].mean(axis=1) #on/off block signals
    

        # Detect large changes in LED ROIs to detect on/off events
        # NB: led_event_thresh refers to the np.diff'd signal!

        # If using avi, the range is pretty small, so use otsu to pick a good dividing number, then simplify to 0 or 1.
        if movie_type == 'avi':
            led_on_thresh = threshold_otsu(led)
            detection_vals = (led > led_on_thresh).astype('int')  # 0 or 1 --> diff is -1 or 1
            led_event_thresh = 0
        elif movie_type == 'mkv':
            detection_vals = led
            led_event_thresh = 2e4  # ie, a change of 2e4 or greater
        elif movie_type == 'basler':
            detection_vals = led
            led_event_thresh = 100  # ie, a change of 100 or greater. basler is uint8, LEDs go from ~250 to ~50.

        led_on = np.where(np.diff(detection_vals) > led_event_thresh)[0]   #rise indices
        led_off = np.where(np.diff(detection_vals) < (-1*led_event_thresh))[0]   #fall indices
        led_vec = np.zeros(frame_data_chunk.shape[0])
        led_vec[led_on] = 1
        led_vec[led_off] = -1
        leds.append(led_vec)
    
    leds = np.vstack(leds) # (nLEDs x nFrames), spiky differenced signals to extract times   

    return leds


def check_led_order(leds, num_leds):
    reverse = 0
    num_events_per_led = np.sum(leds!=0, axis=1)
    max_event_idx = np.where(num_events_per_led == np.max(num_events_per_led))[0]
    
    # Catch rare case where two LEDs share max event count
    if len(max_event_idx)>1:
        if (0 in max_event_idx) and not ((num_leds-1) in max_event_idx):
            reverse = 1
        elif ((num_leds-1) in max_event_idx) and not (0 in max_event_idx):
            pass
        elif (0 in max_event_idx) and ((num_leds-1) in max_event_idx):
            Warning('First / last LEDs had same num events. Cannot validate LED order!')
        else:
            Warning('Multiple max events in LEDs and was not first or last in sort!')
    elif max_event_idx == 0: # Deal with usual case
        reverse = 1
    elif max_event_idx == (num_leds-1):
        pass
        
    return reverse

def split_largest_region(labeled_img, thickness = 10):

    # Find region to split
    rp = regionprops(labeled_img)
    # max_idx = np.argmax([rg.area for rg in rp])
    max_idx = np.argmax([rg.eccentricity for rg in rp])

    # Split the region along its minor axis (ie into 2 LEDs):

    # Find the minor axis endpoints
    center = rp[max_idx].centroid
    dx = np.cos(rp[max_idx].orientation) * rp[max_idx].minor_axis_length 
    dy = np.sin(rp[max_idx].orientation) * rp[max_idx].minor_axis_length
    p1 = (center[0]+dy, center[1]-dx)
    p1_int = tuple(map(int,p1))
    p2 = (center[0]-dy, center[1]+dx)
    p2_int = tuple(map(int,p2))

    # Draw the minor axis with some thickness
    dx_thickness = np.sin(rp[max_idx].orientation) * thickness
    dy_thickness = np.cos(rp[max_idx].orientation) * thickness
    r1 = (p1[0]-dy_thickness, p1[1]-dx_thickness)  # 1,2,3,4 starting top left moving clockwise
    r4 = (p1[0]+dy_thickness, p1[1]+dx_thickness)
    r2 = (p2[0]-dy_thickness, p2[1]-dx_thickness)
    r3 = (p2[0]+dy_thickness, p2[1]+dx_thickness)
    p = polygon((r1[0], r2[0], r3[0], r4[0]), (r1[1], r2[1], r3[1], r4[1]))  # the minor axis with thickness
    
    # Split the image
    labeled_img[p] = 0

    # Relabel (the "relabel_labeled_leds()" function just remaps, it doesn't look for duplicated regions, so need to use ndi.label)
    filled_img = (labeled_img>0).astype('uint8')
    labeled_img, num_features = ndi.label(filled_img)
    
    return p,labeled_img


def get_led_data_with_stds(frame_data_chunk, movie_type, num_leds = 4, chunk_num=0, led_loc=None,
    flip_horizontal=False, flip_vertical=False, sort_by=None, save_path=None):
    """
    Uses std across frames + otsu + cleaning + knowledge about the event sequence to find LED ROIs in a chunk of frames.
    In AVIs, since they're clipped to int8, cleaning is harder. Might be able to solve this by casting to int16 and bumping any value above 250 to 2^16, but it's risky.
    
    frame_data_chunk: nframes, nrows, ncols
    movie_type (str): 'mkv' or 'avi'. Will adjust thresholding beacuse currently avi's with caleb's clipping (uint8) don't have strong enough std
    
    """

    # Flip frames if requested
    if flip_horizontal:
        print('Flipping image horizontally')
        frame_data_chunk = frame_data_chunk[:,:,::-1]
    if flip_vertical:
        print('Flipping image vertically')
        frame_data_chunk = frame_data_chunk[:,::-1,:]
    

    # Convert to uint8
    frames_uint8 = np.asarray(frame_data_chunk / frame_data_chunk.max() * 255, dtype='uint8')

    # Get initial labeled image
    num_features, filled_image, labeled_led_img = extract_initial_labeled_image(frames_uint8, movie_type)
    
    # If too many features, try a series of cleaning steps. Labeled_leds has 0 for background, then 1,2,3...for ROIs of interest

    # If still too many features, check for location parameter and filter by it
    if (num_features > num_leds) and led_loc:
        print('Too many features, using provided LED position...')
        labeled_led_img = clean_by_location(filled_image, labeled_led_img, led_loc)

    # Recompute num features (minus 1 for background)
    num_features = len(np.unique(labeled_led_img)) - 1

    # If still too many features, remove small ones
    if (num_features > num_leds):
        print('Oops! Number of features (%d) did not match the number of LEDs (%d)' % (num_features,num_leds))
        if movie_type == 'mkv' or movie_type == 'avi':
            labeled_led_img = clean_by_size(labeled_led_img, lower_size_thresh=20, upper_size_thresh=100)
        elif movie_type == 'basler':
            # check size of blob in 33000, 36000. Threshold needs to be above that!
            labeled_led_img = clean_by_size(labeled_led_img, lower_size_thresh=500, upper_size_thresh=None)  # sometimes leds are merged so dont remove large ones
    
    # Recompute num features (minus 1 for background)
    num_features = len(np.unique(labeled_led_img)) - 1

    # If using Basler, sometimes two LEDs get merged. If there are too few LEDs, take the biggest one, fit an ellipse, and cut it in half along its minor axis.
    if (num_features < num_leds) and movie_type == 'basler':
        print(f'Oops! Too few features {num_features} in Basler frame, trying to split largest feature')
        split_line, labeled_led_img = split_largest_region(labeled_led_img, thickness = 10)

    
    # Show led labels for debugging
    image_to_show = np.copy(labeled_led_img)
    # for i in range(1,5):
    #     image_to_show[labeled_leds==(sorting[i-1]+1)] = i
    plot_video_frame(image_to_show, 200, f'{save_path}/{movie_type}_frame_{chunk_num}_led_labels_preEvents.png')

    led_labels = [label for label in np.unique(labeled_led_img) if label > 0 ]
    assert led_labels == sorted(led_labels)  # note that these labels aren't guaranteed only the correct ROIs yet... but the labels should be strictly sorted at this point.
    print(f'Found {len(led_labels)} LED ROIs after size- and location-based cleaning...')        

    # At this point, sometimes still weird spots, but they're roughly LED sized.
    # So, to distinguish, get event data and then look for things that don't 
    # look like syncing LEDs in that data.

    # We use led_labels to extract events for each ROI.
    # sorting will be a nLEDs-length list, zero-indexed sort based on ROI horizontal or vertical position.
    # leds wil be an np.array of size (nLEDs, nFrames) with values 1 (on) and -1 (off) for events,
        #  and row index is the sort value.
    # So led_labels[sorting[0]] is the label of the ROI the script thinks belongs to LED #1,
        # and leds[sorting[0]] is the sequence of ONs and OFFs for that LED.
    sorting = get_roi_sorting(labeled_led_img, led_labels, sort_by)
    leds = extract_roi_events(labeled_led_img, led_labels, sorting, frame_data_chunk, movie_type)


    # In the ideal case, there are 4 ROIs, extract events, double check LED 4 is switching each time, and we're done.
    if leds.shape[0] == num_leds:
        reverse = check_led_order(leds, num_leds)
    else:
        # Sometimes though you get little contaminating blips that look like LEDs.
        # They don't tend to have many events -- remove blip with lowest num of events.
        while leds.shape[0] > num_leds:
            # Choose idx to remove
            min_event_idx = np.argmin(np.sum(leds!=0, axis=1))
            row_bool = ~np.isin(np.arange(leds.shape[0]),min_event_idx)
            # Remove it
            print(f'Removing roi #{min_event_idx} based on low event count...')
            leds = leds[row_bool,:]  # drop row
            labeled_led_img[labeled_led_img==led_labels[sorting[min_event_idx]]] = 0  # set ROI to bg
            sorting = sorting[row_bool]  # remove from sort
            
        # Figure out which LED is #4
        reverse = check_led_order(leds, num_leds)
    
    if reverse:
        leds = leds[::-1,:]

    # Re-plot labeled led img, with remaining four led labels mapped to their sort order.
    # Use tmp because if you remap, say, 2 --> 3 before looking for 3, then when you look for 3, you'll also find 2.
    image_to_show = np.copy(labeled_led_img)
    tmp_idx_to_update = []
    for i in range(len(sorting)):
        tmp_idx_to_update.append(image_to_show == led_labels[sorting[i]])
    for i in range(len(sorting)):
        image_to_show[tmp_idx_to_update[i]] = (i+1)
    plot_video_frame(image_to_show, 200, '%s/%s_frame_%d_sort_order_postEvents.png' % (save_path, movie_type, chunk_num) )

    return leds
    

def get_events(leds, timestamps):
    """
    Convert list of led ons/offs + timestamps into list of ordered events

    Inputs:
        leds(np.array): num leds x num frames
        timestamps (np.array): 1 x num frames
    """
    ## e.g. [123,1,-1  ] time was 123rd frame, channel 1 changed from on to off... 

    times = []
    directions = []
    channels = []

    direction_signs = [1, -1]
    led_channels = range(leds.shape[0]) ## actual number of leds in case one is missing in this particular chunk. # range(num_leds)
    
    for channel in led_channels:

        for direction_sign in direction_signs:

            times_of_dir = timestamps[np.where(leds[channel,:] == direction_sign)]  #np.where(leds[channel,:] == direction_sign)[0] + time_offset ## turned off or on
                        
            times.append(times_of_dir)
            channels.append(np.repeat(channel,times_of_dir.shape[0]))
            directions.append(np.repeat(direction_sign,times_of_dir.shape[0] ))


    times = np.hstack(times)
    channels = np.hstack(channels).astype('int')
    directions = np.hstack(directions).astype('int')
    sorting = np.argsort(times)
    events = np.vstack([times[sorting],channels[sorting],directions[sorting]]).T
      
    return events