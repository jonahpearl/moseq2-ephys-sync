import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 'truetype'
import numpy as np


def plot_code_chunk(first_source_led_codes, first_source_name, second_source_led_codes, second_source_name, save_path):
    """
    Visualize a small chunk of the bit codes. do you see a match? 
    ---
    Input: 
        codes : 2d array
        Array of reconstructed pixel clock codes where:
            codes[:,0] = time (already converted to seconds in main script)
            codes[:,1] = code
            codes[:,2] = trigger channel
        These codes are NOT offset for latencies of the triggered channel
    """

    f,axarr = plt.subplots(2,1,dpi=600,sharex=True)

    axarr[0].plot(first_source_led_codes[:,0] - first_source_led_codes[0,0], first_source_led_codes[:,1],label=first_source_name)
    axarr[0].set_title(first_source_name)

    axarr[1].plot(second_source_led_codes[:,0] - second_source_led_codes[0,0],second_source_led_codes[:,1],label=second_source_name)
    axarr[1].set_title(second_source_name)
    

    plt.xlim([-5,300])
    # plt.xlim([-5,50])
    plt.xlabel('time (sec)')
    plt.ylabel('bit code')
    plt.legend()

    f.savefig(f'{save_path}/match_check_{first_source_name}_and_{second_source_name}.png')

    plt.close(f)


## plot the matched codes against each other:
def plot_matched_scatter(matches, first_source_name, second_source_name, save_path):

    f = plt.figure(dpi=600)

    plt.plot([0,3600],[0,3600],c='k',lw=0.5)

    plt.scatter(matches[:,0],matches[:,1],s=1)

    plt.title('Found %d matches' % len(matches))

    plt.xlabel(f'time of {first_source_name} codes')
    plt.ylabel(f'time of {second_source_name} codes')

    f.savefig(f'{save_path}/matched_codes_scatter_{first_source_name}_and_{second_source_name}.png')

    plt.close(f)

## plot model errors:
def plot_model_errors(time_errors, save_path, outname, fname='model_errors'):

    f = plt.figure(dpi=600)
    ax = plt.hist(time_errors)

    plt.title(f'{np.mean(np.abs(time_errors)):0.3f} sec. mean abs. error in second source Times')
    plt.xlabel('Predicted - actual matched video code times')
    f.savefig(f'{save_path}/{fname}_{outname}.png')
    plt.close(f)

## plot the codes on the same time scale
def plot_matched_times(all_predicted_times, t2_codes, t1_codes, n1, n2, save_path, outname):
    f = plt.figure(dpi=600)

    start,stop =  0,100

    # plot t2 codes on t1 timebase
    plt.plot(all_predicted_times[start:stop] , t2_codes[start:stop,1],lw=2,label=f'Predicting {n1} from {n2}')

    # plot t1 codes on t1 timebase
    plt.plot(t1_codes[start:stop,0], t1_codes[start:stop,1],alpha=0.5,lw=1,label=f'Actual {n1}')

    plt.xlabel('Time (sec)')
    plt.ylabel('Bit Code')
    plt.title('Matched times (ok if not entirely overlapping)')

    plt.legend()
    f.savefig(f'{save_path}/matched_codes_video_time_{outname}.png')
    plt.close(f)

def plot_video_frame(frame, dpi, save_path):
    f = plt.figure(dpi=dpi)

    plt.imshow(frame)
    plt.colorbar()

    f.savefig(save_path)

    plt.close(f)

