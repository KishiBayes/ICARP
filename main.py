"""
Loads FIFs (ICA or otherwise) and performs Epoching at certain annotations
"""
import mne
import numpy as np
from scipy import signal

eventLabel = "x"

def EMGAnalysis(eegfif, emgfif, preTime = 3, postTime = 2, startOffset = 0.5, annotation_label = "x", emg_channels=None, threshold=3, duration=3):

    emgraw = mne.io.read_raw_fif(emgfif)
    eegraw = mne.io.read_raw_fif(eegfif, preload=True).filter(l_freq=0.005, h_freq=50)
    eegraw = eegraw.notch_filter([50,100,150,200])
    jerkTimeDict = get_emg_onset(emgraw, startOffset = startOffset, annotation_label = annotation_label, emg_channels=emg_channels, threshold=threshold, duration=duration)
    n_events = len(jerkTimeDict.keys())
    events = np.c_[list(jerkTimeDict.keys())]*emgraw.info["sfreq"]
    events = np.c_[events, np.zeros(n_events), np.ones(n_events)].astype(int)

    Epochs = mne.Epochs(eegraw, events, tmin=-preTime, tmax=postTime, baseline=(-preTime,0), preload=True)


    return Epochs.average()

def get_emg_onset(raw, startOffset = 1, annotation_label = "x", emg_channels=None, threshold=1, duration=3, plot_emg=True, win_size=100):
    """Find the onset of EMG bursts that occur after a set of timestamps.

    Parameters:
    file
    emg_channels (list of str): Names of EMG channels to search for EMG bursts.
    event_times (array-like): Timestamps to search for EMG bursts after.
    threshold (float): Threshold for detecting EMG bursts.
    duration (float): Minimum duration of an EMG burst.
    plot_emg (boolean): Whether or not to plot the EMG burst evoked object for debugging purposes
    win_size (int): Window duration for EMG power computation


    Returns:
    onset_times (array-like): Array of onset times for EMG bursts.
    """

    # Extract the annotations from the raw object
    annotations = raw.annotations

    # Filter the annotations based on the label
    annotations_x = annotations[annotations.description == annotation_label]

    # Get the start times of all annotations
    annotation_times = annotations_x.onset

    epochStarts = [int(x - startOffset) for x in annotation_times]

    onset_times = {}

    if emg_channels == None:
        emg_channels = []
        for ch_name in raw.ch_names:
            if 'EMG' in ch_name:
                emg_channels.append(ch_name)
    elif type(emg_channels) == str:
        emg_channels = [emg_channels]

    # Loop through each EMG channel and search for EMG bursts
    for ch_name in emg_channels:

        ch_idx = raw.ch_names.index(ch_name)
        data = raw.get_data(picks=[ch_idx])
        #print(np.percentile(data,[10,20,30,40,50,60,70,80,90]))
        sfreq = raw.info['sfreq']

        # Compute baseline power in the 1 second window before each event
        baseline_power = []
        baseline_power_SD = []
        window_power = []
        for event_time in epochStarts:
            baseline_start = int((event_time - startOffset) * sfreq)
            baseline_end = int(event_time * sfreq)
            baseline_power.append(np.mean(np.abs(data[0, baseline_start:baseline_end])))
            #window_power.append(np.mean(np.square(data[0, baseline_start:int(baseline_end+duration*sfreq)])))
            baseline_power_SD.append(np.std(np.abs(data[0, baseline_start:baseline_end])))
        # Compute the threshold for each event based on the baseline power
        thresholds = np.array(baseline_power) + threshold*np.array(baseline_power_SD)


        # Loop through each event and find the onset of any EMG bursts after the event
        for i, event_time in enumerate(epochStarts):
            event_start = int(event_time * sfreq) #NB, at this point, event_time starts at "x" - startOffset
            event_end = int((event_time + duration) * sfreq)
            lead_data = data[0, event_start:event_end]
            # rolling window method:
            # Calculate local power using convolution
            window = np.ones(win_size)/win_size
            power = np.convolve(np.abs(lead_data), window, mode='same')
            
            # Set threshold value
            threshold = thresholds[i]
            
            # Find indices where power exceeds threshold
            try:
                jerk_index = np.where(power > threshold)[0][0]
                print(jerk_index)
                print(power[jerk_index], threshold)

                onset_times[event_time + jerk_index / sfreq] = ch_name
            except IndexError as e:
                print(e)

    if plot_emg:
        n_events = len(onset_times.keys())
        events = np.c_[list(onset_times.keys())] * raw.info["sfreq"]
        events = np.c_[events, np.zeros(n_events), np.ones(n_events)].astype(int)
        EMGEpochs = mne.Epochs(raw, events, preload=True, picks=emg_channels, tmin=-2, tmax=5)
        EMGEpochs.apply_function(lambda x: abs(x))
        EMGEvoked = EMGEpochs.average()
        EMGEvoked.plot()

    return onset_times


if __name__ == "__main__":
    #icaEp, icaEv = GenerateRPEpochs(r"C:\Users\rohan\PycharmProjects\ICARP\TestDir\21227972icaraw.fif")
    #rawEp, rawEv = GenerateRPEpochs(r"C:\Users\rohan\PycharmProjects\ICARP\TestDir\21227972raw.fif")
    emgEv = EMGAnalysis(eegfif=r"C:\Users\rohan\PycharmProjects\ICARP\TestDir\21227972eegica.fif", emgfif=r"C:\Users\rohan\PycharmProjects\ICARP\TestDir\21227972emgraw.fif", emg_channels=["EMG2+"])
    emgEv.plot()
    #mne.viz.plot_compare_evokeds([icaEv, rawEv])
