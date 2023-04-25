"""
Loads FIFs (ICA or otherwise) and performs Epoching at certain annotations
"""
import mne
import numpy as np

eventLabel = "x"

def GenerateRPEpochs(fif,preTime=0.5,postTime=0.5):
    raw = mne.io.read_raw_fif(fif)
    events, event_id = mne.events_from_annotations(raw, event_id='auto', regexp='^x$', use_rounding=True)
    RPEpochs = mne.Epochs(raw=raw,events=events, tmin=-preTime, tmax=postTime)
    RPEv = RPEpochs.average()
    return RPEpochs, RPEv

def EMGAnalysis(fif,preTime=0.5,postTime=0.5):
    raw = mne.io.read_raw_fif(fif)

    events, event_id = mne.events_from_annotations(raw, event_id='auto', regexp='^x$', use_rounding=True)

    # Define the epoch length and the baseline period for each lead
    epoch_length = 1.0
    baseline = (None, -1.0)

    # Epoch the raw object based on the labelled events
    epochs = mne.Epochs(raw, events, tmin=-1, tmax=1, baseline=baseline,
                        reject=None, preload=True)

    # Compute the standard deviation of the baseline-corrected data for each epoch for each lead
    stds = epochs.get_data().std(axis=2)

    # Find the time when the value of one of the included leads suddenly deviates from the baseline after each labelled event
    jerks = []
    for i, event in enumerate(events):
        event_time = event[0] / raw.info['sfreq']
        epoch_start_time = event_time - 1.0
        epoch_end_time = event_time
        epoch = raw.copy().crop(epoch_start_time, epoch_end_time)
        epoch_stds = epoch.get_data().std(axis=1)
        lead_index = epoch_stds.argmax()
        lead_data = epoch[lead_index]
        threshold = epoch_stds[lead_index]
        jerk_index = np.where(lead_data > threshold)[1][0]
        jerk_time = epoch.times[jerk_index] + epoch_start_time
        jerks.append((int(round(jerk_time * raw.info['sfreq'])), 0, 1))

    # Create a new events object with the times of those jerks
    jerks = mne.events_from_annotations(raw, event_id={'jerk': 1},
                                             regexp=None, use_rounding=True,
                                             chunk_duration=1.0, verbose=None,
                                             name='auto')[0]

    epochs = mne.Epochs(raw=raw, events=jerks, tmin=-preTime, tmax=postTime, preload=True)

    # Apply the abs function to the data in the epochs
    epochs.apply_function(np.abs)

    # Convert the epoch object to an Evoked object using the average function
    evoked = epochs.average()

    # Plot the resulting Evoked object
    evoked.plot()

    return NotImplementedError

if __name__ == "__main__":
    #icaEp, icaEv = GenerateRPEpochs(r"C:\Users\rohan\PycharmProjects\ICARP\TestDir\21227972icaraw.fif")
    #rawEp, rawEv = GenerateRPEpochs(r"C:\Users\rohan\PycharmProjects\ICARP\TestDir\21227972raw.fif")
    emgEv = EMGAnalysis(r"C:\Users\rohan\PycharmProjects\ICARP\TestDir\21227972emgraw.fif")

    #mne.viz.plot_compare_evokeds([icaEv, rawEv])