import mne
import os
import re
from mne_icalabel import label_components
from mne.preprocessing import ICA

def get_info_with_mne(file_path):
    """ read info from the edf file without loading the data. loading data is done in multiprocessing since it takes
    some time. getting info is done before because some files had corrupted headers or weird sampling frequencies
    that caused the multiprocessing workers to crash. therefore get and check e.g. sampling frequency and duration
    beforehand
    :param file_path: path of the recording file
    :return: file name, sampling frequency, number of samples, number of signals, signal names, duration of the rec
    """
    try:
        raw = mne.io.read_raw_edf(file_path, verbose='error')
    except ValueError:
        return None, None, None, None, None, None
    # some recordings have a very weird sampling frequency. check twice before skipping the file
    sampling_frequency = int(raw.info['sfreq'])
    if sampling_frequency < 10:
        sampling_frequency = 1 / (raw.times[1] - raw.times[0])
        if sampling_frequency < 10:
            return None, sampling_frequency, None, None, None, None
    n_samples = raw.n_times
    signal_names = raw.ch_names
    n_signals = len(signal_names)
    # some weird sampling frequencies are at 1 hz or below, which results in division by zero
    duration = n_samples / max(sampling_frequency, 1)

    return raw, sampling_frequency, n_samples, n_signals, signal_names, duration


def rename_tuh_channels(input_string):
    """Rename TUH channels and ignore non-EEG and custom channels.
    Rules:
    - 'Z' should always be lowercase.
    - 'P' following a 'F' should be lowercase.
    """
    if input_string.startswith("EEG ") and (input_string.endswith("-A1+A2") or input_string.endswith("-G2")):
        input_string = re.findall(r"\s(.*?)\-", input_string)
        return input_string[0].replace('FP', 'Fp').replace('Z', 'z')
    elif input_string.startswith("EEG ") and input_string.endswith("-AVG"):
        input_string = re.findall(r"\s(.*?)\-", input_string)
        return input_string[0].replace('FP', 'Fp').replace('Z', 'z') + "-AVG"
    else:
        return input_string


def mappingMaker(rawChannelNames):
    # function that builds a mapping dict
    notFound = []
    mapping = {}
    for chan in rawChannelNames:
        if chan in mapping.keys():
            pass
        else:
            mapping[chan] = rename_tuh_channels(chan)
    return mapping

def processEdf(fname):
    raw, sfreq, n_samples, n_channels, chan_names, n_sec = get_info_with_mne(
        fname)
    sampling_freq = sfreq

    raw.load_data()

    montage = mne.channels.make_standard_montage('standard_1020')
    mapping = mappingMaker(raw.ch_names)
    submapping = {k:v for k,v in mapping.items() if k in raw.ch_names}
    exclude = [v for v in submapping.values() if v.startswith("E") or v.startswith("M")]
    raw.rename_channels(submapping)
    #copy EMG leads and save as a seperate file
    EMGraw = raw.copy().drop_channels([v for v in submapping.values() if not v.startswith("EMG")])
    EMGraw.save(str(fname.path).split(".")[0] + "emgraw.fif", overwrite=True)

    raw.drop_channels(exclude)  # removing all channels not in montage
    raw.set_montage(montage)

    #Save unedited version as a fif
    raw.save(str(fname.path).split(".")[0] + "eegraw.fif", overwrite=True)

    n_comps = len(raw.ch_names)
    icaraw = Auto_ICA(raw, n_comps, seed=69)

    #save ICA'd version as a fif
    icaraw.save(str(fname.path).split(".")[0] + "eegica.fif", overwrite=True)

    return icaraw

def Auto_ICA(raw, n_comp, seed=1123):
    filt_raw = raw.copy().filter(l_freq=0.005, h_freq=50)
    ica = ICA(n_components=n_comp, max_iter='auto', random_state=seed, method="infomax")
    ica.fit(filt_raw)

    filt_raw.load_data()

    ic_labels = label_components(filt_raw, ica, method="iclabel")

    # ICA0 was correctly identified as an eye blink, whereas ICA12 was
    # classified as a muscle artifact.

    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    print(f"Excluding these ICA components: {exclude_idx}")

    return ica.apply(filt_raw)

def GetEDFs(dir):
    EdfFiles = []
    Directory = os.scandir(dir)
    for file in Directory:
        if file.is_dir():
            # Go deeper
            subdEdfFiles = GetEDFs(file.path)
            EdfFiles += subdEdfFiles
        elif os.path.splitext(file.name)[1] == ".edf":
            EdfFiles.append(file)
    return EdfFiles
def GetFIFs(dir):
    FIFFiles = []
    Directory = os.scandir(dir)
    for file in Directory:
        if file.is_dir():
            # Go deeper
            subdFIFFiles = GetFIFs(file.path)
            FIFFiles += subdFIFFiles
        elif os.path.splitext(file.name)[1] == ".fif":
            FIFFiles.append(file)
    return FIFFiles

def MainSortLoop(dir):
    EdfFiles = GetEDFs(dir)
    for file in EdfFiles:
        raw = processEdf(file)

RPDir = r"C:\Users\rohan\OneDrive - NHS\EEGs\BRP"
TestDir = r"C:\Users\rohan\PycharmProjects\ICARP\TestDir"

if __name__ == "__main__":
    MainSortLoop(TestDir)


