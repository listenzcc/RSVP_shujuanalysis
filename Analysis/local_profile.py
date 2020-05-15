import os
import mne
import pandas as pd


def is_fif_file(path):
    """Whether [path] is legal propressed .fif file

    Arguments:
        path {str} -- Path of a file

    Returns:
        {bool} -- True or False
    """
    assert(isinstance(path, str))
    return path.endswith('_ica-raw.fif')


def get_MEG_events(raw, stim_channel='UPPT001'):
    """Get events from MEG data

    Arguments:
        raw {mne_raw} -- Raw MNE object

    Keyword Arguments:
        stim_channel {str} -- The name of stim_channel (default: {'UPPT001'})

    Returns:
        {array} -- Found events
    """
    return mne.find_events(raw, stim_channel=stim_channel)


def get_EEG_events(raw):
    """Get events from EEG data

    Arguments:
        raw {mne_raw} -- Raw MNE object

    Returns:
        {array} -- Found events
    """
    return mne.events_from_annotations(raw)[0]


def make_profiles():
    """Make profiles for getting raw and epochs object

    Returns:
        {dict} -- The profile dictionary
    """
    profiles = pd.DataFrame()

    # Profiles for EEG
    profiles = profiles.append(pd.Series(dict(
        picks='eeg',
        decim=10,
        get_events=get_EEG_events
    ), name='EEG', dtype=object))

    # Profiles for MEG
    profiles = profiles.append(pd.Series(dict(
        picks='mag',
        decim=12,
        get_events=get_MEG_events
    ), name='MEG', dtype=object))

    # Global profiles
    profiles['n_jobs'] = 'cuda'  # N_JOBS
    profiles['tmin'] = TMIN
    profiles['tmax'] = TMAX

    return profiles


# Basic settings
# Directory of data
HOME = '/home/zcc'
DATA_DIR = os.path.join(
    HOME, 'Documents', 'RSVPshuju_analysis', 'processed_data')
MEMORY_DIR = os.path.join(DATA_DIR, '..', 'memory')
try:
    os.mkdir(MEMORY_DIR)
except:
    pass

# The time range of epochs
TMIN, TMAX = -0.2, 1.2

# The frequency range of bands
BANDS = pd.DataFrame(dict(
    Delta=(1, 4),
    Theta=(4, 7),
    Alpha=(8, 12),
    Beta=(13, 25),
    Gamma=(30, 45),
    U07=(0.1, 7),
    U30=(0.1, 30)))
BANDS.rename({0: 'l_freq', 1: 'h_freq'}, inplace=True)


# DataFrame of prepropressed raw fif file information
RAW_DF = pd.DataFrame(columns=['mode', 'data_dir', 'fif_paths'])
# Get the auto generated DataFrame in preprocessing process
autodf = pd.read_json(os.path.join(DATA_DIR, '..', 'auto_path_table.json'))
for exper in autodf.index:
    data_dir = os.path.join(DATA_DIR, exper)
    fif_paths = sorted([os.path.join(data_dir, e)
                        for e in os.listdir(data_dir) if is_fif_file(e)])

    RAW_DF = RAW_DF.append(pd.Series(dict(
        mode=exper[:3],
        data_dir=data_dir,
        fif_paths=fif_paths
    ), name=exper))

# Profiles DataFrame
# How to correctly get MEG or EEG raw and epochs objects
PROFILES = make_profiles()
