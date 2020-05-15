import os
import mne
import json
import pandas as pd

from local_profile import PROFILES, BANDS, MEMORY_DIR, RAW_DF


def get_raw(series, profiles=PROFILES):
    """Get raw and profile

    Arguments:
        series {Series} -- A pandas Series for raw .fif files information

    Keyword Arguments:
        profiles {dict} -- Profile dictionary (default: {PROFILES})

    Returns:
        raw -- The raw object
        profile -- The profile of the raw object
    """
    assert(hasattr(series, 'fif_paths'))
    assert(hasattr(series, 'mode'))
    raw = mne.concatenate_raws([mne.io.read_raw_fif(e)
                                for e in series['fif_paths']])
    profile = profiles.loc[series['mode']]
    return raw, profile


def relabel(events, sfreq):
    """Re-label 2-> 4 when 2 is near to 1

    Arguments:
        events {array} -- The events array, [[idx], 0, [label]],
                          assume the [idx] column has been sorted.
        sfreq {float} -- The sample frequency

    Returns:
        {array} -- The re-labeled events array
    """
    # Init the pointer [j]
    j = 0
    # For every '1' event, remember as [a]
    for a in events[events[:, -1] == 1]:
        # Do until...
        while True:
            # Break,
            # if [j] is far enough from latest '2' event,
            # it should jump to next [a]
            if events[j, 0] > a[0] + sfreq:
                break
            # Switch '2' into '4' event if it is near enough to the [a] event
            if all([events[j, -1] == 2,
                    abs(events[j, 0] - a[0]) < sfreq]):
                events[j, -1] = 4
            # Add [j]
            j += 1
            # If [j] is out of range of events,
            # break out the 'while True' loop.
            if j == events.shape[0]:
                break
    # Return re-labeled [events]
    return events


def get_epochs(running_name, band, bands=BANDS, memory_name=None):
    """Get epochs

    Arguments:
        running_name {str} -- The running name of the epochs, like 'MEG_S03'
        band {str} -- The band name, it should in [bands]

    Keyword Arguments:
        bands {dict} -- The bands directory (default: {BANDS})
        memory_name {str} -- The epochs will be remembered as the name (default: {None})

    Returns:
        {mne_epochs} -- The epochs MNE object
    """
    # Try to recall the epochs as [memory_name]
    if memory_name is not None:
        # If memory exists,
        # use it
        if os.path.exists(os.path.join(MEMORY_DIR, memory_name)):
            print(f'{memory_name} exists, using it.')
            return mne.read_epochs(os.path.join(MEMORY_DIR, memory_name))

    # Automatically get [raw] and [profile]
    raw, profile = get_raw(RAW_DF.loc[running_name])
    # Get parameters
    l_freq, h_freq = bands[band]
    picks = profile.picks
    tmin = profile.tmin
    tmax = profile.tmax
    decim = profile.decim
    n_jobs = profile.n_jobs

    # Get events array and re-label it
    events = profile.get_events(raw)
    sfreq = raw.info['sfreq']
    events = relabel(events, sfreq)

    # Copy and filter
    raw_copy = raw.copy()
    raw_copy.load_data()
    raw_copy.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)

    # Get epochs
    epochs = mne.Epochs(raw_copy, events=events, picks=picks,
                        tmin=tmin, tmax=tmax, decim=decim)
    epochs.drop_bad()

    # Remember the epochs as [memory_name]
    if memory_name is not None:
        print(f'New memory: {memory_name}')
        epochs.save(os.path.join(MEMORY_DIR, memory_name))

    return epochs
