# %%
# System
import os
import sys
import multiprocessing
import time as builtin_time

# Computing
import numpy as np
from sklearn.manifold import TSNE
from sklearn import cluster as CLUSTER
from scipy.stats import entropy as ENTROPY

# Ploting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Local settings
sys.path.append('..')  # noqa
from local_toolbox import get_raw, get_epochs


# %%
# Custom RUNNING_NAME
# RUNNING_NAME = 'MEG_S03'

# Input RUNNING_NAME
print(sys.argv[1])
if len(sys.argv) > 1:
    RUNNING_NAME = sys.argv[1]

# Set N_CLUSTERS
if RUNNING_NAME.startswith('EEG'):
    N_CLUSTERS = 3
if RUNNING_NAME.startswith('MEG'):
    N_CLUSTERS = 7

# Safety assert,
# RUNNING_NAME is legal,
# N_CLUSTERS is legal.
assert(N_CLUSTERS > 0)

print(f'RUNNING_NAME: {RUNNING_NAME}')
FIGURES = []
SHOW = False

epochs = get_epochs(RUNNING_NAME, band='U07',
                    memory_name=f'{RUNNING_NAME}-U07-epo.fif')

epochs.apply_baseline()

for idx in ['1', '2', '3']:
    title = f'{RUNNING_NAME}-{idx}'
    print(title)
    evoked = epochs[idx].average()
    fig = evoked.plot_joint(times=np.linspace(0, 1.0, 8, endpoint=False),
                            title=f'Evoked of {title}', show=SHOW)
    FIGURES.append(fig)

epochs.load_data()
epochs.crop(tmin=0)
EPOCHS = epochs
EPOCHS

# %%
DATA = EPOCHS['1'].get_data()
NUM, CHANNELS, TIMES = DATA.shape
NUM, CHANNELS, TIMES

# %%


def histogram(d, bounds, bins=100):
    """Compute histogram

    Arguments:
        d {array} -- Data to compute histogram from
        bounds {tuple} -- (lower bound, upper bound)

    Keyword Arguments:
        bins {int} -- The bins to compute the histogram (default: {100})

    Returns:
        {array} -- A [bins] dimension array representing the ratio of value in [bins]
    """
    # Compute histogram
    hist, bin_edges = np.histogram(d,
                                   bins=bins,
                                   range=bounds)
    # Make sure no zero value exists
    hist = hist.astype(float)
    hist[hist == 0] = 0.1
    # Make [h] into pdf
    hist = hist / len(d)
    return hist


class TimeGenEntropyComputer:
    def __init__(self, data=DATA, channels=CHANNELS, times=TIMES, folder='tmp_result'):
        """Init the computer by feeding the data

        Keyword Arguments:
            data {array} -- Data of stimuli 1 (default: {DATA})
            channels {int} -- Number of channels (default: {CHANNELS})
            times {int} -- Number of time points (default: {TIMES})
            folder {str} -- Name of the folder saving tmp result (default: {'tmp_result'})
        """
        self.data = data
        self.channels = channels
        self.times = times
        self.running = True

        self.folder = folder
        self.clear_folder()

    def clear_folder(self):
        """Clear self.folder
        """
        # self.folder should not exist,
        # if it does, empty and remove it
        if os.path.exists(self.folder):
            print(f'Warning {self.folder} exists, emptying it by default.')
            for e in os.listdir(self.folder):
                os.remove(os.path.join(self.folder, e))
            os.removedirs(self.folder)
        # Create self.folder
        os.mkdir(self.folder)

    def tmp_path(self, chan):
        """Get relative path of the tmp file

        Arguments:
            chan {int} -- The index of channel

        Returns:
            {str} -- The relative path of the tmp file
        """
        return os.path.join(self.folder, f'entropy_gen_map_{chan}.npy')

    def save_tmp_file(self, array, chan):
        """Save [array] into tmp file of channel [chan]

        Arguments:
            array {array} -- The array to be saved
            chan {int} -- The channel index
        """
        path = self.tmp_path(chan)
        np.save(path, array)
        print(f'Tmp file saved in {path}')

    def check_tmp_file(self, chan):
        """Check whether the tmp file exists

        Arguments:
            chan {int} -- The channel index

        Returns:
            {bool} -- True or False
        """
        path = self.tmp_path(chan)
        if os.path.exists(path):
            print(f'{path} already exists.')
            return True
        else:
            print(f'{path} not exists.')
            return False

    def compute_entropy(self, chan):
        """Compute entropy in time generation manner

        Arguments:
            chan {int} -- Which channel to compute

        Returns:
            entropy_gen_map {array} -- The matrix of time generation entropy
            Write the time generation entropy matrix into disk
        """
        entropy_gen_map = np.zeros((self.times, self.times))
        print(f'Start {chan}')
        begin_time = builtin_time.time()
        for time1 in range(self.times):
            if time1 % 10 == 0:
                print(f'-- Sub {chan} {time1} | {TIMES}')
            d1 = self.data[:, chan, time1]
            for time2 in range(self.times):
                d2 = self.data[:, chan, time2]
                bounds = (min(d1.min(), d2.min()),
                          max(d1.max(), d2.max()))

                h1 = histogram(d1, bounds=bounds)
                h2 = histogram(d2, bounds=bounds)

                entropy_gen_map[time1, time2] = ENTROPY(h1, h2)

                if not self.running:
                    print(f'Break out {chan}')
                    return 0

        passed_time = builtin_time.time() - begin_time
        print(f'Finish {chan}, {passed_time}')

        self.save_tmp_file(entropy_gen_map, chan)


# %%
TGEC = TimeGenEntropyComputer()

processes = []
for chan in range(CHANNELS):
    p = multiprocessing.Process(target=TGEC.compute_entropy, args=(chan,))
    processes.append(p)
    p.start()
    if chan % 30 == 29:
        p.join()
p.join()

while True:
    if all([TGEC.check_tmp_file(chan) for chan in range(CHANNELS)]):
        break
    builtin_time.sleep(1)

print('Multi-processing has been done, it is good to go.')

# %%
maps = [np.load(TGEC.tmp_path(chan)) for chan in range(CHANNELS)]
ENTROPY_GEN_MAP = np.concatenate([e[np.newaxis, :, :] for e in maps])
BOUNDS = (np.min(ENTROPY_GEN_MAP), np.max(ENTROPY_GEN_MAP))
print(ENTROPY_GEN_MAP.shape, BOUNDS)


TICKS = [e for e in range(0, TIMES, 10)]

fig, ax = plt.subplots()
im = ax.imshow(np.mean(ENTROPY_GEN_MAP, axis=0), origin='lower')

ax.set_xticks(TICKS)
ax.set_xticklabels(EPOCHS.times[TICKS])
ax.set_xlabel('Data 1')

ax.set_yticks(TICKS)
ax.set_yticklabels(EPOCHS.times[TICKS])
ax.set_ylabel('Data 1')
fig.colorbar(im)

FIGURES.append(fig)

# %%
X = np.concatenate([e.ravel()[np.newaxis, :] for e in ENTROPY_GEN_MAP])
X_embedded = TSNE(n_components=2).fit_transform(X)

# %%
spectral = CLUSTER.SpectralClustering(n_clusters=N_CLUSTERS, eigen_solver='arpack',
                                      affinity="nearest_neighbors")
LABELS = spectral.fit_predict(X)
UNIQUE_LABELS = np.unique(LABELS)

# %%
fig, ax = plt.subplots()
for p in UNIQUE_LABELS:
    ax.scatter(X_embedded[LABELS == p, 0], X_embedded[LABELS == p, 1])
FIGURES.append(fig)

# %%

DATA.shape
data = np.mean(DATA, axis=0)
data.shape

fig, axes = plt.subplots(len(UNIQUE_LABELS), 3, figsize=(12, 12))
for p in UNIQUE_LABELS:
    picks = LABELS == p
    axes[p][0].plot(data[picks, :].transpose())
    axes[p][1].imshow(np.mean(ENTROPY_GEN_MAP[picks], axis=0),
                      origin='lower', vmin=BOUNDS[0], vmax=BOUNDS[1])
    axes[p][2].imshow(np.std(ENTROPY_GEN_MAP[picks], axis=0),
                      origin='lower')
    for j in [0, 1, 2]:
        axes[p][j].set_xticks(TICKS)
        axes[p][j].set_xticklabels(EPOCHS.times[TICKS])
        if j == 0:
            continue
        axes[p][j].set_yticks(TICKS)
        axes[p][j].set_yticklabels(EPOCHS.times[TICKS])

FIGURES.append(fig)

# %%
for p in UNIQUE_LABELS:
    picks = np.where(LABELS == p)[0]
    e = EPOCHS['1'].copy()
    e.load_data()
    e.pick(picks)
    ev = e.average()
    fig = ev.plot_joint(times=np.linspace(0, 1.0, 8, endpoint=False),
                        title=f'{p}', show=SHOW)
    FIGURES.append(fig)

# %%

with PdfPages(f'{RUNNING_NAME}.pdf', 'w') as pp:
    for fig in FIGURES:
        pp.savefig(fig)


# %%
