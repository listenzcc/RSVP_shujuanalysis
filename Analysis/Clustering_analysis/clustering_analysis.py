# %%
# System
import os
import sys
import multiprocessing
import time as builtin_time

# Computing
import numpy as np
import mne
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
    N_CLUSTERS = 5
if RUNNING_NAME.startswith('MEG'):
    N_CLUSTERS = 7

# Safety assert,
# RUNNING_NAME is legal,
# N_CLUSTERS is legal.
assert(N_CLUSTERS > 0)

print(f'RUNNING_NAME: {RUNNING_NAME}')
FIGURES = []
SHOW = False

# Get epochs
epochs = get_epochs(RUNNING_NAME, band='U07',
                    memory_name=f'{RUNNING_NAME}-U07-epo.fif')

# Plot evoked
for idx in ['1', '2', '3']:
    # Only plot '1'
    if not idx == '1':
        continue

    title = f'{RUNNING_NAME}-{idx}'
    print(title)
    evoked = epochs[idx].average()
    fig = evoked.plot_joint(times=np.linspace(0, 1.0, 8, endpoint=False),
                            title=f'Evoked of {title}', show=SHOW)
    FIGURES.append(fig)

# Get EPOCHS
epochs.load_data()
epochs.crop(tmin=0, tmax=1)
EPOCHS = epochs['1']

# Get EVOKED
evoked = epochs['1'].average()
EVOKED = evoked

EPOCHS, EVOKED

# %%
CHANNELS = EPOCHS.info['chs']
TIMES = EPOCHS.times


# %%
x = EVOKED.data
x_embedded = TSNE(n_components=2).fit_transform(x)

n_clusters = N_CLUSTERS
spectral = CLUSTER.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack',
                                      affinity='nearest_neighbors')
labels = spectral.fit_predict(x)

fig, ax = plt.subplots()
for label in np.unique(labels):
    ax.scatter(x_embedded[labels == label, 0], x_embedded[labels == label, 1])
FIGURES.append(fig)

times = np.array([0.2, 0.3, 0.4, 0.5, 0.6])

evoked_labels = EVOKED.copy()
evoked_labels.data = evoked_labels.data * 0

for label in np.unique(labels):
    print(label)
    picks = np.where(labels == label)[0]

    evoked_labels.data[picks, label] = 1

    evoked = EVOKED.copy()
    fig = mne.viz.plot_evoked(
        evoked, picks=picks, spatial_colors=True, titles=label, show=SHOW)
    FIGURES.append(fig)

    evoked.data[labels != label, :] = 0
    fig = evoked.plot_topomap(times=times, title=label, show=SHOW)
    FIGURES.append(fig)
    # epochs = EPOCHS.copy()
    # epochs.pick(picks)
    # epochs.average().plot_topomap(ch_type='mag')

fig = evoked_labels.plot_topomap(times=TIMES[:label+1], vmin=0, show=SHOW)
FIGURES.append(fig)


# %%
with PdfPages(f'{RUNNING_NAME}.pdf', 'w') as pp:
    for fig in FIGURES:
        pp.savefig(fig)


# %%
