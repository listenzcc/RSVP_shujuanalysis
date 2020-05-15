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
from sklearn import linear_model
from sklearn import preprocessing
from scipy.stats import entropy as ENTROPY

# Ploting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Local settings
sys.path.append('..')  # noqa
from local_toolbox import get_raw, get_epochs


# %%
# Custom RUNNING_NAME
def get_evoked(running_name):
    print('-' * 80)
    print(f'running_name: {running_name}')

    file_name = f'{running_name}-evo.fif'
    if os.path.exists(file_name):
        os.remove(file_name)

    # Get epochs
    epochs = get_epochs(running_name, band='U07',
                        memory_name=f'{running_name}-U07-epo.fif')

    # Get EPOCHS
    epochs.load_data()
    epochs.crop(tmin=None, tmax=1)

    # Get EVOKED
    evoked = epochs['1'].average()
    evoked.save(f'{running_name}-evo.fif')


NUM = 10

# Whether to load new evoked
RUNNING_NAMES = ['MEG_S%02d' % e for e in range(1, NUM+1)]
if False:
    for name in RUNNING_NAMES:
        t = multiprocessing.Process(target=get_evoked, args=(name,))
        t.start()


# %%
EVOKEDS = [mne.read_evokeds(f'{running_name}-evo.fif')[0]
           for running_name in RUNNING_NAMES]
DATA = np.concatenate([e.data[np.newaxis, :, :] for e in EVOKEDS])
DATA.shape

COLORS = np.load('colors_272.npy')

# %%
evoked = EVOKEDS[0].copy()
evoked.data = np.mean(DATA, axis=0)
fig = evoked.plot_joint()
TIMES = evoked.times

# %%
plt.style.use('ggplot')

x_mean = np.mean(DATA, axis=0).transpose()

SCALE = preprocessing.StandardScaler()
SCALE.fit(x_mean[TIMES < 0.1])

x_mean = SCALE.transform(x_mean)

fig, ax = plt.subplots(1, 1)
lines = ax.plot(x_mean)
for j, line in enumerate(lines):
    line.set_color(COLORS[j])

evoked = EVOKEDS[0].copy()
evoked.data = x_mean.transpose()
fig = evoked.plot_joint()

plt.style.use('default')

DATA_SET = [e for e in range(NUM)]

for j in range(NUM):
    print(j)
    d = SCALE.transform(DATA[j].transpose())

    reg = linear_model.ElasticNet(alpha=1.5, l1_ratio=0.5)
    reg.fit(d, d - x_mean)
    coef = np.eye(reg.coef_.shape[0]) - reg.coef_

    DATA_SET[j] = dict(
        data=d,
        coef=coef,
        inte=reg.intercept_
    )

    fig, axes = plt.subplots(figsize=(4, 4))
    ax = axes
    ax.imshow(coef)
    ax.set_title(j)
    # ax.colorbar()

# %%


def beauty(ax, title='[title]'):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    min_ = min(min(xlim), min(ylim))
    max_ = max(max(xlim), max(ylim))
    ax.set_xlim([min_, max_])
    ax.set_ylim([min_, max_])
    ax.set_aspect(1)
    ax.set_title(title)


def scatter(x_embedded, x_mean_embedded, n, colors=COLORS, style='ggplot'):
    plt.style.use(style)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    for j in range(NUM + 1):
        ax.scatter(x_embedded[n*j:n*(j+1), 0],
                   x_embedded[n*j:n*(j+1), 1], label=j)
    ax.legend(loc='best', bbox_to_anchor=(1, 1))
    beauty(ax, title='All mixed')

    ax = axes[1]
    d = x_embedded[:n]
    ax.scatter(d[:, 0], d[:, 1], c=colors)
    beauty(ax, title='Mean')

    ax = axes[2]
    d = x_mean_embedded
    ax.scatter(d[:, 0], d[:, 1], c=colors)
    beauty(ax, title='Only mean')

    evoked = EVOKEDS[0].copy()
    evoked.data = x_mean
    evoked.plot_joint()


# %%

n = DATA.shape[1]

x_mean = np.mean(DATA, axis=0)
x = np.concatenate([x_mean] + [e for e in DATA])

x_embedded = TSNE(n_components=2).fit_transform(x)
x_mean_embedded = TSNE(n_components=2).fit_transform(x_mean)

scatter(x_embedded, x_mean_embedded, n)

# %%

n = DATA.shape[1]


def compute(e):
    data = e['data']
    coef = e['coef']
    inte = e['inte']
    print(data.shape, coef.shape, inte.shape)
    return SCALE.inverse_transform(np.dot(data, coef) + inte).transpose()


data = np.concatenate([compute(e)[np.newaxis, :, :] for e in DATA_SET])
print(data.shape)

x_mean = np.mean(data, axis=0)
print(x_mean.shape)

x = np.concatenate([x_mean] + [e for e in data])

x_embedded = TSNE(n_components=2).fit_transform(x)
x_mean_embedded = TSNE(n_components=2).fit_transform(x_mean)

scatter(x_embedded, x_mean_embedded, n)

# %%
j = 3
evoked = EVOKEDS[j].copy()
evoked.plot_joint()

evoked.data = compute(DATA_SET[j])
evoked.plot_joint()

print()
# %%
