# Good luck

## Quick start

It is a pipeline for MEG and EEG analysis.

- It requires user put right files into right paths
- It reads data from the [rawdata](./rawdata) and recognize them as folder name
- All follow-up processes are based on correctly recognition of the folders

## Organize Data

The format is quite simple:

*20200101_RSVP_MEG_S00* means *MEG* data of *RSVP* experiment on *20200101* with subject *00*.

Inside the folder is data of sessions:

- We will ignore the files start with '_'
- Files start with 'r' refers *Resting* state
- Others should be correct experiment data

The [parse function](./autobuild_path_table.ipynb) will parse the names and store them in a [table](./auto_path_table.json).
The data should be organized into correct subjects.

Then use [build folder function](./autobuild_folders.ipynb) to build folder for each subject.

The process can be easy and pleasure.

## De-noise

First step is artificial removal using ICA, an automatic [script](./denoise_ica.ipynb) will do that.

It will automatically do following steps:

- Select dataset and fit ICA model for each session
- Automatic detect and mark artificial components
- Remove marked artificial components
- ICA logs will be stored too if there will be in need
- De-noised data will be stored in new [folder](./processed_data)

## MVPA

The labeled sample in MVPA is epochs, ranged from -0.2 to 1.2 seconds respect to the onset of stimuli.
Several bands will be used to filter the epochs, they are:
| Name | Band (Hz) |
| ---- | ---- |
| Delta| 1 - 4 |
| Theta | 4 - 7 |
| Alpha | 8 - 12 |
| Cb_U07 | 0.1 - 7 |
| Cb_U30 | 0.1 - 30 |

MVPA is performed for each subject individually in two folders:

- MVPA with [alltime](./perform_MVPA.ipynb)  
Use all time points in epochs as raw features.
- MVPA with [timeresolution](./perform_MVPA_times.ipynb)  
Perform MVPA in time resolution, use each time points in epochs as raw features.

### MVPA with alltime

The [script](./perform_MVPA.ipynb) will perform MVPA with the epochs.

The predicted labels of each subject will be stored in the current folder.
**User must move them into proper [folder](./MVPA_results_alltime) after the script completes.**

The [script](./MVPA_results_alltime/report_MVPA.ipynb) will automatically generate the report and write it as HTML file.

### MVPA with resulationtime

The [script](./perform_MVPA_times.ipynb) will perform MVPA with the epochs on time resolution.

The predicted labels of each subject will be stored in the current foldr.
**User must move them into proper [folder](./MVPA_results_resolutiontime) after the script completes.**

The scripts will automatically generate reports and write into HTML files, they are [visual mean](./MVPA_results_resolutiontime/report_MVPA-mean.ipynb), [visual separate](./MVPA_results_resolutiontime/report_MVPA-all.ipynb) and [visual together](./MVPA_results_resolutiontime/report_MVPA-together.ipynb).