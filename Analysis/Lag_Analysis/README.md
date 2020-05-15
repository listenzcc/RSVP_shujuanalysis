# Lag Analysis

!! It seems the idea is a fake one, so sad.
But it leads to [clustering analysis](../Clustering_Analysis).
Wish me luck.

## Aim

The lag analysis is taken to find out the time lag of **every time point**.

Specifically,

- Whether the **lag effect** exists?
- How long is it?
- Is it homogeneous for every time point?
- What is the difference between **MEG** and **EEG**?

## Methods

1. Load epochs data
2. Compute **KL divergency** between channels in time generation manner
3. Use **spectral clustering** method to identify different wave pattern

## Conclusion

There seems **several patterns** existing, however the methods I use can not discriminate them.

It leads to frequency analysis for each channels.
To find their common and individual characters in frequency domain.
