# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from sklearn.neighbors import KernelDensity
from scipy.special import rel_entr

class Stats:
    @staticmethod
    def compute_stats(t_array, s_array):
        N = max(s_array) + 1
        T = max(t_array)

        spikes_per_neuron = [list() for i in range(N)]
        for t, n in zip(t_array, s_array):
            spikes_per_neuron[n].append(t)

        interspike_intervals = [list() for i in range(N)]
        for n in range(N):
            for i in range(len(spikes_per_neuron[n]) - 1):
                interspike_intervals[n].append(spikes_per_neuron[n][i + 1] - spikes_per_neuron[n][i])

        mean_of_interspike_intervals = np.array([np.mean(interspike_intervals[n]) for n in range(N)])
        std_of_interspike_intervals = np.array([np.std(interspike_intervals[n]) for n in range(N)])

        mean_of_interspike_intervals = mean_of_interspike_intervals[~np.isnan(mean_of_interspike_intervals)]
        std_of_interspike_intervals = std_of_interspike_intervals[~np.isnan(std_of_interspike_intervals)]

        average_firing_rate = 1 / mean_of_interspike_intervals
        cvisi = std_of_interspike_intervals / mean_of_interspike_intervals

        return average_firing_rate, cvisi


    @staticmethod
    def kernel_density(data, lower_bound = None, upper_bound = None, resolution = 1000, m = 1.5):
        stdev = np.std(data)
        mean = np.mean(data)
        maskMin = mean - stdev * m
        maskMax = mean + stdev * m
        data = np.ma.masked_outside(data, maskMin, maskMax)

        if lower_bound is None:
            lower_bound = data.min() - 1

        if upper_bound is None:
            upper_bound = data.max() + 1

        x_grid = np.linspace(lower_bound, upper_bound, resolution)

        # Create KernelDensity object (note: sklearn expects shape (n_samples, n_features))
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(data.reshape(-1, 1))

        # Evaluate on the grid (log density)
        log_dens = kde.score_samples(x_grid[:, np.newaxis])
        kde_pdf = np.exp(log_dens)

        return x_grid, kde_pdf


    @staticmethod
    def kl_divergence(pdf1, pdf2):
        # Prevent division by zero by adding a very low probability throughout the range
        pdf1_shifted = pdf1 + 1e-15
        pdf2_shifted = pdf2 + 1e-15
        pdf1_shifted /= sum(pdf1_shifted)
        pdf2_shifted /= sum(pdf2_shifted)

        # Compute KL-divergence using relative entropy function of scipy
        return sum(rel_entr(pdf1_shifted, pdf2_shifted))


    @staticmethod
    def histogram(data, bins=100):
        counts, bin_edges = np.histogram(a, density=True)
        return counts, bin_edges

