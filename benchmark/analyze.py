import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from pycuspike import Stats, SpikeData

simulators = {
    'cuSpike': ('t_array_cuspike.dat', 's_array_cuspike.dat'),
    'GeNN': ('t_array_genn.dat', 's_array_genn.dat'),
    'NEST GPU': ('t_array_nestgpu.dat', 's_array_nestgpu.dat'),
    'Brian2Cuda': ('t_array_brian2cuda.dat', 's_array_brian2cuda.dat')
}

# Load saved spikes from all simulators
spike_data = dict()
for model_name, files in simulators.items():
    t_array_filename, s_array_filename = files
    try:
        t_array, s_array = SpikeData.load(t_array_filename, s_array_filename)
        spike_data[model_name] = t_array, s_array
    except:
        continue

# Compute per neuron average firing rate and CV ISI for all simulators
stats_data = dict()
min_firing_rate = 1e30
max_firing_rate = -1e30
min_cv_isi = 1e30
max_cv_isi = -1e30
for model_name, data in spike_data.items():
    t_array, s_array = data
    firing_rate, cv_isi = Stats.compute_stats(t_array, s_array)
    min_firing_rate = min(min_firing_rate, min(firing_rate))
    max_firing_rate = max(max_firing_rate, max(firing_rate))
    min_cv_isi = min(min_cv_isi, min(cv_isi))
    max_cv_isi = max(max_cv_isi, max(cv_isi))
    stats_data[model_name] = firing_rate, cv_isi

colors = {
    'cuSpike': 'red',
    'GeNN': 'green',
    'NEST GPU': 'orange',
    'Brian2Cuda': 'blue'
}

firing_rate_pdfs = dict()
cv_isi_pdfs = dict()

# Compute PDF of per neuron average firing rate
plt.figure()
plt.grid(True, linestyle='--', alpha=0.6)
for model_name, data in stats_data.items():
    firing_rate, cv_isi = data
    firing_rate_grid, firing_rate_pdf = Stats.kernel_density(firing_rate, min_firing_rate, 20)
    firing_rate_pdfs[model_name] = firing_rate_pdf
    plt.plot(firing_rate_grid, firing_rate_pdf, label=model_name, color=colors[model_name])
plt.title('Average Firing Rate Distribution')
plt.legend()
plt.savefig('average-firing-rate.png')

# Compute PDF of per neuron CV ISI
cv_isi_pdfs = dict()
plt.figure()
plt.grid(True, linestyle='--', alpha=0.6)
for model_name, data in stats_data.items():
    firing_rate, cv_isi = data
    cv_isi_grid, cv_isi_pdf = Stats.kernel_density(cv_isi, min_cv_isi, 3)
    cv_isi_pdfs[model_name] = cv_isi_pdf
    plt.plot(cv_isi_grid, cv_isi_pdf, label=model_name, color=colors[model_name])
plt.title('CV ISI Distribution')
plt.legend()
plt.savefig('cv-isi.png')

reference_model_name = 'Brian2Cuda'

# Compute KL-divergences of average firing rate PDFs relative to Brian2Cuda
simulator_name_list = []
kl_divergence_list = []
color_list = []
reference_firing_rate_pdf = firing_rate_pdfs[reference_model_name]
for model_name, firing_rate_pdf in firing_rate_pdfs.items():
    if model_name != reference_model_name:
        kl = Stats.kl_divergence(firing_rate_pdf, reference_firing_rate_pdf)
        simulator_name_list.append(model_name)
        kl_divergence_list.append(kl)
        color_list.append(colors[model_name])
plt.figure(figsize=(10,8))
plt.bar(simulator_name_list, kl_divergence_list, color=color_list)
plt.title(f'Average Firing Rate KL-Divergence Relative to {reference_model_name}')
plt.xlabel("Simulator")
plt.ylabel("KL-Divergence")
plt.savefig("average-firing-rate-kl-divergence.png")

# Compute KL-divergences of CV ISI PDFs relative to Brian2Cuda
simulator_name_list = []
kl_divergence_list = []
color_list = []
reference_cv_isi_pdf = cv_isi_pdfs[reference_model_name]
for model_name, cv_isi_pdf in cv_isi_pdfs.items():
    if model_name != reference_model_name:
        kl = Stats.kl_divergence(cv_isi_pdf, reference_cv_isi_pdf)
        simulator_name_list.append(model_name)
        kl_divergence_list.append(kl)
        color_list.append(colors[model_name])
plt.figure(figsize=(10,8))
plt.bar(simulator_name_list, kl_divergence_list, color=color_list)
plt.title(f'CV ISI KL-Divergence Relative to {reference_model_name}')
plt.xlabel("Simulator")
plt.ylabel("KL-Divergence")
plt.savefig("cv-isi-kl-divergence.png")
