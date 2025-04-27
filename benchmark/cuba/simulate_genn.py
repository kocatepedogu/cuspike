import timeit
t_begin = timeit.default_timer()

import sys
import numpy as np
import matplotlib.pyplot as plt

from pygenn import GeNNModel, init_postsynaptic, init_sparse_connectivity, init_var, init_weight_update

# Units

uF = 1e-6
cm = 1e-2
siemens = 1
mV = 1e-3
ms = 1e-3

# Common Parameters

N = 20000
T = 100000*ms

Cm = 1.0*uF/(cm*cm)
gl = 5e-5*siemens/(cm*cm)

taum = Cm/gl
taue = 5*ms
taui = 10*ms

# GeNN Model

model = GeNNModel("float", "model")
model.dt = 0.1*ms

# Neurons

lif_params = {
    "C": Cm,
    "TauM": taum,
    "Vrest": -49.0*mV,
    "Vreset": -60.0*mV,
    "Vthresh": -50.0*mV,
    "Ioffset": 0.0,
    "TauRefrac": 5.0*ms
}

lif_init = {
    "V": init_var("Uniform", {"min": -60.0*mV, "max": -60.0*mV}),
    "RefracTime": 0.0*ms
}

exc_pop = model.add_neuron_population("E", N * 0.8, "LIF", lif_params, lif_init)
inh_pop = model.add_neuron_population("I", N * 0.2, "LIF", lif_params, lif_init)

exc_pop.spike_recording_enabled = True
inh_pop.spike_recording_enabled = True

# Synapses

Vmean = -60*mV

EE = 0*mV
EI = -80*mV
wE = 0.5 * 0.27e-5*siemens/(cm*cm)
wI = 0.5 * 4.5e-5*siemens/(cm*cm)

we = (EE - Vmean) * wE
wi = (EI - Vmean) * wI

print("we =", we / gl / mV)
print("wi =", wi / gl / mV)

exc_synapse_init = {"g": we}
inh_synapse_init = {"g": wi}

exc_post_syn_params = {"tau": taue}
inh_post_syn_params = {"tau": taui}

fixed_prob = {"prob": 0.02}

model.add_synapse_population("EE", "SPARSE",
    exc_pop, exc_pop,
    init_weight_update("StaticPulseConstantWeight", exc_synapse_init),
    init_postsynaptic("ExpCurr", exc_post_syn_params),
    init_sparse_connectivity("FixedProbabilityNoAutapse", fixed_prob))

model.add_synapse_population("EI", "SPARSE",
    exc_pop, inh_pop,
    init_weight_update("StaticPulseConstantWeight", exc_synapse_init),
    init_postsynaptic("ExpCurr", exc_post_syn_params),
    init_sparse_connectivity("FixedProbability", fixed_prob))

model.add_synapse_population("II", "SPARSE",
    inh_pop, inh_pop,
    init_weight_update("StaticPulseConstantWeight", inh_synapse_init),
    init_postsynaptic("ExpCurr", inh_post_syn_params),
    init_sparse_connectivity("FixedProbabilityNoAutapse", fixed_prob))

model.add_synapse_population("IE", "SPARSE",
    inh_pop, exc_pop,
    init_weight_update("StaticPulseConstantWeight", inh_synapse_init),
    init_postsynaptic("ExpCurr", inh_post_syn_params),
    init_sparse_connectivity("FixedProbability", fixed_prob));

model.build()
model.load(num_recording_timesteps=int(T/model.dt))

while model.timestep < int(T/model.dt):
    model.step_time()

model.pull_recording_buffers_from_device()

exc_spike_times, exc_spike_ids = exc_pop.spike_recording_data[0]
inh_spike_times, inh_spike_ids = inh_pop.spike_recording_data[0]

print(f"Number of spikes: {len(exc_spike_ids) + len(inh_spike_ids)}")

t_end = timeit.default_timer()
with open('genn-elapsedtime.txt', 'w') as f:
    f.write(str(t_end - t_begin))

if 'plot' in sys.argv:
    plt.figure(figsize=(75, 75))
    plt.plot(exc_spike_times, exc_spike_ids, ',k')
    plt.plot(inh_spike_times, inh_spike_ids + int(N*0.8), ',k')
    plt.ylabel("Neuron ID")
    plt.savefig('genn.png')

    from pycuspike import SpikeData
    t_array = np.concatenate([exc_spike_times, inh_spike_times])
    s_array = np.concatenate([exc_spike_ids, inh_spike_ids])
    SpikeData.save(t_array, s_array, 't_array_genn.dat', 's_array_genn.dat')
