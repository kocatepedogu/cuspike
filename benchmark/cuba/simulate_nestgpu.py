import timeit
t_begin = timeit.default_timer()

import matplotlib.pyplot as plt
import nestgpu as ngpu
import numpy as np
import sys

# Units

uF = 1e-6
cm = 1e-2
siemens = 1
mV = 1e-3
ms = 1e-3

# Common Parameters

N = 20000
T = 100000*ms

Cm = 1*uF/(cm*cm)
gl = 5e-5*siemens/(cm*cm)

taum = Cm/gl
taue = 5*ms
taui = 10*ms

# NEST Model

ngpu.SetTimeResolution(0.1*ms)

# Neurons

E_L =  -49.0*mV

properties = {
    "V_m_rel": -60.0*mV - E_L,
    "E_L": E_L,
    "C_m": Cm,
    "tau_m": taum,
    "t_ref": 5*ms,
    "Theta_rel": -50.0*mV - E_L,
    "V_reset_rel": -60.0*mV - E_L,
    "tau_ex": taue,
    "tau_in": taui,
    "I_e": 0
}

exc_pop = ngpu.Create("iaf_psc_exp", n_node = int(N * 0.8))
inh_pop = ngpu.Create("iaf_psc_exp", n_node = int(N * 0.2))

ngpu.SetStatus(exc_pop, properties)
ngpu.SetStatus(inh_pop, properties)

ngpu.ActivateRecSpikeTimes(exc_pop, max_n_rec_spike_times=10000)
ngpu.ActivateRecSpikeTimes(inh_pop, max_n_rec_spike_times=10000)

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

exc_synapse_init = {"weight": we, "delay": 0.1*ms, "receptor":0}
inh_synapse_init = {"weight": wi, "delay": 0.1*ms, "receptor":1}

N_exc = int(N * 0.8)
N_inh = int(N * 0.2)

ngpu.Connect(exc_pop, exc_pop, conn_dict={'rule': 'fixed_total_number', 'total_num': int(N_exc*N_exc*0.02)}, syn_dict=exc_synapse_init)
ngpu.Connect(exc_pop, inh_pop, conn_dict={'rule': 'fixed_total_number', 'total_num': int(N_exc*N_inh*0.02)}, syn_dict=exc_synapse_init)
ngpu.Connect(inh_pop, exc_pop, conn_dict={'rule': 'fixed_total_number', 'total_num': int(N_inh*N_exc*0.02)}, syn_dict=inh_synapse_init)
ngpu.Connect(inh_pop, inh_pop, conn_dict={'rule': 'fixed_total_number', 'total_num': int(N_inh*N_inh*0.02)}, syn_dict=inh_synapse_init)

# Run and plot results

ngpu.Simulate(sim_time=T)

exc_spikes = ngpu.GetRecSpikeTimes(exc_pop)
inh_spikes = ngpu.GetRecSpikeTimes(inh_pop)

t_end = timeit.default_timer()
with open('nestgpu-elapsedtime.txt', 'w') as f:
    f.write(str(t_end - t_begin))

if 'plot' in sys.argv:
    spike_times = []
    spike_ids = []

    for n_id, n_spikes in enumerate(exc_spikes):
        for t in n_spikes:
            spike_times.append(t)
            spike_ids.append(n_id)

    for n_id, n_spikes in enumerate(inh_spikes):
        for t in n_spikes:
            spike_times.append(t)
            spike_ids.append(N_exc + n_id)

    print(f"Number of spikes: {len(spike_ids)}")
    print(f"Number of inhibitory synapses: {int(N_inh*N_exc*0.02) + int(N_inh*N_inh*0.02)}")
    print(f"Number of excitatory synapses: {int(N_exc*N_exc*0.02) + int(N_exc*N_inh*0.02)}")

    plt.figure(figsize=(75, 75))
    plt.plot(spike_times, spike_ids, ',k')
    plt.ylabel("Neuron ID")
    plt.savefig('nestgpu.png')

    from pycuspike import SpikeData
    SpikeData.save(np.array(spike_times), np.array(spike_ids), 't_array_nestgpu.dat', 's_array_nestgpu.dat')
