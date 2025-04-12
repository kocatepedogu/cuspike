import timeit
t_begin = timeit.default_timer()

import sys
import numpy as np
import matplotlib.pyplot as plt

from brian2 import *
import brian2cuda

set_device("cuda_standalone")

# Common Parameters

N = 20000

taum = 20*ms
taue = 5*ms
taui = 10*ms
Vt = -50*mV
Vr = -60*mV
El = -49*mV

eqs = '''
dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
dge/dt = -ge/taue : volt
dgi/dt = -gi/taui : volt
'''

P = NeuronGroup(N, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms,
                method='exact')
P.v = -60*mV
P.ge = 0*mV
P.gi = 0*mV

we = (60*0.27/10)*mV # excitatory synaptic weight (voltage)
wi = (-20*4.5/10)*mV # inhibitory synaptic weight
Ce = Synapses(P, P, on_pre='ge += we')
Ci = Synapses(P, P, on_pre='gi += wi')
Ce.connect(f'i<{N*4//5}', p=0.02)
Ci.connect(f'i>={N*4/5}', p=0.02)

print("we =", we / mV)
print("wi =", wi / mV)

s_mon = SpikeMonitor(P)

run(100 * second)

t_end = timeit.default_timer()
with open('brian2cuda-elapsedtime.txt', 'w') as f:
    f.write(str(t_end - t_begin))

if 'plot' in sys.argv:
    plt.figure(figsize=(75, 75))
    plt.plot(s_mon.t/ms, s_mon.i, ',k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.savefig('brian2cuda.png')

