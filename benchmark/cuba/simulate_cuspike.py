import timeit
t_begin = timeit.default_timer()

from pycuspike import Model

import matplotlib.pyplot as plt
import numpy as np

model = Model('cuba.csm')
model.build()
model.simulate()

t_end = timeit.default_timer()
with open('cuspike-elapsedtime.txt', 'w') as f:
    f.write(str(t_end - t_begin))

t_array, s_array = model.spikes()
plt.figure(figsize=(75, 75))
plt.plot(t_array, s_array, ',k')
plt.savefig('cuspike.png')

from pycuspike import SpikeData
SpikeData.save(t_array, s_array, 't_array_cuspike.dat', 's_array_cuspike.dat')
