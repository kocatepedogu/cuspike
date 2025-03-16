def get_elapsed_time(file_name):
    with open(file_name) as f:
        return float(f.readline().strip())

brian2cuda_elapsed_time = get_elapsed_time('cuba-brian2cuda-elapsedtime.txt')
nestgpu_elapsed_time = get_elapsed_time('cuba-nestgpu-elapsedtime.txt')
genn_elapsed_time = get_elapsed_time('cuba-genn-elapsedtime.txt')
cuspike_elapsed_time = get_elapsed_time('../../cuspike-elapsedtime.txt')

import matplotlib.pyplot as plt
import numpy as np

simulator = ['cuSpike', 'GeNN', 'NEST GPU', 'Brian2Cuda']
elapsed_time = [cuspike_elapsed_time, genn_elapsed_time, nestgpu_elapsed_time, brian2cuda_elapsed_time]

plt.figure(figsize=(10,8))
plt.bar(simulator, elapsed_time)
plt.title('Elapsed Time')
plt.xlabel("Simulator")
plt.ylabel("Elapsed Time (Seconds)")
plt.savefig("cuba-elapsedtime.png")
