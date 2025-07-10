# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

simulators = {
    'cuSpike': 'cuspike-elapsedtime.txt',
    'GeNN': 'genn-elapsedtime.txt',
    'NEST GPU': 'nestgpu-elapsedtime.txt',
    'Brian2Cuda': 'brian2cuda-elapsedtime.txt'
}

simulator_name_list = []
elapsed_time_list = []
for model_name, file_name in simulators.items():
    path = Path(file_name)
    if path.exists() and path.is_file():
        with open(file_name) as f:
            elapsed_time = float(f.readline().strip())
            simulator_name_list.append(model_name)
            elapsed_time_list.append(elapsed_time)

plt.figure(figsize=(10,8))
plt.bar(simulator_name_list, elapsed_time_list)
plt.title('Elapsed Time')
plt.xlabel("Simulator")
plt.ylabel("Elapsed Time (Seconds)")
plt.savefig("elapsedtime.png")
