import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

def read_file(file_name, data_type):
        return np.fromfile(file_name, dtype=data_type)

with mp.Pool(2) as p:
    t_array, s_array = p.starmap(read_file, [
        ('output/t_array.dat', np.float32),
        ('output/s_array.dat', np.int32)
    ])

    plt.figure(figsize=(75, 75))
    plt.plot(t_array, s_array, ',k')
    plt.savefig('output/cuba.png')

