#include <iostream>
#include <fstream>
#include <stdint.h>

#include "config.hpp"

extern uint32_t *spike_times;
extern uint32_t *spike_counts;

void print_results() {
    std::ofstream tfile;
    std::ofstream sfile;

    tfile.open("output/t_array.dat", std::ios::out | std::ios::binary);
    sfile.open("output/s_array.dat", std::ios::out | std::ios::binary);

    int number_of_spikes = 0;

    for (int neuron_index = 0; neuron_index < N; ++neuron_index) {
        for (int spike_index = 0; spike_index < spike_counts[neuron_index]; ++spike_index) {
            float spike_time = spike_times[spike_index * N + neuron_index] * dt;
            tfile.write(reinterpret_cast<const char*>(&spike_time), sizeof(float));

            ++number_of_spikes;
        }
    }

    for (int neuron_index = 0; neuron_index < N; ++neuron_index) {
        for (int spike_index = 0; spike_index < spike_counts[neuron_index]; ++spike_index) {
            sfile.write(reinterpret_cast<const char*>(&neuron_index), sizeof(int));
        }
    }

    std::cout << "Number of spikes: " << number_of_spikes << std::endl;

    tfile.close();
    sfile.close();
}
