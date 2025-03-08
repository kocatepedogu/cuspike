#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cooperative_groups.h>

#include "config.hpp"

extern uint32_t *spike_times;
extern uint32_t *spike_counts;

extern uint32_t *matrix;

#include "./generated/kernel-register/device_functions.cu"

__device__ int spiked_neurons[N];
__device__ int spiked_neurons_cnt;

__global__
void simulate_register(uint32_t *__restrict__ const spike_times,
                       uint32_t *__restrict__ const spike_counts,
                       const uint32_t *__restrict__ const matrix)
{
    cooperative_groups::grid_group barrier = cooperative_groups::this_grid();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    #include "./generated/kernel-register/initialize.cu"

    barrier.sync();

    for (int timestep = 0; timestep < steps; ++timestep) {
        barrier.sync();

        // Reset temporary spike list before starting to record spikes of the new time step
        if (tid == 0) {
            spiked_neurons_cnt = 0;
        }

        barrier.sync();

        /* Parallelize over pre-synaptic neurons */
        if (tid < N) {
            if (r <= 0) {
                // Integrate membrane potential
                #include "./generated/kernel-register/always-at-active.cu"

                // If the membrane potential exceeds the threshold and the neuron is
                // not in refractory period, the neuron fires.
                if (v > Vt) {
                    #include "./generated/kernel-register/always-at-reset.cu"
                    r = refractoriness + dt;

                    // Record spike to global spike arrays for visualization
                    {
                        const int spike_index = spike_counts[tid];
                        spike_times[spike_index * N + tid] = timestep;
                        spike_counts[tid] = spike_index + 1;
                    }

                    // Record spike to temporary spike list for propagation
                    {
                        int cnt = atomicAdd(&spiked_neurons_cnt, 1);
                        spiked_neurons[cnt] = tid;
                    }
                }
            }

            // Integrate synaptic conductances
            #include "./generated/kernel-register/always.cu"
        }

        barrier.sync();

        /* Parallelize over post-synaptic neurons */
        if (tid < N) {
            // Sequentially iterate over incoming spikes
            for (int j = 0; j < spiked_neurons_cnt; ++j) {
                int pre = spiked_neurons[j];
                // Check if the spike is from an adjacent pre-synaptic neuron
                size_t rawIndex = pre * N + tid;
                size_t index = rawIndex / 32;
                size_t offset = rawIndex % 32;

                if (matrix[index] & (1 << offset)) {
                    // Apply synaptic changes
                    #include "./generated/kernel-register/always-at-spike.cu"
                }
            }
        }

        r -= dt;
    }
}
