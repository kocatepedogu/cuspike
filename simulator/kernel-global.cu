#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cooperative_groups.h>

#include "config.hpp"

extern uint32_t *spike_times;
extern uint32_t *spike_counts;

extern uint32_t *synapses;
extern uint32_t *indices;

#include "./generated/kernel-global/device-arrays.cu"

__device__ uint32_t spiked_neurons[N];
__device__ uint32_t spiked_neurons_cnt;

__device__ inline
void propagate(const uint32_t tid, const uint32_t stride,
               const uint32_t spiked_neuron_count,
               const uint32_t *spiked_neuron_list,
               const uint32_t *__restrict__ const synapses,
               const uint32_t *__restrict__ const indices) {
    for (int j = 0; j < spiked_neuron_count; ++j) {
        uint32_t pre = spiked_neuron_list[j];

        uint32_t start = indices[pre];
        uint32_t end = indices[pre + 1];

        /* Parallelize over post-synaptic neurons */
        for (uint32_t k = start + tid; k < end; k += stride) {
            int i = synapses[k];
            #include "./generated/kernel-global/always-at-spike.cu"
        }
    }
}

__global__
void simulate_global(uint32_t *__restrict__ const spike_times,
                     uint32_t *__restrict__ const spike_counts,
                     const uint32_t *__restrict__ const synapses,
                     const uint32_t *__restrict__ const indices)
{
    cooperative_groups::grid_group barrier = cooperative_groups::this_grid();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    __shared__ uint32_t sh_spiked_neurons[sharedSpikesPerBlock];
    __shared__ uint32_t sh_spiked_neurons_cnt;
    __shared__ uint32_t sh_spiked_neurons_cnt_extra;

    barrier.sync();

    for (int i = tid; i < N; i += stride) {
        #include "./generated/kernel-global/initialize.cu"
    }

    barrier.sync();

    for (int timestep = 0; timestep < steps; ++timestep) {
        barrier.sync();

        // Reset temporary spike list before starting to record spikes of the new time step
        if (threadIdx.x == 0) {
            sh_spiked_neurons_cnt = 0;
            sh_spiked_neurons_cnt_extra = 0;
        }
        if (tid == 0) {
            spiked_neurons_cnt = 0;
        }

        barrier.sync();

        /* Parallelize over pre-synaptic neurons */
        for (int i = tid; i < N; i += stride) {
            if (r[i] <= 0) {
                // Integrate membrane potential
                #include "./generated/kernel-global/always-at-active.cu"

                // If the membrane potential exceeds the threshold and the neuron is
                // not in refractory period, the neuron fires.
                if (v[i] > Vt) {
                    #include "./generated/kernel-global/always-at-reset.cu"
                    r[i] = refractoriness + dt;

                    // Record spike to global spike arrays for visualization
                    {
                        const int spike_index = spike_counts[i];
                        spike_times[spike_index * N + i] = timestep;
                        spike_counts[i] = spike_index + 1;
                    }

                    // Record spike to temporary spike list for propagation
                    {
                        // First attempt to store the spike in shared memory
                        uint32_t sh_cnt = atomicAdd(&sh_spiked_neurons_cnt, 1);
                        if (sh_cnt < sharedSpikesPerBlock) {
                            sh_spiked_neurons[sh_cnt] = i;
                        }

                        // Use global memory if the shared memory has become full
                        else {
                            uint32_t cnt = atomicAdd(&spiked_neurons_cnt, 1);
                            spiked_neurons[cnt] = i;
                            atomicAdd(&sh_spiked_neurons_cnt_extra, 1);
                        }
                    }
                }
            }

            // Integrate synaptic conductancess
            #include "./generated/kernel-global/always.cu"
        }

        barrier.sync();

        // Calculate the number of spikes stored in shared memory
        if (threadIdx.x == 0) {
            sh_spiked_neurons_cnt -= sh_spiked_neurons_cnt_extra;
        } __syncthreads();

        // Sequentially iterate over incoming spikes stored in the shared memory
        propagate(threadIdx.x, numThreads, sh_spiked_neurons_cnt, sh_spiked_neurons, synapses, indices);

        // Sequentially iterate over incoming spikes stored in the global memory
        propagate(tid, stride, spiked_neurons_cnt, spiked_neurons, synapses, indices);

        // Decrease refractory time
        for (int i = tid; i < N; i += stride) {
            r[i] -= dt;
        }
    }
}
