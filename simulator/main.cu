// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <stdint.h>

#include "config.hpp"
#include "util.hpp"
#include "initialize.hpp"
#include "kernel-register.hpp"
#include "kernel-global.hpp"
#include "save.hpp"

#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cooperative_groups.h>

int numBlocksPerSmForRegisterKernel;
int numBlocksPerSmForGlobalKernel;
int numBlocksForRegisterKernel;
int numBlocksForGlobalKernel;

uint32_t *spike_times = nullptr;
uint32_t *spike_counts = nullptr;

uint32_t *matrix = nullptr;
uint32_t *synapses = nullptr;
uint32_t *indices = nullptr;

cudaStream_t createStreamWithResidency(int dev, void *ptr) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    CUDA_CALL(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize));

    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));

    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr);
    stream_attribute.accessPolicyWindow.num_bytes = prop.persistingL2CacheMaxSize;
    stream_attribute.accessPolicyWindow.hitRatio  = 1.0;
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    CUDA_CALL(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));

    return stream;
}

void launchRegisterKernel(int dev) {
    dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid(numBlocksForRegisterKernel, 1, 1);

    CUDA_CALL(cudaMallocManaged(&matrix, 1 + N * N * sizeof *matrix / 32));
    initialize_synapses_bitmap(dev);

    void *kernelArgs[] = {&spike_times, &spike_counts, &matrix};

    if constexpr (l2residency) {
        cudaStream_t stream = createStreamWithResidency(dev, matrix);
        CUDA_CALL(cudaLaunchCooperativeKernel((void*)simulate_register, dimGrid, dimBlock, kernelArgs, 0, stream));
    }
    else {
        CUDA_CALL(cudaLaunchCooperativeKernel((void*)simulate_register, dimGrid, dimBlock, kernelArgs));
    }
}

void launchGlobalKernel(int dev) {
    dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid(numBlocksForGlobalKernel, 1, 1);

    CUDA_CALL(cudaMallocManaged(&synapses, N * N * sizeof *synapses));
    CUDA_CALL(cudaMallocManaged(&indices, (N + 1) * sizeof *indices));
    initialize_synapses_csr(dev);

    void *kernelArgs[] = {&spike_times, &spike_counts, &synapses, &indices};

    if constexpr(l2residency) {
        cudaStream_t stream = createStreamWithResidency(dev, synapses);
        CUDA_CALL(cudaLaunchCooperativeKernel((void*)simulate_global, dimGrid, dimBlock, kernelArgs, 0, stream));
    }
    else {
        CUDA_CALL(cudaLaunchCooperativeKernel((void*)simulate_global, dimGrid, dimBlock, kernelArgs));
    }
}

int main(int argc, char **argv) {
    int dev = 0;
    int supportsCoopLaunch = 0;

    CUDA_CALL(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev));
    if( supportsCoopLaunch != 1 ) {
        printf("Cooperative Launch is not supported on this machine.\n");
        abort();
    }

    cudaDeviceProp deviceProp;
    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, dev));

    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSmForRegisterKernel, simulate_register, numThreads, 0));
    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSmForGlobalKernel, simulate_global, numThreads, 0));

    numBlocksForRegisterKernel = numBlocksPerSmForRegisterKernel * deviceProp.multiProcessorCount;
    numBlocksForGlobalKernel = numBlocksPerSmForGlobalKernel * deviceProp.multiProcessorCount;

    CUDA_CALL(cudaMallocManaged(&spike_times, N * steps * sizeof *spike_times));
    CUDA_CALL(cudaMallocManaged(&spike_counts, N * sizeof *spike_counts));

    if (N < numBlocksForRegisterKernel * numThreads) {
        printf("Using registers to store state variables.\n");
        launchRegisterKernel(dev);
    } else {
        printf("Using global memory to store state variables.\n");
        launchGlobalKernel(dev);
    }

    CUDA_CALL(cudaDeviceSynchronize());

    if (argc == 2 && !strcmp(argv[1], "plot")) {
        print_results();
    }

    CUDA_CALL(cudaFree(spike_times));
    CUDA_CALL(cudaFree(spike_counts));

    return 0;
}
