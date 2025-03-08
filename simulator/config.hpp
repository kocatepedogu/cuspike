#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <stdlib.h>

#include "./generated/config.hpp"

constexpr size_t steps = (size_t)(T / dt);

// Number of threads per block
constexpr int numThreads = 1024;

// Number of spikes stored in the shared memory of one block in global kernel
constexpr int sharedSpikesPerThread = 10;
constexpr int sharedSpikesPerBlock = numThreads * sharedSpikesPerThread;

#endif
