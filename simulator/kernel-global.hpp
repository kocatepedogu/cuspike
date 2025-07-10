// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef KERNEL_GLOBAL_HPP
#define KERNEL_GLOBAL_HPP

#include <cuda.h>
#include <stdint.h>

__global__
void simulate_global(uint32_t *__restrict__ const spike_times,
                     uint32_t *__restrict__ const spike_counts,
                     const uint32_t *__restrict__ const synapses,
                     const uint32_t *__restrict__ const indices);

#endif
