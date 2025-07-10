// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef KERNEL_REGISTER_HPP
#define KERNEL_REGISTER_HPP

#include <cuda.h>
#include <stdint.h>

__global__
void simulate_register(uint32_t *__restrict__ const spike_times,
                       uint32_t *__restrict__ const spike_counts,
                       const uint32_t *__restrict__ const matrix);

#endif
