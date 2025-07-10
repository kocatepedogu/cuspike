// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <random>
#include <omp.h>
#include <vector>

#include "config.hpp"

extern uint32_t *matrix;

extern uint32_t *synapses;
extern uint32_t *indices;

void initialize_synapses_bitmap(int dev) {
    #pragma omp parallel
    {
        std::mt19937 mt(time(NULL) * omp_get_thread_num());

        #pragma omp for
        for (size_t i = 0; i < N * N / 32; ++i) {
            matrix[i] = 0;
        }

        #pragma omp for
        for (int pre = 0; pre < N; ++pre) {
            for (int post = 0; post < N; ++post) {
                size_t rawIndex = pre * N + post;
                size_t index = rawIndex / 32;
                size_t offset = rawIndex % 32;

                if ((float)mt()/(float)mt.max() < 0.02) {
                    #pragma omp atomic
                    matrix[index] |= (1 << offset);
                }
            }
        }
    }
}

void initialize_synapses_csr(int dev) {
    std::vector<std::vector<uint32_t>> synapses_local;
    for (int i = 0; i < N; ++i) {
        synapses_local.push_back(std::vector<uint32_t>{});
    }

    int index = 0;

    #pragma omp parallel
    {
        std::mt19937 mt(time(nullptr) << omp_get_thread_num());

        #pragma omp for
        for (int pre = 0; pre < N; ++pre) {
            for (int post = 0; post < N; ++post) {
                if ((float)mt()/(float)mt.max() < 0.02) {
                    synapses_local[pre].push_back(post);
                }
            }
        }

        #pragma omp for reduction(inscan,+:index)
        for (int i = 0; i < N; ++i) {
            indices[i] = index;
            #pragma omp scan exclusive(index)
            index += synapses_local[i].size();
        }
        indices[N] = index;

        #pragma omp for
        for (int i = 0; i < N; ++i) {
            const uint32_t start_index = indices[i];
            const uint32_t length = synapses_local[i].size();
            const uint32_t *synapses_ptr = &synapses_local[i][0];

            #pragma omp simd
            for (int j = 0; j < length; ++j) {
                synapses[start_index + j] = synapses_ptr[j];
            }
        }
    }
}
