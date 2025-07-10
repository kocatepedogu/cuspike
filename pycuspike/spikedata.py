# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

class SpikeData:
    @staticmethod
    def load(t_array_filename, s_array_filename):
        t_array = np.fromfile(t_array_filename, dtype=np.float32)
        s_array = np.fromfile(s_array_filename, dtype=np.int32)

        return t_array, s_array

    @staticmethod
    def save(t_array, s_array, t_array_filename, s_array_filename):
        t_array.astype(dtype=np.float32).tofile(t_array_filename)
        s_array.astype(dtype=np.int32).tofile(s_array_filename)

