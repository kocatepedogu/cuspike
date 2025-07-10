# SPDX-FileCopyrightText: 2025 Doğu Kocatepe
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import subprocess
import shutil
import hashlib
import numpy as np

from pathlib import Path
from .spikedata import SpikeData

class Model:
    def __init__(self, file_name):
        self.file_name = Path(file_name).absolute()

        self.build_directory = Path('cuspike-build')
        self.checksum_file_path = self.build_directory / 'checksum.txt'

        self.module_home = os.path.dirname(__file__)
        self.cuspike_home = os.path.dirname(self.module_home)


    def _run_command(self, command):
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            raise Exception(f'Command {command} failed.')


    def _load_model_and_compare_checksum(self, model_file):
        self.model_contents = model_file.read()
        self.model_checksum = hashlib.md5().hexdigest()

        # If a build directory does not exist, return false
        if not self.build_directory.exists() or not self.build_directory.is_dir():
            return False

        # If a build directory exists, but it does not contain a checksum file, return false
        if not self.checksum_file_path.exists() or not self.checksum_file_path.is_file():
            return False

        # If a build directory and a checksum file exists, compare whether it is the same
        with open(self.checksum_file_path, 'r') as checksum_file:
            return self.model_checksum == checksum_file.read()


    def build(self):
        with open(self.file_name, 'rb') as model_file:
            # If the same built has been done before, return immediately.
            if self._load_model_and_compare_checksum(model_file):
                return

            # Delete files from previous build
            if self.build_directory.exists() and self.build_directory.is_dir():
                shutil.rmtree(self.build_directory)

            # Create new build directory
            self.build_directory.mkdir()

            # Create directories for generated code files
            (self.build_directory / 'generated' / 'kernel-global').mkdir(parents=True)
            (self.build_directory / 'generated' / 'kernel-register').mkdir(parents=True)

            # Compile the model into CUDA files
            self._run_command(f"cd cuspike-build && cpp -E {self.file_name} | grep -wv '#' | {self.cuspike_home}/compiler/compiler")

            # Copy static simulator files into the build directory
            self._run_command(f"cp -r {self.cuspike_home}/simulator/* ./cuspike-build/")

            # Compile the simulator
            self._run_command("cd cuspike-build && make -j$(nproc)")

            # Create output directory
            (self.build_directory / 'output').mkdir(parents=True)

            # Save the new checksum
            with open(self.checksum_file_path, 'w') as f:
                f.write(self.model_checksum)


    def simulate(self):
        self._run_command("cd cuspike-build && ./simulator.exe plot")


    def spikes(self):
        return SpikeData.load('cuspike-build/output/t_array.dat', 'cuspike-build/output/s_array.dat')
