import subprocess
import sys

# Power measurement period in milliseconds
interval = 10

nvidia_smi_process = subprocess.Popen([
    f'nvidia-smi',
    f'--query-gpu=power.draw',
    f'--format=csv',
    f'--loop-ms={interval}'], stdout=subprocess.PIPE, text=True)

nvidia_smi = nvidia_smi_process.pid

# Launch given process until wait until completion
measured_process = subprocess.Popen(sys.argv[1:])
measured_process.wait()

# Terminate power measurement immediately after the given process finishes.
nvidia_smi_process.terminate()
output = nvidia_smi_process.stdout.read()

# Compute total energy consumption by integrating watts over 10 millisecond intervals
total_energy = 0
for line in output.split('\n'):
    if line.strip() == '':
        break
    if 'power.draw' in line:
        continue
    value, watt = line.split(' ')
    total_energy += float(value) * (interval * 1e-3)

# Print the total energy consumption
print(f"Total energy consumption: {total_energy}")
