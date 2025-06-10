import numpy as np
import os

# Path to the original EDA file
eda_path = r"C:\Users\lenin\Documents\GitHub\pipeline-eeg-eda_060524\data\raw\eda\opensignals_lsl_500hz_gain1_0007808C0708_16-32-15_converted.txt"
output_dir = os.path.dirname(eda_path)

# Number of samples to match EEG duration at 500 Hz
n_samples = 105325
n_files = 10

# Read the first 105,325 rows after header (assuming header is 2 lines)
with open(eda_path, 'r') as f:
    header = [next(f) for _ in range(2)]
    data = []
    for _, line in zip(range(n_samples), f):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        try:
            # If three columns, take the last as EDA value
            value = float(parts[-1])
            data.append(value)
        except (ValueError, IndexError):
            continue
base_eda = np.array(data)
actual_samples = len(base_eda)
if actual_samples != n_samples:
    print(f"Warning: Expected {n_samples} samples, but got {actual_samples}. Using {actual_samples} for simulation.")
    n_samples = actual_samples

for i in range(1, n_files + 1):
    # Simulate: random permutation + small jitter
    permuted = np.random.permutation(base_eda)
    jitter = np.random.normal(0, 0.01, n_samples)  # small noise
    simulated = permuted + jitter
    # Save to new file
    out_path = os.path.join(output_dir, f"simulated_eda_{i:02d}_500hz.txt")
    with open(out_path, 'w') as out_f:
        out_f.writelines(header)
        for value in simulated:
            out_f.write(f"{value:.6f}\n")
print(f"Simulated {n_files} EDA files saved to {output_dir}")
