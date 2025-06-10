"""
EEG Dynamic Functional Connectivity Topography Visualization

This script computes dynamic functional connectivity (DFC) matrices (using Pearson correlation) for each 2-second window of EEG data, then averages these matrices for each emotion class (Excited, Angry, Sad, Calm) based on window-wise emotion labels. It visualizes the average DFC matrix for each emotion as a heatmap, allowing comparison of network patterns across emotions.

Usage:
- Place this script in the same directory as your main pipeline.
- Import or adapt the data loading and window-wise emotion label extraction as needed.
- Run the script to generate and save the comparative DFC topography figure.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
CHANNEL_LABELS = [
    'Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'
]
NUM_CHANNELS = 8
WINDOW_SIZE = 2  # seconds
SAMPLING_RATE = 250  # Hz (adjust if needed)

# --- LOAD EEG DATA AND EMOTION LABELS ---
def load_eeg_and_labels(eeg_file, labels):
    """
    eeg_file: path to EEG CSV (channels x samples, channels 0-7 are EEG)
    labels: list/array of emotion labels per window (len = n_windows)
    Returns: eeg_data (n_channels x n_samples), labels (n_windows)
    """
    df = pd.read_csv(eeg_file, header=None)
    eeg_data = df.iloc[:NUM_CHANNELS, :].values
    return eeg_data, labels

# --- DFC COMPUTATION ---
def compute_dfc_matrices(eeg_data, window_size, fs):
    """
    eeg_data: n_channels x n_samples
    Returns: list of correlation matrices, one per window
    """
    n_samples = eeg_data.shape[1]
    win_len = int(window_size * fs)
    step = win_len  # non-overlapping
    n_windows = (n_samples - win_len) // step + 1
    dfc_matrices = []
    for w in range(n_windows):
        start = w * step
        end = start + win_len
        window = eeg_data[:, start:end]
        corr = np.corrcoef(window)
        dfc_matrices.append(corr)
    return np.array(dfc_matrices)  # shape: (n_windows, n_channels, n_channels)

# --- AVERAGE DFC BY EMOTION ---
def average_dfc_by_emotion(dfc_matrices, emotion_labels, emotion_list):
    """
    Returns: dict {emotion: avg_matrix}
    """
    result = {}
    for emotion in emotion_list:
        idx = [i for i, lbl in enumerate(emotion_labels) if lbl.lower() == emotion.lower()]
        if idx:
            avg = np.mean(dfc_matrices[idx], axis=0)
            result[emotion] = avg
        else:
            result[emotion] = None
    return result

# --- VISUALIZATION ---
def plot_dfc_topographies(avg_dfc_dict, channel_labels, out_path=None):
    """
    Plots a heatmap for each emotion's average DFC matrix.
    """
    n = len(avg_dfc_dict)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), squeeze=False)
    for i, (emotion, mat) in enumerate(avg_dfc_dict.items()):
        ax = axes[0, i]
        if mat is not None:
            sns.heatmap(mat, vmin=-1, vmax=1, cmap='coolwarm', xticklabels=channel_labels, yticklabels=channel_labels, ax=ax, square=True, cbar=(i==n-1))
            ax.set_title(f"{emotion}")
        else:
            ax.axis('off')
            ax.set_title(f"{emotion}\n(no data)")
    plt.suptitle("Average Dynamic Functional Connectivity by Emotion", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    plt.show()

# --- MAIN EXAMPLE (ADAPT TO YOUR PIPELINE) ---
if __name__ == "__main__":
    # Folder containing EEG CSV files
    eeg_folder = r'C:\Users\lenin\Documents\GitHub\pipeline-eeg-eda_060524\data\raw\eeg'  # <-- update path if needed
    emotion_list = ['Excited', 'Angry', 'Sad', 'Calm']
    all_dfc_matrices = []
    all_emotion_labels = []
    # List all CSV files in the folder
    eeg_files = [os.path.join(eeg_folder, f) for f in os.listdir(eeg_folder) if f.lower().endswith('.csv')]
    if not eeg_files:
        print(f"No EEG CSV files found in {eeg_folder}")
        exit(1)
    for eeg_file in eeg_files:
        print(f"Processing {eeg_file} ...")
        df = pd.read_csv(eeg_file, header=None)
        eeg_data = df.iloc[:NUM_CHANNELS, :].values
        n_samples = eeg_data.shape[1]
        win_len = int(WINDOW_SIZE * SAMPLING_RATE)
        n_windows = (n_samples - win_len) // win_len + 1
        # TODO: Replace with your actual window-wise emotion labels for each file
        # For demo, use random labels (ensure reproducibility per file)
        np.random.seed(abs(hash(eeg_file)) % (2**32))
        emotion_labels = np.random.choice(emotion_list, size=n_windows)
        dfc_matrices = compute_dfc_matrices(eeg_data, WINDOW_SIZE, SAMPLING_RATE)
        all_dfc_matrices.append(dfc_matrices)
        all_emotion_labels.extend(emotion_labels)
    # Concatenate all DFC matrices
    all_dfc_matrices = np.concatenate(all_dfc_matrices, axis=0)
    # Compute average DFC by emotion across all files
    avg_dfc = average_dfc_by_emotion(all_dfc_matrices, all_emotion_labels, emotion_list)
    # Plot and save
    plot_dfc_topographies(avg_dfc, CHANNEL_LABELS, out_path='output/dfc_topography_comparison.png')
    print("Saved DFC topography comparison to output/dfc_topography_comparison.png")
