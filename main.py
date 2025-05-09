"""
Emotion Classification from EEG and EDA Signals
=================================================
Single-file pipeline that processes raw EEG and EDA signals,
extracts features, generates labels, trains classifiers, and evaluates them.

Author: [Your Name]
Date: [Today]

Expected Inputs:
- EEG CSV file (19 columns: 8 EEG channels, 6 motion sensors, + metadata)
- EDA OpenSignals TXT file (3 columns: nSeq, DI, CH1)

Output:
- Metrics per classifier: Accuracy, F1, Confusion Matrix
- Visualizations of EEG/EDA features and classifier results
- 3D topographic maps of EEG activity
- Interactive time-based visualization

Folder structure:
- /data/raw/: input CSV files
- /data/processed/: intermediate results (e.g., filtered signals, features)
- /output/: results and evaluation plots

Run with:
    python main.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, welch, iirnotch
from sklearn.decomposition import FastICA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
from datetime import timedelta
from tqdm import tqdm
import neurokit2 as nk
import mne
from mne.viz import plot_topomap
from mne.channels import make_standard_montage
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec
import logging

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)

# Add a function to handle errors and continue processing
def safe_execute(func, *args, **kwargs):
    """Execute a function safely and continue with the pipeline."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {str(e)}")
        return None

# Define EEG channel mapping with correct anatomical positions
eeg_channels = {
    0: "Fp1",   # Left frontal pole
    1: "Fp2",   # Right frontal pole
    2: "F3",    # Left frontal
    3: "F4",    # Right frontal
    4: "C3",    # Left central
    5: "C4",    # Right central
    6: "P3",    # Left parietal
    7: "P4"     # Right parietal
}

# Update the channel names list to match the mapping
ch_names = [eeg_channels[i] for i in range(8)]

# -------------------------------------------
# Helper: Filtering Functions
# -------------------------------------------
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)

def notch_filter(data, fs=250, freq=50.0, Q=30):
    b, a = iirnotch(freq / (0.5 * fs), Q)
    return lfilter(b, a, data, axis=0)

# -------------------------------------------
# Helper: Visualization Functions
# -------------------------------------------
def create_3d_topomap(eeg_data, ch_names, title="EEG Topography", show=True, time_index=0, total_windows=None):
    """
    Create a topographic map of EEG activity with keyboard navigation.
    """
    # Ensure data is 1D with length matching number of channels
    if len(eeg_data.shape) > 1:
        if eeg_data.shape[1] == len(ch_names):
            eeg_data = np.mean(eeg_data, axis=0)
        else:
            raise ValueError("Data shape does not match number of channels")
    
    # Create MNE info structure with proper montage
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data.reshape(-1, 1), info)
    
    # Set up the standard 10-20 montage
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)
    
    # Create figure with 16:9 aspect ratio (1920x1080)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_subplot(111)
    
    # Plot topography with adjusted parameters
    mne.viz.plot_topomap(
        eeg_data,
        raw.info,
        show=False,
        axes=ax,
        show_names=True,
        sphere=0.07,
        cmap='RdBu_r',
        outlines='head',
        contours=6,
        sensors=True
    )
    
    plt.title(f"{title} - Window {time_index + 1}/{total_windows if total_windows else 1}", fontsize=16, pad=20)
    
    if show:
        plt.draw()
        plt.pause(0.1)  # Small pause to allow the plot to render
    return fig

def plot_band_power_topomap(eeg_data, ch_names, band='alpha', fs=250, show=True, time_index=0, total_windows=None):
    """
    Create a topographic map of band power with keyboard navigation.
    """
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    
    # Calculate power in the specified band
    band_powers = []
    for ch in range(eeg_data.shape[1]):
        # Calculate power spectrum
        f, Pxx = welch(eeg_data[:, ch], fs=fs, nperseg=min(256, len(eeg_data)))
        
        # Find indices for the frequency band
        idx = np.logical_and(f >= bands[band][0], f < bands[band][1])
        
        # Calculate power using trapezoidal integration
        band_power = np.trapz(Pxx[idx], f[idx])
        band_powers.append(band_power)
    
    # Create MNE info structure with proper montage
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(np.array(band_powers).reshape(-1, 1), info)
    
    # Set up the standard 10-20 montage
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)
    
    # Create figure with 16:9 aspect ratio
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_subplot(111)
    
    # Plot topography
    mne.viz.plot_topomap(
        band_powers,
        raw.info,
        show=False,
        axes=ax,
        show_names=True,
        sphere=0.07,
        cmap='RdBu_r'
    )
    
    plt.title(f'{band.capitalize()} Band Power Topography - Window {time_index + 1}/{total_windows if total_windows else 1}', 
              fontsize=16, pad=20)
    
    if show:
        plt.draw()
        plt.pause(0.1)  # Small pause to allow the plot to render
    return fig

def visualize_window(eeg_window, eda_window, ch_names, window_idx, fs_eeg=250, fs_eda=500):
    """
    Create a comprehensive visualization of a time window.
    """
    # Create figure with 16:9 aspect ratio
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 2], hspace=0.3)
    
    # Plot EEG signals
    ax1 = fig.add_subplot(gs[0])
    time = np.arange(len(eeg_window)) / fs_eeg
    for ch in range(eeg_window.shape[1]):
        ax1.plot(time, eeg_window[:, ch], label=ch_names[ch], linewidth=1.5)
    ax1.set_title(f'EEG Signals - Window {window_idx}', fontsize=16, pad=20)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Amplitude (µV)', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot EDA signal with proper scaling
    ax2 = fig.add_subplot(gs[1])
    time_eda = np.arange(len(eda_window)) / fs_eda
    ax2.plot(time_eda, eda_window, linewidth=1.5, color='green')
    ax2.set_title('EDA Signal', fontsize=16, pad=20)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Conductance (µS)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Set y-limits for EDA with some padding
    eda_min = np.min(eda_window)
    eda_max = np.max(eda_window)
    eda_range = eda_max - eda_min
    ax2.set_ylim(eda_min - 0.1 * eda_range, eda_max + 0.1 * eda_range)
    
    # Plot topography
    ax3 = fig.add_subplot(gs[2])
    mean_activity = np.mean(eeg_window, axis=0)
    
    # Create MNE info structure with proper montage
    info = mne.create_info(ch_names=ch_names, sfreq=fs_eeg, ch_types='eeg')
    raw = mne.io.RawArray(mean_activity.reshape(-1, 1), info)
    
    # Set up the standard 10-20 montage
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)
    
    mne.viz.plot_topomap(
        mean_activity,
        raw.info,
        show=False,
        axes=ax3,
        show_names=True,
        sphere=0.07,
        cmap='RdBu_r'
    )
    ax3.set_title(f'Average EEG Activity - Window {window_idx}', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # Small pause to allow the plot to render
    return fig

# -------------------------------------------
# Helper: Interactive Visualization Functions
# -------------------------------------------
def create_interactive_visualization(eeg_data, eda_data, ch_names, fs_eeg=250, fs_eda=500, window_size=1000):
    """
    Create an interactive visualization with a slider to move through time.
    """
    # Create figure with 16:9 aspect ratio
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 0.5, 3], hspace=0.3)
    
    # EEG plot
    ax_eeg = plt.subplot(gs[0])
    # EDA plot
    ax_eda = plt.subplot(gs[1])
    # Slider
    ax_slider = plt.subplot(gs[2])
    # Topography
    ax_top = plt.subplot(gs[3])
    
    # Initial time point
    init_time = 0
    current_time = init_time
    
    # Create time arrays
    time_eeg = np.arange(len(eeg_data)) / fs_eeg
    time_eda = np.arange(len(eda_data)) / fs_eda
    
    # Plot initial EEG data
    eeg_lines = []
    for ch in range(eeg_data.shape[1]):
        line, = ax_eeg.plot(time_eeg[:window_size], 
                          eeg_data[:window_size, ch],
                          label=ch_names[ch],
                          linewidth=1.5)
        eeg_lines.append(line)
    
    # Plot initial EDA data with proper scaling
    eda_line, = ax_eda.plot(time_eda[:window_size*2], 
                           eda_data[:window_size*2],
                           linewidth=1.5,
                           color='green')
    
    # Set up the slider with better visibility
    slider = Slider(
        ax_slider,
        'Time (s)',
        0,
        len(eeg_data)/fs_eeg - window_size/fs_eeg,
        valinit=init_time,
        color='lightgray',
        initcolor='none'
    )
    slider.label.set_fontsize(12)
    slider.valtext.set_fontsize(12)
    
    # Set up the topography
    info = mne.create_info(ch_names=ch_names, sfreq=fs_eeg, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data[:window_size].T, info)
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    # Initial topography
    mne.viz.plot_topomap(
        np.mean(eeg_data[:window_size], axis=0),
        raw.info,
        show=False,
        axes=ax_top,
        show_names=True,
        sphere=0.07,
        cmap='RdBu_r'
    )
    
    # Set up the plots
    ax_eeg.set_title('EEG Signals', fontsize=16, pad=20)
    ax_eeg.set_xlabel('Time (s)', fontsize=12)
    ax_eeg.set_ylabel('Amplitude (µV)', fontsize=12)
    ax_eeg.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax_eeg.grid(True, alpha=0.3)
    
    ax_eda.set_title('EDA Signal', fontsize=16, pad=20)
    ax_eda.set_xlabel('Time (s)', fontsize=12)
    ax_eda.set_ylabel('Conductance (µS)', fontsize=12)
    ax_eda.grid(True, alpha=0.3)
    
    ax_top.set_title('EEG Topography', fontsize=16, pad=20)
    
    def update(val):
        nonlocal current_time
        current_time = min(max(int(slider.val * fs_eeg), 0), len(eeg_data) - window_size)
        eda_time_point = min(int(current_time * fs_eda / fs_eeg), len(eda_data) - window_size * 2)
        
        # Update EEG plot
        for ch in range(eeg_data.shape[1]):
            eeg_lines[ch].set_xdata(time_eeg[current_time:current_time+window_size])
            eeg_lines[ch].set_ydata(eeg_data[current_time:current_time+window_size, ch])
        
        # Update EDA plot
        eda_line.set_xdata(time_eda[eda_time_point:eda_time_point+window_size*2])
        eda_line.set_ydata(eda_data[eda_time_point:eda_time_point+window_size*2])
        
        # Update topography
        ax_top.clear()
        mne.viz.plot_topomap(
            np.mean(eeg_data[current_time:current_time+window_size], axis=0),
            raw.info,
            show=False,
            axes=ax_top,
            show_names=True,
            sphere=0.07,
            cmap='RdBu_r'
        )
        ax_top.set_title('EEG Topography', fontsize=16, pad=20)
        
        # Update plot limits
        ax_eeg.set_xlim(time_eeg[current_time], time_eeg[current_time+window_size])
        ax_eda.set_xlim(time_eda[eda_time_point], time_eda[eda_time_point+window_size*2])
        
        # Update y-limits with padding
        eeg_min = eeg_data[current_time:current_time+window_size].min()
        eeg_max = eeg_data[current_time:current_time+window_size].max()
        eeg_range = eeg_max - eeg_min
        ax_eeg.set_ylim(eeg_min - 0.1 * eeg_range, eeg_max + 0.1 * eeg_range)
        
        eda_min = eda_data[eda_time_point:eda_time_point+window_size*2].min()
        eda_max = eda_data[eda_time_point:eda_time_point+window_size*2].max()
        eda_range = eda_max - eda_min
        ax_eda.set_ylim(eda_min - 0.1 * eda_range, eda_max + 0.1 * eda_range)
        
        fig.canvas.draw_idle()
    
    def on_key(event):
        nonlocal current_time
        if event.key == 'right':
            # Move forward one window
            new_time = min(current_time + window_size, len(eeg_data) - window_size)
            slider.set_val(new_time / fs_eeg)
        elif event.key == 'left':
            # Move backward one window
            new_time = max(current_time - window_size, 0)
            slider.set_val(new_time / fs_eeg)
    
    # Connect the slider to the update function
    slider.on_changed(update)
    
    # Connect keyboard events
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Add a reset button with better visibility
    reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_ax, 'Reset', color='lightgray', hovercolor='0.975')
    
    def reset(event):
        slider.reset()
    
    reset_button.on_clicked(reset)
    
    # Add keyboard navigation instructions
    plt.figtext(0.02, 0.02, 'Use ← and → arrow keys to navigate', fontsize=10)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # Small pause to allow the plot to render
    return fig

def create_unified_visualization(eeg_data, eda_data, ch_names, fs_eeg=250, fs_eda=500, window_size=1000):
    """
    Create a unified visualization interface with all plots and controls.
    """
    # Create figure with 16:9 aspect ratio
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    gs = gridspec.GridSpec(4, 2, height_ratios=[3, 1, 0.5, 3], hspace=0.3, wspace=0.3)
    
    # EEG plot (left)
    ax_eeg = plt.subplot(gs[0, 0])
    # EDA plot (left)
    ax_eda = plt.subplot(gs[1, 0])
    # Controls (left)
    ax_controls = plt.subplot(gs[2, 0])
    # Topography (left)
    ax_top = plt.subplot(gs[3, 0])
    
    # Band power plots (right)
    ax_bands = plt.subplot(gs[0:2, 1])
    # Confusion matrix (right)
    ax_conf = plt.subplot(gs[2:, 1])
    
    # Initial time point
    init_time = 0
    current_time = init_time
    
    # Create time arrays
    time_eeg = np.arange(len(eeg_data)) / fs_eeg
    time_eda = np.arange(len(eda_data)) / fs_eda
    
    # Calculate maximum valid time
    max_time = len(eeg_data) - window_size
    max_time_eda = len(eda_data) - window_size * 2
    
    # Plot initial EEG data
    eeg_lines = []
    for ch in range(eeg_data.shape[1]):
        line, = ax_eeg.plot(time_eeg[:window_size], 
                          eeg_data[:window_size, ch],
                          label=ch_names[ch],
                          linewidth=1.5)
        eeg_lines.append(line)
    
    # Plot initial EDA data
    eda_line, = ax_eda.plot(time_eda[:window_size*2], 
                           eda_data[:window_size*2],
                           linewidth=1.5,
                           color='green')
    
    # Set up the controls
    slider = Slider(
        ax_controls,
        'Time (s)',
        0,
        max_time / fs_eeg,  # Use max_time instead of calculating it
        valinit=init_time,
        color='lightgray',
        initcolor='none'
    )
    slider.label.set_fontsize(12)
    slider.valtext.set_fontsize(12)
    
    # Add window size control
    window_ax = plt.axes([0.2, 0.02, 0.1, 0.04])
    window_button = Button(window_ax, 'Window Size', color='lightgray', hovercolor='0.975')
    
    # Add play/pause button
    play_ax = plt.axes([0.35, 0.02, 0.1, 0.04])
    play_button = Button(play_ax, 'Play', color='lightgray', hovercolor='0.975')
    
    # Add speed control
    speed_ax = plt.axes([0.5, 0.02, 0.1, 0.04])
    speed_button = Button(speed_ax, 'Speed', color='lightgray', hovercolor='0.975')
    
    # Set up the topography
    info = mne.create_info(ch_names=ch_names, sfreq=fs_eeg, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data[:window_size].T, info)
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)
    
    # Initial topography
    mne.viz.plot_topomap(
        np.mean(eeg_data[:window_size], axis=0),
        raw.info,
        show=False,
        axes=ax_top,
        show_names=True,
        sphere=0.07,
        cmap='RdBu_r',
        outlines='head',
        contours=6,
        sensors=True
    )
    
    # Plot band powers
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    
    band_powers = []
    for band, (low, high) in bands.items():
        powers = []
        for ch in range(eeg_data.shape[1]):
            f, Pxx = welch(eeg_data[:window_size, ch], fs=fs_eeg, nperseg=min(256, window_size))
            idx = np.logical_and(f >= low, f < high)
            power = np.trapz(Pxx[idx], f[idx])
            powers.append(power)
        band_powers.append(powers)
    
    # Plot band powers as bar chart
    x = np.arange(len(ch_names))
    width = 0.15
    for i, (band, powers) in enumerate(zip(bands.keys(), band_powers)):
        ax_bands.bar(x + i*width, powers, width, label=band)
    
    ax_bands.set_ylabel('Power')
    ax_bands.set_title('Band Powers')
    ax_bands.set_xticks(x + width*2)
    ax_bands.set_xticklabels(ch_names)
    ax_bands.legend()
    
    # Set up the plots
    ax_eeg.set_title('EEG Signals', fontsize=16, pad=20)
    ax_eeg.set_xlabel('Time (s)', fontsize=12)
    ax_eeg.set_ylabel('Amplitude (µV)', fontsize=12)
    ax_eeg.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax_eeg.grid(True, alpha=0.3)
    
    ax_eda.set_title('EDA Signal', fontsize=16, pad=20)
    ax_eda.set_xlabel('Time (s)', fontsize=12)
    ax_eda.set_ylabel('Conductance (µS)', fontsize=12)
    ax_eda.grid(True, alpha=0.3)
    
    ax_top.set_title('EEG Topography', fontsize=16, pad=20)
    
    # Playback state
    is_playing = False
    playback_speed = 1.0
    
    def update(val):
        nonlocal current_time
        # Ensure current_time stays within bounds
        current_time = min(max(int(slider.val * fs_eeg), 0), max_time)
        eda_time_point = min(int(current_time * fs_eda / fs_eeg), max_time_eda)
        
        # Update EEG plot
        for ch in range(eeg_data.shape[1]):
            eeg_lines[ch].set_xdata(time_eeg[current_time:current_time+window_size])
            eeg_lines[ch].set_ydata(eeg_data[current_time:current_time+window_size, ch])
        
        # Update EDA plot
        eda_line.set_xdata(time_eda[eda_time_point:eda_time_point+window_size*2])
        eda_line.set_ydata(eda_data[eda_time_point:eda_time_point+window_size*2])
        
        # Update topography
        ax_top.clear()
        mne.viz.plot_topomap(
            np.mean(eeg_data[current_time:current_time+window_size], axis=0),
            raw.info,
            show=False,
            axes=ax_top,
            show_names=True,
            sphere=0.07,
            cmap='RdBu_r',
            outlines='head',
            contours=6,
            sensors=True
        )
        ax_top.set_title('EEG Topography', fontsize=16, pad=20)
        
        # Update band powers
        ax_bands.clear()
        band_powers = []
        for band, (low, high) in bands.items():
            powers = []
            for ch in range(eeg_data.shape[1]):
                f, Pxx = welch(eeg_data[current_time:current_time+window_size, ch], 
                             fs=fs_eeg, nperseg=min(256, window_size))
                idx = np.logical_and(f >= low, f < high)
                power = np.trapz(Pxx[idx], f[idx])
                powers.append(power)
            band_powers.append(powers)
        
        for i, (band, powers) in enumerate(zip(bands.keys(), band_powers)):
            ax_bands.bar(x + i*width, powers, width, label=band)
        
        ax_bands.set_ylabel('Power')
        ax_bands.set_title('Band Powers')
        ax_bands.set_xticks(x + width*2)
        ax_bands.set_xticklabels(ch_names)
        ax_bands.legend()
        
        # Update plot limits
        ax_eeg.set_xlim(time_eeg[current_time], time_eeg[current_time+window_size])
        ax_eda.set_xlim(time_eda[eda_time_point], time_eda[eda_time_point+window_size*2])
        
        # Update y-limits with padding
        eeg_min = eeg_data[current_time:current_time+window_size].min()
        eeg_max = eeg_data[current_time:current_time+window_size].max()
        eeg_range = eeg_max - eeg_min
        ax_eeg.set_ylim(eeg_min - 0.1 * eeg_range, eeg_max + 0.1 * eeg_range)
        
        eda_min = eda_data[eda_time_point:eda_time_point+window_size*2].min()
        eda_max = eda_data[eda_time_point:eda_time_point+window_size*2].max()
        eda_range = eda_max - eda_min
        ax_eda.set_ylim(eda_min - 0.1 * eda_range, eda_max + 0.1 * eda_range)
        
        fig.canvas.draw_idle()
    
    def on_key(event):
        nonlocal current_time, is_playing, playback_speed
        if event.key == 'right':
            # Move forward one window
            new_time = min(current_time + window_size, max_time)
            slider.set_val(new_time / fs_eeg)
        elif event.key == 'left':
            # Move backward one window
            new_time = max(current_time - window_size, 0)
            slider.set_val(new_time / fs_eeg)
        elif event.key == ' ':
            # Toggle play/pause
            is_playing = not is_playing
            play_button.label.set_text('Pause' if is_playing else 'Play')
            if is_playing:
                play_animation()
    
    def play_animation():
        nonlocal current_time, is_playing
        if is_playing:
            new_time = min(current_time + int(window_size * playback_speed), max_time)
            if new_time == max_time:
                is_playing = False
                play_button.label.set_text('Play')
            else:
                slider.set_val(new_time / fs_eeg)
                fig.canvas.callbacks.process('draw_event', None)
                fig.canvas.start_event_loop(0.001)
    
    def toggle_play(event):
        nonlocal is_playing
        is_playing = not is_playing
        play_button.label.set_text('Pause' if is_playing else 'Play')
        if is_playing:
            play_animation()
    
    def change_window_size(event):
        nonlocal window_size
        window_size = int(window_size * 1.5) if window_size < 4000 else 1000
        # Update max_time after changing window size
        max_time = len(eeg_data) - window_size
        max_time_eda = len(eda_data) - window_size * 2
        # Update slider range
        slider.valmax = max_time / fs_eeg
        update(slider.val)
    
    def change_speed(event):
        nonlocal playback_speed
        playback_speed = 2.0 if playback_speed == 1.0 else 1.0
        speed_button.label.set_text(f'Speed: {playback_speed}x')
    
    # Connect the slider to the update function
    slider.on_changed(update)
    
    # Connect keyboard events
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Connect button events
    play_button.on_clicked(toggle_play)
    window_button.on_clicked(change_window_size)
    speed_button.on_clicked(change_speed)
    
    # Add keyboard navigation instructions
    plt.figtext(0.02, 0.02, 'Use ← and → arrow keys to navigate, Space to play/pause', fontsize=10)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # Small pause to allow the plot to render
    return fig

# -------------------------------------------
# Step 0: Initial Data Analysis
# -------------------------------------------
def analyze_file(file_path, file_type):
    """Analyze the content and duration of input files."""
    print(f"\n{'='*50}")
    print(f"Analyzing {file_type} file: {os.path.basename(file_path)}")
    print(f"{'='*50}")
    
    if file_type == 'EEG':
        df = pd.read_csv(file_path)
        fs = 250  # EEG sampling rate
        duration = len(df) / fs
        print(f"Number of samples: {len(df)}")
        print(f"Duration: {timedelta(seconds=duration)}")
        print(f"Number of channels: {len(df.columns)}")
        
        # Map EEG channels with correct positions
        eeg_channels = {
            0: "Fz",    # Forehead
            1: "Cz",    # Crown
            2: "P3",    # Left parietal
            3: "Pz",    # Midline parietal
            4: "P4",    # Right parietal
            5: "PO7",   # Left parieto-occipital
            6: "Oz",    # Occipital
            7: "PO8"    # Right parieto-occipital
        }
        
        # Print channel information
        print("\nChannel mapping:")
        for i in range(8):
            col_name = df.columns[i]
            print(f"- {col_name}: {eeg_channels.get(i, 'Unknown')} (EEG channel {i+1} in microvolts)")
        
        print("\nMotion sensors:")
        print("- ACC X, Y, Z: Accelerometer in g")
        print("- GYR X, Y, Z: Gyroscope in deg/s")
        
        print("\nAdditional channels:")
        print("- CNT: Counter")
        print("- BAT: Battery Level (%)")
        print("- VALID: Validation Indicator")
        print("- DT: Delta time (ms)")
        print("- STATUS: Status/Trigger Value")
        
        # Basic statistics for EEG channels
        print("\nBasic statistics for EEG channels:")
        eeg_cols = df.iloc[:, 0:8].columns
        stats = df[eeg_cols].describe()
        print(stats)
        
        # Create initial topography
        print("\nCreating initial topography...")
        ch_names = [eeg_channels[i] for i in range(8)]
        mean_activity = np.mean(df.iloc[:, 0:8].values, axis=0)
        create_3d_topomap(mean_activity, ch_names, 
                         title="Initial EEG Topography")
        
    elif file_type == 'EDA':
        try:
            # Read header first
            with open(file_path, 'r') as f:
                header_lines = [next(f) for _ in range(3)]
                print("\nHeader lines:")
                for i, line in enumerate(header_lines):
                    print(f"Line {i+1}: {line.strip()}")
            
            # Parse sampling rate from header
            import json
            header_json = json.loads(header_lines[1].strip('# '))
            fs = header_json['00:07:80:8C:07:08']['sampling rate']
            
            # Read data with proper parsing
            print("\nReading data...")
            # Read the raw data first to understand the format
            with open(file_path, 'r') as f:
                # Skip header
                for _ in range(3):
                    next(f)
                # Read first few lines to determine format
                first_lines = [next(f).strip() for _ in range(5)]
                print("\nFirst few raw data lines:")
                for line in first_lines:
                    print(line)
                
                # Reset file pointer to after header
                f.seek(0)
                for _ in range(3):
                    next(f)
                
                # Read data with proper parsing
                data = []
                for line in f:
                    try:
                        # Split by whitespace and convert to float
                        values = [float(x) for x in line.strip().split()]
                        if len(values) >= 3:  # Ensure we have all three columns
                            data.append(values)
                    except ValueError:
                        continue
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['nSeq', 'DI', 'CH1'])
            print(f"\nRaw data shape: {df.shape}")
            print("\nFirst few rows of processed data:")
            print(df.head())
            
            if len(df) == 0:
                raise ValueError("No valid EDA data found in file")
            
            duration = len(df) / fs
            print(f"\nNumber of samples: {len(df)}")
            print(f"Duration: {timedelta(seconds=duration)}")
            print(f"Sampling rate: {fs} Hz")
            
            # Basic statistics for EDA
            print("\nBasic statistics for EDA signal:")
            stats = df['CH1'].describe()
            print(stats)
            
            # Plot a small segment of the signal
            plt.figure(figsize=(12, 4))
            plt.plot(df['CH1'].iloc[:5000])  # First 10 seconds
            plt.title('EDA Signal (First 10 seconds)')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude (µS)')
            plt.grid(True)
            plt.draw()
            plt.pause(0.1)
            
        except Exception as e:
            print(f"\nError loading EDA file: {str(e)}")
            print("Please check if the file format is correct")
            print(f"File path: {file_path}")
            print("\nDebug information:")
            print(f"File exists: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                print(f"File size: {os.path.getsize(file_path)} bytes")
            return None, None
    
    return df, fs

# -------------------------------------------
# Step 1: Load and Analyze EEG + EDA files
# -------------------------------------------
eeg_file = 'data/raw/UnicornRecorder_baseline.csv'
eda_file = 'data/raw/opensignals_lsl_500hz_gain1_0007808C0708_16-32-15_converted.txt'

# Define window size in seconds - adjusted for shorter recordings
window_sec = 5  # 5-second windows instead of 30

print("\nStarting data analysis...")
eeg_df, fs_eeg = analyze_file(eeg_file, 'EEG')
eda_df, fs_eda = analyze_file(eda_file, 'EDA')

# Check if EDA file was loaded successfully
if eda_df is None or fs_eda is None:
    print("\n❌ Failed to load EDA file. Please check the file format and path.")
    print("Expected format: OpenSignals TXT file with header and tab-separated data")
    print("File path: ", eda_file)
    print("\nContinuing with EEG data only...")
    
    # Create a dummy EDA signal with the same duration as EEG
    n_eeg_samples = len(eeg_df)
    eda_signal = np.zeros(n_eeg_samples)  # Dummy EDA signal
    fs_eda = fs_eeg  # Use same sampling rate as EEG
else:
    # EDA signal is the CH1 column
    eda_signal = eda_df['CH1'].values.astype(float)

# Only EEG channels 1–8 (in microvolts)
eeg_data = eeg_df.iloc[:, 0:8].values.astype(float)
gyro_data = eeg_df.iloc[:, 11:14].values.astype(float)

# -------------------------------------------
# Step 2: Filter EEG (Notch + Bandpass), ICA, and Motion Artifact Removal
# -------------------------------------------
print("\nProcessing EEG signals...")
eeg_filtered = notch_filter(eeg_data, fs=fs_eeg)
eeg_filtered = bandpass_filter(eeg_filtered, 0.5, 40, fs_eeg)

# Gyro magnitude: motion filter (simple thresholding method)
gyro_mag = np.linalg.norm(gyro_data, axis=1)
motion_threshold = np.percentile(gyro_mag, 95)
valid_mask = gyro_mag < motion_threshold
eeg_filtered[~valid_mask] = 0  # remove strong movement samples

# ICA for artifact removal
print("Performing ICA...")
ica = FastICA(n_components=8)
eeg_ica = ica.fit_transform(eeg_filtered)
eeg_reconstructed = ica.inverse_transform(eeg_ica)

# -------------------------------------------
# Step 3: Segment Signals into Windows
# -------------------------------------------
print("\nSegmenting signals into windows...")
# Calculate window sizes for both signals
eeg_samples_per_window = fs_eeg * window_sec  # 250 * 5 = 1250 samples
eda_samples_per_window = fs_eda * window_sec  # 500 * 5 = 2500 samples

# Calculate number of complete windows possible
n_eeg_samples = len(eeg_reconstructed)
n_eda_samples = len(eda_signal)

total_windows = min(n_eeg_samples // eeg_samples_per_window, 
                   n_eda_samples // eda_samples_per_window)

print(f"Window size: {window_sec} seconds")
print(f"Number of complete windows possible: {total_windows}")

# Adjust minimum windows required based on data length
min_windows = min(5, total_windows)
if total_windows < min_windows:
    print(f"\n⚠️ Warning: Limited data available. Using {total_windows} windows instead of the ideal {min_windows}.")
    print("Results may be less reliable with fewer windows.")
    if total_windows < 2:
        raise ValueError("Not enough samples to create at least 2 windows. Please record longer data.")

eeg_windows = []
eda_windows = []

for i in tqdm(range(total_windows), desc="Creating windows"):
    # Calculate start and end indices for EEG
    eeg_start = i * eeg_samples_per_window
    eeg_end = eeg_start + eeg_samples_per_window
    eda_start = i * eda_samples_per_window
    eda_end = eda_start + eda_samples_per_window
    
    eeg_win = eeg_reconstructed[eeg_start:eeg_end, :]
    eda_win = eda_signal[eda_start:eda_end]
    
    eeg_windows.append(eeg_win)
    eda_windows.append(eda_win)

eeg_windows = np.array(eeg_windows)
eda_windows = np.array(eda_windows)

# -------------------------------------------
# Step 4: Feature Extraction (EEG + EDA)
# -------------------------------------------
def extract_eeg_features(window, fs=250):
    """Extract EEG features including band ratios."""
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    
    features = []
    for ch in range(window.shape[1]):
        # Calculate power spectrum
        f, Pxx = welch(window[:, ch], fs=fs, nperseg=min(256, len(window)))
        
        # Calculate power in each band
        band_powers = {}
        for band, (low, high) in bands.items():
            # Find indices for the frequency band
            idx = np.logical_and(f >= low, f < high)
            # Calculate power using trapezoidal integration
            band_powers[band] = np.trapz(Pxx[idx], f[idx])
        
        # Calculate band ratios
        theta_alpha = band_powers['theta'] / band_powers['alpha'] if band_powers['alpha'] > 0 else 0
        beta_alpha = band_powers['beta'] / band_powers['alpha'] if band_powers['alpha'] > 0 else 0
        
        # Add all features
        features.extend([
            band_powers['delta'],
            band_powers['theta'],
            band_powers['alpha'],
            band_powers['beta'],
            band_powers['gamma'],
            theta_alpha,
            beta_alpha
        ])
    
    return np.array(features)

def extract_eda_features(window, fs=500):
    """Extract EDA features including phasic component."""
    # Check if window is all zeros (dummy signal)
    if np.all(window == 0):
        # Return dummy features for zero signal
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    try:
        # Extract phasic component using neurokit2
        eda_cleaned = nk.eda_clean(window, sampling_rate=fs)
        eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=fs)
        
        # Get phasic and tonic components
        phasic = eda_decomposed['EDA_Phasic'].values
        tonic = eda_decomposed['EDA_Tonic'].values
        
        # Calculate features
        scl = np.mean(tonic)
        scr_amp = np.mean(phasic)
        
        # Handle peak detection safely
        try:
            peaks = nk.eda_peaks(eda_cleaned, sampling_rate=fs)
            scr_count = len(peaks[0]) if peaks[0] is not None else 0
        except:
            scr_count = 0
        
        return np.array([scl, scr_count, scr_amp, np.mean(phasic), np.std(phasic)])
    except Exception as e:
        print(f"Warning: Error in EDA feature extraction: {str(e)}")
        # Return dummy features if processing fails
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])

print("\nExtracting features...")
X = []
for i in tqdm(range(total_windows), desc="Extracting features", unit="window"):
    print(f"\nProcessing window {i+1}/{total_windows}")
    print("Extracting EEG features...")
    eeg_feat = safe_execute(extract_eeg_features, eeg_windows[i], fs=fs_eeg)
    print("Extracting EDA features...")
    eda_feat = safe_execute(extract_eda_features, eda_windows[i], fs=fs_eda)
    if eeg_feat is not None and eda_feat is not None:
    X.append(np.concatenate([eeg_feat, eda_feat]))

X = np.array(X)
print(f"\n✅ Extracted {X.shape[1]} features from {X.shape[0]} windows")

# -------------------------------------------
# Step 5: Label Creation (Simulated / Ideal)
# -------------------------------------------
emotions = ['calm', 'excited', 'angry', 'sad']
Y = np.array([emotions[i % 4] for i in range(total_windows)])

# Encode as integers
label_map = {e: i for i, e in enumerate(emotions)}
y_encoded = np.array([label_map[y] for y in Y])

print(f"\nFeature matrix shape: {X.shape}")
print(f"Labels shape: {y_encoded.shape}")

# -------------------------------------------
# Step 6: Classifier Training and Evaluation
# -------------------------------------------
print("\n🤖 Step 6: Training and evaluating classifiers...")
classifiers = {
    'SVM': SVC(kernel='rbf', C=1, gamma='scale'),
    'kNN': KNeighborsClassifier(n_neighbors=min(3, total_windows-1)),  # Adjust k based on data size
    'RF': RandomForestClassifier(n_estimators=50),  # Reduced number of trees
    'DT': DecisionTreeClassifier(),
    'MLP': MLPClassifier(hidden_layer_sizes=(20,), max_iter=300)  # Smaller network
}

print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# For small datasets, use a simple train-test split
print("\nUsing train-test split (70-30)")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Create a figure for classification results
plt.figure(figsize=(19.2, 10.8), dpi=100)
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

for idx, (name, clf) in enumerate(tqdm(classifiers.items(), desc="Evaluating classifiers")):
    print(f"\nTraining {name} classifier...")
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{name} Classifier Results:")
    print(f"Accuracy: {acc:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    ax = plt.subplot(gs[idx // 3, idx % 3])
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'{name}\nAccuracy: {acc:.2f}, F1: {f1:.2f}')
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(emotions))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(emotions, rotation=45)
    ax.set_yticklabels(emotions)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

print("\n✅ Pipeline complete. Features extracted and classifiers evaluated.")

# After windowing, add visualization of a sample window
print("\nVisualizing sample window...")
sample_window_idx = 0  # You can change this to view different windows
ch_names = [eeg_channels[i] for i in range(8)]
visualize_window(eeg_windows[sample_window_idx], 
                eda_windows[sample_window_idx],
                ch_names,
                sample_window_idx)

# After feature extraction, visualize band power topographies
print("\nVisualizing band power topographies...")
for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
    plot_band_power_topomap(eeg_windows[sample_window_idx], 
                           ch_names,
                           band=band,
                           fs=fs_eeg)

# After the main processing, add the unified visualization
print("\n🎨 Creating unified visualization...")
ch_names = [eeg_channels[i] for i in range(8)]
unified_fig = create_unified_visualization(
    eeg_data=eeg_reconstructed,
    eda_data=eda_signal,
    ch_names=ch_names,
    fs_eeg=fs_eeg,
    fs_eda=fs_eda,
    window_size=1000  # 4 seconds at 250 Hz
)

# Update the main processing section with better progress reporting
if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("Starting EEG and EDA Processing Pipeline")
        print("="*80)
        
        # Set matplotlib to non-interactive mode
        plt.ioff()
        
        # Step 1: Load and Analyze Files
        print("\n📂 Step 1: Loading and analyzing input files...")
        eeg_df, fs_eeg = safe_execute(analyze_file, eeg_file, 'EEG')
        eda_df, fs_eda = safe_execute(analyze_file, eda_file, 'EDA')
        
        if eeg_df is None or eda_df is None:
            raise ValueError("Failed to load input files")
        
        print("\n✅ Files loaded successfully!")
        print(f"EEG sampling rate: {fs_eeg} Hz")
        print(f"EDA sampling rate: {fs_eda} Hz")
        
        # Extract data with correct column indices
        print("\n📊 Extracting EEG and motion data...")
        # EEG channels (1-8)
        eeg_data = eeg_df.iloc[:, 0:8].values.astype(float)
        # Accelerometer (9-11)
        acc_data = eeg_df.iloc[:, 8:11].values.astype(float)
        # Gyroscope (12-14)
        gyro_data = eeg_df.iloc[:, 11:14].values.astype(float)
        # EDA signal
        eda_signal = eda_df['CH1'].values.astype(float)
        
        print(f"EEG data shape: {eeg_data.shape}")
        print(f"EDA data shape: {eda_signal.shape}")
        
        # Step 2: Filter EEG
        print("\n🔍 Step 2: Filtering EEG signals...")
        print("Applying notch filter...")
        eeg_filtered = safe_execute(notch_filter, eeg_data, fs=fs_eeg)
        if eeg_filtered is not None:
            print("Applying bandpass filter...")
            eeg_filtered = safe_execute(bandpass_filter, eeg_filtered, 0.5, 40, fs_eeg)
        
        if eeg_filtered is None:
            raise ValueError("Failed to filter EEG data")
        
        print("✅ Filtering completed successfully!")
        
        # Motion artifact removal using both accelerometer and gyroscope
        print("\n🧹 Removing motion artifacts...")
        # Calculate magnitude for both accelerometer and gyroscope
        acc_mag = np.linalg.norm(acc_data, axis=1)
        gyro_mag = np.linalg.norm(gyro_data, axis=1)
        
        # Combine both motion metrics
        motion_threshold_acc = np.percentile(acc_mag, 95)
        motion_threshold_gyro = np.percentile(gyro_mag, 95)
        
        # Mark samples as invalid if either accelerometer or gyroscope exceeds threshold
        valid_mask = (acc_mag < motion_threshold_acc) & (gyro_mag < motion_threshold_gyro)
        eeg_filtered[~valid_mask] = 0
        
        print(f"Motion artifact removal statistics:")
        print(f"- Accelerometer threshold: {motion_threshold_acc:.2f} g")
        print(f"- Gyroscope threshold: {motion_threshold_gyro:.2f} deg/s")
        print(f"- Removed {np.sum(~valid_mask)} samples due to motion")
        
        # ICA
        print("\n🔄 Performing ICA...")
        ica = FastICA(n_components=8)
        print("Fitting ICA...")
        eeg_ica = safe_execute(ica.fit_transform, eeg_filtered)
        if eeg_ica is not None:
            print("Reconstructing signals...")
            eeg_reconstructed = safe_execute(ica.inverse_transform, eeg_ica)
        else:
            raise ValueError("Failed to perform ICA")
        
        print("✅ ICA completed successfully!")
        
        # Step 3: Windowing
        print("\n⏱️ Step 3: Creating signal windows...")
        window_sec = 5  # 5-second windows
        eeg_samples_per_window = fs_eeg * window_sec
        eda_samples_per_window = fs_eda * window_sec
        
        n_eeg_samples = len(eeg_reconstructed)
        n_eda_samples = len(eda_signal)
        
        total_windows = min(n_eeg_samples // eeg_samples_per_window, 
                          n_eda_samples // eda_samples_per_window)
        
        print(f"Window size: {window_sec} seconds")
        print(f"Total windows: {total_windows}")
        
        if total_windows < 2:
            raise ValueError(f"Not enough samples for 5-fold cross validation. Only {total_windows} windows possible.")
        
        # Create windows
        print("\nCreating windows...")
        eeg_windows = []
        eda_windows = []
        
        for i in tqdm(range(total_windows), desc="Creating windows", unit="window"):
            eeg_start = i * eeg_samples_per_window
            eeg_end = eeg_start + eeg_samples_per_window
            eda_start = i * eda_samples_per_window
            eda_end = eda_start + eda_samples_per_window
            
            eeg_win = eeg_reconstructed[eeg_start:eeg_end, :]
            eda_win = eda_signal[eda_start:eda_end]
            
            eeg_windows.append(eeg_win)
            eda_windows.append(eda_win)
        
        eeg_windows = np.array(eeg_windows)
        eda_windows = np.array(eda_windows)
        
        print(f"✅ Created {len(eeg_windows)} windows")
        
        # Step 4: Feature Extraction
        print("\n📈 Step 4: Extracting features...")
        X = []
        for i in tqdm(range(total_windows), desc="Extracting features", unit="window"):
            print(f"\nProcessing window {i+1}/{total_windows}")
            print("Extracting EEG features...")
            eeg_feat = safe_execute(extract_eeg_features, eeg_windows[i], fs=fs_eeg)
            print("Extracting EDA features...")
            eda_feat = safe_execute(extract_eda_features, eda_windows[i], fs=fs_eda)
            if eeg_feat is not None and eda_feat is not None:
                X.append(np.concatenate([eeg_feat, eda_feat]))
        
        X = np.array(X)
        print(f"\n✅ Extracted {X.shape[1]} features from {X.shape[0]} windows")
        
        # Step 5: Label Creation
        print("\n🏷️ Step 5: Creating labels...")
        emotions = ['calm', 'excited', 'angry', 'sad']
        Y = np.array([emotions[i % 4] for i in range(total_windows)])
        label_map = {e: i for i, e in enumerate(emotions)}
        y_encoded = np.array([label_map[y] for y in Y])
        
        print("Label distribution:")
        for emotion in emotions:
            count = np.sum(Y == emotion)
            print(f"- {emotion}: {count} windows")
        
        # Step 6: Classification
        print("\n🤖 Step 6: Training and evaluating classifiers...")
        classifiers = {
            'SVM': SVC(kernel='rbf', C=1, gamma='scale'),
            'kNN': KNeighborsClassifier(n_neighbors=min(3, total_windows-1)),  # Adjust k based on data size
            'RF': RandomForestClassifier(n_estimators=50),  # Reduced number of trees
            'DT': DecisionTreeClassifier(),
            'MLP': MLPClassifier(hidden_layer_sizes=(20,), max_iter=300)  # Smaller network
        }
        
        print("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # For small datasets, use a simple train-test split
        print("\nUsing train-test split (70-30)")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
        
        # Create a figure for classification results
        plt.figure(figsize=(19.2, 10.8), dpi=100)
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
        
        for idx, (name, clf) in enumerate(tqdm(classifiers.items(), desc="Evaluating classifiers")):
            print(f"\nTraining {name} classifier...")
            
            # Train the classifier
            clf.fit(X_train, y_train)
            
            # Make predictions
            y_pred = clf.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            
            print(f"\n{name} Classifier Results:")
            print(f"Accuracy: {acc:.2f}")
            print(f"F1 Score: {f1:.2f}")
            print("Confusion Matrix:")
            print(cm)
            
            # Plot confusion matrix
            ax = plt.subplot(gs[idx // 3, idx % 3])
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(f'{name}\nAccuracy: {acc:.2f}, F1: {f1:.2f}')
            plt.colorbar(im, ax=ax)
            tick_marks = np.arange(len(emotions))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(emotions, rotation=45)
            ax.set_yticklabels(emotions)
            ax.set_ylabel('True label')
            ax.set_xlabel('Predicted label')
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
        
        plt.suptitle('Classification Results', fontsize=16, y=0.95)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
        # Step 7: Visualizations
        print("\n🎨 Step 7: Creating visualizations...")
        
        # Sample window visualization
        print("\nCreating sample window visualization...")
        sample_window_idx = 0
        ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
        window_fig = safe_execute(visualize_window, 
                                eeg_windows[sample_window_idx],
                                eda_windows[sample_window_idx],
                                ch_names,
                                sample_window_idx)
        
        # Band power topographies
        print("\nCreating band power topographies...")
        for band in tqdm(['delta', 'theta', 'alpha', 'beta', 'gamma'], desc="Generating topographies"):
            print(f"\nProcessing {band} band...")
            band_fig = safe_execute(plot_band_power_topomap,
                                  eeg_windows[sample_window_idx],
                                  ch_names,
                                  band=band,
                                  fs=fs_eeg,
                                  time_index=sample_window_idx,
                                  total_windows=total_windows)
        
        # Replace all visualization calls with the unified visualization
        print("\n🎨 Creating unified visualization...")
        ch_names = [eeg_channels[i] for i in range(8)]
        unified_fig = create_unified_visualization(
            eeg_data=eeg_reconstructed,
            eda_data=eda_signal,
            ch_names=ch_names,
            fs_eeg=fs_eeg,
            fs_eda=fs_eda,
            window_size=1000  # 4 seconds at 250 Hz
        )
        
        print("\n" + "="*80)
        print("✅ Pipeline completed successfully!")
        print("="*80)
        
        # Show all plots at the end
        plt.show()
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ Pipeline failed!")
        print(f"Error: {str(e)}")
        print("Check pipeline.log for detailed error information")
        print("="*80)
        raise
