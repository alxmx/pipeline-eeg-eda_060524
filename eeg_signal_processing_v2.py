#!/usr/bin/env python3
"""
EEG Signal Processing Pipeline

This script implements advanced signal processing for EEG data analysis, including:
- Data loading and preprocessing
- Signal filtering (bandpass and notch)
- Spectral analysis using Welch's method
- Band power calculations
- Advanced visualization
"""

# === USAGE INSTRUCTIONS ===
# To run the EEG analysis with the added plots and features:
# 1. Place your EEG CSV files in the 'data/eeg' directory.
# 2. Run this script using: python eeg_signal_processing_v2_modified.py
# 3. Check the 'output/' folder for the generated PDF report.
# 4. Use the interactive Plotly charts for each EEG file.
# 5. The PDF includes alpha/beta ratio plots, band power RMS trends, and markers at key timepoints.

import os
import sys
import subprocess

# Check and install required packages
required_packages = ['numpy', 'pandas', 'scipy', 'matplotlib', 'plotly', 'mne', 'scikit-learn', 'seaborn', 'pywavelets']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mne
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.patches as patches
import datetime
import pywt  # For wavelet analysis
from scipy.signal import stft  # For Short-Time Fourier Transform
from scipy.signal import decimate
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, 'data', 'raw', 'eeg')  
output_folder = os.path.join(script_dir, 'output')
processed_folder = os.path.join(script_dir, 'processed')
sampling_rate = 250  # Hz
plot_duration_seconds = 210  # 3.5 minutes
max_samples = plot_duration_seconds * sampling_rate
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
downsample_factor = 5  # Downsample to 50 Hz (250/5)

# Configure plotting settings
sns.set_style("darkgrid")
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)

# Make sure required folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)

# EEG frequency bands with more detailed beta bands
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta_low': (12, 15),
    'beta_mid': (15, 30),
}

# Channel labels
channel_labels = [
    'Fz (ch1)', 'C3 (ch2)', 'Cz (ch3)', 'C4 (ch4)',
    'Pz (ch5)', 'PO7 (ch6)', 'Oz (ch7)', 'PO8 (ch8)',
    'Acc1 (ch9)', 'Acc2 (ch10)', 'Acc3 (ch11)',
    'Gyr1 (ch12)', 'Gyr2 (ch13)', 'Gyr3 (ch14)',
    'Counter (ch15)', 'Valid (ch16)', 'DeltaTime (ch17)', 'Trigger (ch18)'
]

# Constants
NUM_CHANNELS = 8  # First 8 EEG channels

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a bandpass filter to the signal."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def notch_filter(data, freq, fs, q=30):
    """Apply a notch filter to remove power line noise."""
    w0 = freq / (fs/2)
    b, a = signal.iirnotch(w0, q)
    return signal.filtfilt(b, a, data)

def compute_psd(data, fs, nperseg=None):
    """Compute power spectral density using Welch's method."""
    if nperseg is None:
        nperseg = min(256, len(data))
    freqs, psd = signal.welch(data, fs, nperseg=nperseg)
    return freqs, psd

def calculate_band_powers(psd, freqs, bands=FREQ_BANDS):
    """Calculate power in specific frequency bands."""
    powers = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        powers[band_name] = np.trapz(psd[mask], freqs[mask])
    return powers

def process_eeg_data(filepath, channels=None, buffer_size=2.0, buffer_overlap=0.5, downsample=True):
    """Process EEG data from a file with advanced buffer-based analysis.
    
    Args:
        filepath (str): Path to the data file
        channels (list): List of channel indices to process
        buffer_size (float): Size of buffer in seconds
        buffer_overlap (float): Overlap between buffers in seconds
        downsample (bool): Whether to downsample the data
        
    Returns:
        dict: Processed data including time-varying bandpowers and ratios
    """
    # Read data
    df = pd.read_csv(filepath, header=None)
    df = df.iloc[:max_samples]
    
    if channels is None:
        channels = list(range(NUM_CHANNELS))
    
    time = np.arange(len(df)) / sampling_rate
    processed_data = {}
    
    current_fs = sampling_rate  # Keep track of current sampling rate
    
    for channel_index in channels:
        try:
            # Convert to numeric and handle missing values
            data = pd.to_numeric(df[channel_index], errors='coerce').dropna()
            
            # Apply filters
            filtered = filter_eeg(data, current_fs)
            
            # Downsample if requested
            if downsample:
                filtered, new_fs = downsample_with_antialiasing(filtered, current_fs, downsample_factor)
                current_fs = new_fs
                time_cut = np.arange(len(filtered)) / current_fs
            else:
                time_cut = time[:len(filtered)]
            
            # Compute PSD for whole signal
            freqs, psd = compute_psd(filtered, current_fs)
            
            # Calculate overall band powers
            band_powers = calculate_normalized_band_powers(psd, freqs)
            
            # Calculate bandpower ratios
            power_ratios = calculate_bandpower_ratios(band_powers)
            
            # Calculate time-varying bandpowers using buffer
            buffer_times, buffer_powers, buffer_ratios = calculate_bandpower_with_buffer(
                filtered, current_fs, buffer_size, buffer_overlap
            )
            
            # Store all data for this channel
            processed_data[channel_index] = {
                'raw': data,
                'filtered': filtered,
                'time': time_cut,
                'freqs': freqs,
                'psd': psd,
                'powers': band_powers,
                'power_ratios': power_ratios,
                'buffer_data': {
                    'times': buffer_times,
                    'powers': buffer_powers,
                    'ratios': buffer_ratios
                },
                'sampling_rate': current_fs  # Store the actual sampling rate used
            }
            
        except Exception as e:
            print(f"Error processing channel {channel_index}: {e}")
            
    return processed_data

def plot_processed_data(processed_data, filename, output_pdf=None):
    """Create comprehensive visualizations of processed EEG data."""
    n_channels = len(processed_data)
    fig, axes = plt.subplots(8, 1, figsize=(15, 35))  # Added 2 more subplots

    # Plot 1: Time domain
    ax = axes[0]
    for ch_idx, ch_data in processed_data.items():
        ax.plot(ch_data['time'], ch_data['raw'], '--', alpha=0.3,
                label=f'{channel_labels[ch_idx]} (Raw)')
        ax.plot(ch_data['time'], ch_data['filtered'],
                label=f'{channel_labels[ch_idx]} (Filtered)')    # Add time markers for key points
    add_time_markers(ax, [20, 50, 80, 110, 140, 170, 200])
    
    ax.set_title(f'{filename} - Time Domain', fontsize=12)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plots 2-5: Individual bands
    bands = list(FREQ_BANDS.keys())
    for i, band in enumerate(bands):
        ax = axes[i + 1]
        for ch_idx, ch_data in processed_data.items():
            # Apply bandpass filter for the current band
            band_filtered = bandpass_filter(ch_data['raw'],
                                         FREQ_BANDS[band][0],
                                         FREQ_BANDS[band][1],
                                         sampling_rate)
            ax.plot(ch_data['time'], band_filtered,
                   label=f'{channel_labels[ch_idx]} ({band})')
        ax.set_title(f'{filename} - {band.upper()} band', fontsize=12)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    
    if output_pdf is not None:
        output_pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_interactive_visualization(raw_data, file_processed_data, baseline_data=None):
    """Create an interactive visualization including bandpower ratios.
    
    Args:
        raw_data (dict): Raw EEG data
        file_processed_data (dict): Processed data for current file
        baseline_data (dict): Optional baseline data for comparison
    """
    fig = make_subplots(
        rows=6, cols=1,
        subplot_titles=[
            'Time Domain',
            'Power Spectral Density',
            'Band Powers by Channel',
            'Bandpower Ratios by Channel',
            'Time-varying Bandpowers',
            'Time-varying Ratios'
        ],
        vertical_spacing=0.08,
        row_heights=[0.2, 0.2, 0.15, 0.15, 0.15, 0.15]
    )
    
    # Plot 1: Time domain
    for ch_idx, ch_data in file_processed_data.items():
        fig.add_trace(
            go.Scatter(
                x=ch_data['time'],
                y=ch_data['filtered'],
                name=f'{channel_labels[ch_idx]}',
                line=dict(dash='solid')
            ),
            row=1, col=1
        )
    
    # Plot 2: PSD
    for ch_idx, ch_data in file_processed_data.items():
        fig.add_trace(
            go.Scatter(
                x=ch_data['freqs'],
                y=ch_data['psd'],
                name=channel_labels[ch_idx]
            ),
            row=2, col=1
        )
    
    # Plot 3: Band powers by channel
    channels = list(file_processed_data.keys())
    for band in FREQ_BANDS:
        powers = [file_processed_data[ch]['powers'][band] for ch in channels]
        fig.add_trace(
            go.Bar(
                name=band,
                x=[channel_labels[ch] for ch in channels],
                y=powers
            ),
            row=3, col=1
        )
    
    # Plot 4: Power ratios by channel
    ratio_names = [
        'power_ratio_index',
        'delta_alpha_ratio',
        'theta_alpha_ratio',
        'theta_beta_ratio',
        'theta_beta_alpha_ratio',
        'engagement_index'
    ]
    
    for ch_idx in channels:
        ratios = [file_processed_data[ch_idx]['power_ratios'][ratio] for ratio in ratio_names]
        fig.add_trace(
            go.Bar(
                name=channel_labels[ch_idx],
                x=ratio_names,
                y=ratios
            ),
            row=4, col=1
        )
    
    # Plot 5: Time-varying bandpowers
    ref_channel = channels[0]  # Use first channel as reference
    buffer_times = file_processed_data[ref_channel]['buffer_data']['times']
    
    for band in FREQ_BANDS:
        fig.add_trace(
            go.Scatter(
                x=buffer_times,
                y=file_processed_data[ref_channel]['buffer_data']['powers'][band],
                name=f'{band} power',
                mode='lines'
            ),
            row=5, col=1
        )
    
    # Plot 6: Time-varying ratios
    for ratio in ratio_names:
        fig.add_trace(
            go.Scatter(
                x=buffer_times,
                y=file_processed_data[ref_channel]['buffer_data']['ratios'][ratio],
                name=ratio,
                mode='lines'
            ),
            row=6, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=1800,
        showlegend=True,
        legend=dict(x=1.1, y=1),
        title_text="EEG Analysis with Bandpower Ratios"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Power/Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Channel", row=3, col=1)
    fig.update_yaxes(title_text="Power", row=3, col=1)
    fig.update_xaxes(title_text="Ratio Type", row=4, col=1)
    fig.update_yaxes(title_text="Ratio Value", row=4, col=1)
    fig.update_xaxes(title_text="Time (s)", row=5, col=1)
    fig.update_yaxes(title_text="Power", row=5, col=1)
    fig.update_xaxes(title_text="Time (s)", row=6, col=1)
    fig.update_yaxes(title_text="Ratio Value", row=6, col=1)
    
    return fig

def compute_stft(data, fs, nperseg=256, noverlap=None):
    """Compute Short-Time Fourier Transform."""
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Zxx = stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, np.abs(Zxx)

def compute_wavelet(data, fs, wavelet='cmor1.5-1.0'):
    """Compute Wavelet Transform using continuous wavelet transform."""
    # Use Complex Morlet wavelet (cmor) which is better suited for EEG analysis
    scales = np.arange(1, 128)
    frequencies = pywt.scale2frequency(wavelet, scales) * fs
    coef, freqs = pywt.cwt(data, scales, wavelet)
    return frequencies, np.abs(coef)

def calculate_normalized_band_powers(psd, freqs, bands=FREQ_BANDS):
    """Calculate normalized power in specific frequency bands (0-1 scale)."""
    powers = {}
    total_power = 0
    
    # Calculate absolute powers first
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        power = np.trapezoid(psd[mask], freqs[mask])
        powers[band_name] = power
        total_power += power
    
    # Normalize to 0-1 scale
    normalized_powers = {band: power/total_power for band, power in powers.items()}
    return normalized_powers

def calculate_alpha_asymmetry(left_data, right_data, fs):
    """Calculate alpha asymmetry between left and right channels."""
    # Calculate alpha power for both channels
    _, left_psd = compute_psd(left_data, fs)
    _, right_psd = compute_psd(right_data, fs)
    
    # Get alpha band indices
    freqs = np.linspace(0, fs/2, len(left_psd))
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    
    # Calculate alpha power
    left_alpha = np.trapezoid(left_psd[alpha_mask], freqs[alpha_mask])
    right_alpha = np.trapezoid(right_psd[alpha_mask], freqs[alpha_mask])
    
    # Calculate asymmetry score (log of right/left ratio)
    # Add small constant to avoid log(0)
    epsilon = 1e-6
    asymmetry = np.log(right_alpha + epsilon) - np.log(left_alpha + epsilon)
    return asymmetry

def plot_time_frequency_analysis(data, fs, output_pdf=None):
    """Create comprehensive time-frequency analysis plots."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))
    
    # Plot 1: Original signal
    axes[0].plot(np.arange(len(data))/fs, data)
    axes[0].set_title('Original Signal')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    
    # Plot 2: STFT Spectrogram
    f, t, Zxx = compute_stft(data, fs)
    axes[1].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    axes[1].set_title('STFT Spectrogram')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    
    # Plot 3: Wavelet Scalogram
    frequencies, coef = compute_wavelet(data, fs)
    axes[2].pcolormesh(np.arange(len(data))/fs, frequencies, np.abs(coef), 
                      shading='gouraud')
    axes[2].set_title('Wavelet Scalogram')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    if output_pdf is not None:
        output_pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def plot_alpha_beta_ratio_over_time(data, fs, window_sec=5, step_sec=1):
    """Calculate and plot alpha/beta ratio over time using sliding window."""
    window_size = int(window_sec * fs)
    step_size = int(step_sec * fs)
    times = []
    ratios = []
    
    for start in range(0, len(data) - window_size, step_size):
        end = start + window_size
        segment = data[start:end]
        freqs, psd = compute_psd(segment, fs)
        
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        beta_mask = (freqs >= 13) & (freqs <= 30)
        
        alpha_power = np.trapezoid(psd[alpha_mask], freqs[alpha_mask])
        beta_power = np.trapezoid(psd[beta_mask], freqs[beta_mask])
        
        alpha_beta_ratio = alpha_power / (beta_power + 1e-6)
        ratios.append(alpha_beta_ratio)
        times.append(start / fs)
    
    return times, ratios

def plot_band_power_trends(processed_data, fs, output_pdf=None):
    """Plot RMS envelope of each frequency band over time."""
    fig, ax = plt.subplots(figsize=(15, 5))
    bands = list(FREQ_BANDS.keys())
    
    for band in bands:
        for ch_idx, ch_data in processed_data.items():
            filtered = bandpass_filter(ch_data['filtered'], 
                                    FREQ_BANDS[band][0], 
                                    FREQ_BANDS[band][1], 
                                    fs)
            window_size = int(fs * 2)
            rms = np.sqrt(np.convolve(filtered**2, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid'))
            t = ch_data['time'][:len(rms)]
            ax.plot(t, rms, label=f'{channel_labels[ch_idx]}-{band}')
    
    ax.set_title('Band Power Trends Over Time (RMS Envelope)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RMS Amplitude')
    ax.grid(True)
    ax.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_pdf:
        output_pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def add_time_markers(ax, markers_sec):
    """Add vertical time markers to plot."""
    for sec in markers_sec:
        ax.axvline(x=sec, color='red', linestyle='--', alpha=0.5)
        ax.text(sec, ax.get_ylim()[1]*0.95, f'{sec}s', 
                color='red', fontsize=8, alpha=0.7)

def load_eeg_data(filepath, do_downsample=True):
    """Load EEG data from a file with optional downsampling.
    
    Args:
        filepath (str): Path to the data file
        do_downsample (bool): Whether to downsample the data
        
    Returns:
        DataFrame: Loaded (and optionally downsampled) EEG data
    """
    df = pd.read_csv(filepath, header=None, low_memory=False)
    df = df.iloc[:max_samples]  # Limit to plot duration
    
    if do_downsample:
        # Apply anti-aliasing and downsample for each channel
        downsampled_data = {}
        current_fs = sampling_rate
        
        # Process EEG channels
        for ch in range(NUM_CHANNELS):
            # Convert to numeric and handle missing values using ffill()
            data = pd.to_numeric(df[ch], errors='coerce').ffill().fillna(0)  # Use 0 for any remaining NaN
            filtered_data = apply_antialiasing_filter(data, current_fs, 0.8 * (current_fs/(2*downsample_factor)))
            downsampled_data[ch] = signal.decimate(filtered_data, downsample_factor, n=None, ftype='iir', zero_phase=True)
        
        # Create new dataframe with downsampled data
        new_df = pd.DataFrame(downsampled_data)
        
        # Add non-EEG channels without downsampling (metadata columns)
        for col in range(NUM_CHANNELS, df.shape[1]):
            new_df[col] = df[col].iloc[::downsample_factor].reset_index(drop=True)
        
        return new_df
    
    return df

def filter_eeg(data, fs):
    """Apply bandpass and notch filters to EEG data."""
    # Initial high pass filter at 0.5 Hz to remove DC
    data = bandpass_filter(data, 0.5, 30, fs, order=4)
    # Apply 50 Hz notch filter for power line noise
    data = notch_filter(data, 50, fs)
    return data

def normalize_across_files(all_processed_data):
    """Normalize band powers across all files."""
    all_powers = {band: [] for band in FREQ_BANDS}
    
    # Collect all powers
    for file_data in all_processed_data.values():
        for ch_data in file_data.values():
            for band, power in ch_data['powers'].items():
                all_powers[band].append(power)
    
    # Calculate mean and std for each band
    band_stats = {}
    for band, powers in all_powers.items():
        powers = np.array(powers)
        band_stats[band] = {
            'mean': np.mean(powers),
            'std': np.std(powers),
            'min': np.min(powers),
            'max': np.max(powers)
        }
    
    return band_stats

def create_comparative_visualization(all_processed_data, baseline_file, band_stats):
    """Create interactive comparative visualization including bandpower ratios.
    
    Args:
        all_processed_data (dict): Data from all processed files
        baseline_file (str): Name of baseline file
        band_stats (dict): Statistics for normalization
    """
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=[
            'Time Domain Comparison',
            'Normalized Band Powers',
            'Average Power Ratios',
            'Time-varying Power Ratios',
            'Engagement Index Comparison'
        ],
        vertical_spacing=0.08,
        row_heights=[0.25, 0.2, 0.2, 0.2, 0.15]
    )
    
    # Create dropdown menus for file and channel selection
    file_names = list(all_processed_data.keys())
    channels = list(range(NUM_CHANNELS))
    ratio_types = [
        'power_ratio_index',
        'delta_alpha_ratio',
        'theta_alpha_ratio',
        'theta_beta_ratio',
        'theta_beta_alpha_ratio',
        'engagement_index'
    ]
    
    # Plot 1: Time domain comparison
    for file_name in file_names:
        file_data = all_processed_data[file_name]
        for ch_idx in channels:
            ch_data = file_data[ch_idx]
            fig.add_trace(
                go.Scatter(
                    x=ch_data['time'],
                    y=ch_data['filtered'],
                    name=f'{file_name}-{channel_labels[ch_idx]}',
                    visible=(file_name == baseline_file)
                ),
                row=1, col=1
            )
    
    # Plot 2: Normalized band powers comparison
    for file_name in file_names:
        file_data = all_processed_data[file_name]
        for ch_idx in channels:
            powers = []
            for band in FREQ_BANDS:
                norm_power = (file_data[ch_idx]['powers'][band] - band_stats[band]['mean']) / band_stats[band]['std']
                powers.append(norm_power)
            
            fig.add_trace(
                go.Bar(
                    name=f'{file_name}-{channel_labels[ch_idx]}',
                    x=list(FREQ_BANDS.keys()),
                    y=powers
                ),
                row=2, col=1
            )
    
    # Plot 3: Average power ratios across channels
    for file_name in file_names:
        file_data = all_processed_data[file_name]
        avg_ratios = {ratio: [] for ratio in ratio_types}
        
        # Calculate average ratios across channels
        for ch_idx in channels:
            for ratio in ratio_types:
                avg_ratios[ratio].append(file_data[ch_idx]['power_ratios'][ratio])
        
        # Plot average ratios
        fig.add_trace(
            go.Bar(
                name=file_name,
                x=ratio_types,
                y=[np.mean(avg_ratios[ratio]) for ratio in ratio_types]
            ),
            row=3, col=1
        )
    
    # Plot 4: Time-varying power ratios (using first channel)
    ref_channel = channels[0]
    for file_name in file_names:
        file_data = all_processed_data[file_name]
        buffer_times = file_data[ref_channel]['buffer_data']['times']
        
        for ratio in ratio_types:
            fig.add_trace(
                go.Scatter(
                    x=buffer_times,
                    y=file_data[ref_channel]['buffer_data']['ratios'][ratio],
                    name=f'{file_name}-{ratio}',
                    visible='legendonly'  # Hide by default to reduce clutter
                ),
                row=4, col=1
            )
    
    # Plot 5: Engagement Index comparison
    for file_name in file_names:
        file_data = all_processed_data[file_name]
        engagement_values = []
        
        for ch_idx in channels:
            engagement_values.append(
                file_data[ch_idx]['power_ratios']['engagement_index']
            )
        
        fig.add_trace(
            go.Box(
                name=file_name,
                y=engagement_values,
                boxpoints='all',
                pointpos=-1.8
            ),
            row=5, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=2000,
        showlegend=True,
        legend=dict(x=1.1, y=1),
        title_text="Comparative Analysis with Power Ratios"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Frequency Band", row=2, col=1)
    fig.update_yaxes(title_text="Normalized Power", row=2, col=1)
    fig.update_xaxes(title_text="Ratio Type", row=3, col=1)
    fig.update_yaxes(title_text="Average Ratio Value", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_yaxes(title_text="Ratio Value", row=4, col=1)
    fig.update_xaxes(title_text="File", row=5, col=1)
    fig.update_yaxes(title_text="Engagement Index", row=5, col=1)
    
    return fig

def create_alpha_asymmetry_comparison(all_processed_data):
    """Create comparative visualization of alpha asymmetry across all files."""
    fig = go.Figure()
    
    # Calculate alpha asymmetry for each file
    channel_pairs = [(2, 3), (5, 7)]  # C3-C4 and PO7-PO8
    
    for file_name, file_data in all_processed_data.items():
        asymmetry_values = []
        pair_labels = []
        
        for left_ch, right_ch in channel_pairs:
            asymmetry = calculate_alpha_asymmetry(
                file_data[left_ch]['filtered'],
                file_data[right_ch]['filtered'],
                sampling_rate
            )
            asymmetry_values.append(asymmetry)
            pair_labels.append(f"{channel_labels[left_ch]}-{channel_labels[right_ch]}")
        
        fig.add_trace(go.Bar(
            name=file_name,
            x=pair_labels,
            y=asymmetry_values
        ))
    
    fig.update_layout(
        title="Alpha Asymmetry Comparison Across Files",
        xaxis_title="Channel Pairs",
        yaxis_title="Asymmetry Score (log right/left ratio)",
        barmode='group',
        height=600
    )
    
    return fig

def calculate_bandpower_ratios(powers):
    """Calculate various bandpower ratios from frequency bands.
    
    Args:
        powers (dict): Dictionary containing band powers
        
    Returns:
        dict: Dictionary containing calculated ratios
    """
    ratios = {}
    
    # Power Ratio Index = (delta + theta) / (alpha + beta)
    ratios['power_ratio_index'] = (powers['delta'] + powers['theta']) / (powers['alpha'] + powers['beta_mid'])
    
    # Delta/Alpha Ratio
    ratios['delta_alpha_ratio'] = powers['delta'] / powers['alpha']
    
    # Theta/Alpha Ratio
    ratios['theta_alpha_ratio'] = powers['theta'] / powers['alpha']
    
    # Theta/Beta Ratio
    ratios['theta_beta_ratio'] = powers['theta'] / powers['beta_mid']
    
    # Theta/(Alpha + Beta) Ratio
    ratios['theta_beta_alpha_ratio'] = powers['theta'] / (powers['alpha'] + powers['beta_mid'])
    
    # Engagement Index = beta/(theta + alpha)
    ratios['engagement_index'] = powers['beta_mid'] / (powers['theta'] + powers['alpha'])
    
    return ratios

def calculate_bandpower_with_buffer(data, fs, buffer_size=2.0, buffer_overlap=0.5, bands=FREQ_BANDS):
    """Calculate bandpower using sliding buffer.
    
    Args:
        data (array): Input signal
        fs (float): Sampling frequency
        buffer_size (float): Size of buffer in seconds
        buffer_overlap (float): Overlap between buffers in seconds
        bands (dict): Frequency bands
        
    Returns:
        tuple: Time points and bandpowers over time
    """
    buffer_samples = int(buffer_size * fs)
    overlap_samples = int(buffer_overlap * fs)
    step_size = buffer_samples - overlap_samples
    
    # Calculate number of buffers
    n_buffers = (len(data) - overlap_samples) // step_size
    
    # Initialize output arrays
    times = []
    bandpowers = {band: [] for band in bands.keys()}
    bandpower_ratios = {
        'power_ratio_index': [], 
        'delta_alpha_ratio': [],
        'theta_alpha_ratio': [],
        'theta_beta_ratio': [],
        'theta_beta_alpha_ratio': [],
        'engagement_index': []
    }
    
    for i in range(n_buffers):
        start = i * step_size
        end = start + buffer_samples
        
        # Get data segment
        segment = data[start:end]
        
        # Calculate PSD for this segment
        freqs, psd = compute_psd(segment, fs)
        
        # Calculate band powers
        powers = calculate_normalized_band_powers(psd, freqs)
        
        # Calculate ratios
        ratios = calculate_bandpower_ratios(powers)
        
        # Store results
        times.append((start + end) / (2 * fs))  # Center time of buffer
        for band, power in powers.items():
            bandpowers[band].append(power)
        for ratio_name, ratio in ratios.items():
            bandpower_ratios[ratio_name].append(ratio)
    
    return np.array(times), bandpowers, bandpower_ratios

def apply_antialiasing_filter(data, fs, cutoff_freq):
    """Apply anti-aliasing filter before downsampling.
    
    Args:
        data (array): Input signal
        fs (float): Original sampling frequency
        cutoff_freq (float): Cutoff frequency for the filter
        
    Returns:
        array: Filtered signal
    """
    nyquist = fs / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(8, normalized_cutoff, btype='low')
    return signal.filtfilt(b, a, data)

def downsample_with_antialiasing(data, fs, downsample_factor):
    """Downsample signal with proper anti-aliasing filter.
    
    Args:
        data (array): Input signal
        fs (float): Original sampling frequency
        downsample_factor (int): Factor by which to downsample
        
    Returns:
        tuple: (downsampleed_data, new_fs)
    """
    # Apply anti-aliasing filter before decimation
    # Cutoff frequency should be slightly less than the Nyquist frequency of the target sampling rate
    new_fs = fs / downsample_factor
    cutoff_freq = 0.8 * (new_fs / 2)  # Use 80% of new Nyquist frequency
    
    filtered_data = apply_antialiasing_filter(data, fs, cutoff_freq)
    downsampled_data = signal.decimate(filtered_data, downsample_factor, n=None, ftype='iir', zero_phase=True)
    
    return downsampled_data, new_fs

def main():
    input_dir = os.path.join(os.getcwd(), 'data', 'raw', 'eeg')
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    eeg_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
    
    if not eeg_files:
        print(f"No EEG files found in {input_dir}")
        return
        
    baseline_file = eeg_files[1] if len(eeg_files) > 1 else eeg_files[0]  # Use second file as baseline
    
    # Dictionary to store all processed data
    all_processed_data = {}
    
    # Split files for ML
    train_files = eeg_files[:2]
    test_files = eeg_files[2:7]
    validation_files = eeg_files[7:]
    
    # Process all files
    for eeg_file in eeg_files:
        file_path = os.path.join(input_dir, eeg_file)
        try:
            print(f"\nProcessing {eeg_file}...")
              # Load and preprocess data with anti-aliasing and downsampling
            raw_data = load_eeg_data(file_path, do_downsample=True)
            current_fs = sampling_rate / downsample_factor  # Update sampling rate after downsampling
            file_processed_data = {}
            
            for channel in range(NUM_CHANNELS):
                # Filter data
                filtered = filter_eeg(raw_data[channel], sampling_rate)
                
                # Compute PSD
                freqs, psd = compute_psd(filtered, sampling_rate)
                
                # Calculate normalized band powers
                band_powers = calculate_normalized_band_powers(psd, freqs)
                  # Calculate power ratios
                power_ratios = calculate_bandpower_ratios(band_powers)
                
                # Calculate time-varying bandpowers and ratios
                buffer_times, buffer_powers, buffer_ratios = calculate_bandpower_with_buffer(
                    filtered, current_fs, buffer_size=2.0, buffer_overlap=0.5
                )
                
                # Store all data for this channel
                file_processed_data[channel] = {
                    'raw': raw_data[channel],
                    'filtered': filtered,
                    'freqs': freqs,
                    'psd': psd,
                    'powers': band_powers,
                    'power_ratios': power_ratios,
                    'time': np.arange(len(filtered)) / current_fs,
                    'buffer_data': {
                        'times': buffer_times,
                        'powers': buffer_powers,
                        'ratios': buffer_ratios
                    }
                }
            
            all_processed_data[eeg_file] = file_processed_data
            
            # Create individual file visualization
            plot_file = os.path.join(output_dir, f"{os.path.splitext(eeg_file)[0]}_analysis.html")
            fig = create_interactive_visualization(raw_data, file_processed_data, all_processed_data.get(baseline_file))
            fig.write_html(plot_file)
            print(f"Individual analysis plots saved to {plot_file}")
            
        except Exception as e:
            print(f"Error processing {eeg_file}: {str(e)}")
            continue
    
    try:
        # Calculate normalization stats across all files
        band_stats = normalize_across_files(all_processed_data)
        
        # Create comparative visualizations
        comp_plot_file = os.path.join(output_dir, "comparative_analysis.html")
        comp_fig = create_comparative_visualization(all_processed_data, baseline_file, band_stats)
        comp_fig.write_html(comp_plot_file)
        print(f"\nComparative analysis plots saved to {comp_plot_file}")
        
        # Create alpha asymmetry comparison
        asym_plot_file = os.path.join(output_dir, "alpha_asymmetry_comparison.html")
        asym_fig = create_alpha_asymmetry_comparison(all_processed_data)
        asym_fig.write_html(asym_plot_file)
        print(f"Alpha asymmetry comparison saved to {asym_plot_file}")
        
    except Exception as e:
        print(f"Error creating comparative visualizations: {str(e)}")

if __name__ == "__main__":
    main()

def get_channel_positions():
    """Get 2D positions for EEG channels in standard 10-20 layout."""
    # Approximate 2D positions for the channels we have
    positions = {
        'Fz': (0.0, 0.7),      # Front midline
        'C3': (-0.5, 0.0),     # Left central
        'Cz': (0.0, 0.0),      # Central midline
        'C4': (0.5, 0.0),      # Right central
        'Pz': (0.0, -0.5),     # Parietal midline
        'PO7': (-0.7, -0.7),   # Left parieto-occipital
        'Oz': (0.0, -0.7),     # Occipital midline
        'PO8': (0.7, -0.7),    # Right parieto-occipital
    }
    return positions

def calculate_valence_arousal(processed_data, sampling_rate):
    """Calculate valence and arousal metrics from EEG data over time.
    
    Valence: Based on alpha asymmetry between left/right hemispheres
    Arousal: Based on beta/alpha ratio across channels
    
    Returns:
        tuple: (valence_scores, arousal_scores, times)
    """
    # Get relevant channel pairs for valence calculation
    left_channels = [1, 5]  # C3, PO7 
    right_channels = [3, 7]  # C4, PO8
    
    # Initialize arrays for results
    times = []
    valence_scores = []
    arousal_scores = []
    
    # Get buffer size info from first channel
    ref_channel = processed_data[0]
    buffer_times = ref_channel['buffer_data']['times']
    
    # Calculate metrics for each time window
    for t_idx, _ in enumerate(buffer_times):
        # Calculate alpha asymmetry for valence
        left_alpha = 0
        right_alpha = 0
        total_beta = 0
        total_alpha = 0
        
        # Calculate hemispheric alpha power
        for left_ch, right_ch in zip(left_channels, right_channels):
            left_alpha += processed_data[left_ch]['buffer_data']['powers']['alpha'][t_idx]
            right_alpha += processed_data[right_ch]['buffer_data']['powers']['alpha'][t_idx]
        
        # Calculate overall arousal from beta/alpha ratio
        for ch in range(8):  # All EEG channels
            total_beta += processed_data[ch]['buffer_data']['powers']['beta_mid'][t_idx]
            total_alpha += processed_data[ch]['buffer_data']['powers']['alpha'][t_idx]
        
        # Calculate metrics
        valence = np.log(right_alpha + 1e-6) - np.log(left_alpha + 1e-6)  # Alpha asymmetry
        arousal = np.log(total_beta / (total_alpha + 1e-6))  # Beta/alpha ratio
        
        times.append(buffer_times[t_idx])
        valence_scores.append(valence)
        arousal_scores.append(arousal)
    
    return np.array(valence_scores), np.array(arousal_scores), np.array(times)

def plot_valence_arousal_over_time(valence, arousal, times):
    """Plot valence and arousal changes over time."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Valence over Time', 'Arousal over Time', 'Valence-Arousal Space'),
        vertical_spacing=0.12
    )
    
    # Plot valence over time
    fig.add_trace(
        go.Scatter(x=times, y=valence, mode='lines', name='Valence'),
        row=1, col=1
    )
    
    # Plot arousal over time
    fig.add_trace(
        go.Scatter(x=times, y=arousal, mode='lines', name='Arousal'),
        row=2, col=1
    )
    
    # Plot 2D valence-arousal space with time color gradient
    fig.add_trace(
        go.Scatter(
            x=valence, 
            y=arousal,
            mode='markers',
            marker=dict(
                size=8,
                color=times,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Time (s)')
            ),
            name='V-A State'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        title_text="Valence-Arousal Analysis Over Time"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Valence", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Arousal", row=2, col=1)
    fig.update_xaxes(title_text="Valence", row=3, col=1)
    fig.update_yaxes(title_text="Arousal", row=3, col=1)
    
    return fig

def plot_alpha_asymmetry_over_time(processed_data):
    """Plot alpha asymmetry changes over time for different channel pairs."""
    # Channel pairs for asymmetry calculation
    channel_pairs = [
        (1, 3, "C3-C4"),  # Central L-R
        (5, 7, "PO7-PO8")  # Parieto-occipital L-R
    ]
    
    # Get time points from first channel
    times = processed_data[0]['buffer_data']['times']
    
    fig = go.Figure()
    
    for left_ch, right_ch, pair_name in channel_pairs:
        # Calculate asymmetry for each time point
        asymmetry = []
        for t_idx, _ in enumerate(times):
            left_alpha = processed_data[left_ch]['buffer_data']['powers']['alpha'][t_idx]
            right_alpha = processed_data[right_ch]['buffer_data']['powers']['alpha'][t_idx]
            asym = np.log(right_alpha + 1e-6) - np.log(left_alpha + 1e-6)
            asymmetry.append(asym)
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=asymmetry,
                mode='lines',
                name=f'Asymmetry {pair_name}'
            )
        )
    
    fig.update_layout(
        title="Alpha Asymmetry Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Asymmetry Score (log right/left ratio)",
        height=600,
        showlegend=True
    )
    
    return fig
