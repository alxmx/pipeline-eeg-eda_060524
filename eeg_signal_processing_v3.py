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

import os
import sys
import subprocess

# Check and install required packages
required_packages = ['numpy', 'pandas', 'scipy', 'matplotlib', 'plotly', 'mne', 'scikit-learn', 'seaborn']
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

# --- CONFIGURATION ---
# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, 'data', 'raw', 'eeg')  # Fixed path to match workspace structure
output_folder = os.path.join(script_dir, 'output')
processed_folder = os.path.join(script_dir, 'processed')
sampling_rate = 250  # Hz
plot_duration_seconds = 210  # 3.5 minutes
max_samples = plot_duration_seconds * sampling_rate
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Configure plotting
sns.set_style("darkgrid")
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)

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

# Make sure required folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)

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

def process_eeg_data(filepath, channels=None):
    """Process EEG data from a file with advanced analysis."""
    # Read data
    df = pd.read_csv(filepath, header=None)
    df = df.iloc[:max_samples]
    
    if channels is None:
        channels = list(range(8))  # Process first 8 EEG channels by default
    
    time = np.arange(len(df)) / sampling_rate
    processed_data = {}
    
    # Channel pairs for asymmetry analysis (left-right)
    asymmetry_pairs = [
        (2, 3),   # C3-C4
        (5, 7),   # PO7-PO8
    ]
    
    for channel_index in channels:
        try:
            # Convert to numeric and handle missing values
            data = pd.to_numeric(df[channel_index], errors='coerce').dropna()
            time_cut = time[:len(data)]
            
            # Apply filters
            filtered = bandpass_filter(data, 0.5, 45, sampling_rate)
            filtered = notch_filter(filtered, 50, sampling_rate)
              # Calculate PSD
            freqs, psd = compute_psd(filtered, sampling_rate)
            
            # Calculate normalized band powers
            powers = calculate_normalized_band_powers(psd, freqs)
            
            # Compute time-frequency representations
            f_stft, t_stft, Zxx = compute_stft(filtered, sampling_rate)
            freq_cwt, coef_cwt = compute_wavelet(filtered, sampling_rate)
            
            processed_data[channel_index] = {
                'raw': data,
                'filtered': filtered,
                'time': time_cut,
                'freqs': freqs,
                'psd': psd,
                'powers': powers,
                'stft': {'freqs': f_stft, 'times': t_stft, 'values': Zxx},
                'cwt': {'freqs': freq_cwt, 'values': coef_cwt}
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
                label=f'{channel_labels[ch_idx]} (Filtered)')
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

def create_interactive_visualization(processed_data, filename):
    """Create an interactive visualization using Plotly."""
    fig = make_subplots(
        rows=len(FREQ_BANDS) + 1, cols=1,
        subplot_titles=[f'{filename} - Time Domain'] + 
                      [f'{band.upper()} Band Power by Channel' for band in FREQ_BANDS]
    )
    
    # Plot time domain
    for ch_idx, ch_data in processed_data.items():
        fig.add_trace(
            go.Scatter(x=ch_data['time'], y=ch_data['filtered'],
                      name=f'{channel_labels[ch_idx]}'),
            row=1, col=1
        )
    
    # Plot band powers
    for i, (band, (low, high)) in enumerate(FREQ_BANDS.items(), start=2):
        powers = [processed_data[ch]['powers'][band] for ch in processed_data]
        fig.add_trace(
            go.Bar(x=[channel_labels[ch] for ch in processed_data],
                  y=powers,
                  name=f'{band.upper()} ({low}-{high} Hz)'),
            row=i, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=300 * (len(FREQ_BANDS) + 1),
        showlegend=True,
        legend=dict(x=1.05, y=1)
    )
    
    fig.show()

def compute_stft(data, fs, nperseg=256, noverlap=None):
    """Compute Short-Time Fourier Transform."""
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Zxx = stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, np.abs(Zxx)

def compute_wavelet(data, fs, wavelet='morlet'):
    """Compute Wavelet Transform using continuous wavelet transform."""
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
        power = np.trapz(psd[mask], freqs[mask])
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

def main():
    print('Starting EEG signal processing...')
    
    # Get list of CSV files
    csv_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])
    
    if not csv_files:
        print(f"No CSV files found in {data_folder}")
        return
    
    # Create PDF report
    output_pdf_path = os.path.join(output_folder, f'eeg_analysis_report_{timestamp}.pdf')
    with PdfPages(output_pdf_path) as pdf:
        for filename in csv_files:
            print(f'\nProcessing {filename}...')
            filepath = os.path.join(data_folder, filename)
            
            # Process data
            processed_data = process_eeg_data(filepath, channels=list(range(8)))
            
            # Create and save visualizations
            plot_processed_data(processed_data, filename, pdf)
            create_interactive_visualization(processed_data, filename)
            
            # Print band powers for each channel
            print(f"\nBand powers for {filename}:")
            for ch_idx, ch_data in processed_data.items():
                print(f"\nChannel {channel_labels[ch_idx]}:")
                for band, power in ch_data['powers'].items():
                    print(f"{band.upper()}: {power:.2f}")
    
    print(f'\nPDF report saved to: {output_pdf_path}')

if __name__ == "__main__":
    main()
