# -*- coding: utf-8 -*-
"""eeg_signal_processing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RIhEub8iV_VorGKU_ER5V5Opra44uBV-

# EEG Signal Processing Pipeline

This notebook implements advanced signal processing for EEG data analysis, including:
- Data loading and preprocessing
- Signal filtering (bandpass and notch)
- Spectral analysis using Welch's method
- Band power calculations
- Advanced visualization
"""

# Import required libraries
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mne
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns  # Import seaborn
import matplotlib.patches as patches  # Import for creating shaded regions
from google.colab import drive
import datetime

# --- Mount Google Drive ---
drive.mount('/content/drive')  # Mount your Drive to '/content/drive'

# --- CONFIGURATION ---
data_folder = '/content/drive/MyDrive/data'  # Replace with your folder path
output_folder = '/content/drive/MyDrive/output'  # Replace with your folder path
sampling_rate = 250  # Hz
plot_duration_seconds = 210  # 3.5 minutes
max_samples = plot_duration_seconds * sampling_rate
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Configure plotting
# plt.style.use('seaborn')  # Or plt.style.use('seaborn-whitegrid') etc.
# Optionally, you can use seaborn's set_style() to apply a specific seaborn style

sns.set_style("darkgrid") # Set the style before creating any plots
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
# %matplotlib inline

sampling_window = 10  # Initial value: 10 seconds
samples_per_window = int(sampling_window * sampling_rate)


# EEG frequency bands
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Channel labels
channel_labels = [
    'Fz (ch1)', 'C3 (ch2)', 'Cz (ch3)', 'C4 (ch4)',
    'Pz (ch5)', 'PO7 (ch6)', 'Oz (ch7)', 'PO8 (ch8)',
    'Acc1 (ch9)', 'Acc2 (ch10)', 'Acc3 (ch11)',
    'Gyr1 (ch12)', 'Gyr2 (ch13)', 'Gyr3 (ch14)',
    'Counter (ch15)', 'Valid (ch16)', 'DeltaTime (ch17)', 'Trigger (ch18)'
]

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# --- Signal Processing Functions ---
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a bandpass filter to the signal.

    Args:
        data (array): Input signal
        lowcut (float): Lower cutoff frequency
        highcut (float): Upper cutoff frequency
        fs (float): Sampling frequency
        order (int): Filter order

    Returns:
        array: Filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def notch_filter(data, freq, fs, q=30):
    """Apply a notch filter to remove power line noise.

    Args:
        data (array): Input signal
        freq (float): Notch frequency (e.g., 50Hz or 60Hz)
        fs (float): Sampling frequency
        q (float): Quality factor

    Returns:
        array: Filtered signal
    """
    w0 = freq / (fs/2)
    b, a = signal.iirnotch(w0, q)
    return signal.filtfilt(b, a, data)

def compute_psd(data, fs, nperseg=None):
    """Compute power spectral density using Welch's method.

    Args:
        data (array): Input signal
        fs (float): Sampling frequency
        nperseg (int): Length of each segment

    Returns:
        tuple: Frequencies and PSD values
    """
    if nperseg is None:
        nperseg = min(256, len(data))
    freqs, psd = signal.welch(data, fs, nperseg=nperseg)
    return freqs, psd

def calculate_band_powers(psd, freqs, bands=FREQ_BANDS):
    """Calculate power in specific frequency bands.

    Args:
        psd (array): Power spectral density values
        freqs (array): Frequency points
        bands (dict): Dictionary of frequency bands

    Returns:
        dict: Power values for each band
    """
    powers = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        powers[band_name] = np.trapz(psd[mask], freqs[mask])
    return powers

def process_eeg_data(filepath, channels=None):
    """Process EEG data from a file.

    Args:
        filepath (str): Path to the data file
        channels (list): List of channel indices to process

    Returns:
        tuple: Processed data and time points
    """
    # Read data
    df = pd.read_csv(filepath, header=None)
    df = df.iloc[:max_samples]

    if channels is None:
        channels = list(range(8))  # Process first 8 EEG channels by default

    time = np.arange(len(df)) / sampling_rate
    processed_data = {}

    for channel_index in channels:
        try:
            # Convert to numeric and handle missing values
            data = pd.to_numeric(df[channel_index], errors='coerce').dropna()
            time_cut = time[:len(data)]

            # Apply filters
            filtered = bandpass_filter(data, 0.5, 45, sampling_rate)  # Broadband filter
            filtered = notch_filter(filtered, 50, sampling_rate)  # Remove power line noise

            # Calculate PSD
            freqs, psd = compute_psd(filtered, sampling_rate)

            # Calculate band powers
            powers = calculate_band_powers(psd, freqs)

            processed_data[channel_index] = {
                'raw': data,
                'filtered': filtered,
                'time': time_cut,
                'freqs': freqs,
                'psd': psd,
                'powers': powers
            }

        except Exception as e:
            print(f"Error processing channel {channel_index}: {e}")

    return processed_data

def plot_processed_data(processed_data, filename, output_pdf=None):
    """Create comprehensive visualizations of processed EEG data.

    Args:
        processed_data (dict): Dictionary containing processed data
        filename (str): Name of the data file
        output_pdf (PdfPages): PDF pages object for saving
    """

    n_channels = len(processed_data)
    fig, axes = plt.subplots(6, 1, figsize=(15, 25))  # Increased rows for bands

    # Plot 1: Time domain (Raw vs Filtered)
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

    # Plots 2-5: Time domain for each band
    bands = ['delta', 'theta', 'alpha', 'beta']  # Frequency bands
    for i, band in enumerate(bands):
        ax = axes[i + 1]  # Start from second row
        for ch_idx, ch_data in processed_data.items():
            # Apply bandpass filter for the current band
            band_filtered = bandpass_filter(ch_data['raw'],
                                            FREQ_BANDS[band][0],
                                            FREQ_BANDS[band][1],
                                            sampling_rate)
            ax.plot(ch_data['time'], band_filtered,
                    label=f'{channel_labels[ch_idx]} ({band})')
        ax.set_title(f'{filename} - Time Domain ({band})', fontsize=12)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')


    # Plot 6: Averaged Wave Categories and Ratios
    ax = axes[5]  # New subplot for averages

    bands = ['delta', 'theta', 'alpha', 'beta']
    avg_powers = {band: [] for band in bands}
    alpha_theta_ratios = []
    alpha_beta_ratios = []

    for ch_idx in range(8):  # Channels 1-8
        ch_data = processed_data.get(ch_idx)
        if ch_data:
            for band in bands:
                avg_powers[band].append(ch_data['powers'][band])
            alpha_theta_ratios.append(ch_data['powers']['alpha'] / ch_data['powers']['theta'])
            alpha_beta_ratios.append(ch_data['powers']['alpha'] / ch_data['powers']['beta'])

    # Calculate averages
    for band in bands:
        avg_powers[band] = np.mean(avg_powers[band])
    avg_alpha_theta = np.mean(alpha_theta_ratios)
    avg_alpha_beta = np.mean(alpha_beta_ratios)

    # Plot averaged band powers
    ax.plot(bands, [avg_powers[band] for band in bands], label='Avg Band Power', marker='o')
    ax.plot(bands, [avg_alpha_theta, 0, avg_alpha_beta, 0], label='Avg Ratios', marker='x', linestyle='--')

    ax.set_title(f'{filename} - Averaged Wave Categories and Ratios', fontsize=12)
    ax.set_xlabel('Frequency Bands/Ratios')
    ax.set_ylabel('Power/Ratio')
    ax.grid(True)
    ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')


    plt.tight_layout()

    if output_pdf is not None:
        output_pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_interactive_visualization(processed_data, filename):
    """Create an interactive visualization using Plotly.

    Args:
        processed_data (dict): Dictionary containing processed data
        filename (str): Name of the data file
    """
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[f'{filename} - Time Domain',
                       'Power Spectral Density',
                       'Band Powers by Channel',
                       'Band Power Distribution']
    )

    # Plot 1: Time domain
    for ch_idx, ch_data in processed_data.items():
        fig.add_trace(
            go.Scatter(x=ch_data['time'], y=ch_data['raw'],
                      name=f'{channel_labels[ch_idx]} (Raw)',
                      line=dict(dash='dash'), opacity=0.3),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=ch_data['time'], y=ch_data['filtered'],
                      name=f'{channel_labels[ch_idx]} (Filtered)'),
            row=1, col=1
        )

    # Plot 2: PSD
    for ch_idx, ch_data in processed_data.items():
        fig.add_trace(
            go.Scatter(x=ch_data['freqs'], y=ch_data['psd'],
                      name=channel_labels[ch_idx]),
            row=2, col=1
        )

    # Plot 3: Band powers by channel
    channels = list(processed_data.keys())
    bands = list(FREQ_BANDS.keys())

    for band in bands:
        powers = [processed_data[ch]['powers'][band] for ch in channels]
        fig.add_trace(
            go.Bar(name=band,
                   x=[channel_labels[ch] for ch in channels],
                   y=powers),
            row=3, col=1
        )

    # Plot 4: Band power distribution across all channels
    for ch_idx, ch_data in processed_data.items():
        fig.add_trace(
            go.Scatter(x=list(ch_data['powers'].keys()),
                      y=list(ch_data['powers'].values()),
                      name=channel_labels[ch_idx],
                      mode='lines+markers'),
            row=4, col=1
        )

    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=True,
        legend=dict(x=1.05, y=1)
    )

    # Update axes labels
    fig.update_xaxes(title_text='Time (s)', row=1, col=1)
    fig.update_yaxes(title_text='Amplitude', row=1, col=1)
    fig.update_xaxes(title_text='Frequency (Hz)', row=2, col=1)
    fig.update_yaxes(title_text='Power/Frequency', type='log', row=2, col=1)
    fig.update_xaxes(title_text='Channel', row=3, col=1)
    fig.update_yaxes(title_text='Power', row=3, col=1)
    fig.update_xaxes(title_text='Frequency Band', row=4, col=1)
    fig.update_yaxes(title_text='Power', row=4, col=1)

    fig.show()

# Process files and generate visualizations
print('Starting EEG signal processing...')

# Get list of CSV files
csv_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])

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

print(f'\nPDF report saved to: {output_pdf_path}')