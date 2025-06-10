"""
EEG Signal Processing Pipeline

This script implements advanced signal processing for EEG data analysis, including:
- Data loading and preprocessing
- Signal filtering (bandpass and notch)
- Spectral analysis using Welch's method
- Band power calculations
- SVM classifcation of emotions for a 2 second window
- Advanced visualization
"""
"""
Electrode Key:

    Frontal: Fz (ch1)
    Left Central: C3 (ch2)
    Central Midline: Cz (ch3)
    Right Central: C4 (ch4)
    Parietal Midline: Pz (ch5)
    Left Parietal-Occipital: PO7 (ch6)
    Occipital: Oz (ch7)
    Right Parietal-Occipital: PO8 (ch8)
    Accelerometers (ch9–11): Not relevant for EEG analysis.

1. Excited (High Arousal, High Valence)

    EEG: ↑ Beta (frontal/central), ↓ Alpha (posterior)
    Electrodes:
        ↑ Beta: Fz (ch1) (frontal midline), C3 (ch2), Cz (ch3), C4 (ch4) (central regions).
            Why: Beta reflects active engagement; frontal/central regions drive alertness and motor planning.
        ↓ Alpha: PO7 (ch6), Oz (ch7), PO8 (ch8) (posterior).
            Why: Alpha decreases in posterior areas when relaxed focus is disrupted by excitement.

2. Angry (High Arousal, Low Valence)

    EEG: ↑ Right Frontal Beta
    Electrodes:
        ↑ Beta: C4 (ch4) (right central).
            Why: The right central region (C4) approximates right frontal activity due to proximity and shared motor/emotional networks. Beta here aligns with "fight response" motor tension.
        Also possible: Fz (ch1) (midline frontal Beta due to generalized arousal).

3. Sad (Low Arousal, Low Valence)

    EEG: ↑ Right Alpha (posterior)
    Electrodes:
        ↑ Alpha: PO8 (ch8) (right parietal-occipital).
            Why: Posterior Alpha peaks during passive states. Right hemisphere dominance (PO8) links to withdrawal emotions like sadness.

4. Calm (Low Arousal, High Valence)

    EEG: ↑ Left Alpha (posterior)
    Electrodes:
        ↑ Alpha: PO7 (ch6) (left parietal-occipital).
            Why: Left posterior Alpha (PO7) reflects relaxed, positive states. The left hemisphere (PO7) biases toward approach/positivity, even at rest.

Summary Table:
Emotion	Arousal/Valence	EEG Pattern	Relevant Electrodes
Excited	High, High	↑ Beta, ↓ Alpha	Fz, C3, Cz, C4; PO7, Oz, PO8
Angry	High, Low	↑ Right Beta	C4 (right central)
Sad	Low, Low	↑ Right Alpha (posterior)	PO8 (right parietal-occipital)
Calm	Low, High	↑ Left Alpha (posterior)	PO7 (left parietal-occipital)
Key Notes:

    Frontal vs. Central:
        The absence of lateral frontal electrodes (e.g., F3/F4) means nearby central electrodes (C3/C4) may proxy for frontal activity due to overlapping networks.
        Fz (midline) captures generalized frontal arousal.

    Posterior Dominance for Alpha:
        Alpha is strongest in posterior regions (PO7/PO8, Oz), so parietal-occipital electrodes best reflect Alpha changes linked to low arousal.

    Hemispheric Asymmetry:
        Left hemisphere positivity → C3 (central) and PO7 (posterior).
        Right hemisphere negativity → C4 (central) and PO8 (posterior).

        VR Stimulus Analysis for EEG Data

This script analyzes EEG recordings from VR headset sessions with different stimulus periods.

Stimulus Pattern (Total duration: 3 minutes 20 seconds / 200 seconds):
- 0-20 sec: white neutral
- 20-50 sec: warm color
- 50-80 sec: cold color
- 80-110 sec: warm color
- 110-140 sec: cold color
- 140-170 sec: warm color
- 170-200 sec: cold color

The goal is to compare EEG data between warm and cold color stimulus periods
and create visualizations to show differences in brain activity.

"""
# === USAGE INSTRUCTIONS ===
# To run the EEG analysis with the added plots and features:
# 1. Place your EEG CSV files in the 'data/eeg' directory.
# 2. Run this script using: python eeg_signal_processing_v2.py
# 3. Check the 'output/' folder for the generated PDF report.
# 4. Use the interactive Plotly charts for each EEG file.
# 5. The PDF includes alpha/beta ratio plots, band power RMS trends, and markers at key timepoints.

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import stft
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mne
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter
import scipy.signal as signal
from scipy.signal import cwt, morlet2
from fpdf import FPDF
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, roc_curve, auc
import plotly.io as pio
import webbrowser
import joblib
from scipy.signal import correlate
from imblearn.over_sampling import SMOTE  # [NEW] For class balancing
from scipy import stats
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import scipy.signal
from scipy.stats import skew, kurtosis
from scipy.signal import convolve

# --- CONFIGURATION ---
# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, 'data', 'raw', 'eeg')  
output_folder = os.path.join(script_dir, 'output')
processed_folder = os.path.join(script_dir, 'processed')
sampling_rate = 250  # Hz
plot_duration_seconds = 210  # 3.5 minutes
max_samples = plot_duration_seconds * sampling_rate
timestamp = datetime.datetime.now().strftime("__%Y%m%d%H%M%S")
downsample_factor = 1  # Downsample to 50 Hz (250/5)

# Configure plotting settings
sns.set_style("darkgrid")
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)

# Make sure required folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)
os.makedirs('reports', exist_ok=True)

# EEG frequency bands with more detailed beta bands
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta_low': (12, 15),
    'beta_mid': (15, 24),  # 24 is safely below Nyquist for fs=50Hz
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

# Configuration for electrode mapping and emotion detection
ELECTRODES = {
    'Fz': 0,    # Frontal midline
    'C3': 1,    # Left central
    'Cz': 2,    # Central midline
    'C4': 3,    # Right central
    'Pz': 4,    # Parietal midline
    'PO7': 5,   # Left parietal-occipital
    'Oz': 6,    # Occipital midline
    'PO8': 7    # Right parietal-occipital
}

EMOTION_CHANNELS = {
    'excited': ['Fz', 'C3', 'Cz', 'C4'],  # Beta up, Alpha down
    'angry': ['C4'],                       # Right frontal beta up
    'sad': ['PO8'],                        # Right alpha up
    'calm': ['PO7']                        # Left alpha up
}

EMOTION_THRESHOLDS = {
    'beta_high': 5.0,        # µV²/Hz for excited state
    'beta_asymmetry': 2.0,   # Ratio for angry state
    'alpha_low': 2.0         # µV²/Hz for sad state
}

# Window configurations
WINDOW_CONFIGS = {
    'short': {
        'duration': 5.0,     # seconds
        'overlap': 2.5       # 50% overlap
    },
    'long': {
        'duration': 10.0,    # seconds
        'overlap': 5.0       # 50% overlap
    }
}

def calculate_bandpower_ratios(band_powers):
    """Calculate standard EEG bandpower ratios for a given band powers dict."""
    # Avoid division by zero
    def safe_div(a, b):
        return a / b if b != 0 else 0.0

    delta = band_powers.get('delta', 0)
    theta = band_powers.get('theta', 0)
    alpha = band_powers.get('alpha', 0)
    beta_low = band_powers.get('beta_low', 0)
    beta_mid = band_powers.get('beta_mid', 0)
    beta = beta_low + beta_mid

    ratios = {
        'power_ratio_index': safe_div(beta, alpha),
        'delta_alpha_ratio': safe_div(delta, alpha),
        'theta_alpha_ratio': safe_div(theta, alpha),
        'theta_beta_ratio': safe_div(theta, beta),
        'theta_beta_alpha_ratio': safe_div(theta, (beta + alpha)),
        'engagement_index': safe_div(beta, (alpha + theta)),
    }
    return ratios

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a bandpass filter to the signal."""
    nyq = 0.5 * fs
    margin = 0.1  # Hz
    if highcut >= nyq:
        highcut = nyq - margin
    low = lowcut / nyq
    high = highcut / nyq
    if not (0 < low < 1 and 0 < high < 1):
        raise ValueError(f"Invalid cutoff frequencies: low={lowcut}, high={highcut}, fs={fs}")
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def notch_filter(data, freq, fs, q=30):
    """Apply a notch filter to remove power line noise."""
    nyq = 0.5 * fs  # Nyquist frequency
    w0 = freq / nyq
    if not (0 < w0 < 1):
        raise ValueError(f"Invalid notch frequency: freq={freq}, fs={fs}")
    b, a = signal.iirnotch(w0, q)
    return signal.filtfilt(b, a, data)

def compute_psd(data, fs, nperseg=None):
    """Compute power spectral density using Welch's method."""
    if nperseg is None:
        nperseg = min(256, len(data))
    freqs, psd = signal.welch(data, fs, nperseg=nperseg)
    return freqs, psd

def compute_psD(data, fs, nperseg=None):
    # Typo alias for compute_psd for legacy compatibility
    return compute_psd(data, fs, nperseg)

def calculate_band_powers(psd, freqs, bands=FREQ_BANDS):
    """Calculate power in specific frequency bands."""
    powers = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        powers[band_name] = np.trapezoid(psd[mask], freqs[mask])
    return powers


# Now you can use `baseline_processed` for normalization, plotting, or as a reference in your analysis.

def downsample_with_antialiasing(data, fs, factor):
    """Downsample signal with anti-aliasing filter."""
    from scipy.signal import butter, filtfilt, decimate
    if factor == 1:
        return data, fs
    nyq = 0.5 * fs
    cutoff = 0.8 * (nyq / factor)
    b, a = butter(4, cutoff / nyq, btype='low')
    filtered = filtfilt(b, a, data)
    downsampled = decimate(filtered, factor, ftype='iir', zero_phase=True)
    new_fs = fs // factor
    return downsampled, new_fs

def apply_antialiasing_filter(data, fs, cutoff=24, order=4):
    """Apply a lowpass filter for anti-aliasing before downsampling."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low')
    return signal.filtfilt(b, a, data)

def calculate_bandpower_with_buffer(data, fs, buffer_size, buffer_overlap):
    """
    Calculate time-varying band powers and ratios using a sliding buffer.
    Args:
        data (array): 1D EEG data
        fs (float): Sampling rate
        buffer_size (float): Buffer size in seconds
        buffer_overlap (float): Overlap in seconds
    Returns:
        tuple: (buffer_times, buffer_powers, buffer_ratios)
    """
    buffer_len = int(buffer_size * fs)
    step_len = int((buffer_size - buffer_overlap) * fs)
    n_samples = len(data)
    buffer_times = []
    buffer_powers = {band: [] for band in FREQ_BANDS}
    buffer_ratios = {ratio: [] for ratio in [
        'power_ratio_index',
        'delta_alpha_ratio',
        'theta_alpha_ratio',
        'theta_beta_ratio',
        'theta_beta_alpha_ratio',
        'engagement_index']}

    for start in range(0, n_samples - buffer_len + 1, step_len):
        end = start + buffer_len
        segment = data[start:end]
        freqs, psd = compute_psd(segment, fs)
        band_powers = calculate_band_powers(psd, freqs)
        ratios = calculate_bandpower_ratios(band_powers)
        for band in FREQ_BANDS:
            buffer_powers[band].append(band_powers[band])
        for ratio in buffer_ratios:
            buffer_ratios[ratio].append(ratios[ratio])
        buffer_times.append((start + end) / 2 / fs)

    # Convert lists to numpy arrays
    buffer_times = np.array(buffer_times)
    for band in buffer_powers:
        buffer_powers[band] = np.array(buffer_powers[band])
    for ratio in buffer_ratios:
        buffer_ratios[ratio] = np.array(buffer_ratios[ratio])
    return buffer_times, buffer_powers, buffer_ratios

def process_eeg_data(filepath, channels=None, buffer_size=2.0, buffer_overlap=0.5, downsample=True, classification_mode=None, output_folder=None, timestamp=None, pdf=None):
    """Process EEG data from a file with advanced buffer-based analysis.
    Args:
        filepath (str): Path to the data file
        channels (list): List of channel indices to process
        buffer_size (float): Size of buffer in seconds
        buffer_overlap (float): Overlap in seconds
        downsample (bool): Whether to downsample the data
        classification_mode (str): 'selch', 'allch', or None for both
        output_folder (str): Where to save outputs (for autocorr plot)
        timestamp (str): Timestamp for output naming
        pdf (PdfPages): PDF object to add plots to
    Returns:
        dict: Processed data including time-varying bandpowers and ratios
    """
    print(f"Processing EEG data from {os.path.basename(filepath)}...")
    # Read data
    df = pd.read_csv(filepath, header=None, low_memory=False)
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
            
            # Extract statistical features
            stat_features = extract_statistical_features(filtered, window_size=buffer_size, step_size=buffer_overlap, fs=current_fs)
            
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
                'sampling_rate': current_fs,  # Store the actual sampling rate used
                'stat_features': stat_features  # Store statistical features
            }
            
        except Exception as e:
            print(f"Error processing channel {channel_index}: {e}")
    
    # --- SVM window-wise affective state prediction ---
    try:
        modes = ['selch', 'allch'] if classification_mode is None else [classification_mode]
        for mode in modes:
            mode_str = 'allch' if mode == 'allch' else 'selch'
            if mode == 'selch':
                svm_path = os.path.join(script_dir, 'output', 'combined', 'svm_model_selch.joblib')
                scaler_path = os.path.join(script_dir, 'output', 'combined', 'scaler_selch.joblib')
                feature_fn = extract_features_selected
                sel_indices = [ELECTRODES['Fz'], ELECTRODES['C3'], ELECTRODES['C4'], ELECTRODES['PO7'], ELECTRODES['PO8']]
            elif mode == 'allch':
                svm_path = os.path.join(script_dir, 'output', 'combined', 'svm_model_allch.joblib')
                scaler_path = os.path.join(script_dir, 'output', 'combined', 'scaler_allch.joblib')
                feature_fn = extract_features_allch
                sel_indices = list(range(NUM_CHANNELS))
            else:
                continue
            if os.path.exists(svm_path) and os.path.exists(scaler_path):
                svm = joblib.load(svm_path)
                scaler = joblib.load(scaler_path)
                ref_ch = sel_indices[0]
                n_windows = len(processed_data[ref_ch]['buffer_data']['times'])
                feature_mat = []
                for i in range(n_windows):
                    window_data = {}
                    for ch in sel_indices:
                        window_data[ch] = {
                            'powers': {band: processed_data[ch]['buffer_data']['powers'][band][i] for band in FREQ_BANDS},
                            'power_ratios': {ratio: processed_data[ch]['buffer_data']['ratios'][ratio][i] for ratio in [
                                'power_ratio_index',
                                'delta_alpha_ratio',
                                'theta_alpha_ratio',
                                'theta_beta_ratio',
                                'theta_beta_alpha_ratio',
                                'engagement_index']}
                        }
                    feature_vec = feature_fn(window_data)
                    feature_mat.append(feature_vec)
                feature_mat = np.array(feature_mat)
                # For 'allch' mode, average features across all channels (reshape to [n_windows, n_features_per_ch], then mean)
                if mode == 'allch' and feature_mat.ndim == 2:
                    n_features_per_ch = int(feature_mat.shape[1] / NUM_CHANNELS)
                    feature_mat = feature_mat.reshape((n_windows, NUM_CHANNELS, n_features_per_ch)).mean(axis=1)
                # Remove or impute NaNs before classification
                feature_mat = np.nan_to_num(feature_mat, nan=0.0)
                feature_mat_scaled = scaler.transform(feature_mat)
                window_labels = svm.predict(feature_mat_scaled)
                # Map integer labels to affective state names if possible
                if hasattr(svm, 'classes_'):
                    label_map = {i: str(c).capitalize() for i, c in enumerate(svm.classes_)}
                    window_labels_named = [label_map.get(lbl, str(lbl)) for lbl in window_labels]
                else:
                    window_labels_named = [str(lbl) for lbl in window_labels]
                # Store in processed_data dict (mode-specific)
                processed_data[f'affective_state_labels_{mode}'] = window_labels_named
                # --- Rule-based emotion labeling ---
                n_windows = len(processed_data[ref_ch]['buffer_data']['times'])
                rule_labels = []
                for i in range(n_windows):
                    window_band_powers = {}
                    for ch_name, ch_idx in ELECTRODES.items():
                        window_band_powers[ch_idx] = {band: processed_data[ch_idx]['buffer_data']['powers'][band][i] for band in FREQ_BANDS}
                    rule_labels.append(label_emotion_window(window_band_powers, ELECTRODES, EMOTION_THRESHOLDS))
                processed_data[f'affective_state_labels_rule_{mode}'] = rule_labels
            else:
                # Only print the warning if classification_mode is explicitly set (i.e., not None),
                # to avoid spamming during batch feature extraction before models are trained.
                if classification_mode is not None:
                    print(f"[WARN] SVM model or scaler not found at expected paths: {svm_path}, {scaler_path}")
    except Exception as e:
        print(f"[WARN] Could not compute SVM affective state labels: {e}")

    # --- Autocorrelation plot integration ---
    if output_folder is not None and timestamp is not None:
        # Pick a representative channel (e.g., Fz)
        ch_idx = ELECTRODES['Fz'] if 'Fz' in ELECTRODES else 0
        filtered_signal = processed_data[ch_idx]['filtered']
        filename = os.path.basename(filepath)
        class_mode_suffix = f"_{classification_mode}" if classification_mode else ""
        plot_eeg_autocorrelation(filtered_signal, current_fs, output_folder, filename, channel_labels[ch_idx], timestamp, class_mode_suffix, pdf=pdf)

    return processed_data

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
    _, right_psd = compute_psD(right_data, fs)
    
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

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a bandpass filter to the signal."""
    nyq = 0.5 * fs
    margin = 0.1  # Hz
    if highcut >= nyq:
        highcut = nyq - margin
    low = lowcut / nyq
    high = highcut / nyq
    if not (0 < low < 1 and 0 < high < 1):
        raise ValueError(f"Invalid cutoff frequencies: low={lowcut}, high={highcut}, fs={fs}")
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def filter_eeg(data, fs):
    # Use 24 Hz as high cutoff for fs=50Hz
    data = bandpass_filter(data, 0.5, 24, fs, order=4)
    # Only apply 50 Hz notch if fs > 100 Hz
    if fs > 100:
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

def extract_features(file_data):
    """Extract features from processed EEG data for machine learning."""
    features = []
    # Aggregate mean band powers across all channels
    for band in FREQ_BANDS:
        band_powers = [ch_data['powers'][band] for ch_data in file_data.values() if 'powers' in ch_data]
        features.append(np.mean(band_powers))
    # Aggregate mean ratios across all channels
    ratio_names = [
        'power_ratio_index',
        'delta_alpha_ratio',
        'theta_alpha_ratio',
        'theta_beta_ratio',
        'theta_beta_alpha_ratio',
        'engagement_index'
    ]
    for ratio in ratio_names:
        ratio_vals = [ch_data['power_ratios'][ratio] for ch_data in file_data.values() if 'power_ratios' in ch_data]
        features.append(np.mean(ratio_vals))
    return np.array(features)

# Detection and classification functions

def detect_artifacts(acc_data, gyr_data, threshold=0.3):
    """Detect motion artifacts using accelerometer and gyroscope data.
    
    Args:
        acc_data: Accelerometer data (channels 8-10)
        gyr_data: Gyroscope data (channels 11-13)
        threshold: Motion detection threshold
        
    Returns:
        boolean array: True for clean data, False for artifacts
    """
    acc_magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
    gyr_magnitude = np.sqrt(np.sum(gyr_data**2, axis=1))
    motion_score = acc_magnitude + gyr_magnitude
    return motion_score <= threshold


def create_emotion_visualization(processed_data, window_size='short'):
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
    for ch_idx, ch_data in processed_data.items():
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
    for ch_idx, ch_data in processed_data.items():
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
    fig.update_xaxes(title_text="Channel", row=3, col=1)
    fig.update_yaxes(title_text="Power", row=3, col=1)
    fig.update_xaxes(title_text="Ratio Type", row=4, col=1)
    fig.update_yaxes(title_text="Ratio Value", row=4, col=1)
    fig.update_xaxes(title_text="Time (s)", row=5, col=1)
    fig.update_yaxes(title_text="Power", row=5, col=1)
    fig.update_xaxes(title_text="Time (s)", row=6, col=1)
    fig.update_yaxes(title_text="Ratio Value", row=6, col=1)
    
    return fig

# --- Dual-mode SVM feature extraction and classification ---
SELECTED_ELECTRODE_INDICES = [ELECTRODES[ch] for ch in ['Fz', 'C3', 'C4', 'PO7', 'PO8']]
ALL_CHANNEL_INDICES = list(range(NUM_CHANNELS))

# Feature extraction for selected electrodes
def extract_features_selected(processed_data):
    """Extract features from selected channels (Fz, C3, C4, PO7, PO8) for SVM."""
    sel_indices = [ELECTRODES['Fz'], ELECTRODES['C3'], ELECTRODES['C4'], ELECTRODES['PO7'], ELECTRODES['PO8']]
    features = []
    for ch in sel_indices:
        ch_data = processed_data.get(ch, {})
        for band in FREQ_BANDS:
            features.append(ch_data.get('powers', {}).get(band, 0))
        for ratio in [
            'power_ratio_index',
            'delta_alpha_ratio',
            'theta_alpha_ratio',
            'theta_beta_ratio',
            'theta_beta_alpha_ratio',
            'engagement_index']:
            features.append(ch_data.get('power_ratios', {}).get(ratio, 0))
    return np.array(features)

# Feature extraction for all 8 channels
def extract_features_allch(processed_data):
    """Extract features from all 8 EEG channels for SVM."""
    features = []
    for ch in range(NUM_CHANNELS):
        ch_data = processed_data.get(ch, {})
        for band in FREQ_BANDS:
            features.append(ch_data.get('powers', {}).get(band, 0))
        for ratio in [
            'power_ratio_index',
            'delta_alpha_ratio',
            'theta_alpha_ratio',
            'theta_beta_ratio',
            'theta_beta_alpha_ratio',
            'engagement_index']:
            features.append(ch_data.get('power_ratios', {}).get(ratio, 0))
    return np.array(features)

# Helper to get model/scaler paths for each mode
def get_model_paths(mode):
    suffix = '_allch' if mode == 'allch' else '_selch'
    svm_path = os.path.join('output', 'combined', f'svm_model{suffix}.joblib')
    scaler_path = os.path.join('output', 'combined', f'scaler{suffix}.joblib')
    return svm_path, scaler_path

# Train and save SVM/scaler for a given mode
# (Assumes X, y are already prepared for the mode)
def train_and_save_svm(X, y, mode):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_scaled, y)
    svm_path, scaler_path = get_model_paths(mode)
    joblib.dump(svm, svm_path)
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Trained and saved SVM/scaler for mode {mode}.")

def train_advanced_svm(X, y, mode, feature_names=None, plot_dir='reports'):
    """
    Train SVM with SMOTE, RFE feature selection, and GridSearchCV hyperparameter tuning.
    Plots ROC and confusion matrix, saves to 'reports/'.
    Returns best estimator, scaler, and selected features.
    """
    os.makedirs(plot_dir, exist_ok=True)
    # 1. Balance classes with SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    # 2. Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    # 3. Feature selection with RFE (using SVM as estimator)
    base_svm = SVC(kernel='linear', probability=True, random_state=42)
    n_features = min(10, X_scaled.shape[1])
    rfe = RFE(base_svm, n_features_to_select=n_features)
    X_rfe = rfe.fit_transform(X_scaled, y_res)
    selected_features = rfe.get_support(indices=True)
    if feature_names is not None:
        selected_feature_names = [feature_names[i] for i in selected_features]
    else:
        selected_feature_names = selected_features
    # 4. Hyperparameter tuning with GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1, 1],
        'kernel': ['rbf']
    }
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='balanced_accuracy')
    grid.fit(X_rfe, y_res)
    best_svm = grid.best_estimator_
    # 5. Evaluation (train/test split for reporting)
    X_train, X_test, y_train, y_test = train_test_split(X_rfe, y_res, test_size=0.2, random_state=42, stratify=y_res)
    best_svm.fit(X_train, y_train)
    y_pred = best_svm.predict(X_test)
    y_proba = best_svm.predict_proba(X_test)
    # 6. Confusion matrix
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(best_svm, X_test, y_test, ax=ax_cm, cmap='Blues')
    plt.title(f'Confusion Matrix ({mode})')
    cm_path = os.path.join(plot_dir, f'confusion_matrix_{mode}.pdf')
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)
    # 7. ROC curve (one-vs-rest)
    if len(np.unique(y_test)) > 1:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_estimator(best_svm, X_test, y_test, ax=ax_roc)
        plt.title(f'ROC Curve ({mode})')
        roc_path = os.path.join(plot_dir, f'roc_curve_{mode}.pdf')
        fig_roc.savefig(roc_path)
        plt.close(fig_roc)
    else:
        roc_path = None
    # 8. Save model and scaler
    svm_path, scaler_path = get_model_paths(mode)
    joblib.dump(best_svm, svm_path)
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Trained and saved SVM/scaler for mode {mode}.")
    print(f"[INFO] Confusion matrix saved to {cm_path}")
    if roc_path:
        print(f"[INFO] ROC curve saved to {roc_path}")
    return best_svm, scaler, selected_feature_names


def perform_anova_on_bandpowers(X, y, feature_names=None, report_dir='reports', mode='allch'):
    """
    Perform ANOVA for each feature across emotion classes. Save results to PDF in 'reports/'.
    """
    os.makedirs(report_dir, exist_ok=True)
    results = []
    for i in range(X.shape[1]):
        groups = [X[y == label, i] for label in np.unique(y)]
        fval, pval = stats.f_oneway(*groups)
        fname = feature_names[i] if feature_names is not None else f'feature_{i}'
        results.append((fname, fval, pval))
    # Save to PDF
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'ANOVA Results ({mode})', ln=1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, 'Feature\tF-value\tp-value', ln=1)
    for fname, fval, pval in results:
        pdf.cell(0, 10, f'{fname}\t{fval:.3f}\t{pval:.3e}', ln=1)
    pdf_path = os.path.join(report_dir, f'anova_results_{mode}.pdf')
    pdf.output(pdf_path)
    print(f"[INFO] ANOVA results saved to {pdf_path}")
    return results


def calculate_asymmetry_indices(processed_data):
    """
    Calculate alpha asymmetry indices for C3-C4 and PO7-PO8.
    Returns a dict with asymmetry values. Skips pairs if channel data is missing.
    """
    indices = {}
    # C3 (1) vs C4 (3), PO7 (5) vs PO8 (7)
    for (left, right, label) in [(1, 3, 'C3-C4'), (5, 7, 'PO7-PO8')]:
        if left in processed_data and right in processed_data:
            left_data = processed_data[left]['filtered']
            right_data = processed_data[right]['filtered']
            fs = processed_data[left]['sampling_rate']
            asym = calculate_alpha_asymmetry(left_data, right_data, fs)
            indices[label] = asym
        else:
            indices[label] = None  # Mark as missing
    return indices

# --- Methodology clarification ---
# The pipeline now includes:
# - Class balancing with SMOTE
# - Feature selection with RFE
# - SVM hyperparameter tuning with GridSearchCV
# - Statistical validation with ANOVA
# - ROC/confusion matrix and asymmetry index reporting
# - All PDF reports saved to 'reports/'
# - Expanded docstrings and comments for clarity

def label_emotion_window(buffer_powers, electrodes=ELECTRODES, thresholds=EMOTION_THRESHOLDS):
    """Label a single window based on neurophysiological EEG emotion rules."""
    # Extract relevant band powers
    beta_fz = buffer_powers[electrodes['Fz']]['beta_low'] + buffer_powers[electrodes['Fz']]['beta_mid']
    beta_cz = buffer_powers[electrodes['Cz']]['beta_low'] + buffer_powers[electrodes['Cz']]['beta_mid']
    beta_c3 = buffer_powers[electrodes['C3']]['beta_low'] + buffer_powers[electrodes['C3']]['beta_mid']
    beta_c4 = buffer_powers[electrodes['C4']]['beta_low'] + buffer_powers[electrodes['C4']]['beta_mid']
    alpha_po7 = buffer_powers[electrodes['PO7']]['alpha']
    alpha_po8 = buffer_powers[electrodes['PO8']]['alpha']
    alpha_oz = buffer_powers[electrodes['Oz']]['alpha']
    # Excited: High frontal/central beta, low posterior alpha
    if (beta_fz > thresholds['beta_high'] or beta_cz > thresholds['beta_high'] or beta_c3 > thresholds['beta_high'] or beta_c4 > thresholds['beta_high']) and \
       (alpha_po7 < thresholds['alpha_low'] and alpha_po8 < thresholds['alpha_low'] and alpha_oz < thresholds['alpha_low']):
        return 'excited'
    # Angry: High right beta or beta asymmetry
    if (beta_c4 > thresholds['beta_high']) or (beta_c4 / (beta_c3 + 1e-6) > thresholds['beta_asymmetry']):
        return 'angry'
    # Sad: High right posterior alpha
    if alpha_po8 > thresholds['alpha_low']:
        return 'sad'
    # Calm: High left posterior alpha
    if alpha_po7 > thresholds['alpha_low']:
        return 'calm'
    return 'unknown'

def label_emotion_window_rule_based(window_band_powers):
    """
    Classify a window as Excited, Angry, Sad, or Calm based on band powers and neurophysiological rules.
    Args:
        window_band_powers (dict): {ch_idx: {band: value, ...}, ...} for this window
    Returns:
        str: One of 'Excited', 'Angry', 'Sad', 'Calm'
    """
    # Channel indices for relevant electrodes
    Fz, C3, Cz, C4, PO7, Oz, PO8 = 0, 1, 2, 3, 5, 6, 7
    # 1. Excited: High Beta (Fz, C3, Cz, C4), Low Alpha (PO7, Oz, PO8)
    beta_frontal = np.mean([
        window_band_powers.get(ch, {}).get('beta_low', 0) + window_band_powers.get(ch, {}).get('beta_mid', 0)
        for ch in [Fz, C3, Cz, C4]
    ])
    alpha_posterior = np.mean([
        window_band_powers.get(ch, {}).get('alpha', 0)
        for ch in [PO7, Oz, PO8]
    ])
    # 2. Angry: High Beta at C4 (right central)
    beta_c4 = window_band_powers.get(C4, {}).get('beta_low', 0) + window_band_powers.get(C4, {}).get('beta_mid', 0)
    # 3. Sad: High Alpha at PO8 (right posterior)
    alpha_po8 = window_band_powers.get(PO8, {}).get('alpha', 0)
    # 4. Calm: High Alpha at PO7 (left posterior)
    alpha_po7 = window_band_powers.get(PO7, {}).get('alpha', 0)
    # --- Thresholds (relative, not absolute) ---
    # Compute z-scores for each metric within this window (relative to all 4 metrics)
    metrics = np.array([beta_frontal, beta_c4, alpha_po8, alpha_po7])
    zscores = (metrics - np.mean(metrics)) / (np.std(metrics) + 1e-6)
    # Assign emotion based on which metric is highest (z-score)
    idx = np.argmax(zscores)
    if idx == 0:
        # Excited: beta_frontal highest
        return 'Excited'
    elif idx == 1:
        return 'Angry'
    elif idx == 2:
        return 'Sad'
    elif idx == 3:
        return 'Calm'
    else:
        return 'Unknown'

def compute_stft(data, fs, nperseg=256, noverlap=None):
    """Compute Short-Time Fourier Transform."""
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Zxx = stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, np.abs(Zxx)

def compute_wavelet(data, fs, wavelet_width=5.0):
    """Compute Wavelet Transform using continuous wavelet transform."""
    # Use Morlet wavelet for analysis
    scales = np.arange(1, 128)
    frequencies = fs / (scales * wavelet_width)
    coef = cwt(data, morlet2, scales, w=wavelet_width)
    return frequencies, np.abs(coef)

def smooth_emotional_state(processed_data, sampling_rate, smoothing_rate=5):
    smoothed_states = []
    time_points = []

    for ch_idx, ch_data in processed_data.items():
        buffer_times = ch_data['buffer_data']['times']
        ratios = ch_data['buffer_data']['ratios']

        # Smooth using moving average
        for ratio_name, ratio_values in ratios.items():
            smoothed_values = np.convolve(
                ratio_values, np.ones(smoothing_rate) / smoothing_rate, mode='valid'
            )
            smoothed_states.append(smoothed_values)
            time_points.append(buffer_times[:len(smoothed_values)])

    return smoothed_states, time_points

def generate_pdf_report(
    output_dir, emotion_matrix, file_names, valence_arousal_data, png_paths=None, baseline_file=None, settings=None, steps_description=None, results_summary=None,
    all_processed_data=None, classification_results=None, visualizations=None, timestamp=None,
    eda_processed_data=None
):
    """
    Generate a comprehensive PDF report including:
    - Sample/data summary
    - Classification results (confusion matrix, ROC, etc.)
    - Statistical summaries (ANOVA, asymmetry indices)
    - All relevant visualizations
    - Methodology and settings
    - Statistical feature tables/plots for EEG/EDA
    The report is always saved to the 'reports/' directory.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    from fpdf import FPDF
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure reports directory exists
    reports_dir = os.path.join(os.path.dirname(output_dir), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    pdf_path = os.path.join(reports_dir, f'analysis_report_{timestamp}.pdf')

    with PdfPages(pdf_path) as pdf:
        # 1. Title and metadata page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.title('EEG Emotion Analysis Report', fontsize=20, pad=40)
        plt.text(0.1, 0.8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=12)
        if settings:
            plt.text(0.1, 0.7, f"Settings: {settings}", fontsize=10)
        if steps_description:
            plt.text(0.1, 0.6, f"Pipeline Steps: {steps_description}", fontsize=10)
        plt.text(0.1, 0.5, f"Files analyzed: {', '.join(file_names)}", fontsize=10)
        plt.text(0.1, 0.4, f"Baseline: {baseline_file}", fontsize=10)
        plt.text(0.1, 0.3, f"Report location: {pdf_path}", fontsize=8)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # 2. Data/sample summary
        if all_processed_data:
            plt.figure(figsize=(8, 4))
            plt.axis('off')
            plt.title('Sample/Data Summary', fontsize=14)
            lines = []
            for fname, pdata in all_processed_data.items():
                n_ch = len(pdata)
                n_samples = len(next(iter(pdata.values()))['filtered']) if n_ch > 0 else 0
                lines.append(f"{fname}: {n_ch} channels, {n_samples} samples/channel")
            plt.text(0.1, 0.8, '\n'.join(lines), fontsize=10)
            pdf.savefig()
            plt.close()

        # 3. Classification results
        if classification_results:
            plt.figure(figsize=(8, 4))
            plt.axis('off')
            plt.title('Classification Results', fontsize=14)
            for mode, res in classification_results.items():
                plt.text(0.1, 0.8 - 0.1*list(classification_results.keys()).index(mode), f"{mode}: {res}", fontsize=10)
            pdf.savefig()
            plt.close()

        # 4. Statistical summaries (ANOVA, asymmetry)
        if results_summary:
            plt.figure(figsize=(8, 4))
            plt.axis('off')
            plt.title('Statistical Results', fontsize=14)
            plt.text(0.1, 0.8, results_summary, fontsize=10)
            pdf.savefig()
            plt.close()

        # 5. EEG Statistical feature tables/plots
        if all_processed_data:
            for fname, pdata in all_processed_data.items():
                for ch_idx, ch_data in pdata.items():
                    stat_features = ch_data.get('stat_features', [])
                    if stat_features:
                        # Plot each feature over windows
                        win_times = np.arange(len(stat_features)) * 2.0  # 2s window
                        for feat in ['mean', 'std', 'min', 'max', 'rms', 'skewness', 'kurtosis', 'moving_average']:
                            values = [f[feat] for f in stat_features if feat in f]
                            if len(values) > 0:
                                plt.figure(figsize=(8, 2))
                                plt.plot(win_times[:len(values)], values, label=feat)
                                plt.title(f'{fname} - {channel_labels[ch_idx]} - {feat}')
                                plt.xlabel('Time (s)')
                                plt.ylabel(feat)
                                plt.tight_layout()
                                pdf.savefig()
                                plt.close()

        # 6. EDA statistical features and decomposition (if provided)
        if eda_processed_data:
            for fname, eda_dict in eda_processed_data.items():
                for key in ['raw', 'tonic', 'phasic']:
                    stat_features = eda_dict['features'].get(key, [])
                    if stat_features:
                        win_times = np.arange(len(stat_features)) * 2.0
                        for feat in ['mean', 'std', 'min', 'max', 'rms', 'skewness', 'kurtosis', 'moving_average']:
                            values = [f[feat] for f in stat_features if feat in f]
                            if len(values) > 0:
                                plt.figure(figsize=(8, 2))
                                plt.plot(win_times[:len(values)], values, label=feat)
                                plt.title(f'{fname} - EDA {key} - {feat}')
                                plt.xlabel('Time (s)')
                                plt.ylabel(feat)
                                plt.tight_layout()
                                pdf.savefig()
                                plt.close()
                # EDA decomposition plot
                if 'raw' in eda_dict and 'tonic' in eda_dict and 'phagic' in eda_dict:
                    time = np.arange(len(eda_dict['raw'])) / 250.0
                    plt.figure(figsize=(8, 3))
                    plt.plot(time, eda_dict['raw'], label='Raw')
                    plt.plot(time, eda_dict['tonic'], label='Tonic')
                    plt.plot(time, eda_dict['phagic'], label='Phagic')
                    plt.title(f'{fname} - EDA Decomposition')
                    plt.xlabel('Time (s)')
                    plt.ylabel('EDA (uS)')
                    plt.legend()
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()

        # 6b. Comparative tonic/phasic plot across all files (EDA and EEG)
        if eda_processed_data:
            plot_tonic_phasic_comparison_all_files(eda_processed_data, all_processed_data if all_processed_data else None, pdf)

        # 6c. Tonic lateralized alpha asymmetry for low-arousal states (Calm/Sad)
        if all_processed_data:
            plot_tonic_alpha_lateralization_low_arousal(all_processed_data, pdf)

        # 7. Band power/ratio/engagement/asymmetry plots
        if all_processed_data:
            # Band powers
            for fname, pdata in all_processed_data.items():
                for ch_idx, ch_data in pdata.items():
                    # Band powers
                    plt.figure(figsize=(6, 2))
                    plt.bar(list(ch_data['powers'].keys()), list(ch_data['powers'].values()))
                    plt.title(f'{fname} - {channel_labels[ch_idx]} Band Powers')
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                    # Ratios
                    plt.figure(figsize=(6, 2))
                    plt.bar(list(ch_data['power_ratios'].keys()), list(ch_data['power_ratios'].values()))
                    plt.title(f'{fname} - {channel_labels[ch_idx]} Power Ratios')
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
            # Engagement index and alpha asymmetry
            for fname, pdata in all_processed_data.items():
                engagement = [ch_data['power_ratios'].get('engagement_index', 0) for ch_data in pdata.values()]
                plt.figure(figsize=(6, 2))
                plt.bar(range(len(engagement)), engagement)
                plt.title(f'{fname} - Engagement Index (all channels)')
                plt.tight_layout()
                pdf.savefig()
                plt.close()
            # Alpha asymmetry
            for fname, pdata in all_processed_data.items():
                if 1 in pdata and 3 in pdata:
                    left = pdata[1]['filtered']
                    right = pdata[3]['filtered']
                    fs = pdata[1]['sampling_rate']
                    asym = calculate_alpha_asymmetry(left, right, fs)
                    plt.figure(figsize=(6, 2))
                    plt.bar(['C3-C4'], [asym])
                    plt.title(f'{fname} - Alpha Asymmetry (C3-C4)')
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                if 5 in pdata and 7 in pdata:
                    left = pdata[5]['filtered']
                    right = pdata[7]['filtered']
                    fs = pdata[5]['sampling_rate']
                    asym = calculate_alpha_asymmetry(left, right, fs)
                    plt.figure(figsize=(6, 2))
                    plt.bar(['PO7-PO8'], [asym])
                    plt.title(f'{fname} - Alpha Asymmetry (PO7-PO8)')
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()

        # 8. Visualizations (figures, confusion matrix, ROC, etc.)
        if visualizations:
            for fig in visualizations:
                pdf.savefig(fig)
                plt.close(fig)
        if png_paths:
            for png in png_paths:
                img = plt.imread(png)
                plt.figure(figsize=(8, 4))
                plt.imshow(img)
                plt.axis('off')
                pdf.savefig()
                plt.close()

        # 9. If nothing was added, add a placeholder page
        if not (all_processed_data or classification_results or results_summary or visualizations or png_paths or eda_processed_data):
            plt.figure(figsize=(8, 4))
            plt.axis('off')
            plt.title('No data available for report.', fontsize=16)
            pdf.savefig()
            plt.close()

    print(f"[INFO] PDF report saved to {pdf_path}")
    return pdf_path

def main():
    import glob
    import os
    import datetime
    import numpy as np
    import pandas as pd

    # Set up paths and timestamp
    data_folder = os.path.join(script_dir, 'data', 'raw', 'eeg')
    reports_dir = os.path.join(script_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # 1. Collect all EEG files
    eeg_files = glob.glob(os.path.join(data_folder, '*.csv'))
    all_processed_data = {}
    file_names = []
    for eeg_file in eeg_files:
        processed = process_eeg_data(eeg_file)
        all_processed_data[os.path.basename(eeg_file)] = processed
        file_names.append(os.path.basename(eeg_file))

    # 2. Prepare features and labels for SVM (example: using allch mode)
    X, y = [], []
    for fname, pdata in all_processed_data.items():
        # Placeholder: extract label from filename (customize as needed)
        label = fname.split('_')[0]
        X.append(extract_features_allch(pdata))
        y.append(label)
    X = np.array(X)
    y = np.array(y)

    # 3. Train advanced SVM and perform ANOVA if there is more than one class
    classification_results = {}
    results_summary = ''
    if len(np.unique(y)) > 1:
        best_svm, scaler, selected_features = train_advanced_svm(X, y, mode='allch')
        classification_results['allch'] = f"SVM trained with {len(selected_features)} features."
        anova_results = perform_anova_on_bandpowers(X, y, feature_names=None, report_dir=reports_dir, mode='allch')
        results_summary += f"ANOVA performed on {X.shape[1]} features.\n"
    else:
        results_summary += "Not enough classes for SVM/ANOVA.\n"

    # 4. Calculate asymmetry indices for each file
    asymmetry_indices = {}
    for fname, pdata in all_processed_data.items():
        asymmetry_indices[fname] = calculate_asymmetry_indices(pdata)
    results_summary += f"Alpha asymmetry indices calculated for all files.\n"

    # 5. Generate PDF report (ensure all relevant data is passed)
    generate_pdf_report(
        output_dir=reports_dir,
        emotion_matrix=None,
        file_names=file_names,
        valence_arousal_data=None,
        png_paths=None,
        baseline_file=None,
        settings=f"Sampling rate: {sampling_rate}, Downsample factor: {downsample_factor}",
        steps_description="Advanced SVM, SMOTE, RFE, ANOVA, Asymmetry, Visualization",
        results_summary=results_summary,
        all_processed_data=all_processed_data,
        classification_results=classification_results,
        visualizations=None,
        timestamp=timestamp
    )

# --- COMPARATIVE TONIC/PHASIC VISUALIZATION ACROSS ALL FILES ---
def plot_tonic_phasic_comparison_all_files(eda_processed_data, eeg_processed_data=None, pdf=None):
    """
    Plot tonic and phasic EDA components for all files on the same axes for comparison.
    Optionally, plot EEG slow/fast band power envelopes at key electrodes to mimic tonic/phasic and lateralization.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    colors = plt.cm.tab10.colors
    # --- EDA: Tonic and Phasic ---
    plt.figure(figsize=(12, 6))
    for i, (fname, eda_dict) in enumerate(eda_processed_data.items()):
        n = len(eda_dict['tonic'])
        t = np.arange(n) / 250.0
        plt.plot(t, eda_dict['tonic'], label=f"{fname} tonic", color=colors[i % 10], alpha=0.7, linestyle='-')
    plt.title('EDA Tonic Component Comparison Across Files')
    plt.xlabel('Time (s)')
    plt.ylabel('EDA (uS)')
    plt.legend(fontsize=8)
    plt.tight_layout()
    if pdf is not None:
        pdf.savefig()
    plt.close()

    plt.figure(figsize=(12, 6))
    for i, (fname, eda_dict) in enumerate(eda_processed_data.items()):
        n = len(eda_dict['phagic']) if 'phagic' in eda_dict else len(eda_dict['phasic'])
        t = np.arange(n) / 250.0
        phasic = eda_dict.get('phagic', eda_dict.get('phasic'))
        plt.plot(t, phasic, label=f"{fname} phasic", color=colors[i % 10], alpha=0.7, linestyle='-')
    plt.title('EDA Phasic Component Comparison Across Files')
    plt.xlabel('Time (s)')
    plt.ylabel('EDA (uS)')
    plt.legend(fontsize=8)
    plt.tight_layout()
    if pdf is not None:
        pdf.savefig()
    plt.close()

    # --- EEG: Tonic/Phasic-like (slow/fast band power) at key electrodes ---
    if eeg_processed_data is not None:
        # Define slow (delta+theta+alpha) and fast (beta) bands
        slow_bands = ['delta', 'theta', 'alpha']
        fast_bands = ['beta_low', 'beta_mid']
        # Key electrodes for lateralization: PO7 (5), PO8 (7), C3 (1), C4 (3)
        key_electrodes = {'PO7': 5, 'PO8': 7, 'C3': 1, 'C4': 3}
        for label, ch_idx in key_electrodes.items():
            plt.figure(figsize=(12, 5))
            for i, (fname, pdata) in enumerate(eeg_processed_data.items()):
                ch_data = pdata.get(ch_idx)
                if ch_data is None:
                    continue
                buffer_times = ch_data['buffer_data']['times']
                slow_power = np.zeros_like(buffer_times)
                fast_power = np.zeros_like(buffer_times)
                for band in slow_bands:
                    slow_power += ch_data['buffer_data']['powers'][band]
                for band in fast_bands:
                    fast_power += ch_data['buffer_data']['powers'][band]
                plt.plot(buffer_times, slow_power, label=f"{fname} slow (tonic)", color=colors[i % 10], alpha=0.7, linestyle='-')
                plt.plot(buffer_times, fast_power, label=f"{fname} fast (phasic)", color=colors[i % 10], alpha=0.7, linestyle='--')
            plt.title(f'EEG Slow (Tonic) and Fast (Phasic) Band Power at {label} Across Files')
            plt.xlabel('Time (s)')
            plt.ylabel('Band Power (a.u.)')
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            if pdf is not None:
                pdf.savefig()
            plt.close()

def plot_tonic_alpha_lateralization_low_arousal(all_processed_data, pdf=None):
    """
    For each file, extract tonic (mean) alpha power at PO7 (left) and PO8 (right) for windows labeled as 'Calm' or 'Sad'.
    Compute and plot the alpha asymmetry index (right - left, log scale) grouped by emotion label.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    colors = plt.cm.Set2.colors
    results = {'Calm': [], 'Sad': []}
    file_labels = []
    for fname, pdata in all_processed_data.items():
        # Find window labels and band powers for PO7 (5) and PO8 (7)
        ch_left = pdata.get(5, {})  # PO7
        ch_right = pdata.get(7, {}) # PO8
        if not ch_left or not ch_right:
            continue
        # Assume window labels and band powers are stored per window
        window_labels = ch_left.get('window_labels', [])
        alpha_left = [w.get('alpha', 0) for w in ch_left.get('window_band_powers', [])]
        alpha_right = [w.get('alpha', 0) for w in ch_right.get('window_band_powers', [])]
        for lbl, aL, aR in zip(window_labels, alpha_left, alpha_right):
            if lbl in results:
                # Tonic = mean alpha in window
                asym = np.log(aR + 1e-6) - np.log(aL + 1e-6)
                results[lbl].append(asym)
        file_labels.append(fname)
    # Plot
    plt.figure(figsize=(8, 4))
    data = [results['Calm'], results['Sad']]
    plt.boxplot(data, labels=['Calm', 'Sad'], patch_artist=True, boxprops=dict(facecolor=colors[0], alpha=0.5))
    plt.title('Tonic Lateralized Alpha Asymmetry (PO8-PO7) in Low-Arousal States')
    plt.ylabel('Alpha Asymmetry (log right - log left)')
    plt.xlabel('Emotion (Low Arousal)')
    plt.grid(True, axis='y', alpha=0.3)
    if pdf is not None:
        pdf.savefig()
    plt.close()
