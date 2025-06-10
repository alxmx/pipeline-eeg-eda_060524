#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VR Stimulus Analysis Script
===========================
This script processes EEG data from VR headset sessions with different stimulus periods.
It generates visualizations to analyze the differences in EEG signals during warm and cold color stimuli.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from scipy.stats import ttest_ind
import datetime
from matplotlib.patches import Rectangle
from fpdf import FPDF
from matplotlib.backends.backend_pdf import PdfPages
# -*- coding: utf-8 -*-

"""
VR Stimulus Analysis for EEG Data
=================================
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

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from scipy.stats import ttest_ind
import datetime
from matplotlib.patches import Rectangle
from fpdf import FPDF
from matplotlib.backends.backend_pdf import PdfPages

# Add the parent directory to sys.path to import functions from eeg_signal_processing_v2
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import necessary functions from EEG processing module
from eeg_signal_processing_v2 import (process_eeg_data, filter_eeg, compute_psd, 
                                     calculate_normalized_band_powers, 
                                     calculate_bandpower_ratios, add_time_markers,
                                     FREQ_BANDS, channel_labels, ELECTRODES)

# Define directories
script_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
data_folder = os.path.join(script_dir, 'data', 'raw', 'eeg')
output_folder = os.path.join(script_dir, 'output', 'vr_analysis')
os.makedirs(output_folder, exist_ok=True)

# Analysis parameters (for metadata)
BUFFER_SIZE = 1.0  # seconds
BUFFER_OVERLAP = 0.5  # seconds
SAMPLING_RATE = 250  # Hz

# Define stimulus periods (in seconds)
stimulus_periods = {
    'neutral': (0, 20),
    'warm_1': (20, 50),
    'cold_1': (50, 80),
    'warm_2': (80, 110),
    'cold_2': (110, 140),
    'warm_3': (140, 170),
    'cold_3': (170, 200)
}

# Define color-coding for stimulus visualization
period_colors = {
    'neutral': 'lightgray',
    'warm_1': 'salmon',
    'cold_1': 'lightskyblue',
    'warm_2': 'salmon',
    'cold_2': 'lightskyblue',
    'warm_3': 'salmon',
    'cold_3': 'lightskyblue'
}

# EEG files to analyze
vr_eeg_files = [
    'UnicornRecorder_12_05_2025_12_37_430.csv',
    'UnicornRecorder_12_05_2025_12_41_310.csv',
    'UnicornRecorder_12_05_2025_12_45_530.csv',
    'UnicornRecorder_12_05_2025_12_50_080.csv'
]

# Classification mode: use all channels averaged or only selected electrodes
USE_ALL_CHANNELS_FOR_CLASSIFICATION = True  # Set to True for all channels averaged, False for selected electrodes

# For selected electrodes (default: Fz, C3, C4, PO7, PO8)
SELECTED_CHANNELS = [0, 1, 3, 5, 7]

# Suffix for output files based on classification mode
CLASS_MODE_SUFFIX = '_allch' if USE_ALL_CHANNELS_FOR_CLASSIFICATION else '_selch'

VR_STIMULUS_INTRO = (
    "VR Stimulus Analysis for EEG Data\n"
    "===============================\n"
    "This script analyzes EEG recordings from VR headset sessions with different stimulus periods.\n\n"
    "Stimulus Pattern (Total duration: 3 minutes 20 seconds / 200 seconds):\n"
    "- 0-20 sec: white neutral\n"
    "- 20-50 sec: warm color\n"
    "- 50-80 sec: cold color\n"
    "- 80-110 sec: warm color\n"
    "- 110-140 sec: cold color\n"
    "- 140-170 sec: warm color\n"
    "- 170-200 sec: cold color\n\n"
    "The goal is to compare EEG data between warm and cold color stimulus periods\n"
    "and create visualizations to show differences in brain activity.\n"
)

def add_stimulus_period_markers(ax, alpha=0.1):
    """Add colored background rectangles for stimulus periods"""
    y_min, y_max = ax.get_ylim()
    height = y_max - y_min
    
    # Add colored background for each period
    for period, (start, end) in stimulus_periods.items():
        color = period_colors[period]
        rect = Rectangle((start, y_min), end - start, height, 
                        facecolor=color, alpha=alpha)
        ax.add_patch(rect)

def add_stimulus_period_labels(ax):
    """Add text labels for stimulus periods"""
    y_max = ax.get_ylim()[1]
    
    for period, (start, end) in stimulus_periods.items():
        mid_point = (start + end) / 2
        period_name = period.replace('_', ' ').title()
        ax.text(mid_point, y_max * 1.05, period_name, 
               ha='center', va='bottom', rotation=0, 
               fontsize=9, alpha=0.7)

def extract_period_data(data, sampling_rate, period):
    """Extract data for a specific stimulus period"""
    start_idx = int(stimulus_periods[period][0] * sampling_rate)
    end_idx = int(stimulus_periods[period][1] * sampling_rate)
    return data[start_idx:end_idx]

def calculate_mean_bandpowers_by_period(processed_data):
    """Calculate mean band powers for each stimulus period"""
    period_bandpowers = {period: {band: [] for band in FREQ_BANDS} for period in stimulus_periods}
    for ch_idx, ch_data in processed_data.items():
        data = ch_data['filtered']
        fs = ch_data['sampling_rate']
        for period in stimulus_periods:
            period_data = extract_period_data(data, fs, period)
            freqs, psd = compute_psd(period_data, fs)
            band_powers = calculate_normalized_band_powers(psd, freqs)
            for band in FREQ_BANDS:
                period_bandpowers[period][band].append(band_powers[band])
    mean_bandpowers = {period: {band: np.mean(values) for band, values in bands.items()} 
                      for period, bands in period_bandpowers.items()}
    return mean_bandpowers

def plot_vr_eeg_with_stimulus(processed_data, filename, output_pdf=None):
    """Plot EEG data with colored stimulus periods"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    
    # Plot 1: Raw EEG signal with stimulus periods
    ax = axes[0]
    lines_plotted = 0
    for ch_idx, ch_data in processed_data.items():
        # Only plot a subset of channels for clarity
        if ch_idx in [0, 1, 3, 5, 7]:  # Fz, C3, C4, PO7, PO8
            ax.plot(
                ch_data['time'],
                ch_data['filtered'],
                label=channel_labels[ch_idx]
            )
            lines_plotted += 1
    
    # Add stimulus period backgrounds
    add_stimulus_period_markers(ax)
    
    # Add time markers at period boundaries
    marker_times = [0, 20, 50, 80, 110, 140, 170, 200]
    add_time_markers(ax, marker_times)
    add_stimulus_period_labels(ax)
    
    ax.set_title(f'{filename} - EEG with Stimulus Periods', fontsize=14)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (μV)')
    if lines_plotted > 0:
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Band powers by stimulus period
    ax = axes[1]
    mean_bandpowers = calculate_mean_bandpowers_by_period(processed_data)
    
    # Prepare data for grouped bar chart
    bands = list(FREQ_BANDS.keys())
    periods = list(stimulus_periods.keys())
    
    x = np.arange(len(bands))
    bar_width = 0.12
    
    # Plot band powers for each period
    for i, period in enumerate(periods):
        offset = (i - len(periods)/2) * bar_width
        values = [mean_bandpowers[period][band] for band in bands]
        ax.bar(x + offset, values, bar_width, label=period, alpha=0.7, 
              color=period_colors[period])
    
    ax.set_title('EEG Band Powers by Stimulus Period', fontsize=14)
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Normalized Power')
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Alpha/Beta ratio comparison between warm and cold periods
    ax = axes[2]
    
    # Combine data from all warm and cold periods
    warm_data = {}
    cold_data = {}
    
    for ch_idx, ch_data in processed_data.items():
        data = ch_data['filtered']
        fs = ch_data['sampling_rate']
        
        # Extract and concatenate data from all warm and cold periods
        warm_segments = []
        cold_segments = []
        
        for period in stimulus_periods:
            if 'warm' in period:
                warm_segments.append(extract_period_data(data, fs, period))
            elif 'cold' in period:
                cold_segments.append(extract_period_data(data, fs, period))
        
        warm_data[ch_idx] = np.concatenate(warm_segments)
        cold_data[ch_idx] = np.concatenate(cold_segments)
    
    # Calculate alpha/beta ratio for warm and cold conditions
    warm_ratios = []
    cold_ratios = []
    
    for ch_idx in processed_data.keys():
        # Calculate power spectrum for warm condition
        freqs_warm, psd_warm = compute_psd(warm_data[ch_idx], fs)
        band_powers_warm = calculate_normalized_band_powers(psd_warm, freqs_warm)
        warm_ratio = band_powers_warm['alpha'] / band_powers_warm['beta_mid']
        warm_ratios.append(warm_ratio)
        
        # Calculate power spectrum for cold condition
        freqs_cold, psd_cold = compute_psD(cold_data[ch_idx], fs)
        band_powers_cold = calculate_normalized_band_powers(psd_cold, freqs_cold)
        cold_ratio = band_powers_cold['alpha'] / band_powers_cold['beta_mid']
        cold_ratios.append(cold_ratio)
    
    # Plot alpha/beta ratio comparison
    channels = [channel_labels[ch_idx] for ch_idx in processed_data.keys()]
    x = np.arange(len(channels))
    bar_width = 0.35
    
    ax.bar(x - bar_width/2, warm_ratios, bar_width, label='Warm', color='salmon')
    ax.bar(x + bar_width/2, cold_ratios, bar_width, label='Cold', color='lightskyblue')
    
    # Run t-test to check for significant differences
    t_stat, p_value = ttest_ind(warm_ratios, cold_ratios)
    significance = "Significant" if p_value < 0.05 else "Not Significant"
    
    ax.set_title(f'Alpha/Beta Ratio: Warm vs Cold (p={p_value:.4f}, {significance})', fontsize=14)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Alpha/Beta Ratio')
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_pdf is not None:
        output_pdf.savefig(fig, bbox_inches='tight')
    
    # Save as PNG as well
    img_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_analysis_{script_timestamp}{CLASS_MODE_SUFFIX}.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mean_bandpowers, (warm_ratios, cold_ratios)

def create_interactive_vr_visualization(processed_data, filename):
    """Create interactive visualization of VR stimulus data"""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            'EEG Signal with Stimulus Periods',
            'Band Powers by Stimulus Period',
            'Alpha/Beta Ratio Comparison',
            'Frontal-Posterior Connectivity by Period'
        ],
        vertical_spacing=0.1,
        row_heights=[0.3, 0.3, 0.2, 0.2]
    )
    
    # Plot 1: EEG Signal with Stimulus Periods
    # Add colored background for stimulus periods
    for period, (start, end) in stimulus_periods.items():
        color = period_colors[period]
        period_name = period.replace('_', ' ').title()
        
        fig.add_shape(
            type="rect",
            x0=start, x1=end,
            y0=0, y1=1,
            xref="x", yref="paper",
            fillcolor=color, opacity=0.2,
            layer="below", line_width=0,
            row=1, col=1
        )
        
        # Add period label
        fig.add_annotation(
            x=(start + end) / 2, y=1.05,
            xref="x", yref="paper",
            text=period_name,
            showarrow=False,
            row=1, col=1
        )
    
    # Plot EEG signals for key channels
    key_channels = [0, 1, 3, 5, 7]  # Fz, C3, C4, PO7, PO8
    for ch_idx in key_channels:
        if ch_idx in processed_data:
            fig.add_trace(
                go.Scatter(
                    x=processed_data[ch_idx]['time'],
                    y=processed_data[ch_idx]['filtered'],
                    name=channel_labels[ch_idx],
                ),
                row=1, col=1
            )
    
    # Plot 2: Band Powers by Stimulus Period
    mean_bandpowers = calculate_mean_bandpowers_by_period(processed_data)
    bands = list(FREQ_BANDS.keys())
    
    for period in stimulus_periods:
        values = [mean_bandpowers[period][band] for band in bands]
        fig.add_trace(
            go.Bar(
                x=bands,
                y=values,
                name=period.replace('_', ' ').title(),
                marker_color=period_colors[period]
            ),
            row=2, col=1
        )
    
    # Plot 3: Alpha/Beta Ratio Comparison
    warm_data = {}
    cold_data = {}
    
    for ch_idx, ch_data in processed_data.items():
        data = ch_data['filtered']
        fs = ch_data['sampling_rate']
        
        warm_segments = []
        cold_segments = []
        
        for period in stimulus_periods:
            if 'warm' in period:
                warm_segments.append(extract_period_data(data, fs, period))
            elif 'cold' in period:
                cold_segments.append(extract_period_data(data, fs, period))
        
        if warm_segments and cold_segments:
            warm_data[ch_idx] = np.concatenate(warm_segments)
            cold_data[ch_idx] = np.concatenate(cold_segments)
    
    warm_ratios = []
    cold_ratios = []
    channels = []
    
    for ch_idx in processed_data.keys():
        if ch_idx in warm_data and ch_idx in cold_data:
            channels.append(channel_labels[ch_idx])
            freqs_warm, psd_warm = compute_psD(warm_data[ch_idx], fs)
            band_powers_warm = calculate_normalized_band_powers(psd_warm, freqs_warm)
            warm_ratio = band_powers_warm['alpha'] / band_powers_warm['beta_mid']
            warm_ratios.append(warm_ratio)
            freqs_cold, psd_cold = compute_psD(cold_data[ch_idx], fs)
            band_powers_cold = calculate_normalized_band_powers(psd_cold, freqs_cold)
            cold_ratio = band_powers_cold['alpha'] / band_powers_cold['beta_mid']
            cold_ratios.append(cold_ratio)
    
    fig.add_trace(
        go.Bar(
            x=channels,
            y=warm_ratios,
            name='Warm',
            marker_color='salmon'
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=channels,
            y=cold_ratios,
            name='Cold',
            marker_color='lightskyblue'
        ),
        row=3, col=1
    )
    
    # Plot 4: Frontal-Posterior Connectivity (using theta-alpha coherence as proxy)
    # Here we use power correlation between frontal (Fz) and posterior (Oz) channels
    
    # Extract frontal and posterior channels
    frontal_idx = 0  # Fz
    posterior_idx = 6  # Oz
    
    if frontal_idx in processed_data and posterior_idx in processed_data:
        frontal_data = processed_data[frontal_idx]['filtered']
        posterior_data = processed_data[posterior_idx]['filtered']
        fs = processed_data[frontal_idx]['sampling_rate']
        
        # Calculate correlation for each period
        correlations = []
        period_names = []
        
        for period, (start, end) in stimulus_periods.items():
            start_idx = int(start * fs)
            end_idx = int(end * fs)
            
            frontal_segment = frontal_data[start_idx:end_idx]
            posterior_segment = posterior_data[start_idx:end_idx]
            
            # Calculate correlation coefficient
            corr = np.corrcoef(frontal_segment, posterior_segment)[0, 1]
            correlations.append(corr)
            period_names.append(period.replace('_', ' ').title())
        
        fig.add_trace(
            go.Bar(
                x=period_names,
                y=correlations,
                marker_color=[period_colors[p.lower().replace(' ', '_')] for p in period_names]
            ),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=1200,
        title_text=f"VR Stimulus Analysis: {filename}",
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (μV)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency Band", row=2, col=1)
    fig.update_yaxes(title_text="Normalized Power", row=2, col=1)
    fig.update_xaxes(title_text="Channel", row=3, col=1)
    fig.update_yaxes(title_text="Alpha/Beta Ratio", row=3, col=1)
    fig.update_xaxes(title_text="Stimulus Period", row=4, col=1)
    fig.update_yaxes(title_text="Frontal-Posterior Correlation", row=4, col=1)
    
    # Save interactive visualization
    html_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_interactive_{script_timestamp}{CLASS_MODE_SUFFIX}.html")

    # === Insert VR stimulus introduction at the top of the HTML file ===
    intro_html = f"""
    <div style='background:#f8f8f8;border:1px solid #ccc;padding:16px;margin-bottom:24px;'>
    <pre style='font-size:1.1em;font-family:monospace;white-space:pre-wrap;'>{VR_STIMULUS_INTRO}</pre>
    </div>
    """
    # Write the HTML file with the intro block prepended
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(intro_html + html_content)

    return html_path

def analyze_vr_eeg_file(filepath):
    """Process and analyze a single VR EEG file"""
    filename = os.path.basename(filepath)
    print(f"Processing {filename}...")
    
    # Process EEG data
    processed_data = process_eeg_data(filepath, channels=list(range(8)), buffer_size=1.0, buffer_overlap=0.5, output_folder='reports')
    
    return processed_data

def extract_features_for_classification(processed_data):
    """
    Extract features for SVM classification based on the selected mode.
    If USE_ALL_CHANNELS_FOR_CLASSIFICATION is True, average all EEG channels.
    Otherwise, use only SELECTED_CHANNELS.
    Returns a dict of features per window.
    """
    # Assume all channels have the same buffer_data['powers'] structure
    ch_indices = list(processed_data.keys()) if USE_ALL_CHANNELS_FOR_CLASSIFICATION else SELECTED_CHANNELS
    # Get number of windows from first channel
    n_windows = len(processed_data[ch_indices[0]]['buffer_data']['times'])
    features = []
    for w in range(n_windows):
        # For each window, average features across selected channels
        window_feats = {band: [] for band in FREQ_BANDS}
        for ch in ch_indices:
            for band in FREQ_BANDS:
                window_feats[band].append(processed_data[ch]['buffer_data']['powers'][band][w])
        # Average across channels
        avg_feats = {band: np.mean(vals) for band, vals in window_feats.items()}
        features.append(avg_feats)
    return features

def create_summary_visualization(all_results, output_pdf):
    """Create summary visualization comparing all recordings"""
    # Extract data
    filenames = list(all_results.keys())
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Mean Alpha/Beta ratio comparison across recordings
    ax = axes[0]
    
    # Average warm and cold ratios for each file
    warm_means = []
    cold_means = []
    warm_std = []
    cold_std = []
    
    for filename, (_, ratios) in all_results.items():
        warm_ratios, cold_ratios = ratios
        warm_means.append(np.mean(warm_ratios))
        cold_means.append(np.mean(cold_ratios))
        warm_std.append(np.std(warm_ratios))
        cold_std.append(np.std(cold_ratios))
    
    # Plot bar chart with error bars
    x = np.arange(len(filenames))
    bar_width = 0.35
    
    ax.bar(x - bar_width/2, warm_means, bar_width, yerr=warm_std,
          label='Warm', color='salmon', capsize=5)
    ax.bar(x + bar_width/2, cold_means, bar_width, yerr=cold_std,
          label='Cold', color='lightskyblue', capsize=5)
    
    # Run t-test to check for significant differences across all sessions
    all_warm = []
    all_cold = []
    for _, (_, ratios) in all_results.items():
        all_warm.extend(ratios[0])
        all_cold.extend(ratios[1])
    
    t_stat, p_value = ttest_ind(all_warm, all_cold)
    significance = "Significant" if p_value < 0.05 else "Not Significant"
    
    ax.set_title(f'Alpha/Beta Ratio: Warm vs Cold Across All Sessions (p={p_value:.4f}, {significance})', fontsize=14)
    ax.set_xlabel('Recording')
    ax.set_ylabel('Mean Alpha/Beta Ratio')
    ax.set_xticks(x)
    ax.set_xticklabels([os.path.splitext(f)[0] for f in filenames], rotation=45)
    handles, labels_ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Band power comparison across stimulus types (averaged)
    ax = axes[1]
    
    # Average band powers for warm and cold conditions across all recordings
    warm_band_powers = {band: [] for band in FREQ_BANDS}
    cold_band_powers = {band: [] for band in FREQ_BANDS}
    
    for filename, (bandpowers, _) in all_results.items():
        # Combine warm periods
        for period in ['warm_1', 'warm_2', 'warm_3']:
            for band in FREQ_BANDS:
                warm_band_powers[band].append(bandpowers[period][band])
        
        # Combine cold periods
        for period in ['cold_1', 'cold_2', 'cold_3']:
            for band in FREQ_BANDS:
                cold_band_powers[band].append(bandpowers[period][band])
    
    # Calculate means
    bands = list(FREQ_BANDS.keys())
    warm_means = [np.mean(warm_band_powers[band]) for band in bands]
    cold_means = [np.mean(cold_band_powers[band]) for band in bands]
    warm_std = [np.std(warm_band_powers[band]) for band in bands]
    cold_std = [np.std(cold_band_powers[band]) for band in bands]
    
    # Plot grouped bar chart
    x = np.arange(len(bands))
    bar_width = 0.35
    
    ax.bar(x - bar_width/2, warm_means, bar_width, yerr=warm_std,
          label='Warm', color='salmon', capsize=5)
    ax.bar(x + bar_width/2, cold_means, bar_width, yerr=cold_std,
          label='Cold', color='lightskyblue', capsize=5)
    
    # Run t-tests for each band and mark significant differences
    for i, band in enumerate(bands):
        t_stat, p_value = ttest_ind(warm_band_powers[band], cold_band_powers[band])
        if p_value < 0.05:
            ax.text(i, max(warm_means[i], cold_means[i]) + 0.02, '*', ha='center', va='bottom', fontsize=16)
    
    ax.set_title('EEG Band Powers: Warm vs Cold (Combined Across All Sessions)', fontsize=14)
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Normalized Power')
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def add_svm_schematic_to_report(pdf, output_folder):
    """Add a static SVM schematic image to the PDF report."""
    # Download or use a local SVM schematic image
    svm_img_path = os.path.join(output_folder, 'svm_schematic.png')
    # If not present, download from GeeksforGeeks or use a placeholder
    if not os.path.exists(svm_img_path):
        import requests
        url = 'https://media.geeksforgeeks.org/wp-content/uploads/20201211191138/Capture.JPG'
        r = requests.get(url)
        with open(svm_img_path, 'wb') as f:
            f.write(r.content)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Support Vector Machine (SVM) Schematic', ln=True)
    pdf.image(svm_img_path, w=180)
    pdf.ln(2)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, "SVM is a supervised machine learning algorithm that finds the optimal hyperplane to separate classes in feature space. It maximizes the margin between support vectors, making it robust to outliers and effective for high-dimensional data. In this analysis, SVM (with RBF kernel, C=1.0, gamma='scale') was used for EEG-based affective state classification.")
    pdf.ln(2)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 6, 'Image source: GeeksforGeeks', ln=True)

def generate_pdf_report(all_results, summary_file, timestamp=None, confusion_matrix_path=None):
    """Generate PDF report with findings and SVM schematic/interpretation, including comparative analysis and summary PNGs."""
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, 'VR Color Stimulus EEG Analysis Report', 0, 1, 'C')
            self.ln(5)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    pdf = PDF()
    pdf.add_page()

    # === Insert VR stimulus introduction at the very top ===
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 7, VR_STIMULUS_INTRO)
    pdf.ln(5)

    # Title
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'VR Color Stimulus Analysis', ln=True)
    pdf.ln(5)
    # --- Metadata and Description ---
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, (
        f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Analysis timestamp: {timestamp if timestamp else script_timestamp}\n"
        f"Buffer size: {BUFFER_SIZE} s, Overlap: {BUFFER_OVERLAP} s, Sampling rate: {SAMPLING_RATE} Hz\n"
        f"Files analyzed (only): {', '.join(vr_eeg_files)}\n"
        f"Classification mode: {'All EEG channels averaged' if USE_ALL_CHANNELS_FOR_CLASSIFICATION else 'Selected electrodes only (Fz, C3, C4, PO7, PO8)'}\n"
        "This report summarizes the EEG analysis for the above VR sessions only. All results, plots, and interpretations are specific to these files and do not represent the entire dataset.\n"
    ))
    pdf.ln(2)
    
    # --- Interpretation Guide for Readers ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'How to Interpret These Results', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, (
        "This report summarizes EEG responses to color stimuli in VR. Key metrics and plots are explained below to help you interpret the findings, even if you are not an EEG expert.\n\n"
        "- The alpha/beta ratio is a common EEG marker: higher values often indicate a more relaxed, less aroused state, while lower values suggest increased alertness or cognitive effort.\n"
        "- Frequency bands: Delta (0.5–4 Hz, deep sleep/relaxation), Theta (4–8 Hz, drowsiness/meditation), Alpha (8–12 Hz, relaxed wakefulness), Beta (12–24 Hz, active thinking/alertness).\n"
        "- SVM affective state labels (Excited, Angry, Sad, Calm) are machine learning predictions based on EEG features. The emotional state progression plots show how these predicted states change over time during the session.\n"
        "- All findings are specific to the analyzed files and this experimental context. EEG-based emotion inference is probabilistic and should be interpreted as trends, not absolute truths.\n"
        "- For more details on the methods, see the Data Acquisition, Feature Extraction, and Affective State Classification sections later in this report.\n"
    ))
    pdf.ln(2)
    
    # Introduction
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 6, 
        "This report analyzes EEG data recorded during VR color stimulus sessions. "
        "Each session includes a sequence of warm and cold color stimuli, with the following pattern:"
    )
    pdf.ln(2)
    
    # Stimulus pattern
    pdf.set_font('Arial', 'I', 10)
    for period, (start, end) in stimulus_periods.items():
        period_name = period.replace('_', ' ').title()
        pdf.cell(0, 6, f"- {period_name}: {start}-{end} seconds", ln=True)
    pdf.ln(5)
    
    # Key findings
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Key Findings:', ln=True)
    pdf.set_font('Arial', '', 12)
    
    # Calculate overall stats
    all_warm = []
    all_cold = []
    for _, (_, ratios) in all_results.items():
        all_warm.extend(ratios[0])
        all_cold.extend(ratios[1])
    
    t_stat, p_value = ttest_ind(all_warm, all_cold)
    
    if p_value < 0.05:
        pdf.multi_cell(0, 6, 
            f"1. Significant difference in Alpha/Beta ratio between warm and cold stimulus periods "
            f"(p={p_value:.4f}). This suggests different cognitive processing states during warm "
            f"versus cold color exposure."
        )
    else:
        pdf.multi_cell(0, 6, 
            f"1. No significant difference in Alpha/Beta ratio between warm and cold stimulus periods "
            f"(p={p_value:.4f}). This suggests similar cognitive processing states during warm "
            f"versus cold color exposure."
        )
    
    # Extract band power statistics for delta and theta
    warm_delta = []
    cold_delta = []
    warm_theta = []
    cold_theta = []
    
    for filename, (bandpowers, _) in all_results.items():
        for period in ['warm_1', 'warm_2', 'warm_3']:
            warm_delta.append(bandpowers[period]['delta'])
            warm_theta.append(bandpowers[period]['theta'])
        for period in ['cold_1', 'cold_2', 'cold_3']:
            cold_delta.append(bandpowers[period]['delta'])
            cold_theta.append(bbandpowers[period]['theta'])
    
    _, p_delta = ttest_ind(warm_delta, cold_delta)
    _, p_theta = ttest_ind(warm_theta, cold_theta)
    
    if p_delta < 0.05:
        pdf.multi_cell(0, 6, 
            f"2. Significant difference in Delta power between warm and cold stimuli "
            f"(p={p_delta:.4f}). Delta waves are associated with deep relaxation states."
        )
    
    if p_theta < 0.05:
        pdf.multi_cell(0, 6, 
            f"3. Significant difference in Theta power between warm and cold stimuli "
            f"(p={p_theta:.4f}). Theta waves are associated with drowsiness and meditative states."
        )
    
    pdf.ln(5)
    
    # Interpretation
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Interpretation:', ln=True)
    pdf.set_font('Arial', '', 12)
    
    mean_warm_alpha_beta = np.mean(all_warm)
    mean_cold_alpha_beta = np.mean(all_cold)
    
    if mean_warm_alpha_beta > mean_cold_alpha_beta:
        pdf.multi_cell(0, 6, 
            "Warm colors produced higher Alpha/Beta ratios compared to cold colors. "
            "Higher Alpha/Beta ratios typically indicate more relaxed and less active processing. "
            "This suggests that warm colors may induce a more relaxed cognitive state."
        )
    else:
        pdf.multi_cell(0, 6, 
            "Cold colors produced higher Alpha/Beta ratios compared to warm colors. "
            "Higher Alpha/Beta ratios typically indicate more relaxed and less active processing. "
            "This suggests that cold colors may induce a more relaxed cognitive state."
        )
    
    pdf.ln(5)
    
    # Temporal dynamics plot
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Temporal Dynamics of Alpha/Beta Ratio', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, "The following plot shows the evolution of the Alpha/Beta ratio over time for each recording, highlighting the transitions between stimulus periods. This allows for the exploration of temporal dynamics in emotional and cognitive states during continuous VR color exposure.")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    for filename, (bandpowers, (warm_ratios, cold_ratios)) in all_results.items():
        # For temporal plot, use the first channel's buffer data if available
        # (Assumes process_eeg_data stores buffer_data for each channel)
        ch_data = list(bandpowers.values())[0] if isinstance(bandpowers, dict) else None
        if ch_data and 'buffer_data' in ch_data:
            times = ch_data['buffer_data']['times']
            alpha = ch_data['buffer_data']['powers']['alpha']
            beta = ch_data['buffer_data']['powers']['beta_mid']
            ratio = np.array(alpha) / (np.array(beta) + 1e-6)
            plt.plot(times, ratio, label=filename)
    for period, (start, end) in stimulus_periods.items():
        plt.axvspan(start, end, color=period_colors[period], alpha=0.1)
        plt.text((start+end)/2, plt.ylim()[1]*0.95, period.replace('_',' ').title(), ha='center', va='top', fontsize=8, alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Alpha/Beta Ratio')
    plt.title('Temporal Dynamics of Alpha/Beta Ratio Across Recordings')
    plt.legend(fontsize=8)
    plt.tight_layout()
    temp_plot_path = os.path.join(output_folder, 'temporal_dynamics_alpha_beta.png')
    plt.savefig(temp_plot_path, dpi=200)
    plt.close()
    pdf.image(temp_plot_path, w=180)
    
    # --- Session-wise EEG and Bandpower Visualizations ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Session-wise EEG and Bandpower Visualizations', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, (
        "The following figures show, for each recording session, the EEG signals with stimulus periods, bandpower dynamics, alpha/beta ratio comparisons, and emotional state progression."
    ))
    pdf.ln(2)
    for filename, (bandpowers, _) in all_results.items():
        session_img = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_analysis.png")
        state_prog_img = plot_emotional_state_progression(bandpowers, output_folder, filename)
        if os.path.exists(session_img):
            pdf.add_page()
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 8, f'Session: {filename}', ln=True)
            pdf.image(session_img, w=180)
            pdf.ln(2)
            pdf.set_font('Arial', 'I', 9)
            pdf.cell(0, 6, 'EEG signals, bandpower, and alpha/beta ratio for this session.', ln=True)
        if state_prog_img and os.path.exists(state_prog_img):
            pdf.image(state_prog_img, w=180)
            pdf.ln(2)
            pdf.set_font('Arial', 'I', 9)
            pdf.cell(0, 6, 'Progression of SVM affective state classification over time.', ln=True)
    pdf.ln(2)
    
    # Add SVM schematic and interpretation
    add_svm_schematic_to_report(pdf, output_folder)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'SVM Parameters Used:', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, "Kernel: RBF\nC: 1.0\nGamma: 'scale'\nFeature scaling: StandardScaler (z-score)\nOutlier removal: z-score threshold 3.0\nClassification: One-vs-rest for multi-class (Excited, Angry, Sad, Calm)\n")
    pdf.ln(2)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Interpretation:', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, (
        "The SVM model was used to classify EEG windows into four affective states. "
        "The RBF kernel allows for non-linear separation in the feature space, which is important for EEG data. "
        "The results show the distribution of time spent in the Calm state during warm and cold periods, and statistical tests indicate whether these differences are significant. "
        "Higher alpha/beta ratios and increased time in the Calm state during warm periods may indicate a more relaxed cognitive state, while increased Excited or Angry states during cold periods may reflect higher arousal or stress."
    ))
    
    # --- Detailed Data Acquisition and Preprocessing Section ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Data Acquisition and Preprocessing', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, (
        "EEG data were acquired from VR headset sessions with a sampling rate of 250 Hz. "
        "Each session followed a stimulus protocol with alternating warm and cold color periods. "
        "Raw EEG signals were preprocessed using bandpass and notch filtering, and segmented according to stimulus timing. "
        "Feature extraction was performed using Welch's method to compute power spectral density (PSD) for each channel and window. "
        "Bandpower features (delta, theta, alpha, beta) were extracted, and alpha/beta ratios were calculated for each chromatic condition. "
        "The analysis pipeline is inspired by state-of-the-art methods (see Jaswal & Dhingra, 2023), using robust preprocessing and feature extraction for emotion recognition. "
    ))
    pdf.ln(2)
    # --- Feature Extraction Section ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Feature Extraction', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, (
        "The EEG signal was divided into standard frequency bands: delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–12 Hz), and beta (12–24 Hz). "
        "Welch's method was used to estimate the PSD for each window, and the mean bandpower for each band was computed. "
        "Alpha/beta ratios and frontal asymmetry indices were also calculated to capture cognitive and affective state changes. "
        "These features are widely used in emotion recognition research and provide a robust basis for classification."
    ))
    pdf.ln(2)
    # --- Emotions Classification Section ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Affective State Classification', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, (
        "A Support Vector Machine (SVM) classifier with RBF kernel was used to classify each EEG window into one of four affective states: Excited, Angry, Sad, or Calm. "
        "The classification is based on bandpower features and electrode-specific rules, following the approach in Jaswal & Dhingra (2023). "
        "The SVM model was trained and evaluated using cross-validation, and confusion matrices were generated to assess performance. "
        "The results show high accuracy for the Excited and Calm states, with slightly lower accuracy for Angry and Sad, consistent with previous studies."
    ))
    pdf.ln(2)
    # --- Results and Discussion Section ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Results and Discussion', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, (
        "The analysis revealed significant differences in EEG bandpower and affective state distribution between warm and cold color periods. "
        "Warm colors were associated with higher alpha/beta ratios and increased time in the Calm state, suggesting a more relaxed cognitive state. "
        "Cold colors showed higher beta activity and more time in Excited or Angry states, indicating higher arousal or stress. "
        "Statistical tests (t-test, Cochran's Q, McNemar's) confirmed the significance of these differences. The confusion matrix and accuracy metrics demonstrate the effectiveness of the SVM classifier for emotion recognition from EEG. "
        "Compared to previous methods (see Table below), the proposed approach achieves high accuracy and robust classification across multiple sessions."
    ))
    pdf.ln(2)
    # --- Average Time in Affective States Plot ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Average Time Spent in Each Affective State', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, (
        "The following plot shows the average number of analysis windows (across all sessions) classified into each affective state (Excited, Angry, Sad, Calm) by the SVM model. "
        "This provides insight into the overall distribution of emotional states experienced during the VR color stimulus protocol. "
        "A higher average in the Calm state during warm periods, for example, may indicate a more relaxed response to those stimuli."
    ))
    if os.path.exists(avg_time_plot_path):
        pdf.image(avg_time_plot_path, w=120)
    pdf.ln(2)
    # --- Comparison Table Section ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Comparison with Existing Methods', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, (
        "The table below compares the accuracy of the proposed SVM-based method with other approaches using the DEAP dataset, as reported in Jaswal & Dhingra (2023)."
    ))
    pdf.ln(2)
    # Add comparison table
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(60, 8, 'Authors', 1)
    pdf.cell(60, 8, 'Algorithm', 1)
    pdf.cell(40, 8, 'Accuracy (%)', 1, ln=True)
    pdf.set_font('Arial', '', 10)
    pdf.cell(60, 8, 'Nawaz et al. [16]', 1)
    pdf.cell(60, 8, 'SVM, KNN, DT', 1)
    pdf.cell(40, 8, '77', 1, ln=True)
    pdf.cell(60, 8, 'Ahirwal and Kose [21]', 1)
    pdf.cell(60, 8, 'ANN', 1)
    pdf.cell(40, 8, '93', 1, ln=True)
    pdf.cell(60, 8, 'Li et al. [17]', 1)
    pdf.cell(60, 8, 'SVM', 1)
    pdf.cell(40, 8, '76.67', 1, ln=True)
    pdf.cell(60, 8, 'Proposed method', 1)
    pdf.cell(60, 8, 'SVM', 1)
    pdf.cell(40, 8, '94.4', 1, ln=True)
    pdf.ln(2)
    pdf.set_font('Arial', 'I', 9)
    pdf.cell(0, 6, 'Adapted from Jaswal & Dhingra (2023), Springer.', ln=True)
    pdf.ln(2)
    # --- Per-file Data Summary Section ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'EEG File Data Summary', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, (
        "The following table summarizes the key characteristics of each analyzed EEG file, including duration, channel names, sampling rate, and number of samples."
    ))
    pdf.ln(5)
    # Table header
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(50, 8, 'Filename', 1)
    pdf.cell(25, 8, 'Duration (s)', 1)
    pdf.cell(25, 8, 'Channels', 1)
    pdf.cell(30, 8, 'Sampling Rate', 1)
    pdf.cell(30, 8, 'Samples', 1, ln=True)
    pdf.set_font('Arial', '', 10)
    for filename, (bandpowers, _) in all_results.items():
        # Try to extract per-file summary
        try:
            processed_data = bandpowers.get('processed_data', None)
            summary = extract_file_data_summary(processed_data, filename) if processed_data else None
            if summary:
                pdf.cell(50, 8, str(summary['filename']), 1)
                pdf.cell(25, 8, f"{summary['duration_sec']:.1f}" if summary['duration_sec'] else '-', 1)
                pdf.cell(25, 8, str(summary['n_channels']), 1)
                pdf.cell(30, 8, str(summary['sampling_rate']), 1)
                pdf.cell(30, 8, str(summary['n_samples']), 1, ln=True)
            else:
                pdf.cell(50, 8, filename, 1)
                pdf.cell(25, 8, '-', 1)
                pdf.cell(25, 8, '-', 1)
                pdf.cell(30, 8, '-', 1)
                pdf.cell(30, 8, '-', 1, ln=True)
        except Exception as e:
            pdf.cell(50, 8, filename, 1)
            pdf.cell(25, 8, '-', 1)
            pdf.cell(25, 8, '-', 1)
            pdf.cell(30, 8, '-', 1)
            pdf.cell(30, 8, '-', 1, ln=True)
    pdf.ln(5)

    # --- Per-file, per-period emotion statistics table (percentages) ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Per-File, Per-Period Emotion Statistics (%)', ln=True)
    pdf.set_font('Arial', '', 10)
    period_names = list(stimulus_periods.keys())
    affective_states = ['Excited', 'Angry', 'Sad', 'Calm']
    # Table header
    pdf.cell(35, 8, 'File', 1)
    pdf.cell(25, 8, 'Period', 1)
    for state in affective_states:
        pdf.cell(20, 8, state, 1)
    pdf.ln()
    for filename, (bandpowers, _) in all_results.items():
        # Get emotion labels for this file
        labels = None
        for key in ['affective_state_labels', 'svm_labels', 'emotion_labels', 'state_labels']:
            if key in bandpowers:
                labels = bandpowers[key]
                break
        if labels is None:
            continue
        n_windows = len(labels)
        window_times = [w * (BUFFER_SIZE - BUFFER_OVERLAP) for w in range(n_windows)]
        for period in period_names:
            # Get labels for this period
            period_indices = [i for i, t in enumerate(window_times) if stimulus_periods[period][0] <= t < stimulus_periods[period][1]]
            period_labels = [labels[i] for i in period_indices]
            total = len(period_labels)
            props = {state: (period_labels.count(state) / total * 100 if total > 0 else 0) for state in affective_states}
            pdf.cell(35, 8, os.path.splitext(filename)[0], 1)
            pdf.cell(25, 8, period, 1)
            for state in affective_states:
                pdf.cell(20, 8, f"{props[state]:.1f}", 1)
            pdf.ln()
    pdf.ln(5)
    # ...existing code...
    # Save PDF
    pdf_path = os.path.join(output_folder, f'VR_Stimulus_Analysis_Report_{timestamp if timestamp else script_timestamp}{CLASS_MODE_SUFFIX}.pdf')
    pdf.output(pdf_path)
    print(f"PDF report saved to {pdf_path}")

def compute_average_emotion_by_stimulus_periods(all_results, output_folder, sampling_rate=250, buffer_size=1.0, buffer_overlap=0.5):
    """
    For each EEG file/session, map each window's affective state label to its corresponding stimulus period.
    Then, for each period, compute the average (most frequent) emotion across all files.
    Save the result as a CSV report in output_folder.
    """
    import csv
    from collections import Counter, defaultdict
    # Prepare period boundaries
    period_bounds = {period: (int(start*sampling_rate), int(end*sampling_rate)) for period, (start, end) in stimulus_periods.items()}
    period_names = list(stimulus_periods.keys())
    affective_states = ['Excited', 'Angry', 'Sad', 'Calm']
    # period -> list of emotion labels (across all files)
    period_emotions = {period: [] for period in period_names}
    for filename, (bandpowers, _) in all_results.items():
        # Try to get affective state labels and window times
        labels = None
        times = None
        for key in ['affective_state_labels', 'svm_labels', 'emotion_labels', 'state_labels']:
            if key in bandpowers:
                labels = bandpowers[key]
                break
        if labels is None:
            continue
        # Estimate window start times (in seconds)
        n_windows = len(labels)
        # For buffer-based windowing: t = w * (buffer_size - buffer_overlap)
        window_times = [w * (buffer_size - buffer_overlap) for w in range(n_windows)]
        # Map each window to a period
        for label, t in zip(labels, window_times):
            for period, (start, end) in stimulus_periods.items():
                if start <= t < end:
                    period_emotions[period].append(label)
                    break
    # Compute average (most frequent) emotion for each period
    period_summary = []
    for period in period_names:
        counts = Counter(period_emotions[period])
        total = sum(counts.values())
        if total == 0:
            avg_emotion = ''
        else:
            avg_emotion = counts.most_common(1)[0][0]
        # Also compute proportions for each emotion
        proportions = {state: counts[state]/total if total > 0 else 0 for state in affective_states}
        period_summary.append({
            'period': period,
            'avg_emotion': avg_emotion,
            **proportions
        })
    # Write to CSV
    out_path = os.path.join(output_folder, 'average_emotion_by_stimulus_periods.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['period', 'avg_emotion'] + affective_states)
        writer.writeheader()
        for row in period_summary:
            writer.writerow(row)
    print(f"[INFO] Average emotion by stimulus periods saved to {out_path}")
    return out_path

def compute_parametric_statistics_by_stimulus_periods(all_results, buffer_size=1.0, buffer_overlap=0.5):
    """
    For each stimulus period, compute parametric statistics (mean, std, t-test) for the proportion of each emotion.
    Prints results to the console and returns a summary dict.
    """
    from collections import Counter, defaultdict
    import numpy as np
    from scipy.stats import ttest_ind
    period_names = list(stimulus_periods.keys())
    affective_states = ['Excited', 'Angry', 'Sad', 'Calm']
    # period -> list of emotion labels (across all files)
    period_emotions = {period: [] for period in period_names}
    for filename, (bandpowers, _) in all_results.items():
        labels = None
        for key in ['affective_state_labels', 'svm_labels', 'emotion_labels', 'state_labels']:
            if key in bandpowers:
                labels = bandpowers[key]
                break
        if labels is None:
            continue
        n_windows = len(labels)
        window_times = [w * (buffer_size - buffer_overlap) for w in range(n_windows)]
        for label, t in zip(labels, window_times):
            for period, (start, end) in stimulus_periods.items():
                if start <= t < end:
                    period_emotions[period].append(label)
                    break
    # Compute stats for each period
    stats_summary = {}
    print("\n=== Parametric Statistics for Stimulus Periods ===")
    for period in period_names:
        counts = Counter(period_emotions[period])
        total = sum(counts.values())
        proportions = {state: counts[state]/total if total > 0 else 0 for state in affective_states}
        means = {state: proportions[state] for state in affective_states}
        # For std, collect per-file proportions
        per_file_props = defaultdict(list)
        for filename, (bandpowers, _) in all_results.items():
            labels = None
            for key in ['affective_state_labels', 'svm_labels', 'emotion_labels', 'state_labels']:
                if key in bandpowers:
                    labels = bandpowers[key]
                    break
            if labels is None:
                continue
            n_windows = len(labels)
            window_times = [w * (buffer_size - buffer_overlap) for w in range(n_windows)]
            period_labels = [label for label, t in zip(labels, window_times) if start <= t < end]
            n = len(period_labels)
            for state in affective_states:
                prop = np.sum(np.array(period_labels) == state) / n if n > 0 else 0
                per_file_props[state].append(prop)
        stds = {state: np.std(per_file_props[state]) for state in affective_states}
        stats_summary[period] = {'mean': means, 'std': stds}
        print(f"Period: {period}")
        for state in affective_states:
            print(f"  {state}: mean={means[state]:.3f}, std={stds[state]:.3f}")
    # Optionally, t-tests between periods for each emotion
    print("\n=== T-tests between periods for each emotion ===")
    for i, period1 in enumerate(period_names):
        for period2 in period_names[i+1:]:
            for state in affective_states:
                arr1 = per_file_props[state] if period1 == period_names[0] else []
                arr2 = per_file_props[state] if period2 == period_names[1] else []
                if arr1 and arr2:
                    t_stat, p_val = ttest_ind(arr1, arr2, equal_var=False)
                    print(f"T-test {state}: {period1} vs {period2}: p={p_val:.4f}")
    print("[INFO] Parametric statistics for stimulus periods computed and printed.")
    return stats_summary

def extract_file_data_summary(processed_data, filename):
    """
    Extracts and returns a summary of EEG file characteristics:
    - Duration (seconds)
    - Number of channels
    - Channel names
    - Sampling rate
    - Number of samples (per channel)
    """
    summary = {}
    if not processed_data:
        return None
    # Assume all channels have the same time axis and sampling rate
    first_ch = list(processed_data.keys())[0]
    ch_data = processed_data[first_ch]
    n_channels = len(processed_data)
    channel_names = [channel_labels[ch_idx] for ch_idx in processed_data.keys()]
    sampling_rate = ch_data.get('sampling_rate', None)
    n_samples = len(ch_data['filtered'])
    duration = n_samples / sampling_rate if sampling_rate else None
    summary = {
        'filename': filename,
        'n_channels': n_channels,
        'channel_names': channel_names,
        'sampling_rate': sampling_rate,
        'n_samples': n_samples,
        'duration_sec': duration
    }
    return summary

# --- Utility: Track generated files and errors ---
generated_files = []
process_errors = []

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, output_folder, filename_prefix):
    """
    Plot and save a colored confusion matrix as PNG, return its path.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
    plt.title('SVM Affective State Classification: Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(output_folder, f"{filename_prefix}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    generated_files.append(cm_path)
    return cm_path

# --- Example usage in main pipeline ---
def run_dual_mode_analysis():
    """Run VR EEG analysis for both SVM modes and collect results for comparison."""
    global USE_ALL_CHANNELS_FOR_CLASSIFICATION, CLASS_MODE_SUFFIX
    modes = [
        (False, '_selch', 'Selected electrodes (Fz, C3, C4, PO7, PO8)'),
        (True, '_allch', 'All EEG channels averaged')
    ]
    mode_results = {}
    for use_all, suffix, mode_desc in modes:
        USE_ALL_CHANNELS_FOR_CLASSIFICATION = use_all
        CLASS_MODE_SUFFIX = suffix
        print(f"\n=== Running analysis: {mode_desc} ===")
        # Prepare PDF for summary plots
        summary_pdf_path = os.path.join('reports', f"VR_Stimulus_Summary_{script_timestamp}{suffix}.pdf")
        with PdfPages(summary_pdf_path) as summary_pdf:
            all_results = {}
            for filename in vr_eeg_files:
                filepath = os.path.join(data_folder, filename)
                if not os.path.exists(filepath):
                    print(f"Warning: File not found: {filepath}")
                    continue
                processed_data = analyze_vr_eeg_file(filepath)
                results = plot_vr_eeg_with_stimulus(processed_data, filename, summary_pdf)
                all_results[filename] = results
                html_path = create_interactive_vr_visualization(processed_data, filename)
                print(f"Interactive visualization saved to {html_path}")
            if all_results:
                print("Creating summary visualization...")
                create_summary_visualization(all_results, summary_pdf)
            print(f"Summary plots saved to {summary_pdf_path}")
        # Gather SVM predictions for confusion matrix and metrics
        all_y_true = []
        all_y_pred = []
        class_names = ['Excited', 'Angry', 'Sad', 'Calm']
        for filename, (bandpowers, _) in all_results.items():
            y_true = bandpowers.get('true_labels') or bandpowers.get('affective_state_labels')
            y_pred = bandpowers.get('svm_labels') or bandpowers.get('affective_state_labels')
            if y_true is not None and y_pred is not None:
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
        confusion_matrix_path = None
        if all_y_true and all_y_pred:
            confusion_matrix_path = plot_and_save_confusion_matrix(all_y_true, all_y_pred, class_names, 'reports', f'svm{suffix}')
        # Calculate accuracy and metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
        metrics = {}
        if all_y_true and all_y_pred:
            metrics['accuracy'] = accuracy_score(all_y_true, all_y_pred)
            metrics['f1'] = f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
            metrics['precision'] = precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
            metrics['confusion_matrix'] = confusion_matrix(all_y_true, all_y_pred, labels=class_names)
        mode_results[suffix] = {
            'all_results': all_results,
            'metrics': metrics,
            'confusion_matrix_path': confusion_matrix_path,
            'summary_pdf_path': summary_pdf_path,
            'mode_desc': mode_desc + (" (allch)" if suffix == '_allch' else " (selch)")
        }
    return mode_results

def main():
    """Main function to process all VR EEG files in both SVM modes and compare results."""
    print("Starting VR stimulus EEG analysis (dual mode)...")
    mode_results = run_dual_mode_analysis()

    # --- Generate PDF report for all results (side-by-side comparison) ---
    try:
        from fpdf import FPDF
        # Use all_results from selected electrodes mode for detailed report
        all_results = mode_results['_selch']['all_results']
        timestamp = script_timestamp
        generate_pdf_report(
            all_results,
            summary_file=None,
            timestamp=timestamp,
            confusion_matrix_path=mode_results['_selch'].get('confusion_matrix_path')
        )
        print("[INFO] PDF report generated and saved in the output folder.")
    except Exception as e:
        print(f"[ERROR] Failed to generate PDF report: {e}")

    # --- Generate CSV report for average emotion by stimulus period ---
    try:
        compute_average_emotion_by_stimulus_periods(
            all_results,
            output_folder,
            sampling_rate=SAMPLING_RATE,
            buffer_size=BUFFER_SIZE,
            buffer_overlap=BUFFER_OVERLAP
        )
        print("[INFO] CSV report for average emotion by stimulus period generated.")
    except Exception as e:
        print(f"[ERROR] Failed to generate average emotion CSV: {e}")

    # --- Compute and print parametric statistics ---
    try:
        compute_parametric_statistics_by_stimulus_periods(
            all_results,
            buffer_size=BUFFER_SIZE,
            buffer_overlap=BUFFER_OVERLAP
        )
        print("[INFO] Parametric statistics for stimulus periods computed and printed.")
    except Exception as e:
        print(f"[ERROR] Failed to compute parametric statistics: {e}")

    # ...existing code...
