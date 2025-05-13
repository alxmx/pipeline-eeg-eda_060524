# EEG-EDA Signal Processing Pipeline

## Overview
This project implements a comprehensive signal processing pipeline for analyzing EEG (Electroencephalogram) and EDA (Electrodermal Activity) data. It provides real-time visualization, spectral analysis, and mental state classification through a interactive web dashboard.

## Features

### 1. Signal Processing
- **EEG Processing**
  - Sampling Rate: 250 Hz
  - Channels: Fp1, Fp2, F3, F4, C3, C4, P3, P4 (10-20 system)
  - Filters: 
    - Notch Filter (50 Hz)
    - Bandpass Filter (0.5-40 Hz)
  - Window Length: 1 second (250 samples)
  - Window Overlap: 96% (240/250 samples)

- **EDA Processing**
  - Sampling Rate: 500 Hz
  - Window Length: 1 second (500 samples)
  - Window Overlap: 90% (450/500 samples)
  - Components: Tonic (SCL) and Phasic (SCR)

### 2. Analysis Capabilities
- **Spectral Analysis**
  - Band Power Analysis:
    - Delta (0.5-4 Hz)
    - Theta (4-8 Hz)
    - Alpha (8-13 Hz)
    - Beta (13-30 Hz)
    - Gamma (30-40 Hz)
  - Power Spectral Density using Welch's method
  - Spectral Edge Frequency
  - Band Power Ratios

- **EEG Features**
  - Time Domain: Mean, SD, Max, Min, IQR
  - Frequency Domain: Band powers, ratios, spectral entropy, flatness
  - Cross-channel coherence
  - Real-time topographical mapping

- **EDA Features**
  - Skin Conductance Level (SCL)
  - Skin Conductance Response (SCR)
  - Peak detection and analysis
  - Statistical measures

### 3. Machine Learning
- **Mental State Classification**
  - Support Vector Machine (SVM) with RBF kernel
  - States: Relaxed, Focused, Stressed
  - Real-time prediction
  - Interactive training interface

### 4. Visualization
- Interactive Dashboard with:
  - Real-time signal displays
  - Topographical brain mapping
  - Spectral analysis plots
  - Band power visualizations
  - Mental state predictions
  - Process logging
  - EDA component separation

## Setup

1. Create and activate a Python virtual environment:
\`\`\`powershell
python -m venv eeg_python
.\eeg_python\Scripts\activate
\`\`\`

2. Install required packages:
\`\`\`powershell
pip install -r requirements.txt
\`\`\`

3. Ensure data files are in the correct location:
   - EEG data: `data/raw/UnicornRecorder_baseline.csv`
   - EDA data: `data/raw/opensignals_lsl_500hz_gain1_0007808C0708_16-32-15_converted.txt`

## Usage

1. Start the application:
\`\`\`powershell
python run_app.py
\`\`\`

2. Access the dashboard at `http://localhost:8050`

3. Using the Dashboard:
   - Upload or confirm data sources
   - Adjust window size and overlap settings
   - View real-time signal processing
   - Train and use the mental state classifier
   - Generate analysis reports

## File Structure
- `app.py`: Main application and dashboard implementation
- `generate_report.py`: PDF report generation
- `run_app.py`: Application entry point
- `requirements.txt`: Package dependencies
- `data/raw/`: Raw data files
- `output/`: Trained models and analysis results
- `processed/`: Processed data files

## Analysis Parameters
- Window Size: 1 second (adjustable 0.005-2s)
- Overlap: 96% for EEG, 90% for EDA
- Sampling Rates: 250 Hz (EEG), 500 Hz (EDA)
- Filter Settings: 
  - Bandpass: 0.5-40 Hz
  - Notch: 50 Hz

## Data Requirements
- EEG: CSV format, 8 channels (Fp1, Fp2, F3, F4, C3, C4, P3, P4)
- EDA: TXT format, single channel with metadata header

## Output
- Interactive visualizations
- PDF analysis reports
- Trained classification models
- Processed signal data
- Analysis logs

## Dependencies
- Python 3.8+
- Dash
- NumPy
- Pandas
- SciPy
- Plotly
- scikit-learn
- MNE-Python
- FPDF
- joblib
