import os
import numpy as np
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

class AnalysisReport(FPDF):
    def header(self):
        # Add logo if available
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'EEG-EDA Analysis Report', 0, 1, 'C')
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_analysis_table():
    # Create the analysis parameters table
    parameters = {
        'Parameter': [
            'Sampling Rate',
            'Window Length',
            'Window Overlap',
            'Filters Applied',
            'Frequency Bands',
            'Features Extracted',
            'Spectral Methods',
            'Cross-Channel Analysis',
            'Classification',
            'Topographical Mapping'
        ],            'EEG': [
            '250 Hz',
            '1 second (250 samples)',
            '96% (240/250 samples)',            'Notch Filter (50 Hz, Butterworth)\nBandpass Filter (0.5-40 Hz, 4th order Butterworth)',
            'Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-40 Hz)',
            'Time: Mean, SD, Max, Min, IQR\nFreq: Band powers, ratios, spectral entropy, flatness',
            'Welch\'s PSD (nperseg=250), Periodogram with high overlap, Spectral edge frequency, coherence',
            'Mean coherence between all electrode pairs',
            'SVM (RBF kernel) with z-score normalization',
            '10-20 standard montage (Fp1-P4) with 2D interpolation and dynamic visualization'
        ],            'EDA': [
            '500 Hz',
            '1 second (500 samples)',
            '90% (450/500 samples)',
            'Optional smoothing; resampled and baseline corrected',
            'Not applicable',
            'Tonic (SCL), Phasic (SCR peaks), peak count, amplitude, IQR, SD',
            'Welch\'s PSD (nperseg=500), enhanced SCR periodogram, SCL smoothing (4s window)',
            'Not applicable',
            'Integrated features with EEG, trained via user-labeled states (Relaxed, Focused, Stressed)',
            'Not applicable'
        ]
    }
    return pd.DataFrame(parameters)

def generate_pdf_report(eeg_data=None, eda_data=None):
    # Create PDF object
    pdf = AnalysisReport()
    
    # Add title page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 60, 'EEG-EDA Analysis Report', 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
    
    # Add analysis parameters
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Analysis Parameters', 0, 1, 'L')
    pdf.ln(5)
    
    # Create and add the parameters table
    df = create_analysis_table()
    
    # Set up the table
    pdf.set_font('Arial', 'B', 10)
    col_width = [40, 75, 75]
    row_height = 10
    
    # Add headers
    for i, col in enumerate(['Parameter', 'EEG', 'EDA']):
        pdf.cell(col_width[i], row_height, col, 1, 0, 'C')
    pdf.ln()
    
    # Add data
    pdf.set_font('Arial', '', 8)
    for _, row in df.iterrows():
        # Calculate required height
        lines_eeg = len(str(row['EEG']).split('\n'))
        lines_eda = len(str(row['EDA']).split('\n'))
        max_lines = max(lines_eeg, lines_eda, 1)
        height = row_height * max_lines
        
        # Print cells
        pdf.multi_cell(col_width[0], height/max_lines, row['Parameter'], 1, 'L', False)
        x = pdf.get_x()
        y = pdf.get_y() - height
        pdf.set_xy(x + col_width[0], y)
        
        pdf.multi_cell(col_width[1], height/max_lines, str(row['EEG']), 1, 'L', False)
        pdf.set_xy(x + sum(col_width[:2]), y)
        
        pdf.multi_cell(col_width[2], height/max_lines, str(row['EDA']), 1, 'L', False)
        pdf.ln()
    
    # Add analysis results section
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Analysis Results', 0, 1, 'L')
    pdf.ln(5)
    
    # Add model performance metrics if available
    try:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Classification Performance', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        
        # Load model metrics if available
        if os.path.exists('output/svm_model.joblib'):
            from joblib import load
            model = load('output/svm_model.joblib')
            pdf.cell(0, 10, f'Model Type: Support Vector Machine (RBF kernel)', 0, 1, 'L')
            # Add more metrics if available
    except Exception as e:
        pdf.cell(0, 10, 'Model metrics not available', 0, 1, 'L')
    
    # Add spectral analysis results section
    add_spectral_analysis_section(pdf, eeg_data, eda_data)
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save the report
    output_path = 'output/analysis_report.pdf'
    pdf.output(output_path)
    print(f"Report generated successfully: {output_path}")

def add_spectral_analysis_section(pdf, eeg_data, eda_data):
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Spectral Analysis Results', 0, 1, 'L')
    pdf.ln(5)
    
    # EEG Spectral Analysis
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'EEG Power Spectral Density', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    # Load and summarize spectral results if available
    try:
        spectral_results = np.load('output/eeg_spectral.npz')
        pdf.multi_cell(0, 5, f"Window Size: 250 samples (1s)\nOverlap: 96% (240/250 samples)\nSpectral Edge Frequency: {spectral_results['edge_freq']:.2f} Hz\n", 0, 'L')
        pdf.ln(5)
    except:
        pdf.cell(0, 10, 'EEG spectral results not available', 0, 1, 'L')
    
    # EDA Spectral Analysis
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'EDA Component Analysis', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    try:
        eda_results = np.load('output/eda_spectral.npz')
        pdf.multi_cell(0, 5, f"Window Size: 500 samples (1s)\nOverlap: 90% (450/500 samples)\nSCR Frequency: {eda_results['scr_freq']:.2f} Hz\n", 0, 'L')
        pdf.ln(5)
    except:
        pdf.cell(0, 10, 'EDA spectral results not available', 0, 1, 'L')

def load_data():
    try:
        eeg_data = pd.read_csv('data/raw/UnicornRecorder_baseline.csv', low_memory=False).values
        eda_data = pd.read_csv('data/raw/opensignals_lsl_500hz_gain1_0007808C0708_16-32-15_converted.txt', low_memory=False).values
        return eeg_data, eda_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

if __name__ == "__main__":
    eeg_data, eda_data = load_data()
    generate_pdf_report()
