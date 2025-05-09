import os
import numpy as np
import pandas as pd
import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import io
import json
from datetime import datetime
from scipy import signal, stats, interpolate
from scipy.integrate import trapezoid
from scipy.signal import welch, butter, lfilter, find_peaks
from scipy.fft import fft, fftfreq
from mne.channels import make_standard_montage
import mne
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Create standard EEG montage
montage = make_standard_montage('standard_1020')

def parse_eeg_contents(contents):
    """Parse EEG CSV file contents."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Read CSV file
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    # Extract EEG channels (first 8 columns)
    eeg_data = df.iloc[:, 0:8].values.astype(float)
    
    return eeg_data

def parse_eda_contents(contents):
    """Parse EDA file contents."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Read header first
    header_lines = []
    data_lines = []
    for line in decoded.decode('utf-8').split('\n'):
        if line.startswith('#'):
            header_lines.append(line)
        else:
            data_lines.append(line)
    
    # Parse sampling rate from header
    header_json = json.loads(header_lines[1].strip('# '))
    fs = header_json['00:07:80:8C:07:08']['sampling rate']
    
    # Read data
    data = []
    for line in data_lines:
        if line.strip():
            values = [float(x) for x in line.strip().split()]
            if len(values) >= 3:
                data.append(values)
    
    # Convert to DataFrame and extract EDA signal
    df = pd.DataFrame(data, columns=['nSeq', 'DI', 'CH1'])
    eda_data = df['CH1'].values.astype(float)
    
    return eda_data, fs

def find_first_file(folder, exts):
    try:
        # Ensure the folder path exists
        if not os.path.exists(folder):
            print(f"Folder {folder} not found")
            return None
            
        for fname in os.listdir(folder):
            if any(fname.lower().endswith(ext) for ext in exts):
                full_path = os.path.join(folder, fname)
                print(f"Found file: {full_path}")
                return full_path
        print(f"No files with extensions {exts} found in {folder}")
        return None
    except Exception as e:
        print(f"Error in find_first_file: {str(e)}")
        return None

def process_signals(eeg_data, eda_data):
    """Process EEG and EDA signals."""
    # EEG processing
    fs_eeg = 250  # EEG sampling rate
    
    # Apply filters
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data, axis=0)
    
    def notch_filter(data, fs=250, freq=50.0, Q=30):
        b, a = butter(2, [freq-1, freq+1], btype='bandstop', fs=fs)
        return lfilter(b, a, data, axis=0)
    
    # Filter EEG
    eeg_filtered = notch_filter(eeg_data, fs=fs_eeg)
    eeg_filtered = bandpass_filter(eeg_filtered, 0.5, 40, fs_eeg)
    
    # EDA processing
    fs_eda = 500  # EDA sampling rate
    
    # Resample EDA to match EEG length if necessary
    if len(eda_data) > len(eeg_data) * 2:
        eda_data = eda_data[:len(eeg_data) * 2]
    elif len(eda_data) < len(eeg_data) * 2:
        eda_data = np.pad(eda_data, (0, len(eeg_data) * 2 - len(eda_data)))
    
    return eeg_filtered, eda_data

def load_initial_data():
    print("Starting initial data load...")
    try:
        # Use absolute path for data folder
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
        print(f"Looking for files in: {data_folder}")
        
        eeg_path = find_first_file(data_folder, ['.csv'])
        eda_path = find_first_file(data_folder, ['.txt'])
        print(f"Found EEG file: {eeg_path}")
        print(f"Found EDA file: {eda_path}")
        
        if not eeg_path or not eda_path:
            print("Could not find required files")
            return None, None, None, None
        
        # Calculate and display file durations
        print("\nCalculating file durations...")
        
        # Calculate EEG duration (250 Hz)
        with open(eeg_path, 'r', encoding='utf-8') as f:
            eeg_lines = sum(1 for line in f)
            eeg_duration_sec = (eeg_lines - 1) / 250  # -1 for header line
            eeg_duration_min = eeg_duration_sec / 60
            print(f"EEG file duration: {eeg_duration_sec:.2f} seconds ({eeg_duration_min:.2f} minutes)")
        
        # Calculate EDA duration (500 Hz)
        with open(eda_path, 'r', encoding='utf-8') as f:
            eda_lines = sum(1 for line in f if not line.startswith('#'))
            eda_duration_sec = eda_lines / 500
            eda_duration_min = eda_duration_sec / 60
            print(f"EDA file duration: {eda_duration_sec:.2f} seconds ({eda_duration_min:.2f} minutes)")
        
        print("\nLoading EEG data...")
        with open(eeg_path, 'r', encoding='utf-8') as f:
            eeg_contents = f.read()
            eeg_b64 = 'data:text/csv;base64,' + base64.b64encode(eeg_contents.encode('utf-8')).decode('utf-8')
        
        print("Loading EDA data...")
        with open(eda_path, 'r', encoding='utf-8') as f:
            eda_contents = f.read()
            eda_b64 = 'data:text/plain;base64,' + base64.b64encode(eda_contents.encode('utf-8')).decode('utf-8')
        
        return eeg_path, eda_path, eeg_b64, eda_b64
    
    except Exception as e:
        print(f"Error in load_initial_data: {str(e)}")
        return None, None, None, None

# Load initial data if available
initial_eeg, initial_eda, initial_eeg_path, initial_eda_path = load_initial_data()

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define EEG channel mapping
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

# Create the layout
app.layout = html.Div([
    html.H1("EEG and EDA Analysis Dashboard", style={'textAlign': 'center'}),
    
    # File Configuration Section
    html.Div([
        html.H3("Data Source Configuration"),
        html.Div([
            html.Div([
                html.Label("EEG Data Source:"),
                html.Div(id='eeg-file-path', style={'marginBottom': '10px', 'fontFamily': 'monospace'}),
                html.Label("EDA Data Source:"),
                html.Div(id='eda-file-path', style={'marginBottom': '10px', 'fontFamily': 'monospace'}),
            ]),
            html.Button('Confirm Data Sources', id='confirm-sources-button', n_clicks=0),
            html.Div(id='source-confirmation-status'),
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px'}),
        
        # Data Summary Report
        html.Div([
            html.H4("Data Summary"),
            html.Div(id='data-summary-text'),
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px'}),
    ], style={'marginBottom': '20px'}),
    
    # Interpretation Guide
    html.Div([
        html.H3("Signal Processing and Interpretation Guide", style={'marginBottom': '10px'}),
        html.Div([
            html.H4("Time-Frequency Analysis"),
            html.P([
                "Window Size Effects:",
                html.Ul([
                    html.Li("Shorter windows (0.005-0.1s): Better temporal resolution, poorer frequency resolution"),
                    html.Li("Longer windows (0.5-2s): Better frequency resolution, poorer temporal resolution"),
                    html.Li("Recommended: 1s window for balanced analysis")
                ])
            ]),
            html.H4("Filter Settings"),
            html.P([
                "Signal Filtering:",
                html.Ul([
                    html.Li("Bandpass (0.5-40 Hz): Removes drift and high-frequency noise"),
                    html.Li("Notch (50 Hz): Removes power line interference"),
                    html.Li("Filter response shown in frequency plot")
                ])
            ]),
            html.H4("Interpretation Guide"),
            html.P([
                "Brain State Indicators:",
                html.Ul([
                    html.Li("Alpha/Theta > 1: High alertness and attention"),
                    html.Li("Strong alpha (8-13 Hz): Relaxed wakefulness"),
                    html.Li("Strong theta (4-8 Hz): Deep relaxation or drowsiness"),
                    html.Li("High beta (13-30 Hz): Active thinking or anxiety"),
                    html.Li("Balanced gamma (30-40 Hz): Complex cognitive processing")
                ])
            ]),
            html.H4("EDA Components"),
            html.P([
                "Skin Conductance:",
                html.Ul([
                    html.Li("SCL (blue line): Tonic level, overall arousal"),
                    html.Li("SCR (red peaks): Phasic responses to stimuli"),
                    html.Li("Higher values indicate increased sympathetic activity")
                ])
            ])
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
    ], style={'marginBottom': '20px'}),
    
    # File Upload Section
    html.Div([
        html.H3("Data Upload"),
        html.Div([
            html.Div(f"Auto-loaded EEG: {os.path.basename(initial_eeg_path) if initial_eeg_path else 'None'}", style={'color': 'green' if initial_eeg_path else 'gray'}),
            html.Div(f"Auto-loaded EDA: {os.path.basename(initial_eda_path) if initial_eda_path else 'None'}", style={'color': 'green' if initial_eda_path else 'gray'}),
        ]),
        dcc.Upload(
            id='upload-eeg',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select EEG File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        dcc.Upload(
            id='upload-eda',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select EDA File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        html.Button('Process Data', id='process-button', n_clicks=0),
        html.Button('Run Complete Analysis', id='analyze-button', n_clicks=0, 
                   style={'marginLeft': '10px', 'backgroundColor': '#4CAF50', 'color': 'white'}),
        html.Div(id='upload-status')
    ], style={'width': '100%', 'padding': '20px'}),
    
    # Main Content (initially hidden)
    html.Div(id='main-content', style={'display': 'none'}, children=[
        # Control Panel
        html.Div([
            html.Div([
                html.H3("Controls"),
                html.Label("Time Window (seconds)"),
                dcc.Slider(
                    id='time-slider',
                    min=0,
                    max=30,
                    step=0.005,
                    value=0,
                    marks={float(i): f'{i:.3f}' for i in np.arange(0, 31, 5)},
                ),
                html.Div([
                    html.Label("Window Size (seconds)"),
                    dcc.Slider(
                        id='window-size-slider',
                        min=0.005,
                        max=2,
                        step=0.005,
                        value=1,
                        marks={float(i): f'{i:.3f}' for i in [0.005, 0.1, 0.5, 1, 1.5, 2]},
                    ),
                ]),
                html.Div([
                    html.Label("Playback Speed (samples/sec)"),
                    dcc.Input(
                        id='speed-input',
                        type='number',
                        value=10,
                        min=1,
                        max=100,
                        step=1,
                        style={'width': '100px', 'margin': '10px 0'}
                    ),
                ]),
                html.Button('Play/Pause', id='play-button', n_clicks=0),
                html.Button('Reset', id='reset-button', n_clicks=0),
                dcc.Interval(
                    id='interval-component',
                    interval=100,  # in milliseconds
                    n_intervals=0,
                    disabled=True
                ),
            ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
            
            # Main Visualization Area
            html.Div([
                html.Div([
                    html.H3("Signal Processing"),
                    # Filter Response
                    dcc.Graph(id='filter-response-plot', style={'height': '300px'}),
                    # EEG Topography
                    dcc.Graph(id='topography-plot', style={'height': '500px'}),
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.H3("Time Series"),
                    # EEG Time Series
                    dcc.Graph(id='eeg-plot', style={'height': '300px'}),
                    # EDA Signal and Features
                    dcc.Graph(id='eda-plot', style={'height': '200px'}),
                    dcc.Graph(id='eda-features-plot', style={'height': '300px'}),
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.H3("Spectral Analysis"),
                    # Band Powers
                    dcc.Graph(id='band-powers-plot', style={'height': '400px'}),
                    # Average Band Powers
                    dcc.Graph(id='avg-powers-plot', style={'height': '400px'}),
                    # Power Spectral Density Plots
                    dcc.Graph(id='psd-plots', style={'height': '800px'}),
                    # Overlap Analysis
                    dcc.Graph(id='overlap-comparison-plot', style={'height': '400px'}),
                    # Alpha-Theta Analysis
                    dcc.Graph(id='analysis-plot', style={'height': '400px'}),
                ], style={'marginBottom': '20px'}),
                
                # New SVM Section
                html.Div([
                    html.H3("Mental State Classification"),
                    html.Div([
                        html.H4("Model Training"),
                        html.P("Label current window:"),
                        dcc.Dropdown(
                            id='state-label-dropdown',
                            options=[
                                {'label': 'Relaxed', 'value': 'relaxed'},
                                {'label': 'Focused', 'value': 'focused'},
                                {'label': 'Stressed', 'value': 'stressed'}
                            ],
                            value='relaxed'
                        ),
                        html.Button('Add Training Sample', id='add-sample-button', n_clicks=0),
                        html.Button('Train Model', id='train-model-button', n_clicks=0),
                        html.Div(id='training-status'),
                        
                        html.H4("Current State Prediction"),
                        dcc.Graph(id='state-prediction-plot', style={'height': '200px'}),
                        html.Div(id='prediction-text'),
                        
                        # Hidden storage for training data
                        dcc.Store(id='training-features'),
                        dcc.Store(id='training-labels')
                    ])
                ], style={'marginBottom': '20px'})
            ], style={'width': '75%', 'display': 'inline-block', 'padding': '20px'}),
        ], style={'display': 'flex'}),
    ]),
    
    # Hidden divs for storing data
    dcc.Store(id='eeg-data', data=initial_eeg),
    dcc.Store(id='eda-data', data=initial_eda),
    dcc.Store(id='processed-data'),
    
    # Add Log Messages Section at the bottom
    html.Div([
        html.H3("Process Log"),
        html.Div(id='log-messages', 
                style={
                    'maxHeight': '200px',
                    'overflowY': 'auto',
                    'padding': '10px',
                    'border': '1px solid #ddd',
                    'borderRadius': '5px',
                    'backgroundColor': '#f8f9fa',
                    'fontFamily': 'monospace',
                    'whiteSpace': 'pre-wrap'
                })
    ], style={'marginTop': '20px'}),
    
    # Add storage for log messages
    dcc.Store(id='log-storage', data=[]),
    
    # Add storage for file paths
    dcc.Store(id='file-paths', data={'eeg': None, 'eda': None}),
])

# Function to extract features from EEG and EDA data
def extract_features(eeg_data, eda_data, window_size=250, overlap=0.5):
    """Extract features from EEG and EDA data for classification.
    
    Features extracted:
    1. EEG Features (per channel):
        Time Domain:
        - Mean
        - Standard deviation
        - Maximum value
        - Minimum value
        - Interquartile range (IQR)
        
        Frequency Domain:
        - Band powers for delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz),
          beta (13-30 Hz), and gamma (30-40 Hz)
        - Band power ratios (alpha/theta, beta/alpha, gamma/beta)
        - Spectral edge frequency (95th percentile)
        - Spectral entropy (measure of signal regularity)
        - Spectral flatness (Wiener entropy, measure of signal tonality)
        
        Cross-channel:
        - Coherence between all channel pairs
    
    2. EDA Features:
        - Skin conductance level (SCL, tonic component)
        - Number of skin conductance responses (SCR peaks)
        - Mean SCR peak amplitude
        - Standard deviation
        - Range (max - min)
        - Interquartile range
    
    Parameters:
        eeg_data (np.ndarray): EEG data array of shape (samples, channels)
        eda_data (np.ndarray): EDA data array of shape (samples,)
        window_size (int): Size of the analysis window in samples
        overlap (float): Overlap between consecutive windows (0 to 1)
    
    Returns:
        np.ndarray: Feature matrix of shape (n_windows, n_features)
    """
    features = []
    
    # Calculate step size for overlapping windows
    step = int(window_size * (1 - overlap))
    
    for i in range(0, len(eeg_data) - window_size, step):
        window_eeg = eeg_data[i:i + window_size]
        window_eda = eda_data[i*2:(i + window_size)*2]  # EDA has 2x sampling rate
        
        # EEG features
        eeg_features = []
        for ch in range(window_eeg.shape[1]):
            ch_features = []
            
            # 1. Time domain features
            ch_features.extend([
                np.mean(window_eeg[:, ch]),
                np.std(window_eeg[:, ch]),
                np.max(window_eeg[:, ch]),
                np.min(window_eeg[:, ch]),
                np.percentile(window_eeg[:, ch], 75) - np.percentile(window_eeg[:, ch], 25)  # IQR
            ])
            
            # 2. Frequency domain features
            fs = 250  # EEG sampling rate
            # Ensure nperseg is not larger than window length
            nperseg = min(256, len(window_eeg))
            f, psd = welch(window_eeg[:, ch], fs=fs, nperseg=nperseg)
            
            # Traditional frequency bands
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 40)
            }
            
            # 2.1 Band powers using scipy.integrate.trapezoid
            band_powers = {}
            for band, (low, high) in bands.items():
                idx = np.logical_and(f >= low, f < high)
                band_powers[band] = trapezoid(psd[idx], f[idx])
                ch_features.append(band_powers[band])
            
            # 2.2 Band power ratios (from MLSys example)
            ch_features.extend([
                band_powers['alpha'] / band_powers['theta'] if band_powers['theta'] > 0 else 0,
                band_powers['beta'] / band_powers['alpha'] if band_powers['alpha'] > 0 else 0,
                band_powers['gamma'] / band_powers['beta'] if band_powers['beta'] > 0 else 0
            ])
            
            # 2.3 Spectral edge frequency (95% of total power)
            total_power = np.sum(psd)
            cumulative_power = np.cumsum(psd) / total_power
            spectral_edge_idx = np.where(cumulative_power >= 0.95)[0][0]
            ch_features.append(f[spectral_edge_idx])
            
            # 2.4 Spectral entropy (from MLSys example)
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            ch_features.append(spectral_entropy)
            
            # 2.5 Spectral flatness
            spectral_flatness = np.exp(np.mean(np.log(psd + 1e-10))) / (np.mean(psd) + 1e-10)
            ch_features.append(spectral_flatness)
            
            eeg_features.extend(ch_features)
        
        # 3. Cross-channel features
        for i in range(window_eeg.shape[1]):
            for j in range(i + 1, window_eeg.shape[1]):
                # 3.1 Coherence between channels
                _, coherence = signal.coherence(window_eeg[:, i], window_eeg[:, j], fs=fs)
                eeg_features.append(np.mean(coherence))
        
        # 4. EDA features
        # 4.1 Basic statistics
        scl = np.mean(window_eda)  # Skin conductance level
        scr = window_eda - scl  # Skin conductance response
        
        # 4.2 SCR peaks analysis
        scr_peaks, properties = signal.find_peaks(scr, height=0.1, distance=int(500*0.5))  # Min 0.5s between peaks
        
        eda_features = [
            scl,  # Tonic component (SCL)
            len(scr_peaks),  # Number of SCR peaks
            np.mean(properties['peak_heights']) if len(scr_peaks) > 0 else 0,  # Mean peak amplitude
            np.std(window_eda),  # Standard deviation
            np.max(window_eda) - np.min(window_eda),  # Range
            np.percentile(window_eda, 75) - np.percentile(window_eda, 25)  # IQR
        ]
        
        # Combine all features
        features.append(np.concatenate([eeg_features, eda_features]))
    
    return np.array(features)

# SVM model training function
def train_svm_model(features, labels):
    """Train SVM model with the extracted features."""
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.2, random_state=42
    )
    
    # Train SVM
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    
    # Save model and scaler
    model_path = os.path.join('output', 'svm_model.joblib')
    scaler_path = os.path.join('output', 'scaler.joblib')
    os.makedirs('output', exist_ok=True)
    joblib.dump(svm, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Evaluate model
    train_score = svm.score(X_train, y_train)
    test_score = svm.score(X_test, y_test)
    
    return train_score, test_score

# Function to load and predict with SVM model
def predict_mental_state(eeg_data, eda_data, window_size=250):
    """Predict mental state using trained SVM model."""
    try:
        # Load model and scaler
        model = joblib.load(os.path.join('output', 'svm_model.joblib'))
        scaler = joblib.load(os.path.join('output', 'scaler.joblib'))
        
        # Extract features for the current window
        features = extract_features(eeg_data, eda_data, window_size=window_size, overlap=0)
        
        if len(features) == 0:
            return None
        
        # Scale features and predict
        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        return predictions[-1], probabilities[-1]
    except:
        return None

@callback(
    [Output('upload-status', 'children'),
     Output('main-content', 'style'),
     Output('eeg-data', 'data'),
     Output('eda-data', 'data'),
     Output('file-paths', 'data'),
     Output('data-summary-text', 'children'),
     Output('source-confirmation-status', 'children'),
     Output('log-storage', 'data')],
    [Input('process-button', 'n_clicks'),
     Input('confirm-sources-button', 'n_clicks')],
    [State('upload-eeg', 'contents'),
     State('upload-eda', 'contents'),
     State('eeg-data', 'data'),
     State('eda-data', 'data'),
     State('file-paths', 'data'),
     State('log-storage', 'data')]
)
def process_data(process_clicks, confirm_clicks, eeg_contents, eda_contents, current_eeg_data, current_eda_data, current_paths, current_logs):
    ctx = dash.callback_context
    if not ctx.triggered:
        return '', {'display': 'none'}, None, None, current_paths, "", "Awaiting data", current_logs
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'process-button' and process_clicks > 0:
        try:
            # Use uploaded data if available, otherwise use current data
            if eeg_contents and eda_contents:
                # Parse uploaded files
                eeg_data = parse_eeg_contents(eeg_contents)
                eda_data, _ = parse_eda_contents(eda_contents)
                new_paths = {'eeg': 'uploaded_file.csv', 'eda': 'uploaded_file.txt'}
            elif current_eeg_data and current_eda_data:
                # Use auto-loaded data
                eeg_data = np.array(current_eeg_data)
                eda_data = np.array(current_eda_data)
                new_paths = current_paths
            else:
                return (html.Div('No data available. Please upload files or check auto-loaded data.', 
                              style={'color': 'red'}),
                        {'display': 'none'},
                        None,
                        None,
                        current_paths,
                        "",
                        "",
                        current_logs)
            
            # Process signals
            eeg_processed, eda_processed = process_signals(eeg_data, eda_data)
            
            # Update log
            timestamp = datetime.now().strftime("%H:%M:%S")
            new_log = current_logs + [f"[{timestamp}] Data processed successfully"]
            
            return (html.Div('Files processed successfully!', style={'color': 'green'}),
                    {'display': 'block'},
                    eeg_processed.tolist(),
                    eda_processed.tolist(),
                    new_paths,
                    "",
                    "",
                    new_log)
        
        except Exception as e:
            return (html.Div(f'Error processing files: {str(e)}', style={'color': 'red'}),
                    {'display': 'none'},
                    None,
                    None,
                    current_paths,
                    "",
                    "",
                    current_logs)
    
    elif trigger_id == 'confirm-sources-button' and confirm_clicks > 0:
        if not current_eeg_data or not current_eda_data:
            return (html.Div(''),
                    {'display': 'none'},
                    current_eeg_data,
                    current_eda_data,
                    current_paths,
                    html.Div("No data loaded", style={'color': 'red'}),
                    html.Div("No data loaded", style={'color': 'red'}),
                    current_logs)
        
        try:
            # Generate data summary
            eeg_array = np.array(current_eeg_data)
            eda_array = np.array(current_eda_data)
            
            eeg_summary = f"""EEG Data Summary:
- File: {os.path.basename(current_paths['eeg'])}
- Channels: {len(eeg_channels)}
- Sample Rate: 250 Hz
- Duration: {len(eeg_array)/250:.2f} seconds
- Channel Names: {', '.join(eeg_channels.values())}"""
            
            eda_summary = f"""EDA Data Summary:
- File: {os.path.basename(current_paths['eda'])}
- Sample Rate: 500 Hz
- Duration: {len(eda_array)/500:.2f} seconds"""
            
            summary = html.Div([
                html.Pre(eeg_summary, style={'whiteSpace': 'pre-wrap'}),
                html.Pre(eda_summary, style={'whiteSpace': 'pre-wrap'})
            ])
            
            # Update log
            timestamp = datetime.now().strftime("%H:%M:%S")
            new_log = current_logs + [f"[{timestamp}] Data sources confirmed successfully"]
            
            return (html.Div(''),
                    {'display': 'block'},
                    current_eeg_data,
                    current_eda_data,
                    current_paths,
                    summary,
                    html.Div("Data sources confirmed", style={'color': 'green'}),
                    new_log)
            
        except Exception as e:
            return (html.Div(''),
                    {'display': 'none'},
                    current_eeg_data,
                    current_eda_data,
                    current_paths,
                    html.Div(f"Error: {str(e)}", style={'color': 'red'}),
                    html.Div(f"Error: {str(e)}", style={'color': 'red'}),
                    current_logs)
    
    return '', {'display': 'none'}, None, None, current_paths, "", "", current_logs

# Callback to update log messages display
@callback(
    Output('log-messages', 'children'),
    [Input('log-storage', 'data')]
)
def update_log_display(logs):
    if not logs:
        return "No log messages"
    return html.Pre('\n'.join(logs[-50:]))  # Show last 50 messages

# Process upload callback is defined above

def create_topography_plot(eeg_data, time_idx, window_size=1000):
    """Create a topographic map using plotly with correct 2D channel positions and scale."""
    # Ensure indices are integers and within bounds
    start_idx = max(0, min(int(time_idx), len(eeg_data)-1))
    end_idx = max(0, min(start_idx + int(window_size), len(eeg_data)))
    
    # Calculate mean activity for the current window
    mean_activity = np.mean(eeg_data[start_idx:end_idx], axis=0)
    
    # Define standard 10-20 positions for our channels
    standard_positions = {
        'Fp1': [-0.3, 0.7], 'Fp2': [0.3, 0.7],
        'F3': [-0.5, 0.4], 'F4': [0.5, 0.4],
        'C3': [-0.7, 0], 'C4': [0.7, 0],
        'P3': [-0.5, -0.4], 'P4': [0.5, -0.4]
    }
    
    # Channel names in order
    ch_names = [eeg_channels[i] for i in range(8)]
    
    # Get positions array
    pos = np.array([standard_positions[name] for name in ch_names])
    
    # Get correct 2D positions from montage
    pos_dict = montage.get_positions()['ch_pos']
    pos = np.array([pos_dict[ch] for ch in ch_names])[:, :2]  # Only x, y
    # Flip y-axis so nose is at the top (standard EEG convention)
    pos[:, 1] *= -1
    # Compute scale from montage coordinates
    x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
    y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
    pad = 0.02
    x_range = [x_min - pad, x_max + pad]
    y_range = [y_min - pad, y_max + pad]
    head_radius = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max)) + pad
    # Create the figure
    fig = go.Figure()
    # Add the head outline
    theta = np.linspace(0, 2*np.pi, 100)
    x_head = head_radius * np.cos(theta)
    y_head = head_radius * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=x_head, y=y_head,
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    # Add the nose (now at the top)
    nose_x = [0, 0.02, 0]
    nose_y = [head_radius, head_radius+0.03, head_radius]
    fig.add_trace(go.Scatter(
        x=nose_x, y=nose_y,
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    # Add the ears
    ear_x = [x_min, x_min-0.01, x_min]
    ear_y = [0.01, 0, -0.01]
    fig.add_trace(go.Scatter(
        x=ear_x, y=ear_y,
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    ear_x2 = [x_max, x_max+0.01, x_max]
    ear_y2 = [0.01, 0, -0.01]
    fig.add_trace(go.Scatter(
        x=ear_x2, y=ear_y2,
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    # Interpolation grid
    x_grid = np.linspace(x_range[0], x_range[1], 100)
    y_grid = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    from scipy.interpolate import griddata
    Z = griddata(pos, mean_activity, (X, Y), method='cubic')
    # Color scale
    vmin = np.min(mean_activity)
    vmax = np.max(mean_activity)
    abs_max = max(abs(vmin), abs(vmax))
    # Add the interpolated heatmap
    fig.add_trace(go.Heatmap(
        z=Z,
        x=x_grid,
        y=y_grid,
        colorscale='RdBu_r',
        showscale=True,
        zmin=-abs_max,
        zmax=abs_max,
        colorbar=dict(
            title=dict(
                text='µV',
                font=dict(size=14)
            ),
            thickness=20,
            len=0.8,
            y=0.5,
            yanchor='middle'
        )
    ))
    # Add channel markers
    fig.add_trace(go.Scatter(
        x=pos[:, 0], y=pos[:, 1],
        mode='markers+text',
        marker=dict(size=15, color='black'),
        text=ch_names,
        textposition="top center",
        showlegend=False
    ))
    # Update layout
    fig.update_layout(
        title='EEG Topography',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=x_range
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=y_range,
            scaleanchor="x",
            scaleratio=1
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        width=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

def create_eeg_plot(eeg_data, time_idx, window_size=1000):
    """Create EEG time series plot."""
    time = np.arange(len(eeg_data)) / 250  # 250 Hz sampling rate
    fig = go.Figure()
    
    for ch in range(eeg_data.shape[1]):
        fig.add_trace(go.Scatter(
            x=time[time_idx:time_idx+window_size],
            y=eeg_data[time_idx:time_idx+window_size, ch],
            name=eeg_channels[ch],
            mode='lines'
        ))
    
    fig.update_layout(
        title='EEG Signals',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude (µV)',
        showlegend=True,
        height=300
    )
    
    return fig

def create_eda_plot(eda_data, time_idx, window_size=1000):
    """Create EDA signal plot."""
    time = np.arange(len(eda_data)) / 500  # 500 Hz sampling rate
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time[time_idx:time_idx+window_size*2],
        y=eda_data[time_idx:time_idx+window_size*2],
        name='EDA',
        mode='lines',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title='EDA Signal',
        xaxis_title='Time (s)',
        yaxis_title='Conductance (µS)',
        showlegend=False,
        height=200
    )
    
    return fig

def create_band_powers_plot(eeg_data, time_idx, window_size=1000):
    """Create band powers plot."""
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    
    # Calculate band powers
    band_powers = []
    avg_powers = []
    for band, (low, high) in bands.items():
        powers = []
        for ch in range(eeg_data.shape[1]):
            f, Pxx = welch(eeg_data[time_idx:time_idx+window_size, ch], 
                          fs=250, nperseg=min(256, window_size))
            idx = np.logical_and(f >= low, f < high)
            power = np.trapezoid(Pxx[idx], f[idx])
            powers.append(power)
        band_powers.append(powers)
        avg_powers.append(np.mean(powers))
    
    # Calculate alpha-theta ratio
    alpha_idx = list(bands.keys()).index('alpha')
    theta_idx = list(bands.keys()).index('theta')
    alpha_theta_ratio = avg_powers[alpha_idx] / avg_powers[theta_idx]
    
    # Create the figure with subplots
    fig = go.Figure()
    
    # Plot individual channel powers
    x = np.arange(len(eeg_channels))
    width = 0.15
    
    for i, (band, powers) in enumerate(zip(bands.keys(), band_powers)):
        fig.add_trace(go.Bar(
            x=x + i*width,
            y=powers,
            name=band,
            width=width,
            opacity=0.7
        ))
    
    fig.update_layout(
        title='Band Powers by Channel',
        xaxis_title='Channel',
        yaxis_title='Power (µV²/Hz)',
        barmode='group',
        xaxis=dict(
            ticktext=list(eeg_channels.values()),
            tickvals=x + width*2
        ),
        height=400,
        yaxis=dict(
            range=[0, max([max(powers) for powers in band_powers]) * 1.2]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Create average powers plot
    avg_fig = go.Figure()
    
    # Plot average powers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (band, avg_power) in enumerate(zip(bands.keys(), avg_powers)):
        avg_fig.add_trace(go.Bar(
            x=[band],
            y=[avg_power],
            name=band,
            marker_color=colors[i]
        ))
    
    avg_fig.update_layout(
        title='Average Band Powers Across Channels',
        xaxis_title='Frequency Band',
        yaxis_title='Average Power (µV²/Hz)',
        height=400,
        yaxis=dict(
            range=[0, max(avg_powers) * 1.2]
        ),
        showlegend=False
    )
    
    # Create alpha-theta analysis plot
    analysis_fig = go.Figure()
    
    # Add alpha-theta ratio
    analysis_fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=alpha_theta_ratio,
        title={'text': "Alpha-Theta Ratio"},
        gauge={'axis': {'range': [0, 5]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 1], 'color': "lightgray"},
                   {'range': [1, 2], 'color': "lightblue"},
                   {'range': [2, 3], 'color': "blue"},
                   {'range': [3, 4], 'color': "darkblue"},
                   {'range': [4, 5], 'color': "navy"}
               ]}
    ))
    
    # Add statistical analysis
    alpha_powers = band_powers[alpha_idx]
    theta_powers = band_powers[theta_idx]
    
    stats = {
        'Alpha Mean': np.mean(alpha_powers),
        'Alpha Std': np.std(alpha_powers),
        'Theta Mean': np.mean(theta_powers),
        'Theta Std': np.std(theta_powers),
        'Alpha-Theta Ratio': alpha_theta_ratio
    }
    
    # Add statistics as annotations
    analysis_fig.add_annotation(
        x=0.5,
        y=0.15,
        text=(
            f"<span style='font-size:18px'><b>Alpha Mean:</b> {stats['Alpha Mean']:.2f} µV²/Hz<br>"
            f"<b>Alpha Std:</b> {stats['Alpha Std']:.2f} µV²/Hz<br>"
            f"<b>Theta Mean:</b> {stats['Theta Mean']:.2f} µV²/Hz<br>"
            f"<b>Theta Std:</b> {stats['Theta Std']:.2f} µV²/Hz</span>"
        ),
        showarrow=False,
        font=dict(size=16, color='black'),
        align="center",
        xref='paper',
        yref='paper',
        bordercolor='black',
        borderwidth=1,
        borderpad=4,
        bgcolor='white',
        opacity=0.9
    )
    
    analysis_fig.update_layout(
        title='Alpha-Theta Analysis',
        height=400,
        annotations=[
            dict(
                x=0.5,
                y=0.1,
                text="Reference: Alpha-Theta ratio > 1 indicates increased alertness and attention<br>" +
                     "Alpha waves (8-13 Hz) are associated with relaxed wakefulness<br>" +
                     "Theta waves (4-8 Hz) are associated with drowsiness and meditation",
                showarrow=False,
                font=dict(size=12),
                align="center"
            )
        ]
    )
    
    return fig, avg_fig, analysis_fig

def create_filter_response_plot(fs_eeg=250):
    """Create filter frequency response visualization."""
    # Generate frequency response for the filters
    freqs = np.logspace(0, 2, 1000)
    
    # Bandpass response
    nyq = 0.5 * fs_eeg
    low = 0.5 / nyq
    high = 40 / nyq
    b, a = butter(4, [low, high], btype='band')
    w_band, h_band = signal.freqz(b, a, freqs)
    
    # Notch response
    b_notch, a_notch = butter(2, [48/nyq, 52/nyq], btype='bandstop')
    w_notch, h_notch = signal.freqz(b_notch, a_notch, freqs)
    
    # Create combined response plot
    fig = go.Figure()
    
    # Add bandpass response
    fig.add_trace(go.Scatter(
        x=w_band * fs_eeg / (2*np.pi),
        y=20 * np.log10(np.abs(h_band)),
        name='Bandpass Filter (0.5-40 Hz)',
        line=dict(color='blue')
    ))
    
    # Add notch response
    fig.add_trace(go.Scatter(
        x=w_notch * fs_eeg / (2*np.pi),
        y=20 * np.log10(np.abs(h_notch)),
        name='Notch Filter (50 Hz)',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Filter Frequency Response',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude (dB)',
        xaxis_type='log',
        showlegend=True,
        height=300
    )
    
    return fig

def create_eda_features_plot(eda_data, time_idx, window_size=1000):
    """Create EDA features plot showing SCL and SCR."""
    from scipy.signal import find_peaks
    
    # Convert indices to time
    fs_eda = 500  # EDA sampling rate
    time = np.arange(len(eda_data)) / fs_eda
    
    # Extract window of data
    eda_window = eda_data[time_idx:time_idx+window_size*2]
    time_window = time[time_idx:time_idx+window_size*2]
    
    # Calculate SCL (baseline) using moving average
    # Make sure window size is not larger than data
    window_size_scl = min(int(fs_eda * 4), len(eda_window)-1)  # 4-second window or smaller
    scl = np.convolve(eda_window, np.ones(window_size_scl)/window_size_scl, mode='valid')
    # Pad the SCL signal to match the original length
    pad_size = len(eda_window) - len(scl)
    left_pad = pad_size // 2
    right_pad = pad_size - left_pad
    scl = np.pad(scl, (left_pad, right_pad), mode='edge')
    
    # Find SCR peaks
    scr = eda_window - scl
    peaks, properties = find_peaks(scr, height=0.1, distance=int(fs_eda*0.5))
    
    # Create plot
    fig = go.Figure()
    
    # Plot raw EDA
    fig.add_trace(go.Scatter(
        x=time_window,
        y=eda_window,
        name='Raw EDA',
        line=dict(color='green')
    ))
    
    # Plot SCL
    fig.add_trace(go.Scatter(
        x=time_window,
        y=scl,
        name='SCL (baseline)',
        line=dict(color='blue', dash='dash')
    ))
    
    # Plot SCR peaks
    fig.add_trace(go.Scatter(
        x=time_window[peaks],
        y=eda_window[peaks],
        mode='markers',
        name='SCR peaks',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(
        title='EDA Components Analysis',
        xaxis_title='Time (s)',
        yaxis_title='Conductance (µS)',
        showlegend=True,
        height=300
    )
    
    return fig

def create_spectral_density_plots(eeg_data, time_idx, window_size=1000, fs=250):
    """Create power spectral density plots using both periodogram and Welch's method with high overlap."""
    # Get window of data
    start_idx = max(0, min(int(time_idx), len(eeg_data)-1))
    end_idx = max(0, min(start_idx + int(window_size), len(eeg_data)))
    window_data = eeg_data[start_idx:end_idx]
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('EEG Periodogram (96% overlap)', 'EDA Periodogram (90% overlap)',
                      'EEG Welch PSD (96% overlap)', 'EDA Welch PSD (90% overlap)')
    )
    
    # Colors for different channels
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
    
    # Calculate window sizes and overlaps
    eeg_nperseg = 250  # 1 second for EEG
    eda_nperseg = 500  # 1 second for EDA
    eeg_noverlap = 240  # 96% overlap for EEG
    eda_noverlap = 450  # 90% overlap for EDA
    
    # Calculate and plot periodogram and Welch for each channel
    for ch in range(window_data.shape[1]):
        # EEG Periodogram (96% overlap)
        freqs_p_eeg, psd_p_eeg = signal.welch(window_data[:, ch], fs=fs, 
                                             nperseg=eeg_nperseg, noverlap=eeg_noverlap)
        fig.add_trace(
            go.Scatter(
                x=freqs_p_eeg,
                y=psd_p_eeg,
                mode='lines',
                name=f'{eeg_channels[ch]} (EEG)',
                line=dict(color=colors[ch]),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # EDA Periodogram (90% overlap)
        freqs_p_eda, psd_p_eda = signal.welch(window_data[:, ch], fs=fs*2,  # Note: EDA fs = 500Hz
                                             nperseg=eda_nperseg, noverlap=eda_noverlap)
        fig.add_trace(
            go.Scatter(
                x=freqs_p_eda,
                y=psd_p_eda,
                mode='lines',
                name=f'{eeg_channels[ch]} (EDA)',
                line=dict(color=colors[ch], dash='dot'),
                showlegend=True
            ),
            row=1, col=2
        )
        
        # EEG Welch (96% overlap)
        freqs_w_eeg, psd_w_eeg = signal.welch(window_data[:, ch], fs=fs,
                                             nperseg=eeg_nperseg, noverlap=eeg_noverlap)
        fig.add_trace(
            go.Scatter(
                x=freqs_w_eeg,
                y=psd_w_eeg,
                mode='lines',
                name=f'{eeg_channels[ch]} (EEG Welch)',
                line=dict(color=colors[ch]),
                showlegend=True
            ),
            row=2, col=1
        )
        
        # EDA Welch (90% overlap)
        freqs_w_eda, psd_w_eda = signal.welch(window_data[:, ch], fs=fs*2,
                                             nperseg=eda_nperseg, noverlap=eda_noverlap)
        fig.add_trace(
            go.Scatter(
                x=freqs_w_eda,
                y=psd_w_eda,
                mode='lines',
                name=f'{eeg_channels[ch]} (EDA Welch)',
                line=dict(color=colors[ch], dash='dot'),
                showlegend=True
            ),
            row=2, col=2
        )
    
    # Update layout for all subplots
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(type='log', range=[np.log10(0.5), np.log10(100)], row=i, col=j)
            fig.update_yaxes(type='log', range=[np.log10(0.000001), np.log10(10)], row=i, col=j)
            fig.update_xaxes(title_text='Frequency (Hz)', row=i, col=j)
            fig.update_yaxes(title_text='Power', row=i, col=j)
    
    # Update overall layout
    fig.update_layout(
        height=1000,  # Increased height for better visibility
        title='Power Spectral Density Analysis with High Overlap',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_overlap_comparison_plot(eeg_data, window_samples, overlap_percent):
    """Create a plot comparing different window overlaps for EEG signal analysis."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Calculate samples for overlap
    overlap_samples = int((overlap_percent/100) * window_samples)
    hop_samples = window_samples - overlap_samples
    
    # Create windows with specified overlap
    windows = []
    start = 0
    while start + window_samples <= len(eeg_data):
        windows.append(eeg_data[start:start + window_samples])
        start += hop_samples
    
    # Create subplot
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=[f'Window Overlap Analysis ({overlap_percent}% overlap)',
                                     'Window Transitions'])
    
    # Plot first few windows
    for i, window in enumerate(windows[:3]):
        fig.add_trace(
            go.Scatter(y=window, name=f'Window {i+1}',
                      line=dict(width=2)),
            row=1, col=1
        )
    
    # Plot transition between windows
    transition_data = eeg_data[:window_samples*3]
    fig.add_trace(
        go.Scatter(y=transition_data, name='Signal',
                  line=dict(color='black', width=2)),
        row=2, col=1
    )
    
    # Add vertical lines showing window boundaries
    for i in range(0, len(transition_data), hop_samples):
        fig.add_vline(x=i, line_dash="dash", line_color="gray",
                     row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text=f"Window Analysis (Size: {window_samples}, Overlap: {overlap_percent}%)",
    )
    
    return fig

@callback(
    [Output('topography-plot', 'figure'),
     Output('eeg-plot', 'figure'),
     Output('eda-plot', 'figure'),
     Output('band-powers-plot', 'figure'),
     Output('avg-powers-plot', 'figure'),
     Output('analysis-plot', 'figure'),
     Output('filter-response-plot', 'figure'),
     Output('eda-features-plot', 'figure'),
     Output('psd-plots', 'figure')],
    [Input('time-slider', 'value'),
     Input('window-size-slider', 'value'),
     Input('interval-component', 'n_intervals')],
    [State('eeg-data', 'data'),
     State('eda-data', 'data')]
)
def update_plots(time_idx, window_size_seconds, n_intervals, eeg_data, eda_data):
    if eeg_data is None or eda_data is None:
        return [go.Figure()] * 10
    
    # Convert data to numpy arrays
    eeg_array = np.array(eeg_data)
    eda_array = np.array(eda_data)
    
    # Convert time values from seconds to samples
    time_samples = int(time_idx * 250)  # 250 Hz sampling rate for EEG
    window_samples = int(window_size_seconds * 250)
    
    # Create all plots
    topo_fig = create_topography_plot(eeg_array, time_samples, window_samples)
    eeg_fig = create_eeg_plot(eeg_array, time_samples, window_samples)
    eda_fig = create_eda_plot(eda_array, time_samples, window_samples)
    band_fig, avg_fig, analysis_fig = create_band_powers_plot(eeg_array, time_samples, window_samples)
    filter_fig = create_filter_response_plot()
    eda_features_fig = create_eda_features_plot(eda_array, time_samples, window_samples)
    psd_fig = create_spectral_density_plots(eeg_array, time_samples, window_samples)
    overlap_fig = create_overlap_comparison_plot(eeg_array, eda_array)
    
    return topo_fig, eeg_fig, eda_fig, band_fig, avg_fig, analysis_fig, filter_fig, eda_features_fig, psd_fig, overlap_fig

@callback(
    Output('interval-component', 'interval'),
    [Input('speed-input', 'value')]
)
def update_interval(speed):
    if speed is None or speed <= 0:
        return 100
    # Convert samples per second to milliseconds per sample
    return int(1000 / speed)

@callback(
    [Output('time-slider', 'value'),
     Output('interval-component', 'disabled')],
    [Input('interval-component', 'n_intervals'),
     Input('play-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    [State('time-slider', 'value'),
     State('time-slider', 'max')]
)
def update_time(n_intervals, n_clicks_play, n_clicks_reset, current_value, max_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_value, True
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'reset-button':
        return 0, True
    elif trigger_id == 'play-button':
        return current_value, not (n_clicks_play % 2 == 1)
    elif trigger_id == 'interval-component':
        new_value = current_value + 1
        if new_value > max_value:
            return 0, True
        return new_value, False
    
    return current_value, True

# SVM model training and prediction callbacks
@callback(
    [Output('training-status', 'children'),
     Output('training-features', 'data'),
     Output('training-labels', 'data')],
    [Input('add-sample-button', 'n_clicks')],
    [State('state-label-dropdown', 'value'),
     State('eeg-data', 'data'),
     State('eda-data', 'data'),
     State('time-slider', 'value'),
     State('window-size-slider', 'value'),
     State('training-features', 'data'),
     State('training-labels', 'data')]
)
def add_training_sample(n_clicks, label, eeg_data, eda_data, time_idx, window_size, stored_features, stored_labels):
    if n_clicks is None or n_clicks == 0:
        return "Ready to collect training samples", [], []
    
    if eeg_data is None or eda_data is None:
        return "No data available", stored_features or [], stored_labels or []
    
    # Convert data to numpy arrays
    eeg_array = np.array(eeg_data)
    eda_array = np.array(eda_data)
    
    # Extract features for current window
    window_samples = int(window_size * 250)  # Convert seconds to samples
    features = extract_features(eeg_array[int(time_idx*250):], eda_array[int(time_idx*500):], 
                              window_size=window_samples, overlap=0)
    
    if len(features) == 0:
        return "Error extracting features", stored_features or [], stored_labels or []
    
    # Update stored features and labels
    new_features = features[0:1]  # Take only first window
    new_features_list = stored_features + new_features.tolist() if stored_features else new_features.tolist()
    new_labels_list = stored_labels + [label] if stored_labels else [label]
    
    return f"Added sample for state: {label}", new_features_list, new_labels_list

@callback(
    [Output('state-prediction-plot', 'figure'),
     Output('prediction-text', 'children')],
    [Input('training-features', 'data'),
     Input('training-labels', 'data'),
     Input('train-model-button', 'n_clicks'),
     Input('time-slider', 'value')],
    [State('eeg-data', 'data'),
     State('eda-data', 'data'),
     State('window-size-slider', 'value')]
)
def update_prediction(features, labels, train_clicks, time_idx, eeg_data, eda_data, window_size):
    if not features or not labels or train_clicks is None or train_clicks == 0:
        return go.Figure(), "Train the model first"
    
    try:
        # Train model if button clicked
        if train_clicks > 0:
            train_score, test_score = train_svm_model(np.array(features), labels)
            
        # Make prediction for current window
        prediction, probabilities = predict_mental_state(
            np.array(eeg_data), 
            np.array(eda_data), 
            window_size=int(window_size * 250)
        )
        
        if prediction is None:
            return go.Figure(), "Could not make prediction"
        
        # Create probability bar plot
        fig = go.Figure()
        states = ['relaxed', 'focused', 'stressed']
        fig.add_trace(go.Bar(
            x=states,
            y=probabilities,
            marker_color=['green', 'blue', 'red']
        ))
        
        fig.update_layout(
            title='Mental State Probabilities',
            xaxis_title='State',
            yaxis_title='Probability',
            yaxis_range=[0, 1]
        )
        
        return fig, f"Current mental state: {prediction}"
        
    except Exception as e:
        return go.Figure(), f"Error: {str(e)}"

def generate_analysis_report(eeg_data, eda_data, file_paths):
    """Generate a comprehensive analysis report for EEG and EDA data.
    
    Args:
        eeg_data (numpy.ndarray): Processed EEG data
        eda_data (numpy.ndarray): Processed EDA data
        file_paths (dict): Dictionary containing file paths for EEG and EDA data
    
    Returns:
        str: HTML formatted analysis report
    """
    import numpy as np
    from scipy import signal
    
    # Constants
    fs_eeg = 250  # EEG sampling rate
    fs_eda = 500  # EDA sampling rate
    
    # EEG Analysis
    def calculate_band_power(data, fs, band):
        """Calculate power in a specific frequency band."""
        freqs, psd = signal.welch(data, fs, nperseg=256)
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.mean(psd[idx])
    
    # Calculate EEG frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 40)
    }
    
    band_powers = {band: calculate_band_power(eeg_data, fs_eeg, freq_range) 
                  for band, freq_range in bands.items()}
    
    # Calculate Alpha/Theta ratio (indicator of cognitive performance)
    alpha_theta_ratio = band_powers['Alpha'] / band_powers['Beta']
    
    # EDA Analysis
    # Calculate Skin Conductance Level (SCL) - tonic component
    scl = np.mean(eda_data)
    
    # Calculate Skin Conductance Response (SCR) frequency
    # Define SCR as peaks above threshold
    peaks, _ = signal.find_peaks(eda_data, height=0.1, distance=fs_eda)
    scr_freq = len(peaks) / (len(eda_data) / fs_eda)  # SCRs per second
    
    # Signal Quality Assessment
    eeg_quality = {
        'variance': np.var(eeg_data),
        'max_amplitude': np.max(np.abs(eeg_data)),
        'zero_crossings': np.sum(np.diff(np.signbit(eeg_data)))
    }
    
    eda_quality = {
        'variance': np.var(eda_data),
        'drift': np.mean(np.diff(eda_data)),
        'noise_ratio': np.std(np.diff(eda_data)) / np.mean(eda_data)
    }
    
    # Generate HTML report
    report = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px;">
        <h2>Analysis Report</h2>
        
        <h3>Data Sources</h3>
        <p>EEG File: {os.path.basename(file_paths['eeg'])}<br>
        EDA File: {os.path.basename(file_paths['eda'])}</p>
        
        <h3>EEG Analysis</h3>
        <h4>Frequency Band Power</h4>
        <ul>
            {''.join([f"<li><strong>{band}:</strong> {power:.2e}</li>" 
                     for band, power in band_powers.items()])}
        </ul>
        <p><strong>Alpha/Theta Ratio:</strong> {alpha_theta_ratio:.2f}</p>
        
        <h3>EDA Analysis</h3>
        <p><strong>Skin Conductance Level (SCL):</strong> {scl:.2f} µS</p>
        <p><strong>SCR Frequency:</strong> {scr_freq:.2f} per second</p>
        
        <h3>Signal Quality Assessment</h3>
        <h4>EEG Signal</h4>
        <ul>
            <li>Variance: {eeg_quality['variance']:.2e}</li>
            <li>Max Amplitude: {eeg_quality['max_amplitude']:.2f}</li>
            <li>Zero Crossings: {eeg_quality['zero_crossings']}</li>
        </ul>
        <h4>EDA Signal</h4>
        <ul>
            <li>Variance: {eda_quality['variance']:.2e}</li>
            <li>Baseline Drift: {eda_quality['drift']:.2e}</li>
            <li>Noise Ratio: {eda_quality['noise_ratio']:.2f}</li>
        </ul>
        
        <h3>Physiological State Interpretation</h3>
        <p>{interpret_physiological_state(band_powers, scl, scr_freq)}</p>
    </div>
    """
    
    return report

def interpret_physiological_state(band_powers, scl, scr_freq):
    """Interpret the physiological state based on EEG and EDA metrics."""
    
    # Initialize interpretation parts
    arousal_level = "moderate"
    cognitive_state = "neutral"
    stress_indication = "normal"
    
    # Analyze arousal from EDA
    if scl > 10 or scr_freq > 0.5:
        arousal_level = "high"
    elif scl < 2 or scr_freq < 0.1:
        arousal_level = "low"
    
    # Analyze cognitive state from EEG
    alpha_ratio = band_powers['Alpha'] / sum(band_powers.values())
    theta_ratio = band_powers['Theta'] / sum(band_powers.values())
    
    if alpha_ratio > 0.3:
        cognitive_state = "relaxed"
    elif theta_ratio > 0.3:
        cognitive_state = "drowsy"
    elif band_powers['Beta'] / sum(bband_powers.values()) > 0.3:
        cognitive_state = "alert"
    
    # Analyze stress from combined indicators
    if arousal_level == "high" and band_powers['Beta'] > band_powers['Alpha'] * 2:
        stress_indication = "elevated"
    
    return f"""
    Based on the combined EEG and EDA analysis:
    - Arousal level appears to be {arousal_level}
    - Cognitive state indicates {cognitive_state} condition
    - Stress levels are {stress_indication}
    
    Note: This is an automated interpretation and should be considered alongside other behavioral and contextual factors.
    """

def generate_sequential_analysis_report(eeg_data, eda_data, output_folder='processed'):
    """Generate a sequential analysis report with all processing steps.
    
    Parameters:
        eeg_data: Raw EEG data
        eda_data: Raw EDA data
        output_folder: Folder to save report images
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Raw Signal Visualization
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Raw EEG', 'Raw EDA'))
    
    # Plot raw EEG
    for ch in range(eeg_data.shape[1]):
        fig.add_trace(
            go.Scatter(y=eeg_data[:, ch], name=f'EEG Ch{ch+1}'),
            row=1, col=1
        )
    
    # Plot raw EDA
    fig.add_trace(
        go.Scatter(y=eda_data, name='EDA'),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Step 1: Raw Signals',
        height=720,
        width=1280,
        showlegend=True
    )
    fig.write_image(os.path.join(output_folder, '01_raw_signals.png'))
    
    # 2. Filtering Stage
    fs_eeg = 250
    fs_eda = 500
    
    # Apply filters
    eeg_notch = notch_filter(eeg_data, fs=fs_eeg)
    eeg_filtered = bandpass_filter(eeg_notch, 0.5, 40, fs_eeg)
    
    # Create filter response visualization
    fig = create_filter_response_plot()
    fig.update_layout(
        title='Step 2: Filter Responses',
        height=720,
        width=1280
    )
    fig.write_image(os.path.join(output_folder, '02_filter_response.png'))
    
    # 3. Time-Frequency Analysis
    window_size = 250  # 1 second
    f, t, Sxx = signal.spectrogram(eeg_filtered[:, 0], fs=fs_eeg, nperseg=window_size, noverlap=window_size//2)
    
    fig = go.Figure(data=go.Heatmap(
        x=t,
        y=f,
        z=10 * np.log10(Sxx),
        colorscale='Jet'
    ))
    
    fig.update_layout(
        title='Step 3: Time-Frequency Analysis (Channel Fp1)',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        height=720,
        width=1280
    )
    fig.write_image(os.path.join(output_folder, '03_timefreq.png'))
    
    # 4. Band Power Analysis
    band_fig, avg_fig, analysis_fig = create_band_powers_plot(eeg_filtered, 0, window_size)
    
    band_fig.update_layout(
        title='Step 4: Frequency Band Analysis',
        height=720,
        width=1280
    )
    band_fig.write_image(os.path.join(output_folder, '04_band_powers.png'))
    
    # 5. EDA Component Analysis
    eda_fig = create_eda_features_plot(eda_data, 0, window_size*2)  # *2 because of EDA sampling rate
    eda_fig.update_layout(
        title='Step 5: EDA Component Analysis',
        height=720,
        width=1280
    )
    eda_fig.write_image(os.path.join(output_folder, '05_eda_analysis.png'))
    
    # 6. Spectral Analysis
    psd_fig = create_spectral_density_plots(eeg_filtered, 0, window_size)
    psd_fig.update_layout(
        title='Step 6: Power Spectral Density Analysis',
        height=720,
        width=1280
    )
    psd_fig.write_image(os.path.join(output_folder, '06_psd_analysis.png'))
    
    # 7. Brain Topography
    topo_fig = create_topography_plot(eeg_filtered, 0, window_size)
    topo_fig.update_layout(
        title='Step 7: Brain Topography',
        height=720,
        width=1280
    )
    topo_fig.write_image(os.path.join(output_folder, '07_topography.png'))
    
    # Create analysis summary text
    summary = f"""Analysis Parameters and Settings:

1. Sampling Rates:
   - EEG: 250 Hz
   - EDA: 500 Hz

2. Window Settings:
   - Length: 1 second (250 samples for EEG, 500 for EDA)
   - Overlap: 50%
   - Reason: Optimal balance between frequency resolution (1 Hz) and temporal precision

3. Filters Applied:
  
   - Notch Filter: 50 Hz (power line interference)
   - Bandpass Filter: 0.5-40 Hz (remove drift and high-frequency noise)
   - Reason: Remove common artifacts while preserving relevant neural oscillations

4. Frequency Bands Analyzed:
  - Delta (0.5-4 Hz): Sleep, deep relaxation
  - Theta (4-8 Hz): Drowsiness, meditation
  - Alpha (8-13 Hz): Relaxed wakefulness
  - Beta (13-30 Hz): Active thinking
  - Gamma (30-40 Hz): Complex cognitive processing

5. EDA Components:
  - Tonic (SCL): Overall arousal level
  - Phasic (SCR): Event-related responses
  - Analysis window: 4 seconds for SCL estimation

6. Power Spectral Density:
  - Methods: Welch's method and Periodogram
  - Reason: Welch's method reduces noise through averaging
   
7. Statistical Measures:
  - Time domain: Mean, SD, IQR
  - Frequency domain: Band powers, ratios
  - Cross-channel: Coherence
  - Reason: Capture both temporal and spectral characteristics

Pipeline Order:
1. Raw signal acquisition
2. Filtering and artifact removal
3. Time-frequency decomposition
4. Band power extraction
5. EDA component separation
6. Spectral analysis
7. Topographical mapping

Note: Settings optimized for cognitive state analysis and artifact removal while preserving physiologically relevant signals.
"""
    
    with open(os.path.join(output_folder, 'analysis_summary.txt'), 'w') as f:
        f.write(summary)
        
    return "Analysis report generated in the 'processed' folder"

def generate_analysis_summary_slides(output_folder='processed'):
    """Generate summary slides with parameter explanations for each analysis step."""
    
    summaries = [
        {
            'title': '1. Data Acquisition and Raw Signals',
            'parameters': {
                'EEG Sampling Rate': '250 Hz - Captures neural oscillations up to 125 Hz (Nyquist)',
                'EDA Sampling Rate': '500 Hz - High resolution for SCR detection',
                'EEG Channels': '8 channels (Fp1, Fp2, F3, F4, C3, C4, P3, P4) - Standard 10-20 system',
                'Signal Resolution': '24-bit for precise amplitude detection'
            },
            'relevance': """
            Raw signal quality is crucial for subsequent analysis:
            - Higher sampling rates enable better temporal resolution
            - Multiple EEG channels allow spatial analysis
            - High resolution captures subtle amplitude changes
            """
        },
        {
            'title': '2. Signal Filtering',
            'parameters': {
                'Notch Filter': '50 Hz - Removes power line interference',
                'Bandpass Filter': '0.5-40 Hz - Preserves relevant brain rhythms',
                'Filter Order': '4th order Butterworth - Optimal flatness in passband',
                'Zero-phase': 'Applied to prevent temporal distortion'
            },
            'relevance': """
            Filtering removes noise while preserving signal integrity:
            - Notch filter essential for clean EEG recording
            - Bandpass range captures all major brain rhythms
            - Butterworth characteristics minimize signal distortion
            """
        },
        {
            'title': '3. Time-Frequency Analysis',
            'parameters': {
                'Window Size': '1 second (250 samples) - Balanced resolution',
                'Overlap': '50% - Smooth transitions between windows',
                'Frequency Range': '0.5-40 Hz - Covers major brain rhythms',
                'Method': 'Short-time Fourier Transform with Hanning window'
            },
            'relevance': """
            Time-frequency analysis reveals dynamic spectral changes:
            - 1s window provides 1 Hz frequency resolution
            - 50% overlap prevents information loss at boundaries
            - Hanning window reduces spectral leakage
            """
        },
        {
            'title': '4. Band Power Analysis',
            'parameters': {
                'Delta': '0.5-4 Hz - Sleep, deep relaxation',
                'Theta': '4-8 Hz - Drowsiness, meditation',
                'Alpha': '8-13 Hz - Relaxed wakefulness',
                'Beta': '13-30 Hz - Active thinking',
                'Gamma': '30-40 Hz - Complex processing'
            },
            'relevance': """
            Band powers indicate different cognitive states:
            - Ratios between bands reveal mental state
            - Alpha/Theta ratio indicates arousal level
            - Beta/Alpha ratio reflects cognitive load
            """
        },
        {
            'title': '5. EDA Component Analysis',
            'parameters': {
                'SCL Window': '4 seconds - Stable baseline estimation',
                'SCR Threshold': '0.1 µS - Standard response detection',
                'Minimum SCR Interval': '0.5 seconds - Prevents false positives',
                'Analysis Window': '2 seconds post-stimulus'
            },
            'relevance': """
            EDA components reveal autonomic nervous system activity:
            - SCL indicates general arousal level
            - SCRs show phasic responses to stimuli
            - Component separation enables detailed analysis
            """
        },
        {
            'title': '6. Spectral Analysis',
            'parameters': {
                'Welch Method': 'Reduces noise through averaging',
                'Segment Length': '256 samples - Good frequency resolution',
                'Overlap': '50% - Smooth spectral estimates',
                'Window': 'Hanning - Reduces spectral leakage'
            },
            'relevance': """
            Spectral analysis reveals frequency content:
            - Welch's method provides stable estimates
            - Multiple segments improve reliability
            - Frequency resolution of ~1 Hz suitable for EEG
            """
        },
        {
            'title': '7. Brain Topography',
            'parameters': {
                'Interpolation': 'Cubic spline - Smooth spatial mapping',
                'Resolution': '100x100 grid - Detailed visualization',
                'Reference': 'Average reference - Balanced activity',
                'Scale': 'Symmetric ±µV - Facilitates interpretation'
            },
            'relevance': """
            Topographical mapping shows spatial distribution:
            - Reveals regional brain activity patterns
            - Helps identify focal activity changes
            - Enables comparison with standard patterns
            """
        }
    ]
    
    for i, summary in enumerate(summaries, 1):
        fig = go.Figure()
        
        # Add title
        fig.add_annotation(
            text=summary['title'],
            xref="paper", yref="paper",
            x=0.5, y=1,
            showarrow=False,
            font=dict(size=24, color="black"),
            yshift=20
        )
        
        # Add parameters
        param_text = "<br>".join([f"<b>{k}:</b> {v}" for k, v in summary['parameters'].items()])
        fig.add_annotation(
            text=param_text,
            xref="paper", yref="paper",
            x=0.5, y=0.7,
            showarrow=False,
            font=dict(size=14),
            align="left",
            bordercolor="black",
            borderwidth=1,
            borderpad=10,
            bgcolor="white"
        )
        
        # Add relevance
        fig.add_annotation(
            text=summary['relevance'],
            xref="paper", yref="paper",
            x=0.5, y=0.3,
            showarrow=False,
            font=dict(size=14),
            align="left",
            bordercolor="black",
            borderwidth=1,
            borderpad=10,
            bgcolor="white"
        )
        
        # Update layout for 16:9 aspect ratio
        fig.update_layout(
            width=1920,
            height=1080,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='lightgray'
        )
        
        # Save figure
        fig.write_image(os.path.join(output_folder, f'{i:02d}_summary_{summary["title"].lower().replace(" ", "_")}.png'))
    
    return "Summary slides generated in the processed folder"

def run_complete_analysis(eeg_data, eda_data, output_folder='processed'):
    """Run complete analysis pipeline and generate all reports and summaries."""
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Generate sequential analysis report with technical visualizations
    print("Generating technical analysis report...")
    generate_sequential_analysis_report(eeg_data, eda_data, output_folder)
    
    # 2. Generate summary slides with parameter explanations
    print("Generating summary slides...")
    generate_analysis_summary_slides(output_folder)
    
    # 3. Create a master summary text file
    summary_text = """EEG/EDA Analysis Pipeline Summary

Data Characteristics:
- EEG: 8-channel recording at 250 Hz
- EDA: Single channel at 500 Hz
- Analysis window: 1 second with 50% overlap

Key Processing Steps:
1. Raw Signal Processing
   - Quality check and artifact identification
   - Optimal window length determination
   - Signal standardization

2. Filtering Pipeline
   - 50 Hz notch filter for power line interference
   - 0.5-40 Hz bandpass for physiological signals
   - Zero-phase filtering to preserve timing

3. Spectral Analysis
   - Welch's method for stable estimates
   - Periodogram for high-resolution analysis
   - Time-frequency analysis for dynamic changes

4. Feature Extraction
   - Band power calculation (δ, θ, α, β, γ)
   - EDA component separation (SCL, SCR)
   - Cross-channel coherence analysis

Analysis Parameters:
- Window Size: 1 second (250 EEG samples, 500 EDA samples)
  Rationale: Balances frequency resolution (1 Hz) with temporal precision
  
- Overlap: 50%
  Rationale: Ensures smooth transitions and prevents information loss
  
- Frequency Bands:
  δ (0.5-4 Hz): Sleep patterns
  θ (4-8 Hz): Drowsiness/meditation
  α (8-13 Hz): Relaxed wakefulness
  β (13-30 Hz): Active thinking
  γ (30-40 Hz): Complex processing

- EDA Features:
  SCL: 4-second windows for baseline
  SCR: 0.1 µS threshold, 0.5s minimum interval

Quality Metrics:
- Signal-to-Noise Ratio (SNR)
- Spectral coherence
- Component separation quality
- Statistical validation

Note: Parameters optimized for cognitive state analysis and physiological monitoring.
Results should be interpreted in context with behavioral observations.
"""
    
    with open(os.path.join(output_folder, 'master_summary.txt'), 'w') as f:
        f.write(summary_text)
    
    return "Complete analysis pipeline executed and documented"

# Start the server
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
