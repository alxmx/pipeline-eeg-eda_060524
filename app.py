import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, lfilter
import mne
import base64
import io
import json
from datetime import timedelta
import os

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
                # EEG Topography
                dcc.Graph(id='topography-plot', style={'height': '500px'}),
                
                # EEG Time Series
                dcc.Graph(id='eeg-plot', style={'height': '300px'}),
                
                # EDA Signal
                dcc.Graph(id='eda-plot', style={'height': '200px'}),
                
                # Band Powers
                dcc.Graph(id='band-powers-plot', style={'height': '400px'}),
                
                # Average Band Powers
                dcc.Graph(id='avg-powers-plot', style={'height': '400px'}),
                
                # Alpha-Theta Analysis
                dcc.Graph(id='analysis-plot', style={'height': '400px'}),
            ], style={'width': '75%', 'display': 'inline-block', 'padding': '20px'}),
        ], style={'display': 'flex'}),
    ]),
    
    # Hidden divs for storing data
    dcc.Store(id='eeg-data', data=initial_eeg),
    dcc.Store(id='eda-data', data=initial_eda),
    dcc.Store(id='processed-data'),
])

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

@callback(
    [Output('upload-status', 'children'),
     Output('main-content', 'style'),
     Output('eeg-data', 'data'),
     Output('eda-data', 'data')],
    [Input('process-button', 'n_clicks')],
    [State('upload-eeg', 'contents'),
     State('upload-eda', 'contents')]
)
def process_upload(n_clicks, eeg_contents, eda_contents):
    if n_clicks == 0:
        return '', {'display': 'none'}, None, None
    
    if not eeg_contents or not eda_contents:
        return html.Div('Please upload both EEG and EDA files', style={'color': 'red'}), {'display': 'none'}, None, None
    
    try:
        # Parse files
        eeg_data = parse_eeg_contents(eeg_contents)
        eda_data, _ = parse_eda_contents(eda_contents)
        
        # Process signals
        eeg_processed, eda_processed = process_signals(eeg_data, eda_data)
        
        return html.Div('Files processed successfully!', style={'color': 'green'}), {'display': 'block'}, eeg_processed.tolist(), eda_processed.tolist()
    
    except Exception as e:
        return html.Div(f'Error processing files: {str(e)}', style={'color': 'red'}), {'display': 'none'}, None, None

def create_topography_plot(eeg_data, time_idx, window_size=1000):
    """Create a topographic map using plotly with correct 2D channel positions and scale."""
    # Calculate mean activity for the current window
    mean_activity = np.mean(eeg_data[time_idx:time_idx+window_size], axis=0)
    
    # Channel names
    ch_names = [eeg_channels[i] for i in range(8)]
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, match_case=False)
    
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
            power = np.trapz(Pxx[idx], f[idx])
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

@callback(
    [Output('topography-plot', 'figure'),
     Output('eeg-plot', 'figure'),
     Output('eda-plot', 'figure'),
     Output('band-powers-plot', 'figure'),
     Output('avg-powers-plot', 'figure'),
     Output('analysis-plot', 'figure')],
    [Input('time-slider', 'value'),
     Input('window-size-slider', 'value'),
     Input('interval-component', 'n_intervals')],
    [State('eeg-data', 'data'),
     State('eda-data', 'data')]
)
def update_plots(time_idx, window_size_seconds, n_intervals, eeg_data, eda_data):
    if eeg_data is None or eda_data is None:
        return {}, {}, {}, {}, {}, {}
    
    eeg_data = np.array(eeg_data)
    eda_data = np.array(eda_data)
    
    # Convert window size from seconds to samples
    window_size_eeg = int(window_size_seconds * 250)  # EEG sampling rate
    window_size_eda = int(window_size_seconds * 500)  # EDA sampling rate
    
    # Create all plots
    topo_fig = create_topography_plot(eeg_data, time_idx, window_size_eeg)
    eeg_fig = create_eeg_plot(eeg_data, time_idx, window_size_eeg)
    eda_fig = create_eda_plot(eda_data, time_idx, window_size_eda)
    band_fig, avg_fig, analysis_fig = create_band_powers_plot(eeg_data, time_idx, window_size_eeg)
    
    return topo_fig, eeg_fig, eda_fig, band_fig, avg_fig, analysis_fig

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

# --- Auto-load EEG and EDA files from data/raw/ ---
def find_first_file(folder, exts):
    for fname in os.listdir(folder):
        if any(fname.lower().endswith(ext) for ext in exts):
            return os.path.join(folder, fname)
    return None

def load_initial_data():
    eeg_path = find_first_file('data/raw', ['.csv'])
    eda_path = find_first_file('data/raw', ['.txt'])
    eeg_data, eda_data = None, None
    if eeg_path and eda_path:
        # Load EEG
        with open(eeg_path, 'r', encoding='utf-8') as f:
            eeg_contents = f.read()
        eeg_b64 = 'data:text/csv;base64,' + base64.b64encode(eeg_contents.encode('utf-8')).decode('utf-8')
        eeg_data = parse_eeg_contents(eeg_b64)
        # Load EDA
        with open(eda_path, 'r', encoding='utf-8') as f:
            eda_contents = f.read()
        eda_b64 = 'data:text/plain;base64,' + base64.b64encode(eda_contents.encode('utf-8')).decode('utf-8')
        eda_data, _ = parse_eda_contents(eda_b64)
        # Process signals
        eeg_processed, eda_processed = process_signals(eeg_data, eda_data)
        return eeg_processed.tolist(), eda_processed.tolist(), eeg_path, eda_path
    return None, None, None, None

# Load initial data if available
initial_eeg, initial_eda, initial_eeg_path, initial_eda_path = load_initial_data()

if __name__ == '__main__':
    app.run(debug=True) 