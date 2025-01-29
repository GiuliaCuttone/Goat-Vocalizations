"""
Goat vocalizations
Università degli studi di Milano

@author: Giulia Cuttone
"""

import ntpath

import librosa
import librosa.feature as ftr

import numpy as np
import pandas as pd


'----------------------------------------------------------------------------------------------------------------'

def extract_features(sequence, s_rate=16000):
    
    """
    Temporal and spectral audio features extraction.
    ------------------------------------------------
    Parameters:
        sequence (array): Audio file sequence
        label (string): Audio file label.
        s_rate (int, optional): Sampling rate; default is 16000 Hz.
    ---------------------------------------------------------------
    Returns:
        numpy.ndarray: unidimensional array containing the following features:
            - mfccs: 40 Mel-frequency cepstral coefficients (MFCCs)
            - rms: Root mean square
            - sp_ce: Spectral centroid
            - sp_bw: Spectral bandwidth
            - sp_ro: Spectral roll-off
            - zero_cr: Zero crossing rate
    """
    
    mfccs = ftr.mfcc(y=sequence, sr=s_rate, n_mfcc=40)     # Mel-frequency cepstral coefficients
    rms = ftr.rms(y=sequence)                                   # root mean square
    sp_ce = ftr.spectral_centroid(y=sequence, sr=s_rate)        # spectral centroid
    sp_bw = ftr.spectral_bandwidth(y=sequence, sr=s_rate)       # spectral bandwidth
    sp_ro = ftr.spectral_rolloff(y=sequence, sr=s_rate)         # spectral rolloff
    zero_cr = ftr.zero_crossing_rate(y=sequence)                # zero crossing rate
    
    return np.hstack([mfccs.mean(axis=1), rms.mean(), sp_ce.mean(), sp_bw.mean(), sp_ro.mean(), zero_cr.mean()])

'---------------------------------------------------------------------------------------------------------------'


"Main code:"

# Load audio files
file_path = './VOCAPRA_all/'
file_list = librosa.util.find_files(file_path, ext=['wav'])

# Initialize an empty list to store features and classes
data = []

for file in file_list:
    
    # Sound name
    sound_name = ntpath.basename(file)
       
    # Sound file              
    sound_file, sample_rate = librosa.load(file, sr=None)

    # Sound class
    sound_class = sound_name.split('.')[0]

    # Extract features from the audio file
    features = extract_features(sound_file, sample_rate)

    data.append(features.tolist() + [sound_class])


# Create a DataFrame with the extracted features and labels
column_names = [f'MFCCs_{i+1}' for i in range(40)] + ['RMS', 'Spectral_Centroid', 'Spectral_Bandwidth', 'Spectral_Rolloff', 'Zero_Crossing_Rate', 'Class']
df = pd.DataFrame(data, columns=column_names)

# Display the first few rows of the DataFrame
print(df.head())

# Save the DataFrame to a CSV file
df.to_csv('Vocapra_dataset.csv', index=False)