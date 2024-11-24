import tensorflow as tf
import librosa
import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, InputLayer
from itertools import groupby
from operator import itemgetter
from better_onset import improved_beat_tracking
from robust_model_maker import create_ffnn_model
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors
tf.get_logger().setLevel('ERROR')

def load_trained_model(model_path):
    """
    Load the model with trained weights.
    For inference, we only need the model architecture and weights, not the optimizer state.
    """
    try:
        # First try to load as a complete model
        model = load_model(model_path)
    except:
        # If that fails, create new model and load just the weights
        model = create_ffnn_model()
        # Load weights without optimizer state
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
    
    # Recompile the model for inference only (no training needed)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    return model

def extract_robust_features(segment, sr, filename=""):
    """
    Extract robust features from audio segment including:
    - 12 PCP (median aggregated)
    - 6 Tonnetz features
    - 6 Spectral contrast features
    - 20 MFCCs
    - 1 Zero crossing rate
    
    Args:
        segment: Audio time series
        sr: Sampling rate
        filename: Original filename for reporting
    
    Returns:
        numpy.ndarray: 45-dimensional feature vector
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        # Pre-emphasis to enhance high frequencies
        y_pre = librosa.effects.preemphasis(segment, coef=0.97)
        
        # Harmonic-Percussive Source Separation
        y_harmonic, y_percussive = librosa.effects.hpss(
            y_pre,
            kernel_size=31,
            margin=8.0,
            power=2.0
        )
        
        # 1. Chromagram from harmonic component (12 features)
        chromagram = librosa.feature.chroma_cqt(
            y=y_harmonic,
            sr=sr,
            hop_length=512,
            n_chroma=12,
            threshold=0.05,
            norm=2
        )
        chroma_features = np.median(chromagram, axis=1)
        
        # 2. Tonnetz features (6 features)
        tonnetz = librosa.feature.tonnetz(
            y=y_harmonic,
            sr=sr,
            chroma=chromagram
        )
        tonnetz_features = np.mean(tonnetz, axis=1)
        
        # 3. Spectral Contrast (6 features)
        S = np.abs(librosa.stft(y_harmonic, n_fft=2048, hop_length=512))
        contrast = librosa.feature.spectral_contrast(
            S=S,
            sr=sr,
            n_bands=6,
            fmin=200.0,
            quantile=0.02,
            linear=True
        )
        contrast_features = np.mean(contrast, axis=1)
        
        # 4. MFCC (20 features)
        mfcc = librosa.feature.mfcc(
            y=y_harmonic,
            sr=sr,
            n_mfcc=20,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            fmin=0,
            fmax=sr/2
        )
        mfcc_features = np.mean(mfcc, axis=1)
        
        # 5. Zero Crossing Rate (1 feature)
        zcr = librosa.feature.zero_crossing_rate(
            y=y_harmonic,
            frame_length=2048,
            hop_length=512
        )
        zcr_feature = np.mean(zcr)
        
        # Combine all features
        all_features = np.concatenate([
            chroma_features,      # 12 features
            tonnetz_features,     # 6 features
            contrast_features,    # 6 features
            mfcc_features,        # 20 features
            [zcr_feature]         # 1 feature
        ])
        
        # Normalize features
        all_features = librosa.util.normalize(all_features)
        return all_features
        
    except Exception as e:
        print(f"Error extracting features from segment {filename}: {str(e)}")
        return None

def predict_chord(features, model):
    """
    Predict chord from feature vector using the trained model.
    
    Args:
        features (numpy.ndarray): 45-dimensional feature vector
        model (tensorflow.keras.Model): Trained chord prediction model
    
    Returns:
        str: Predicted chord label
    """
    chord_list = ['Cmaj', 'Cmin', 'C#maj', 'C#min', 'Dmaj', 'Dmin', 'D#maj', 'D#min', 
                  'Emaj', 'Emin', 'Fmaj', 'Fmin', 'F#maj', 'F#min', 'Gmaj', 'Gmin', 
                  'G#maj', 'G#min', 'Amaj', 'Amin', 'A#maj', 'A#min', 'Bmaj', 'Bmin']
    
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features, verbose=0)
    chord_index = np.argmax(prediction)
    
    return chord_list[chord_index]

def apply_audio_filters(audio_data, sr):
    """
    Apply preprocessing filters to clean the audio signal.
    
    Args:
        audio_data (numpy.ndarray): Raw audio signal
        sr (int): Sampling rate
    
    Returns:
        numpy.ndarray: Filtered audio signal
    """
    # Apply pre-emphasis filter
    y_filtered = librosa.effects.preemphasis(audio_data)
    
    # Apply harmonic-percussive separation
    y_harmonic, _ = librosa.effects.hpss(y_filtered)
    
    return y_harmonic

def group_consecutive_chords(times, chords):
    """
    Group consecutive segments with the same chord prediction.
    
    Args:
        times (numpy.ndarray): Array of segment boundary timestamps
        chords (list): List of chord predictions
    
    Returns:
        list: List of (start_time, end_time, chord) tuples
    """
    grouped_segments = []
    chord_segments = list(zip(times[:-1], times[1:], chords))
    
    for chord, group in groupby(chord_segments, key=lambda x: x[2]):
        group_list = list(group)
        start_time = group_list[0][0]
        end_time = group_list[-1][1]
        grouped_segments.append((start_time, end_time, chord))
    
    return grouped_segments

def infer_chords(audio_file, model_weights_path):
    """
    Process an audio file to detect and predict its chord progression.
    
    Args:
        audio_file (str): Path to input audio file
        model_weights_path (str): Path to model weights
    
    Returns:
        list: List of (start_time, end_time, chord) tuples
    """
    print(f"Loading audio file: {audio_file}")
    y, sr = librosa.load(audio_file)
    beat_times, tempo = improved_beat_tracking(y, sr)
    print(f"Detected tempo: {tempo:.1f} BPM")
    print(f"Found {len(beat_times)} beats")

    
    # Generate half-beat times
    half_beat_times = []
    for i in range(len(beat_times) - 1):
        start_time = beat_times[i]
        end_time = beat_times[i + 1]
        mid_time = start_time + (end_time - start_time) / 2
        half_beat_times.extend([start_time, mid_time])
    half_beat_times.append(beat_times[-1])
    times = np.array(half_beat_times)
    
    # Load model
    model = load_trained_model(model_weights_path)
    
    # Process each segment
    predictions = []
    for i in range(len(times) - 1):
        start_time = times[i]
        end_time = times[i + 1]
        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)
        
        # Extract segment and apply filtering
        segment = y[start_idx:end_idx]
        filtered_segment = apply_audio_filters(segment, sr)
        
        # Extract features and predict
        features = extract_robust_features(filtered_segment, sr, f"segment_{start_time}")
        if features is not None:
            chord = predict_chord(features, model)
            predictions.append(chord)
        else:
            # If feature extraction fails, use previous chord or "N.C." if first segment
            predictions.append(predictions[-1] if predictions else "N.C.")
    
    # Group consecutive identical chords
    grouped_segments = group_consecutive_chords(times, predictions)
    
    return grouped_segments

def format_time(seconds: float):
    """Format time in seconds to MM:SS.mmm"""
    minutes = int(seconds // 60)
    seconds_remainder = seconds % 60
    return f"{minutes:02d}:{seconds_remainder:06.3f}"

if __name__ == "__main__":
    audio_file = "infer3.mp3"
    model_weights_path = "robust_model_80_20_split.h5"
    
    # Run inference
    chord_segments = infer_chords(audio_file, model_weights_path)
    
    # Display results
    print("\nPredicted Chord Progression:")
    print("-----------------------------")
    for start_time, end_time, chord in chord_segments:
        print(f"{format_time(start_time)} - {format_time(end_time)}: {chord}")