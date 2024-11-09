import tensorflow as tf
import librosa
import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, InputLayer
from model_maker import create_ffnn_model
from pcp_module import pcp_vectorise_segment
from itertools import groupby
import logging
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # Specify GPU device
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Suppress NUMA warnings
os.environ['TF_CPP_NUMA_POLICY'] = '0'    # Disable NUMA warnings

# Configure Python logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# Disable all warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logging even further
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Optional: Suppress specific NVIDIA/CUDA warnings if they appear
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

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
   

def predict_chord(pcp_vector, model):
    """
    Predict chord from PCP vector using the trained model
    Returns the predicted chord label
    """
    chord_list = ['Cmaj', 'Cmin', 'C#maj', 'C#min', 'Dmaj', 'Dmin', 'D#maj', 'D#min', 
                  'Emaj', 'Emin', 'Fmaj', 'Fmin', 'F#maj', 'F#min', 'Gmaj', 'Gmin', 
                  'G#maj', 'G#min', 'Amaj', 'Amin', 'A#maj', 'A#min', 'Bmaj', 'Bmin']
    
    # Reshape PCP vector for model input
    pcp_vector = np.array(pcp_vector).reshape(1, -1)
    
    # Get model prediction
    prediction = model.predict(pcp_vector, verbose=0)
    chord_index = np.argmax(prediction)
    
    return chord_list[chord_index]

def apply_audio_filters(audio_data, sr):
    """Apply audio filters to clean the signal"""
    # Apply a bandpass filter (keeping frequencies between 50Hz and 2000Hz)
    y_filtered = librosa.effects.preemphasis(audio_data)
    
    # Apply HPSS (Harmonic-Percussive Source Separation)
    y_harmonic, _ = librosa.effects.hpss(y_filtered)
    
    return y_harmonic

def group_consecutive_chords(times, chords):
    """Group consecutive identical chords and their time intervals"""
    grouped_segments = []
    
    # Create pairs of (time, chord)
    chord_segments = list(zip(times[:-1], times[1:], chords))
    
    # Group by chord
    for chord, group in groupby(chord_segments, key=lambda x: x[2]):
        group_list = list(group)
        start_time = group_list[0][0]
        end_time = group_list[-1][1]
        grouped_segments.append((start_time, end_time, chord))
    
    return grouped_segments

def infer_chords(audio_file, model_weights_path):
    """
    Main inference function that processes audio and returns chord predictions
    """
    # Load the audio file
    print(f"Loading audio file: {audio_file}")
    y, sr = librosa.load(audio_file)
    
    # Get tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    print(f"Detected tempo: {tempo} BPM")
    
    # Convert beat frames to time
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Generate half-beat times by interpolating between beats
    half_beat_times = []
    for i in range(len(beat_times) - 1):
        start_time = beat_times[i]
        end_time = beat_times[i + 1]
        mid_time = start_time + (end_time - start_time) / 2
        half_beat_times.extend([start_time, mid_time])
    # Add the last beat time
    half_beat_times.append(beat_times[-1])
    
    # Convert to numpy array for easier handling
    times = np.array(half_beat_times)
    
    # Load trained model
    model = load_trained_model(model_weights_path)
    
    # Process each segment
    predictions = []
    for i in range(len(times) - 1):
        start_time = times[i]
        end_time = times[i + 1]
        
        # Convert times to sample indices
        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)
        
        # Extract segment
        segment = y[start_idx:end_idx]
        
        # Apply filters
        filtered_segment = apply_audio_filters(segment, sr)
        
        # Get PCP vector
        pcp_vector_str = pcp_vectorise_segment(filtered_segment, sr, f"segment_{start_time}")
        pcp_vector = [float(x) for x in pcp_vector_str.strip('[]').split(',')]
        
        # Predict chord
        chord = predict_chord(pcp_vector, model)
        predictions.append(chord)
    
    # Group consecutive identical chords
    grouped_segments = group_consecutive_chords(times, predictions)
    
    return grouped_segments

def format_time(seconds):
    """Format time in seconds to MM:SS.mmm"""
    minutes = int(seconds // 60)
    seconds_remainder = seconds % 60
    return f"{minutes:02d}:{seconds_remainder:06.3f}"

if __name__ == "__main__":
    audio_file = "0001_infer.mp3"
    model_weights_path = "model.h5"
    
    # Run inference
    chord_segments = infer_chords(audio_file, model_weights_path)
    
    # Display results
    print("\nPredicted Chord Progression:")
    print("-----------------------------")
    for start_time, end_time, chord in chord_segments:
        print(f"{format_time(start_time)} - {format_time(end_time)}: {chord}")