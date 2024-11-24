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
from better_onset import improved_beat_tracking
from tqdm import tqdm

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
    Process an audio file to detect and predict its chord progression using improved beat tracking.
    
    Parameters:
        audio_file (str): Path to the input audio file
        model_weights_path (str): Path to the saved model weights
    
    Returns:
        list: List of tuples (start_time, end_time, chord) representing the detected chord progression
    
    Raises:
        FileNotFoundError: If audio_file or model_weights_path doesn't exist
        RuntimeError: If beat detection or chord prediction fails
    """
    print(f"Loading audio file: {audio_file}")
    y, sr = librosa.load(audio_file)
    
    # Use improved beat tracking instead of librosa's
    beat_times, tempo = improved_beat_tracking(y, sr)
    print(f"Detected tempo: {tempo:.1f} BPM")
    print(f"Found {len(beat_times)} beats")
    
    # Generate half-beat times for finer chord detection
    half_beat_times = []
    for i in range(len(beat_times) - 1):
        start_time = beat_times[i]
        end_time = beat_times[i + 1]
        mid_time = start_time + (end_time - start_time) / 2
        half_beat_times.extend([start_time, mid_time])
    
    # Add the last beat
    if len(beat_times) > 0:
        half_beat_times.append(beat_times[-1])
        
        # Add one more half beat after the last beat if possible
        if beat_times[-1] + (end_time - start_time) / 2 < len(y) / sr:
            half_beat_times.append(beat_times[-1] + (end_time - start_time) / 2)
    
    times = np.array(half_beat_times)
    
    if len(times) < 2:
        raise RuntimeError("Insufficient beats detected for chord analysis")
    
    # Load the trained model
    try:
        model = load_trained_model(model_weights_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
    
    # Process each segment
    predictions = []
    print("Processing segments for chord detection...")
    for i in tqdm(range(len(times) - 1)):
        start_time = times[i]
        end_time = times[i + 1]
        
        # Convert times to samples
        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, len(y))
        
        if end_idx <= start_idx:
            continue
            
        # Extract segment
        segment = y[start_idx:end_idx]
        
        # Apply audio filtering
        filtered_segment = apply_audio_filters(segment, sr)
        
        try:
            # Extract PCP vector
            pcp_vector_str = pcp_vectorise_segment(filtered_segment, sr, f"segment_{start_time}")
            if pcp_vector_str is None:
                continue
                
            pcp_vector = np.array([float(x) for x in pcp_vector_str.strip('[]').split(',')])
            
            # Predict chord
            chord = predict_chord(pcp_vector, model)
            predictions.append(chord)
            
        except Exception as e:
            print(f"Warning: Failed to process segment at {start_time:.2f}s: {str(e)}")
            if len(predictions) > 0:
                # Use previous chord as fallback
                predictions.append(predictions[-1])
            continue
    
    # Group consecutive identical chords
    grouped_segments = []
    if len(predictions) > 0:
        current_chord = predictions[0]
        current_start = times[0]
        
        for i in range(1, len(predictions)):
            if predictions[i] != current_chord:
                # Add the completed segment
                grouped_segments.append((current_start, times[i], current_chord))
                # Start new segment
                current_chord = predictions[i]
                current_start = times[i]
        
        # Add the last segment
        if len(times) > i:
            grouped_segments.append((current_start, times[i+1], current_chord))
    
    # Post-process the segments
    if len(grouped_segments) > 0:
        # Merge very short segments (less than 0.1 seconds) with their neighbors
        MIN_SEGMENT_DURATION = 0.1
        processed_segments = []
        
        for i, (start, end, chord) in enumerate(grouped_segments):
            if i == 0:
                processed_segments.append((start, end, chord))
                continue
                
            last_start, last_end, last_chord = processed_segments[-1]
            
            # If this segment is very short, merge it with the previous one
            if end - start < MIN_SEGMENT_DURATION:
                if len(processed_segments) > 0:
                    processed_segments[-1] = (last_start, end, last_chord)
            else:
                processed_segments.append((start, end, chord))
        
        grouped_segments = processed_segments
    
    print(f"Detected {len(grouped_segments)} chord changes")
    return grouped_segments

def format_time(seconds):
    """Format time in seconds to MM:SS.mmm"""
    minutes = int(seconds // 60)
    seconds_remainder = seconds % 60
    return f"{minutes:02d}:{seconds_remainder:06.3f}"

if __name__ == "__main__":
    audio_file = "infer3.mp3"
    model_weights_path = "pcpmodel_1000_regular_learning.h5"
    
    # Run inference
    chord_segments = infer_chords(audio_file, model_weights_path)
    
    # Display results
    print("\nPredicted Chord Progression:")
    print("-----------------------------")
    for start_time, end_time, chord in chord_segments:
        print(f"{format_time(start_time)} - {format_time(end_time)}: {chord}")