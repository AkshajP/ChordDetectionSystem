import pandas as pd
import soundfile as sf
import librosa
import numpy as np
import os
import sys
from tqdm import tqdm
import warnings 


def load_annotations(csv_file):
    """Load and process the annotations CSV file."""
    # Read CSV file with specified column names
    df = pd.read_csv(csv_file, header=None, 
                     names=['start_time', 'bar', 'beat', 'chord'])
    
    # Calculate end times by shifting start times
    df['end_time'] = df['start_time'].shift(-1)
    # For the last segment, we'll need to handle it separately
    df = df.ffill()
    
    return df


def pcp_vectorise_segment(segment, sr, filename):
    """
    Process audio segment to extract harmonic-based chroma features.
    
    Parameters:
        segment: Audio time series
        sr: Sampling rate
        filename: Original filename for reporting
    
    Returns:
        String: {filename}:{vector_str}
    """
    n_fft = 512
    warnings.filterwarnings("ignore", category=UserWarning, module="librosa.core.spectrum")
    try:
        # Harmonic-percussive source separation
        padded = librosa.util.fix_length(segment, size = n_fft)
        y_harmonic, y_percussive = librosa.effects.hpss(segment, n_fft=n_fft)
        
        # Compute CQT-based chromagram from harmonic signal
        chromagram = librosa.feature.chroma_cqt(
            y=y_harmonic,
            sr=sr,
            #norm=None  # Keep the original magnitude
        )
        
        # Reduce the chromagram to a single 12-dimensional vector using median
        chroma_reduced = np.median(chromagram, axis=1)
        
        # Ensure we have a 12-dimensional vector
        assert len(chroma_reduced) == 12, f"Expected 12 dimensions, got {len(chroma_reduced)}"
        
        # Create formatted string of the vector with 6 decimal places
        vector_str = ','.join([f"{x:.6f}" for x in chroma_reduced])
    
        return f"[{vector_str}]"
        
    except Exception as e:
        print(f"Error processing segment {filename}: {str(e)}")

def one_hot_encoder(chord: str) -> list[int]:
    chord_list = ['Cmaj', 'Cmin', 'C#maj', 'C#min', 'Dmaj', 'Dmin', 'D#maj', 'D#min', 
              'Emaj', 'Emin', 'Fmaj', 'Fmin', 'F#maj', 'F#min', 'Gmaj', 'Gmin', 
              'G#maj', 'G#min', 'Amaj', 'Amin', 'A#maj', 'A#min', 'Bmaj', 'Bmin']
    encoding = [0] * 24
    if chord in chord_list:
        encoding[chord_list.index(chord)] = 1
    else:
        raise ValueError(f"Chord '{chord}' not found in chord_list.")
    
    return encoding

def process_audio_and_save_pcp(audio_file_name, dataset_location, annotations_df, output_dir, logging_level=0):
    """
    Slice the audio file according to the annotations

    Parameters:
        audio_file_name: Name of audio file with extension. Do not put . or / before 
        dataset_location: Path till dataset directory
        annotations_df: Pandas dataframe of csv file read
        output_dir: Name of directory in which all vectors are to be stored
        logging_level: Default 0
            0 - None, 1 - Info level
    
    Returns:
        None. Saves pcp vector in csv file with file name as audio name
    
    """
    audio_file_path = os.path.join(dataset_location, audio_file_name)
    try:
        if not os.path.exists(os.path.join(dataset_location,audio_file_name)):
            raise FileNotFoundError(f"Audio file not found: {audio_file_name}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Loading audio file: {audio_file_name}")
        warnings.filterwarnings("ignore", category=UserWarning, module="librosa.core.spectrum")
        y, sr = librosa.load(audio_file_path, sr=None)
        print(f"Audio loaded with sampling rate: {sr} Hz")
        
        # Calculate samples per microsecond for precision checking
        samples_per_microsecond = sr / 1_000_000
        print(f"Samples per microsecond: {samples_per_microsecond}")
        
        with open(f'./{output_dir}/{audio_file_name}_pcpvectors.csv', 'w') as file:
            # Process all rows except the last one since last file will be of 0 bytes
            for idx, row in tqdm(annotations_df.iloc[:-1].iterrows(), unit=" segments"):
                chord = row['chord'].strip().replace("'", "").replace('"', "") 
                if chord == 'N.C.': # Discard the lines that are corrupt in dataset
                    continue

                # Convert times to sample indices with high precision
                start_idx = int(np.floor(row['start_time'] * sr))
                end_idx = int(np.floor(row['end_time'] * sr))
                
                # Ensure we don't exceed array bounds
                end_idx = min(end_idx, len(y))
                segment = y[start_idx:end_idx] # Slicing
                
                # Create filename with new format: line_chord_start_end.mp3
                filename = f"{idx+1}_{chord}_{row['start_time']:.6f}_{row['end_time']:.6f}.mp3"
                
                # Save the segment with original sampling rate
                # output_path = os.path.join(output_dir, filename)
                # sf.write(output_path, segment, sr)
                
                pcp_vector = pcp_vectorise_segment(segment, sr, filename)
                one_hot_encoded_list = one_hot_encoder(chord)
                file.write(str(one_hot_encoded_list)+ ','+ pcp_vector + '\n')

                if logging_level == 1:
                # Print detailed timing information
                    print(f"Processed: {filename}")
                    print(f"Segment info:")
                    print(f"  Start time: {row['start_time']:.6f} seconds")
                    print(f"  End time: {row['end_time']:.6f} seconds")
                    # segment_duration = len(segment) / sr
                    # print(f"  Duration: {segment_duration:.6f} seconds")
                    print(f"  Samples: {len(segment)}")

        print(f"Processed: {audio_file_name}")      
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    audio_file_name = '0001_mix.mp3'
    dataset_location = "./datasetmini/audio-mixes/"
    output_dir = 'modifications'
    annotations_dir_loc = "./datasetmini/annotations/"
    annotations_file_name = "0001_beatinfo.csv"

    annotations_file = os.path.join(annotations_dir_loc, annotations_file_name)
    annotations_df = load_annotations(annotations_file)
    process_audio_and_save_pcp(audio_file_name, dataset_location, annotations_df, output_dir)