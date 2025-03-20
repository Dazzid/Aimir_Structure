# extract_full_midi.py
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import librosa
import numpy as np
import os
import soundfile as sf
import tempfile

def extract_full_song(audio_path, output_midi_path, verbose=False):
    """
    Extract notes from the full audio file using basic-pitch,
    after normalizing the audio to ensure consistent amplitude.

    Parameters:
    -----------
    audio_path : str
        Path to the audio file to process.
    output_midi_path : str
        Path where the MIDI file will be saved.
    verbose : bool
        If True, prints debug information.
        
    Returns:
    --------
    tuple
        (model_output, midi_data, note_events) from basic-pitch.
    """
    if verbose: 
        print(f"Extracting notes from full audio: {audio_path}")
    
    # Load the entire audio file
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    if verbose: 
        print(f"Audio duration: {duration:.2f} seconds, Sample rate: {sr} Hz")
    
    # Normalize the audio: scale signal so max absolute amplitude is 1.
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y_normalized = y / max_amp
        if verbose: 
            print("Audio normalized successfully.")
    else:
        y_normalized = y
        if verbose:
            print("Audio normalization skipped due to zero amplitude.")
    
    # Write the normalized audio to a temporary file (this file won't clutter your folders)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        normalized_audio_path = tmp.name
    sf.write(normalized_audio_path, y_normalized, sr)
    if verbose:
        print(f"Normalized audio temporarily saved to {normalized_audio_path}")
    
    try:
        # Process the normalized audio using basic-pitch's predict function
        model_output, midi_data, note_events = predict(
            normalized_audio_path, 
            ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5,      # Default is 0.5, lower for more notes
            frame_threshold=0.3,      # Default is 0.3, lower for more notes
            minimum_note_length=58,   # In ms, default is 58ms
            minimum_frequency=30,     # In Hz, default is 30Hz
            maximum_frequency=3000,   # In Hz, default is 3000Hz
            multiple_pitch_bends=False,  # Default is False
            melodia_trick=True        # Default is True, helps with melody extraction
        )
    finally:
        # Remove the temporary file so it doesn't clutter your folders
        os.remove(normalized_audio_path)
        if verbose:
            print(f"Temporary file {normalized_audio_path} deleted.")
    
    # Ensure the output directory exists and save the MIDI file
    os.makedirs(os.path.dirname(output_midi_path), exist_ok=True)
    midi_data.write(output_midi_path)
    
    # Count total notes extracted (assuming midi_data is a PrettyMIDI object)
    total_instruments = len(midi_data.instruments)
    total_notes = sum(len(instrument.notes) for instrument in midi_data.instruments)
    if verbose:
        print(f"Extracted {total_notes} notes across {total_instruments} instruments to {output_midi_path}")
    
    return model_output, midi_data, note_events
