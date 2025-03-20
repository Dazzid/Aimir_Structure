# extract_full_midi.py
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import librosa
import numpy as np
import os

def extract_full_song(audio_path, output_midi_path, verbose=False):
    """
    Extract notes from the full audio file using basic-pitch
    with proper configuration to ensure the entire song is processed
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file to process
    output_midi_path : str
        Path where the MIDI file will be saved
        
    Returns:
    --------
    tuple
        (model_output, midi_data, note_events) from basic-pitch
    """
    if verbose: print(f"Extracting notes from full audio: {audio_path}")
    
    # Load audio file info to check duration
    y, sr = librosa.load(audio_path, sr=None, duration=10)  # Just load a small part to get the sample rate
    duration = librosa.get_duration(filename=audio_path)
    if verbose: print(f"Audio duration: {duration:.2f} seconds, Sample rate: {sr} Hz")
    
    # Configure basic-pitch for full song extraction
    model_output, midi_data, note_events = predict(
        audio_path, 
        ICASSP_2022_MODEL_PATH,
        onset_threshold=0.5,   # Default is 0.5, lower for more notes
        frame_threshold=0.3,   # Default is 0.3, lower for more notes
        minimum_note_length=58,  # In ms, default is 58ms
        minimum_frequency=30,  # In Hz, default is 30Hz
        maximum_frequency=3000,  # In Hz, default is 3000Hz
        multiple_pitch_bends=False,  # Default is False
        melodia_trick=True  # Default is True, helps with melody extraction
    )
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_midi_path), exist_ok=True)
    
    # Save the MIDI file
    midi_data.write(output_midi_path)
    
    # Count total notes extracted
    # The structure depends on the type of MIDI object returned
    # PrettyMIDI uses instruments instead of tracks
    total_instruments = len(midi_data.instruments)
    total_notes = 0
    for instrument in midi_data.instruments:
        total_notes += len(instrument.notes)
    if verbose: print(f"Extracted {total_notes} notes across {total_instruments} instruments to {output_midi_path}")
    
    return model_output, midi_data, note_events