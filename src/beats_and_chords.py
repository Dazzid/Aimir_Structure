import numpy as np
import librosa
import pretty_midi
import matplotlib.pyplot as plt
from IPython.display import Audio
import os
from chordAnalyzer import (load_midi_notes, clean_bass_line, identify_unique_chord, plot_chord_timeline_multirow)

def extract_beats_with_shift(audio_path, ms_shift=20):
    """
    Extract beats from audio using librosa's beat tracking algorithm
    and apply a millisecond shift to align with MIDI.
    
    Args:
        audio_path: Path to the audio file
        ms_shift: Milliseconds to shift beats earlier
        
    Returns:
        y: Audio data
        sr: Sample rate
        beats_shifted: Beat positions in seconds, shifted earlier by ms_shift
    """
    print(f"Extracting beats from {audio_path} with {ms_shift}ms shift")
    
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # Extract beats
    tempo, beats_static = librosa.beat.beat_track(y=y, sr=sr, units='time', trim=False)
    
    # Convert milliseconds to seconds
    seconds_shift = ms_shift / 1000
    
    # Shift the beat times earlier
    beats_shifted = beats_static - seconds_shift
    
    # Ensure no negative beat times
    beats_shifted = np.maximum(beats_shifted, 0)
    
    print(f"Tempo estimate: {tempo:.2f} BPM, extracted {len(beats_shifted)} beats")
    
    return y, sr, beats_shifted, tempo

def convert_beats_to_beat_positions(beats_shifted, offset=0.0):
    """
    Convert absolute beat timings to beat positions (e.g., 1.0, 2.0, 3.0...)
    
    Args:
        beats_shifted: Array of beat times in seconds
        offset: Optional offset to shift all beat positions
        
    Returns:
        Array of beat positions (0, 1, 2, 3...)
    """
    # Create beat positions (0, 1, 2, 3...)
    beat_positions = np.arange(len(beats_shifted)) + offset
    
    return beat_positions

def create_beat_time_mapping(beats_shifted):
    """
    Create a mapping from beat positions to time positions
    and a function to convert arbitrary beat positions to time
    
    Args:
        beats_shifted: Array of beat times in seconds
        
    Returns:
        beat_to_time: Function that converts beat positions to time
    """
    beat_positions = np.arange(len(beats_shifted))
    
    # Create interpolation function to map any beat position to time
    from scipy.interpolate import interp1d
    beat_to_time = interp1d(beat_positions, beats_shifted, 
                            bounds_error=False, fill_value="extrapolate")
    
    return beat_to_time

def convert_midi_timing_to_beat_positions(midi_notes, beat_to_time, beat_positions):
    """
    Convert MIDI note timings (which are in beats) to match the extracted beat positions
    
    Args:
        midi_notes: List of MIDI notes with 'start' and 'end' in beats
        beat_to_time: Function to convert beat positions to time
        beat_positions: Array of beat positions
        
    Returns:
        Updated MIDI notes with aligned start and end times
    """
    # Create time-to-beat mapping (inverse of beat_to_time)
    from scipy.interpolate import interp1d
    beat_times = beat_to_time(beat_positions)
    time_to_beat = interp1d(beat_times, beat_positions, 
                            bounds_error=False, fill_value="extrapolate")
    
    # Update each note with aligned start and end times
    updated_notes = []
    
    for note in midi_notes:
        # Convert start and end from beat to time domain
        start_time = beat_to_time(note['start'])
        end_time = beat_to_time(note['end'])
        
        # Create a copy of the note
        new_note = note.copy()
        
        # Update with aligned times
        new_note['start_time'] = start_time
        new_note['end_time'] = end_time
        
        # Keep original beat positions for reference
        new_note['start_beat'] = note['start']
        new_note['end_beat'] = note['end']
        
        updated_notes.append(new_note)
    
    return updated_notes

def merge_adjacent_identical_chords(chords):
    """
    Merge adjacent chords that have the same chord name to create a cleaner progression.
    
    Args:
        chords: List of chord dictionaries from extract_chords_with_beat_windows
        
    Returns:
        List of merged chord dictionaries
    """
    if not chords:
        return []
    
    # Ensure chords are sorted by beat_start
    chords = sorted(chords, key=lambda x: x['beat_start'])
    
    # Initialize merged list with the first chord
    merged_chords = []
    current_chord = chords[0].copy()
    
    # Process remaining chords
    for next_chord in chords[1:]:
        # If same chord name and adjacent (beat_end == beat_start)
        if (next_chord['chord_name'] == current_chord['chord_name'] and 
            next_chord['beat_start'] == current_chord['beat_end']):
            
            # Extend current chord
            current_chord['beat_end'] = next_chord['beat_end']
            current_chord['time_end'] = next_chord['time_end']
            
            # Average the confidence values when merging
            if 'confidence' in current_chord and 'confidence' in next_chord:
                current_chord_duration = current_chord['time_end'] - current_chord['time_start']
                next_chord_duration = next_chord['time_end'] - next_chord['time_start']
                total_duration = current_chord_duration + next_chord_duration
                
                # Weighted average of confidence based on duration
                current_chord['confidence'] = (
                    (current_chord['confidence'] * current_chord_duration) +
                    (next_chord['confidence'] * next_chord_duration)
                ) / total_duration
        else:
            # Different chord or not adjacent, add current to results and start a new one
            merged_chords.append(current_chord)
            current_chord = next_chord.copy()
    
    # Add the last chord
    merged_chords.append(current_chord)
    
    return merged_chords

def visualize_beats_and_chords(audio_path, beats_shifted, chords, start_beat=0, num_beats=16):
    """
    Visualize audio waveform with beat markers and chord annotations
    
    Args:
        audio_path: Path to the audio file
        beats_shifted: Array of beat times in seconds
        chords: List of chord dictionaries
        start_beat: First beat to show
        num_beats: Number of beats to display
    """
    # Calculate time boundaries from beats
    if start_beat >= len(beats_shifted):
        start_beat = 0
    
    end_beat = min(start_beat + num_beats, len(beats_shifted) - 1)
    
    start_time = beats_shifted[start_beat]
    end_time = beats_shifted[end_beat] if end_beat < len(beats_shifted) else beats_shifted[-1] + 0.5
    
    # Load audio segment
    y, sr = librosa.load(audio_path, offset=start_time, duration=(end_time - start_time))
    
    # Set up plot
    plt.figure(figsize=(15, 8))
    
    # Plot waveform
    times = np.linspace(start_time, end_time, len(y))
    plt.plot(times, y, color='gray', alpha=0.6)
    
    # Add beat markers
    beat_times = beats_shifted[start_beat:end_beat+1]
    for beat_idx, beat_time in enumerate(beat_times):
        plt.axvline(x=beat_time, color='r', linestyle='--', alpha=0.7, 
                   label='Beat' if beat_idx == 0 else "")
        
        # Add beat number
        plt.text(beat_time, plt.ylim()[1] * 0.9, f"{start_beat + beat_idx}", 
                fontsize=8, ha='center')
    
    # Filter chords for this segment
    segment_chords = [c for c in chords 
                     if c['beat_end'] > start_beat and c['beat_start'] < end_beat]
    
    # Add chord annotations
    for chord in segment_chords:
        chord_start = max(chord['time_start'], start_time)
        chord_end = min(chord['time_end'], end_time)
        
        # Highlight chord segment
        plt.axvspan(chord_start, chord_end, color='blue', alpha=0.1)
        
        # Add chord name
        plt.text(chord_start + (chord_end - chord_start)/2, plt.ylim()[0] * 0.8, 
                chord['chord_name'], fontsize=10, ha='center', 
                bbox=dict(facecolor='white', alpha=0.8, pad=1))
    
    # Add labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(f'Audio Waveform with Beats and Chords (Beats {start_beat}-{end_beat})')
    
    # Add legend
    plt.legend()
    
    # Show grid and plot
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Return audio snippet for playback
    return Audio(data=y, rate=sr)

def compare_window_sizes(audio_path, midi_bass_path, midi_harmony_path, ms_shift=20, start_bar=0, num_bars=4):
    """
    Compare chord extraction with 1-beat vs 2-beat windows
    
    Args:
        audio_path: Path to audio file
        midi_bass_path: Path to bass MIDI file
        midi_harmony_path: Path to harmony MIDI file
        ms_shift: Millisecond shift to apply
        start_bar: First bar to analyze
        num_bars: Number of bars to analyze
    """
    # Extract beats from audio with shift
    y, sr, beats_shifted, tempo = extract_beats_with_shift(audio_path, ms_shift)
    
    # Load MIDI notes
    bass_notes = load_midi_notes(midi_bass_path, source_name='bass')
    harmony_notes = load_midi_notes(midi_harmony_path, source_name='harmony')
    
    # Clean the bass line (ensure monophonic)
    cleaned_bass = clean_bass_line(bass_notes)
    
    # Combine cleaned bass with harmony notes
    combined_notes = cleaned_bass + harmony_notes
    combined_notes.sort(key=lambda x: x['start'])
    
    # Extract chords with 1-beat windows
    chords_1beat = extract_chords_with_beat_windows(
        combined_notes, beats_shifted, window_size=1, verbose=True
    )
    
    # Extract chords with 2-beat windows
    chords_2beat = extract_chords_with_beat_windows(
        combined_notes, beats_shifted, window_size=2, verbose=True
    )
    
    # Calculate total bars based on number of beats
    total_bars = (len(beats_shifted) + 3) // 4  # Assuming 4/4 time
    
    # Visualize 1-beat window results
    print("\n1-Beat Window Chord Analysis:")
    plot_chord_timeline_multirow(chords_1beat, len(beats_shifted), beats_per_bar=None, beats_per_row=32)

    
    # Visualize 2-beat window results
    print("\n2-Beat Window Chord Analysis:")
    plot_chord_timeline_multirow(chords_2beat, len(beats_shifted), beats_per_bar=None, beats_per_row=32)

    
    # Visualize a section of audio with beats and 1-beat window chords
    start_beat = start_bar * 4  # Assuming 4/4 time
    num_beats = num_bars * 4
    
    print(f"\nVisualizing beats {start_beat}-{start_beat+num_beats} with 1-beat window chords:")
    audio_1beat = visualize_beats_and_chords(
        audio_path, beats_shifted, chords_1beat, start_beat, num_beats
    )
    
    print(f"\nVisualizing beats {start_beat}-{start_beat+num_beats} with 2-beat window chords:")
    audio_2beat = visualize_beats_and_chords(
        audio_path, beats_shifted, chords_2beat, start_beat, num_beats
    )
    
    # Print chord progression comparison for the visualized section
    print("\nChord Progression Comparison:")
    print(f"{'Bar':<5} {'Beat':<5} {'1-Beat Window':<15} {'2-Beat Window':<15}")
    print("-" * 45)
    
    for beat in range(start_beat, start_beat + num_beats):
        bar = beat // 4 + 1
        beat_in_bar = beat % 4 + 1
        
        # Find chords at this beat
        chord_1beat = next((c['chord_name'] for c in chords_1beat 
                           if c['beat_start'] <= beat and c['beat_end'] > beat), "-")
        
        chord_2beat = next((c['chord_name'] for c in chords_2beat 
                           if c['beat_start'] <= beat and c['beat_end'] > beat), "-")
        
        print(f"{bar:<5} {beat_in_bar:<5} {chord_1beat:<15} {chord_2beat:<15}")
    
    return audio_1beat, audio_2beat, chords_1beat, chords_2beat

# Update the extract_chords_with_beat_windows function to remove bar references

def extract_chords_with_beat_windows(notes, beats_shifted, window_size=1, merge=True, verbose=False, use_bars=False):
    """
    Extract chords using fixed windows based on precise beat timings
    
    Args:
        notes: Combined notes from bass and harmony
        beats_shifted: Array of beat times in seconds
        window_size: Number of beats per window (1 or 2)
        merge: Whether to merge adjacent identical chords
        verbose: Whether to output detailed information
        use_bars: Whether to include bar information in output (default: False)
        
    Returns:
        List of chord dictionaries
    """
    # Ensure we have notes and beats
    if not notes or len(beats_shifted) < 2:
        print("No notes or beats to analyze")
        return []
    
    # Get beat positions
    beat_positions = np.arange(len(beats_shifted))
    
    # Create mapping from beat positions to time
    beat_to_time = create_beat_time_mapping(beats_shifted)
    
    # Convert MIDI note timings to match extracted beats
    aligned_notes = convert_midi_timing_to_beat_positions(notes, beat_to_time, beat_positions)
    
    # Set up output
    all_chords = []
    total_beats = len(beats_shifted)
    
    # Calculate total number of windows
    num_windows = total_beats - (window_size - 1)
    
    # Process each window
    for window_idx in range(0, num_windows):
        # Define window boundaries in beat positions
        window_start_beat = window_idx
        window_end_beat = window_idx + window_size
        
        # Convert to time domain
        window_start_time = beats_shifted[window_start_beat]
        window_end_time = beats_shifted[window_end_beat] if window_end_beat < len(beats_shifted) else beats_shifted[-1] + (beats_shifted[-1] - beats_shifted[-2])
        
        # Find notes active in this window (using time domain)
        active_notes = [n for n in aligned_notes 
                       if n['end_time'] > window_start_time and n['start_time'] < window_end_time]
        
        # Skip empty windows
        if not active_notes:
            continue
        
        # Separate bass and harmony notes
        bass_notes = [n for n in active_notes if n['source'] == 'bass']
        harmony_notes = [n for n in active_notes if n['source'] == 'harmony']
        
        # Skip if no harmony notes (need harmony for a chord)
        if not harmony_notes:
            continue
        
        # Calculate pitch class durations in the window for ALL notes
        pc_durations = {}
        pc_to_name = {}
        pc_to_lowest_pitch = {}
        pc_to_source = {}  # Track whether a pitch class appears in bass, harmony, or both
        
        # Process ALL notes to get combined pitch class statistics
        for note in active_notes:
            # Calculate overlap with the window
            overlap_start = max(window_start_time, note['start_time'])
            overlap_end = min(window_end_time, note['end_time'])
            overlap_duration = overlap_end - overlap_start
            
            pc = note['pitch_class']
            source = note['source']
            
            if pc not in pc_durations:
                pc_durations[pc] = 0
                pc_to_name[pc] = note['name']
                pc_to_lowest_pitch[pc] = note['pitch']
                pc_to_source[pc] = set([source])
            else:
                if note['pitch'] < pc_to_lowest_pitch[pc]:
                    pc_to_lowest_pitch[pc] = note['pitch']
                pc_to_source[pc].add(source)
            
            pc_durations[pc] += overlap_duration
        
        # Determine root candidates
        root_candidates = []
        
        # First, check if any pitch classes appear in BOTH bass and harmony
        both_sources_pcs = [pc for pc, sources in pc_to_source.items() 
                         if 'bass' in sources and 'harmony' in sources]
        
        if both_sources_pcs:
            # Prioritize notes that appear in both bass and harmony
            for pc in both_sources_pcs:
                position_score = 120 - pc_to_lowest_pitch[pc]  # Lower pitches score higher
                duration_score = pc_durations[pc]
                bass_confirmation_bonus = 50  # High bonus for appearing in both bass and harmony
                
                total_score = position_score * 2 + duration_score * 2 + bass_confirmation_bonus
                
                root_candidates.append({
                    'pitch_class': pc,
                    'name': pc_to_name[pc],
                    'score': total_score
                })
        
        # If no pitch classes appear in both, check bass notes first
        if not root_candidates and bass_notes:
            # Get unique pitch classes from bass
            bass_pcs = set(bn['pitch_class'] for bn in bass_notes)
            
            for pc in bass_pcs:
                # Calculate bass duration
                bass_duration = sum(
                    min(bn['end_time'], window_end_time) - max(bn['start_time'], window_start_time)
                    for bn in bass_notes if bn['pitch_class'] == pc
                )
                
                position_score = 120 - pc_to_lowest_pitch[pc]
                duration_score = bass_duration
                bass_bonus = 30  # Bonus just for being in the bass
                
                total_score = position_score * 2 + duration_score * 3 + bass_bonus
                
                root_candidates.append({
                    'pitch_class': pc,
                    'name': pc_to_name[pc],
                    'score': total_score
                })
        
        # If still no candidates, use harmony-only approach
        if not root_candidates:
            # Sort harmony notes by pitch (lowest first)
            sorted_harmony = sorted(harmony_notes, key=lambda x: x['pitch'])
            
            # Take the lowest 3 notes as candidates
            for note in sorted_harmony[:3]:
                pc = note['pitch_class']
                lowest_pitch = pc_to_lowest_pitch[pc]
                
                # Calculate note duration within window
                overlap_start = max(window_start_time, note['start_time'])
                overlap_end = min(window_end_time, note['end_time'])
                overlap_duration = overlap_end - overlap_start
                
                position_score = 120 - lowest_pitch  # Lower pitches score higher
                duration_score = pc_durations[pc]
                
                # Prioritize strong beats
                beat_score = 0
                if window_start_beat % 4 == 0:  # Every 4th beat (potential downbeat)
                    beat_score = 10
                    
                total_score = position_score * 4 + duration_score * 2 + beat_score
                
                root_candidates.append({
                    'pitch_class': pc,
                    'name': pc_to_name[pc],
                    'score': total_score
                })
        
        # Select the highest-scoring root candidate
        if not root_candidates:
            continue
            
        root_candidates.sort(key=lambda x: x['score'], reverse=True)
        root_pc = root_candidates[0]['pitch_class']
        root_name = root_candidates[0]['name']
        
        # Calculate intervals for chord identification relative to selected root
        interval_durations = {}
        
        # Always include the root interval
        interval_durations[0] = window_end_time - window_start_time
        
        # Calculate intervals for harmony notes
        for hn in harmony_notes:
            # Calculate overlap with the window
            overlap_start = max(window_start_time, hn['start_time'])
            overlap_end = min(window_end_time, hn['end_time'])
            overlap_duration = overlap_end - overlap_start
            
            interval = (hn['pitch_class'] - root_pc) % 12
            if interval not in interval_durations:
                interval_durations[interval] = 0
            interval_durations[interval] += overlap_duration
        
        # Create window result for chord identification
        window_result = {
            'intervals': list(interval_durations.keys()),
            'interval_durations': interval_durations,
            'root_name': root_name
        }
        
        # Use identify_unique_chord function to get refined chord type
        refined_results = identify_unique_chord([window_result])
        refined_chord_type = refined_results[0]['refined_chord_type']
        confidence = refined_results[0].get('confidence', 0)
        
        chord_name = f"{root_name}{refined_chord_type}"
        
        # Calculate normalized weights for reference
        total_duration = sum(interval_durations.values())
        normalized_weights = {
            i: d/total_duration for i, d in interval_durations.items()
        }
        
        # Calculate significant intervals based on normalized weights
        significant_intervals = [i for i, w in normalized_weights.items() 
                               if w >= 0.1 or i == 0]
        
        # Determine source type based on root origin
        if 'bass' in pc_to_source.get(root_pc, []):
            source_type = 'bass-confirmed'
        else:
            source_type = 'harmony-derived'
        
        # Create the chord dictionary with basic info
        chord_dict = {
            'beat_start': window_start_beat,
            'beat_end': window_end_beat,
            'time_start': window_start_time,
            'time_end': window_end_time,
            'root': root_name,
            'chord_type': refined_chord_type,
            'chord_name': chord_name,
            'intervals': significant_intervals,
            'source': source_type,
            'confidence': confidence
        }
        
        # Add bar info if requested
        if use_bars:
            chord_dict['bar'] = window_start_beat // 4 + 1
        
        # Store chord
        all_chords.append(chord_dict)
    
    # Sort chords by start time
    all_chords.sort(key=lambda x: x['beat_start'])
    
    # Merge adjacent identical chords if requested
    if merge:
        all_chords = merge_adjacent_identical_chords(all_chords)
        if verbose:
            print(f"Merged adjacent identical chords: {len(all_chords)} chords after merging")
    
    # Print summary
    if verbose:
        print(f"Analyzed {total_beats} beats")
        if use_bars:
            total_bars = (total_beats + 3) // 4  # Assuming 4/4 time signature
            print(f"Approximate bars: {total_bars} (assuming 4/4 time)")
        print(f"Extracted {len(all_chords)} chords using {window_size}-beat windows")
        
        bass_confirmed = sum(1 for c in all_chords if c.get('source') == 'bass-confirmed')
        harmony_derived = sum(1 for c in all_chords if c.get('source') == 'harmony-derived')
        print(f"  - {bass_confirmed} bass-confirmed chords")
        print(f"  - {harmony_derived} harmony-derived chords")
    
    return all_chords

def analyze_song_chords(song_id, ms_shift=20):
    """
    Run the complete analysis process for a song
    
    Args:
        song_id: Song ID (used for paths)
        ms_shift: Millisecond shift to apply to beats
    """
    # Setup paths
    base_path = "../samples/suno_samples"
    audio_path = f"{base_path}/audio/{song_id}.mp3"
    midi_base_path = f"{base_path}/segmented/{song_id}/midi"
    bass_midi_path = f"{midi_base_path}/bass.mid"
    harmony_midi_path = f"{midi_base_path}/harmony.mid"
    
    # Check if files exist
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return
    
    if not os.path.exists(bass_midi_path) or not os.path.exists(harmony_midi_path):
        print(f"Error: MIDI files not found at {midi_base_path}")
        return
    
    # Extract beats with shift
    y, sr, beats_shifted, tempo = extract_beats_with_shift(audio_path, ms_shift)
    
    # Load and prepare MIDI notes
    bass_notes = load_midi_notes(bass_midi_path, source_name='bass')
    harmony_notes = load_midi_notes(harmony_midi_path, source_name='harmony')
    
    # Clean the bass line
    cleaned_bass = clean_bass_line(bass_notes)
    
    # Combine cleaned bass with harmony
    combined_notes = cleaned_bass + harmony_notes
    combined_notes.sort(key=lambda x: x['start'])
    
    print(f"Combined {len(cleaned_bass)} bass notes with {len(harmony_notes)} harmony notes")
    
    # Extract chords with 1-beat windows
    print("\nExtracting chords with 1-beat windows:")
    chords_1beat = extract_chords_with_beat_windows(
        combined_notes, beats_shifted, window_size=1, merge=False, verbose=True
    )
    
    # Extract chords with 2-beat windows
    print("\nExtracting chords with 2-beat windows:")
    chords_2beat = extract_chords_with_beat_windows(
        combined_notes, beats_shifted, window_size=2, merge=False, verbose=True
    )
    
    # Calculate total bars
    total_bars = (len(beats_shifted) + 3) // 4  # Assuming 4/4 time
    
    # Visualize results
    print("\n1-Beat Window Chord Analysis:")
    plot_chord_timeline_multirow(chords_1beat, total_bars, beats_per_bar=4, bars_per_row=8)
    
    print("\n2-Beat Window Chord Analysis:")
    plot_chord_timeline_multirow(chords_2beat, total_bars, beats_per_bar=4, bars_per_row=8)
    
    # Create audio with beat clicks for verification
    # Ensure click_track is the same length as y
    click_track = librosa.clicks(
        times=beats_shifted, 
        sr=sr, 
        click_freq=660, 
        click_duration=0.1,
        length=len(y)  # Add this parameter to ensure matching length
    )
    
    audio_with_clicks = y + click_track * 0.5
    
    print("\nAudio with beat clicks (to verify beat alignment):")
    
    return {
        'y': y,
        'sr': sr,
        'beats_shifted': beats_shifted,
        'tempo': tempo,
        'chords_1beat': chords_1beat,
        'chords_2beat': chords_2beat,
        'audio_with_clicks': audio_with_clicks
    }