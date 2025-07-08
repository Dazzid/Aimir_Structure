import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio, display, Markdown, clear_output
import music21
from music21 import converter, chord, note, stream, pitch
from matplotlib.widgets import Slider, Button
import ipywidgets as widgets
from ipywidgets import interact, fixed

def analyze_song_structure(audio_path, max_duration=60):
    """
    Perform detailed analysis of the song structure to determine beats and bars
    
    Args:
        audio_path: Path to the audio file
        max_duration: Maximum duration to analyze in seconds
        
    Returns:
        Dictionary containing analysis results
    """
    print(f"Loading audio from {audio_path}...")
    
    try:
        # Load audio file with original sample rate
        y, sr = librosa.load(audio_path, sr=None, duration=max_duration)
        
        # Get duration information
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"Audio duration: {duration:.2f} seconds")
        
        # Compute onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Dynamic tempo estimation (allows for tempo changes)
        dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        
        # Use more accurate beat tracking with adjustable parameters
        # Try multiple settings and take the most consistent
        beat_results = []
        
        # Standard beat tracking
        tempo, beats_standard = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beat_times_standard = librosa.frames_to_time(beats_standard, sr=sr)
        beat_results.append((tempo, beat_times_standard, "Standard"))
        
        # Beat tracking with higher/lower tempo bias
        for tempo_bias in [0.9, 1.0, 1.1]:
            try:
                tempo, beats_custom = librosa.beat.beat_track(
                    onset_envelope=onset_env, 
                    sr=sr,
                    start_bpm=tempo * tempo_bias,
                    tightness=100  # Make it stick closer to the provided tempo
                )
                beat_times_custom = librosa.frames_to_time(beats_custom, sr=sr)
                beat_results.append((tempo, beat_times_custom, f"Tempo bias {tempo_bias:.1f}"))
            except:
                pass
                
        # Find the most consistent beat tracking result
        # (The one with the most consistent inter-beat intervals)
        best_consistency = float('inf')
        best_result = None
        
        for result in beat_results:
            tempo, beat_times, method = result
            
            # Calculate inter-beat intervals
            if len(beat_times) > 1:
                ibis = np.diff(beat_times)
                consistency = np.std(ibis) / np.mean(ibis)  # Coefficient of variation
                
                if consistency < best_consistency:
                    best_consistency = consistency
                    best_result = result
        
        if best_result:
            tempo, beat_times, method = best_result
            print(f"Selected beat tracking method: {method}")
        else:
            tempo, beat_times = beat_results[0][0], beat_results[0][1]
            print("Using default beat tracking result")
            
        print(f"Estimated tempo: {tempo:.1f} BPM")
        print(f"Detected {len(beat_times)} beats")
        
        # Detect downbeats (first beat of each bar)
        # This is an estimate - we'll provide tools to adjust manually
        
        # First approach: use librosa's beat plp
        if len(beat_times) > 4:
            S = librosa.stft(y)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            _, beats_plp = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            # Estimate 4/4 time signature (most common)
            estimated_bars = []
            
            # Method 1: Try to find every 4th beat
            for i in range(0, len(beats_plp), 4):
                if i < len(beats_plp):
                    estimated_bars.append(beats_plp[i])
                    
            # Convert to times
            estimated_bar_times = librosa.frames_to_time(estimated_bars, sr=sr)
            
            # Method 2: Use spectral flux for stronger beats
            # (More reliable for finding true downbeats)
            S = np.abs(librosa.stft(y))
            flux = np.sum(np.diff(S, axis=1), axis=0)
            flux = np.maximum(flux, 0.0)  # Keep only increases in energy
            
            # Normalize
            flux = flux / np.max(flux)
            
            # Find which beats have the highest flux
            beat_flux = []
            for beat in beats_plp:
                if beat < len(flux):
                    beat_flux.append((beat, flux[beat]))
                    
            # Sort by flux value
            beat_flux.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 25% of beats by energy
            strongest_beats = [bf[0] for bf in beat_flux[:len(beat_flux)//4]]
            strongest_beats.sort()  # Sort back by time
            
            # Convert to times
            strongest_beat_times = librosa.frames_to_time(strongest_beats, sr=sr)
        else:
            estimated_bar_times = []
            strongest_beat_times = []
        
        # Estimate offset (if song doesn't start on a downbeat)
        # This is a rough estimate - will need manual adjustment
        first_beat_time = beat_times[0] if beat_times.size > 0 else 0
        
        # Calculate beat period (seconds per beat) from tempo
        beat_period = 60.0 / tempo
        
        # Rough estimate of offset (how far into a bar the song starts)
        # Assumption: Most songs start close to a bar boundary
        # This will be adjustable in the final interface
        offset_beats = (first_beat_time / beat_period) % 4
        
        # Return all analysis results
        return {
            'y': y,
            'sr': sr,
            'duration': duration,
            'tempo': tempo,
            'beat_times': beat_times,
            'beat_period': beat_period,
            'estimated_bar_times': estimated_bar_times,
            'strongest_beat_times': strongest_beat_times,
            'offset_beats': offset_beats,
            'onset_env': onset_env
        }
        
    except Exception as e:
        print(f"Error in audio analysis: {e}")
        return None

def visualize_with_adjustable_bars(analysis_results, midi_notes=None, bpm=None):
    """
    Interactive visualization to manually adjust bar boundaries
    
    Args:
        analysis_results: Results from analyze_song_structure
        midi_notes: Optional MIDI notes to display
        bpm: BPM to use (if None, uses the detected tempo)
    """
    if not analysis_results:
        print("No analysis results to visualize")
        return
    
    # Extract data from analysis
    y = analysis_results['y']
    sr = analysis_results['sr']
    beat_times = analysis_results['beat_times']
    detected_tempo = analysis_results['tempo']
    duration = analysis_results['duration']
    
    # Use provided BPM or detected tempo
    tempo = bpm if bpm is not None else detected_tempo
    
    # Create interactive widgets
    tempo_slider = widgets.FloatSlider(
        value=tempo,
        min=tempo*0.8,
        max=tempo*1.2,
        step=0.1,
        description='BPM:',
        continuous_update=False
    )
    
    # Offset slider (how far into a bar the first beat is)
    offset_slider = widgets.FloatSlider(
        value=0,
        min=0,
        max=3.99,
        step=0.01,
        description='Offset (beats):',
        continuous_update=False
    )
    
    # Time signature widget (for different bar lengths)
    time_sig_dropdown = widgets.Dropdown(
        options=[('4/4', 4), ('3/4', 3), ('6/8', 6)],
        value=4,
        description='Time Signature:',
    )
    
    # Window size to display
    window_slider = widgets.IntSlider(
        value=15,
        min=5,
        max=min(60, int(duration)),
        step=1,
        description='Window (s):',
    )
    
    # Starting point in the song
    start_time_slider = widgets.FloatSlider(
        value=0,
        min=0,
        max=max(0, duration - 5),
        step=0.1,
        description='Start Time (s):',
    )
    
    # Peak detection threshold for beat validation
    threshold_slider = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1,
        step=0.01,
        description='Peak Threshold:',
    )
    
    # Button to recompute beats with current settings
    recompute_button = widgets.Button(
        description='Recompute Beats',
        button_style='primary',
        tooltip='Recompute beats with current settings'
    )
    
    # Button to confirm and save the settings
    save_button = widgets.Button(
        description='Save Settings',
        button_style='success',
        tooltip='Save these settings as the best match'
    )
    
    # Output area for calculated values and messages
    output_area = widgets.Output()
    
    # Function to update the visualization
    def update_visualization(tempo, offset, beats_per_bar, window_size, start_time, threshold):
        clear_output(wait=True)
        
        # Calculate beat period from tempo
        beat_period = 60.0 / tempo
        
        # Calculate absolute beat positions (start from 0)
        absolute_beat_positions = np.arange(0, duration + beat_period, beat_period)
        
        # Apply the offset to get the actual beat positions
        beat_positions = absolute_beat_positions - (offset * beat_period)
        
        # Calculate bar positions (every beats_per_bar)
        # Need to ensure we don't start with a negative bar
        first_bar_start = beat_positions[0] - (beat_positions[0] % (beats_per_bar * beat_period))
        bar_positions = np.arange(first_bar_start, duration + beat_period, beats_per_bar * beat_period)
        
        # Create figure with shared x axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
        
        # Load audio segment for display
        segment_end = min(start_time + window_size, duration)
        segment_duration = segment_end - start_time
        
        # Extract audio segment for display
        samples_start = int(start_time * sr)
        samples_end = int(segment_end * sr)
        y_segment = y[samples_start:samples_end]
        
        # Get detected onset strength for this segment
        onset_env = analysis_results['onset_env']
        onset_times = librosa.times_like(onset_env, sr=sr)
        segment_onset_indices = np.where((onset_times >= start_time) & (onset_times < segment_end))[0]
        segment_onset_times = onset_times[segment_onset_indices] - start_time
        segment_onset_values = onset_env[segment_onset_indices]
        
        # Normalize onset strength
        if len(segment_onset_values) > 0:
            segment_onset_values = segment_onset_values / np.max(segment_onset_values)
        
        # Create time array for segment
        segment_times = np.linspace(0, segment_duration, len(y_segment))
        
        # Plot audio waveform
        ax1.plot(segment_times, y_segment, color='blue', linewidth=0.5)
        
        # Overlay onset strength
        if len(segment_onset_times) > 0:
            ax1.plot(segment_onset_times, segment_onset_values, color='green', alpha=0.7)
            ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
        
        # Find which beats and bars fall within our display window
        visible_beat_positions = [bp - start_time for bp in beat_positions 
                                if bp >= start_time and bp < segment_end]
        visible_bar_positions = [bp - start_time for bp in bar_positions 
                                if bp >= start_time and bp < segment_end]
        
        # Plot beat lines
        for beat_pos in visible_beat_positions:
            ax1.axvline(x=beat_pos, color='gray', linestyle=':', alpha=0.5)
            ax2.axvline(x=beat_pos, color='gray', linestyle=':', alpha=0.5)
        
        # Plot bar lines
        for i, bar_pos in enumerate(visible_bar_positions):
            # Calculate the bar number based on tempo, offset, and beats_per_bar
            bar_number = int((bar_pos + start_time) / (beat_period * beats_per_bar)) + 1
            
            # Draw bar lines
            ax1.axvline(x=bar_pos, color='black', linestyle='-', alpha=0.7)
            ax2.axvline(x=bar_pos, color='black', linestyle='-', alpha=0.7)
            
            # Label bar number
            ax2.text(bar_pos + 0.1, 0.95, f"Bar {bar_number}", 
                     transform=ax2.get_xaxis_transform(), fontsize=9)
        
        # If MIDI notes are provided, plot them in the lower subplot
        if midi_notes:
            # Handle both single-track and multi-track MIDI
            if isinstance(midi_notes, list) and isinstance(midi_notes[0], dict):
                # Single track
                plot_midi_notes(ax2, midi_notes, start_time, segment_end, tempo, offset)
            elif isinstance(midi_notes, dict):
                # Multiple tracks
                for track_name, track_notes in midi_notes.items():
                    color = 'red' if 'bass' in track_name.lower() else 'blue'
                    plot_midi_notes(ax2, track_notes, start_time, segment_end, tempo, offset, color=color)
        else:
            # If no MIDI, just set reasonable y limits
            ax2.set_ylim(0, 1)
            ax2.set_yticks([])
            ax2.text(0.5, 0.5, "No MIDI data provided", 
                     ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        
        # Set axes labels and limits
        ax1.set_title(f"Audio Waveform & Detected Beats (Tempo: {tempo:.1f} BPM, Offset: {offset:.2f} beats)")
        ax1.set_ylabel("Amplitude")
        ax2.set_xlabel("Time (seconds)")
        
        # Set x limits to display window
        ax1.set_xlim(0, segment_duration)
        
        # Layout and display
        plt.tight_layout()
        plt.show()
        
        # Display the audio segment
        display(Audio(y_segment, rate=sr))
        
        # Display the calculated values in the output area
        with output_area:
            clear_output()
            print(f"BPM: {tempo:.1f}")
            print(f"Offset: {offset:.2f} beats")
            print(f"Beats per bar: {beats_per_bar}")
            
            # Calculate actual first beat and first bar
            first_visible_beat = -1
            if visible_beat_positions:
                first_visible_beat = visible_beat_positions[0] + start_time
                
            first_visible_bar = -1
            if visible_bar_positions:
                first_visible_bar = visible_bar_positions[0] + start_time
                
            # Calculate the seconds per beat and beats per bar
            seconds_per_beat = 60.0 / tempo
            
            print(f"\nBeats:")
            for i, beat_pos in enumerate(visible_beat_positions[:10]):
                absolute_beat_pos = beat_pos + start_time
                beat_number = int(round(absolute_beat_pos / seconds_per_beat))
                print(f"  Beat {beat_number}: {absolute_beat_pos:.2f}s")
                
            print(f"\nBars:")
            for i, bar_pos in enumerate(visible_bar_positions[:5]):
                absolute_bar_pos = bar_pos + start_time
                bar_number = int(absolute_bar_pos / (seconds_per_beat * beats_per_bar)) + 1
                print(f"  Bar {bar_number}: {absolute_bar_pos:.2f}s")
    
    def plot_midi_notes(ax, notes, start_time, end_time, tempo, offset, color='blue'):
        # Convert BPM to seconds per beat
        seconds_per_beat = 60.0 / tempo
        
        # Apply offset to MIDI notes (shifting by the given offset in beats)
        adjusted_notes = []
        
        for note in notes:
            # Calculate start and end times in seconds
            if 'start' in note and 'end' in note:
                # Convert from beat positions to seconds
                note_start_sec = note['start'] * seconds_per_beat
                note_end_sec = note['end'] * seconds_per_beat
                
                # Adjust with offset (positive offset means song starts N beats into first bar)
                offset_sec = offset * seconds_per_beat
                
                # Skip notes outside our display window
                if note_end_sec < start_time or note_start_sec > end_time:
                    continue
                
                # Only keep notes that are visible in this window
                visible_start = max(0, note_start_sec - start_time)
                visible_end = min(end_time - start_time, note_end_sec - start_time)
                
                if visible_end <= visible_start:
                    continue
                
                # Get pitch
                if 'pitch' in note:
                    pitch_value = note['pitch']
                elif 'pitch_class' in note:
                    pitch_value = note['pitch_class']
                else:
                    continue
                
                # Store adjusted note
                adjusted_notes.append({
                    'start': visible_start,
                    'end': visible_end,
                    'pitch': pitch_value,
                    'name': note.get('name', 'Note')
                })
        
        # Plot MIDI notes
        if adjusted_notes:
            # Set y-axis to display MIDI note range
            if all('pitch' in note and isinstance(note['pitch'], int) for note in adjusted_notes):
                # For numeric MIDI pitch values
                min_pitch = max(0, min(note['pitch'] for note in adjusted_notes) - 5)
                max_pitch = min(127, max(note['pitch'] for note in adjusted_notes) + 5)
                ax.set_ylim(min_pitch, max_pitch)
                
                # Set y ticks to show note names
                yticks = range(min_pitch, max_pitch + 1, 12)
                ylabels = [pitch.Pitch(p).nameWithOctave for p in yticks] if yticks else []
                ax.set_yticks(yticks)
                ax.set_yticklabels(ylabels)
                
                # Plot each note
                for note in adjusted_notes:
                    # Draw note rectangle
                    rect = plt.Rectangle(
                        (note['start'], note['pitch'] - 0.4),
                        note['end'] - note['start'],
                        0.8,
                        color=color,
                        alpha=0.7
                    )
                    ax.add_patch(rect)
                    
                    # Add note name if there's enough space
                    if note['end'] - note['start'] > 0.2:
                        ax.text(
                            note['start'] + (note['end'] - note['start']) / 2,
                            note['pitch'] + 0.2,
                            note.get('name', ''),
                            ha='center',
                            va='center',
                            fontsize=8
                        )
            else:
                # For other display (e.g., pitch class)
                ax.set_ylim(0, 12)
                ax.set_yticks(range(12))
                ax.set_yticklabels(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
                
                # Plot each pitch class
                for note in adjusted_notes:
                    pitch_class = note['pitch'] % 12 if isinstance(note['pitch'], int) else note['pitch']
                    rect = plt.Rectangle(
                        (note['start'], pitch_class - 0.4),
                        note['end'] - note['start'],
                        0.8,
                        color=color,
                        alpha=0.7
                    )
                    ax.add_patch(rect)
        else:
            ax.set_ylim(0, 127)
            ax.text(0.5, 0.5, "No notes visible in this window", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            
        ax.set_ylabel("MIDI Pitch")
    
    # Function to handle recompute button click
    def on_recompute_button_clicked(b):
        with output_area:
            clear_output()
            print("Recomputing beats with current settings...")
            
        # Get current values
        current_tempo = tempo_slider.value
        current_offset = offset_slider.value
        current_time_sig = time_sig_dropdown.value
        
        # Update visualization with new values
        update_visualization(
            current_tempo, 
            current_offset, 
            current_time_sig, 
            window_slider.value, 
            start_time_slider.value,
            threshold_slider.value
        )
    
    # Function to handle save button click
    def on_save_button_clicked(b):
        # Get current values
        current_tempo = tempo_slider.value
        current_offset = offset_slider.value
        current_time_sig = time_sig_dropdown.value
        
        with output_area:
            clear_output()
            print("Saving settings:")
            print(f"- BPM: {current_tempo:.1f}")
            print(f"- Offset: {current_offset:.2f} beats")
            print(f"- Time Signature: {current_time_sig}/4")
            print("\nThese settings can be used for proper bar alignment in chord analysis.")
            
            # Calculate beat period
            beat_period = 60.0 / current_tempo
            
            # Calculate actual first downbeat time
            first_downbeat = current_offset * beat_period
            
            print(f"\nFirst downbeat occurs at: {first_downbeat:.2f} seconds")
            print(f"Seconds per beat: {beat_period:.4f}")
            print(f"Seconds per bar: {beat_period * current_time_sig:.4f}")
            
            # If there's a significant offset, provide adjustment code
            if current_offset > 0.1:
                print("\nAdjustment code to use in chord analysis:")
                print("```python")
                print(f"# Correct bar alignment settings")
                print(f"bpm = {current_tempo:.1f}")
                print(f"offset_beats = {current_offset:.2f}")
                print(f"beats_per_bar = {current_time_sig}")
                print(f"seconds_per_beat = 60.0 / bpm  # {beat_period:.4f} seconds")
                print("\n# Apply these settings in your chord analysis")
                print("# Example: Adjust note start/end times")
                print("for note in notes:")
                print("    # Convert beat positions to seconds with correct offset")
                print("    note['start_adjusted'] = note['start'] - (offset_beats * seconds_per_beat)")
                print("    note['end_adjusted'] = note['end'] - (offset_beats * seconds_per_beat)")
                print("```")
    
    # Connect buttons to functions
    recompute_button.on_click(on_recompute_button_clicked)
    save_button.on_click(on_save_button_clicked)
    
    # Create interactive function
    def update(tempo, offset, beats_per_bar, window_size, start_time, threshold):
        update_visualization(tempo, offset, beats_per_bar, window_size, start_time, threshold)
    
    # Create widget layout
    top_controls = widgets.HBox([tempo_slider, offset_slider, time_sig_dropdown])
    mid_controls = widgets.HBox([window_slider, start_time_slider, threshold_slider])
    bottom_controls = widgets.HBox([recompute_button, save_button])
    
    # Display widgets and initial visualization
    display(top_controls)
    display(mid_controls)
    display(bottom_controls)
    display(output_area)
    
    # Initial update
    update_visualization(
        tempo_slider.value, 
        offset_slider.value, 
        time_sig_dropdown.value, 
        window_slider.value, 
        start_time_slider.value,
        threshold_slider.value
    )

def convert_midi_to_beats(midi_path, track_name="MIDI"):
    """
    Convert a MIDI file to a list of notes with beat positions
    
    Args:
        midi_path: Path to the MIDI file
        track_name: Name for the track (for multi-track display)
        
    Returns:
        List of notes with beat positions
    """
    print(f"Loading MIDI: {midi_path}")
    
    try:
        # Load the MIDI file
        midi = converter.parse(midi_path)
        
        all_notes = []
        for part in midi.parts:
            for note_obj in part.flatten().notesAndRests:
                if isinstance(note_obj, note.Note):
                    all_notes.append({
                        'start': float(note_obj.offset),
                        'end': float(note_obj.offset + note_obj.duration.quarterLength),
                        'pitch': note_obj.pitch.midi,
                        'name': note_obj.pitch.nameWithOctave,
                        'track': track_name
                    })
                elif isinstance(note_obj, chord.Chord):
                    for p in note_obj.pitches:
                        all_notes.append({
                            'start': float(note_obj.offset),
                            'end': float(note_obj.offset + note_obj.duration.quarterLength),
                            'pitch': p.midi,
                            'name': p.nameWithOctave,
                            'track': track_name
                        })
        
        # Sort by start time
        all_notes.sort(key=lambda x: x['start'])
        print(f"Loaded {len(all_notes)} notes from {midi_path}")
        
        return all_notes
    
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        return []

def align_beat_structure(audio_path, midi_paths=None):
    """
    Main function to align beat structure between audio and MIDI
    
    Args:
        audio_path: Path to the audio file
        midi_paths: Dictionary of MIDI paths (e.g., {'bass': 'bass.mid', 'harmony': 'harmony.mid'})
    """
    # First analyze audio
    analysis = analyze_song_structure(audio_path)
    
    if not analysis:
        print("Audio analysis failed. Cannot continue.")
        return
    
    # Load MIDI files if provided
    midi_notes = {}
    if midi_paths:
        for track_name, path in midi_paths.items():
            midi_notes[track_name] = convert_midi_to_beats(path, track_name)
    
    # Display interactive alignment tool
    visualize_with_adjustable_bars(analysis, midi_notes)

def apply_bar_alignment_to_notes(notes, bpm, offset_beats, beats_per_bar=4):
    """
    Apply bar alignment settings to notes
    
    Args:
        notes: List of note dictionaries with 'start' and 'end' beat positions
        bpm: Beats per minute
        offset_beats: Offset in beats (how far into a bar the song starts)
        beats_per_bar: Number of beats per bar
        
    Returns:
        Copy of notes with adjusted start/end times and bar information
    """
    # Convert BPM to seconds per beat
    seconds_per_beat = 60.0 / bpm
    
    # Create a copy of the notes with adjusted information
    adjusted_notes = []
    
    for note in notes:
        # Copy the original note
        adjusted_note = note.copy()
        
        # Add original beat positions
        adjusted_note['start_beat_original'] = note['start']
        adjusted_note['end_beat_original'] = note['end']
        
        # Apply offset to get aligned beat positions
        # If offset is positive, it means the song starts offset beats into a bar
        # So we need to subtract the offset from the MIDI note positions
        adjusted_note['start_beat_aligned'] = note['start'] - offset_beats
        adjusted_note['end_beat_aligned'] = note['end'] - offset_beats
        
        # Convert to bar and beat positions
        # For start position
        start_bar = int(adjusted_note['start_beat_aligned'] / beats_per_bar) + 1
        start_beat_in_bar = (adjusted_note['start_beat_aligned'] % beats_per_bar) + 1
        
        # For end position
        end_bar = int(adjusted_note['end_beat_aligned'] / beats_per_bar) + 1
        end_beat_in_bar = (adjusted_note['end_beat_aligned'] % beats_per_bar) + 1
        
        # Add bar and beat information
        adjusted_note['start_bar'] = start_bar
        adjusted_note['start_beat_in_bar'] = start_beat_in_bar
        adjusted_note['end_bar'] = end_bar
        adjusted_note['end_beat_in_bar'] = end_beat_in_bar
        
        # Add to results
        adjusted_notes.append(adjusted_note)
    
    return adjusted_notes

def test_alignment_with_chords(audio_path, midi_paths, bpm, offset_beats, beats_per_bar=4):
    """
    Test if the alignment works by extracting chords
    
    Args:
        audio_path: Path to audio file
        midi_paths: Dictionary of MIDI paths
        bpm: Corrected BPM
        offset_beats: Offset in beats
        beats_per_bar: Number of beats per bar
    """
    from IPython.display import display, Markdown
    
    display(Markdown("## Testing Alignment with Chord Extraction"))
    
    # Load MIDI files
    midi_notes = {}
    for track_name, path in midi_paths.items():
        midi_notes[track_name] = convert_midi_to_beats(path, track_name)
    
    # Apply alignment
    aligned_notes = {}
    for track_name, notes in midi_notes.items():
        aligned_notes[track_name] = apply_bar_alignment_to_notes(notes, bpm, offset_beats, beats_per_bar)
    
    # Visualize a few bars with proper alignment
    # Load audio analysis
    analysis = analyze_song_structure(audio_path)
    
    if not analysis:
        print("Audio analysis failed. Cannot continue.")
        return
    
    # Update analysis with corrected values
    analysis['tempo'] = bpm
    
    # Visualize with corrected alignment
    print(f"Visualizing with corrected alignment: BPM={bpm}, Offset={offset_beats} beats")
    visualize_with_adjustable_bars(analysis, aligned_notes, bpm)
    
    return aligned_notes

# Example usage (uncomment to test directly)
# audio_path = 'path/to/your/audio.mp3'
# midi_paths = {
#     'bass': 'path/to/your/bass.mid',
#     'harmony': 'path/to/your/harmony.mid'
# }
# align_beat_structure(audio_path, midi_paths)