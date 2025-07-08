#Analyse chords in a MIDI file
import numpy as np
import matplotlib.pyplot as plt
import music21
from music21 import converter, chord, note, stream, pitch
from collections import Counter
import librosa 
import os
import re

#---------------------------------------------------------------------------
CHORD_COLORS = {
    # Basic triads
    "": "#ffb703",       # Major (blue)
    "m": "#8ecae6",      # Minor (mint green)
    "dim": "#219ebc",    # Diminished (pink)
    "aug": "#fc7600",    # Augmented (gold)
    
    # Seventh chords
    "7": "#ffffff",      # Dominant 7th (red)
    "7alt": "#ffffff",   # Altered 7th (white)
    "7b13": "#ffffff",   # Altered 7th (white)
    "maj7": "#fcba03",   # Major 7th (purple)
    "m7": "#03b5fc",     # Minor 7th (cyan)
    "dim7": "#ff33eb",   # Diminished 7th (magenta)
    "m7b5": "#8a82ff",   # Half-diminished (orange)
    
    # Extended chords
    "9": "#ffffff",      # Dominant 9th (violet)
    "maj9": "#fcba03",   # Major 9th (lime green)
    "m9": "#03b5fc",     # Minor 9th (royal blue)
    
    # Suspended & other
    "sus4": "#fb8500",   # Suspended (light gray)
    "5": "#fb8500",      # Power chord (gray)
    
    # Default
    "default": "#d4d4d4" # Other chord types (light gray)
}

#---------------------------------------------------------------------------
# Cell 2: Define chord structures with their characteristic intervals
CHORD_TYPES = {
    # Basic triads
    "": {  # Major triad
        "intervals": [0, 4, 7],
        "required": [0, 4]
    },
    "m": {  # Minor triad
        "intervals": [0, 3, 7],
        "required": [0, 3]
    },
    "dim": {  # Diminished triad
        "intervals": [0, 3, 6],
        "required": [0, 3, 6]
    },
    "aug": {  # Augmented triad
        "intervals": [0, 4, 8],
        "required": [0, 4, 8]
    },
    
    # Seventh chords
    "7": {  # Dominant seventh
        "intervals": [0, 4, 7, 10],
        "required": [0, 4, 10]
    },
    "maj7": {  # Major seventh
        "intervals": [0, 4, 7, 11],
        "required": [0, 4, 11]
    },
    "m7": {  # Minor seventh
        "intervals": [0, 3, 7, 10],
        "required": [0, 3, 10]
    },
    "dim7": {  # Diminished seventh
        "intervals": [0, 3, 6, 9],
        "required": [0, 3, 9]
    },
    "m7b5": {  # Half-diminished seventh
        "intervals": [0, 3, 6, 10],
        "required": [0, 3, 6, 10]
    },
    
    # Extended chords
    "9": {  # Dominant ninth
        "intervals": [0, 4, 7, 10, 2],
        "required": [0, 4, 10, 2]
    },
    "maj9": {  # Major ninth
        "intervals": [0, 4, 7, 11, 2],
        "required": [0, 4, 11, 2]
    },
    "m9": {  # Minor ninth
        "intervals": [0, 3, 7, 10, 2],
        "required": [0, 3, 10, 2]
    },
    
    # Altered chords
    "7b9": {  # Dominant seventh flat ninth
        "intervals": [0, 4, 7, 10, 1],
        "required": [0, 4, 10, 1]
    },
    "7#9": {  # Dominant seventh sharp ninth
        "intervals": [0, 4, 7, 10, 3],
        "required": [0, 4, 10, 3]
    },
    "7#11": {  # Dominant seventh sharp eleventh
        "intervals": [0, 4, 7, 10, 6],
        "required": [0, 4, 10, 6]
    },
    
    # Suspended chords
    "sus4": {  # Suspended fourth
        "intervals": [0, 5, 7],
        "required": [0, 5]
    },
    "7sus4": {  # Dominant seventh suspended fourth
        "intervals": [0, 5, 7, 10],
        "required": [0, 5, 10]
    },
    "7alt": {  # Dominant seventh altered (with b5 and b9)
        "intervals": [0, 4, 6, 10, 1],
        "required": [0, 4, 10]
    },
    "7b13": {  # Dominant seventh flat thirteenth (same as #5)
        "intervals": [0, 4, 7, 8, 10],
        "required": [0, 4, 10, 8]
    }
}

#get the bars of the song
#---------------------------------------------------------------------------
def getBeatsFromSong(path):
    """
    Extract beat timestamps from an audio file.
    Returns an array of times (in seconds) corresponding to detected beats.
    """
    y, sr = librosa.load(path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times

#---------------------------------------------------------------------------
def get_bpm_from_audio(path):
    y, sr = librosa.load(path)
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    return bpm

#---------------------------------------------------------------------------
# Cell 3: Define function to load MIDI notes
def load_midi_notes(midi_path, source_name="unknown"):
    """Load notes from a MIDI file with source labeling"""
    print(f"Loading MIDI file: {midi_path}")
    
    try:
        # Load MIDI file
        midi = converter.parse(midi_path)
        
        # Extract notes with pitch information
        all_notes = []
        for part in midi.parts:
            for note_obj in part.flatten().notesAndRests:
                if isinstance(note_obj, note.Note):
                    all_notes.append({
                        'start': float(note_obj.offset),
                        'end': float(note_obj.offset + note_obj.duration.quarterLength),
                        'pitch': note_obj.pitch.midi,
                        'name': fix_note_name(note_obj.pitch.name),
                        'pitch_class': note_obj.pitch.pitchClass,
                        'velocity': note_obj.volume.velocity if hasattr(note_obj.volume, 'velocity') else 64,
                        'source': source_name
                    })
                elif isinstance(note_obj, chord.Chord):
                    for p in note_obj.pitches:
                        all_notes.append({
                            'start': float(note_obj.offset),
                            'end': float(note_obj.offset + note_obj.duration.quarterLength),
                            'pitch': p.midi,
                            'name': fix_note_name(p.name),
                            'pitch_class': p.pitchClass,
                            'velocity': note_obj.volume.velocity if hasattr(note_obj.volume, 'velocity') else 64,
                            'source': source_name
                        })
        
        # Sort notes by start time
        all_notes.sort(key=lambda x: x['start'])
        print(f"Extracted {len(all_notes)} notes from {midi_path}")
        
        return all_notes
    
    except Exception as e:
        print(f"Error loading MIDI file {midi_path}: {e}")
        return []
    
#---------------------------------------------------------------------------
# Cell 4: Function to visualize notes in a bar with enhanced display
def visualize_notes_in_bars(notes, num_bars=1, beats_per_bar=4, start_bar=0):
    """
    Visualize notes in the specified bars with enhanced display
    """
    # Calculate start and end beats
    start_beat = start_bar * beats_per_bar
    end_beat = (start_bar + num_bars) * beats_per_bar
    
    # Filter notes within the specified bars
    bar_notes = [n for n in notes if n['start'] < end_beat and n['end'] > start_beat]
    
    if not bar_notes:
        print(f"No notes found in bars {start_bar+1} to {start_bar+num_bars}")
        return
    
    # Set up the plot
    plt.figure(figsize=(15, 8))
    
    # Plot notes as rectangles
    for note_data in bar_notes:
        # Trim notes to the visible range
        start = max(note_data['start'], start_beat)
        end = min(note_data['end'], end_beat)
        
        # Determine color based on source
        color = 'blue' if note_data['source'] == 'harmony' else 'red'
        alpha = min(1.0, note_data['velocity'] / 127 + 0.3) # Adjust transparency by velocity
        
        # Plot the note
        plt.barh(
            note_data['pitch'], 
            end - start, 
            left=start, 
            height=0.5, 
            color=color, 
            alpha=alpha
        )

        # Add note name               
        plt.text(
            start, 
            note_data['pitch']+0.5, 
            note_data['name'], 
            fontsize=8,
            color='black',
            bbox=dict(facecolor='white', alpha=0.5, pad=0, edgecolor="white")
        )
    
    # Add bar lines
    for bar in range(start_bar, start_bar + num_bars + 1):
        bar_pos = bar * beats_per_bar
        plt.axvline(x=bar_pos, color='black', linestyle='--', alpha=0.5)
        if bar < start_bar + num_bars:  # Add bar numbers
            plt.text(bar_pos + 0.1, plt.ylim()[1] - 3, f"Bar {bar+1}", fontsize=10)
    
    # Add beat markers
    for beat in range(int(start_beat), int(end_beat) + 1):
        plt.axvline(x=beat, color='gray', linestyle=':', alpha=0.3)
    
    # Set labels and title
    plt.xlabel('Beat Position')
    plt.ylabel('MIDI Pitch')
    plt.title(f'Notes in Bars {start_bar+1} to {start_bar+num_bars}')
    
    # Set y-axis to show note names
    yticks = range(0, 128, 12)
    ylabels = [pitch.Pitch(p).nameWithOctave for p in yticks]
    plt.yticks(yticks, ylabels)
    
    # Set x-axis ticks to show beats
    plt.xticks(range(int(start_beat), int(end_beat) + 1))
    
    # Add legend
    plt.scatter([], [], color='red', label='Bass')
    plt.scatter([], [], color='blue', label='Harmony')
    plt.legend()
    
    # Show grid and plot
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
#---------------------------------------------------------------------------    
# Add or check that this function is defined before it's used in analyze_time_windows
def get_interval_name(interval):
    """Get a readable name for an interval number"""
    interval_names = {
        0: "Root",
        1: "b9",
        2: "9",
        3: "b3",
        4: "3",
        5: "11",
        6: "b5",
        7: "5",
        8: "#5",
        9: "6",
        10: "b7",
        11: "7"
    }
    return interval_names.get(interval, f"Int{interval}")

#---------------------------------------------------------------------------
# Cell 5: Improved chord identification from intervals
def identify_chord_from_intervals(intervals, interval_weights=None):
    """
    Identify chord type from intervals, with improved dominant chord detection
    
    Args:
        intervals: List of intervals relative to root
        interval_weights: Optional dictionary of interval weights for better decision making
    
    Returns:
        String describing the chord type
    """
    # First handle the case where we have weights to make better decisions
    if interval_weights is not None:
        # If both b7 and maj7 are present, use the one with higher weight
        if 10 in intervals and 11 in intervals:
            flat_7_weight = interval_weights.get(10, 0)
            maj_7_weight = interval_weights.get(11, 0)
            
            # If flat 7 is significant enough, remove maj7 from consideration
            if flat_7_weight > maj_7_weight * 1.2:  # 20% stronger presence
                intervals = [i for i in intervals if i != 11]
                print(f"Prioritizing flat 7 (weight {flat_7_weight:.2f}) over maj7 (weight {maj_7_weight:.2f})")
            # If maj7 is significantly stronger, remove flat 7
            elif maj_7_weight > flat_7_weight * 1.2:
                intervals = [i for i in intervals if i != 10]
                print(f"Prioritizing maj7 (weight {maj_7_weight:.2f}) over flat 7 (weight {flat_7_weight:.2f})")
            # If they're close in weight, slightly prefer dominant (b7)
            else:
                intervals = [i for i in intervals if i != 11]
                print(f"Both 7ths similar weight, preferring dominant: b7={flat_7_weight:.2f}, maj7={maj_7_weight:.2f}")
    
    # First check for diminished chords
    if 0 in intervals and 3 in intervals and 6 in intervals:
        if 9 in intervals:
            return "dim7"
        if 10 in intervals:
            return "m7b5"
        return "dim"
    
    # Special case for altered dominant chords: Root + b7 + b5
    if 0 in intervals and 10 in intervals and 6 in intervals and 4 not in intervals and 3 not in intervals:
        return "7b5"  # Dominant 7 flat 5
    
    # Special case for dominant with no 3rd but has b7
    if 0 in intervals and 10 in intervals and 7 in intervals and 4 not in intervals and 3 not in intervals:
        return "7no3"  # Dominant 7 no 3rd
    
    # Check exact match for specific chord types
    for chord_type, data in CHORD_TYPES.items():
        required_intervals = data["required"]
        
        # All required intervals must be present
        if all(req in intervals for req in required_intervals):
            # For cases where multiple chord types might match, prioritize:
            # 1. Extended chords (9, 11, 13)
            # 2. Seventh chords
            # 3. Triads
            if chord_type in ["9", "maj9", "m9", "7b9", "7#9", "7#11"]:
                return chord_type
    
    # Second pass for seventh chords
    for chord_type in ["7", "maj7", "m7", "7sus4"]:
        data = CHORD_TYPES[chord_type]
        if all(req in intervals for req in data["required"]):
            return chord_type
    
    # Third pass for triads
    for chord_type in ["", "m", "aug", "sus4"]:
        data = CHORD_TYPES[chord_type]
        if all(req in intervals for req in data["required"]):
            return chord_type
    
    # If we have a bare minimum, try to find the closest match
    if 0 in intervals:
        # Check for key intervals
        has_minor_third = 3 in intervals
        has_major_third = 4 in intervals
        has_perfect_fifth = 7 in intervals
        has_diminished_fifth = 6 in intervals
        has_augmented_fifth = 8 in intervals
        has_minor_seventh = 10 in intervals
        has_major_seventh = 11 in intervals
        has_ninth = 2 in intervals
        has_flat_ninth = 1 in intervals
        
        # Identify based on key intervals
        if has_major_third:
            if has_augmented_fifth:
                return "aug"
            if has_minor_seventh and has_flat_ninth:
                return "7b9"
            if has_minor_seventh and has_ninth:
                return "9"
            if has_minor_seventh:
                return "7"
            if has_major_seventh and has_ninth:
                return "maj9"
            if has_major_seventh:
                return "maj7"
            return ""  # Major triad
        
        elif has_minor_third:
            if has_diminished_fifth and has_minor_seventh:
                return "m7b5"
            if has_diminished_fifth:
                return "dim"
            if has_minor_seventh and has_ninth:
                return "m9"
            if has_minor_seventh:
                return "m7"
            return "m"  # Minor triad
        
        # No third but has dominant 7th
        elif has_minor_seventh:
            if has_diminished_fifth:
                return "7b5"  # Dom7 with flat 5, no 3rd
            if has_perfect_fifth:
                return "7no3"  # Dom7 with no 3rd
            return "7"  # Best guess for a dominant with missing 3rd and 5th
            
        elif 5 in intervals:  # No third but has fourth
            if has_minor_seventh:
                return "7sus4"
            return "sus4"
    
    # If still no match, return power chord (root + fifth) or unrecognized
    if 7 in intervals:
        return "5"  # Power chord
    
    return ""  # Unrecognized or incomplete

#---------------------------------------------------------------------------

def fix_note_name(note_name):
    """
    Fix note names by replacing music21's terrible minus sign flats with proper 'b' flats
    
    Args:
        note_name: Note name from music21 (like 'B-' or 'E-')
    
    Returns:
        Fixed note name (like 'Bb' or 'Eb')
    """
    # Replace the minus sign used for flats with 'b'
    return note_name.replace('-', 'b')

#---------------------------------------------------------------------------
# Cell 6: Enhanced visualization of time windows with RAW WEIGHTS
# Replace your existing analyze_time_windows function with this modified version
def analyze_time_windows(notes, start_bar, beats_per_bar=4, min_chord_duration=0.5):
    """
    Analyze exactly which notes are active during each bass note's duration,
    with improved handling of short bass notes and chord identification
    """
    # Calculate bar boundaries
    start_beat = start_bar * beats_per_bar
    end_beat = (start_bar + 1) * beats_per_bar
    
    # Get notes in this bar
    bar_notes = [n for n in notes if n['end'] > start_beat and n['start'] < end_beat]
    
    # Separate bass and harmony
    bass_notes = [n for n in bar_notes if n['source'] == 'bass']
    harmony_notes = [n for n in bar_notes if n['source'] == 'harmony']
    
    # Sort bass notes by start time
    bass_notes.sort(key=lambda x: x['start'])
    
    # Filter out very short bass notes (passing notes)
    significant_bass_notes = []
    for bn in bass_notes:
        duration = min(bn['end'], end_beat) - max(bn['start'], start_beat)
        if duration >= min_chord_duration:
            significant_bass_notes.append(bn)
    
    # If no significant bass notes found, use the longest one we have
    if not significant_bass_notes and bass_notes:
        longest_bass = max(bass_notes, 
                          key=lambda x: min(x['end'], end_beat) - max(x['start'], start_beat))
        significant_bass_notes = [longest_bass]
    
    # Debug print active bass notes
    print(f"\nBass notes in bar {start_bar+1}:")
    print(f"  Total bass notes: {len(bass_notes)}")
    print(f"  Significant bass notes (duration >= {min_chord_duration}): {len(significant_bass_notes)}")
    for i, bn in enumerate(significant_bass_notes):
        duration = min(bn['end'], end_beat) - max(bn['start'], start_beat)
        print(f"  Bass {i+1}: {bn['name']} (PC {bn['pitch_class']}): {bn['start']:.2f}-{bn['end']:.2f}, duration: {duration:.2f}")
    
    # 1. FIRST VISUALIZATION: Show all notes in the bar with chord windows highlighted
    plt.figure(figsize=(15, 8))
    
    # Plot all notes as rectangles
    for note_data in bar_notes:
        # Trim notes to the visible range
        start = max(note_data['start'], start_beat)
        end = min(note_data['end'], end_beat)
        
        # Determine color based on source
        color = 'blue' if note_data['source'] == 'harmony' else 'red'
        alpha = min(1.0, note_data['velocity'] / 127 + 0.3) # Adjust transparency by velocity
        
        # Plot the note
        plt.barh(
            note_data['pitch'], 
            end - start, 
            left=start, 
            height=0.8, 
            color=color, 
            alpha=alpha
        )
        
        # Add note name for all notes - improved visibility
        plt.text(
            start + 0.05, 
            note_data['pitch'] + 0.3, 
            note_data['name'], 
            fontsize=8,
            color='black',
            bbox=dict(facecolor='white', alpha=0.7, pad=1)
        )
    
    # Highlight the bass note time windows
    for i, bn in enumerate(significant_bass_notes):
        window_start = max(bn['start'], start_beat)
        window_end = min(bn['end'], end_beat)
        
        # Draw a semi-transparent rectangle covering the window
        plt.axvspan(window_start, window_end, alpha=0.2, color='yellow')
        
        # Add chord name at the top
        plt.text(
            window_start + (window_end - window_start)/2, 
            plt.ylim()[1] - 2,
            f"Chord {i+1}: {bn['name']} root",
            ha='center',
            fontsize=9,
            bbox=dict(facecolor='yellow', alpha=0.5, pad=2)
        )
    
    # Add bar lines
    bar_pos = start_bar * beats_per_bar
    plt.axvline(x=bar_pos, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=end_beat, color='black', linestyle='--', alpha=0.5)
    plt.text(bar_pos + 0.1, plt.ylim()[1] - 3, f"Bar {start_bar+1}", fontsize=10)
    
    # Add beat markers
    for beat in range(int(start_beat), int(end_beat) + 1):
        plt.axvline(x=beat, color='gray', linestyle=':', alpha=0.3)
    
    # Set labels and title
    plt.xlabel('Beat Position')
    plt.ylabel('MIDI Pitch')
    plt.title(f'Notes in Bar {start_bar+1} with Chord Windows Highlighted')
    
    # Set y-axis to show note names
    yticks = range(0, 128, 12)
    ylabels = [pitch.Pitch(p).nameWithOctave for p in yticks]
    plt.yticks(yticks, ylabels)
    
    # Set x-axis ticks to show beats
    plt.xticks(range(int(start_beat), int(end_beat) + 1))
    
    # Add legend
    plt.scatter([], [], color='red', label='Bass')
    plt.scatter([], [], color='blue', label='Harmony')
    plt.scatter([], [], color='yellow', alpha=0.5, s=100, label='Chord Window')
    plt.legend(loc='upper right')
    
    # Show grid and plot
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Create a list to store results
    chord_windows = []
    
    # 2. ANALYZE EACH SIGNIFICANT BASS NOTE TIME WINDOW
    for i, bass_note in enumerate(significant_bass_notes):
        # Define precise time window
        window_start = max(bass_note['start'], start_beat)
        window_end = min(bass_note['end'], end_beat)
        root_pc = bass_note['pitch_class']
        
        # Find exactly which harmony notes are active during this window
        active_harmony = []
        for hn in harmony_notes:
            # Note must overlap with the current bass note's time window
            if hn['start'] < window_end and hn['end'] > window_start:
                overlap_start = max(window_start, hn['start'])
                overlap_end = min(window_end, hn['end'])
                overlap_duration = overlap_end - overlap_start
                
                active_harmony.append({
                    'name': hn['name'],
                    'pitch': hn['pitch'],
                    'pitch_class': hn['pitch_class'],
                    'duration': overlap_duration,
                    'interval': (hn['pitch_class'] - root_pc) % 12
                })
        
        # Calculate RAW duration for each interval
        interval_durations = {}
        interval_notes = {}
        for interval in range(12):
            # Root (interval 0) gets duration of bass note
            if interval == 0:
                interval_durations[interval] = window_end - window_start
                interval_notes[interval] = [bass_note['name']]
                continue
                
            # Other intervals get sum of all harmony note durations with that interval
            matching_notes = [n for n in active_harmony if n['interval'] == interval]
            if matching_notes:
                interval_durations[interval] = sum(n['duration'] for n in matching_notes)
                interval_notes[interval] = sorted(set(n['name'] for n in matching_notes))
        
        # Calculate normalized weights for intervals
        total_duration = window_end - window_start  # Use window duration as denominator
        interval_weights = {i: d for i, d in interval_durations.items()}
        
        # Determine threshold for including intervals - intervals present for at least 25% of window
        threshold = 0.25
        
        # Identify significant intervals
        significant_intervals = sorted([
            i for i, w in interval_weights.items() if w >= threshold or i == 0
        ])
        
        # Special consideration for thirds, fifths, and sevenths (common chord tones)
        for special_interval in [3, 4, 7, 10, 11]:  # Minor/major 3rd, 5th, minor/major 7th
            if special_interval in interval_durations and special_interval not in significant_intervals:
                # Include them if they're at least 15% present
                if interval_weights[special_interval] >= 0.15:
                    significant_intervals.append(special_interval)
                    significant_intervals.sort()
                    
        # Identify chord based on significant intervals
        chord_type = identify_chord_from_intervals(significant_intervals)
        chord_name = f"{fix_note_name(bass_note['name'])}{chord_type}"
        
        # Store the result
        chord_windows.append({
            'start': window_start,
            'end': window_end,
            'duration': window_end - window_start,
            'root_pc': root_pc,
            'root_name': bass_note['name'],
            'intervals': significant_intervals,
            'interval_durations': interval_durations,
            'interval_weights': interval_weights,
            'chord_type': chord_type,
            'chord_name': chord_name,
            'active_notes': [hn['name'] for hn in active_harmony]
        })
        
        # 3. VISUALIZATION: Show notes specifically in this chord window
        plt.figure(figsize=(15, 8))
        
        # Only show notes active during this window
        window_notes = [bass_note] + [n for n in harmony_notes 
                                    if n['start'] < window_end and n['end'] > window_start]
        window_notes.sort(key=lambda x: x['pitch'])
        
        for note_data in window_notes:
            # Trim notes to the visible range
            start = max(note_data['start'], window_start)
            end = min(note_data['end'], window_end)
            
            # Determine color based on source
            color = 'blue' if note_data['source'] == 'harmony' else 'red'
            
            # Plot the note
            plt.barh(
                note_data['pitch'], 
                end - start, 
                left=start, 
                height=0.8, 
                color=color
            )
            
            # Add note name and interval for all notes
            interval = (note_data['pitch_class'] - root_pc) % 12
            interval_name = get_interval_name(interval)
            
            # More visible labels with white background
            plt.text(
                start + 0.05, 
                note_data['pitch'] + 0.3, 
                f"{note_data['name']} ({interval_name})", 
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, pad=1)
            )
        
        # Add window boundaries
        plt.axvline(x=window_start, color='black', linestyle='-', alpha=0.7)
        plt.axvline(x=window_end, color='black', linestyle='-', alpha=0.7)
        
        # Add beat markers
        for beat in range(int(window_start), int(window_end) + 1):
            plt.axvline(x=beat, color='gray', linestyle=':', alpha=0.3)
        
        # Set labels and title
        plt.xlabel('Beat Position')
        plt.ylabel('MIDI Pitch')
        plt.title(f'Chord Window for {chord_name} (Beats {window_start:.2f}-{window_end:.2f})')
        
        # Set y-axis to show note names
        yticks = [n['pitch'] for n in window_notes]
        ylabels = [f"{n['name']} ({get_interval_name((n['pitch_class']-root_pc)%12)})" for n in window_notes]
        plt.yticks(yticks, ylabels)
        
        # Add legend
        plt.scatter([], [], color='red', label='Bass (Root)')
        plt.scatter([], [], color='blue', label='Harmony')
        plt.legend(loc='upper right')
        
        # Show grid and plot
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print the result for this time window
        print(f"\nBass note window {i+1}: {bass_note['name']} ({window_start:.2f}-{window_end:.2f})")
        print(f"  Active harmony notes: {[n['name'] for n in active_harmony]}")
        print(f"  Significant intervals: {significant_intervals}")
        print(f"  Identified chord: {chord_name}")
        
        # 4. VISUALIZE INTERVAL WEIGHTS
        plt.figure(figsize=(12, 6))
        
        # Show weights for all 12 intervals
        interval_names = ["Root", "b9", "9", "b3", "3", "11", "b5", "5", "#5", "6", "b7", "maj7"]
        
        # Get weights for all intervals (0 if not present)
        weights = [interval_weights.get(i, 0) for i in range(12)]
        
        # Create colors (significant intervals are colored)
        colors = ['lightgray'] * 12
        for i in significant_intervals:
            colors[i] = 'red' if i == 0 else 'blue'
        
        # Create the chart with weights
        bars = plt.bar(interval_names, weights, color=colors)
        
        # Add a threshold line
        plt.axhline(y=threshold, color='green', linestyle='--', alpha=0.7)
        plt.text(11.5, threshold + 0.02, f"Threshold ({threshold:.2f})", color='green', ha='right')
        
        # Add secondary threshold for special intervals
        plt.axhline(y=0.15, color='orange', linestyle=':', alpha=0.7)
        plt.text(11.5, 0.15 + 0.02, "Special intervals threshold (0.15)", color='orange', ha='right')
        
        plt.ylabel('Relative Weight')
        plt.title(f"Interval Analysis for {chord_name}")
        
        # Annotate ALL intervals with notes (including small values)
        for i in range(12):
            if i in interval_notes:
                note_str = ", ".join(interval_notes[i])
                plt.text(
                    i, 
                    weights[i] + 0.01, 
                    note_str, 
                    ha='center', 
                    va='bottom',
                    fontsize=8,
                    rotation=0
                )
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed weights breakdown
        print("\nINTERVAL WEIGHTS:")
        print(f"{'Interval':<8} {'Name':<6} {'Weight':<10} {'Duration':<10} {'Notes'}")
        print("-" * 70)
        
        # Sort by weight (largest first) to see most important intervals
        sorted_intervals = sorted(
            [(i, interval_weights.get(i, 0)) for i in range(12)],
            key=lambda x: x[1],
            reverse=True
        )
        
        for interval, weight in sorted_intervals:
            if weight > 0:
                duration = interval_durations.get(interval, 0)
                notes_str = ", ".join(interval_notes.get(interval, []))
                print(f"{interval:<8} {interval_names[interval]:<6} {weight:<10.3f} {duration:<10.3f} {notes_str}")
    
    # Return the list of chord windows
    return chord_windows

#---------------------------------------------------------------------------
# Cell 7: Function to analyze a progression with time windows
def analyze_progression_with_time_windows(notes, start_bar, num_bars=4, beats_per_bar=4):
    """
    Analyze a progression of bars using time-sensitive chord recognition
    """
    # Visualize the entire section first
    visualize_notes_in_bars(notes, num_bars=num_bars, start_bar=start_bar, beats_per_bar=beats_per_bar)
    
    # Analyze each bar individually
    all_results = []
    for bar_idx in range(start_bar, start_bar + num_bars):
        print(f"\n=== Analyzing Bar {bar_idx+1} with Time Windows ===")
        bar_results = analyze_time_windows(notes, bar_idx, beats_per_bar=beats_per_bar)
        all_results.append(bar_results)
    
    # Summarize the progression
    print("\n" + "=" * 60)
    print("CHORD PROGRESSION SUMMARY (TIME-WINDOWED):")
    print("=" * 60)
    print(f"{'Bar':<6} {'Chords'}")
    print("-" * 60)
    
    for i, bar_results in enumerate(all_results):
        if bar_results:
            bar_num = start_bar + i + 1
            chord_names = [window['chord_name'] for window in bar_results]
            print(f"{bar_num:<6} {' → '.join(chord_names)}")
    
    print("=" * 60)
    
    return all_results

#---------------------------------------------------------------------------
# Example usage in the notebook
def apply_unique_chord_identification(original_results):
    """
    Apply unique chord identification to the original time window results
    """
    refined_results = identify_unique_chord(original_results)
    
    # Print detailed results
    print("\n=== Refined Chord Identification ===")
    for result in refined_results:
        print(f"\nRoot: {result['root_name']}")
        print(f"Original Chord: {result['chord_name']}")
        print(f"Refined Chord: {result['root_name']}{result['refined_chord_type']}")
        
        # Print normalized weights for transparency
        print("Normalized Interval Weights:")
        for interval, weight in sorted(result['normalized_weights'].items(), key=lambda x: x[1], reverse=True):
            if weight > 0.05:  # Only show significant weights
                print(f"  {get_interval_name(interval)}: {weight:.3f}")
    
    return refined_results

#---------------------------------------------------------------------------
# Modify the main analysis to use this approach
def analyze_progression_with_unique_chords(notes, start_bar, num_bars=4, beats_per_bar=4):
    """
    Analyze progression with unique chord identification
    """
    # First do the original time window analysis
    original_results = analyze_progression_with_time_windows(notes, start_bar, num_bars, beats_per_bar)
    
    # Then apply unique chord identification
    refined_results = [
        apply_unique_chord_identification(bar_results) 
        for bar_results in original_results
    ]
    
    return refined_results


#---------------------------------------------------------------------------------------
def clean_chord_quality(quality):
    mapping = {
        '5': 'power',
        '59': 'power9',
        '7': '7',
        '7#11': '7#11',
        '7#9': '7alt',
        '7alt': '7alt',
        '7b13': '7b13',
        '7b9': '7b9',
        '7sus': '7sus',
        '7sus4': '7sus4',
        '9': '9',
        'aug': 'aug',
        'b5': 'm7b5',
        'b59': 'm7b5',
        'b5b9': 'dim7',
        'b9': '7b9',
        'dim': 'dim',
        'dim7': 'dim7',
        'm': 'm',
        'm7': 'm7',
        'm79': 'm7',
        'm7b5': 'm7b5',
        'm9': 'm7',
        'maj7': 'maj7',
        'maj7b9': 'maj7',
        'maj9': 'maj7',
        'mb9': '7b9',
        'mmaj7': 'mmaj7',
        'mmaj7b9': 'mmaj7',
        'sus4': 'sus4',
        '': '',
        '5b9': '7b9',    # still ambiguous
    }
    return mapping.get(quality, quality)

#---------------------------------------------------------------------------------------

def classify_chord_by_weights(normalized_weights, min_threshold=0.01, root_pc=None):
    """
    Hybrid chord classification: exact structural match (via CHORD_TYPES) + fallback heuristics.
    """
    # Create interval set with more lenient threshold for diminished chord detection
    interval_set = {i for i, w in normalized_weights.items() if w > 0.05}
    
    # Special handling for diminished chords - use lower threshold
    dim_interval_set = {i for i, w in normalized_weights.items() if w > 0.03}
    
    # 1. Exact or template-based match from CHORD_TYPES
    best_label = None
    best_score = 0
    
    for label, spec in CHORD_TYPES.items():
        required = set(spec["required"])
        
        # Use different thresholds for different chord types
        if label in ["dim", "dim7", "m7b5"]:
            # For diminished chords, use more lenient checking
            # Check if at least 80% of required intervals are present
            present_required = sum(1 for i in required if i in dim_interval_set)
            required_ratio = present_required / len(required)
            
            if required_ratio >= 0.8:  # At least 80% of required intervals
                # Calculate score based on actual weights
                score = sum(normalized_weights.get(i, 0) for i in spec["intervals"])
                if score > best_score:
                    best_score = score
                    best_label = label
        else:
            # For other chords, use original strict checking
            if not required.issubset(interval_set):
                continue
            score = sum(normalized_weights.get(i, 0) for i in spec["intervals"])
            if score > best_score:
                best_score = score
                best_label = label
    
    if best_label:
        return best_label
    
    # 2. Manual diminished chord detection if not caught above
    has_minor_third = normalized_weights.get(3, 0) >= 0.05
    has_dim_fifth = normalized_weights.get(6, 0) >= 0.03  # Lower threshold
    has_dim_seventh = normalized_weights.get(9, 0) >= 0.05
    has_minor_seventh = normalized_weights.get(10, 0) >= 0.05
    
    if has_minor_third and has_dim_fifth:
        if has_dim_seventh:
            return "dim7"
        elif has_minor_seventh:
            return "m7b5"
        else:
            return "dim"
    
    # 3. Heuristic fallback: altered dominants, sus chords, power chords, etc.
    third_intervals = {3: 'm', 4: ''}  # minor or major third
    fifth_intervals = {6: 'b5', 7: '', 8: 'b13'}
    seventh_intervals = {10: '7', 11: 'maj7'}
    ninth_intervals = {1: 'b9', 2: '9'}
    
    max_third = max([(normalized_weights.get(i, 0), i) for i in third_intervals], 
                   key=lambda x: x[0], default=(0, None))[1]
    max_fifth = max([(normalized_weights.get(i, 0), i) for i in fifth_intervals], 
                   key=lambda x: x[0], default=(0, None))[1]
    max_seventh = max([(normalized_weights.get(i, 0), i) for i in seventh_intervals], 
                     key=lambda x: x[0], default=(0, None))[1]
    max_ninth = max([(normalized_weights.get(i, 0), i) for i in ninth_intervals], 
                   key=lambda x: x[0], default=(0, None))[1]
    
    is_dominant = max_seventh == 10 and normalized_weights.get(10, 0) >= min_threshold
    is_maj7 = max_seventh == 11 and normalized_weights.get(11, 0) >= min_threshold
    has_aug_fifth = max_fifth == 8 and normalized_weights.get(8, 0) >= min_threshold
    has_b9 = max_ninth == 1 and normalized_weights.get(1, 0) >= min_threshold
    has_b5 = max_fifth == 6 and normalized_weights.get(6, 0) >= min_threshold
    has_fourth = normalized_weights.get(5, 0) >= min_threshold
    no_third = (normalized_weights.get(3, 0) < 0.01 and normalized_weights.get(4, 0) < 0.01)
    is_sus = has_fourth and no_third
    
    # Altered dominant detection
    if is_dominant and has_aug_fifth and has_b9:
        return "7alt"
    elif is_dominant and has_aug_fifth:
        return "7b13"
    elif is_dominant and has_b9:
        return "7b9"
    elif is_dominant and has_b5:
        return "7#11"
    
    # Suspended chords
    if is_sus:
        return "7sus4" if is_dominant else "sus4"
    
    # Construct basic chord label
    chord_type = ""
    if not is_sus and max_third and normalized_weights.get(max_third, 0) > min_threshold:
        chord_type = third_intervals.get(max_third, '')
    
    if max_seventh and normalized_weights.get(max_seventh, 0) > min_threshold:
        chord_type += seventh_intervals.get(max_seventh, '')
    
    # Handle flat 5 in non-dominant contexts
    if max_fifth == 6 and normalized_weights.get(6, 0) > min_threshold and not is_dominant:
        if chord_type.startswith("m") and "7" in chord_type:
            return "m7b5"
        elif not chord_type.startswith("m"):
            chord_type += "b5"
    
    # Add ninth extensions
    if max_ninth and normalized_weights.get(max_ninth, 0) > min_threshold:
        chord_type += ninth_intervals.get(max_ninth, '')
    
    # Power chord fallback
    if chord_type == "" and max_fifth == 7 and normalized_weights.get(7, 0) > min_threshold:
        return "5"
    
    chord_type = clean_chord_quality(chord_type)
    return chord_type


#---------------------------------------------------------------------------
def identify_unique_chord(time_window_results):
    """
    Refine chord identification using weighted interval analysis with better bass handling.
    
    Args:
        time_window_results (list): Results from time window analysis.
        
    Returns:
        list: Refined chord identification results with a 'confidence' value.
    """
    refined_results = []
    
    for window in time_window_results:
        # Extract key information - now including absolute pitch information
        intervals = window.get('intervals', [])
        interval_durations = window.get('interval_durations', {}).copy()
        root_pitch = window.get('root_pitch', None)  # Get absolute root pitch if available
        absolute_pitches = window.get('absolute_pitches', [])  # Get absolute pitches if available
        
        # Ensure the root is always present with a nonzero duration
        if 0 not in interval_durations or interval_durations[0] == 0:
            if 'start' in window and 'end' in window:
                interval_durations[0] = window['end'] - window['start']
            else:
                interval_durations[0] = 1e-6  # Minimal fallback value
        
        # Calculate normalized weights for intervals
        total_duration = sum(interval_durations.values())
        
        if total_duration == 0:
            normalized_weights = {i: 0 for i in interval_durations}
        else:
            normalized_weights = {interval: duration / total_duration
                                for interval, duration in interval_durations.items()}
        
        # If we have absolute pitch information, use it to enhance chord detection
        enhanced_weights = normalized_weights.copy()
        
        if root_pitch is not None and absolute_pitches:
            # Calculate weights based on note distribution in actual chord voicing
            # This can help prioritize structural notes when they appear multiple times
            pitch_counts = {}
            for pitch in absolute_pitches:
                pc = pitch % 12
                if pc not in pitch_counts:
                    pitch_counts[pc] = 0
                pitch_counts[pc] += 1
            
            # Enhance weights of intervals that appear multiple times in the voicing
            root_pc = root_pitch % 12
            for pc, count in pitch_counts.items():
                interval = (pc - root_pc) % 12
                if interval in enhanced_weights and count > 1:
                    # Increase weight for intervals that appear multiple times
                    enhanced_weights[interval] *= (1 + 0.2 * (count - 1))
            
            # Renormalize weights
            total_enhanced = sum(enhanced_weights.values())
            if total_enhanced > 0:
                enhanced_weights = {k: v / total_enhanced for k, v in enhanced_weights.items()}
        
        # Get the unique chord type using the enhanced weights
        chord_type = classify_chord_by_weights(enhanced_weights)
        
        # Validate the chord type against music theory
        chord_type = validate_chord_type(chord_type)
        
        # Determine required intervals
        if chord_type in CHORD_TYPES:
            required_intervals = CHORD_TYPES[chord_type]["required"]
        else:
            required_intervals = [0]
        
        # Compute confidence as the sum of normalized weights for the required intervals
        confidence = sum(normalized_weights.get(i, 0) for i in required_intervals)
        
        # Update the window result with refined chord type and confidence
        updated_window = window.copy()
        updated_window['refined_chord_type'] = chord_type
        updated_window['normalized_weights'] = normalized_weights
        updated_window['enhanced_weights'] = enhanced_weights  # Store enhanced weights
        updated_window['confidence'] = confidence
        
        refined_results.append(updated_window)
    
    return refined_results


#---------------------------------------------------------------------------
# Cell 5: Function to validate and fix chord types
def validate_chord_type(chord_type):
    """
    Validate chord type against music theory to prevent invalid combinations
    and fix ordering to follow standard conventions
    
    Args:
        chord_type: Chord type string from classification
        
    Returns:
        Validated chord type string
    """
    # List of valid chord types
    valid_types = [
        "", "m", "dim", "aug",  # Triads
        "7", "maj7", "m7", "dim7", "m7b5", "7#11", "7alt", "7b13", # Sevenths
        "9", "maj9", "m9", "7b9", "7#9", # Extended
        "sus4", "7sus4", "7sus", "sus", "5" # Suspended and power
    ]
    
    # Direct match to valid types
    if chord_type in valid_types:
        return chord_type
        
    # Fix D79 to 7sus
    if chord_type == "79" or chord_type == "7+9" or chord_type == "9sus":
        return "7sus"
    
    # Fix ordering of altered fifths in dominant chords
    if "#11" in chord_type and "7" in chord_type:
        if chord_type.find("#11") < chord_type.find("7"):
            parts = []
            if chord_type.startswith("m") and chord_type != "maj7":
                parts.append("m")
            parts.append("7")
            parts.append("#11")
            return "".join(parts)
    
    # Fix 7b13 ordering
    if "b13" in chord_type and "7" in chord_type:
        if chord_type.find("b13") < chord_type.find("7"):
            parts = []
            if chord_type.startswith("m") and chord_type != "maj7":
                parts.append("m")
            parts.append("7")
            parts.append("b13")
            return "".join(parts)
    
    # Cannot have both minor and augmented
    if "m" in chord_type and "aug" in chord_type:
        # Choose based on which appears first (priority)
        if chord_type.find("m") < chord_type.find("aug"):
            return chord_type.replace("aug", "")
        else:
            return chord_type.replace("m", "")
    
    # Cannot have both diminished and augmented
    if "dim" in chord_type and "aug" in chord_type:
        return chord_type.replace("aug", "")
    
    # Fix ordering issues (e.g., "maj7m" should be "mmaj7")
    if "maj7" in chord_type and "m" in chord_type and chord_type.find("maj7") < chord_type.find("m"):
        return "m" + chord_type.replace("m", "")
    
    # Some specific replacements for common invalid combinations
    replacements = {
        "maj79": "maj7",  # Major 7 with 9 is just major 7
        "mdim": "dim",     # Minor diminished is just diminished
        "maug": "aug",     # Minor augmented doesn't exist
        "dimm": "dim",     # Diminished minor is just diminished
        "augm": "aug",     # Augmented minor doesn't exist
        "7#5": "7b13",     # Dominant with #5 should be 7b13 in jazz notation
        "79sus": "7sus",   # Dom7 with 9 and sus is just 7sus
        "9 ": "7sus",      # Dom9 without 3rd is 7sus
        "7#5b9": "7alt",   # Dominant with #5 and b9 is 7alt
        "7b9#5": "7alt",   # Dominant with b9 and #5 is 7alt
        "7#9b13": "7alt",  # Multiple alterations = 7alt
        "7b9b13": "7alt"   # Multiple alterations = 7alt
    }
    
    # Apply replacements
    for bad, good in replacements.items():
        if bad in chord_type:
            chord_type = chord_type.replace(bad, good)
    
    return chord_type

#---------------------------------------------------------------------------------------
def extract_harmony_only_chords(harmony_notes, start_beat, end_beat, beats_per_bar, bar_idx):
    """
    Extract chords from harmony notes when bass notes are not available
    
    Args:
        harmony_notes: List of harmony notes in the current bar
        start_beat: Start beat of the bar
        end_beat: End beat of the bar
        beats_per_bar: Number of beats per bar
        bar_idx: Current bar index
        
    Returns:
        List of chord dictionaries
    """
    if not harmony_notes:
        return []
    
    # Divide the bar into equal windows (half-bar divisions by default)
    window_size = beats_per_bar / 2
    harmony_windows = []
    
    # Create windows
    for window_start in np.arange(start_beat, end_beat, window_size):
        window_end = min(window_start + window_size, end_beat)
        
        # Find notes active in this window
        active_notes = []
        for hn in harmony_notes:
            if hn['start'] < window_end and hn['end'] > window_start:
                overlap_start = max(window_start, hn['start'])
                overlap_end = min(window_end, hn['end'])
                overlap_duration = overlap_end - overlap_start
                
                active_notes.append({
                    'name': hn['name'],
                    'pitch': hn['pitch'],  # Include MIDI pitch for finding lowest notes
                    'pitch_class': hn['pitch_class'], 
                    'duration': overlap_duration,
                    'start': hn['start'],  # Keep original start time for beat position analysis
                    'end': hn['end']       # Keep original end time
                })
        
        if active_notes:
            harmony_windows.append({
                'start': window_start,
                'end': window_end,
                'notes': active_notes
            })
    
    # Extract chords from each window
    window_chords = []
    
    for window in harmony_windows:
        if not window['notes']:
            continue
            
        # Find the lowest notes (potentially bass function)
        # Sort by pitch (ascending)
        sorted_by_pitch = sorted(window['notes'], key=lambda x: x['pitch'])
        
        # Take the lowest 2-3 notes that have significant duration
        lowest_notes = []
        for note in sorted_by_pitch:
            if note['duration'] >= 0.25 * window_size:  # At least 25% of window duration
                lowest_notes.append(note)
                if len(lowest_notes) >= 3:  # Take up to 3 lowest notes
                    break
        
        # If we don't have significant lowest notes, take the 2 absolute lowest
        if not lowest_notes:
            lowest_notes = sorted_by_pitch[:2] if len(sorted_by_pitch) >= 2 else sorted_by_pitch
        
        # Calculate note durations by pitch class for all notes
        pc_durations = {}
        pc_to_name = {}  # Map pitch class to note name
        pc_to_lowest_pitch = {}  # Map pitch class to its lowest occurrence
        
        for note in window['notes']:
            pc = note['pitch_class']
            if pc not in pc_durations:
                pc_durations[pc] = 0
                pc_to_name[pc] = note['name']
                pc_to_lowest_pitch[pc] = note['pitch']
            else:
                # Keep track of the lowest occurrence of each pitch class
                if note['pitch'] < pc_to_lowest_pitch[pc]:
                    pc_to_lowest_pitch[pc] = note['pitch']
            
            pc_durations[pc] += note['duration']
        
        # Determine likely root using a weighted approach:
        # 1. Heavily favor lowest notes (potential bass function)
        # 2. Factor in duration/prominence
        
        # First, create a list of candidate roots
        root_candidates = []
        
        # The pitch classes of the lowest notes become candidates
        for note in lowest_notes:
            pc = note['pitch_class']
            # Calculate a root score: combine bass position and duration
            # Lower pitch = higher score, longer duration = higher score
            lowest_pitch_for_pc = pc_to_lowest_pitch[pc]
            position_score = 120 - lowest_pitch_for_pc  # Lower pitches get higher scores (increased weight)
            duration_score = pc_durations[pc]
            
            # Prioritize strong beats (first and third beat in 4/4 time)
            beat_score = 0
            if 'start' in note:
                beat_in_bar = note['start'] % beats_per_bar
                if beat_in_bar < 0.5 or (beat_in_bar >= 2 and beat_in_bar < 2.5):
                    beat_score = 10  # Bonus for strong beats
            
            # Total score with position (bass function) weighted heaviest
            total_score = position_score * 4 + duration_score * 2 + beat_score  # Increased bass weighting
            
            root_candidates.append({
                'pitch_class': pc,
                'name': note['name'],
                'score': total_score
            })
        
        if not root_candidates:
            continue
            
        # Sort candidates by score
        root_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Select the highest-scoring candidate
        root_pc = root_candidates[0]['pitch_class']
        root_name = root_candidates[0]['name']
        
        # Calculate intervals relative to the root
        intervals = {}
        for pc, duration in pc_durations.items():
            interval = (pc - root_pc) % 12
            intervals[interval] = duration
        
        # Always include the root interval
        if 0 not in intervals:
            intervals[0] = window['end'] - window['start']
        
        # Create a window result in format needed for identify_unique_chord
        interval_durations = {i: intervals.get(i, 0) for i in range(12)}
        
        window_result = {
            'intervals': list(intervals.keys()),
            'interval_durations': interval_durations,
            'root_name': root_name
        }
        
        # Use the chord identification function to get the chord type
        refined_results = identify_unique_chord([window_result])
        refined_chord_type = refined_results[0]['refined_chord_type']
        
        chord_name = f"{fix_note_name(root_name)}{refined_chord_type}"
        
        # Store the chord
        window_chords.append({
            'bar': bar_idx + 1,
            'beat_start': window['start'],
            'beat_end': window['end'],
            'root': root_name,
            'chord_type': refined_chord_type,
            'chord_name': chord_name,
            'intervals': list(intervals.keys()),
            'source': 'harmony-only'  # Mark as derived from harmony without bass
        })
    
    return window_chords

#---------------------------------------------------------------------------------------
def extract_all_chords_automatically(notes, beats_per_bar=4, min_chord_duration=0.5, bars_per_row=16, verbose=False, plot=True, analyze_harmony_only=True):
    """
    Extract ALL chords from the entire piece automatically with harmony-only analysis
    
    Args:
        notes: Combined notes from bass and harmony
        beats_per_bar: Beats per bar
        min_chord_duration: Minimum duration for chord identification
        bars_per_row: Number of bars to display in each row of the visualization
        verbose: If True, return full chord data; if False, return summary string
        plot: If True, plot the multi-row timeline; if False, skip plotting
        analyze_harmony_only: If True, analyze sections without bass notes using harmony information
    """
    # Temporarily disable all print statements
    import sys
    original_stdout = sys.stdout
    sys.stdout = open('/dev/null', 'w')
    
    # Temporarily disable all plotting
    original_plt_show = plt.show
    plt.show = lambda: None
    
    # Close all existing figures to prevent warnings
    plt.close('all')
    
    try:
        # Determine the total length of the song automatically
        if not notes:
            return [] if verbose else "No notes found"
            
        # Find the last note's end time
        last_note_end = max(note['end'] for note in notes)
        
        # Calculate total number of bars (rounding up to include partial bars)
        import math
        total_bars = math.ceil(last_note_end / beats_per_bar)
        
        # Extract all chords from all bars
        all_chords = []
        
        for bar_idx in range(total_bars):
            # Calculate bar boundaries
            start_beat = bar_idx * beats_per_bar
            end_beat = (bar_idx + 1) * beats_per_bar
            
            # Get notes in this bar
            bar_notes = [n for n in notes if n['end'] > start_beat and n['start'] < end_beat]
            
            # Skip empty bars
            if not bar_notes:
                continue
                
            # Separate bass and harmony
            bass_notes = [n for n in bar_notes if n['source'] == 'bass']
            harmony_notes = [n for n in bar_notes if n['source'] == 'harmony']
            
            # Check if we have bass notes
            if bass_notes:
                # Process with normal bass-based chord extraction
                # Sort bass notes by start time
                bass_notes.sort(key=lambda x: x['start'])
                
                # Filter out very short bass notes
                significant_bass_notes = []
                for bn in bass_notes:
                    duration = min(bn['end'], end_beat) - max(bn['start'], start_beat)
                    if duration >= min_chord_duration:
                        significant_bass_notes.append(bn)
                
                # If no significant bass notes, use the longest one
                if not significant_bass_notes and bass_notes:
                    longest_bass = max(bass_notes, 
                                      key=lambda x: min(x['end'], end_beat) - max(x['start'], start_beat))
                    significant_bass_notes = [longest_bass]
                
                # Analyze each significant bass note
                for bass_note in significant_bass_notes:
                    # Define precise time window
                    window_start = max(bass_note['start'], start_beat)
                    window_end = min(bass_note['end'], end_beat)
                    root_pc = bass_note['pitch_class']
                    
                    # Find harmony notes active during this window
                    active_harmony = []
                    for hn in harmony_notes:
                        if hn['start'] < window_end and hn['end'] > window_start:
                            overlap_start = max(window_start, hn['start'])
                            overlap_end = min(window_end, hn['end'])
                            overlap_duration = overlap_end - overlap_start
                            
                            active_harmony.append({
                                'name': hn['name'],
                                'pitch_class': hn['pitch_class'],
                                'duration': overlap_duration,
                                'interval': (hn['pitch_class'] - root_pc) % 12
                            })
                    
                    # Calculate duration for each interval
                    interval_durations = {}
                    interval_notes = {}
                    for interval in range(12):
                        # Root (interval 0) gets duration of bass note
                        if interval == 0:
                            interval_durations[interval] = window_end - window_start
                            interval_notes[interval] = [bass_note['name']]
                            continue
                        
                        # Other intervals get sum of all harmony note durations
                        matching_notes = [n for n in active_harmony if n['interval'] == interval]
                        if matching_notes:
                            interval_durations[interval] = sum(n['duration'] for n in matching_notes)
                            interval_notes[interval] = sorted(set(n['name'] for n in matching_notes))
                    
                    # Calculate normalized weights for intervals
                    total_duration = sum(interval_durations.values()) if interval_durations else 1
                    normalized_weights = {
                        interval: duration / total_duration 
                        for interval, duration in interval_durations.items()
                    }
                    
                    # Create window result in format needed for identify_unique_chord
                    window_result = {
                        'intervals': [i for i in range(12) if i in interval_durations],
                        'interval_durations': interval_durations,
                        'root_name': bass_note['name']
                    }
                    
                    # Use the identify_unique_chord function to get refined chord type
                    refined_results = identify_unique_chord([window_result])
                    refined_chord_type = refined_results[0]['refined_chord_type']
                    
                    chord_name = f"{fix_note_name(bass_note['name'])}{refined_chord_type}"
                    
                    # Calculate significant intervals for reference
                    significant_intervals = [i for i, w in normalized_weights.items() 
                                            if w >= 0.15 or i == 0]
                    
                    # Determine source type based on root origin
                    if 'bass' in pc_to_source.get(root_pc, []):
                        source_type = 'bass-confirmed'
                    else:
                        source_type = 'harmony-derived'
                        
                    # Store chord with refined type
                    all_chords.append({
                        'bar': bar_idx + 1,
                        'beat_start': window['start'],
                        'beat_end': window['end'],
                        'root': root_name,
                        'chord_type': refined_chord_type,
                        'chord_name': chord_name,
                        'intervals': significant_intervals,
                        'source': source_type,
                        'confidence': confidence,
                        'timestamp': window_start * (60 / bpm) 
                    })
            
            # Handle bars without bass but with harmony notes
            elif analyze_harmony_only and harmony_notes:
                harmony_chords = extract_harmony_only_chords(
                    harmony_notes, start_beat, end_beat, beats_per_bar, bar_idx
                )
                all_chords.extend(harmony_chords)
                
    finally:
        # Restore stdout and plt.show
        sys.stdout = original_stdout
        plt.show = original_plt_show
    
    # Sort chords by start time
    all_chords.sort(key=lambda x: x['beat_start'])
    
    # Plot the multi-row timeline only if requested
    if plot:
        plot_chord_timeline_multirow(all_chords, total_bars, beats_per_bar, bars_per_row)
    
    # Return only a summary string if not verbose
    if not verbose:
        bass_chords = sum(1 for c in all_chords if c.get('source') == 'bass')
        harmony_chords = sum(1 for c in all_chords if c.get('source') == 'harmony-only')
        return f"Extracted {len(all_chords)} chords across {total_bars} bars ({bass_chords} from bass, {harmony_chords} from harmony only)"
    
    # Otherwise return the full chord data
    return all_chords

#---------------------------------------------------------------------------
def align_midi_to_audio(notes, ms_shift=20):
    """
    Shift MIDI note start/end times earlier by ms_shift milliseconds to align with audio beats.
    Args:
        notes: List of MIDI note dicts with 'start' and 'end' in seconds.
        ms_shift: Milliseconds to shift earlier (default 20ms, as in beat_tracking.ipynb)
    Returns:
        List of shifted notes (deep copy)
    """
    seconds_shift = ms_shift / 1000.0
    shifted = []
    for n in notes:
        n_copy = n.copy()
        n_copy['start'] = max(0, n_copy['start'] - seconds_shift)
        n_copy['end'] = max(0, n_copy['end'] - seconds_shift)
        shifted.append(n_copy)
    return shifted

#---------------------------------------------------------------------------
def plot_chord_timeline(chords, total_bars, beats_per_bar=4):
    """
    Plot ONE timeline with all extracted chords
    """
    # Close any existing plots
    plt.close('all')
    
    # Create the figure with appropriate width based on song length
    width = max(20, total_bars * 1.2)  # Minimum width of 20, scales with song length
    fig, ax = plt.subplots(figsize=(width, 6))
    
    # Draw the chords
    for chord in chords:
        start = chord['beat_start']
        end = chord['beat_end']
        name = chord['chord_name']
        chord_type = chord['chord_type']
        
        # Determine color
        if chord_type.startswith('m7'):
            color = '#8dd3c7'  # Minor seventh
        elif chord_type.startswith('m'):
            color = '#ffffb3'  # Minor
        elif chord_type == '7' or chord_type.endswith('7'):
            color = '#fdb462'  # Dominant seventh
        elif chord_type.startswith('maj'):
            color = '#b3de69'  # Major seventh or major ninth
        elif chord_type in ['dim', 'dim7']:
            color = '#fb8072'  # Diminished
        elif chord_type == '':
            color = '#80b1d3'  # Major
        else:
            color = '#d9d9d9'  # Other
        
        # Draw chord rectangle
        ax.add_patch(plt.Rectangle(
            (start, 0.1), 
            end - start, 
            0.8, 
            facecolor=color,
            edgecolor='black',
            alpha=0.7
        ))
        
        # Add chord name
        plt.text(
            start + (end - start)/2, 
            0.5, 
            name, 
            ha='center', 
            va='center',
            fontsize=10,
            fontweight='bold'
        )
    
    # Draw bar lines and numbers
    for bar in range(total_bars + 1):
        bar_pos = bar * beats_per_bar
        plt.axvline(x=bar_pos, color='black', linestyle='-', alpha=0.5)
        if bar < total_bars:
            plt.text(bar_pos + 0.1, 0.95, f"{bar+1}", fontsize=9)
    
    # Draw beat lines (only if the visualization isn't too crowded)
    if total_bars <= 32:  # Only show beat lines for shorter songs
        for beat in range(total_bars * beats_per_bar + 1):
            plt.axvline(x=beat, color='gray', linestyle=':', alpha=0.3)
    
    # Configure axes
    plt.xlim(0, total_bars * beats_per_bar)
    plt.ylim(0, 1)
    plt.title('Complete Chord Progression Timeline', fontsize=14)
    plt.xlabel('Beats', fontsize=12)
    
    # Remove y-axis ticks and labels
    plt.yticks([])
    
    # Add legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS[''], alpha=0.7, edgecolor='black', label='Major'),
        plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS['m'], alpha=0.7, edgecolor='black', label='Minor'),
        plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS['7'], alpha=0.7, edgecolor='black', label='Dominant 7th'),
        plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS['maj7'], alpha=0.7, edgecolor='black', label='Major 7th'),
        plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS['m7'], alpha=0.7, edgecolor='black', label='Minor 7th'),
        plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS['dim'], alpha=0.7, edgecolor='black', label='Diminished'),
        # Add more legend items as needed
    ]
    plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=7, frameon=False)
    
    plt.tight_layout()
    plt.show()
    

#---------------------------------------------------------------------------------------
# Create a custom text color function that selects an appropriate text color
# based on the background color brightness:
def get_text_color(background_color):
    """
    Determine whether to use black or white text based on background color brightness
    
    Args:
        background_color: Hex color string (e.g., '#f55442')
        
    Returns:
        String 'black' or 'white' for optimal contrast
    """
    # Convert hex color to RGB values
    r = int(background_color[1:3], 16) / 255.0
    g = int(background_color[3:5], 16) / 255.0
    b = int(background_color[5:7], 16) / 255.0
    
    # Calculate perceived brightness (ITU-R BT.709 standard)
    brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    # Use white text on dark backgrounds, black text on light backgrounds
    return 'white' if brightness < 0.5 else 'black'

#---------------------------------------------------------------------------------------
# Create a function to plot the chord timeline with multiple rows
def plot_chord_timeline_multirow(chords, total_beats, beats_per_row=32):
    """
    Plot a chord timeline with multiple rows for better readability in publications.
    Completely removes bar-based visualization and focuses only on beats.
    
    Args:
        chords: List of chord dictionaries from analysis
        total_beats: Total number of beats in the piece
        beats_per_row: Number of beats to display in each row
    """
    # Calculate number of rows needed
    num_rows = (total_beats + beats_per_row - 1) // beats_per_row
    
    # Create figure with appropriate dimensions
    fig, axes = plt.subplots(num_rows, 1, figsize=(15, 2.5*num_rows + 1), sharex=False)
    
    # Ensure axes is always a list even with single row
    if num_rows == 1:
        axes = [axes]
    
    # Process each row
    for row in range(num_rows):
        ax = axes[row]
        
        # Calculate beat range for this row
        start_beat = row * beats_per_row
        end_beat = min((row + 1) * beats_per_row, total_beats)
        
        # Find chords that overlap with this row's time window
        row_chords = [c for c in chords 
                     if c['beat_end'] > start_beat and c['beat_start'] < end_beat]
        
        # Draw the chords for this row
        for chord in row_chords:
            # Get chord properties with proper bounds for this row
            chord_start = max(chord['beat_start'], start_beat)
            chord_end = min(chord['beat_end'], end_beat)
            name = chord['chord_name']
            chord_type = chord['chord_type']
            
            # Skip if chord doesn't actually appear in this row's range
            if chord_end <= chord_start:
                continue
                
            # Adjust position to be relative to the row's start
            relative_start = chord_start - start_beat
            relative_end = chord_end - start_beat
            
            # First try the exact chord type
            if chord_type in CHORD_COLORS:
                color = CHORD_COLORS[chord_type]
            # Then try prefixes for types like m7, maj7, etc.
            elif chord_type.startswith('m7'):
                color = CHORD_COLORS['m7']
            elif chord_type.startswith('m'):
                color = CHORD_COLORS['m']
            elif chord_type.startswith('maj'):
                color = CHORD_COLORS['maj7']
            elif chord_type.endswith('7'):
                color = CHORD_COLORS['7']
            else:
                color = CHORD_COLORS['default']
            
            # Determine optimal text color based on background
            text_color = get_text_color(color)
            
            # Draw chord rectangle
            ax.add_patch(plt.Rectangle(
                (relative_start, 0.1), 
                relative_end - relative_start, 
                0.8, 
                facecolor=color,
                edgecolor='black',
                linewidth=0.25,
                alpha=0.7
            ))
            
            # Add chord name
            ax.text(
                relative_start + (relative_end - relative_start) / 2, 
                0.5, 
                name,
                ha='center',
                va='center',
                fontsize=10,
                rotation=90,
                rotation_mode='anchor',
                color=text_color
            )

            # Add the confidence number if available
            if 'confidence' in chord:
                confidence = chord.get('confidence', 0)
                ax.text(
                    relative_start + (relative_end - relative_start) / 2, 
                    0.8,
                    f"{confidence:.1f}",
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    color=text_color
                )
        
        # Draw beat lines as a grid
        for beat_offset in range(beats_per_row):
            # Only draw if within row's range
            if beat_offset <= end_beat - start_beat:
                # Determine beat line style - make every 4th beat slightly darker for rhythm reference
                # but don't label it as a bar or measure
                is_quaternary_beat = (start_beat + beat_offset) % 4 == 0
                
                ax.axvline(
                    x=beat_offset, 
                    color='gray',
                    linestyle='-' if is_quaternary_beat else ':',
                    alpha=0.4 if is_quaternary_beat else 0.2,
                    linewidth=0.7 if is_quaternary_beat else 0.5
                )
        
        # Configure this row's axes
        ax.set_xlim(0, end_beat - start_beat)
        ax.set_ylim(0, 1)
        
        # Row title shows beat range
        ax.set_title(f"Beats {start_beat} to {end_beat-1}", fontsize=10)
        
        # Remove y-axis ticks and labels
        ax.set_yticks([])
        
        # Add x-axis with beat numbers
        beat_ticks = list(range(0, min(beats_per_row, end_beat - start_beat) + 1, 4))
        beat_labels = [f"{start_beat + i}" for i in beat_ticks]
        ax.set_xticks(beat_ticks)
        ax.set_xticklabels(beat_labels)
        ax.set_xlabel("Beat", fontsize=8)
    
    # Add a common legend at the bottom
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS[''],     alpha=0.7, edgecolor='black', linewidth=0.25, label='Major'),
        plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS['m'],    alpha=0.7, edgecolor='black', linewidth=0.25, label='Minor'),
        plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS['7'],    alpha=0.7, edgecolor='black', linewidth=0.25, label='Dominant 7th'),
        plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS['maj7'], alpha=0.7, edgecolor='black', linewidth=0.25, label='Major 7th'),
        plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS['m7'],   alpha=0.7, edgecolor='black', linewidth=0.25, label='Minor 7th'),
        plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS['dim'],  alpha=0.7, edgecolor='black', linewidth=0.25, label='Diminished'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.03), 
          ncol=4, frameon=False)
    
    # Add overall title focusing on beats
    plt.suptitle(f'Chord Progression Timeline ({total_beats} beats)', fontsize=14, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.92)
    
    # Save as vector graphic for paper
    # filePath = "Figures/chord_progression"
    # plt.savefig(f"{filePath}.pdf", bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    print(f"Visualization saved as: chord_progression.pdf")
    
    
#----------------------------------------------------------------------------------
# Function to extract all chords with defined length
def extract_all_chords_with_defined_length(notes, bpm,  beats_per_bar=4, window_size_beats=1, merge=False, verbose=False, plot=True):
    """
    Extract chords using fixed-length windows (half-bar by default), using bass notes to confirm root
    
    Args:
        notes: Combined notes from bass and harmony
        beats_per_bar: Beats per bar
        window_size_beats: Window size in beats (default: 2, half-bar in 4/4 time)
        bars_per_row: Number of bars to display in each row of the visualization
        verbose: If True, return full chord data; if False, return summary string
        plot: If True, plot the multi-row timeline; if False, skip plotting
    """
    # Temporarily disable all print statements
    import sys
    original_stdout = sys.stdout
    sys.stdout = open('/dev/null', 'w')
    
    # Temporarily disable all plotting
    original_plt_show = plt.show
    plt.show = lambda: None
    
    # Close all existing figures to prevent warnings
    plt.close('all')
    
    try:
        # Determine the total length of the song automatically
        if not notes:
            return [] if verbose else "No notes found"
            
        # Find the last note's end time
        last_note_end = max(note['end'] for note in notes)
        
        # Calculate total number of bars (rounding up to include partial bars)
        import math
        total_bars = math.ceil(last_note_end / beats_per_bar)
        
        # Calculate total song beats
        total_beats = total_bars * beats_per_bar
        
        # Extract all chords using consistent fixed-size windows
        all_chords = []
        
        # Process the song in fixed-size windows
        windows_per_bar = beats_per_bar / window_size_beats
        
        # For each window
        for window_idx in range(math.ceil(total_beats / window_size_beats)):
            window_start = window_idx * window_size_beats
            window_end = window_start + window_size_beats
            
            # Ensure we don't go beyond the song length
            window_end = min(window_end, total_beats)
            
            # Calculate which bar this window is in
            bar_idx = int(window_start / beats_per_bar)
            
            # Find notes active in this window
            active_notes = [n for n in notes if n['start'] < window_end and n['end'] > window_start]
            
            # Skip empty windows
            if not active_notes:
                continue
            
            # Separate bass and harmony notes
            bass_notes = [n for n in active_notes if n['source'] == 'bass']
            harmony_notes = [n for n in active_notes if n['source'] == 'harmony']
            
            # Skip if no harmony notes (need some harmony for a chord)
            if not harmony_notes:
                continue
            
            # Calculate pitch class durations in the window for ALL notes (bass and harmony)
            pc_durations = {}
            pc_to_name = {}
            pc_to_lowest_pitch = {}
            pc_to_source = {}  # Track whether a pitch class appears in bass, harmony, or both
            
            # Process ALL notes to get combined pitch class statistics
            for note in active_notes:
                # Calculate overlap with the window
                overlap_start = max(window_start, note['start'])
                overlap_end = min(window_end, note['end'])
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
                        min(bn['end'], window_end) - max(bn['start'], window_start)
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
                    overlap_start = max(window_start, note['start'])
                    overlap_end = min(window_end, note['end'])
                    overlap_duration = overlap_end - overlap_start
                    
                    position_score = 120 - lowest_pitch  # Lower pitches score higher
                    duration_score = pc_durations[pc]
                    
                    # Prioritize strong beats
                    beat_score = 0
                    beat_in_bar = note['start'] % beats_per_bar
                    if beat_in_bar < 0.5 or (beat_in_bar >= 2 and beat_in_bar < 2.5):
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
            interval_durations[0] = window_end - window_start
            
            # Calculate intervals for harmony notes
            for hn in harmony_notes:
                # Calculate overlap with the window
                overlap_start = max(window_start, hn['start'])
                overlap_end = min(window_end, hn['end'])
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
            
            # Use the identify_unique_chord function to get refined chord type
            refined_results = identify_unique_chord([window_result])
            refined_chord_type = refined_results[0]['refined_chord_type']
            confidence = refined_results[0].get('confidence', 0)
            
            chord_name = f"{fix_note_name(root_name)}{refined_chord_type}"
            
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
                
            # Store chord with fixed window boundaries
            all_chords.append({
                'beat_start': window_start,
                'beat_end': window_end,
                'root': root_name,
                'chord_type': refined_chord_type,
                'chord_name': chord_name,
                'intervals': significant_intervals,
                'source': source_type,
                'confidence': confidence,
                'timestamp': window_start * (60 / bpm) 
            })
                
    finally:
        # Restore stdout and plt.show
        sys.stdout = original_stdout
        plt.show = original_plt_show
    
    # Sort chords by start time
    all_chords.sort(key=lambda x: x['beat_start'])
    if merge: all_chords = merge_adjacent_chords(all_chords)
    
    if plot:
        # Plot the multi-row timeline only if requested
        total_beats = total_bars * beats_per_bar
        plot_chord_timeline_multirow(all_chords, total_beats, 64)
    
    # Otherwise return the full chord data
    return all_chords

#---------------------------------------------------------------------------
def merge_adjacent_chords(chords, gap_tolerance=0.0):
    """
    Merge adjacent chords that have the same root. Among chords with slight differences,
    choose the one with the highest confidence as representative.
    
    Args:
        chords (list): List of chord dictionaries. Each chord should have at least:
                       'root', 'beat_start', 'beat_end', and 'confidence'.
        gap_tolerance (float): Maximum gap (in beats) allowed between chords to consider them adjacent.
        
    Returns:
        list: A new list of merged chord dictionaries.
    """
    if not chords:
        return []
    
    # Ensure chords are sorted by start time.
    chords.sort(key=lambda c: c['beat_start'])
    
    merged_chords = []
    # Start with the first chord as the current group.
    current_group = [chords[0]]
    
    for chord in chords[1:]:
        # If the current chord has the same root and is contiguous (allowing a small gap),
        # then add it to the current group.
        if (chord['root'] == current_group[-1]['root'] and 
            chord['beat_start'] - current_group[-1]['beat_end'] <= gap_tolerance):
            current_group.append(chord)
        else:
            # Process the current group into one merged chord.
            merged_chord = merge_group(current_group)
            merged_chords.append(merged_chord)
            # Start a new group with the current chord.
            current_group = [chord]
    
    # Process any remaining group.
    if current_group:
        merged_chords.append(merge_group(current_group))
    
    return merged_chords

#---------------------------------------------------------------------------
def merge_group(group):
    """
    Merge a group of chords (all with the same root) into one chord.
    The merged chord will span from the start of the first chord to the end of the last chord.
    The representative chord (for chord type and confidence) is chosen as the one with the highest confidence.
    
    Args:
        group (list): List of chord dictionaries with the same 'root'.
        
    Returns:
        dict: A single chord dictionary representing the merged group.
    """
    # Determine the overall start and end
    merged_start = min(chord['beat_start'] for chord in group)
    merged_end = max(chord['beat_end'] for chord in group)
    merged_duration = merged_end - merged_start

    # Pick the chord with the highest confidence as the representative
    best_chord = max(group, key=lambda c: c.get('confidence', 0))
    
    # Create a merged chord dictionary.
    merged_chord = best_chord.copy()
    merged_chord['beat_start'] = merged_start
    merged_chord['beat_end'] = merged_end
    merged_chord['duration'] = merged_duration
    # Optionally, you could recompute the confidence as, say, the average or maximum.
    # Here we simply keep the highest confidence.
    
    return merged_chord


def clean_bass_line(bass_notes, verbose=True):
    """
    Clean a potentially polyphonic bass line to produce a monophonic bass line.
    At each point in time, select only the lowest-pitched note.
    
    Args:
        bass_notes: List of bass notes as extracted by load_midi_notes
        verbose: If True, print information about processing
        
    Returns:
        List of bass notes with only a single note active at any time
    """
    if not bass_notes:
        return []
    
    # Track all times where the active note set changes
    change_points = []
    for note in bass_notes:
        change_points.append((note['start'], 'start', note))
        change_points.append((note['end'], 'end', note))
    
    # Sort by time, then put 'end' events before 'start' events at the same time
    # This ensures we remove ending notes before adding new ones at the same time point
    change_points.sort(key=lambda x: (x[0], 0 if x[1] == 'end' else 1))
    
    # Process the change points to build the cleaned bass line
    active_notes = []  # Notes currently active
    current_lowest = None  # Current lowest active note
    bass_line = []  # Our output
    last_time = None  # Last time we saw a change
    
    for time, event_type, note in change_points:
        # If we have a current lowest note and the time has changed,
        # add the note segment to our output
        if current_lowest and last_time is not None and time > last_time:
            # Create a new note object copying all properties from the original
            new_note = current_lowest.copy()
            new_note['start'] = last_time
            new_note['end'] = time
            bass_line.append(new_note)
        
        # Update active notes based on event
        if event_type == 'start':
            active_notes.append(note)
        else:  # event_type == 'end'
            if note in active_notes:
                active_notes.remove(note)
        
        # Find the current lowest note
        if active_notes:
            current_lowest = min(active_notes, key=lambda x: x['pitch'])
        else:
            current_lowest = None
        
        # Update last_time
        last_time = time
    
    # Merge adjacent segments with the same pitch
    merged_bass = []
    current_note = None
    
    for note in bass_line:
        if current_note is None:
            current_note = note.copy()
        elif (current_note['pitch'] == note['pitch'] and 
              abs(current_note['end'] - note['start']) < 1e-6):  # Allow for floating point imprecision
            # Extend current note
            current_note['end'] = note['end']
        else:
            # Add completed note and start a new one
            merged_bass.append(current_note)
            current_note = note.copy()
    
    # Add the last note if it exists
    if current_note is not None:
        merged_bass.append(current_note)
    
    if verbose:
        parallel_notes_removed = sum(1 for t, e, n in change_points if e == 'start') - len(merged_bass)
        print(f"Original bass notes: {len(bass_notes)}")
        print(f"Cleaned bass notes: {len(merged_bass)}")
        print(f"Removed {parallel_notes_removed} parallel bass notes")
        
        original_duration = sum(n['end'] - n['start'] for n in bass_notes)
        cleaned_duration = sum(n['end'] - n['start'] for n in merged_bass)
        print(f"Original total duration: {original_duration:.2f} beats")
        print(f"Cleaned total duration: {cleaned_duration:.2f} beats")
        
        if bass_notes:
            print(f"Original time span: {min(n['start'] for n in bass_notes):.2f} to {max(n['end'] for n in bass_notes):.2f}")
        if merged_bass:
            print(f"Cleaned time span: {min(n['start'] for n in merged_bass):.2f} to {max(n['end'] for n in merged_bass):.2f}")
    
    return merged_bass

#---------------------------------------------------------------------------
def play_audio_file(song_id, base_path="../samples/suno_samples/audio", verbose=True):
    """
    Create an audio player widget for a song file
    
    Args:
        song_id: ID of the song (filename without extension)
        base_path: Base path to the audio files
        verbose: Whether to print detailed information
        
    Returns:
        IPython audio widget that plays the song
    """
    from IPython.display import Audio, display
    
    # Construct file path
    file_path = os.path.join(base_path, f"{song_id}.mp3")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        print("Please check the path to your audio file")
        return None
    
    if verbose:
        # Get file info
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        print(f"✓ File exists: {file_path}")
        print(f"✓ File size: {file_size:.2f} MB")
    
    # Create and return the audio player widget
    return Audio(file_path, autoplay=False)

#---------------------------------------------------------------------------
def compare_bass_lines(original_bass, cleaned_bass, num_bars=4, start_bar=0, beats_per_bar=4):
    """
    Visualize the original and cleaned bass lines side by side for comparison
    
    Args:
        original_bass: Original potentially polyphonic bass notes
        cleaned_bass: Cleaned monophonic bass notes
        num_bars: Number of bars to display
        start_bar: Starting bar index
        beats_per_bar: Number of beats per bar
    """
    import matplotlib.pyplot as plt
    import music21.pitch as pitch
    
    # Calculate start and end beats
    start_beat = start_bar * beats_per_bar
    end_beat = (start_bar + num_bars) * beats_per_bar
    
    # Filter notes within the specified bars
    original_notes = [n for n in original_bass if n['end'] > start_beat and n['start'] < end_beat]
    cleaned_notes = [n for n in cleaned_bass if n['end'] > start_beat and n['start'] < end_beat]
    
    if not original_notes and not cleaned_notes:
        print(f"No bass notes found in bars {start_bar+1} to {start_bar+num_bars}")
        return
    
    # Set up the plot with two subplots (original and cleaned)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    
    # Plot original bass notes
    ax1.set_title("Original Bass Line (Potentially Polyphonic)")
    for note_data in original_notes:
        # Trim notes to the visible range
        start = max(note_data['start'], start_beat)
        end = min(note_data['end'], end_beat)
        
        # Plot the note
        ax1.barh(
            note_data['pitch'], 
            end - start, 
            left=start, 
            height=0.8, 
            color='red', 
            alpha=0.7
        )
        
        # Add note name
        ax1.text(
            start + 0.1, 
            note_data['pitch'] + 0.3, 
            note_data['name'], 
            fontsize=9,
            color='black',
            bbox=dict(facecolor='white', alpha=0.7, pad=1)
        )
    
    # Plot cleaned bass notes
    ax2.set_title("Cleaned Bass Line (Monophonic)")
    for note_data in cleaned_notes:
        # Trim notes to the visible range
        start = max(note_data['start'], start_beat)
        end = min(note_data['end'], end_beat)
        
        # Plot the note
        ax2.barh(
            note_data['pitch'], 
            end - start, 
            left=start, 
            height=0.8, 
            color='blue', 
            alpha=0.7
        )
        
        # Add note name
        ax2.text(
            start + 0.1, 
            note_data['pitch'] + 0.3, 
            note_data['name'], 
            fontsize=9,
            color='black',
            bbox=dict(facecolor='white', alpha=0.7, pad=1)
        )
    
    # Add bar lines to both plots
    for ax in [ax1, ax2]:
        for bar in range(start_bar, start_bar + num_bars + 1):
            bar_pos = bar * beats_per_bar
            ax.axvline(x=bar_pos, color='black', linestyle='--', alpha=0.5)
            if bar < start_bar + num_bars:
                ax.text(bar_pos + 0.1, ax.get_ylim()[1] - 3, f"Bar {bar+1}", fontsize=10)
        
        # Add beat markers
        for beat in range(int(start_beat), int(end_beat) + 1):
            ax.axvline(x=beat, color='gray', linestyle=':', alpha=0.3)
    
    # Set common labels
    fig.text(0.5, 0.04, 'Beat Position', ha='center', va='center', fontsize=12)
    fig.text(0.06, 0.5, 'MIDI Pitch', ha='center', va='center', rotation='vertical', fontsize=12)
    
    # Set y-axis to show note names on both plots
    min_pitch = min([n['pitch'] for n in original_notes + cleaned_notes]) - 5
    max_pitch = max([n['pitch'] for n in original_notes + cleaned_notes]) + 5
    
    yticks = range(min_pitch, max_pitch + 1, 2)
    ylabels = [pitch.Pitch(p).nameWithOctave for p in yticks]
    
    for ax in [ax1, ax2]:
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_xlim(start_beat, end_beat)
        ax.set_ylim(min_pitch, max_pitch)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    
    # Print statistics
    original_notes_count = len(original_notes)
    cleaned_notes_count = len(cleaned_notes)
    
    print(f"Original bass notes in selected bars: {original_notes_count}")
    print(f"Cleaned bass notes in selected bars: {cleaned_notes_count}")
    
    # Check for polyphonic sections
    time_points = sorted(set([n['start'] for n in original_notes] + [n['end'] for n in original_notes]))
    polyphonic_moments = 0
    
    for i in range(len(time_points) - 1):
        t_start = time_points[i]
        t_end = time_points[i + 1]
        
        if t_start >= end_beat or t_end <= start_beat:
            continue
            
        # Count active notes in this time slice
        active_notes = [n for n in original_notes 
                      if n['start'] <= t_start and n['end'] >= t_end]
                      
        if len(active_notes) > 1:
            polyphonic_moments += 1
    
    print(f"Polyphonic moments in these bars: {polyphonic_moments}")
    

#---------------------------------------------------------------------------------------
def visualize_waveform_with_bass(audio_path, bass_notes, bpm=120, start_time=0, duration=30, beats_per_bar=4):
    """
    Visualize audio waveform alongside bass notes to check alignment.
    
    Args:
        audio_path: Path to audio file
        bass_notes: List of bass notes from MIDI
        bpm: Beats per minute
        start_time: Start time in seconds for visualization
        duration: Duration in seconds to visualize
        beats_per_bar: Number of beats per bar
    """
    import librosa
    import librosa.display
    import numpy as np
    import matplotlib.pyplot as plt
    import music21.pitch as pitch
    from IPython.display import Audio, display
    
    # Load audio file
    print(f"Loading audio: {audio_path}")
    try:
        y, sr = librosa.load(audio_path, offset=start_time, duration=duration)
        print(f"✓ Audio loaded: {len(y)/sr:.2f} seconds @ {sr}Hz")
    except Exception as e:
        print(f"Error loading audio: {str(e)}")
        return
    
    # Create figure with two aligned subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    
    # Plot waveform on top subplot - explicitly set color to avoid cycler issue
    try:
        librosa.display.waveshow(y, sr=sr, ax=ax1, color='blue')
        ax1.set_title("Audio Waveform")
        ax1.label_outer()  # Hide x-axis labels for top subplot
    except AttributeError:
        # Manual fallback if librosa.display.waveshow fails
        print("Using fallback waveform display method")
        times = np.linspace(0, len(y)/sr, len(y))
        ax1.plot(times, y, color='blue', linewidth=0.5)
        ax1.set_title("Audio Waveform")
        ax1.label_outer()  # Hide x-axis labels for top subplot
    
    # Highlight louder parts of the waveform that may contain bass
    y_abs = np.abs(y)
    threshold = np.percentile(y_abs, 90)  # Top 10% of amplitude
    high_amplitude = np.where(y_abs > threshold)[0]
    
    # Add a thinner plot highlighting potential bass frequencies
    try:
        # Extract bass frequencies (20-200Hz)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        y_bass = librosa.effects.low_pass(y_harmonic, sr=sr, cutoff=200)
        y_bass_envelope = np.abs(librosa.stft(y_bass)).mean(axis=0)
        
        # Normalize and plot bass envelope
        y_bass_envelope = y_bass_envelope / np.max(y_bass_envelope) if np.max(y_bass_envelope) > 0 else y_bass_envelope
        times = librosa.times_like(y_bass_envelope, sr=sr)
        ax1.plot(times, y_bass_envelope * 0.5, color='red', alpha=0.8, linewidth=1, label='Bass envelope')
        ax1.legend()
    except Exception as e:
        print(f"Could not extract bass frequencies: {str(e)}")
    
    # Convert beat positions to time positions
    seconds_per_beat = 60 / bpm
    
    # Filter bass notes within the time range
    start_beat = start_time / seconds_per_beat
    end_beat = (start_time + duration) / seconds_per_beat
    
    visible_notes = [n for n in bass_notes 
                    if n['end'] > start_beat and n['start'] < end_beat]
    
    # Plot bass notes on bottom subplot
    if visible_notes:
        min_pitch = max(24, min(n['pitch'] for n in visible_notes) - 3)
        max_pitch = min(60, max(n['pitch'] for n in visible_notes) + 3)
    else:
        min_pitch = 30
        max_pitch = 50
    
    # Create vertical grid lines for beats and bars
    num_beats = int(duration / seconds_per_beat) + 1
    for i in range(num_beats):
        beat_time = i * seconds_per_beat
        bar_position = (start_beat + i) % beats_per_bar
        
        if bar_position == 0:
            # Bar line (thicker)
            ax1.axvline(x=beat_time, color='black', linestyle='-', linewidth=0.8, alpha=0.6)
            ax2.axvline(x=beat_time, color='black', linestyle='-', linewidth=0.8, alpha=0.6)
            
            # Add bar number
            bar_num = int((start_beat + i) / beats_per_bar) + 1
            ax2.text(beat_time + 0.1, max_pitch + 1, f"Bar {bar_num}", fontsize=8)
        else:
            # Regular beat line
            ax1.axvline(x=beat_time, color='gray', linestyle=':', linewidth=0.5, alpha=0.4)
            ax2.axvline(x=beat_time, color='gray', linestyle=':', linewidth=0.5, alpha=0.4)
    
    # Plot each bass note
    for note in visible_notes:
        # Convert beat positions to seconds
        note_start_sec = max(0, (note['start'] - start_beat) * seconds_per_beat)
        note_end_sec = min(duration, (note['end'] - start_beat) * seconds_per_beat)
        
        if note_end_sec <= note_start_sec:
            continue
            
        # Draw rectangle for the note
        rect = plt.Rectangle(
            (note_start_sec, note['pitch'] - 0.4),
            note_end_sec - note_start_sec,
            0.8,
            color='blue',
            alpha=0.7
        )
        ax2.add_patch(rect)
        
        # Add note name
        ax2.text(
            note_start_sec + (note_end_sec - note_start_sec) / 2,
            note['pitch'] + 0.4,
            note['name'],
            ha='center',
            va='center',
            fontsize=9
        )
    
    # Set axes properties for bass notes
    ax2.set_ylim(min_pitch - 3, max_pitch + 3)
    ax2.set_ylabel("MIDI Pitch")
    ax2.set_xlabel("Time (seconds)")
    
    # Set y-axis to show note names
    yticks = list(range(min_pitch, max_pitch + 1, 2))
    ylabels = [pitch.Pitch(p).nameWithOctave for p in yticks]
    
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabels)
    ax2.set_xlim(start_beat, end_beat)
    ax2.set_ylim(min_pitch, max_pitch)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    
    # Show the plot
    plt.show()
    
    # Display audio player for this segment
    print(f"Audio segment ({start_time}-{start_time+duration} seconds):")
    display(Audio(y, rate=sr))
    
    return fig, (ax1, ax2)

#---------------------------------------------------------------------------
def analyze_audio_bass_alignment(audio_path, bass_notes, bpm=120, segments=3, segment_duration=20):
    """
    Analyze alignment between audio and bass notes across multiple segments
    
    Args:
        audio_path: Path to audio file
        bass_notes: List of bass notes from MIDI
        bpm: Beats per minute
        segments: Number of segments to analyze
        segment_duration: Duration in seconds for each segment
    """
    from IPython.display import display, Markdown
    import librosa
    
    # Get audio duration
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=10)  # Load just a bit to get sample rate
        audio_info = librosa.get_duration(filename=audio_path)
        print(f"Audio duration: {audio_info:.2f} seconds ({audio_info/60:.2f} minutes)")
    except Exception as e:
        print(f"Error loading audio file: {str(e)}")
        audio_info = segment_duration * segments  # Fallback duration
    
    # If very short audio, adjust segments
    if audio_info < segment_duration * segments:
        segments = max(1, int(audio_info / segment_duration))
        print(f"Adjusted to {segments} segments due to audio duration")
    
    # Calculate seconds per beat
    seconds_per_beat = 60 / bpm
    
    # Convert beat positions to time
    max_time = max(n['end'] for n in bass_notes) * seconds_per_beat if bass_notes else audio_info
    
    # Choose spaced out segments to analyze
    segment_points = []
    if segments == 1:
        segment_points = [0]  # Just the beginning
    else:
        # Distribute segments evenly across the track
        segment_points = [i * (min(max_time, audio_info) - segment_duration) / (segments - 1) for i in range(segments)]
    
    # Analyze each segment
    for i, start_time in enumerate(segment_points):
        start_time = int(start_time)  # Round to nearest second
        display(Markdown(f"## Segment {i+1}: {start_time}-{start_time+segment_duration} seconds"))
        
        # Calculate beat range for reference
        start_beat = start_time / seconds_per_beat
        end_beat = (start_time + segment_duration) / seconds_per_beat
        
        # Count bass notes in this segment
        segment_notes = [n for n in bass_notes 
                       if n['end'] * seconds_per_beat > start_time and 
                          n['start'] * seconds_per_beat < start_time + segment_duration]
        
        print(f"Bass notes in segment: {len(segment_notes)}")
        
        # Visualize this segment
        visualize_waveform_with_bass(
            audio_path, 
            bass_notes, 
            bpm=bpm,
            start_time=start_time,
            duration=segment_duration
        )
        
        print("\n" + "-"*80 + "\n")
        
#---------------------------------------------------------------------------

def convert_to_music21_notation(chord_string):
    """
    Converts standard chord notation to music21-compatible notation.
    Args:
        chord_string: A chord name like 'Bb', 'Ebm7', 'Abmaj7'
    Returns:
        The chord name with music21 notation
    """
    if not isinstance(chord_string, str):
        return chord_string
        
    # First handle double flats (bb) - must be done before single flats
    chord_string = chord_string.replace('bb', '--')
    
    # Regex to find a note letter followed by a flat symbol
    pattern = r'([A-G])b'
    # Replace each occurrence with the note followed by '-'
    result = re.sub(pattern, r'\1-', chord_string)
    
    # Fix chord types that music21 doesn't understand
    # Convert extended chords to basic seventh chords for analysis
    result = re.sub(r'maj9', r'maj7', result)
    result = re.sub(r'9', r'7', result)  # Change dominant 9 to dominant 7
    result = re.sub(r'11', r'7', result)  # Change 11th to 7th
    result = re.sub(r'13', r'7', result)  # Change 13th to 7th
    result = re.sub(r'm9', r'm7', result)  # Change minor 9 to minor 7
    
    return result

#---------------------------------------------------------------------------
# def separate_roman_numeral(figure):
#     """Separate a Roman numeral figure into base function and alterations"""
#     # Order Roman numerals by length (descending) to avoid partial matches
#     base_numerals = sorted(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 
#                             'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii'], 
#                            key=len, reverse=True)
    
#     # Find the base Roman numeral
#     base = ""
#     alterations = ""
#     found_base = False
    
#     for numeral in base_numerals:
#         if figure.startswith(numeral):
#             base = numeral
#             found_base = True
#             alterations = figure[len(numeral):]
#             break
    
#     # Handle special cases
#     if not found_base:
#         # Check for flat or sharp modifiers
#         for prefix in ['b', '#']:
#             for numeral in base_numerals:
#                 if figure.startswith(prefix + numeral):
#                     base = prefix + numeral
#                     alterations = figure[len(base):]
#                     found_base = True
#                     break
#             if found_base:
#                 break
    
#     # If still not found, just return the whole thing as base
#     if not found_base:
#         base = figure
#         alterations = ""
        
#     return base, alterations

# ---------------------------------------------------------------------------
# --- CHORD NAME CANONICALIZATION ---
CHORD_NAME_CANONICAL = {
    # Half-diminished
    'b5b7': 'ø7', '-5-7': 'ø7', '-5b7': 'ø7', 'b5-7': 'ø7', 'm7b5': 'ø7', 'm7-5': 'ø7',
    # Diminished (triad)
    'b5': 'o', '-5': 'o',
    # Minor seventh
    'mb7': 'm7', 'm-7': 'm7',
    # Faulty analyzer output
    '7b7': '7', 'b7': '7', '57': '7', '-7': '7', '5b7': '7', '-57': '7',
    'maj7b7': 'M7',
}

# --- CHORD TYPE MAPPING FOR MUSIC21 ---

CHORD_TYPE_TO_MUSIC21 = {
    # Basic triads
    '': '',
    'm': 'm',
    'dim': 'dim', 'o': 'dim',
    'aug': 'aug', '+': 'aug',

    # Sevenths and extensions
    '7': '7',
    'm7': 'm7',
    'M7': 'M7', 'maj7': 'M7',
    'ø7': 'm7b5',           # half-dim, music21 likes m7b5
    'dim7': 'dim7', 'o7': 'dim7',
    'mM7': 'mM7', 'mmaj7': 'mM7',
    
    'maj7b9': 'maj7b9',
    '59': 'add9',        # or '' if you prefer major triad
    '5b9': '7b9',
    'b59': '7b9',

    # Ninths and further
    '9': '9', 'm9': 'm9', 'M9': 'M9', 'maj9': 'M9',
    'm7b5': 'm7b5',  # valid for music21
    'm7b9': 'm7b9',
    'm9b9': 'm9b9',
    'mb9': 'm7b9',
    '13': '13', 'm13': 'm13',
    '11': '11', 'm11': 'm11',

    # Common alterations/extensions (dominant by default)
    '7b9': '7b9', '7#9': '7#9', '7#11': '7#11', '7b13': '7b13',
    '7b5': '7b5', '7#5': '7#5', '7b5b9': '7b5b9', '7#5b9': '7#5b9',
    '7#5#9': '7#5#9', '7#9b5': '7b5#9',  # music21 accepts

    # Suspended and power chords
    'sus4': 'sus4', '7sus4': '7sus4', '7sus': '7sus4', 'sus': 'sus4',
    '5': '',  # power chord: treat as major

    # "Alt" is ambiguous; use closest music21 approximation
    '7alt': '7b13',
}

# --- EXTENSION-ONLY TYPES: PREPEND '7' ---

DOMINANT_EXTENSIONS = {
    'b9', '#9', 'b5', '#5', 'b5b9', 'b9#5', '#9b5', 'b13', '#13', '#11'
}

# --- FLAT/SHARP HANDLING ---

def convert_to_music21_notation(chord_name):
    """
    Converts 'b' and '#' in chord root to Music21-friendly notation.
    Example: 'Bbmaj7' -> 'B-maj7', 'F#7' -> 'F#7'
    """
    m = re.match(r'^([A-Ga-g])([b#-]?)(.*)$', chord_name)
    if not m:
        return chord_name
    root, acc, rest = m.groups()
    if acc == 'b':
        root += '-'
    elif acc == '#':
        root += '#'
    elif acc == '-':
        root += '-'
    return root + rest

# --- LOGGING FOR UNHANDLED CHORDS (DEDUPE) ---

_unhandled_chord_types_seen = set()
def log_unhandled_chord_type(label):
    if label not in _unhandled_chord_types_seen:
        print(f"[Warning] No Music21 conversion found for chord type '{label}', using as-is")
        _unhandled_chord_types_seen.add(label)

# --- MAIN CONVERSION FUNCTION ---

def convert_to_music21_chord_name(chord_name):
    """
    Convert internal chord names to Music21's chord symbol format.

    Args:
        chord_name (str): Chord name from the analyzer.

    Returns:
        str: Music21-compatible chord symbol.
    """
    if not isinstance(chord_name, str) or not chord_name:
        return chord_name

    # 1. Canonicalize: fix non-standard labels
    base = chord_name.strip()
    base = CHORD_NAME_CANONICAL.get(base, base)

    # 2. Separate root and chord type
    base = convert_to_music21_notation(base)
    if len(base) > 1 and base[1] in ['-', '#']:
        root = base[:2]
        chord_type = base[2:]
    else:
        root = base[:1]
        chord_type = base[1:]

    root = root.strip().capitalize()
    chord_type = chord_type.strip()

    # 3. Extension-only types: assume dominant ('7')
    if chord_type in DOMINANT_EXTENSIONS:
        chord_type = '7' + chord_type

    # 4. Map to music21
    music21_chord_type = CHORD_TYPE_TO_MUSIC21.get(chord_type)
    if music21_chord_type is None:
        # As last resort, pass as-is (music21 may still recognize) and warn
        music21_chord_type = chord_type
        log_unhandled_chord_type(chord_type)

    return f"{root}{music21_chord_type}"

#---------------------------------------------------------------------------
def separate_roman_numeral(figure):
    """
    Splits the figure into function (accidentals + roman numeral) and alteration (the rest).
    """
    m = re.match(r"^([b#]*[IViv]+)(.*)$", figure)
    if m:
        return m.group(1), m.group(2) or ''
    return figure, ''

    
#---------------------------------------------------------------------------
def clean_half_diminished(roman, alt, chord_type=None):
    if alt == "b5b3" and chord_type in ("m7b5", "ø7"):
        return roman, "m7b5"
    return roman, alt

#---------------------------------------------------------------------------
def functional_chords(chord_data, tonality='C'):
    """
    Enhances the existing chord data by adding functional harmony information.
    Returns each chord with a functional_harmony dict:
      {'functional': full roman figure,
       'roman_numeral': base (pure roman),
       'alterations': trailing digits or modifications,
       'music21_name': ... }
    """
    
    if isinstance(chord_data, dict) and 'chords' in chord_data:
        chords = chord_data['chords']
        is_dict_format = True
    else:
        chords = chord_data
        is_dict_format = False
    
    # Properly parse the tonality
    tonic = None
    mode = 'major'
    if isinstance(tonality, str):
        tonality = convert_to_music21_notation(tonality)
        tonality = tonality.strip()
        if ' ' in tonality:
            parts = tonality.split()
            tonic = parts[0]
            mode = parts[1].lower()
        elif tonality.endswith('m') and len(tonality) > 1:
            tonic = tonality[:-1]
            mode = 'minor'
        else:
            tonic = tonality
    
    try:
        k = music21.key.Key(tonic, mode)
    except Exception as e:
        print(f"Error creating key from {tonality}: {str(e)}")
        assert False, f"Invalid tonality: {tonality}"
    
    for chord in chords:
        try:
            if 'chord_name' in chord:
                chord_name_m21 = convert_to_music21_chord_name(chord['chord_name'])
                c = music21.harmony.ChordSymbol(chord_name_m21)
                rn = music21.roman.romanNumeralFromChord(c, k)

                # NEW: split base and alteration as you want
                base, alteration = separate_roman_numeral(rn.figure)
                # print(f"DEBUG: {rn.figure} -> function: {base}, alterations: {alteration}")
                base, alteration = clean_half_diminished(base, alteration, chord.get('chord_type', None))
                chord['functional_harmony'] = {
                    'functional': rn.figure,
                    'roman_numeral': base,
                    'alterations': alteration,
                    'music21_name': chord_name_m21
                }
            else:
                print("not doing it")   
        except Exception as e:
            chord['functional_harmony'] = {
                'functional': 'Unknown',
                'alterations': '',
                'roman_numeral': 'Unknown',
                'error': str(e),
                'original_chord': chord.get('chord_name', 'Unknown')
            }
            print(f"Error analyzing chord {chord.get('chord_name', 'Unknown')}: {str(e)}")
    return chords
    
#---------------------------------------------------------------------------
def enhance_midi_chords_with_audio(chords_midi, chords_audio, time_threshold=0.5, consistency_threshold=0.6):
    """
    Enhance the MIDI chords with more detailed information from audio chords
    when there's a consistent correlation pattern.
    
    Args:
        chords_midi: Dictionary with 'chords' key containing list of chord dictionaries from MIDI extraction
        chords_audio: List of ChordChange objects from audio extraction method
        time_threshold: Maximum time difference to consider chords correlated (seconds)
        consistency_threshold: Minimum ratio of consistent matches to consider a pattern
    
    Returns:
        List of enhanced MIDI chord dictionaries
    """
    
    # Handle different possible input formats
    if isinstance(chords_midi, dict) and 'chords' in chords_midi:
        # Extract the list of chords from the dictionary
        midi_chords = chords_midi['chords']
    elif isinstance(chords_midi, list):
        # Already a list
        midi_chords = chords_midi
    else:
        # Try to convert to a list if it's a string representation
        import ast
        try:
            if isinstance(chords_midi, str):
                midi_chords = ast.literal_eval(chords_midi)
            else:
                raise ValueError("Unsupported format for chords_midi")
        except:
            raise ValueError("Could not parse chords_midi data")
    
    # Parse chords_audio if it's a string representation
    if isinstance(chords_audio, str):
        # Extract ChordChange objects from the string representation
        import re
        chord_pattern = r"ChordChange\(chord='([^']+)',\s*timestamp=([\d.]+)\)"
        chord_matches = re.findall(chord_pattern, chords_audio)
        
        # Convert to a format similar to ChordChange objects
        class ChordChange:
            def __init__(self, chord, timestamp):
                self.chord = chord
                self.timestamp = timestamp
        
        parsed_chords_audio = [
            ChordChange(chord=chord, timestamp=float(timestamp))
            for chord, timestamp in chord_matches
        ]
        chords_audio = parsed_chords_audio
    
    # Create a copy of the MIDI chords to avoid modifying the original
    enhanced_chords = []
    for chord in midi_chords:
        # Create a new dict with the same structure
        new_chord = {}
        for key, value in chord.items():
            if isinstance(value, (list, set, dict)):
                # Make a deep copy of complex structures
                import copy
                new_chord[key] = copy.deepcopy(value)
            else:
                new_chord[key] = value
        enhanced_chords.append(new_chord)
    
    # Create a mapping of simple chord roots to potential enhancements
    root_to_extensions = {}
    
    # Step 1: Create a dictionary to track all possible extensions for each root
    for midi_chord in midi_chords:
        # Get the timestamp - either from beat_start or directly from timestamp
        midi_time = midi_chord.get('timestamp', midi_chord.get('beat_start', 0))
        midi_name = midi_chord.get('chord_name', '')
        
        # Extract the root of the MIDI chord (remove all extensions)
        midi_root = midi_name.rstrip('maj7').rstrip('7').rstrip('sus4').rstrip('5').rstrip('9').rstrip('m').rstrip('dim')
        
        # Find all audio chords within the time threshold
        nearby_audio_chords = [
            c for c in chords_audio 
            if abs(c.timestamp - midi_time) < time_threshold
        ]
        
        # If we found any nearby audio chords
        if nearby_audio_chords:
            # Sort by time difference to prioritize closest
            nearby_audio_chords.sort(key=lambda c: abs(c.timestamp - midi_time))
            
            # Get the closest audio chord
            closest_audio = nearby_audio_chords[0]
            audio_chord = closest_audio.chord
            
            # Extract the root of the audio chord
            audio_root = audio_chord.rstrip('maj7').rstrip('7').rstrip('sus4').rstrip('5').rstrip('9').rstrip('m').rstrip('dim')
            
            # If the roots match, record this potential extension
            if midi_root == audio_root:
                if midi_root not in root_to_extensions:
                    root_to_extensions[midi_root] = []
                
                # Record the extension pattern (from midi_name to audio_chord)
                root_to_extensions[midi_root].append({
                    'midi_chord': midi_name,
                    'audio_chord': audio_chord,
                    'midi_time': midi_time,
                    'audio_time': closest_audio.timestamp
                })
    
    # Step 2: Analyze the patterns to find consistent extensions
    consistent_extensions = {}
    
    for root, extensions in root_to_extensions.items():
        # Group extensions by MIDI chord type
        extensions_by_midi = {}
        for ext in extensions:
            midi_chord = ext['midi_chord']
            if midi_chord not in extensions_by_midi:
                extensions_by_midi[midi_chord] = []
            extensions_by_midi[midi_chord].append(ext)
        
        # For each MIDI chord type, find the most consistent audio extension
        for midi_chord, matches in extensions_by_midi.items():
            if len(matches) < 2:  # Need at least 2 occurrences to establish pattern
                continue
            
            # Count occurrences of each audio extension
            audio_counts = {}
            for match in matches:
                audio_chord = match['audio_chord']
                if audio_chord not in audio_counts:
                    audio_counts[audio_chord] = 0
                audio_counts[audio_chord] += 1
            
            # Find the most frequent audio extension
            most_common_audio = max(audio_counts.items(), key=lambda x: x[1])
            audio_chord, count = most_common_audio
            
            # Check if it's consistent enough
            consistency = count / len(matches)
            if consistency >= consistency_threshold:
                consistent_extensions[midi_chord] = {
                    'enhanced_chord': audio_chord,
                    'consistency': consistency,
                    'count': count,
                    'total': len(matches)
                }
    
    # Step 3: Apply the enhancements to our copy of MIDI chords
    for i, chord in enumerate(enhanced_chords):
        midi_name = chord.get('chord_name', '')
        if midi_name in consistent_extensions:
            # Replace with the enhanced chord name
            enhanced_chords[i]['chord_name'] = consistent_extensions[midi_name]['enhanced_chord']
            # Add metadata about the enhancement
            enhanced_chords[i]['enhancement_data'] = consistent_extensions[midi_name]
            enhanced_chords[i]['original_chord'] = midi_name
    
    # Return the enhanced chords and the identified patterns
    return enhanced_chords, consistent_extensions

def analyze_chord_functionality(data):
    """
    Analyze if chords are functionally correct for any tonality
    
    Parameters:
    data (list): List containing tonality, scale, and chord data
    
    Returns:
    dict: Analysis results
    """
    # Extract tonality and scale information
    tonality_info = next((item for item in data if isinstance(item, dict) and 'tonality' in item), None)
    scale_info = next((item for item in data if isinstance(item, dict) and 'scale' in item), None)
    
    
    
    if not tonality_info or not scale_info:
        return {"error": "Missing tonality or scale information"}
    
    tonality = tonality_info['tonality']
    scale = scale_info['scale']
    
    print("tonality:", tonality)
    print("scale:", scale)
    
    # Parse tonality to get root and quality
    tonality_parts = tonality.split()
    root = tonality_parts[0]
    quality = tonality_parts[1] if len(tonality_parts) > 1 else "major"  # Default to major if not specified
    is_minor = quality.lower() == "minor"
    
    # Extract all chord entries
    chords = [item for item in data if isinstance(item, dict) and 'chord_name' in item]
    
    # Get scale degree indexes for faster lookups
    scale_degrees = {note: idx for idx, note in enumerate(scale)}
    
    # Roman numerals for proper notation
    roman_numerals = ["I", "II", "III", "IV", "V", "VI", "VII"]
    roman_numerals_minor = ["i", "ii°", "III", "iv", "v", "VI", "VII"]
    
    # Define functional roles based on scale degrees for any key
    if is_minor:
        # Natural minor scale degrees and their functional roles
        # Corrected to properly classify diatonic chords in minor
        functional_roles = {
            0: {  # i/I - Tonic: minor is diatonic, major is borrowed
                "primary": ["m", "m7", "m6", "madd9"],  # Minor tonic (i) - diatonic
                "secondary": ["", "maj7", "6", "add9", "sus4", "7sus4"],  # Major tonic (I) - borrowed
                "role": "tonic",
                "roman": lambda chord_type: "i" if chord_type in ["m", "m7", "m6", "madd9"] else "I"
            },
            1: {  # ii° - Supertonic: diminished is diatonic, minor/major are borrowed
                "primary": ["dim", "m7b5", "ø"],  # Diminished supertonic (ii°) - diatonic
                "secondary": ["m", "m7", "", "7", "sus4"],  # Other supertonic forms - borrowed
                "role": "supertonic",
                "roman": lambda chord_type: "ii°" if chord_type in ["dim", "m7b5", "ø"] else 
                                        "ii" if chord_type in ["m", "m7"] else "II"
            },
            2: {  # III/iii - Mediant: major is diatonic, minor is borrowed
                "primary": ["", "maj7", "6", "add9"],  # Major mediant (III) - diatonic
                "secondary": ["m", "m7", "madd9"],  # Minor mediant (iii) - borrowed
                "role": "mediant",
                "roman": lambda chord_type: "III" if chord_type in ["", "maj7", "6", "add9"] else "iii"
            },
            3: {  # iv/IV - Subdominant: both common in minor context
                "primary": ["", "maj7", "6", "add9", "sus4"],  # Major subdominant (IV) - diatonic in natural minor
                "secondary": ["m", "m7", "m6", "madd9"],  # Minor subdominant (iv) - common alternate
                "role": "subdominant",
                "roman": lambda chord_type: "IV" if chord_type in ["", "maj7", "6", "add9", "sus4"] else "iv"
            },
            4: {  # v/V - Dominant: minor is diatonic in natural minor, major is from harmonic minor
                "primary": ["m", "m7"],  # Minor dominant (v) - diatonic in natural minor
                "secondary": ["", "7", "maj7", "9", "13", "sus4", "7sus4", "5"],  # Major dominant (V) - from harmonic minor
                "role": "dominant",
                "roman": lambda chord_type: "v" if chord_type in ["m", "m7"] else "V"
            },
            5: {  # VI/vi - Submediant: major is diatonic, minor is borrowed
                "primary": ["", "maj7", "6", "add9"],  # Major submediant (VI) - diatonic
                "secondary": ["m", "m7", "m6", "madd9"],  # Minor submediant (vi) - borrowed
                "role": "submediant",
                "roman": lambda chord_type: "VI" if chord_type in ["", "maj7", "6", "add9"] else "vi"
            },
            6: {  # VII/vii° - Subtonic/Leading tone: major is diatonic in natural minor, dim is from harmonic
                "primary": ["", "7", "maj7"],  # Major subtonic (VII) - diatonic in natural minor
                "secondary": ["dim", "m7b5", "ø", "m", "m7"],  # Diminished leading tone (vii°) - from harmonic minor
                "role": "subtonic",
                "roman": lambda chord_type: "VII" if chord_type in ["", "7", "maj7"] else "vii°"
            }
        }
    else:
        # Major scale degrees and their functional roles
        functional_roles = {
            0: {  # I/i - Tonic: major is diatonic, minor is borrowed
                "primary": ["", "maj7", "6", "add9", "69", "sus4"],  # Major tonic (I) - diatonic
                "secondary": ["m", "m7", "m6", "7sus4"],  # Minor tonic (i) - borrowed
                "role": "tonic",
                "roman": lambda chord_type: "I" if chord_type not in ["m", "m7", "m6"] else "i"
            },
            1: {  # ii/II - Supertonic: minor is diatonic, major/dom7 are borrowed
                "primary": ["m", "m7", "m9", "madd9"],  # Minor supertonic (ii) - diatonic
                "secondary": ["", "7", "sus4"],  # Major supertonic (II) - borrowed
                "role": "supertonic",
                "roman": lambda chord_type: "ii" if chord_type in ["m", "m7", "m9", "madd9"] else "II"
            },
            2: {  # iii/III - Mediant: minor is diatonic, major is borrowed
                "primary": ["m", "m7", "madd9"],  # Minor mediant (iii) - diatonic
                "secondary": ["", "maj7", "7"],  # Major mediant (III) - borrowed
                "role": "mediant",
                "roman": lambda chord_type: "iii" if chord_type in ["m", "m7", "madd9"] else "III"
            },
            3: {  # IV/iv - Subdominant: major is diatonic, minor is borrowed
                "primary": ["", "maj7", "6", "add9", "sus4"],  # Major subdominant (IV) - diatonic
                "secondary": ["m", "m7", "m6", "madd9"],  # Minor subdominant (iv) - borrowed
                "role": "subdominant",
                "roman": lambda chord_type: "IV" if chord_type in ["", "maj7", "6", "add9", "sus4"] else "iv"
            },
            4: {  # V/v - Dominant: major is diatonic, minor is borrowed
                "primary": ["", "7", "9", "13", "sus4", "7sus4"],  # Major dominant (V) - diatonic
                "secondary": ["m", "m7", "madd9"],  # Minor dominant (v) - borrowed
                "role": "dominant",
                "roman": lambda chord_type: "V" if chord_type not in ["m", "m7", "madd9"] else "v"
            },
            5: {  # vi/VI - Submediant: minor is diatonic, major is borrowed
                "primary": ["m", "m7", "m6", "m9"],  # Minor submediant (vi) - diatonic
                "secondary": ["", "maj7", "6"],  # Major submediant (VI) - borrowed
                "role": "submediant",
                "roman": lambda chord_type: "vi" if chord_type in ["m", "m7", "m6", "m9"] else "VI"
            },
            6: {  # vii°/VII - Leading tone: dim is diatonic, major is borrowed
                "primary": ["dim", "m7b5", "ø"],  # Diminished leading tone (vii°) - diatonic
                "secondary": ["", "m", "m7", "7"],  # Major subtonic (VII) - borrowed
                "role": "leading tone",
                "roman": lambda chord_type: "vii°" if chord_type in ["dim", "m7b5", "ø"] else "VII"
            }
        }
    
    # Process each chord
    results = {
        'tonality': tonality,
        'is_minor': is_minor,
        'total_chords': len(chords),
        'functional_chords': 0,
        'diatonic_chords': 0,
        'borrowed_chords': 0,
        'non_functional_chords': 0,
        'non_functional_list': [],
        'chord_analysis': []
    }
    
    for chord in chords:
        chord_name = chord['chord_name']
        root_name = chord['root_name']
        chord_type = chord['chord_type'] if chord['chord_type'] else ""  # Empty string for major triads
        
        # Handle power chords and sus chords specially
        if chord_type == '5':
            # Power chords can function in place of either major or minor triads
            simplified_type = ""  # Treat as major for functional analysis
        elif chord_type.startswith('sus'):
            # Sus chords typically function like their root's primary function
            simplified_type = ""  # For functional analysis purposes
        else:
            simplified_type = chord_type
        
        # Determine scale degree of the chord's root
        if root_name in scale_degrees:
            scale_degree = scale_degrees[root_name]
            
            # Check if chord is functional
            if scale_degree in functional_roles:
                role = functional_roles[scale_degree]
                
                # Check if chord type is primary or secondary for this scale degree
                if simplified_type in role["primary"]:
                    is_functional = True
                    is_diatonic = True
                    is_borrowed = False
                    results['diatonic_chords'] += 1
                    results['functional_chords'] += 1
                    roman = role["roman"](chord_type)
                    function = f"{role['role']} ({roman})"
                
                elif simplified_type in role["secondary"]:
                    is_functional = True
                    is_diatonic = False
                    is_borrowed = True
                    results['borrowed_chords'] += 1
                    results['functional_chords'] += 1
                    roman = role["roman"](chord_type)
                    function = f"{role['role']} ({roman}) - borrowed"
                
                else:
                    # Chord root is in scale but type doesn't match functional expectations
                    is_functional = False
                    is_diatonic = False
                    is_borrowed = False
                    results['non_functional_chords'] += 1
                    results['non_functional_list'].append(chord_name)
                    function = f"Non-functional (scale degree {scale_degree+1})"
            else:
                # Scale degree not defined in functional roles (shouldn't happen)
                is_functional = False
                is_diatonic = False
                is_borrowed = False
                results['non_functional_chords'] += 1
                results['non_functional_list'].append(chord_name)
                function = "Invalid scale degree"
        else:
            # Chord root not in scale
            is_functional = False
            is_diatonic = False
            is_borrowed = False
            results['non_functional_chords'] += 1
            results['non_functional_list'].append(chord_name)
            function = "Non-diatonic root"
        
        # Store analysis of this chord
        results['chord_analysis'].append({
            'chord_name': chord_name,
            'root_name': root_name,
            'chord_type': chord_type,
            'is_functional': is_functional,
            'is_diatonic': is_diatonic,
            'is_borrowed': is_borrowed,
            'function': function
        })
    
    # Calculate percentages
    if results['total_chords'] > 0:
        results['functional_percentage'] = (results['functional_chords'] / results['total_chords']) * 100
        results['diatonic_percentage'] = (results['diatonic_chords'] / results['total_chords']) * 100
        results['borrowed_percentage'] = (results['borrowed_chords'] / results['total_chords']) * 100
    else:
        results['functional_percentage'] = 0
        results['diatonic_percentage'] = 0
        results['borrowed_percentage'] = 0
    
    # Final determination
    results['is_tonality_correct'] = results['functional_percentage'] >= 80
    
    return results