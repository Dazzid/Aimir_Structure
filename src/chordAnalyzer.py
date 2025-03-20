#Analyse chords in a MIDI file
import numpy as np
import matplotlib.pyplot as plt
import music21
from music21 import converter, chord, note, stream, pitch
from collections import Counter
import os

#---------------------------------------------------------------------------
CHORD_COLORS = {
    # Basic triads
    "": "#fcc203",       # Major (blue)
    "m": "#03b5fc",      # Minor (mint green)
    "dim": "#ff33eb",    # Diminished (pink)
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
    "sus4": "#b8b8b8",   # Suspended (light gray)
    "5": "#b8b8b8",      # Power chord (gray)
    
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
    }
}
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
def classify_chord_by_weights(normalized_weights, min_threshold = 0.1, root_pc=None):
    """
    Classify chord based on normalized interval weights with improved altered dominant detection
    
    Args:
        normalized_weights: Dictionary mapping intervals to their normalized weights
        root_pc: Pitch class of the root (optional, for additional context)
        
    Returns:
        Chord type string
    """
    # Key interval thresholds
    third_intervals = {3: 'm', 4: ''}  # minor or major third
    fifth_intervals = {6: 'b5', 7: '', 8: 'b13'}  # diminished, perfect, augmented/b13
    seventh_intervals = {10: '7', 11: 'maj7'}  # minor or major seventh
    ninth_intervals = {1: 'b9', 2: '9'}  # flat or natural ninth
    
    # Find most significant intervals
    max_third = max([(normalized_weights.get(i, 0), i) for i in third_intervals.keys()], 
                   key=lambda x: x[0], default=(0, None))[1]
    
    max_fifth = max([(normalized_weights.get(i, 0), i) for i in fifth_intervals.keys()], 
                   key=lambda x: x[0], default=(0, None))[1]
    
    # weighted_sevenths = [(normalized_weights.get(i, 0) * 1.7 if i in (10, 11) else normalized_weights.get(i, 0), i)
    #                  for i in seventh_intervals.keys()]
    # max_seventh = max(weighted_sevenths, key=lambda x: x[0], default=(0, None))[1]
    
    max_seventh = max([(normalized_weights.get(i, 0), i) for i in seventh_intervals.keys()],
                     key=lambda x: x[0], default=(0, None))[1]
    
    max_ninth = max([(normalized_weights.get(i, 0), i) for i in ninth_intervals.keys()],
                   key=lambda x: x[0], default=(0, None))[1] if ninth_intervals else None
    
    # Check for dominant context (presence of b7)
    is_dominant = max_seventh == 10 and normalized_weights.get(10, 0) >= min_threshold
    
    # Check for maj7 context
    is_maj7 = max_seventh == 11 and normalized_weights.get(11, 0) >= min_threshold
    
    # Check for augmented fifth/#5/b13
    has_aug_fifth = max_fifth == 8 and normalized_weights.get(8, 0) >= min_threshold
    
    # Check for suspended context (presence of 4th/11th without 3rd)
    has_fourth = normalized_weights.get(5, 0) >= min_threshold
    no_third = (normalized_weights.get(3, 0) < 0.01 and normalized_weights.get(4, 0) < 0.01)
    is_sus = has_fourth and no_third
    
    # Check for altered dominant features
    has_b9 = max_ninth == 1 and normalized_weights.get(1, 0) >= min_threshold
    has_b5 = max_fifth == 6 and normalized_weights.get(6, 0) >= min_threshold
    
    # Detect altered dominant (a dominant 7th with tensions)
    is_altered_dominant = is_dominant and (has_b9 or has_aug_fifth or has_b5)
    
    # CRITICAL FIX: If we have both dominant 7th AND augmented 5th
    if is_dominant and has_aug_fifth:
        # Check for b9 for fully altered dominant
        if has_b9:
            return "7alt"  # Dominant with b9 and #5/b13
        else:
            return "7b13"  # Dominant with #5/b13
    
    # Special case for maj7 with augmented fifth
    if is_maj7 and has_aug_fifth:
        return "maj7"  # Major 7 with b13
    
    # # Special case for pure augmented triad (only root, major 3rd, #5)
    # if (max_third == 4 and has_aug_fifth and normalized_weights.get(4, 0) > min_threshold):
    #     # Only classify as augmented if it's a pure triad with no 7th
    #     no_sevenths = (normalized_weights.get(10, 0) < min_threshold and normalized_weights.get(11, 0) < min_threshold)
    #     only_core_tones = sum(1 for i, w in normalized_weights.items() 
    #                           if i not in [0, 4, 8] and w >= min_threshold) == 0
        
    #     if no_sevenths and only_core_tones:
    #         return "aug" 
    
    # Special case for diminished chords
    if max_third == 3 and max_fifth == 6 and normalized_weights.get(3, 0) > min_threshold and normalized_weights.get(6, 0) > min_threshold:
        if normalized_weights.get(9, 0) > min_threshold:
            return "dim7"
        elif normalized_weights.get(10, 0) > min_threshold:
            return "m7b5"
        return "dim"
    
    # Handle altered dominant chords
    if is_altered_dominant:
        # Base type: dominant 7th
        chord_type = "7"
        
        # Add specific alterations
        if has_b9 and has_aug_fifth:
            return chord_type + "alt"  # Multiple alterations = 7alt
        elif has_b9:
            return chord_type + "b9"   # Just b9
        elif has_aug_fifth:
            return chord_type + "b13"  # Just b13
        elif has_b5:
            return chord_type + "#11"  # Jazz convention for b5 in dominant
            
        return chord_type  # Basic dominant if no specific alterations identified
    
    # Handle suspended chords
    if is_sus:
        if is_dominant:  # with dominant 7th
            return "7sus4"
        else:
            return "sus4"
    
    # Construct chord type for other cases
    chord_type = ""
    
    # Add third quality if not suspended
    if not is_sus and max_third and normalized_weights.get(max_third, 0) > min_threshold:
        chord_type = third_intervals.get(max_third, '')
    
    # Add seventh if significant
    if max_seventh and normalized_weights.get(max_seventh, 0) > min_threshold:
        chord_type += seventh_intervals.get(max_seventh, '')
    
    # Handle alterations to fifth in non-dominant contexts
    if max_fifth == 6 and normalized_weights.get(6, 0) > min_threshold and not is_dominant:
        if chord_type.startswith("m") and "7" in chord_type:
            return "m7b5"  # Minor 7 flat 5
        elif not chord_type.startswith("m"):
            chord_type += "b5"  # Other contexts    
    
    # Add ninth if significant and not already handled
    if max_ninth and normalized_weights.get(max_ninth, 0) > min_threshold and not is_altered_dominant:
        chord_type += ninth_intervals.get(max_ninth, '')
    
    # Handle power chords
    if chord_type == "" and max_fifth == 7 and normalized_weights.get(7, 0) > min_threshold:
        return "5"  # Power chord
        
    return chord_type

#---------------------------------------------------------------------------
def identify_unique_chord(time_window_results):
    """
    Refine chord identification using weighted interval analysis.
    
    Args:
        time_window_results (list): Results from time window analysis.
        
    Returns:
        list: Refined chord identification results with a 'confidence' value.
    """
    refined_results = []
    
    for window in time_window_results:
        # Extract key information
        intervals = window.get('intervals', [])
        interval_durations = window.get('interval_durations', {}).copy()
        
        # Ensure the root (interval 0) is always present with a nonzero duration.
        # If not present, try to compute it using the window's start and end times.
        if 0 not in interval_durations or interval_durations[0] == 0:
            if 'start' in window and 'end' in window:
                interval_durations[0] = window['end'] - window['start']
            else:
                interval_durations[0] = 1e-6  # Minimal fallback value
        
        # Calculate normalized weights for intervals.
        total_duration = sum(interval_durations.values())
        if total_duration == 0:
            normalized_weights = {i: 0 for i in interval_durations}
        else:
            normalized_weights = {interval: duration / total_duration 
                                  for interval, duration in interval_durations.items()}
        
        # Get the unique chord type using the improved method
        chord_type = classify_chord_by_weights(normalized_weights)
        
        # Validate the chord type against music theory
        chord_type = validate_chord_type(chord_type)
        
        # Determine required intervals; if chord_type is not found, fallback to [0] (root only)
        if chord_type in CHORD_TYPES:
            required_intervals = CHORD_TYPES[chord_type]["required"]
        else:
            required_intervals = [0]
        
        # Debug: Uncomment the line below to inspect intermediate values.
        # print(f"Chord Type: {chord_type}, Required: {required_intervals}, Normalized Weights: {normalized_weights}")
        
        # Compute confidence as the sum of normalized weights for the required intervals.
        confidence = sum(normalized_weights.get(i, 0) for i in required_intervals)
        
        # Update the window result with refined chord type and confidence
        updated_window = window.copy()
        updated_window['refined_chord_type'] = chord_type
        updated_window['normalized_weights'] = normalized_weights
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
                        
                        # Other intervals get sum of harmony note durations
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
                    
                    # Store chord with refined type
                    all_chords.append({
                        'bar': bar_idx + 1,
                        'beat_start': window_start,
                        'beat_end': window_end,
                        'root': bass_note['name'],
                        'chord_type': refined_chord_type,
                        'chord_name': chord_name,
                        'intervals': significant_intervals,
                        'source': 'bass'  # Mark as derived from bass notes
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

#---------------------------------------------------------------------------------------
# Cell 5: Function to extract all chords automatically
def extract_all_chords_automatically(notes, beats_per_bar=4, min_chord_duration=0.5, bars_per_row=16, verbose=False, plot=True, analyze_harmony_only=True):
    """
    Extract ALL chords from the entire piece automatically using the superior chord identification from identify_unique_chord
    With enhanced capability to analyze sections without bass notes
    
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
                        
                        # Other intervals get sum of harmony note durations
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
                    
                    # Use the identify_unique_chord function directly to get refined chord type
                    refined_results = identify_unique_chord([window_result])
                    refined_chord_type = refined_results[0]['refined_chord_type']
                    
                    # Also validate to prevent invalid chord names
                    refined_chord_type = validate_chord_type(refined_chord_type)
                    
                    chord_name = f"{fix_note_name(bass_note['name'])}{refined_chord_type}"
                    
                    # Calculate significant intervals for reference
                    significant_intervals = [i for i, w in normalized_weights.items() 
                                            if w >= 0.15 or i == 0]
                    
                    # Store chord with refined type
                    all_chords.append({
                        'bar': bar_idx + 1,
                        'beat_start': window_start,
                        'beat_end': window_end,
                        'root': bass_note['name'],
                        'chord_type': refined_chord_type,
                        'chord_name': chord_name,
                        'intervals': significant_intervals,
                        'source': 'bass'  # Mark as derived from bass notes
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
def plot_chord_timeline_multirow(chords, total_bars, beats_per_bar=4, bars_per_row=16):
    """
    Plot a chord timeline with multiple rows for better readability in publications
    
    Args:
        chords: List of chord dictionaries from analysis
        total_bars: Total number of bars in the piece
        beats_per_bar: Number of beats per bar
        bars_per_row: Number of bars to display in each row
    """
    # Calculate number of rows needed
    num_rows = (total_bars + bars_per_row - 1) // bars_per_row
    
    # Create figure with appropriate dimensions
    fig, axes = plt.subplots(num_rows, 1, figsize=(15, 2.5*num_rows + 1), sharex=False)
    
    # Ensure axes is always a list even with single row
    if num_rows == 1:
        axes = [axes]
    
    # Process each row
    for row in range(num_rows):
        ax = axes[row]
        
        # Calculate bar range for this row
        start_bar = row * bars_per_row
        end_bar = min((row + 1) * bars_per_row, total_bars)
        
        # Calculate beat range
        start_beat = start_bar * beats_per_bar
        end_beat = end_bar * beats_per_bar
        
        # Find chords that overlap with this row's time window
        row_chords = [c for c in chords 
                     if c['beat_end'] > start_beat and c['beat_start'] < end_beat]
        
        # Debug info
        # print(f"Row {row+1}: Bars {start_bar+1}-{end_bar}, Beats {start_beat}-{end_beat}, Chords: {len(row_chords)}")
        
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
            
            # Add chord name if there's enough space
            
            # Draw the chord name as before
            ax.text(
                relative_start + (relative_end - relative_start) / 2, 
                0.5, 
                name,
                ha='center',
                va='center',
                fontsize=10,
                # fontweight='bold',
                rotation=90,
                rotation_mode='anchor',
                color=text_color
            )

            # Add the confidence number at the top center of the chord rectangle
            confidence = chord.get('confidence', 0)
            ax.text(
                relative_start + (relative_end - relative_start) / 2, 
                0.8,  # Adjust this value as needed to position above the chord rectangle
                f"{confidence:.1f}",
                ha='center',
                va='bottom',
                fontsize=7,  # Smaller font for the confidence number
                color=text_color
            )
        
        # Draw bar lines and numbers for this row
        for bar in range(start_bar, end_bar + 1):
            # Calculate beat position relative to row start
            bar_pos = (bar - start_bar) * beats_per_bar
            
            # Draw the bar line
            ax.axvline(x=bar_pos, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
            
            # Add bar number (except for the end of row line)
            if bar < end_bar:
                ax.text(bar_pos + 0.1, 0.93, f"{bar+1}", fontsize=8)
        
        # Draw beat lines
        for beat in range(start_bar * beats_per_bar, end_bar * beats_per_bar + 1):
            # Calculate beat position relative to row start
            beat_pos = beat - start_beat
            
            # Draw beat line
            ax.axvline(x=beat_pos, color='gray', linestyle=':', alpha=0.3)
        
        # Configure this row's axes
        ax.set_xlim(0, (end_bar - start_bar) * beats_per_bar)
        ax.set_ylim(0, 1)
        ax.set_title(f"Bars {start_bar+1} to {end_bar}", fontsize=10)
        
        # Remove y-axis ticks and labels
        ax.set_yticks([])
        
        # Add x-axis (beats)
        beat_ticks = [i for i in range(0, (end_bar - start_bar) * beats_per_bar + 1, beats_per_bar)]
        beat_labels = [f"{start_bar * beats_per_bar + i}" for i in beat_ticks]
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
        # Add more legend items as needed
    ]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.03), 
          ncol=4, frameon=False)
    
    # Add overall title
    plt.suptitle(f'Chord Progression Timeline ({total_bars} bars)', fontsize=14, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.92)
    
    # Save as vector graphic for paper
    filePath = "Figures/chord_progression"
    plt.savefig(f"{filePath}.pdf", bbox_inches='tight')
    #plt.savefig(f"{filePath}.svg", bbox_inches='tight')
    #plt.savefig(f"{filePath}.png", dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    print(f"Visualization saved as: chord_progression.pdf")
    
    
#----------------------------------------------------------------------------------
# Function to extract all chords with defined length
def extract_all_chords_with_defined_length(notes, beats_per_bar=4, window_size_beats=2, bars_per_row=16, merge=False, verbose=False, plot=True):
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
                'bar': bar_idx + 1,
                'beat_start': window_start,
                'beat_end': window_end,
                'root': root_name,
                'chord_type': refined_chord_type,
                'chord_name': chord_name,
                'intervals': significant_intervals,
                'source': source_type,
                'confidence': confidence
            })
                
    finally:
        # Restore stdout and plt.show
        sys.stdout = original_stdout
        plt.show = original_plt_show
    
    # Sort chords by start time
    all_chords.sort(key=lambda x: x['beat_start'])
    if merge: all_chords = merge_adjacent_chords(all_chords)
    
    # Plot the multi-row timeline only if requested
    plot_chord_timeline_multirow(all_chords, total_bars, beats_per_bar, bars_per_row)
    
    # Return only a summary string if not verbose
    if not verbose:
        bass_confirmed = sum(1 for c in all_chords if c.get('source') == 'bass-confirmed')
        harmony_derived = sum(1 for c in all_chords if c.get('source') == 'harmony-derived')
        return f"Extracted {len(all_chords)} chords across {total_bars} bars ({bass_confirmed} bass-confirmed, {harmony_derived} harmony-derived)"
    
    # Otherwise return the full chord data
    return all_chords


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
