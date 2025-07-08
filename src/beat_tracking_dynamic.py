import numpy as np
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import pretty_midi
import os
import soundfile as sf
from collections import defaultdict
from chordAnalyzer import identify_unique_chord, validate_chord_type, fix_note_name
from music21 import pitch

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
    "5": "#ffb703",      # Power chord (gray)
    
    # Default
    "default": "#d4d4d4" # Other chord types (light gray)
}


class Beat:
    def __init__(self, index, time, duration):
        self.index = index          # Integer index of this beat
        self.time = time            # Absolute time position in seconds
        self.duration = duration    # Duration to next beat in seconds
        self.notes = []             # List of MIDI notes that start on this beat
        self.bass_note = None       # The bass note (lowest note or from bass instrument)
        self.chord = None           # Chord information for this beat
        
class BeatGrid:
    def __init__(self, beats):
        self.beats = beats          # List of Beat objects
        self.beat_map = {beat.index: beat for beat in beats}  # Map of index to Beat objects
        
    def get_beat_by_index(self, index):
        return self.beat_map.get(index)
        
    def get_beat_by_time(self, time):
        # Find the beat closest to the given time
        closest_beat = min(self.beats, key=lambda beat: abs(beat.time - time))
        return closest_beat
    
    def assign_notes_from_beat_blocks(self, midi_beat_blocks):
        """Assign notes from midi_beat_blocks to beat objects"""
        for i, midi_block in enumerate(midi_beat_blocks):
            if i >= len(self.beats):
                break
                
            beat = self.get_beat_by_index(i)
            for instrument in midi_block.instruments:
                is_bass = 'bass' in instrument.name.lower() if instrument.name else False
                
                for note in instrument.notes:
                    note_info = {
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'pitch_class': note.pitch % 12,
                        'absolute_pitch': note.pitch,  # Keep absolute pitch
                        'octave': note.pitch // 12 - 1,  # Store octave (C4 = MIDI 60 = octave 4)
                        'duration': note.end - note.start,
                        'is_bass': is_bass,
                        'instrument': instrument.name
                    }
                    beat.notes.append(note_info)
                    
                    # Check if this is a bass note
                    if is_bass and (beat.bass_note is None or note.pitch < beat.bass_note['pitch']):
                        beat.bass_note = note_info
            
            # If no explicit bass note was found, use the lowest note
            if not beat.bass_note and beat.notes:
                beat.bass_note = min(beat.notes, key=lambda n: n['pitch'])
        
    #----------------------------------------------------------------------------
    def analyze_chords(self):
        """
        Analyze chords based on actual MIDI notes, using the lowest note as the root
        """
        for beat in self.beats:
            if not beat.notes or len(beat.notes) == 0:
                continue
                
            # Sort all notes by actual MIDI pitch to find the bass/lowest note
            sorted_notes = sorted(beat.notes, key=lambda x: x['pitch'])
            
            # Set the lowest note as the bass
            beat.bass_note = sorted_notes[0]
            
            # Get root information
            root_pitch = beat.bass_note['pitch']  # Actual MIDI note number
            root_pc = root_pitch % 12  # Pitch class (0-11)
             
            root_name = fix_note_name(pitch.Pitch(root_pitch).name)
            
            # Create array of absolute pitches
            midi_pitches = np.array([note['pitch'] for note in sorted_notes])
            
            # Calculate both absolute intervals (in semitones) and pitch class intervals
            absolute_intervals = midi_pitches - root_pitch  # Semitone distances from root
            pc_intervals = [(pitch % 12 - root_pc) % 12 for pitch in midi_pitches]  # Pitch class intervals
            
            # Create a more accurate representation for chord analysis
            # Include both absolute MIDI pitches and intervals
            interval_durations = {}
            for i, note in enumerate(sorted_notes):
                interval = pc_intervals[i]
                if interval not in interval_durations:
                    interval_durations[interval] = 0
                interval_durations[interval] += note.get('duration', beat.duration)
            
            # Ensure root/bass is always present
            if 0 not in interval_durations:
                interval_durations[0] = beat.duration
            
            # Create window result with all needed information
            window_result = {
                'intervals': pc_intervals,  # All intervals in pitch class space
                'interval_durations': interval_durations,  # Duration of each interval
                'root_name': root_name,  # Name of the root note
                'root_pitch': root_pitch,  # MIDI pitch of the root
                'root_pc': root_pc,  # Pitch class of the root
                'absolute_pitches': midi_pitches.tolist(),  # All MIDI pitches in the chord
                'absolute_intervals': absolute_intervals.tolist(),  # Semitone intervals from root
                'lowest_pitch': root_pitch  # Explicitly include lowest pitch
            }
            
            # Use chord analyzer to determine chord quality
            refined_results = identify_unique_chord([window_result])
            chord_type = refined_results[0]['refined_chord_type']
            
            # Store complete chord information
            beat.chord = {
                'beat_idx': beat.index,
                'beat_start': beat.time,
                'beat_end': beat.time + beat.duration,
                'root_midi': root_pc,
                'root_pitch': root_pitch,  # Keep absolute MIDI pitch
                'root_name': root_name,
                'chord_type': chord_type,
                'chord_name': f"{root_name}{chord_type}" if chord_type != 'N.C.' else 'N.C.',
                'notes': midi_pitches.tolist(),  # Actual MIDI pitches
                'absolute_intervals': absolute_intervals.tolist(),  # Actual intervals in semitones
                'pitch_classes': set(pitch % 12 for pitch in midi_pitches),  # Set of pitch classes
                'intervals': pc_intervals,  # Pitch class intervals
                'voicing': midi_pitches.tolist()  # Complete voicing information
            }
        
        # Fill in empty beats with previous chord
        prev_chord = None
        for beat in self.beats:
            if not beat.chord and prev_chord:
                beat.chord = prev_chord.copy()
                beat.chord['beat_idx'] = beat.index
                beat.chord['beat_start'] = beat.time
                beat.chord['beat_end'] = beat.time + beat.duration
            
            if beat.chord:
                prev_chord = beat.chord
    
    #----------------------------------------------------------------------------
    def merge_consecutive_identical_chords(self):
        """Merge beats with identical bass notes and analyze the resulting bag of notes"""
        merged_chords = []
        
        i = 0
        while i < len(self.beats):
            current_beat = self.beats[i]
            
            # Skip beats without chords or bass
            if not current_beat.chord or not current_beat.bass_note:
                i += 1
                continue
            
            # Get current bass note
            current_bass_pc = current_beat.bass_note['pitch_class']
            current_bass_name = current_beat.chord['root_name']  # Use the chord root name as it's derived from bass
            
            # Find all consecutive beats with the same bass note
            j = i + 1
            same_bass_beats = [current_beat]
            
            while j < len(self.beats) and self.beats[j].bass_note and \
                self.beats[j].bass_note['pitch_class'] == current_bass_pc:
                same_bass_beats.append(self.beats[j])
                j += 1
                
            # Create a bag of notes from all beats with the same bass
            if same_bass_beats:
                all_notes = []
                total_duration = 0
                for beat in same_bass_beats:
                    all_notes.extend(beat.notes)
                    total_duration += beat.duration
                    
                # delete repeated notes 
                all_notes = list({note['pitch']: note for note in all_notes}.values())
                
                # Sort notes by pitch
                all_notes.sort(key=lambda x: x['pitch'])
                
                # Create a chord using the bag of notes
                chord_info = self.analyze_chord_from_note_bag(all_notes, current_bass_pc, current_bass_name)
                
                # Set correct beat range
                chord_info['beat_idx'] = same_bass_beats[0].index
                chord_info['beat_idx_end'] = same_bass_beats[-1].index
                chord_info['beat_start'] = same_bass_beats[0].time
                chord_info['beat_end'] = same_bass_beats[-1].time + same_bass_beats[-1].duration
                chord_info['duration'] = total_duration
                
                merged_chords.append(chord_info)
                i = j  # Skip to the end of this group
            else:
                # No group found, just add the current chord
                if current_beat.chord:
                    merged_chords.append(current_beat.chord)
                i += 1
        
        return merged_chords
    
#----------------------------------------------------------------------------
    def analyze_chord_from_note_bag(self, harmony_notes, bass_pc, bass_name):
        """
        Improved chord analysis using the sophisticated identify_unique_chord function
        """

        
        # Combine bass and harmony notes for analysis
        all_notes = harmony_notes.copy()
        
        # Add the bass note to the analysis (create a synthetic bass note)
        bass_note = {
            'pitch': bass_pc + 60,  # Put in middle octave for consistency
            'pitch_class': bass_pc,
            'duration': 1.0  # Default duration
        }
        all_notes.insert(0, bass_note)  # Put bass note first (lowest)
        
        # Sort all notes by pitch
        sorted_notes = sorted(all_notes, key=lambda x: x['pitch'])
        
        # Create array of absolute pitches
        midi_pitches = np.array([note['pitch'] for note in sorted_notes])
        
        # Calculate pitch class intervals from bass as root
        pc_intervals = [(p % 12 - bass_pc) % 12 for p in midi_pitches]
        
        # Create interval durations (use beat duration if available, otherwise default)
        beat_duration = 1.0
        if beats is not None and beat_idx < len(beats) - 1:
            beat_duration = beats[beat_idx+1] - beats[beat_idx] if beat_idx+1 < len(beats) else 0.5
        
        interval_durations = {}
        for i, note in enumerate(sorted_notes):
            interval = pc_intervals[i]
            if interval not in interval_durations:
                interval_durations[interval] = 0
            interval_durations[interval] += note.get('duration', beat_duration)
        
        # Ensure root is always present
        if 0 not in interval_durations:
            interval_durations[0] = beat_duration
        
        # Create window result for chord analyzer
        window_result = {
            'intervals': pc_intervals,
            'interval_durations': interval_durations,
            'root_name': bass_name,
            'root_pitch': bass_note['pitch'],
            'root_pc': bass_pc,
            'absolute_pitches': midi_pitches.tolist(),
            'lowest_pitch': bass_note['pitch']
        }
        
        # Use the sophisticated chord analyzer
        refined_results = identify_unique_chord([window_result])
        chord_type = refined_results[0]['refined_chord_type']
        
        return {
            'root_midi': bass_pc,
            'root_name': bass_name,
            'chord_type': chord_type,
            'chord_name': f"{bass_name}{chord_type}",
            'intervals': list(set(pc_intervals)),
            'pitch_classes': set(p % 12 for p in midi_pitches),
            'notes': midi_pitches,
        }

#----------------------------------------------------------------------------
    def visualize_chords(self, merged_chords, beats_per_row=64):
        """Create visualization of chord progression"""
        # Get merged chords for cleaner visualization
        # merged_chords = self.merge_consecutive_identical_chords()
        
        if not merged_chords:
            print("No chords to visualize")
            return None
            
        # Calculate rows needed
        total_beats = len(self.beats)
        num_rows = int(np.ceil(total_beats / beats_per_row))
        
        # Create figure
        fig, axes = plt.subplots(num_rows, 1, figsize=(18, 2.5 * num_rows + 1), sharex=False)
        if num_rows == 1:
            axes = [axes]
            
        # Draw each row
        for row in range(num_rows):
            ax = axes[row]
            start_beat_idx = row * beats_per_row
            end_beat_idx = min((row + 1) * beats_per_row, total_beats)
            
            # Draw beat lines first
            for i in range(start_beat_idx, end_beat_idx + 1):
                ax.axvline(
                    x=i - start_beat_idx,  # Relative position in this row
                    color='gray',
                    linestyle='--',
                    alpha=0.8,
                    linewidth=1.0,
                    zorder=2
                )
                
            # Draw chord blocks
            for chord in merged_chords:
                # Skip chords outside this row
                if chord['beat_idx'] >= end_beat_idx or \
                (chord.get('beat_idx_end', chord['beat_idx']) < start_beat_idx):
                    continue
                    
                # Adjust chord boundaries to row limits
                start_x = max(chord['beat_idx'], start_beat_idx) - start_beat_idx
                end_x = min(chord.get('beat_idx_end', chord['beat_idx']), end_beat_idx-1) - start_beat_idx + 1
                
                # Select color based on chord type
                chord_type = chord['chord_type']
                if chord_type in CHORD_COLORS:
                    color = CHORD_COLORS[chord_type]
                elif chord_type.startswith('m7'):
                    color = CHORD_COLORS['m7']
                elif chord_type.startswith('m'):
                    color = CHORD_COLORS['m']
                elif chord_type.startswith(''):
                    color = CHORD_COLORS['maj7']
                elif chord_type.endswith('7'):
                    color = CHORD_COLORS['7']
                else:
                    color = CHORD_COLORS['default']
                
                text_color = 'black'
                
                # Draw chord rectangle
                ax.add_patch(plt.Rectangle(
                    (start_x, 0.1),
                    end_x - start_x,
                    0.8,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.85,
                    zorder=5
                ))
                
                # Add chord name - use the actual chord_name field from the data
                ax.text(
                    start_x + (end_x - start_x) / 2,
                    0.5,
                    chord['chord_name'],  # This is the key fix - use the exact chord name
                    ha='center',
                    va='center',
                    fontsize=11,
                    fontweight='bold',
                    color=text_color,
                    zorder=6
                )
                
            # Configure axis
            ax.set_xlim(0, end_beat_idx - start_beat_idx)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xticks(np.arange(0, end_beat_idx - start_beat_idx + 1, 4))
            ax.set_xticklabels([str(start_beat_idx + i) for i in range(0, end_beat_idx - start_beat_idx + 1, 4)])
            ax.set_xlabel("Beat", fontsize=8)
            ax.set_title(f"Beats {start_beat_idx + 1} to {end_beat_idx}", fontsize=10)
        
        # Add legend for chord types
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS[''], alpha=0.8, edgecolor='black', label='Major'),
            plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS['m'], alpha=0.8, edgecolor='black', label='Minor'),
            plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS['7'], alpha=0.8, edgecolor='black', label='Dominant 7th'),
            plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS['maj7'], alpha=0.8, edgecolor='black', label='Major 7th'),
            plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS['m7'], alpha=0.8, edgecolor='black', label='Minor 7th'),
            plt.Rectangle((0, 0), 1, 1, facecolor=CHORD_COLORS['dim'], alpha=0.8, edgecolor='black', label='Diminished')
        ]
        fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.03),
                ncol=4, frameon=False)
        
        plt.suptitle(f'Chord Progression Timeline ({beats_per_row} beats per row)', fontsize=14, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.92)
        
        return fig

#--------------------------------------------------------------------------------
def listen_to_midi_beat_blocks(midi_beat_blocks, bpm=120, sr=22050):
    """Simpler version with fixed duration for each beat"""
    
    # Calculate beat duration from BPM
    beat_duration = 60.0 / bpm  # seconds per beat
    
    # Initialize combined audio
    combined_audio = np.array([])
    
    # For each beat
    for i, midi_block in enumerate(midi_beat_blocks):
        try:
            # Create a new MIDI for this beat with the right tempo
            fixed_midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
            
            # Copy instruments
            for instrument in midi_block.instruments:
                new_instrument = pretty_midi.Instrument(
                    program=instrument.program,
                    is_drum=instrument.is_drum,
                    name=instrument.name
                )
                
                # Copy notes with fixed duration
                for note in instrument.notes:
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=0.0,  # Start at beginning of beat
                        end=min(beat_duration, beat_duration * 0.95)  # End slightly before next beat
                    )
                    new_instrument.notes.append(new_note)
                
                if new_instrument.notes:
                    fixed_midi.instruments.append(new_instrument)
            
            # Synthesize
            beat_audio = fixed_midi.fluidsynth(fs=sr)
            
            # Ensure mono
            if len(beat_audio.shape) > 1:
                beat_audio = np.mean(beat_audio, axis=1)
            
            # Add to combined audio with a gap of exactly one beat
            if len(combined_audio) == 0:
                combined_audio = beat_audio
            else:
                # Calculate how many samples one beat should be
                beat_samples = int(beat_duration * sr)
                
                # If beat audio is shorter than a full beat, pad it
                if len(beat_audio) < beat_samples:
                    beat_audio = np.pad(beat_audio, (0, beat_samples - len(beat_audio)))
                else:
                    # If longer, truncate
                    beat_audio = beat_audio[:beat_samples]
                
                # Concatenate
                combined_audio = np.concatenate([combined_audio, beat_audio])
            
            if (i + 1) % 10 == 0:
                print(f"Synthesized {i + 1}/{len(midi_beat_blocks)} beats")
                
        except Exception as e:
            print(f"Error synthesizing beat {i}: {str(e)}")
    
    # Normalize
    if len(combined_audio) > 0:
        combined_audio = combined_audio / np.max(np.abs(combined_audio)) * 0.9
    
    # Play
    print(f"\nPlaying combined audio at {bpm} BPM:")
    audio_obj = Audio(data=combined_audio, rate=sr)
    display(audio_obj)
    
    return {
        'combined_audio': combined_audio,
        'sr': sr,
        'bpm': bpm
    }

def create_beat_grid(results):
    """
    Create a BeatGrid from analysis results
    
    Parameters:
    -----------
    results : dict
        Results from run_time_varying_tempo
        
    Returns:
    --------
    BeatGrid
        Organized beat grid with all musical information
    """
    beats = results['beats']
    midi_beat_blocks = results['midi_beat_blocks']
    
    # listen_to_midi_beat_blocks(midi_beat_blocks)
    
    # Create beat objects
    beat_objects = []
    for i, beat_time in enumerate(beats):
        # Calculate duration to next beat
        if i < len(beats) - 1:
            duration = beats[i+1] - beat_time
        else:
            # For the last beat, use a default duration or estimate from previous beats
            if i > 0:
                duration = beats[i] - beats[i-1]  # Use same as previous beat
            else:
                duration = 0.5  # Default fallback
        
        beat_objects.append(Beat(i, beat_time, duration))
    
    # Create beat grid
    beat_grid = BeatGrid(beat_objects)
    
    # Assign notes from MIDI blocks
    beat_grid.assign_notes_from_beat_blocks(midi_beat_blocks)
    
    # Analyze chords
    beat_grid.analyze_chords()
    
    return beat_grid

def static_tempo_beat_tracking(audio_path, audio_display=False):
    y, sr = librosa.load(audio_path)
    
    # Get the original beats and tempo
    original_tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units='time', trim=False)
    
    # Calculate intervals between consecutive beats
    beat_intervals = np.diff(beat_times)
    
    # Calculate the average interval (in seconds)
    avg_beat_interval = np.mean(beat_intervals)
    
    # Convert to BPM
    avg_tempo = 60.0 / avg_beat_interval
    
    # Generate evenly spaced beats at the average tempo
    # Start from the first detected beat
    first_beat = beat_times[0]
    # Calculate how many beats will fit in the duration of the audio
    audio_duration = len(y) / sr
    num_beats_to_generate = int((audio_duration - first_beat) / avg_beat_interval) + 1
    
    # Create evenly spaced beats at the average tempo
    average_beats = first_beat + np.arange(num_beats_to_generate) * avg_beat_interval
    
    # Create the click track
    click_track = librosa.clicks(times=average_beats, sr=sr, click_freq=660, 
                                click_duration=0.25, length=len(y))
    
    print(f"1. Original tempo estimate: {original_tempo:.2f} BPM")
    print(f"Calculated average tempo: {avg_tempo:.2f} BPM")
    mix = y + click_track
    if audio_display: display(Audio(data=mix, rate=sr))
    
    return mix, average_beats, original_tempo
    

def dynamic_tempo_beat_tracking(audio_path, plotme=False, audio_display=False, hop_length=512):
    """
    Track beats with time-varying tempo and compare with static tempo tracking
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file
    plotme : bool
        Whether to plot the tempo analysis
    hop_length : int
        Hop length for STFT (default: 512)
        
    Returns:
    --------
    dict
        Dictionary containing tempo analysis results
    """
    # Load audio file
    y, sr = librosa.load(audio_path)
    audio_duration = len(y) / sr
    
    # Define time array for plotting
    times = np.arange(len(y)) / sr
    
    # 1. Static tempo beat tracking
    mix_static, static_beats, static_original_tempo = static_tempo_beat_tracking(audio_path)
    
    if len(static_beats) > 1:
        static_intervals = np.diff(static_beats)
        static_avg_interval = np.mean(static_intervals)
        static_tempo = 60.0 / static_avg_interval
    else:
        static_tempo = static_original_tempo
    
    click_static = mix_static - y
    
    # 2. Dynamic tempo tracking - using plp for time-varying tempo
    # This is the key change - directly use librosa.beat.beat_track
    dynamic_tempo, dynamic_beats = librosa.beat.beat_track(y=y, sr=sr, units='time', trim=False)
    
    # Calculate tempo stats
    if len(dynamic_beats) > 1:
        dynamic_intervals = np.diff(dynamic_beats)
        median_tempo = 60.0 / np.median(dynamic_intervals)
        min_tempo = 60.0 / np.max(dynamic_intervals)  # Slowest tempo = longest interval
        max_tempo = 60.0 / np.min(dynamic_intervals)  # Fastest tempo = shortest interval
        std_tempo = np.std(60.0 / dynamic_intervals)
    else:
        median_tempo = dynamic_tempo
        min_tempo = dynamic_tempo
        max_tempo = dynamic_tempo
        std_tempo = 0
    
    click_dynamic = librosa.clicks(times=dynamic_beats, sr=sr, click_freq=500, click_duration=0.25, length=len(y))
    
    # 3. Average beats for stable tempo
    if len(dynamic_beats) > 1:
        dynamic_avg_interval = np.mean(dynamic_intervals)
        dynamic_avg_tempo = 60.0 / dynamic_avg_interval
        
        dynamic_first_beat = dynamic_beats[0]
        dynamic_num_beats = int((audio_duration - dynamic_first_beat) / dynamic_avg_interval) + 1
        dynamic_avg_beats = dynamic_first_beat + np.arange(dynamic_num_beats) * dynamic_avg_interval
    else:
        dynamic_avg_tempo = dynamic_tempo
        dynamic_avg_interval = 60.0 / dynamic_tempo
        
        dynamic_first_beat = 0
        dynamic_num_beats = int(audio_duration / dynamic_avg_interval) + 1
        dynamic_avg_beats = np.arange(dynamic_num_beats) * dynamic_avg_interval
    
    click_avg_dynamic = librosa.clicks(times=dynamic_avg_beats, sr=sr, click_freq=1000, click_duration=0.25, length=len(y))
    
    # Visualization (if requested)
    if plotme:
        # Create visualization code (unchanged)
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        
        # For dynamic tempo plot, use the calculated BPM from intervals
        if len(dynamic_beats) > 1:
            beat_times = dynamic_beats[:-1]  # Starting time of each interval
            beat_bpms = 60.0 / dynamic_intervals  # BPM of each interval
            
            axs[0].scatter(beat_times, beat_bpms, label='Dynamic tempo', color='blue', alpha=0.7, s=10)
            axs[0].plot(beat_times, beat_bpms, color='blue', alpha=0.4)
        
        axs[0].axhline(static_tempo, label=f'Static tempo: {static_tempo:.1f} BPM', color='red', linestyle='--')
        axs[0].axhline(dynamic_avg_tempo, label=f'Avg dynamic tempo: {dynamic_avg_tempo:.1f} BPM', color='green', linestyle='-.')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Tempo (BPM)')
        axs[0].set_title(f'Tempo Analysis\nRange: {min_tempo:.1f}-{max_tempo:.1f} BPM, Std: {std_tempo:.1f}')
        axs[0].legend()
        axs[0].grid(alpha=0.3)
        
        # Beat comparison plot
        axs[1].plot(times, y, color='gray', alpha=0.5, label='Waveform')
        axs[1].vlines(static_beats, 0.6, 1.0, label='Static beats', color='red', alpha=0.7)
        axs[1].vlines(dynamic_beats, 0.1, 0.5, label='Dynamic beats', color='blue', alpha=0.7)
        axs[1].vlines(dynamic_avg_beats, -0.5, -0.1, label='Avg dynamic beats', color='green', alpha=0.7)
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Amplitude')
        axs[1].set_title('Beat Tracking Comparison')
        axs[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    # Audio outputs
    mix_dynamic = y + click_dynamic
    print(f"\n2. Dynamic tempo beats ({dynamic_tempo:.2f} BPM):")
    if audio_display: display(Audio(data=mix_dynamic, rate=sr))
    
    mix_avg_dynamic = y + click_avg_dynamic
    print(f"\n3. Stable beats based on dynamic average ({dynamic_avg_tempo:.2f} BPM):")
    if audio_display: display(Audio(data=mix_avg_dynamic, rate=sr))
    
    # Return results
    return {
        "audio": y,
        "sr": sr,
        "static_tempo": static_tempo,
        "static_beats": static_beats,
        "median_dynamic_tempo": median_tempo,
        "dynamic_tempo": dynamic_tempo,
        "dynamic_beats": dynamic_beats,
        "dynamic_avg_tempo": dynamic_avg_tempo,
        "dynamic_avg_beats": dynamic_avg_beats,
        "std_tempo": std_tempo,
        "audio_with_static_clicks": mix_static,
        "audio_with_dynamic_clicks": mix_dynamic,
        "audio_with_avg_dynamic_clicks": mix_avg_dynamic
    }

def create_midi_blocks_per_beat(beat_notes, beats):
    midi_blocks = []
    for i in range(len(beats) - 1):
        midi = pretty_midi.PrettyMIDI()
        inst_dict = {}
        for note_info in beat_notes.get(i, []):
            inst_key = (note_info['program'], note_info['is_drum'])
            if inst_key not in inst_dict:
                inst_dict[inst_key] = pretty_midi.Instrument(
                    program=note_info['program'],
                    is_drum=note_info['is_drum'],
                    name=note_info.get('instrument', "Beat Instrument")
                )
            rel_start = 0.0
            rel_end = min(beats[i+1] - beats[i], note_info['duration'])
            new_note = pretty_midi.Note(
                velocity=note_info['velocity'],
                pitch=note_info['pitch'],
                start=rel_start,
                end=rel_end
            )
            inst_dict[inst_key].notes.append(new_note)
        for inst in inst_dict.values():
            if inst.notes:
                midi.instruments.append(inst)
        midi_blocks.append(midi)
    return midi_blocks

# --------------------------------------------------------------------------------
# MIDI Alignment Functions (Added to work with dynamic beat tracking)
# --------------------------------------------------------------------------------

def create_click_track(times, freqs, sr, click_duration=0.1, length=None):
    """
    Create a click track with multiple frequencies.
    This is a custom implementation to work around librosa.clicks API changes.
    
    Parameters:
    -----------
    times : list of float
        Click times in seconds
    freqs : list of float
        Click frequencies (one per time)
    sr : int
        Sample rate
    click_duration : float
        Duration of each click
    length : int or None
        Length of the output array
        
    Returns:
    --------
    click_track : np.ndarray
        Audio with clicks at specified times and frequencies
    """
    # Initialize empty click track
    clicks = np.zeros(length)
    
    # First, determine which version of librosa.clicks we have by checking available parameters
    import inspect
    click_params = inspect.signature(librosa.clicks).parameters
    has_click_freq = 'click_freq' in click_params
    has_frequencies = 'frequencies' in click_params
    
    # Add individual clicks
    for time, freq in zip(times, freqs):
        # Use appropriate parameters based on the librosa version
        if has_click_freq:
            # Newer API with click_freq
            click = librosa.clicks(times=[time], click_freq=freq, sr=sr,
                                click_duration=click_duration, length=length)
        elif has_frequencies:
            # Older API with frequencies
            click = librosa.clicks(times=[time], frequencies=[freq], sr=sr, 
                                click_duration=click_duration, length=length)
        else:
            # If neither parameter works, use a simple sine wave as fallback
            t = np.arange(int(sr * click_duration)) / sr
            click = np.sin(2 * np.pi * freq * t) * 0.2 * np.exp(-t * 10)
            
            # Place the click at the correct time
            click_full = np.zeros(length)
            start_idx = int(time * sr)
            end_idx = min(start_idx + len(click), length)
            if end_idx > start_idx:
                click_full[start_idx:end_idx] = click[:end_idx-start_idx]
            click = click_full
        
        # Add to output
        clicks += click
        
    return clicks

def plot_midi_piano_roll(midi_data, ax, start_time, end_time, title='MIDI Piano Roll'):
    """Helper function to plot MIDI piano roll"""
    # Create different colors for each instrument
    instruments = midi_data.instruments
    instrument_colors = plt.cm.tab10(np.linspace(0, 1, len(instruments)))
    
    # Get all notes within time range
    for i, instrument in enumerate(instruments):
        inst_name = instrument.name if instrument.name else f'Instrument {i}'
        
        # Filter notes in the time window
        visible_notes = [note for note in instrument.notes 
                        if note.end >= start_time and note.start <= end_time]
        
        # Plot each note as a horizontal line
        for note in visible_notes:
            note_start = max(note.start, start_time)
            note_end = min(note.end, end_time)
            
            ax.plot(
                [note_start, note_end], 
                [note.pitch, note.pitch], 
                color=instrument_colors[i], 
                alpha=0.7, 
                linewidth=2 * (note.velocity / 127)
            )
        
        # Add a legend entry if we have notes to show
        if visible_notes:
            ax.plot([], [], color=instrument_colors[i], label=inst_name)
    
    ax.set_title(title)
    ax.set_ylabel('MIDI Note Number')
    ax.set_xlim(start_time, end_time)
    ax.set_ylim(0, 127)
    ax.grid(alpha=0.2)
    ax.legend(loc='upper right')

def plot_bass(results, beats_per_row=16):
    """
    Plot the bass note per beat, before chord analysis.
    This provides visualization of the bass line used for chord root determination.
    
    Parameters:
    -----------
    results : dict
        Results from run_time_varying_tempo
    beats_per_row : int
        Number of beats to display in each row
        
    Returns:
    --------
    fig : matplotlib Figure
        Figure with bass visualization
    """
    midi_beat_blocks = results['midi_beat_blocks']
    beats = results['beats']
    
    if not midi_beat_blocks or len(midi_beat_blocks) == 0:
        print("No MIDI beat blocks found.")
        return None
        
    # Extract bass notes per beat
    bass_info = []
    for beat_idx, midi_block in enumerate(midi_beat_blocks):
        # Find bass notes in this beat
        bass_note = None
        for instrument in midi_block.instruments:
            is_bass = 'bass' in instrument.name.lower() if instrument.name else False
            if is_bass and instrument.notes:
                # For explicitly labeled bass instruments, use the lowest note
                lowest_note = min(instrument.notes, key=lambda note: note.pitch)
                bass_note = {
                    'pitch': lowest_note.pitch, 
                    'pitch_class': lowest_note.pitch % 12,
                    'name': None
                }
                break
        
        # If no explicit bass instrument, use the lowest note from any instrument
        if not bass_note:
            all_notes = []
            for instrument in midi_block.instruments:
                all_notes.extend(instrument.notes)
            if all_notes:
                lowest_note = min(all_notes, key=lambda note: note.pitch)
                bass_note = {
                    'pitch': lowest_note.pitch, 
                    'pitch_class': lowest_note.pitch % 12,
                    'name': None
                }
        
        # Convert pitch to note name
        if bass_note:
            bass_note['name'] = fix_note_name(pitch.Pitch(bass_note['pitch']).name)
        else:
            bass_note = {'name': 'N.C.', 'pitch': None, 'pitch_class': None}
            
        bass_info.append(bass_note)
    
    # Calculate visualization parameters
    total_beats = len(bass_info)
    num_rows = int(np.ceil(total_beats / beats_per_row))
    
    # Create the figure
    fig, axes = plt.subplots(num_rows, 1, figsize=(18, 1.8 * num_rows + 1), sharex=False)
    if num_rows == 1:
        axes = [axes]
        
    # Define basic colors for the bass notes
    note_colors = {
        'C': '#8dd3c7', 'C#': '#ffffb3', 'Db': '#ffffb3',
        'D': '#bebada', 'D#': '#fb8072', 'Eb': '#fb8072',
        'E': '#80b1d3', 'F': '#fdb462',
        'F#': '#b3de69', 'Gb': '#b3de69',
        'G': '#fccde5', 'G#': '#d9d9d9', 'Ab': '#d9d9d9',
        'A': '#bc80bd', 'A#': '#ccebc5', 'Bb': '#ccebc5',
        'B': '#ffed6f', 'N.C.': '#d9d9d9'
    }
    
    # Draw each row
    for row in range(num_rows):
        ax = axes[row]
        start_beat = row * beats_per_row
        end_beat = min((row + 1) * beats_per_row, total_beats)
        
        # Draw each bass note as a colored rectangle
        for beat_idx in range(start_beat, end_beat):
            relative_beat = beat_idx - start_beat
            bass_name = bass_info[beat_idx]['name']
            color = note_colors.get(bass_name, '#d9d9d9')  # Default gray for unknown notes
            
            # Draw rectangle for this bass note
            ax.add_patch(plt.Rectangle(
                (relative_beat, 0.1),
                1.0,  # Width of 1 beat
                0.8,  # Height
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.85
            ))
            
            # Add the note name text
            text_color = 'black'
            ax.text(
                relative_beat + 0.5,
                0.5,
                bass_name,
                ha='center',
                va='center',
                fontsize=12,
                fontweight='bold',
                color=text_color
            )
            
        # Draw beat lines (gray)
        # for beat_offset in range(0, end_beat - start_beat + 1):
        #     ax.axvline(
        #         x=beat_offset,
        #         color='black',
        #         linestyle='--',
        #         alpha=0.8,
        #         linewidth=1.0,
        #         zorder=10
        #     )
            
        # Configure axis
        ax.set_xlim(0, end_beat - start_beat)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks(np.arange(0, end_beat - start_beat + 1, 4))
        ax.set_xticklabels([str(start_beat + i) for i in range(0, end_beat - start_beat + 1, 4)])
        ax.set_xlabel("Beat", fontsize=8)
        ax.set_title(f"Bass Notes: Beats {start_beat + 1} to {end_beat}", fontsize=10)
    
    # Add title
    plt.suptitle(f'Bass Note Timeline ({beats_per_row} beats per row)', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.92)
    plt.show()
    
    return fig

def align_midi_to_beats(midi_data, beats, tempo, audio_duration, fine_adjust_ms=0, verbose=False):
    """
    Align MIDI files to detected beats.
    Adapted from align_midi_to_stable_grid in beat_tracking.py.
    
    Parameters:
    -----------
    midi_data : PrettyMIDI
        MIDI data to align
    beats : array-like
        Beat times in seconds
    tempo : float
        Tempo in BPM
    audio_duration : float
        Duration of audio in seconds
    fine_adjust_ms : float
        Fine adjustment in milliseconds

    Returns:
    --------
    aligned_midi : PrettyMIDI
        Aligned MIDI data
    """
    if verbose: 
        print("\n[DEBUG] --- align_midi_to_beats called ---")
        print(f"[DEBUG] Number of audio beats: {len(beats)} | Number of MIDI beats: {len(midi_data.get_beats())}")
    midi_note_starts = [note.start for inst in midi_data.instruments for note in inst.notes]

    # Get MIDI tempo with robust estimation
    midi_tempo = midi_data.estimate_tempo()
    midi_beats = midi_data.get_beats()

    # If tempo estimation fails, calculate from beats or note patterns
    if midi_tempo == 0 or midi_tempo < 20 or midi_tempo > 300:
        if len(midi_beats) >= 2:
            midi_beat_intervals = np.diff(midi_beats)
            midi_tempo = 60 / np.median(midi_beat_intervals)
            if verbose: print(f"[DEBUG] Calculated MIDI tempo from beat intervals: {midi_tempo:.2f} BPM")
        else:
            note_onsets = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    note_onsets.append(note.start)
            if len(note_onsets) >= 2:
                note_onsets = sorted(note_onsets)
                onset_intervals = np.diff(note_onsets)
                if len(onset_intervals) > 0:
                    hist, bins = np.histogram(onset_intervals, bins=50, range=(0, min(2.0, max(onset_intervals))))
                    if np.max(hist) > 0:
                        most_common_interval = bins[np.argmax(hist)]
                        midi_tempo = 60 / most_common_interval
                        if verbose: print(f"[DEBUG] Estimated MIDI tempo from note patterns: {midi_tempo:.2f} BPM")
                    else:
                        midi_tempo = tempo
                else:
                    midi_tempo = tempo
            else:
                midi_tempo = tempo
    else:
        if verbose: print(f"[DEBUG] MIDI tempo from file metadata: {midi_tempo:.2f} BPM")

    tempo_ratio = tempo / midi_tempo
    if verbose: print(f"[DEBUG] Audio tempo: {tempo:.2f} BPM | MIDI tempo: {midi_tempo:.2f} BPM | Tempo ratio: {tempo_ratio:.4f}")

    fine_adjust_sec = fine_adjust_ms / 1000.0

    # --- RELIABILITY CHECK ---
    min_beats_required = 8
    tempo_ratio_close = abs(tempo_ratio - 1) < 0.1
    midi_beats_ok = len(midi_beats) >= min_beats_required
    audio_beats_ok = len(beats) >= min_beats_required
    tempo_ok = 20 < midi_tempo < 300
    mapping_allowed = midi_beats_ok and audio_beats_ok and tempo_ok and tempo_ratio_close

    if not mapping_allowed:
        if verbose: 
            print(f"[DEBUG] Mapping/scaling NOT allowed. Reason(s):")
            if not midi_beats_ok: print("[DEBUG]   - Not enough MIDI beats")
            if not audio_beats_ok: print("[DEBUG]   - Not enough audio beats")
            if not tempo_ok: print("[DEBUG]   - MIDI tempo out of range")
            if not tempo_ratio_close: print("[DEBUG]   - Tempo ratio not close enough to 1.0")
            print("[DEBUG] Only applying fine offset, no warping or mapping.")
        aligned_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        for instrument in midi_data.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name
            )
            for note in instrument.notes:
                start_time = note.start + fine_adjust_sec
                end_time = note.end + fine_adjust_sec
                start_time = max(0, start_time)
                end_time = max(start_time + 0.01, end_time)
                if start_time < audio_duration:
                    end_time = min(end_time, audio_duration)
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=start_time,
                        end=end_time
                    )
                    new_instrument.notes.append(new_note)
            if len(new_instrument.notes) > 0:
                aligned_midi.instruments.append(new_instrument)
        aligned_note_starts = [note.start for inst in aligned_midi.instruments for note in inst.notes]
        if aligned_note_starts:
            if verbose: print(f"[DEBUG] After alignment: First note {min(aligned_note_starts):.3f} | Last note {max(aligned_note_starts):.3f}")
        return aligned_midi

    # --- ELSE: USE BEAT-TO-BEAT MAPPING/SCALING ---
    if verbose: print("[DEBUG] Using beat-to-beat mapping for direct beat alignment...")
    from scipy.interpolate import interp1d
    max_beats_to_use = min(len(midi_beats), len(beats))
    max_midi_time = midi_beats[-1] if len(midi_beats) > 0 else 0
    
    try:
        mapping_fn = interp1d(
            midi_beats[:max_beats_to_use],
            beats[:max_beats_to_use],
            kind='linear',
            bounds_error=False,
            fill_value=(beats[0] if len(beats) > 0 else 0, 
                       beats[-1] if len(beats) > 0 else audio_duration)
        )
        use_mapping = True
        print(f"[DEBUG] Beat mapping created successfully with {max_beats_to_use} beats")
    except Exception as e:
        print(f"[DEBUG] Beat mapping failed: {str(e)}")
        use_mapping = False

    aligned_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for instrument in midi_data.instruments:
        new_instrument = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name
        )
        for note in instrument.notes:
            if use_mapping and note.start <= max_midi_time:
                start_time = mapping_fn(note.start) + fine_adjust_sec
                if note.end <= max_midi_time:
                    end_time = mapping_fn(note.end) + fine_adjust_sec
                else:
                    duration = note.end - note.start
                    end_time = start_time + duration * tempo_ratio
            else:
                # Use first beat as reference if mapping fails
                start_time = (beats[0] if len(beats) > 0 else 0) + (note.start * tempo_ratio) + fine_adjust_sec
                end_time = (beats[0] if len(beats) > 0 else 0) + (note.end * tempo_ratio) + fine_adjust_sec
            
            start_time = max(0, start_time)
            end_time = max(start_time + 0.01, end_time)
            
            if start_time < audio_duration:
                end_time = min(end_time, audio_duration)
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=start_time,
                    end=end_time
                )
                new_instrument.notes.append(new_note)
        if len(new_instrument.notes) > 0:
            aligned_midi.instruments.append(new_instrument)
    
    aligned_note_starts = [note.start for inst in aligned_midi.instruments for note in inst.notes]
    if aligned_note_starts and verbose:
        print(f"[DEBUG] After mapping: First note {min(aligned_note_starts):.3f} | Last note {max(aligned_note_starts):.3f}")
    
    # Verify alignment quality
    total_notes = sum(len(instrument.notes) for instrument in aligned_midi.instruments)
    notes_on_beats = 0
    beat_tolerance = 0.05  # 50ms tolerance
    for instrument in aligned_midi.instruments:
        for note in instrument.notes:
            min_dist = np.min(np.abs(beats - note.start)) if len(beats) > 0 else float('inf')
            if min_dist < beat_tolerance:
                notes_on_beats += 1
    if total_notes > 0:
        alignment_quality = notes_on_beats / total_notes
        if verbose: print(f"[DEBUG] Alignment quality: {alignment_quality:.2%} of notes aligned with beats")
    
    return aligned_midi

def group_notes_by_beat(aligned_midi, beats):
    """
    Group MIDI notes by the closest beat
    
    Parameters:
    -----------
    aligned_midi : PrettyMIDI
        Aligned MIDI data
    beats : array-like
        Beat times in seconds
        
    Returns:
    --------
    beat_notes : dict
        Dictionary mapping beat indices to lists of notes
    """
    print("\nGrouping MIDI notes by beat...")
    
    beat_notes = {i: [] for i in range(len(beats))}
    
    # Assign each note to the closest beat
    for instrument in aligned_midi.instruments:
        for note in instrument.notes:
            # Find the closest beat to note start
            beat_distances = np.abs(beats - note.start)
            closest_beat_idx = np.argmin(beat_distances)
            
            # Add this note to the appropriate beat group
            beat_notes[closest_beat_idx].append({
                'instrument': instrument.name if instrument.name else "",
                'program': instrument.program,
                'is_drum': instrument.is_drum,
                'pitch': note.pitch,
                'velocity': note.velocity,
                'start': note.start,
                'end': note.end,
                'duration': note.end - note.start
            })
    
    # Print summary statistics
    total_notes = sum(len(notes) for notes in beat_notes.values())
    beats_with_notes = sum(1 for notes in beat_notes.values() if len(notes) > 0)
    
    print(f"Total notes grouped: {total_notes}")
    print(f"Beats with notes: {beats_with_notes}/{len(beats)} ({beats_with_notes/len(beats)*100:.1f}%)")
    
    
    return beat_notes

def create_midi_beat_blocks(beat_notes, beats, audio_duration, sr):
    """
    Create audio blocks with MIDI notes per beat
    
    Parameters:
    -----------
    beat_notes : dict
        Dictionary mapping beat indices to lists of notes
    beats : array-like
        Beat times in seconds
    audio_duration : float
        Duration of audio in seconds
    sr : int
        Sample rate
        
    Returns:
    --------
    beat_block_audio : np.ndarray
        Audio with MIDI notes arranged by beat
    """
    print("\n5. Creating MIDI beat blocks...")
    
    # Handle edge case of no beats
    if len(beats) == 0:
        print("Warning: No beats detected, cannot create beat blocks")
        return np.zeros(int(audio_duration * sr))
    
    # Initialize empty audio
    total_samples = int(audio_duration * sr)
    beat_block_audio = np.zeros(total_samples)
    
    # Create a simple click track to mark beat positions
    beat_clicks = create_click_track(beats, [880] * len(beats), sr, click_duration=0.1, length=total_samples)
    beat_block_audio += beat_clicks * 0.3  # Add clicks at 30% volume
    
    # For each beat, synthesize its notes
    for i, beat_time in enumerate(beats):
        if i >= len(beats) - 1:
            # For the last beat, use the remaining duration
            next_beat_time = audio_duration
        else:
            next_beat_time = beats[i + 1]
        
        beat_duration = next_beat_time - beat_time
        
        # Skip if the beat isn't in the dictionary (shouldn't happen but just in case)
        if i not in beat_notes:
            continue
            
        beat_notes_list = beat_notes[i]
        
        if not beat_notes_list:
            continue  # Skip if no notes at this beat
        
        # Create a temporary MIDI object for this beat
        beat_midi = pretty_midi.PrettyMIDI(initial_tempo=60.0/beat_duration*60.0)  # Set tempo based on beat duration
        
        # Collect notes by instrument
        instruments_dict = {}
        for note_info in beat_notes_list:
            inst_key = (note_info['program'], note_info['is_drum'])
            if inst_key not in instruments_dict:
                instruments_dict[inst_key] = pretty_midi.Instrument(
                    program=note_info['program'],
                    is_drum=note_info['is_drum'],
                    name=note_info.get('instrument', "Beat Instrument")
                )
            
            # Add the note to the instrument
            # Normalize note timing to the beat window
            rel_start = 0.0  # Always start at the beginning of the beat
            rel_end = min(beat_duration, note_info['duration'])  # Cap duration to beat length
            
            new_note = pretty_midi.Note(
                velocity=note_info['velocity'],
                pitch=note_info['pitch'],
                start=rel_start,
                end=rel_end
            )
            instruments_dict[inst_key].notes.append(new_note)
        
        # Add non-empty instruments to the MIDI
        for instrument in instruments_dict.values():
            if instrument.notes:
                beat_midi.instruments.append(instrument)
        
        # Synthesize audio for this beat
        try:
            beat_audio = beat_midi.fluidsynth(fs=sr)
            
            # Ensure beat audio is mono
            if len(beat_audio.shape) > 1:
                beat_audio = np.mean(beat_audio, axis=1)
            
            # Calculate sample positions
            start_sample = int(beat_time * sr)
            beat_samples = len(beat_audio)
            end_sample = min(start_sample + beat_samples, total_samples)
            
            # Add synthesized audio to the output at the correct position
            if end_sample > start_sample:
                # Apply fade-in and fade-out to avoid clicks
                fade_samples = min(int(0.01 * sr), beat_samples // 4)  # 10ms fade or 1/4 of beat, whichever is smaller
                
                # Apply fades if enough samples
                if beat_samples > fade_samples * 2:
                    fade_in = np.linspace(0, 1, fade_samples)
                    fade_out = np.linspace(1, 0, fade_samples)
                    beat_audio[:fade_samples] *= fade_in
                    beat_audio[-fade_samples:] *= fade_out
                
                # Trim beat audio to fit
                samples_to_copy = end_sample - start_sample
                beat_audio_trimmed = beat_audio[:samples_to_copy]
                
                # Add to output
                beat_block_audio[start_sample:end_sample] += beat_audio_trimmed * 0.7  # Reduce volume to avoid clipping
                
        except Exception as e:
            print(f"Error synthesizing beat {i} at {beat_time:.2f}s: {str(e)}")
            # Use simple synthesis as fallback
            for note_info in beat_notes_list:
                freq = librosa.midi_to_hz(note_info['pitch'])
                start_sample = int(beat_time * sr)
                end_sample = min(int(next_beat_time * sr), total_samples)
                
                if end_sample > start_sample:
                    # Simple sine wave synthesis
                    t = np.arange(end_sample - start_sample) / sr
                    amplitude = 0.1 * (note_info['velocity'] / 127.0)
                    sine_wave = amplitude * np.sin(2 * np.pi * freq * t)
                    
                    # Apply simple envelope
                    env_length = min(int(0.01 * sr), len(sine_wave) // 2)
                    if env_length > 0:
                        attack = np.linspace(0, 1, env_length)
                        release = np.linspace(1, 0, env_length)
                        
                        sine_wave[:env_length] *= attack
                        sine_wave[-env_length:] *= release
                    
                    beat_block_audio[start_sample:end_sample] += sine_wave
    
    # Normalize to prevent clipping
    if np.max(np.abs(beat_block_audio)) > 0:
        beat_block_audio = beat_block_audio / np.max(np.abs(beat_block_audio)) * 0.9
    
    print("MIDI beat blocks created successfully.")
    return beat_block_audio

def visualize_midi_alignment_with_beats(y, sr, beats, original_midi, aligned_midi, start_time=0, end_time=20):
    """
    Visualize MIDI alignment with detected beats
    
    Parameters:
    -----------
    y : np.ndarray
        Audio signal
    sr : int
        Sample rate
    beats : array-like
        Beat times in seconds
    original_midi : PrettyMIDI
        Original MIDI data
    aligned_midi : PrettyMIDI
        Aligned MIDI data
    start_time : float
        Start time for visualization in seconds
    end_time : float
        End time for visualization in seconds
    """
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
    
    # Plot audio waveform with beat grid
    ax1 = axes[0]
    sample_start = int(start_time * sr)
    sample_end = int(min(end_time * sr, len(y)))
    times = np.arange(sample_start, sample_end) / sr
    
    ax1.plot(times, y[sample_start:sample_end], color='gray', alpha=0.6)
    
    # Color downbeats differently (assuming 4/4 time signature)
    for i, beat in enumerate(beats):
        if beat >= start_time and beat <= end_time:
            if i % 4 == 0:  # Downbeat
                ax1.axvline(x=beat, color='r', linewidth=1.5, alpha=0.7)
            else:  # Regular beat
                ax1.axvline(x=beat, color='b', linewidth=0.8, alpha=0.5)
    
    ax1.set_title('Audio Waveform with Beat Grid')
    ax1.set_ylim(-1, 1)
    
    # Plot original MIDI
    ax2 = axes[1]
    plot_midi_piano_roll(original_midi, ax2, start_time, end_time, 
                        title='Original MIDI Notes')
    
    # Plot aligned MIDI
    ax3 = axes[2]
    plot_midi_piano_roll(aligned_midi, ax3, start_time, end_time, 
                        title='Aligned MIDI Notes')
    
    # Add beat grid to MIDI plots
    for ax in [ax2, ax3]:
        for i, beat in enumerate(beats):
            if beat >= start_time and beat <= end_time:
                if i % 4 == 0:  # Downbeat
                    ax.axvline(x=beat, color='r', linewidth=1.5, alpha=0.7)
                else:  # Regular beat
                    ax.axvline(x=beat, color='b', linewidth=0.8, alpha=0.5)
    
    # Set common labels
    ax3.set_xlabel('Time (s)')
    fig.tight_layout()
    
    return fig

def find_midi_fine_adjustment(y, sr, aligned_midi, max_offset_ms=30, step_ms=2):
    """
    Find optimal fine adjustment for MIDI timing
    
    Parameters:
    -----------
    y : numpy.ndarray
        Audio time series
    sr : int
        Sample rate
    aligned_midi : PrettyMIDI object
        MIDI that's already aligned to beats but needs fine adjustment
    max_offset_ms : int
        Maximum milliseconds to adjust in either direction
    step_ms : int
        Step size for testing different offsets in milliseconds
    
    Returns:
    --------
    float
        Optimal offset in milliseconds
    """
    print("Finding optimal fine adjustment for MIDI...")
    
    # Extract audio percussive component
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Get audio onsets
    onset_env = librosa.onset.onset_strength(
        y=y_percussive, 
        sr=sr,
        hop_length=512
    )
    
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=512,
        backtrack=True
    )
    
    audio_onsets = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    
    # Get MIDI note onsets
    midi_note_onsets = []
    midi_velocities = []
    for instrument in aligned_midi.instruments:
        for note in instrument.notes:
            midi_note_onsets.append(note.start)
            midi_velocities.append(note.velocity)
    
    # Sort onsets and corresponding velocities
    if len(midi_note_onsets) > 0:
        sorted_indices = np.argsort(midi_note_onsets)
        midi_note_onsets = np.array(midi_note_onsets)[sorted_indices]
        midi_velocities = np.array(midi_velocities)[sorted_indices]
    else:
        midi_note_onsets = np.array([])
        midi_velocities = np.array([])
    
    # Test different offsets
    offsets_ms = np.arange(-max_offset_ms, max_offset_ms + step_ms, step_ms)
    offsets_sec = offsets_ms / 1000.0
    scores = []
    
    for offset in offsets_sec:
        # Shift MIDI onsets
        shifted_onsets = midi_note_onsets + offset
        
        # Count how many MIDI onsets are close to audio onsets
        matches = 0
        tolerance = 0.03  # 30ms tolerance
        
        # Weight by velocity - higher velocity notes are more important for alignment
        velocity_weight_sum = 0
        
        for i, midi_onset in enumerate(shifted_onsets):
            if len(audio_onsets) > 0:
                min_distance = np.min(np.abs(audio_onsets - midi_onset))
                if min_distance < tolerance:
                    # Use velocity as a weight (normalized to 0-1)
                    velocity_weight = midi_velocities[i] / 127.0
                    matches += velocity_weight
                    velocity_weight_sum += velocity_weight
        
        # Score is percentage of matches, weighted by velocity
        if velocity_weight_sum > 0:
            score = matches / velocity_weight_sum
        else:
            score = 0
        scores.append(score)
    
    # Find best offset (apply smoothing if results are noisy)
    if len(scores) > 0:
        from scipy.ndimage import gaussian_filter1d
        smoothed_scores = gaussian_filter1d(scores, sigma=1.0)
        best_idx = np.argmax(smoothed_scores)
        optimal_offset_ms = offsets_ms[best_idx]
    else:
        optimal_offset_ms = 0
    
    print(f"Optimal MIDI fine adjustment: {optimal_offset_ms:.2f} ms")
    
    return optimal_offset_ms

def create_synchronized_audio(y, sr, beats, aligned_midi):
    """Create audio with synchronized beat clicks and MIDI"""
    # Create a click track with different sounds for downbeats vs. regular beats
    click_times = []
    click_freqs = []
    
    for i, beat in enumerate(beats):
        click_times.append(beat)
        if i % 4 == 0:  # Every 4th beat is a downbeat
            click_freqs.append(1760)  # Higher pitch for downbeats
        else:
            click_freqs.append(880)   # Lower pitch for other beats
    
    # Create the click track
    click_track = create_click_track(click_times, click_freqs, sr, click_duration=0.1, length=len(y))
    
    # Synthesize aligned MIDI
    try:
        midi_audio = aligned_midi.fluidsynth(fs=sr)
        
        # Check if successful
        if np.max(np.abs(midi_audio)) < 0.001:
            raise Exception("Silent MIDI synthesis")
    except Exception as e:
        print(f"Error in MIDI synthesis: {str(e)}")
        print("Using simple synthesis as fallback...")
        
        # Create a simple sine wave synthesis
        from scipy import signal
        
        # Initialize empty audio
        midi_audio = np.zeros(len(y))
        
        # Simple synthesis for each note
        for instrument in aligned_midi.instruments:
            for note in instrument.notes:
                freq = librosa.midi_to_hz(note.pitch)
                start_sample = int(note.start * sr)
                end_sample = min(int(note.end * sr), len(midi_audio))
                
                if end_sample > start_sample:
                    # Generate sine wave for this note
                    t = np.arange(end_sample - start_sample) / sr
                    sine_wave = 0.1 * (note.velocity / 127.0) * np.sin(2 * np.pi * freq * t)
                    
                    # Add to the output
                    midi_audio[start_sample:end_sample] += sine_wave
    
    # Ensure MIDI audio is mono channel
    if len(midi_audio.shape) > 1:
        midi_audio = np.mean(midi_audio, axis=1)
    
    # Normalize audio streams
    y_norm = y / np.max(np.abs(y)) * 0.8 if np.max(np.abs(y)) > 0 else y
    
    # Adjust MIDI length to match original audio
    if len(midi_audio) > len(y):
        midi_audio = midi_audio[:len(y)]
    else:
        midi_audio = np.pad(midi_audio, (0, max(0, len(y) - len(midi_audio))))
    
    # Normalize MIDI
    midi_norm = midi_audio / np.max(np.abs(midi_audio)) * 0.8 if np.max(np.abs(midi_audio)) > 0 else midi_audio
    
    # Create outputs with different combinations
    # 1. Original audio with beat clicks
    audio_with_clicks = y_norm + click_track * 0.4
    
    # 2. Aligned MIDI with beat clicks  
    midi_with_clicks = midi_norm * 0.7 + click_track * 0.4
    
    # 3. Full mix: original + MIDI + clicks
    full_mix = y_norm * 0.6 + midi_norm * 0.6 + click_track * 0.3
    
    # Clip to prevent distortion
    audio_with_clicks = np.clip(audio_with_clicks, -0.95, 0.95)
    midi_with_clicks = np.clip(midi_with_clicks, -0.95, 0.95)
    full_mix = np.clip(full_mix, -0.95, 0.95)
    
    return audio_with_clicks, midi_with_clicks, full_mix

#----------------------------------------------------------------------------------------------
def run_time_varying_tempo(song_path, midi_paths, audio_display=False, hop_length=512):
    """
    Main function to run dynamic tempo analysis with MIDI alignment
    
    Parameters:
    -----------
    song_path : str
        Path to the audio file
    midi_paths : dict
        Dictionary of MIDI paths (e.g. {'bass': bass_path, 'harmony': harmony_path})
    output_path : str, optional
        Directory to save outputs (if provided)
    hop_length : int
        Hop length for STFT
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    print(f"Analyzing song with dynamic tempo: {song_path}")
    
    # Check if audio file exists
    if not os.path.exists(song_path):
        print(f"Error: Audio file not found at {song_path}")
        return None
    
    # Run beat tracking with dynamic tempo
    results = dynamic_tempo_beat_tracking(song_path, plotme=False, hop_length=hop_length)
    
    # Step 4: Load and combine MIDI files
    print("\n4. Loading and combining MIDI files...")
    combined_midi = pretty_midi.PrettyMIDI()
    
    # Verify at least one MIDI file exists
    valid_midi_found = False
    
    for midi_name, midi_path in midi_paths.items():
        if not os.path.exists(midi_path):
            print(f"Warning: MIDI file not found at {midi_path}")
            continue
            
        print(f"Loading {midi_name} MIDI: {midi_path}")
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            
            # Check if MIDI has any notes
            note_count = sum(len(instrument.notes) for instrument in midi_data.instruments)
            if note_count == 0:
                print(f"Warning: No notes found in MIDI file {midi_path}")
                continue
                
            # Rename instruments to include source file name for clarity
            for instrument in midi_data.instruments:
                instrument.name = f"{midi_name}"
            
            combined_midi.instruments.extend(midi_data.instruments)
            valid_midi_found = True
        except Exception as e:
            print(f"Error loading MIDI file {midi_path}: {str(e)}")
    
    if not valid_midi_found:
        #stop the process with error message
        raise ValueError("Error: No valid MIDI files could be loaded", midi_path)
        
    print(f"Combined MIDI has {len(combined_midi.instruments)} instruments")
    
    # Extract audio and beats from the results
    y = results['audio']
    sr = results['sr']
    audio_duration = len(y) / sr
    
    # Choose which beat tracking to use - I'll use the average of dynamic beats
    # as it provides stable timing but is based on the dynamic tempo analysis
    beats = results['dynamic_avg_beats']
    tempo = results['dynamic_avg_tempo']
    
    # Check if we have enough beats
    if len(beats) < 4:
        print(f"Warning: Very few beats detected ({len(beats)}). Results may be unreliable.")
    
    print(f"\nUsing dynamic average beats: {len(beats)} beats at {tempo:.2f} BPM")
    
    # Align MIDI to detected beats
    try:
        aligned_midi = align_midi_to_beats(combined_midi, beats, tempo, audio_duration=audio_duration)
        
    except Exception as e:
        print(f"Error in MIDI alignment: {str(e)}")
        # Return partial results
        return {
            'beats': beats,
            'tempo': tempo,
            'audio': results['audio'],
            'sr': sr,
            'error': f"MIDI alignment failed: {str(e)}"
        }
    
    # Find optimal fine adjustment for MIDI
    try:
        midi_offset_ms = find_midi_fine_adjustment(y, sr, aligned_midi, max_offset_ms=50, step_ms=2)
        
        # Apply the fine adjustment to create final MIDI
        final_midi = align_midi_to_beats(combined_midi, beats, tempo, audio_duration=audio_duration, fine_adjust_ms=midi_offset_ms)
    
    except Exception as e:
        print(f"Error in MIDI fine adjustment: {str(e)}")
        # Continue with the original aligned MIDI
        midi_offset_ms = 0
        final_midi = aligned_midi
    
    # Group MIDI notes by beat
    try:
        beat_notes = group_notes_by_beat(final_midi, beats)
        rms_per_beat = compute_per_beat_rms(y, beats, sr)
        # print(rms_per_beat)
        SILENCE_THRESHOLD = 0.02  # Or tune as needed

        for idx, rms in enumerate(rms_per_beat):
            if rms < SILENCE_THRESHOLD:
                beat_notes[idx] = []
    except Exception as e:
         raise ValueError(f"Error grouping notes by beat: {str(e)}", midi_path)
    
    # Create beat-aligned MIDI blocks (PrettyMIDI objects, for chord analysis)
    try:
        midi_beat_blocks = create_midi_blocks_per_beat(beat_notes, beats)
    except Exception as e:
        print(f"Error creating MIDI beat blocks: {str(e)}")
        midi_beat_blocks = []

    # Create beat-aligned audio blocks (for playback)
    try:
        beat_blocks_audio = create_midi_beat_blocks(beat_notes, beats, audio_duration, sr)
    except Exception as e:
        print(f"Error creating beat block audio: {str(e)}")
        beat_blocks_audio = np.zeros_like(y)
    
    # Create synchronized audio
    try:
        audio_with_clicks, midi_with_clicks, full_mix = create_synchronized_audio(
            y, sr, beats, final_midi
        )
    except Exception as e:
        print(f"Error creating synchronized audio: {str(e)}")
        # Create fallback audio
        beat_clicks = create_click_track(beats, [880] * len(beats), sr, click_duration=0.1, length=len(y))
        audio_with_clicks = y + beat_clicks * 0.3
        midi_with_clicks = beat_clicks * 0.3  # Just clicks if MIDI fails
    
    # Create beat blocks + original audio mix
    # Ensure both arrays have the same length before adding
    min_length = min(len(y), len(beat_blocks_audio))

    # If there's a difference, pad the shorter one to match
    if len(y) > min_length:
        # Trim the original audio to match beat blocks audio
        y_to_use = y[:min_length]
    elif len(beat_blocks_audio) > min_length:
        # Pad the beat blocks audio to match original audio length
        beat_blocks_audio = np.pad(beat_blocks_audio, (0, len(y) - len(beat_blocks_audio)))
        min_length = len(y)
        y_to_use = y
    else:
        # Both already same length
        y_to_use = y

    # Normalize and mix
    y_normalized = y_to_use / np.max(np.abs(y_to_use)) * 0.7 if np.max(np.abs(y_to_use)) > 0 else y_to_use
    beat_blocks_with_audio = y_normalized + beat_blocks_audio[:min_length] * 0.8
    beat_blocks_with_audio = np.clip(beat_blocks_with_audio, -0.95, 0.95)
    
    # Step 10: Play audio results
    if audio_display: 
        display(Audio(data=beat_blocks_with_audio, rate=sr))
        display(Audio(data=beat_blocks_audio, rate=sr))
    
    # Return results
    return {
        'sr': sr,
        'audio_duration': audio_duration,
        'beats': beats,
        'tempo': tempo,
        'midi_offset_ms': midi_offset_ms,
        'final_midi': final_midi,
        'beat_notes': beat_notes,
        'midi_beat_blocks': midi_beat_blocks,
        'audio': {
            'original_with_clicks': audio_with_clicks,
            'midi_with_clicks': midi_with_clicks,
            'beat_blocks': beat_blocks_audio,
            'beat_blocks_with_audio': beat_blocks_with_audio,
        }
    }

#-----------------------------------------------------------------------------------------
def generate_chord_progression_summary(chords, beats_per_group=16):
    """
    Generate a text summary of the chord progression organized by beat groups
    
    Parameters:
    -----------
    chords : list
        List of chord dictionaries
    beats_per_group : int
        Number of beats to group together for display purposes
        
    Returns:
    --------
    str
        Text summary of the chord progression
    """
    if not chords:
        return "No chords detected"
    
    # Group chords by beat groups
    chord_groups = {}
    
    for chord in chords:
        # Handle both index-based and time-based chord dictionaries
        if 'beat_idx' in chord:
            # Index-based chord (from BeatGrid)
            start_idx = chord['beat_idx']
            end_idx = chord.get('beat_idx_end', start_idx)
            start_group = start_idx // beats_per_group
            end_group = end_idx // beats_per_group
        else:
            # Time-based chord (from old format)
            # Estimate beat index based on average beat duration
            if len(chords) > 1:
                avg_beat_duration = sum(c['beat_end'] - c['beat_start'] for c in chords) / sum(1 for c in chords if 'beat_end' in c and 'beat_start' in c)
                start_beat = int(chord['beat_start'] / avg_beat_duration)
                end_beat = int(chord['beat_end'] / avg_beat_duration)
                start_group = start_beat // beats_per_group
                end_group = end_beat // beats_per_group
            else:
                # Fallback for single chord
                start_group = 0
                end_group = 0
        
        # Add chord to each group it spans
        for group in range(start_group, end_group + 1):
            if group not in chord_groups:
                chord_groups[group] = []
            chord_groups[group].append(chord)
    
    # Create the summary text
    summary = []
    summary.append("## Chord Progression Summary")
    summary.append("")
    
    # Add a table of beat groups and chords
    summary.append("| Beats | Chords |")
    summary.append("|-------|--------|")
    
    # Process each group in order
    for group in sorted(chord_groups.keys()):
        # Get chords for this group
        group_chords = chord_groups[group]
        
        # Sort chords by start time/index
        if 'beat_idx' in group_chords[0]:
            group_chords.sort(key=lambda c: c['beat_idx'])
        else:
            group_chords.sort(key=lambda c: c['beat_start'])
        
        # Extract unique chord names in order
        chord_names = []
        for chord in group_chords:
            if not chord_names or chord['chord_name'] != chord_names[-1]:
                chord_names.append(chord['chord_name'])
        
        # Add to summary
        start_beat = group * beats_per_group
        end_beat = (group + 1) * beats_per_group - 1
        summary.append(f"| {start_beat}-{end_beat} | {' → '.join(chord_names)} |")
    
    # Add chord frequencies
    summary.append("")
    summary.append("## Chord Frequencies")
    summary.append("")
    
    # Count occurrences of each chord
    chord_counts = {}
    for chord in chords:
        name = chord['chord_name']
        
        # Calculate duration
        if 'beat_idx' in chord and 'beat_idx_end' in chord:
            duration = chord['beat_idx_end'] - chord['beat_idx'] + 1
        elif 'beat_start' in chord and 'beat_end' in chord:
            duration = chord['beat_end'] - chord['beat_start']
        else:
            duration = 1  # Default if no duration info
        
        if name not in chord_counts:
            chord_counts[name] = {'count': 0, 'duration': 0}
        
        chord_counts[name]['count'] += 1
        chord_counts[name]['duration'] += duration
    
    # Sort chords by frequency
    sorted_chords = sorted(chord_counts.items(), 
                          key=lambda x: x[1]['duration'], 
                          reverse=True)
    
    # Add frequency table
    summary.append("| Chord | Count | Duration (beats) |")
    summary.append("|-------|-------|-----------------|")
    
    for name, stats in sorted_chords:
        summary.append(f"| {name} | {stats['count']} | {stats['duration']:.1f} |")
    
    return "\n".join(summary)


def visualize_midi_notes_by_beat(results, start_beat=0, num_beats=16):
    """
    Visualize MIDI notes in each beat as a clear piano roll
    """
    
    midi_beat_blocks = results['midi_beat_blocks']
    beats = results['beats']
    
    if start_beat >= len(midi_beat_blocks) or len(midi_beat_blocks) == 0:
        print("No MIDI beat blocks available to visualize.")
        return None
    
    if len(midi_beat_blocks) < start_beat + num_beats:
        print(f"Not enough beats available. Showing {len(midi_beat_blocks) - start_beat} beats instead.")
        num_beats = len(midi_beat_blocks) - start_beat
    
    # Create subplots - one row per beat, taller rows
    fig, axes = plt.subplots(num_beats, 1, figsize=(15, num_beats * 3))
    if num_beats == 1:
        axes = [axes]  # Ensure axes is always a list
    
    # For each beat...
    for i in range(num_beats):
        beat_idx = start_beat + i
        ax = axes[i]
        
        # Get MIDI data for this beat
        midi_block = midi_beat_blocks[beat_idx]
        
        # Process notes
        bass_notes = []
        harmony_notes = []
        
        for instrument in midi_block.instruments:
            is_bass = 'bass' in instrument.name.lower() if instrument.name else False
            
            for note in instrument.notes:
                note_name = fix_note_name(pitch.Pitch(note.pitch).name)
                
                note_info = {
                    'pitch': note.pitch,
                    'name': note_name,
                    'octave': note.pitch // 12 - 1,  # C4 is MIDI 60, so MIDI / 12 - 1 = octave
                    'pitch_class': note.pitch % 12,
                    'is_bass': is_bass
                }
                
                if is_bass:
                    bass_notes.append(note_info)
                else:
                    harmony_notes.append(note_info)
        
        # Determine root (lowest bass note, or if no bass notes, lowest note)
        if bass_notes:
            bass_notes.sort(key=lambda x: x['pitch'])
            root_note = bass_notes[0]
        elif harmony_notes:
            harmony_notes.sort(key=lambda x: x['pitch'])
            root_note = harmony_notes[0]
        else:
            # No notes in this beat
            ax.text(0.5, 0.5, "No notes in this beat", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_yticks([])
            ax.set_xticks([])
            continue
        
        root_midi = root_note['pitch_class']
        root_name = root_note['name']
        
        # Combine all notes
        all_notes = bass_notes + harmony_notes
        all_notes.sort(key=lambda x: x['pitch'])
        
        # Find pitch range for display
        if all_notes:
            min_pitch = max(0, min(n['pitch'] for n in all_notes) - 12)
            max_pitch = min(127, max(n['pitch'] for n in all_notes) + 12)
        else:
            min_pitch = 48  # Default to C3
            max_pitch = 72  # Default to C5
        
        # Draw piano roll background
        for p in range(min_pitch, max_pitch + 1):
            pc = p % 12
            # White keys: C, D, E, F, G, A, B (0, 2, 4, 5, 7, 9, 11)
            is_white = pc in [0, 2, 4, 5, 7, 9, 11]
            color = 'lightgray' if is_white else 'darkgray'
            alpha = 0.3 if is_white else 0.2
            
            ax.add_patch(patches.Rectangle(
                (0, p - 0.5), 10, 1.0, 
                facecolor=color, edgecolor='gray',
                alpha=alpha, linewidth=0.5, zorder=1
            ))
            
            # Add faint line at C notes
            if pc == 0:
                ax.axhline(y=p, color='gray', linestyle='-', alpha=0.4, zorder=2)
                ax.text(-0.5, p, f"C{p//12-1}", fontsize=8, va='center', ha='right')
        
        # Calculate intervals for each note
        all_interval_names = {
            0: "Root", 1: "b9", 2: "9", 3: "m3", 4: "M3", 
            5: "4", 6: "b5", 7: "5", 8: "#5", 9: "6", 
            10: "b7", 11: "M7"
        }
        
        intervals = set()
        for note in all_notes:
            interval = (note['pitch_class'] - root_midi) % 12
            note['interval'] = interval
            note['interval_name'] = all_interval_names.get(interval, str(interval))
            intervals.add(interval)
        
        # Now create the visualization
        for j, note in enumerate(all_notes):
            # Color notes based on role
            is_root = note['pitch'] == root_note['pitch']
            is_bass = note['is_bass']
            
            if is_root:
                color = 'red'  # Root note
                alpha = 0.9
                zorder = 4
            elif is_bass:
                color = 'orange'  # Other bass notes
                alpha = 0.7
                zorder = 3
            else:
                color = 'blue'  # Harmony notes
                alpha = 0.5
                zorder = 2
            
            # Draw note block
            ax.add_patch(patches.Rectangle(
                (1, note['pitch'] - 0.4), 8, 0.8, 
                facecolor=color, edgecolor='black',
                alpha=alpha, linewidth=1, zorder=zorder
            ))
            
            # Add note label in black
            label = f"{note['name']} ({note['interval_name']})"
            text_color = 'black'
            
            ax.text(5, note['pitch'], label, fontsize=10, color=text_color,
                   ha='center', va='center', fontweight='bold' if is_root else 'normal',
                   zorder=zorder+1, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))
        
        # Analyze chord type
        chord_type = "N.C."
        
        has_minor_third = 3 in intervals
        has_major_third = 4 in intervals
        has_perfect_fifth = 7 in intervals
        has_minor_seventh = 10 in intervals
        has_major_seventh = 11 in intervals
        has_ninth = 2 in intervals
        has_fourth = 5 in intervals
        
        if has_fourth and not has_minor_third and not has_major_third:
            if has_minor_seventh:
                chord_type = "7sus4"
            else:
                chord_type = "sus4"
        elif has_minor_third and has_perfect_fifth and has_minor_seventh:
            if has_ninth:
                chord_type = "m9"
            else:
                chord_type = "m7"
        elif has_major_third and has_perfect_fifth and has_minor_seventh:
            if has_ninth:
                chord_type = "9"
            else:
                chord_type = "7"
        elif has_major_third and has_perfect_fifth and has_major_seventh:
            chord_type = "maj7"
        elif has_minor_third and has_perfect_fifth:
            chord_type = "m"
        elif has_major_third and has_perfect_fifth:
            chord_type = ""  # Major triad
        elif has_perfect_fifth:
            chord_type = "5"  # Power chord
        
        # Add chord title
        chord_name = f"{root_name}{chord_type}"
        
        # Add title with beat number and chord
        ax.set_title(f"Beat {beat_idx+1}: {chord_name}", fontsize=14, fontweight='bold')
        
        # List actual notes directly on the visualization
        note_names = [f"{n['name']}" for n in all_notes]
        ax.text(5, min_pitch - 5, f"Notes: {', '.join(note_names)}", 
               ha='center', va='center', fontsize=12,
               bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.3'))
        
        # Add legend for note types
        legend_elements = []
        if any(n['pitch'] == root_note['pitch'] for n in all_notes):
            legend_elements.append(patches.Patch(facecolor='red', alpha=0.9, label='Root'))
        if any(n['is_bass'] and n['pitch'] != root_note['pitch'] for n in all_notes):
            legend_elements.append(patches.Patch(facecolor='orange', alpha=0.7, label='Bass'))
        if any(not n['is_bass'] for n in all_notes):
            legend_elements.append(patches.Patch(facecolor='blue', alpha=0.5, label='Harmony'))
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Configure axis
        ax.set_xlim(-1, 11)
        ax.set_ylim(min_pitch - 8, max_pitch + 2)  # More space at bottom for note list
        ax.set_xticks([])  # Hide x-axis ticks
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed beat analysis (only once)
    print("\nDetailed MIDI Analysis:")
    print("=====================")
    for i in range(num_beats):
        beat_idx = start_beat + i
        if beat_idx >= len(midi_beat_blocks):
            break
            
        midi_block = midi_beat_blocks[beat_idx]
        
        print(f"\nBeat {beat_idx+1}:")
        print("-" * 20)
        
        # Get all notes
        all_notes = []
        for instrument in midi_block.instruments:
            is_bass = 'bass' in instrument.name.lower() if instrument.name else False
            inst_name = instrument.name if instrument.name else "Unknown"
            
            for note in instrument.notes:
                note_name = fix_note_name(pitch.Pitch(note.pitch).name)
                all_notes.append({
                    'pitch': note.pitch,
                    'name': note_name,
                    'pitch_class': note.pitch % 12,
                    'instrument': inst_name,
                    'is_bass': is_bass
                })
        
        # Group by instrument
        by_instrument = {}
        for note in all_notes:
            if note['instrument'] not in by_instrument:
                by_instrument[note['instrument']] = []
            by_instrument[note['instrument']].append(note)
        
        # Print notes by instrument
        for inst, notes in by_instrument.items():
            notes.sort(key=lambda n: n['pitch'])
            note_names = [n['name'] for n in notes]
            if note_names:
                print(f"  {inst}: {', '.join(note_names)}")
        
        # Determine root note and analyze chord
        if all_notes:
            # Determine root
            bass_notes = [n for n in all_notes if n['is_bass']]
            
            if bass_notes:
                lowest_bass = min(bass_notes, key=lambda n: n['pitch'])
                root_name = lowest_bass['name']
                root_midi = lowest_bass['pitch_class']
                # print(f"  Root from bass: {root_name}")
            else:
                # No bass, use lowest note as root
                all_notes.sort(key=lambda n: n['pitch'])
                root_name = all_notes[0]['name']
                root_midi = all_notes[0]['pitch_class']
                # print(f"  Root from lowest note: {root_name}")
            
            # Calculate intervals from root
            intervals = set()
            for note in all_notes:
                interval = (note['pitch_class'] - root_midi) % 12
                intervals.add(interval)
            
            interval_names = {
                0: "Root", 1: "b9", 2: "9", 3: "m3", 4: "M3", 
                5: "4", 6: "b5", 7: "5", 8: "#5", 9: "6", 
                10: "b7", 11: "M7"
            }
            
            # Print intervals as text
            int_list = sorted(intervals)
            named_intervals = [f"{i} ({interval_names.get(i, i)})" for i in int_list]
            print(f"  Intervals from root: {', '.join(named_intervals)}")
            
            # Analyze chord type
            has_minor_third = 3 in intervals
            has_major_third = 4 in intervals
            has_perfect_fifth = 7 in intervals
            has_minor_seventh = 10 in intervals
            has_major_seventh = 11 in intervals
            has_ninth = 2 in intervals
            has_fourth = 5 in intervals
            
            # Determine chord type
            chord_type = "N.C."
            if has_fourth and not has_minor_third and not has_major_third:
                if has_minor_seventh:
                    chord_type = "7sus4"
                else:
                    chord_type = "sus4"
            elif has_minor_third and has_perfect_fifth and has_minor_seventh:
                if has_ninth:
                    chord_type = "m9"
                else:
                    chord_type = "m7"
            elif has_major_third and has_perfect_fifth and has_minor_seventh:
                if has_ninth:
                    chord_type = "9"
                else:
                    chord_type = "7"
            elif has_major_third and has_perfect_fifth and has_major_seventh:
                chord_type = "maj7"
            elif has_minor_third and has_perfect_fifth:
                chord_type = "m"
            elif has_major_third and has_perfect_fifth:
                chord_type = ""  # Major triad
            elif has_perfect_fifth:
                chord_type = "5"  # Power chord
                
            print(f"  Determined chord: {root_name}{chord_type}")
        else:
            print("  No notes in this beat")
    
    return fig

def correct_enharmonics(chord_data, tonality, alterations, scale):
    """
    Corrects chord names to use the appropriate enharmonic spelling based on the key.
    
    Args:
        chord_data: Dictionary with 'chords' key containing list of chord dictionaries
        tonality: The key of the song (e.g., "F minor")
        alterations: List of altered notes in the key (e.g., ["Bb", "Eb", "Ab", "Db"])
        scale: List of notes in the scale (e.g., ["F", "G", "Ab", "Bb", "C", "Db", "Eb"])
        
    Returns:
        A new dictionary with corrected chord names
    """
    # Define enharmonic equivalents (sharp to flat and flat to sharp)
    enharmonic_map = {
        'C#': 'Db', 'Db': 'C#',
        'D#': 'Eb', 'Eb': 'D#',
        'F#': 'Gb', 'Gb': 'F#',
        'G#': 'Ab', 'Ab': 'G#',
        'A#': 'Bb', 'Bb': 'A#',
        'E#': 'F', 'F': 'E#',  # Less common
        'B#': 'C', 'C': 'B#',  # Less common
        'Cb': 'B', 'B': 'Cb'   # Less common
    }
    
    # Create a flattened set of all notes in the scale (for quick lookup)
    scale_notes = set(scale)
    
    # Create the preference map based on the key
    preferred_spellings = {}
    
    # For each possible note that has an enharmonic equivalent
    for sharp_note, flat_note in [('C#', 'Db'), ('D#', 'Eb'), ('F#', 'Gb'), 
                                 ('G#', 'Ab'), ('A#', 'Bb')]:
        # Check which spelling appears in the scale
        if sharp_note in scale_notes:
            preferred_spellings[sharp_note] = sharp_note
            preferred_spellings[flat_note] = sharp_note
        elif flat_note in scale_notes:
            preferred_spellings[sharp_note] = flat_note
            preferred_spellings[flat_note] = flat_note
        else:
            # If neither is in the scale, prefer the one from alterations
            if flat_note in alterations:
                preferred_spellings[sharp_note] = flat_note
                preferred_spellings[flat_note] = flat_note
            elif sharp_note in alterations:
                preferred_spellings[sharp_note] = sharp_note
                preferred_spellings[flat_note] = sharp_note
            else:
                # Default to flat in minor keys, sharp in major keys
                if "minor" in tonality.lower():
                    preferred_spellings[sharp_note] = flat_note
                    preferred_spellings[flat_note] = flat_note
                else:
                    preferred_spellings[sharp_note] = sharp_note
                    preferred_spellings[flat_note] = sharp_note
    
    # Check if chord_data is a dictionary with 'chords' key or a list
    if isinstance(chord_data, dict) and 'chords' in chord_data:
        chords_list = chord_data['chords']
    else:
        chords_list = chord_data
    
    corrected_chords = []
    
    # Process each chord in the list
    for chord_dict in chords_list:
        # Create a copy of the original chord dictionary
        corrected_chord = chord_dict.copy()
        corrected_chord['tonality'] = tonality
        corrected_chord['scale'] = scale
        
        # Correct the chord name
        if 'chord_name' in chord_dict and chord_dict['chord_name'] != 'N.C.':
            chord_name = chord_dict['chord_name']
            
            # Extract root and quality
            if len(chord_name) > 1 and chord_name[1] in ['#', 'b']:
                root = chord_name[:2]
                quality = chord_name[2:]
            else:
                root = chord_name[:1]
                quality = chord_name[1:]
            
            # If the root has an enharmonic equivalent and a preferred spelling
            if root in enharmonic_map and root in preferred_spellings:
                corrected_root = preferred_spellings[root]
                corrected_chord['chord_name'] = corrected_root + quality
        
        # Also correct the root name if present
        if 'root_name' in chord_dict:
            root = chord_dict['root_name']
            if root in enharmonic_map and root in preferred_spellings:
                corrected_chord['root_name'] = preferred_spellings[root]
        
        corrected_chords.append(corrected_chord)
    
    # Return in the same format as input
    # if isinstance(chord_data, dict) and 'chords' in chord_data:
    #     result = chord_data.copy()
    #     result['chords'] = corrected_chords
    #     return result
    # else:
    return corrected_chords


def visualize_midi_files(midi_paths, component='both', start_beat=0, num_beats=20):
    """
    Visualize the bass, harmony, or both MIDI files directly (before merging)
    
    Parameters:
    -----------
    midi_paths : dict
        Dictionary with paths to MIDI files ('bass' and 'harmony')
    component : str
        'bass', 'harmony', or 'both'
    start_beat : int
        First beat to display
    num_beats : int
        Number of beats to display
    """
    # Load MIDI files based on component
    midi_data = {}
    if component in ['bass', 'both']:
        midi_data['bass'] = pretty_midi.PrettyMIDI(midi_paths['bass'])
    if component in ['harmony', 'both']:
        midi_data['harmony'] = pretty_midi.PrettyMIDI(midi_paths['harmony'])
    
    # Extract beats from the first available MIDI file
    first_key = next(iter(midi_data))
    beats = midi_data[first_key].get_beats()
    
    # Handle case with no beats detected
    if len(beats) == 0:
        print("No beats detected in MIDI file. Using time-based slices instead.")
        # Create artificial beats every quarter second
        first_midi = midi_data[first_key]
        end_time = max([note.end for inst in first_midi.instruments for note in inst.notes]) if any(first_midi.instruments) else 60
        beats = np.arange(0, end_time, 0.5)  # Create beats every half-second
    
    # Limit to available beats
    if start_beat >= len(beats):
        print(f"Start beat {start_beat} exceeds available beats ({len(beats)})")
        start_beat = 0
    
    end_beat = min(start_beat + num_beats, len(beats))
    
    # Create figure
    fig, axes = plt.subplots(end_beat - start_beat, 1, figsize=(15, (end_beat - start_beat) * 2.5))
    if end_beat - start_beat == 1:
        axes = [axes]  # Ensure axes is always a list
    
    # For each beat
    for i in range(start_beat, end_beat):
        ax_idx = i - start_beat
        ax = axes[ax_idx]
        
        # Get time boundaries for this beat
        beat_start = beats[i]
        beat_end = beats[i+1] if i+1 < len(beats) else beat_start + 0.5  # Default to half-second if last beat
        
        # Collect notes within this beat time frame
        beat_notes = []
        
        for midi_type, midi in midi_data.items():
            is_bass = midi_type == 'bass'
            
            for inst_idx, instrument in enumerate(midi.instruments):
                for note in instrument.notes:
                    # Note is in this beat if:
                    # 1. It starts during this beat, OR
                    # 2. It started before but is still playing during this beat
                    if (beat_start <= note.start < beat_end) or \
                       (note.start < beat_start and note.end > beat_start):
                        
                        note_name = pitch.Pitch(note.pitch).name
                        note_info = {
                            'pitch': note.pitch,
                            'name': note_name,
                            'start': note.start,
                            'end': note.end,
                            'velocity': note.velocity,
                            'is_bass': is_bass,
                            'source': midi_type
                        }
                        beat_notes.append(note_info)
        
        # If no notes in this beat
        if not beat_notes:
            ax.text(0.5, 0.5, f"No notes in beat {i+1} ({beat_start:.2f}s - {beat_end:.2f}s)", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_yticks([])
            ax.set_xticks([])
            continue
        
        # Find bass/root note
        if any(note['is_bass'] for note in beat_notes):
            bass_notes = [note for note in beat_notes if note['is_bass']]
            bass_notes.sort(key=lambda x: x['pitch'])
            root_note = bass_notes[0]
        else:
            # Use lowest note as root if no bass
            beat_notes.sort(key=lambda x: x['pitch'])
            root_note = beat_notes[0]
        
        root_midi = root_note['pitch'] % 12
        root_name = root_note['name']
        
        # Find pitch range for display
        min_pitch = max(0, min(n['pitch'] for n in beat_notes) - 5)
        max_pitch = min(127, max(n['pitch'] for n in beat_notes) + 5)
        
        # Draw piano roll background
        for p in range(min_pitch, max_pitch + 1):
            pc = p % 12
            # White keys: C, D, E, F, G, A, B (0, 2, 4, 5, 7, 9, 11)
            is_white = pc in [0, 2, 4, 5, 7, 9, 11]
            color = 'lightgray' if is_white else 'darkgray'
            alpha = 0.3 if is_white else 0.2
            
            ax.add_patch(plt.Rectangle(
                (0, p - 0.5), 10, 1.0, 
                facecolor=color, edgecolor='gray',
                alpha=alpha, linewidth=0.5, zorder=1
            ))
            
            # Add faint line at C notes
            if pc == 0:
                ax.axhline(y=p, color='gray', linestyle='-', alpha=0.4, zorder=2)
                ax.text(-0.5, p, f"C{p//12-1}", fontsize=8, va='center', ha='right')
        
        # Calculate intervals
        all_interval_names = {
            0: "Root", 1: "b9", 2: "9", 3: "m3", 4: "M3", 
            5: "4", 6: "b5", 7: "5", 8: "#5", 9: "6", 
            10: "b7", 11: "M7"
        }
        
        # Now draw the notes
        for note in beat_notes:
            # Calculate interval from root
            interval = (note['pitch'] % 12 - root_midi) % 12
            interval_name = all_interval_names.get(interval, str(interval))
            
            # Determine note color
            if note['pitch'] % 12 == root_midi and note['is_bass']:
                color = 'red'  # Root bass note
                alpha = 0.9
            elif note['is_bass']:
                color = 'orange'  # Other bass notes
                alpha = 0.7
            else:
                color = 'blue'  # Harmony notes
                alpha = 0.5
            
            # Draw note block
            ax.add_patch(plt.Rectangle(
                (1, note['pitch'] - 0.4), 8, 0.8, 
                facecolor=color, edgecolor='black',
                alpha=alpha, linewidth=1, zorder=3
            ))
            
            # Add note label
            label = f"{note['name']} ({interval_name})"
            ax.text(5, note['pitch'], label, fontsize=10, color='black',
                   ha='center', va='center', fontweight='bold' if note['pitch'] % 12 == root_midi else 'normal',
                   zorder=4, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))
        
        # Add title with beat and time info
        ax.set_title(f"Beat {i+1} (Time: {beat_start:.2f}-{beat_end:.2f}s)", fontsize=12)
        
        # Add note names for this beat
        note_names = [f"{n['name']} ({n['source']})" for n in beat_notes]
        ax.text(5, min_pitch - 3, f"Notes: {', '.join(note_names)}", 
               ha='center', va='center', fontsize=10,
               bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.3'))
        
        # Configure axis
        ax.set_xlim(-1, 11)
        ax.set_ylim(min_pitch - 5, max_pitch + 2)
        ax.set_xticks([])  # Hide x-axis ticks
    
    # Add overall title
    component_titles = {
        'bass': 'Bass MIDI Only',
        'harmony': 'Harmony MIDI Only',
        'both': 'Bass & Harmony MIDI Combined'
    }
    plt.suptitle(f"{component_titles[component]} - Beats {start_beat+1} to {end_beat}", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return fig


#-------------------------------------------------------------------------------
def compute_per_beat_rms(audio, beats, sr):
    rms_per_beat = []
    for i in range(len(beats)-1):
        start = int(beats[i] * sr)
        end = int(beats[i+1] * sr)
        segment = audio[start:end]
        if len(segment) == 0:
            rms = 0
        else:
            rms = np.sqrt(np.mean(segment**2))
        rms_per_beat.append(rms)
    return np.array(rms_per_beat)


#-------------------------------------------------------------------------------
def analyze_chords_from_beat_results(results, beats_per_row=64, plot_it=False):
    """
    Analyze chords per beat from beat alignment results, using bass as root and harmony for quality.
    Empty bass/harmony are filled from the previous beat ("hold over" logic).
    Returns merged chords and the beat grid object.
    """
    
    # Create BeatGrid object
    beat_grid = create_beat_grid(results)

    # Get the beat_notes dict: beat_idx -> list of note dicts (each with 'instrument', 'pitch', etc.)
    beat_notes = results['beat_notes']
    beats = results['beats']
    num_beats = len(beat_notes)

    # Hold-over for last seen non-empty bass/harmony
    last_bass_notes = []
    last_harmony_notes = []

    # Store per-beat chord info
    chord_results_per_beat = []

    for beat_idx in range(num_beats):
        notes = beat_notes.get(beat_idx, [])
        # Separate instruments
        bass_notes = [n for n in notes if n['instrument'] == 'bass']
        harmony_notes = [n for n in notes if n['instrument'] == 'harmony']

        # Hold over previous non-empty bass/harmony
        if not bass_notes:
            bass_notes = last_bass_notes
        else:
            last_bass_notes = bass_notes
        if not harmony_notes:
            harmony_notes = last_harmony_notes
        else:
            last_harmony_notes = harmony_notes

        # If still no bass, cannot define chord for this beat
        if not bass_notes:
            chord_results_per_beat.append(None)
            continue

        # Use *lowest* bass as root
        root_note = min(bass_notes, key=lambda n: n['pitch'])
        root_pc = root_note['pitch'] % 12
        root_pitch = root_note['pitch']
        # Use fix_note_name to get proper note name
        root_name = fix_note_name(pitch.Pitch(root_pitch).name)

        # Combine bass and harmony notes for chord analysis
        all_notes = bass_notes + harmony_notes
        
        
        # Sort all notes by pitch
        sorted_notes = sorted(all_notes, key=lambda x: x['pitch'])
        
        # Create array of absolute pitches
        midi_pitches = np.array([note['pitch'] for note in sorted_notes])
        
        # Calculate pitch class intervals from root
        pc_intervals = [(p % 12 - root_pc) % 12 for p in midi_pitches]
        
        # Create interval durations (use beat duration for simplicity)
        beat_duration = beats[beat_idx+1] - beats[beat_idx] if beat_idx+1 < len(beats) else 0.5
        interval_durations = {}
        for i, note in enumerate(sorted_notes):
            interval = pc_intervals[i]
            if interval not in interval_durations:
                interval_durations[interval] = 0
            interval_durations[interval] += note.get('duration', beat_duration)
        
        # Ensure root is always present
        if 0 not in interval_durations:
            interval_durations[0] = beat_duration
        
        # Create window result for chord analyzer - DIRECT CALL TO identify_unique_chord
        window_result = {
            'intervals': pc_intervals,
            'interval_durations': interval_durations,
            'root_name': root_name,
            'root_pitch': root_pitch,
            'root_pc': root_pc,
            'absolute_pitches': midi_pitches.tolist(),
            'lowest_pitch': root_pitch
        }
        
        # Use identify_unique_chord directly instead of analyze_chord_from_note_bag
        refined_results = identify_unique_chord([window_result])
        chord_type = refined_results[0]['refined_chord_type']
        
        # Create chord info
        chord_info = {
            'beat_idx': beat_idx,
            'beat_start': beats[beat_idx],
            'beat_end': beats[beat_idx] + beat_duration,
            'root_midi': root_pc,
            'root_pitch': root_pitch,
            'root_name': root_name,
            'chord_type': chord_type,
            'chord_name': f"{root_name}{chord_type}" if chord_type != 'N.C.' else 'N.C.',
            'notes': midi_pitches.tolist(),
            'pitch_classes': set(p % 12 for p in midi_pitches),
            'intervals': pc_intervals,
        }
        
        chord_results_per_beat.append(chord_info)

    # Store chord results in beat grid
    beat_grid.chord_results = chord_results_per_beat
    
    # Create merged chords (consecutive identical chords)
    merged_chords = []
    if chord_results_per_beat:
        current_chord = None
        current_start_idx = None
        
        for i, chord in enumerate(chord_results_per_beat):
            if chord is None:
                continue
                
            if current_chord is None:
                # First chord
                current_chord = chord.copy()
                current_start_idx = i
            elif (chord['chord_name'] == current_chord['chord_name'] and 
                  chord['root_name'] == current_chord['root_name']):
                # Same chord continues, just update end
                current_chord['beat_end'] = chord['beat_end']
                current_chord['beat_idx_end'] = i
            else:
                # Different chord, save previous and start new
                current_chord['beat_idx_end'] = i - 1
                merged_chords.append(current_chord)
                current_chord = chord.copy()
                current_start_idx = i
        
        # Don't forget the last chord
        if current_chord is not None:
            current_chord['beat_idx_end'] = len(chord_results_per_beat) - 1
            merged_chords.append(current_chord)

    # Optionally plot
    if plot_it:
        chord_fig = beat_grid.visualize_chords(merged_chords, beats_per_row)

    return {
        'chords': merged_chords,
        'beat_grid': beat_grid  # For further analysis if needed
    }