import numpy as np
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import pretty_midi
import soundfile as sf
import os

def detect_beats_plp(y, sr, hop_length=512, dynamic_tempo=False):
    """
    Detect beats using librosa's PLP (Probabilistic Local Pulse) method.
    
    This function supports both stable tempo and varying tempo detection.
    
    Parameters:
    -----------
    y : numpy.ndarray
        Audio time series
    sr : int
        Sample rate
    hop_length : int
        Hop length for STFT
    dynamic_tempo : bool
        Whether to use dynamic tempo tracking (for songs with tempo changes)
    
    Returns:
    --------
    beats : array of beat times (seconds)
    tempo : float or array, estimated tempo in BPM
    """
    print("Detecting beats using PLP method...")
    
    # Separate harmonic and percussive components for better beat detection
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Calculate onset strength envelope - focus on percussive content
    onset_env = librosa.onset.onset_strength(
        y=y_percussive, 
        sr=sr,
        hop_length=hop_length
    )
    
    # Get tempo estimation
    if dynamic_tempo:
        # For dynamic tempo tracking, calculate tempo over time
        dtempo = librosa.beat.tempo(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            aggregate=None  # No aggregation for dynamic tempo
        )
        # Use median tempo as reference
        tempo = np.median(dtempo)
        print(f"Estimated tempo range: {np.min(dtempo):.2f} - {np.max(dtempo):.2f} BPM (median: {tempo:.2f} BPM)")
    else:
        # For stable tempo, use the mean estimator
        tempo = librosa.beat.tempo(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            aggregate=np.mean  # Use mean for more stability
        )[0]
        print(f"Estimated stable tempo: {tempo:.2f} BPM")
    
    # Use PLP beat tracker - compatible with your librosa version
    # Note: older versions of librosa don't accept the tempo parameter
    prior = librosa.beat.plp(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length
        # tempo parameter removed as it's not supported in this version
    )
    
    # Get beat positions using dynamic programming
    beat_plp = librosa.beat.beat_track(
        onset_envelope=prior,  # Use PLP enhanced onset envelope
        sr=sr,
        hop_length=hop_length,
        units='time',
        trim=False
    )[1]
    
    if len(beat_plp) == 0:
        # Fallback if PLP fails to detect beats
        print("PLP beat detection produced no beats. Falling back to standard method.")
        beat_plp = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            units='time', 
            trim=False
        )[1]
    
    print(f"Detected {len(beat_plp)} beats")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Waveform with beat positions
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.5, color='gray')
    plt.vlines(beat_plp, -1, 1, color='r', alpha=0.7, label='PLP Beats')
    plt.title(f'Audio Waveform with Detected Beats (Tempo: {tempo:.2f} BPM)')
    plt.legend()
    plt.xlim(0, min(30, len(y)/sr))  # Show first 30 seconds or less
    
    # Plot 2: Onset strength with beat positions
    plt.subplot(2, 1, 2)
    hop_time = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    plt.plot(hop_time, onset_env, alpha=0.7, label='Onset Strength')
    plt.vlines(beat_plp, 0, onset_env.max(), color='r', alpha=0.7, label='PLP Beats')
    plt.title('Onset Strength and Beat Positions')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.xlim(0, min(30, len(y)/sr))  # Show first 30 seconds or less
    
    plt.tight_layout()
    plt.show()
    
    # Create and play click track
    click_times = beat_plp
    click_track = create_click_track(click_times, [880] * len(click_times), sr, length=len(y))
    
    print("\nPlaying audio with detected beat grid:")
    display(Audio(data=y + click_track * 0.4, rate=sr))
    
    return beat_plp, tempo

def detect_beats_dynamic(y, sr, hop_length=512):
    """
    Detect beats in songs with varying tempo changes.
    Uses the approach from librosa's dynamic beat tracking example.
    
    Parameters:
    -----------
    y : numpy.ndarray
        Audio time series
    sr : int
        Sample rate
    hop_length : int
        Hop length for STFT
    
    Returns:
    --------
    beats : array of beat times (seconds)
    tempo : float, estimated median tempo in BPM
    """
    print("Detecting beats for varying tempo...")
    
    # Separate harmonic and percussive components for better beat detection
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Calculate onset strength envelope - focus on percussive content
    onset_env = librosa.onset.onset_strength(
        y=y_percussive, 
        sr=sr,
        hop_length=hop_length
    )
    
    # Dynamic tempo analysis
    dtempo = librosa.beat.tempo(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        aggregate=None  # No aggregation for dynamic tempo
    )
    
    # Get median tempo
    tempo = np.median(dtempo)
    print(f"Tempo range: {np.min(dtempo):.2f} - {np.max(dtempo):.2f} BPM (median: {tempo:.2f} BPM)")
    
    # Adaptive beat tracking - automatically follows tempo changes
    ac = librosa.autocorrelate(onset_env, max_size=2 * sr // hop_length)
    
    # Get adaptive prior to follow varying tempo
    prior = librosa.beat.plp(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length
        # tempo=None parameter removed as it's not supported in this version
    )
    
    # Get beat positions using dynamic programming
    beats = librosa.beat.beat_track(
        onset_envelope=prior,  # Use PLP enhanced onset envelope
        sr=sr,
        hop_length=hop_length,
        units='time',
        trim=False
    )[1]
    
    # Visualize the results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Waveform with beat positions
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.5, color='gray')
    plt.vlines(beats, -1, 1, color='r', alpha=0.7, label='Dynamic Beats')
    plt.title(f'Audio Waveform with Detected Beats (Median Tempo: {tempo:.2f} BPM)')
    plt.legend()
    plt.xlim(0, min(30, len(y)/sr))  # Show first 30 seconds or less
    
    # Plot 2: Onset strength with beat positions
    plt.subplot(3, 1, 2)
    hop_time = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    plt.plot(hop_time, onset_env, alpha=0.7, label='Onset Strength')
    plt.vlines(beats, 0, onset_env.max(), color='r', alpha=0.7, label='Dynamic Beats')
    plt.title('Onset Strength and Beat Positions')
    plt.legend()
    plt.xlim(0, min(30, len(y)/sr))  # Show first 30 seconds or less
    
    # Plot 3: Visualize tempo changes over time
    plt.subplot(3, 1, 3)
    frames = np.arange(len(dtempo))
    beat_times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    plt.plot(beat_times, dtempo, label='Tempo Estimate')
    plt.axhline(y=tempo, color='r', linestyle='--', alpha=0.5, label=f'Median: {tempo:.2f} BPM')
    plt.title('Tempo Changes Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Tempo (BPM)')
    plt.legend()
    plt.xlim(0, min(30, len(y)/sr))  # Show first 30 seconds or less
    
    plt.tight_layout()
    plt.show()
    
    # Create and play click track
    click_track = create_click_track(beats, [880] * len(beats), sr, length=len(y))
    
    print("\nPlaying audio with adaptive beat grid:")
    display(Audio(data=y + click_track * 0.4, rate=sr))
    
    return beats, tempo

def detect_stable_beats(y, sr):
    """
    Detect a stable, consistent beat grid for a song with constant tempo.
    
    This function:
    1. Estimates the global tempo
    2. Creates a perfectly regular beat grid at that tempo
    3. Finds the optimal phase offset to align the grid with the audio
    
    Parameters:
    -----------
    y : audio signal
    sr : sample rate
    
    Returns:
    --------
    beats : array of beat times (seconds)
    tempo : detected tempo in BPM
    offset : optimal phase offset in seconds
    """
    print("Detecting stable beat grid...")
    
    # Calculate overall tempo using multiple estimators
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Focus on percussive content for tempo estimation
    onset_env = librosa.onset.onset_strength(
        y=y_percussive, 
        sr=sr,
        hop_length=512
    )
    
    # Get tempo estimation
    tempo = librosa.beat.tempo(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=512,
        aggregate=np.mean  # Use mean for more stability
    )[0]
    
    print(f"Estimated tempo: {tempo:.2f} BPM")
    
    # Create a perfectly regular beat grid at the estimated tempo
    beat_period = 60.0 / tempo  # seconds per beat
    song_duration = len(y) / sr
    
    # Create a perfect beat grid (without phase offset yet)
    num_beats = int(np.ceil(song_duration / beat_period))
    perfect_beats = np.arange(0, num_beats) * beat_period
    
    # Find the optimal phase offset to align the grid with audio content
    # We'll test different offsets and choose the one with maximum energy
    
    # Define range of offsets to test (in seconds)
    max_offset = beat_period  # One full beat period
    num_offsets = 100  # Test 100 different offsets within one beat
    offsets = np.linspace(0, max_offset, num_offsets, endpoint=False)
    
    # Calculate audio energy (focus on low frequencies for kick drum)
    S = np.abs(librosa.stft(y_percussive, hop_length=512))
    
    # Focus on lower frequencies (where kick drums are typically found)
    low_freq_energy = np.sum(S[:40, :], axis=0)  # Sum the first 40 frequency bins
    energy_times = librosa.frames_to_time(np.arange(len(low_freq_energy)), sr=sr, hop_length=512)
    
    # Evaluate each offset
    offset_scores = []
    
    for offset in offsets:
        # Shift the beat grid
        shifted_beats = perfect_beats + offset
        
        # Keep only beats within the song duration
        shifted_beats = shifted_beats[shifted_beats < song_duration]
        
        # Convert beats to frames
        beat_frames = librosa.time_to_frames(shifted_beats, sr=sr, hop_length=512)
        beat_frames = beat_frames[beat_frames < len(low_freq_energy)]
        
        # Calculate energy at these beat positions
        if len(beat_frames) > 0:
            beat_energies = low_freq_energy[beat_frames]
            score = np.mean(beat_energies)
        else:
            score = 0
            
        offset_scores.append(score)
    
    # Find best offset
    best_idx = np.argmax(offset_scores)
    best_offset = offsets[best_idx]
    
    print(f"Optimal beat phase offset: {best_offset:.4f} seconds")
    
    # Create final beat grid with optimal offset
    beats = np.arange(0, num_beats) * beat_period + best_offset
    beats = beats[beats < song_duration]
    
    # Check if we need to invert the beat pattern (if upbeats were detected as downbeats)
    # For most EDM and pop music, we can check if strong/weak beat pattern matches
    
    # Create two separate beat groups: even and odd indexed beats
    even_beats = beats[0::2]
    odd_beats = beats[1::2]
    
    # Convert to frames
    even_frames = librosa.time_to_frames(even_beats, sr=sr, hop_length=512)
    even_frames = even_frames[even_frames < len(low_freq_energy)]
    
    odd_frames = librosa.time_to_frames(odd_beats, sr=sr, hop_length=512)
    odd_frames = odd_frames[odd_frames < len(low_freq_energy)]
    
    # Calculate average energy for each group
    even_energy = np.mean(low_freq_energy[even_frames]) if len(even_frames) > 0 else 0
    odd_energy = np.mean(low_freq_energy[odd_frames]) if len(odd_frames) > 0 else 0
    
    # If odd beats have more energy, we might need to shift by half a beat
    if odd_energy > even_energy * 1.1:  # 10% threshold
        print("Odd beats have more energy, shifting half a beat...")
        half_beat = beat_period / 2
        beats = np.arange(0, num_beats) * beat_period + best_offset + half_beat
        beats = beats[beats < song_duration]
    
    # Visualize the result
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Offset scores
    plt.subplot(3, 1, 1)
    plt.plot(offsets, offset_scores)
    plt.axvline(x=best_offset, color='r', linestyle='--')
    plt.xlabel('Offset (seconds)')
    plt.ylabel('Energy Score')
    plt.title(f'Beat Phase Alignment (Optimal: {best_offset:.4f} sec)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Waveform with beat grid
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(y))/sr, y, color='gray', alpha=0.5)
    plt.vlines(beats, -0.5, 0.5, color='r', alpha=0.7)
    plt.title('Waveform with Stable Beat Grid')
    plt.xlim(0, min(30, len(y)/sr))  # Show first 30 seconds or less
    
    # Plot 3: Low-frequency energy with beat positions
    plt.subplot(3, 1, 3)
    plt.plot(energy_times, low_freq_energy, alpha=0.5)
    
    # Create beat pattern: emphasize every 4th beat for 4/4 time
    colors = ['r', 'b', 'b', 'b'] * (len(beats) // 4 + 1)
    colors = colors[:len(beats)]
    alphas = [0.8, 0.4, 0.4, 0.4] * (len(beats) // 4 + 1)
    alphas = alphas[:len(beats)]
    
    for i, beat in enumerate(beats):
        if beat < max(energy_times):
            plt.axvline(x=beat, color=colors[i], alpha=alphas[i])
    
    plt.title('Low-Frequency Energy with Beat Grid')
    plt.xlabel('Time (seconds)')
    plt.xlim(0, min(30, len(y)/sr))  # Show first 30 seconds or less
    
    plt.tight_layout()
    plt.show()
    
    # Create click track
    # Emphasize every 4th beat (assuming 4/4 time signature)
    click_times = []
    click_freqs = []
    
    for i, beat in enumerate(beats):
        click_times.append(beat)
        if i % 4 == 0:  # Every 4th beat is a downbeat
            click_freqs.append(1760)  # Higher pitch for downbeats
        else:
            click_freqs.append(880)   # Lower pitch for other beats
    
    # Create the click track using our custom function that works with both API versions
    click_track = create_click_track(click_times, click_freqs, sr, click_duration=0.1, length=len(y))
    
    print("\nPlaying audio with stable beat grid:")
    display(Audio(data=y + click_track * 0.4, rate=sr))
    
    return beats, tempo

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

def align_midi_to_stable_grid(midi_data, beats, tempo, audio_duration, fine_adjust_ms=0):
    print("\n[DEBUG] --- align_midi_to_stable_grid called ---")
    print(f"[DEBUG] Number of audio beats: {len(beats)} | Number of MIDI beats: {len(midi_data.get_beats())}")
    midi_note_starts = [note.start for inst in midi_data.instruments for note in inst.notes]
    if midi_note_starts:
        print(f"[DEBUG] First MIDI note start: {min(midi_note_starts):.3f} | Last MIDI note start: {max(midi_note_starts):.3f}")
    else:
        print("[DEBUG] No MIDI notes found!")

    # Get MIDI tempo with robust estimation
    midi_tempo = midi_data.estimate_tempo()
    midi_beats = midi_data.get_beats()

    # If tempo estimation fails, calculate from beats or note patterns
    if midi_tempo == 0 or midi_tempo < 20 or midi_tempo > 300:
        if len(midi_beats) >= 2:
            midi_beat_intervals = np.diff(midi_beats)
            midi_tempo = 60 / np.median(midi_beat_intervals)
            print(f"[DEBUG] Calculated MIDI tempo from beat intervals: {midi_tempo:.2f} BPM")
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
                        print(f"[DEBUG] Estimated MIDI tempo from note patterns: {midi_tempo:.2f} BPM")
                    else:
                        midi_tempo = tempo
                else:
                    midi_tempo = tempo
            else:
                midi_tempo = tempo
    else:
        print(f"[DEBUG] MIDI tempo from file metadata: {midi_tempo:.2f} BPM")

    tempo_ratio = tempo / midi_tempo
    print(f"[DEBUG] Audio tempo: {tempo:.2f} BPM | MIDI tempo: {midi_tempo:.2f} BPM | Tempo ratio: {tempo_ratio:.4f}")

    fine_adjust_sec = fine_adjust_ms / 1000.0

    # --- RELIABILITY CHECK ---
    min_beats_required = 8
    tempo_ratio_close = abs(tempo_ratio - 1) < 0.1
    midi_beats_ok = len(midi_beats) >= min_beats_required
    audio_beats_ok = len(beats) >= min_beats_required
    tempo_ok = 20 < midi_tempo < 300
    mapping_allowed = midi_beats_ok and audio_beats_ok and tempo_ok and tempo_ratio_close

    if not mapping_allowed:
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
            print(f"[DEBUG] After alignment: First note {min(aligned_note_starts):.3f} | Last note {max(aligned_note_starts):.3f}")
        return aligned_midi

    # --- ELSE: USE BEAT-TO-BEAT MAPPING/SCALING ---
    print("[DEBUG] Using beat-to-beat mapping for direct beat alignment...")
    from scipy.interpolate import interp1d
    max_beats_to_use = min(len(midi_beats), len(beats))
    max_midi_time = midi_beats[-1]
    try:
        mapping_fn = interp1d(
            midi_beats[:max_beats_to_use],
            beats[:max_beats_to_use],
            kind='linear',
            bounds_error=False,
            fill_value=(beats[0], beats[-1])
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
                start_time = beats[0] + (note.start * tempo_ratio) + fine_adjust_sec
                end_time = beats[0] + (note.end * tempo_ratio) + fine_adjust_sec
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
        print(f"[DEBUG] Alignment quality: {alignment_quality:.2%} of notes aligned with beats")
    return aligned_midi

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
    
    # Create the click track using our custom function that works with both API versions
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

def visualize_midi_alignment(y, sr, beats, original_midi, aligned_midi, start_time=0, end_time=20):
    """Visualize MIDI alignment with stable beat grid"""
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
    
    # Plot audio waveform with beat grid
    ax1 = axes[0]
    sample_start = int(start_time * sr)
    sample_end = int(min(end_time * sr, len(y)))
    times = np.arange(sample_start, sample_end) / sr
    
    ax1.plot(times, y[sample_start:sample_end], color='gray', alpha=0.6)
    
    # Color downbeats differently
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

def stable_beat_alignment(song_path, midi_paths, output_path=None, method='plp', dynamic_tempo=False):
    """
    Run complete alignment workflow with beat grid for audio
    
    Parameters:
    -----------
    song_path : path to audio file
    midi_paths : dictionary of midi paths (e.g. {'bass': bass_path, 'harmony': harmony_path})
    output_path : directory to save outputs (optional) - Note: disabled to avoid excessive file creation
    method : str, beat detection method ('plp', 'stable', 'dynamic')
    dynamic_tempo : bool, whether to track tempo changes (only applies to 'plp' method)
    """
    # Step 1: Load audio
    print(f"Loading audio: {song_path}")
    y, sr = librosa.load(song_path)
    print(f"Audio duration: {len(y)/sr:.2f} seconds, Sample rate: {sr} Hz")
    
    # Step 2: Detect beat grid using the selected method
    if method == 'plp':
        # Use our new PLP method
        beats, tempo = detect_beats_plp(y, sr, dynamic_tempo=dynamic_tempo)
        # For compatibility with old code
        beat_offset = 0.0 if len(beats) == 0 else beats[0]
    elif method == 'dynamic':
        # Use dynamic tempo tracking method
        beats, tempo = detect_beats_dynamic(y, sr)
        # For compatibility with old code
        beat_offset = 0.0 if len(beats) == 0 else beats[0]
    else:
        # Use original stable beat detection
        beats, tempo, beat_offset = detect_stable_beats(y, sr)
    
    # Step 3: Load and combine MIDI files
    print("\nLoading and combining MIDI files...")
    combined_midi = pretty_midi.PrettyMIDI()
    
    for midi_name, midi_path in midi_paths.items():
        print(f"Loading {midi_name} MIDI: {midi_path}")
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            combined_midi.instruments.extend(midi_data.instruments)
        except Exception as e:
            print(f"Error loading MIDI file {midi_path}: {str(e)}")
    
    print(f"Combined MIDI has {len(combined_midi.instruments)} instruments")
    
    # Step 4: Align MIDI to beat grid
    aligned_midi = align_midi_to_stable_grid(
        combined_midi, 
        beats, 
        tempo,
        audio_duration=len(y)/sr
    )
    
    # Step 5: Find optimal fine adjustment for MIDI
    midi_offset_ms = find_midi_fine_adjustment(
        y, sr, aligned_midi, max_offset_ms=30, step_ms=2
    )
    
    # Apply the fine adjustment to create final MIDI
    final_midi = align_midi_to_stable_grid(
        combined_midi, 
        beats, 
        tempo,
        audio_duration=len(y)/sr,
        fine_adjust_ms=midi_offset_ms
    )
    
    # Step 6: Create synchronized audio
    print("\nCreating synchronized audio...")
    audio_with_clicks, midi_with_clicks, full_mix = create_synchronized_audio(
        y, sr, beats, final_midi
    )
    
    # Play the results
    print("\nResults:")
    print("1. Original audio with beat grid:")
    display(Audio(data=audio_with_clicks, rate=sr))
    
    print("\n2. Aligned MIDI with beat clicks:")
    display(Audio(data=midi_with_clicks, rate=sr))
    
    print("\n3. Mixed audio (Original + MIDI + clicks):")
    display(Audio(data=full_mix, rate=sr))
    
    # Step 7: Visualize alignment
    print("\nVisualizing alignment...")
    viz_end = min(20, len(y)/sr)
    fig = visualize_midi_alignment(
        y, sr, beats, combined_midi, final_midi, 
        start_time=0, end_time=viz_end
    )
    plt.show()
    
    
    return {
        'beats': beats,
        'tempo': tempo,
        'beat_offset': beat_offset,
        'midi_offset_ms': midi_offset_ms,
        'final_midi': final_midi,
        'audio': {
            'original_with_clicks': audio_with_clicks,
            'midi_with_clicks': midi_with_clicks,
            'full_mix': full_mix
        }
    }

def run_stable_beat_test(song_path, midi_paths, output_path=None, method='plp', dynamic_tempo=False):
    """
    Run the beat grid alignment test
    
    Parameters:
    -----------
    song_path : str
        Path to the audio file
    midi_paths : dict
        Dictionary of MIDI paths (e.g. {'bass': bass_path, 'harmony': harmony_path})
    output_path : str, optional
        Directory to save outputs
    method : str
        Beat detection method ('plp', 'stable', 'dynamic')
    dynamic_tempo : bool
        Whether to track tempo changes (only applies to 'plp' method)
    """
    
    # Run the alignment process
    results = stable_beat_alignment(song_path, midi_paths, output_path, method, dynamic_tempo)
    
    print("\nSummary of Results:")
    print(f"Detected tempo: {results['tempo']:.2f} BPM")
    if 'beat_offset' in results:
        print(f"Beat phase offset: {results['beat_offset']:.4f} seconds")
    print(f"MIDI fine adjustment: {results['midi_offset_ms']:.2f} ms")
    print(f"Number of beats: {len(results['beats'])}")
    
    if output_path:
        print(f"Output files saved to: {output_path}")
    
    return results

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
        Maximum milliseconds to adjust in either direction (positive or negative)
    step_ms : int
        Step size for testing different offsets in milliseconds
    
    Returns:
    --------
    optimal_offset_ms : float
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
    
    # Visualize
    plt.figure(figsize=(12, 5))
    plt.plot(offsets_ms, scores, 'b-', alpha=0.5, label='Raw scores')
    plt.plot(offsets_ms, smoothed_scores, 'r-', linewidth=2, label='Smoothed scores')
    plt.axvline(x=optimal_offset_ms, color='g', linestyle='--')
    plt.xlabel('Fine Adjustment (ms)')
    plt.ylabel('Onset Match Score')
    plt.title(f'MIDI Fine Adjustment (Optimal: {optimal_offset_ms:.2f} ms)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"Optimal MIDI fine adjustment: {optimal_offset_ms:.2f} ms")
    
    return optimal_offset_ms


def direct_midi_alignment(audio_path, midi_paths, target_tempo=None, remove_pitch_bend=True, output_path=None, offset_seconds=0):
    """
    Directly align MIDI to audio by applying a simple time offset.
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file
    midi_paths : dict or str
        Dictionary of MIDI paths or a single MIDI path
    target_tempo : float or None
        Target tempo in BPM. If None, it will be detected from the audio
    remove_pitch_bend : bool
        Whether to remove pitch bend events that may cause issues
    output_path : str, optional
        Directory to save aligned MIDI (if provided)
    offset_seconds : float
        Direct time offset to apply to MIDI notes (positive: delay MIDI, negative: advance MIDI)
        For example, -0.03 would shift all MIDI notes 30ms earlier
    
    Returns:
    --------
    dict
        Dictionary containing alignment results
    """
    # Load audio file
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path)
    audio_duration = len(y) / sr
    print(f"Audio duration: {audio_duration:.2f} seconds")
    
    # Detect beats and tempo from audio if not provided
    if target_tempo is None:
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        target_tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, units='time')
        print(f"Detected audio tempo: {target_tempo:.2f} BPM")
        print(f"Detected {len(beats)} beats")
    else:
        # If tempo is provided, generate a regular beat grid at that tempo
        beat_period = 60.0 / target_tempo  # seconds per beat
        num_beats = int(np.ceil(audio_duration / beat_period))
        beats = np.arange(0, num_beats) * beat_period
        print(f"Using provided tempo: {target_tempo:.2f} BPM")
        print(f"Generated {len(beats)} beats")
    
    # Create click track
    click_times = beats
    click_freqs = []
    for i, _ in enumerate(beats):
        if i % 4 == 0:  # Every 4th beat is a downbeat (assuming 4/4 time)
            click_freqs.append(1760)  # Higher pitch for downbeats
        else:
            click_freqs.append(880)   # Lower pitch for other beats
    
    click_track = create_click_track(click_times, click_freqs, sr, length=len(y))
    
    # Load and process MIDI file(s)
    if isinstance(midi_paths, dict):
        # Multiple MIDI files
        midi_files = {}
        for name, path in midi_paths.items():
            print(f"Loading {name} MIDI: {path}")
            try:
                midi_data = pretty_midi.PrettyMIDI(path)
                
                # Remove pitch bends if requested
                if remove_pitch_bend:
                    for instrument in midi_data.instruments:
                        if len(instrument.pitch_bends) > 0:
                            print(f"  Removing {len(instrument.pitch_bends)} pitch bend events from {instrument.name if instrument.name else 'unnamed'}")
                            instrument.pitch_bends = []
                
                # Store the MIDI data
                midi_files[name] = midi_data
            except Exception as e:
                print(f"Error loading MIDI file {path}: {str(e)}")
    else:
        # Single MIDI file
        midi_files = {"main": None}
        try:
            print(f"Loading MIDI: {midi_paths}")
            midi_data = pretty_midi.PrettyMIDI(midi_paths)
            
            # Remove pitch bends if requested
            if remove_pitch_bend:
                for instrument in midi_data.instruments:
                    if len(instrument.pitch_bends) > 0:
                        print(f"  Removing {len(instrument.pitch_bends)} pitch bend events from {instrument.name if instrument.name else 'unnamed'}")
                        instrument.pitch_bends = []
            
            midi_files["main"] = midi_data
        except Exception as e:
            print(f"Error loading MIDI file {midi_paths}: {str(e)}")
    
    # Process each MIDI file for alignment
    aligned_midis = {}
    original_midis = {}  # Store original MIDI for comparison
    aligned_audios = {}
    original_audios = {}  # Store original MIDI audio for comparison
    
    for name, midi_data in midi_files.items():
        if midi_data is None:
            continue
            
        # Check MIDI tempo
        midi_tempo = midi_data.estimate_tempo()
        if midi_tempo == 0:
            # Try to infer from time signature
            time_sig_changes = midi_data.time_signature_changes
            if len(time_sig_changes) > 0:
                print(f"Found time signature: {time_sig_changes[0].numerator}/{time_sig_changes[0].denominator}")
                # For most MIDIs, the nominal tempo is often 120 BPM if not specified
                midi_tempo = 120.0
            else:
                # Default fallback
                midi_tempo = 120.0
        
        print(f"{name} MIDI original tempo: {midi_tempo:.2f} BPM")
        print(f"Applying offset of {offset_seconds * 1000:.1f} ms to MIDI notes")
        
        # Store original MIDI for comparison
        original_midis[name] = midi_data
        
        # Create aligned MIDI with offset adjustment
        aligned_midi = pretty_midi.PrettyMIDI(initial_tempo=target_tempo)
        
        # Set time signature if present in original
        if len(midi_data.time_signature_changes) > 0:
            for ts in midi_data.time_signature_changes:
                aligned_midi.time_signature_changes.append(ts)
        
        # Copy instruments and adjust note timings with direct offset
        for instrument in midi_data.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name
            )
            
            # Adjust all notes with the offset
            for note in instrument.notes:
                # Apply the direct offset
                start_time = note.start + offset_seconds
                end_time = note.end + offset_seconds
                
                # Create new note with adjusted timing
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=max(0, start_time),  # Ensure we don't have negative start times
                    end=max(0.001, end_time)   # Ensure notes have positive duration
                )
                
                # Add note if it falls within audio duration
                if start_time < audio_duration:
                    new_instrument.notes.append(new_note)
            
            # Add instrument if it has notes
            if len(new_instrument.notes) > 0:
                aligned_midi.instruments.append(new_instrument)
            
        # Store aligned MIDI
        aligned_midis[name] = aligned_midi
        
        # Synthesize the aligned MIDI
        try:
            # Check for common soundfont locations
            sf2_paths = [
                '/usr/share/sounds/sf2/FluidR3_GM.sf2',  # Linux
                '/usr/local/share/soundfonts/default.sf2',  # Other Linux
                'C:\\soundfonts\\FluidR3_GM.sf2',  # Windows
                '/Library/Audio/Sounds/Banks/FluidR3_GM.sf2'  # macOS
            ]
            
            # Find first available soundfont
            sf2_path = None
            for path in sf2_paths:
                if os.path.exists(path):
                    sf2_path = path
                    print(f"Using soundfont: {sf2_path}")
                    break
            
            # Synthesize original MIDI
            if sf2_path:
                original_audio = original_midis[name].fluidsynth(fs=sr, sf2_path=sf2_path)
                midi_audio = aligned_midi.fluidsynth(fs=sr, sf2_path=sf2_path)
            else:
                original_audio = original_midis[name].fluidsynth(fs=sr)
                midi_audio = aligned_midi.fluidsynth(fs=sr)
                
        except Exception as e:
            print(f"FluidSynth error: {str(e)}")
            print("Using simple synthesis fallback...")
            
            # Create simple synthesis fallback for original
            original_audio = np.zeros(len(y))
            for instrument in original_midis[name].instruments:
                for note in instrument.notes:
                    freq = librosa.midi_to_hz(note.pitch)
                    start_sample = int(note.start * sr)
                    end_sample = min(int(note.end * sr), len(original_audio))
                    
                    if end_sample > start_sample:
                        # Generate sine wave for note
                        t = np.arange(end_sample - start_sample) / sr
                        sine_wave = 0.1 * (note.velocity / 127.0) * np.sin(2 * np.pi * freq * t)
                        
                        # Apply envelope
                        envelope = np.exp(-t * 4) if instrument.is_drum else np.exp(-t * 2)
                        sine_wave = sine_wave * envelope
                        
                        # Add to output
                        original_audio[start_sample:end_sample] += sine_wave
            
            # Create simple synthesis fallback for aligned MIDI
            midi_audio = np.zeros(len(y))
            for instrument in aligned_midi.instruments:
                for note in instrument.notes:
                    freq = librosa.midi_to_hz(note.pitch)
                    start_sample = int(note.start * sr)
                    end_sample = min(int(note.end * sr), len(midi_audio))
                    
                    if end_sample > start_sample:
                        # Generate sine wave for note
                        t = np.arange(end_sample - start_sample) / sr
                        sine_wave = 0.1 * (note.velocity / 127.0) * np.sin(2 * np.pi * freq * t)
                        
                        # Apply envelope
                        envelope = np.exp(-t * 4) if instrument.is_drum else np.exp(-t * 2)
                        sine_wave = sine_wave * envelope
                        
                        # Add to output
                        midi_audio[start_sample:end_sample] += sine_wave
        
        # Ensure MIDI audio is mono channel and matches length
        if len(midi_audio.shape) > 1:
            midi_audio = np.mean(midi_audio, axis=1)
        
        if len(midi_audio) > len(y):
            midi_audio = midi_audio[:len(y)]
        else:
            midi_audio = np.pad(midi_audio, (0, max(0, len(y) - len(midi_audio))))
        
        # Do the same for original MIDI audio
        if len(original_audio.shape) > 1:
            original_audio = np.mean(original_audio, axis=1)
        
        if len(original_audio) > len(y):
            original_audio = original_audio[:len(y)]
        else:
            original_audio = np.pad(original_audio, (0, max(0, len(y) - len(original_audio))))
        
        # Store aligned audio
        aligned_audios[name] = midi_audio
        original_audios[name] = original_audio
    
    # Combine all MIDI audio tracks for visualization and playback
    combined_midi_audio = np.zeros_like(y)
    combined_original_audio = np.zeros_like(y)
    
    if aligned_audios:
        for name, audio in aligned_audios.items():
            combined_midi_audio += audio / len(aligned_audios)  # Simple mixing
    
    if original_audios:
        for name, audio in original_audios.items():
            combined_original_audio += audio / len(original_audios)  # Simple mixing
    
    # Normalize audio
    y_normalized = y / np.max(np.abs(y)) * 0.8 if np.max(np.abs(y)) > 0 else y
    midi_normalized = combined_midi_audio / np.max(np.abs(combined_midi_audio)) * 0.8 if np.max(np.abs(combined_midi_audio)) > 0 else combined_midi_audio
    original_normalized = combined_original_audio / np.max(np.abs(combined_original_audio)) * 0.8 if np.max(np.abs(combined_original_audio)) > 0 else combined_original_audio
    
    # Mix with click track
    clicks = click_track * 0.3
    audio_with_clicks = y_normalized + clicks
    midi_with_clicks = midi_normalized + clicks
    original_with_clicks = original_normalized + clicks
    mixed = (y_normalized * 0.6) + (midi_normalized * 0.6) + (clicks * 0.5)
    mixed_original = (y_normalized * 0.6) + (original_normalized * 0.6) + (clicks * 0.5)
    
    # Ensure all are within valid range
    audio_with_clicks = np.clip(audio_with_clicks, -0.9, 0.9)
    midi_with_clicks = np.clip(midi_with_clicks, -0.9, 0.9)
    original_with_clicks = np.clip(original_with_clicks, -0.9, 0.9)
    mixed = np.clip(mixed, -0.9, 0.9)
    mixed_original = np.clip(mixed_original, -0.9, 0.9)
    
    # Make sure sample rate is acceptable
    target_sr = min(sr, 44100)  # Use 44100 as maximum to be safe
    if sr != target_sr:
        from librosa import resample
        audio_with_clicks = resample(audio_with_clicks, orig_sr=sr, target_sr=target_sr)
        midi_with_clicks = resample(midi_with_clicks, orig_sr=sr, target_sr=target_sr)
        original_with_clicks = resample(original_with_clicks, orig_sr=sr, target_sr=target_sr)
        mixed = resample(mixed, orig_sr=sr, target_sr=target_sr)
        mixed_original = resample(mixed_original, orig_sr=sr, target_sr=target_sr)
        playback_sr = target_sr
    else:
        playback_sr = sr
    
    # Add interactive offset adjustment widget if in Jupyter notebook
    try:
        from ipywidgets import interact, FloatSlider, Layout
        import matplotlib.pyplot as plt
        from IPython.display import display, clear_output
        
        is_jupyter = True
    except ImportError:
        is_jupyter = False
    
    if is_jupyter:
        # Create a function to preview different offsets
        def preview_offset(offset_ms=0):
            # Convert to seconds for calculation
            offset_sec = offset_ms / 1000.0
            
            # Clear previous output
            clear_output(wait=True)
            
            # Visualize the alignment
            fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
            
            # Plot audio waveform with beats
            ax1 = axes[0]
            librosa.display.waveshow(y, sr=sr, ax=ax1, alpha=0.6, color='gray')
            ax1.vlines(beats, -1, 1, color='r', alpha=0.7, label='Beats')
            ax1.set_title(f'Audio Waveform with Beat Grid (Tempo: {target_tempo:.2f} BPM)')
            ax1.legend()
            ax1.set_xlim(0, min(20, audio_duration))  # Show first 20 seconds
            
            # Plot original MIDI piano roll
            ax2 = axes[1]
            if original_midis:
                # Use the first MIDI for visualization
                first_midi_name = list(original_midis.keys())[0]
                plot_midi_piano_roll(original_midis[first_midi_name], ax2, 0, min(20, audio_duration), 
                                    title=f'Original MIDI Notes ({first_midi_name})')
            else:
                ax2.set_title('No valid MIDI data')
            
            # Add beat markers to piano roll
            for i, beat in enumerate(beats):
                if beat <= 20:  # Show first 20 seconds
                    if i % 4 == 0:  # Downbeat
                        ax2.axvline(x=beat, color='r', linewidth=1.2, alpha=0.7)
                    else:  # Regular beat
                        ax2.axvline(x=beat, color='b', linewidth=0.8, alpha=0.5)
            
            # Create a temporary aligned MIDI for preview with current offset
            temp_aligned_midi = pretty_midi.PrettyMIDI(initial_tempo=target_tempo)
            
            # Use the first MIDI for visualization
            first_midi_name = list(original_midis.keys())[0]
            original_midi = original_midis[first_midi_name]
            
            # Set time signature if present in original
            if len(original_midi.time_signature_changes) > 0:
                for ts in original_midi.time_signature_changes:
                    temp_aligned_midi.time_signature_changes.append(ts)
            
            # Copy instruments and adjust note timings with direct offset
            for instrument in original_midi.instruments:
                new_instrument = pretty_midi.Instrument(
                    program=instrument.program,
                    is_drum=instrument.is_drum,
                    name=instrument.name
                )
                
                # Adjust all notes with the offset
                for note in instrument.notes:
                    # Apply the direct offset
                    start_time = note.start + offset_sec
                    end_time = note.end + offset_sec
                    
                    # Create new note with adjusted timing
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=max(0, start_time),  # Ensure we don't have negative start times
                        end=max(0.001, end_time)   # Ensure notes have positive duration
                    )
                    
                    # Add note if it falls within audio duration
                    if start_time < audio_duration:
                        new_instrument.notes.append(new_note)
                
                # Add instrument if it has notes
                if len(new_instrument.notes) > 0:
                    temp_aligned_midi.instruments.append(new_instrument)
            
            # Plot aligned MIDI piano roll
            ax3 = axes[2]
            plot_midi_piano_roll(temp_aligned_midi, ax3, 0, min(20, audio_duration), 
                                title=f'Aligned MIDI Notes with {offset_ms:.1f}ms offset')
            
            # Add beat markers to piano roll
            for i, beat in enumerate(beats):
                if beat <= 20:  # Show first 20 seconds
                    if i % 4 == 0:  # Downbeat
                        ax3.axvline(x=beat, color='r', linewidth=1.2, alpha=0.7)
                    else:  # Regular beat
                        ax3.axvline(x=beat, color='b', linewidth=0.8, alpha=0.5)
            
            # Plot waveform comparison
            ax4 = axes[3]
            # Plot original audio
            ax4.plot(np.arange(len(y[:sr*20 if sr*20 < len(y) else len(y)])) / sr, 
                    y[:sr*20 if sr*20 < len(y) else len(y)], 
                    color='gray', alpha=0.5)
            
            # Plot original MIDI audio
            ax4.plot(np.arange(len(combined_original_audio[:sr*20 if sr*20 < len(combined_original_audio) else len(combined_original_audio)])) / sr, 
                    combined_original_audio[:sr*20 if sr*20 < len(combined_original_audio) else len(combined_original_audio)], 
                    color='green', alpha=0.5, label='Original MIDI Audio')
            
            # Plot aligned MIDI audio
            ax4.plot(np.arange(len(combined_midi_audio[:sr*20 if sr*20 < len(combined_midi_audio) else len(combined_midi_audio)])) / sr, 
                    combined_midi_audio[:sr*20 if sr*20 < len(combined_midi_audio) else len(combined_midi_audio)], 
                    color='blue', alpha=0.5, label='Aligned MIDI Audio')
            
            # Add beat markers
            for i, beat in enumerate(beats):
                if beat <= 20:  # Show first 20 seconds
                    if i % 4 == 0:  # Downbeat
                        ax4.axvline(x=beat, color='r', linewidth=1.2, alpha=0.7)
                    else:  # Regular beat
                        ax4.axvline(x=beat, color='b', linewidth=0.8, alpha=0.5)
            
            ax4.set_title(f'Waveform Preview - Current offset: {offset_ms:.1f}ms ({offset_sec:.4f}s)')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Amplitude')
            ax4.legend()
            
            plt.tight_layout()
            plt.show()
            
            print(f"Preview with offset: {offset_ms:.1f}ms ({offset_sec:.4f}s)")
            print("Adjust the slider to find the optimal MIDI offset.")
            print("Once you find the best value, rerun the function with this offset value.")
            
        # Create slider for interactive adjustment
        slider = FloatSlider(
            value=offset_seconds * 1000,  # Convert to ms for display
            min=-100,
            max=100,
            step=1,
            description='Offset (ms):',
            layout=Layout(width='600px')
        )
        
        # Display interactive widget
        print("\nMIDI Fine Adjustment:")
        print("Adjust the offset to align MIDI notes with audio beats")
        interact(preview_offset, offset_ms=slider)
    
    # Visualize the alignment
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
    
    # Plot audio waveform with beats
    ax1 = axes[0]
    librosa.display.waveshow(y, sr=sr, ax=ax1, alpha=0.6, color='gray')
    ax1.vlines(beats, -1, 1, color='r', alpha=0.7, label='Beats')
    ax1.set_title(f'Audio Waveform with Beat Grid (Tempo: {target_tempo:.2f} BPM)')
    ax1.legend()
    ax1.set_xlim(0, min(20, audio_duration))  # Show first 20 seconds
    
    # Plot original MIDI piano roll
    ax2 = axes[1]
    if original_midis:
        # Use the first MIDI for visualization
        first_midi_name = list(original_midis.keys())[0]
        plot_midi_piano_roll(original_midis[first_midi_name], ax2, 0, min(20, audio_duration), 
                           title=f'Original MIDI Notes ({first_midi_name})')
    else:
        ax2.set_title('No valid MIDI data')
    
    # Add beat markers to piano roll
    for i, beat in enumerate(beats):
        if beat <= 20:  # Show first 20 seconds
            if i % 4 == 0:  # Downbeat
                ax2.axvline(x=beat, color='r', linewidth=1.2, alpha=0.7)
            else:  # Regular beat
                ax2.axvline(x=beat, color='b', linewidth=0.8, alpha=0.5)
    
    # Plot aligned MIDI piano roll
    ax3 = axes[2]
    if aligned_midis:
        # Use the first MIDI for visualization
        first_midi_name = list(aligned_midis.keys())[0]
        plot_midi_piano_roll(aligned_midis[first_midi_name], ax3, 0, min(20, audio_duration), 
                           title=f'Aligned MIDI Notes with {offset_seconds*1000:.1f}ms offset ({first_midi_name})')
    else:
        ax3.set_title('No valid aligned MIDI data')
    
    # Add beat markers to piano roll
    for i, beat in enumerate(beats):
        if beat <= 20:  # Show first 20 seconds
            if i % 4 == 0:  # Downbeat
                ax3.axvline(x=beat, color='r', linewidth=1.2, alpha=0.7)
            else:  # Regular beat
                ax3.axvline(x=beat, color='b', linewidth=0.8, alpha=0.5)
    
    # Plot waveform comparison
    ax4 = axes[3]
    # Plot original audio
    ax4.plot(np.arange(len(y[:sr*20 if sr*20 < len(y) else len(y)])) / sr, 
             y[:sr*20 if sr*20 < len(y) else len(y)], 
             color='gray', alpha=0.5, label='Original Audio')
    
    # Plot original MIDI audio
    ax4.plot(np.arange(len(combined_original_audio[:sr*20 if sr*20 < len(combined_original_audio) else len(combined_original_audio)])) / sr, 
            combined_original_audio[:sr*20 if sr*20 < len(combined_original_audio) else len(combined_original_audio)], 
            color='green', alpha=0.5, label='Original MIDI Audio')
    
    # Plot aligned MIDI audio
    ax4.plot(np.arange(len(combined_midi_audio[:sr*20 if sr*20 < len(combined_midi_audio) else len(combined_midi_audio)])) / sr, 
            combined_midi_audio[:sr*20 if sr*20 < len(combined_midi_audio) else len(combined_midi_audio)], 
            color='blue', alpha=0.5, label='Aligned MIDI Audio')
    
    # Add beat markers
    for i, beat in enumerate(beats):
        if beat <= 20:  # Show first 20 seconds
            if i % 4 == 0:  # Downbeat
                ax4.axvline(x=beat, color='r', linewidth=1.2, alpha=0.7)
            else:  # Regular beat
                ax4.axvline(x=beat, color='b', linewidth=0.8, alpha=0.5)
    
    ax4.set_title(f'Waveform Comparison (Offset: {offset_seconds*1000:.1f}ms)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Play the results
    print("\nAudio with beats:")
    display(Audio(data=audio_with_clicks, rate=playback_sr))
    
    print("\nOriginal MIDI with beats:")
    display(Audio(data=original_with_clicks, rate=playback_sr))
    
    print("\nAligned MIDI with beats:")
    display(Audio(data=midi_with_clicks, rate=playback_sr))
    
    print("\nMixed audio (Original + original MIDI + beats):")
    display(Audio(data=mixed_original, rate=playback_sr))
    
    print("\nMixed audio (Original + aligned MIDI + beats):")
    display(Audio(data=mixed, rate=playback_sr))
    
    # Save MIDI files if output path is provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        for name, midi in aligned_midis.items():
            output_file = os.path.join(output_path, f"{name}_aligned_offset_{int(offset_seconds*1000)}ms.mid")
            midi.write(output_file)
            print(f"Saved aligned MIDI: {output_file}")
            
            # Also save original for comparison
            output_file_orig = os.path.join(output_path, f"{name}_original.mid")
            original_midis[name].write(output_file_orig)
            print(f"Saved original MIDI: {output_file_orig}")
    
    return {
        'tempo': target_tempo,
        'beats': beats,
        'original_midis': original_midis,
        'aligned_midis': aligned_midis,
        'audio': {
            'audio_with_clicks': audio_with_clicks,
            'midi_with_clicks': midi_with_clicks,
            'original_with_clicks': original_with_clicks,
            'mixed': mixed,
            'mixed_original': mixed_original
        },
        'offset_seconds': offset_seconds,
        'sample_rate': playback_sr
    }