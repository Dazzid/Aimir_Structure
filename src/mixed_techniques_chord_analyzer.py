import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from scipy import stats as scipy_stats
import music21
import re

#----------------------------------------------------------------------------------------
# Function to correct a single chord name
def correct_chord_name(chord_name, preferred_spellings, enharmonic_map):
    # If the chord name is just a root note (like 'G#')
    if len(chord_name) == 1 or (len(chord_name) == 2 and chord_name[1] in ['#', 'b']):
        if chord_name in preferred_spellings:
            return preferred_spellings[chord_name]
        return chord_name
    
    # If the chord name has a quality (like 'G#maj7', 'Bbm')
    for note, enharmonic in enharmonic_map.items():
        if chord_name.startswith(note):
            # Replace only the root note, keeping the rest of the chord name
            if note in preferred_spellings:
                return preferred_spellings[note] + chord_name[len(note):]
    
    return chord_name
    
def correct_chord_name(chord_name, preferred_spellings, scale_notes):
    """
    Corrects the root note of the chord name using preferred_spellings, but only if in the scale.
    Returns the chord name with the corrected root.
    """
    import re
    # Match root note at the start, e.g., C, Bb, F#, Eb...
    match = re.match(r'^([A-G][b#]?)(.*)$', chord_name)
    if not match:
        return chord_name  # Nothing to correct
    root, rest = match.groups()
    # Only correct if in the scale
    if root in scale_notes and root in preferred_spellings:
        root = preferred_spellings[root]
    return root + rest

def correct_enharmonics(chord_data, tonality, alterations, scale):
    """
    Corrects chord names to use the appropriate enharmonic spelling based on the key,
    but only for notes actually in the scale (or alterations).
    Preserves the original object types (ChordChange or dict).
    """
    enharmonic_map = {
        'C#': 'Db', 'Db': 'C#',
        'D#': 'Eb', 'Eb': 'D#',
        'F#': 'Gb', 'Gb': 'F#',
        'G#': 'Ab', 'Ab': 'G#',
        'A#': 'Bb', 'Bb': 'A#',
        'E#': 'F', 'F': 'E#',
        'B#': 'C', 'C': 'B#',
        'Cb': 'B', 'B': 'Cb'
    }

    scale_notes = set(scale)
    alteration_notes = set(alterations)

    preferred_spellings = {}

    # Only make mappings for notes actually in the scale (or alterations)
    for sharp_note, flat_note in [('C#', 'Db'), ('D#', 'Eb'), ('F#', 'Gb'), 
                                 ('G#', 'Ab'), ('A#', 'Bb')]:
        if sharp_note in scale_notes:
            preferred = sharp_note
        elif flat_note in scale_notes:
            preferred = flat_note
        elif flat_note in alteration_notes:
            preferred = flat_note
        elif sharp_note in alteration_notes:
            preferred = sharp_note
        else:
            preferred = flat_note if "minor" in tonality.lower() else sharp_note
        # Set mapping only for notes in scale or alterations
        if sharp_note in scale_notes or sharp_note in alteration_notes:
            preferred_spellings[sharp_note] = preferred
        if flat_note in scale_notes or flat_note in alteration_notes:
            preferred_spellings[flat_note] = preferred

    corrected_data = []
    # Determine object type: ChordChange or dict
    if chord_data and hasattr(chord_data[0], 'chord') and hasattr(chord_data[0], 'timestamp'):
        for chord_obj in chord_data:
            corrected_chord = type(chord_obj)(
                chord=correct_chord_name(chord_obj.chord, preferred_spellings, scale_notes | alteration_notes),
                timestamp=chord_obj.timestamp
            )
            corrected_data.append(corrected_chord)
    elif chord_data and isinstance(chord_data[0], dict):
        for chord_dict in chord_data:
            corrected_chord = chord_dict.copy()
            if 'chord_name' in chord_dict:
                corrected_chord['chord_name'] = correct_chord_name(
                    chord_dict['chord_name'], preferred_spellings, scale_notes | alteration_notes)
            if 'root' in chord_dict:
                corrected_chord['root'] = correct_chord_name(
                    chord_dict['root'], preferred_spellings, scale_notes | alteration_notes)
            corrected_data.append(corrected_chord)
    else:
        # Return the data unchanged if unknown structure
        return chord_data

    return corrected_data


#----------------------------------------------------------------------------------------
def compare_chord_extractions(chords_midi, chords_audio, max_time=120, figsize=(15, 10)):
    """
    Create a focused comparison visualization of two chord extraction methods
    with quantitative metrics to determine which method is better
    
    Args:
        chords_midi: List of dictionaries from MIDI extraction method
        chords_audio: List of ChordChange objects from audio extraction method
        max_time: Maximum time to display in seconds
        figsize: Figure size as tuple (width, height)
    """
    # Convert the data to pandas DataFrames for easier manipulation
    
    # MIDI chords (method 1)
    midi_df = pd.DataFrame(chords_midi)
    midi_df = midi_df[midi_df['timestamp'] <= max_time]
    midi_df['method'] = 'MIDI'
    
    # Audio chords (method 2)
    audio_data = [{'chord_name': c.chord, 'timestamp': c.timestamp, 'method': 'Audio'} 
                 for c in chords_audio if c.timestamp <= max_time]
    audio_df = pd.DataFrame(audio_data)
    
    # Create a figure with two subplots: timeline and timing differences
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Main comparison timeline (top plot)
    midi_y = 1.0  # y-position for MIDI chords
    audio_y = 0.0  # y-position for Audio chords
    
    # Plot MIDI chord markers
    for _, row in midi_df.iterrows():
        # Plot vertical line for chord change
        ax1.axvline(x=row['timestamp'], ymin=0.48, ymax=0.52, color='blue', alpha=0.3, linewidth=1)
        # Plot the chord name
        ax1.text(row['timestamp'], midi_y + 0.05, row['chord_name'], 
                 rotation=90, ha='left', va='bottom', fontsize=8, alpha=0.7)
        # Plot confidence level if available
        if 'confidence' in row:
            # Vary the size based on confidence
            size = 40 * row['confidence']
            ax1.scatter(row['timestamp'], midi_y, s=size, color='blue', alpha=0.7)
        else:
            ax1.scatter(row['timestamp'], midi_y, s=30, color='blue', alpha=0.7)
    
    # Plot Audio chord markers
    for _, row in audio_df.iterrows():
        # Plot vertical line for chord change
        ax1.axvline(x=row['timestamp'], ymin=0.48, ymax=0.52, color='red', alpha=0.3, linewidth=1)
        # Plot the chord name
        ax1.text(row['timestamp'], audio_y - 0.05, row['chord_name'], 
                rotation=90, ha='left', va='top', fontsize=8, alpha=0.7)
        ax1.scatter(row['timestamp'], audio_y, s=30, color='red', alpha=0.7)
    
    # Set up the main timeline plot
    ax1.set_yticks([audio_y, midi_y])
    ax1.set_yticklabels(['Audio Method', 'MIDI Method'])
    ax1.set_xlabel('Time (seconds)')
    ax1.set_title('Chord Extraction Methods Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_time)
    ax1.set_ylim(-0.5, 1.5)
    
    # Find matches and differences between methods
    def find_closest_chord(target_time, chords_list, time_field='timestamp', max_dist=0.5):
        """Find the closest chord in the list to the target time"""
        closest = None
        min_dist = float('inf')
        
        for chord in chords_list:
            time = chord[time_field] if isinstance(chord, dict) else getattr(chord, time_field)
            dist = abs(time - target_time)
            if dist < min_dist and dist <= max_dist:
                min_dist = dist
                closest = chord
                
        return closest, min_dist
    
    # Calculate matching statistics
    exact_matches = 0
    similar_matches = 0
    timing_matches = 0
    no_matches = 0
    
    # Lists to store matched pairs for analysis
    midi_matched_times = []
    audio_matched_times = []
    time_diffs = []
    match_types = []  # 2=exact, 1=similar, 0=timing only
    
    for _, row in midi_df.iterrows():
        closest_audio, dist = find_closest_chord(row['timestamp'], chords_audio)
        
        if closest_audio:
            midi_chord = row['chord_name']
            audio_chord = closest_audio.chord
            
            # Record times for correlation analysis
            if dist <= 0.5:  # Only consider reasonably close matches
                midi_matched_times.append(row['timestamp'])
                audio_matched_times.append(closest_audio.timestamp)
                time_diffs.append(closest_audio.timestamp - row['timestamp'])
            
            # Simplify chords for comparison (remove extensions)
            midi_base = midi_chord.rstrip('maj7').rstrip('7').rstrip('sus4').rstrip('5').rstrip('9').rstrip('m')
            audio_base = audio_chord.rstrip('maj7').rstrip('7').rstrip('sus4').rstrip('5').rstrip('9').rstrip('m')
            
            if midi_chord == audio_chord:
                exact_matches += 1
                # Mark exact matches with a green line
                if dist <= 0.5:  # Only if they're reasonably close in time
                    ax1.plot([row['timestamp'], closest_audio.timestamp], 
                             [midi_y, audio_y], 'g-', alpha=0.5, linewidth=1.5)
                    match_types.append(2)  # Exact match
            elif midi_base == audio_base:
                similar_matches += 1
                # Mark similar matches with a yellow line
                if dist <= 0.5:
                    ax1.plot([row['timestamp'], closest_audio.timestamp], 
                             [midi_y, audio_y], 'y-', alpha=0.5, linewidth=1.5)
                    match_types.append(1)  # Similar match
            elif dist <= 0.3:  # Different chords but very close in time
                timing_matches += 1
                # Mark timing matches with a gray line
                ax1.plot([row['timestamp'], closest_audio.timestamp], 
                         [midi_y, audio_y], 'gray', alpha=0.3, linewidth=1)
                match_types.append(0)  # Timing only match
            else:
                no_matches += 1
        else:
            no_matches += 1
    
    # Add a custom legend for the match lines
    custom_lines = [
        Line2D([0], [0], color='g', lw=2, alpha=0.7),
        Line2D([0], [0], color='y', lw=2, alpha=0.7),
        Line2D([0], [0], color='gray', lw=2, alpha=0.5)
    ]
    ax1.legend(custom_lines, ['Exact Match', 'Similar Chord', 'Timing Match'], loc='upper right')
    
    # Add match statistics to the title
    total = exact_matches + similar_matches + timing_matches + no_matches
    ax1.set_title(f'Chord Extraction Comparison\nExact: {exact_matches}/{total} ({exact_matches/total:.1%}), '
                 f'Similar: {similar_matches}/{total} ({similar_matches/total:.1%}), '
                 f'Timing: {timing_matches}/{total} ({timing_matches/total:.1%})')
    
    # Calculate correlation between matched timestamps
    correlation = 0
    if len(midi_matched_times) >= 2:
        correlation, p_value = scipy_stats.pearsonr(midi_matched_times, audio_matched_times)
    
    # Calculate timing accuracy metrics
    time_diffs_abs = [abs(diff) for diff in time_diffs]
    mean_abs_diff = np.mean(time_diffs_abs) if time_diffs_abs else 0
    median_abs_diff = np.median(time_diffs_abs) if time_diffs_abs else 0
    std_diff = np.std(time_diffs) if time_diffs else 0
    
    # Plot timing differences in bottom plot
    if time_diffs:
        # Make sure all arrays have the same length before creating the DataFrame
        min_len = min(len(midi_matched_times), len(time_diffs), len(match_types))
        
        # Create a DataFrame with the information for plotting (ensuring same length for all arrays)
        df_plot = pd.DataFrame({
            'midi_time': midi_matched_times[:min_len],
            'time_diff': time_diffs[:min_len],
            'match_type': match_types[:min_len]
        })
        df_plot = df_plot.sort_values('midi_time')
        
        # Plot different match types with different colors
        exact_matches_df = df_plot[df_plot['match_type'] == 2]
        similar_matches_df = df_plot[df_plot['match_type'] == 1]
        timing_matches_df = df_plot[df_plot['match_type'] == 0]
        
        ax2.scatter(exact_matches_df['midi_time'], exact_matches_df['time_diff'], 
                    color='green', alpha=0.6, label='Exact Match')
        ax2.scatter(similar_matches_df['midi_time'], similar_matches_df['time_diff'], 
                    color='yellow', alpha=0.6, label='Similar Match')
        ax2.scatter(timing_matches_df['midi_time'], timing_matches_df['time_diff'], 
                    color='gray', alpha=0.6, label='Timing Only')
        
        # Add trend line
        if len(time_diffs) > 1:
            z = np.polyfit(midi_matched_times, time_diffs, 1)
            p = np.poly1d(z)
            ax2.plot(sorted(midi_matched_times), p(sorted(midi_matched_times)), 
                    "r--", alpha=0.6, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        
        # Add zero line and formatting
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlim(0, max_time)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Time Difference (s)\nAudio - MIDI')
        ax2.set_title(f'Timing Differences: Mean Abs={mean_abs_diff:.3f}s, Median={median_abs_diff:.3f}s, StdDev={std_diff:.3f}s')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
    
    # Method evaluation metrics
    # 1. Proportion of exact/similar matches of total MIDI chords
    midi_match_rate = (exact_matches + similar_matches) / len(midi_df) if len(midi_df) > 0 else 0
    
    # 2. Proportion of exact/similar matches of total Audio chords
    audio_match_rate = (exact_matches + similar_matches) / len(audio_df) if len(audio_df) > 0 else 0
    
    # 3. F1 score between methods (harmonic mean of precision and recall)
    f1_score = 2 * (midi_match_rate * audio_match_rate) / (midi_match_rate + audio_match_rate) if (midi_match_rate + audio_match_rate) > 0 else 0
    
    # 4. Method comparison summary
    conclusion = ""
    if f1_score > 0.7:
        conclusion = "Methods show high agreement, both likely accurate"
    elif f1_score > 0.5:
        conclusion = "Methods show moderate agreement"
    else:
        conclusion = "Methods show significant disagreement"
        
    if len(midi_df) > len(audio_df) * 1.3:
        conclusion += "\nMIDI method detects more chords"
    elif len(audio_df) > len(midi_df) * 1.3:
        conclusion += "\nAudio method detects more chords"
    
    if mean_abs_diff < 0.2:
        conclusion += "\nExcellent timing precision between methods"
    elif mean_abs_diff < 0.4:
        conclusion += "\nGood timing precision between methods"
    else:
        conclusion += "\nPoor timing alignment between methods"
    
    # Create a text box with evaluation metrics
    # eval_text = (
    #     f"Method Evaluation Metrics:\n"
    #     f"Correlation: {correlation:.3f}\n"
    #     f"F1 Score: {f1_score:.3f}\n"
    #     f"MIDI Match Rate: {midi_match_rate:.3f}\n"
    #     f"Audio Match Rate: {audio_match_rate:.3f}\n"
    #     f"Mean Time Difference: {mean_abs_diff:.3f}s\n\n"
    #     f"Chord Counts:\n"
    #     f"MIDI: {len(midi_df)} chords\n"
    #     f"Audio: {len(audio_df)} chords\n\n"
    #     f"SUMMARY:\n{conclusion}"
    # )
    
    # Add the evaluation text to the main plot
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax1.text(0.02, 0.3, eval_text, transform=ax1.transAxes, fontsize=9, verticalalignment='top', bbox=props)
    
    # Tighten layout and show
    plt.tight_layout()
    plt.show()
    
    # Return comprehensive evaluation metrics
    return {
        'exact_matches': exact_matches,
        'similar_matches': similar_matches,
        'timing_matches': timing_matches,
        'no_matches': no_matches,
        'total_midi_chords': len(midi_df),
        'total_audio_chords': len(audio_df),
        'midi_match_rate': midi_match_rate,
        'audio_match_rate': audio_match_rate,
        'f1_score': f1_score,
        'correlation': correlation,
        'mean_absolute_time_difference': mean_abs_diff,
        'median_time_difference': median_abs_diff,
        'std_time_difference': std_diff,
        'conclusion': conclusion
    }
    
#----------------------------------------------------------------------------------------
def analyze_chord_progression_segments(chords_midi, chords_audio, max_time=120, segment_size=10, figsize=(15, 6)):
    """
    Analyze chord progressions by segments to identify where methods agree or disagree
    
    Args:
        chords_midi: List of dictionaries from MIDI extraction method
        chords_audio: List of ChordChange objects from audio extraction method
        max_time: Maximum time to display in seconds
        segment_size: Size of each analysis segment in seconds
        figsize: Figure size as tuple (width, height)
    """
    # Filter data by max_time
    midi_data = [c for c in chords_midi if c['timestamp'] <= max_time]
    audio_data = [c for c in chords_audio if c.timestamp <= max_time]
    
    # Create segments
    num_segments = int(np.ceil(max_time / segment_size))
    segment_bounds = [(i * segment_size, min((i + 1) * segment_size, max_time)) 
                      for i in range(num_segments)]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 1]})
    
    # Count chords and matches in each segment
    segment_centers = [sum(bounds)/2 for bounds in segment_bounds]
    midi_counts = np.zeros(num_segments)
    audio_counts = np.zeros(num_segments)
    match_counts = np.zeros(num_segments)
    
    # Count MIDI chords in each segment
    for chord in midi_data:
        segment_idx = min(int(chord['timestamp'] / segment_size), num_segments - 1)
        midi_counts[segment_idx] += 1
    
    # Count Audio chords in each segment
    for chord in audio_data:
        segment_idx = min(int(chord.timestamp / segment_size), num_segments - 1)
        audio_counts[segment_idx] += 1
    
    # Find matches in each segment
    for midi_chord in midi_data:
        segment_idx = min(int(midi_chord['timestamp'] / segment_size), num_segments - 1)
        segment_start, segment_end = segment_bounds[segment_idx]
        
        # Get Audio chords in this segment
        segment_audio_chords = [c for c in audio_data 
                              if segment_start <= c.timestamp < segment_end]
        
        # Check for matches
        for audio_chord in segment_audio_chords:
            if (abs(midi_chord['timestamp'] - audio_chord.timestamp) < 0.5 and
                (midi_chord['chord_name'] == audio_chord.chord or
                 midi_chord['chord_name'].rstrip('maj7').rstrip('7').rstrip('sus4').rstrip('5').rstrip('9').rstrip('m') == 
                 audio_chord.chord.rstrip('maj7').rstrip('7').rstrip('sus4').rstrip('5').rstrip('9').rstrip('m'))):
                match_counts[segment_idx] += 1
                break
    
    # Plot segment chord counts
    ax1.bar(segment_centers, midi_counts, width=segment_size*0.35, alpha=0.7, 
           color='blue', label='MIDI Chords')
    ax1.bar([c + segment_size*0.35/2 for c in segment_centers], audio_counts, 
           width=segment_size*0.35, alpha=0.7, color='red', label='Audio Chords')
    
    # Calculate match rate by segment
    match_rate = np.zeros(num_segments)
    for i in range(num_segments):
        if midi_counts[i] > 0:
            match_rate[i] = match_counts[i] / midi_counts[i]
    
    # Plot match rate line on separate axis
    ax1_rate = ax1.twinx()
    ax1_rate.plot(segment_centers, match_rate, 'g-', marker='o', linewidth=2, label='Match Rate')
    ax1_rate.set_ylim(0, 1.1)
    ax1_rate.set_ylabel('Match Rate', color='g')
    ax1_rate.tick_params(axis='y', labelcolor='g')
    
    # Add legend with all items
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_rate.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Set up the segment analysis plot
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Chord Count')
    ax1.set_title('Method Comparison by Segment')
    ax1.set_xlim(0, max_time)
    ax1.grid(True, alpha=0.3)
    
    # Plot chord density difference
    density_diff = midi_counts - audio_counts
    colors = ['blue' if x >= 0 else 'red' for x in density_diff]
    ax2.bar(segment_centers, density_diff, width=segment_size*0.6, alpha=0.6, color=colors)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Set up the difference plot
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('MIDI - Audio\nChord Count')
    ax2.set_title('Method Difference by Segment')
    ax2.set_xlim(0, max_time)
    ax2.grid(True, alpha=0.3)
    
    # Identify problematic segments (low match rate or high difference)
    problem_segments = []
    for i, (start, end) in enumerate(segment_bounds):
        if (match_rate[i] < 0.5 and midi_counts[i] > 2) or abs(density_diff[i]) > 5:
            problem_segments.append((start, end, match_rate[i], density_diff[i]))
            # Highlight problematic segment
            ax1.axvspan(start, end, alpha=0.2, color='red')
            ax2.axvspan(start, end, alpha=0.2, color='red')
    
    # Create a text box with analysis results
    avg_match_rate = np.mean(match_rate[midi_counts > 0]) if any(midi_counts > 0) else 0
    
    analysis_text = (
        f"Segment Analysis:\n"
        f"Average Match Rate: {avg_match_rate:.2f}\n"
        f"Segments with significant disagreement: {len(problem_segments)}\n\n"
    )
    
    if problem_segments:
        analysis_text += "Problem segments (seconds):\n"
        for start, end, rate, diff in problem_segments[:3]:  # Show top 3 problem segments
            analysis_text += f"{start:.1f}-{end:.1f}: match rate={rate:.2f}, diff={diff:.1f}\n"
        if len(problem_segments) > 3:
            analysis_text += f"...and {len(problem_segments)-3} more\n"
    
    # Add methodology recommendation
    if avg_match_rate > 0.7:
        analysis_text += "\nRECOMMENDATION:\nBoth methods show good agreement overall."
        if any(abs(diff) > 3 for _, _, _, diff in problem_segments):
            analysis_text += "\nConsider combining methods for best results."
        else:
            if np.sum(midi_counts) > np.sum(audio_counts):
                analysis_text += "\nMIDI method appears to be more detailed."
            else:
                analysis_text += "\nAudio method appears to be more detailed."
    else:
        if np.sum(midi_counts) > np.sum(audio_counts) * 1.3:
            analysis_text += "\nRECOMMENDATION:\nMIDI method detects more chords but verify accuracy."
        elif np.sum(audio_counts) > np.sum(midi_counts) * 1.3:
            analysis_text += "\nRECOMMENDATION:\nAudio method detects more chords but verify accuracy."
        else:
            analysis_text += "\nRECOMMENDATION:\nManual review needed - methods show significant differences."
    
    # Add the analysis text to the first plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.98, analysis_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'segment_bounds': segment_bounds,
        'midi_counts': midi_counts.tolist(),
        'audio_counts': audio_counts.tolist(),
        'match_counts': match_counts.tolist(),
        'match_rate': match_rate.tolist(),
        'problem_segments': problem_segments,
        'avg_match_rate': avg_match_rate
    }

#----------------------------------------------------------------------------------------
def enhance_midi_chords_with_audio(chords_midi, chords_audio, time_threshold=0.5, consistency_threshold=0.6):
    """
    Enhance the MIDI chords with more detailed information from audio chords
    when there's a consistent correlation pattern.
    
    Args:
        chords_midi: List of dictionaries from MIDI extraction method
        chords_audio: List of ChordChange objects from audio extraction method
        time_threshold: Maximum time difference to consider chords correlated (seconds)
        consistency_threshold: Minimum ratio of consistent matches to consider a pattern
    
    Returns:
        List of enhanced MIDI chord dictionaries
    """
    # Create a copy of the MIDI chords to avoid modifying the original
    enhanced_chords = [chord.copy() for chord in chords_midi]
    
    # Create a mapping of simple chord roots to potential enhancements
    root_to_extensions = {}
    
    # Step 1: Create a dictionary to track all possible extensions for each root
    for midi_chord in chords_midi:
        midi_time = midi_chord['timestamp']
        midi_name = midi_chord['chord_name']
        
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
        midi_name = chord['chord_name']
        if midi_name in consistent_extensions:
            # Replace with the enhanced chord name
            enhanced_chords[i]['chord_name'] = consistent_extensions[midi_name]['enhanced_chord']
            # Add metadata about the enhancement
            enhanced_chords[i]['enhancement_data'] = consistent_extensions[midi_name]
            enhanced_chords[i]['original_chord'] = midi_name
    
    # Return the enhanced chords and the identified patterns
    return enhanced_chords, consistent_extensions

#------------------------------------------------------------------------------------------
# Function to visualize the chord enhancements
def visualize_chord_enhancements(chords_midi, enhanced_chords, consistent_extensions, figsize=(15, 8)):
    """
    Visualize the chord enhancements applied to MIDI chords
    
    Args:
        chords_midi: Original MIDI chord dictionaries
        enhanced_chords: Enhanced MIDI chord dictionaries
        consistent_extensions: Dictionary of consistent extension patterns
        figsize: Figure size as tuple (width, height)
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Create a figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Extract timestamps from the original and enhanced chords
    timestamps = [chord['timestamp'] for chord in chords_midi]
    
    # Define y-positions
    original_y = 0.0
    enhanced_y = 1.0
    
    # Plot original MIDI chords
    for i, chord in enumerate(chords_midi):
        ax1.scatter(chord['timestamp'], original_y, color='blue', alpha=0.7, s=30)
        ax1.text(chord['timestamp'], original_y - 0.05, chord['chord_name'], 
                rotation=90, ha='right', va='top', fontsize=8, alpha=0.7)
    
    # Plot enhanced MIDI chords
    for i, chord in enumerate(enhanced_chords):
        # Determine if this chord was enhanced
        was_enhanced = 'original_chord' in chord
        color = 'green' if was_enhanced else 'blue'
        
        ax1.scatter(chord['timestamp'], enhanced_y, color=color, alpha=0.7, s=30)
        ax1.text(chord['timestamp'], enhanced_y + 0.05, chord['chord_name'], 
                rotation=90, ha='left', va='bottom', fontsize=8, alpha=0.7)
        
        # Draw a line connecting original to enhanced if changed
        if was_enhanced:
            ax1.plot([chord['timestamp'], chord['timestamp']], 
                    [original_y, enhanced_y], color='green', alpha=0.5, linewidth=1)
    
    # Set up the main plot
    ax1.set_yticks([original_y, enhanced_y])
    ax1.set_yticklabels(['Original MIDI', 'Enhanced MIDI'])
    ax1.set_xlabel('Time (seconds)')
    ax1.set_title('MIDI Chord Enhancement Visualization')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min(timestamps) - 1, max(timestamps) + 1)
    ax1.set_ylim(-0.5, 1.5)
    
    # Create a table of enhancement patterns in the second subplot
    if consistent_extensions:
        patterns_data = []
        for midi_chord, info in consistent_extensions.items():
            patterns_data.append({
                'Original': midi_chord,
                'Enhanced': info['enhanced_chord'],
                'Consistency': f"{info['consistency']:.0%}",
                'Occurrences': f"{info['count']}/{info['total']}"
            })
        
        # Create a pandas DataFrame for the table
        df = pd.DataFrame(patterns_data)
        
        # Hide the subplot's axes
        ax2.axis('off')
        
        # Create a table
        table = ax2.table(
            cellText=df.values,
            colLabels=df.columns,
            loc='center',
            cellLoc='center',
            colWidths=[0.25, 0.25, 0.2, 0.2]
        )
        
        # Customize the table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        ax2.set_title('Consistent Chord Enhancement Patterns')
    else:
        ax2.text(0.5, 0.5, 'No consistent enhancement patterns found', 
                 ha='center', va='center', fontsize=12)
        ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    enhanced_count = sum(1 for chord in enhanced_chords if 'original_chord' in chord)
    total_count = len(chords_midi)
    print(f"Enhanced {enhanced_count} out of {total_count} chords ({enhanced_count/total_count:.1%})")
    print(f"Found {len(consistent_extensions)} consistent enhancement patterns")

#----------------------------------------------------------------------------------------
def print_enhancement_report(consistent_extensions):
    """
    Print a detailed report of the chord enhancement patterns
    
    Args:
        consistent_extensions: Dictionary of consistent extension patterns
    """
    if not consistent_extensions:
        print("No consistent enhancement patterns found.")
        return
    
    print("\n==== CHORD ENHANCEMENT PATTERNS ====")
    print(f"Found {len(consistent_extensions)} consistent patterns:\n")
    
    # Sort patterns by consistency (highest first)
    sorted_patterns = sorted(
        consistent_extensions.items(), 
        key=lambda x: x[1]['consistency'], 
        reverse=True
    )
    
    for midi_chord, info in sorted_patterns:
        print(f"Original: {midi_chord:8} → Enhanced: {info['enhanced_chord']:8} "
              f"(Consistency: {info['consistency']:.1%}, Occurrences: {info['count']}/{info['total']})")
    
    print("\n==== ENHANCEMENT SUMMARY ====")
    
    # Group by chord type
    chord_types = {}
    for midi_chord, info in consistent_extensions.items():
        # Extract only the extension part (e.g., '7', 'maj7', etc.)
        midi_base = midi_chord.rstrip('maj7').rstrip('7').rstrip('sus4').rstrip('5').rstrip('9').rstrip('m').rstrip('dim')
        extension = info['enhanced_chord'][len(midi_base):]
        
        if extension not in chord_types:
            chord_types[extension] = 0
        chord_types[extension] += 1
    
    print("Common extension patterns:")
    for ext, count in sorted(chord_types.items(), key=lambda x: x[1], reverse=True):
        ext_display = ext if ext else "(no extension)"
        print(f"  {ext_display}: {count} chords")
    
    print("\n==== RECOMMENDATION ====")
    avg_consistency = sum(info['consistency'] for _, info in consistent_extensions.items()) / len(consistent_extensions)
    
    if avg_consistency > 0.8:
        print("High consistency in enhancements - recommended to use enhanced chords")
    elif avg_consistency > 0.6:
        print("Moderate consistency - enhanced chords are likely beneficial but verify important sections")
    else:
        print("Low consistency - use enhanced chords with caution and manual verification")

#----------------------------------------------------------------------------------------
def convert_to_music21_notation(chord_string):
    """
    Converts standard chord notation with 'b' for flats to music21's notation with '-' for flats.
    
    Args:
        chord_string: A chord name like 'Bb', 'Ebm7', 'Abmaj7'
        
    Returns:
        The chord name with music21 notation
    """
    # First handle double flats (bb) - must be done before single flats
    chord_string = chord_string.replace('bb', '--')
    
    # Regex to find a note letter followed by a flat symbol
    pattern = r'([A-G])b'
    
    # Replace each occurrence with the note followed by '-'
    result = re.sub(pattern, r'\1-', chord_string)
    
    return result

#----------------------------------------------------------------------------------------
def separate_roman_numeral(rn_figure):
    """
    Separates a Roman numeral figure into base function and alterations/extensions.
    
    Args:
        rn_figure: The Roman numeral figure from music21 (e.g., 'V7', '#II#7#5#3')
        
    Returns:
        Tuple of (base_function, alterations)
    """
    # Extract the core Roman numeral using regex
    base_match = re.match(r'([#b]*)([ivIV]+)', rn_figure)
    
    if not base_match:
        return rn_figure, ""  # Return original if no match
    
    # The base function includes accidentals and the Roman numeral
    accidental = base_match.group(1)  # Any leading # or b
    roman = base_match.group(2)       # The Roman numeral itself
    base_function = accidental + roman
    
    # Get everything after the Roman numeral as alterations
    start_pos = len(base_function)
    alterations = rn_figure[start_pos:] if start_pos < len(rn_figure) else ""
    
    return base_function, alterations

#----------------------------------------------------------------------------------------
def extract_functional_chords(enhanced_chords, tonality='C'):
    """
    Creates a new array with functional chords using music21,
    keeping beat_start, beat_end, duration, and timestamp, and
    separating functional harmony into base function and alterations.
    
    Args:
        enhanced_chords: List of chord dictionaries
        tonality: Key of the song (e.g., 'C', 'Fm', 'F minor')
        
    Returns:
        List of dictionaries with functional harmony information
    """
    # Properly parse the tonality
    tonic = None
    mode = 'major'
    
    # Handle different formats: 'F minor', 'F Minor', 'Fm', etc.
    if isinstance(tonality, str):
        # Convert flats in tonality to music21 notation
        tonality = convert_to_music21_notation(tonality)
        
        tonality = tonality.strip()
        # Check for formats like 'F minor'
        if ' ' in tonality:
            parts = tonality.split()
            tonic = parts[0]
            mode = parts[1].lower()
        # Check for formats like 'Fm'
        elif tonality.endswith('m') and len(tonality) > 1:
            tonic = tonality[:-1]
            mode = 'minor'
        # Standard format like 'F'
        else:
            tonic = tonality
    
    # Create a key object for analysis
    try:
        k = music21.key.Key(tonic, mode)
        # print(f"Successfully created key: {k}")
    except Exception as e:
        print(f"Error creating key from {tonality}: {str(e)}")
        #rise a notification of the error
        assert False, f"Invalid tonality: {tonality}"
    
    functional_chords = []
    
    for chord in enhanced_chords:
        # Extract the requested fields
        functional_chord = {
            'beat_start': chord['beat_start'],
            'beat_end': chord['beat_end'],
            'duration': chord['duration'],
            'timestamp': chord['timestamp']
        }
        
        # Use music21 to get functional harmony
        try:
            # Convert chord_name to music21 notation
            original_chord_name = chord['chord_name']
            chord_name_m21 = convert_to_music21_notation(original_chord_name)
            
            # Parse the chord with music21 notation
            c = music21.harmony.ChordSymbol(chord_name_m21)
            
            # Get Roman numeral representation
            rn = music21.roman.romanNumeralFromChord(c, k)
            
            # Separate the functional harmony into base function and alterations
            base_function, alterations = separate_roman_numeral(rn.figure)
            
            # Replace chord_name with separate functional components
            functional_chord['functional'] = base_function
            functional_chord['alterations'] = alterations
            
        except Exception as e:
            # If music21 can't parse the chord, use placeholders and log the error
            functional_chord['functional'] = 'Unknown'
            functional_chord['alterations'] = ''
            print(f"Error analyzing chord {original_chord_name} (converted to {chord_name_m21}): {str(e)}")
        
        functional_chords.append(functional_chord)
    
    return functional_chords