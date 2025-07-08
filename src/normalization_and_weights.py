
import numpy as np
import copy
from typing import List, Dict, Any


def compute_chord_weights(chords_data):
    """
    Computes the weight for each chord based on its duration in beats.
    
    Args:
        chords_data: list of dicts, each representing a chord with 'beat_idx' and 'beat_idx_end'
    
    Returns:
        List of dicts, each with original chord info plus a 'weight' key.
    """
    weighted_chords = []
    for chord in chords_data:
        start = chord['beat_idx']
        end = chord['beat_idx_end']
        # Weight is number of beats the chord is present (inclusive)
        weight = (end - start) + 1
        # Copy chord data and add 'weight'
        chord_weighted = copy.deepcopy(chord)
        chord_weighted['weight'] = weight
        weighted_chords.append(chord_weighted)
    return weighted_chords


def find_temporal_gaps(chords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find temporal gaps between chords and create placeholder chords for them.
    
    Args:
        chords: List of chord dictionaries sorted by beat_start
        
    Returns:
        List of gap chord dictionaries
    """
    gaps = []
    
    for i in range(len(chords) - 1):
        current_end = chords[i]['beat_end']
        next_start = chords[i + 1]['beat_start']
        
        # Check if there's a gap (allowing small tolerance for floating point precision)
        if next_start - current_end > 0.001:
            # Create gap chord with all required fields
            gap_chord = {
                'beat_start': current_end,
                'beat_end': next_start,
                'beat_idx': chords[i]['beat_idx_end'] + 1,
                'beat_idx_end': chords[i + 1]['beat_idx'] - 1,
                'duration': next_start - current_end,
                'weight': 1,
                'is_temporal_gap': True,  # Flag to identify gaps
                # Musical content fields - empty but present
                'chord_name': '',
                'functional': '',
                'chord_type': '',
                'root_name': '',
                'roman_numeral': '',
                'root_pc': None,
                'intervals': [],
                'interval_durations': {},
                'pitch_classes': set(),
                'notes': [],
                'functional_harmony': {'functional': '', 'alterations': '', 'roman_numeral': ''}
            }
            gaps.append(gap_chord)
    
    return gaps

def insert_gaps_and_sort(chords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Insert temporal gap chords and sort all chords by beat_start.
    
    Args:
        chords: Original list of chord dictionaries
        
    Returns:
        Combined list with gaps inserted and sorted by time
    """
    gaps = find_temporal_gaps(chords)
    # if gaps:
        # print(f"Found {len(gaps)} temporal gaps")
    
    # Combine original chords and gaps, then sort by beat_start
    all_chords = chords + gaps
    all_chords.sort(key=lambda x: x['beat_start'])
    
    return all_chords

def is_empty_chord(chord: Dict[str, Any]) -> bool:
    """
    Check if a chord represents an empty beat (temporal gap or no musical information).
    
    Args:
        chord: Chord dictionary to check
        
    Returns:
        True if the chord is empty (temporal gap or no meaningful musical content)
    """
    # Check if it's explicitly marked as a temporal gap
    if chord.get('is_temporal_gap', False):
        return True
    
    # Check various indicators of empty/missing musical content
    empty_indicators = [
        not chord.get('chord_name') or chord.get('chord_name') == '',
        not chord.get('functional') or chord.get('functional') == '',
        not chord.get('root_name') or chord.get('root_name') == '',
        len(chord.get('notes', [])) == 0,
        len(chord.get('pitch_classes', set())) == 0,
        len(chord.get('intervals', [])) == 0
    ]
    
    # If most indicators suggest emptiness, consider it empty
    return sum(empty_indicators) >= 4

def merge_isolated_chords(chords_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Iteratively merge isolated chords (weight=1) with their dominant neighbors
    until no isolated chords remain.
    
    Args:
        chords_data: List of chord dictionaries with timing and harmony information
        
    Returns:
        List of merged chord dictionaries
    """
    # Work with a deep copy to avoid modifying original data
    chords = copy.deepcopy(chords_data)
    
    # First, insert temporal gaps as placeholder chords
    chords = insert_gaps_and_sort(chords)
    
    iteration = 0
    max_iterations = 100  # Safety limit to prevent infinite loops
    
    while iteration < max_iterations:
        iteration += 1
        # print(f"\n--- Iteration {iteration} ---")
        # print(f"Starting with {len(chords)} chords")
        
        # Step 1: Find and translate isolated chords
        translations_made = 0
        
        for i in range(len(chords)):
            # Check for isolated chords (weight=1) OR empty beats (no musical content)
            is_isolated = chords[i]['weight'] == 1
            is_empty_beat = is_empty_chord(chords[i])
            
            if is_isolated or is_empty_beat:
                # Found an isolated chord or empty beat
                prev_neighbor = None
                next_neighbor = None
                
                # Check previous neighbor
                if i > 0:
                    prev_neighbor = chords[i-1]
                    
                # Check next neighbor
                if i < len(chords) - 1:
                    next_neighbor = chords[i+1]
                
                # Determine which neighbor to use as dominant
                dominant_neighbor = None
                
                if prev_neighbor and next_neighbor:
                    # Both neighbors exist
                    prev_has_content = not is_empty_chord(prev_neighbor) and prev_neighbor['weight'] > 1
                    next_has_content = not is_empty_chord(next_neighbor) and next_neighbor['weight'] > 1
                    
                    if prev_has_content and next_has_content:
                        # Both have content and weight > 1, choose previous (left) neighbor
                        dominant_neighbor = prev_neighbor
                    elif prev_has_content:
                        dominant_neighbor = prev_neighbor
                    elif next_has_content:
                        dominant_neighbor = next_neighbor
                    # If both are empty or weight=1, skip (will be caught later)
                    
                elif prev_neighbor and not is_empty_chord(prev_neighbor) and prev_neighbor['weight'] > 1:
                    dominant_neighbor = prev_neighbor
                elif next_neighbor and not is_empty_chord(next_neighbor) and next_neighbor['weight'] > 1:
                    dominant_neighbor = next_neighbor
                
                # Translate isolated chord to dominant neighbor's identity
                if dominant_neighbor:
                    # Copy harmonic identity fields while preserving timing
                    harmonic_fields = ['chord_name', 'functional', 'chord_type', 'root_name', 
                                     'roman_numeral', 'functional_harmony', 'root_pc', 
                                     'intervals', 'pitch_classes']
                    
                    for field in harmonic_fields:
                        if field in dominant_neighbor:
                            chords[i][field] = copy.deepcopy(dominant_neighbor[field])
                    
                    # Remove temporal gap flag
                    chords[i].pop('is_temporal_gap', None)
                    
                    translations_made += 1
        
        # print(f"  Made {translations_made} translations")
        
        # Step 2: Merge consecutive chords with same identity
        merged_chords = []
        current_group = None
        
        for chord in chords:
            # Use harmonic fields for identity comparison
            chord_identity = (
                chord.get('chord_name', ''),
                chord.get('functional', ''),
                chord.get('chord_type', ''),
                chord.get('root_name', ''),
                chord.get('roman_numeral', ''),
                tuple(sorted(chord.get('intervals', []))),
                frozenset(chord.get('pitch_classes', set()))
            )
            
            if current_group is None:
                # Start new group
                current_group = {
                    'chords': [chord],
                    'identity': chord_identity
                }
            elif current_group['identity'] == chord_identity:
                # Add to current group
                current_group['chords'].append(chord)
            else:
                # Finalize current group and start new one
                merged_chords.append(merge_chord_group(current_group['chords']))
                current_group = {
                    'chords': [chord],
                    'identity': chord_identity
                }
        
        # Don't forget the last group
        if current_group:
            merged_chords.append(merge_chord_group(current_group['chords']))
        
        # print(f"  After merging: {len(merged_chords)} chords")
        
        # Check if we have any isolated chords or empty beats left
        isolated_count = sum(1 for chord in merged_chords if chord['weight'] == 1 or is_empty_chord(chord))
        # print(f"  Isolated/empty chords remaining: {isolated_count}")
        
        chords = merged_chords
        
        # Stop if no isolated chords or empty beats remain
        if isolated_count == 0:
            # print(f"\nConverged after {iteration} iterations!")
            break
            
        # Also stop if no translations were made (stuck in infinite loop)
        if translations_made == 0:
            # print(f"\nNo more translations possible. Stopping after {iteration} iterations.")
            # print(f"Final isolated/empty chords: {isolated_count}")
            break
    
    return chords

def merge_chord_group(chord_group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge a group of consecutive identical chords into a single chord.
    
    Args:
        chord_group: List of chord dictionaries with the same identity
        
    Returns:
        Single merged chord dictionary
    """
    if len(chord_group) == 1:
        return chord_group[0]
    
    # Use the first chord as the base and preserve ALL its fields
    merged = copy.deepcopy(chord_group[0])
    
    # Update timing information
    merged['beat_start'] = chord_group[0]['beat_start']
    merged['beat_end'] = chord_group[-1]['beat_end']
    merged['beat_idx'] = chord_group[0]['beat_idx']
    merged['beat_idx_end'] = chord_group[-1]['beat_idx_end']
    
    # Update duration
    merged['duration'] = merged['beat_end'] - merged['beat_start']
    
    # Update weight (sum of all weights)
    merged['weight'] = sum(chord['weight'] for chord in chord_group)
    
    # Remove temporal gap flag if it exists (merged chord is no longer a gap)
    merged.pop('is_temporal_gap', None)
    
    # For interval_durations, merge them if they exist
    if 'interval_durations' in merged:
        all_interval_durations = {}
        for chord in chord_group:
            if 'interval_durations' in chord:
                all_interval_durations.update(chord['interval_durations'])
        merged['interval_durations'] = all_interval_durations
    
    # For intervals, combine unique intervals
    if 'intervals' in merged:
        all_intervals = set()
        for chord in chord_group:
            if 'intervals' in chord:
                all_intervals.update(chord['intervals'])
        merged['intervals'] = sorted(list(all_intervals))
    
    # For pitch_classes, combine unique pitch classes
    if 'pitch_classes' in merged:
        all_pitch_classes = set()
        for chord in chord_group:
            if 'pitch_classes' in chord and chord['pitch_classes']:
                all_pitch_classes.update(chord['pitch_classes'])
        merged['pitch_classes'] = all_pitch_classes
    
    # For notes, combine unique notes but keep as string list
    if 'notes' in merged:
        all_notes = set()
        for chord in chord_group:
            if 'notes' in chord:
                all_notes.update(chord['notes'])
        merged['notes'] = sorted(list(all_notes))
    
    return merged