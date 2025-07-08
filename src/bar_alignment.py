"""
Bar Alignment Script with Self-Similarity Matrix
KTH Royal Institute of Technology
AIMIR Project

This script aligns bars with structural boundaries detected using self-similarity matrix analysis.
It can be run independently without requiring class modifications.
"""

"""
This version uses a simpler approach that avoids the problematic visualization 
code causing errors with Matplotlib/Librosa interaction.
"""
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy.sparse
import scipy.linalg
import scipy.ndimage
from sklearn.cluster import KMeans

def optimize_bar_alignment(y, sr, section_boundaries, beats=None, tempo=None, visualize=True):
    """Find the optimal alignment of bars with structural section boundaries."""
    # Step 1: Get beat and tempo information if not provided
    if beats is None or tempo is None:
        # Use harmonic-percussive source separation for better beat tracking
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Onset strength using the percussive component for better beat detection
        oenv = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        
        if tempo is None:
            # Estimate tempo from percussive component
            tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr)[0]
            print(f"Estimated tempo: {tempo:.1f} BPM")
        
        if beats is None:
            # Get beat frames based on the estimated tempo
            _, beat_frames = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, trim=False, start_bpm=tempo)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    else:
        beat_times = beats
    
    # Step 2: Calculate the theoretical bar length based on tempo (assuming 4/4 time signature)
    seconds_per_beat = 60.0 / tempo
    seconds_per_bar = 4 * seconds_per_beat  # 4 beats per bar
    
    # Step 3: Calculate downbeat likelihood using spectral flux at beat positions
    # This helps identify which beats are more likely to be downbeats (start of bars)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_diff = np.diff(mfcc, axis=1)  # Calculate frame-to-frame differences
    
    # Get MFCC values at beat positions
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    beat_frames = np.minimum(beat_frames, mfcc.shape[1] - 1)  # Ensure within bounds
    
    # Calculate spectral flux (sum of squared differences) between consecutive beats
    beat_flux = np.zeros(len(beat_frames))
    for i in range(1, len(beat_frames)):
        if i < len(beat_frames):
            beat_flux[i] = np.sum(np.square(mfcc[:, beat_frames[i]] - mfcc[:, beat_frames[i-1]]))
    
    # Normalize and find peaks in beat flux (potential downbeats)
    beat_flux = (beat_flux - np.min(beat_flux)) / (np.max(beat_flux) - np.min(beat_flux) + 1e-8)
    
    # Step 4: Try different bar offsets and score each
    possible_offsets = np.arange(4)  # Try offsets 0, 1, 2, 3 beats
    offset_scores = np.zeros(4)
    detailed_scores = []
    
    duration = librosa.get_duration(y=y, sr=sr)
    
    # For each possible offset, evaluate alignment with structural boundaries
    for offset_idx, offset in enumerate(possible_offsets):
        # Apply offset to bar times
        offset_seconds = offset * seconds_per_beat
        bar_times = np.arange(offset_seconds, duration + seconds_per_bar, seconds_per_bar)
        
        # Calculate alignment with each structural boundary
        boundary_scores = []
        
        for boundary_time in section_boundaries:
            # Skip boundaries outside of our audio time
            if boundary_time <= 0 or boundary_time >= duration:
                continue
                
            # Find distance to the nearest bar
            distances = np.abs(bar_times - boundary_time)
            min_distance = np.min(distances)
            nearest_bar_idx = np.argmin(distances)
            
            # Calculate score based on distance (exponential decay, closer is better)
            # Penalize more as distance increases, normalized by bar length
            alignment_score = np.exp(-min_distance / (0.15 * seconds_per_bar))  # 0.15 is tolerance factor
            
            # Check if this boundary aligns with a downbeat (using beat flux)
            nearest_beat_idx = int(round((bar_times[nearest_bar_idx] - beat_times[0]) / seconds_per_beat))
            if 0 <= nearest_beat_idx < len(beat_flux):
                # Boost score if the boundary aligns with a likely downbeat
                alignment_score *= (1.0 + 0.5 * beat_flux[nearest_beat_idx])
                
            boundary_scores.append({
                'boundary_time': boundary_time,
                'nearest_bar': bar_times[nearest_bar_idx],
                'distance': min_distance,
                'score': alignment_score
            })
        
        # Calculate overall score for this offset
        if boundary_scores:
            # Average score across all boundaries
            mean_score = np.mean([s['score'] for s in boundary_scores])
            
            # Add a factor based on how evenly the sections are distributed in bars
            # (prefer sections that align with even number of bars)
            bar_counts = []
            for i in range(len(section_boundaries) - 1):
                start_bar = np.searchsorted(bar_times, section_boundaries[i])
                end_bar = np.searchsorted(bar_times, section_boundaries[i+1])
                bar_count = end_bar - start_bar
                bar_counts.append(bar_count)
            
            # Check if sections tend to be even numbers of bars (musically common)
            evenness_factor = np.mean([1.0 if bc % 2 == 0 else 0.8 for bc in bar_counts]) if bar_counts else 1.0
            
            # Final score is a combination of alignment and evenness
            offset_scores[offset_idx] = mean_score * evenness_factor
            detailed_scores.append({
                'offset': offset,
                'mean_score': mean_score,
                'evenness_factor': evenness_factor,
                'final_score': offset_scores[offset_idx],
                'boundary_scores': boundary_scores
            })
    
    # Find best offset
    best_offset_idx = np.argmax(offset_scores)
    best_offset = possible_offsets[best_offset_idx]
    best_score = offset_scores[best_offset_idx]
    
    # Get the final bar positions with the optimal offset
    offset_seconds = best_offset * seconds_per_beat
    aligned_bar_times = np.arange(offset_seconds, duration + seconds_per_bar, seconds_per_bar)
    
    # Create bar positions data structure
    bar_positions = []
    for i, bar_start in enumerate(aligned_bar_times):
        if bar_start < duration:
            # Calculate beats in this bar
            bar_beats = []
            for beat_offset in range(4):
                beat_time = bar_start + beat_offset * seconds_per_beat
                if beat_time < duration:
                    bar_beats.append(beat_time)
            
            bar_positions.append({
                'bar_number': i + 1,
                'start_time': bar_start,
                'beats': bar_beats
            })
    
    # Skip visualization if there are issues
    if visualize:
        try:
            # Simple visualization to show the offset scores
            plt.figure(figsize=(8, 4))
            plt.bar(possible_offsets, offset_scores, color='blue')
            plt.xlabel('Beat Offset')
            plt.ylabel('Alignment Score')
            plt.title(f'Bar Alignment Scores (Best Offset: {best_offset}, Score: {best_score:.3f})')
            plt.xticks(possible_offsets)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
            
            # Print detailed analysis
            print(f"Bar alignment analysis complete.")
            print(f"Optimal offset: {best_offset} beats ({best_offset * seconds_per_beat:.2f} seconds)")
            print(f"Tempo: {tempo:.1f} BPM, Bar length: {seconds_per_bar:.2f} seconds")
            print(f"Alignment score: {best_score:.3f}")
            
            # Print section info
            print("\nSection analysis:")
            print("-" * 60)
            print(f"{'Section':<10}{'Start (s)':<12}{'End (s)':<12}{'Duration (s)':<12}{'Bars':<10}")
            print("-" * 60)
            
            for i in range(len(section_boundaries) - 1):
                start = section_boundaries[i]
                end = section_boundaries[i+1]
                duration = end - start
                
                # Find bar indices for this section
                start_bar_idx = np.searchsorted(aligned_bar_times, start)
                end_bar_idx = np.searchsorted(aligned_bar_times, end)
                bar_count = end_bar_idx - start_bar_idx
                
                print(f"{i+1:<10}{start:<12.2f}{end:<12.2f}{duration:<12.2f}{bar_count:<10}")
        
        except Exception as e:
            print(f"Visualization error: {e}")
            print("Analysis completed but visualization failed.")
    
    # Return the results
    return {
        'optimal_offset': best_offset,
        'optimal_offset_seconds': best_offset * seconds_per_beat,
        'tempo': tempo,
        'seconds_per_beat': seconds_per_beat,
        'seconds_per_bar': seconds_per_bar,
        'aligned_bar_times': aligned_bar_times,
        'bar_positions': bar_positions,
        'alignment_score': best_score,
        'detailed_scores': detailed_scores
    }


def analyze_musical_structure(y, sr, k=4, min_duration=3.5, visualize=True):
    """Comprehensive musical structure analysis that integrates form extraction and bar alignment."""
    # Step 1: Perform the structure segmentation analysis
    # Compute CQT for better harmonic representation
    BINS_PER_OCTAVE = 12 * 3
    N_OCTAVES = 7
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr, 
                                               bins_per_octave=BINS_PER_OCTAVE, 
                                               n_bins=N_OCTAVES * BINS_PER_OCTAVE)), 
                            ref=np.max)
    
    # Beat tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    print(f"Detected tempo: {tempo:.1f} BPM")
    
    # Synchronize features to beats
    Csync = librosa.util.sync(C, beats, aggregate=np.median)
    
    # Compute recurrence matrix
    R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity', sym=True)
    
    # Apply median filter to enhance diagonal structures
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))
    
    # Compute MFCC for timbral features
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    Msync = librosa.util.sync(mfcc, beats)
    
    # Compute path similarity
    path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
    
    # Compute balanced combination
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)
    A = mu * Rf + (1 - mu) * R_path
    
    # Compute normalized Laplacian
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    
    # Compute spectral decomposition
    evals, evecs = scipy.linalg.eigh(L)
    
    # Clean up eigenvectors with median filter
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
    
    # Compute cumulative normalization
    Cnorm = np.cumsum(evecs**2, axis=1)**0.5
    
    # Use k components for clustering
    X = evecs[:, :k] / Cnorm[:, k-1:k]
    
    # Fix NaN values
    if np.isnan(X).any() or np.isinf(X).any():
        X = np.nan_to_num(X)
    
    # Cluster beats into segments
    KM = KMeans(n_clusters=k, n_init=10)
    seg_ids = KM.fit_predict(X)
    
    # Find segment boundaries
    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])
    bound_beats = np.concatenate(([0], bound_beats))
    
    # Convert to frames and times
    bound_frames = beats[bound_beats]
    bound_times = librosa.frames_to_time(bound_frames, sr=sr)
    
    # Add the end of the song
    if bound_times[-1] < librosa.get_duration(y=y, sr=sr):
        bound_times = np.append(bound_times, librosa.get_duration(y=y, sr=sr))
    
    # Get segment labels
    bound_segs = list(seg_ids[bound_beats])
    if len(bound_segs) < len(bound_times):
        bound_segs.append(bound_segs[-1])
    
    # Enforce minimum segment duration
    if min_duration > 0:
        # Merge segments shorter than min_duration
        i = 1
        while i < len(bound_times) - 1:
            if bound_times[i] - bound_times[i-1] < min_duration:
                # Remove this boundary
                bound_times = np.delete(bound_times, i)
                bound_segs.pop(i)
            else:
                i += 1
    
    # Map segment labels to sequential numbers
    unique_labels = []
    for label in bound_segs:
        if label not in unique_labels:
            unique_labels.append(label)
    
    label_mapping = {old: new+1 for new, old in enumerate(unique_labels)}
    new_bound_segs = [label_mapping[label] for label in bound_segs]
    
    # Step 2: Optimize bar alignment with section boundaries
    alignment_results = optimize_bar_alignment(
        y=y, 
        sr=sr, 
        section_boundaries=bound_times, 
        beats=beat_times,
        tempo=tempo, 
        visualize=visualize
    )
    
    # Step 3: Prepare final results
    results = {
        'tempo': tempo,
        'section_boundaries': bound_times,
        'section_labels': new_bound_segs,
        'bar_alignment': alignment_results
    }
    
    return results


def format_bars_for_form_extractor(bar_positions):
    """
    Convert the optimized bar positions from the bar alignment algorithm
    to the format expected by formExtractor.
    
    Parameters:
    -----------
    bar_positions : list of dict
        Bar positions from the bar alignment algorithm
        
    Returns:
    --------
    list of list
        Bars in the format expected by formExtractor (list of beat times)
    """
    bars = []
    for bar in bar_positions:
        bars.append(bar['beats'])
    return bars


def integrate_with_form_extractor(form_extractor, audio_path, k=4, min_duration=3.5):
    """
    Integrate the bar alignment algorithm with the formExtractor class.
    
    Parameters:
    -----------
    form_extractor : formExtractor
        An instance of the formExtractor class
    audio_path : str
        Path to the audio file
    k : int, optional
        Number of clusters for segmentation
    min_duration : float, optional
        Minimum duration of segments in seconds
        
    Returns:
    --------
    dict
        Updated data dictionary with properly aligned bars and sections
    """
    # First get data using the formExtractor
    form_extractor.getData(audio_path)
    y, sr = form_extractor.y, form_extractor.sr
    
    # Run the musical structure analysis
    results = analyze_musical_structure(y, sr, k=k, min_duration=min_duration, visualize=True)
    
    # Format the bars for formExtractor
    formatted_bars = format_bars_for_form_extractor(results['bar_alignment']['bar_positions'])
    
    # Replace the bars in the formExtractor instance
    form_extractor.bars = formatted_bars
    
    # Convert the section boundaries to frames
    bound_frames = librosa.time_to_frames(results['section_boundaries'], sr=sr)
    
    # Update the data dictionary
    data_dict = form_extractor.populateDict(
        sr=sr,
        chords=form_extractor.chords,
        bars=formatted_bars,
        bound_frames=bound_frames,
        bound_segs=results['section_labels']
    )
    
    return data_dict