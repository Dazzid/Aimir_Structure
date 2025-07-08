import librosa
import numpy as np
from collections import namedtuple

class TriadExtractor:
    def __init__(self, hop_length=512, scale=['C', 'D', 'E', 'F', 'G', 'A', 'B']):
        self.hop_length = hop_length
        self.maj_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        self.min_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        self.N_template = np.full(12, 1/4)
        self.labels_sharp = [
            'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
            'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm', 'N'
        ]
        self.labels_flat = [
            'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
            'Cm', 'Dbm', 'Dm', 'Ebm', 'Em', 'Fm', 'Gbm', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm', 'N'
        ]
        self.labels = self.labels_sharp if any(note in self.labels_sharp for note in scale) else self.labels_flat
        self.weights = self._generate_weights()
        self.trans = librosa.sequence.transition_loop(25, 0.99)

    def _generate_weights(self):
        weights = np.zeros((25, 12))
        for c in range(12):
            weights[c] = np.roll(self.maj_template, c)
            weights[c + 12] = np.roll(self.min_template, c)
        weights[-1] = self.N_template
        return weights

    def extract_chords(self, song_path, threshold=0.3, check_on_beat=False):
        y, sr = librosa.load(song_path)
        y = librosa.effects.harmonic(y, margin=1)
        chroma = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=self.hop_length, n_chroma=12
        )
        chroma = librosa.util.normalize(chroma, norm=1, axis=0)

        # Compute the onset envelope once
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.hop_length
        )

        if check_on_beat:
            beats = self.get_beats(onset_env, sr)
            # Calculate average beat duration for the last beat estimation
            if len(beats) > 1:
                avg_beat_duration = np.mean(np.diff(beats))
            else:
                tempo = librosa.beat.tempo(
                    onset_envelope=onset_env, sr=sr, hop_length=self.hop_length
                )
                if tempo.size > 0 and tempo[0] > 0:
                    avg_beat_duration = 60.0 / tempo[0]
                else:
                    avg_beat_duration = 0.5  # Fallback value
            chord_progression = self.extract_chords_on_beat(chroma, sr, beats)
        else:
            chord_progression = self.extract_chords_viterbi(chroma, sr)
            avg_beat_duration = 0.5  # Default value when not checking on beats

        # Use average beat duration for seventh chord analysis
        window_range = int((avg_beat_duration * sr) / self.hop_length)
        updated_cp = self._check_for_sevenths(
            chroma, chord_progression, sr, window_range, threshold
        )
        ChordChange = namedtuple('ChordChange', ['chord', 'timestamp'])
        return [
            ChordChange(chord=chord, timestamp=timestamp)
            for chord, timestamp in updated_cp
        ]

    def extract_chords_on_beat(self, chroma, sr, beats):
        chord_progression = []
        previous_chord = None

        for i in range(len(beats) - 1):
            start_time = beats[i]
            end_time = beats[i + 1]
            frame_start = librosa.time_to_frames(start_time, sr=sr, hop_length=self.hop_length)
            frame_end = librosa.time_to_frames(end_time, sr=sr, hop_length=self.hop_length)

            chroma_window = chroma[:, frame_start:frame_end]
            if chroma_window.size == 0:
                continue

            # Compute emission probabilities
            probs = np.exp(self.weights.dot(chroma_window))
            probs /= probs.sum(axis=0, keepdims=True)

            # Apply Viterbi algorithm to the beat window
            chords_vit = librosa.sequence.viterbi_discriminative(probs, self.trans)

            # Determine the chord for the beat (most frequent chord)
            chord_counts = np.bincount(chords_vit, minlength=len(self.labels))
            chord_index = np.argmax(chord_counts)
            beat_chord = self.labels[chord_index]

            if beat_chord != previous_chord:
                chord_progression.append((beat_chord, start_time))

            previous_chord = beat_chord


        return chord_progression

    def extract_chords_viterbi(self, chroma, sr):
        # Apply Viterbi algorithm to the entire song
        probs = np.exp(self.weights.dot(chroma))
        probs /= probs.sum(axis=0, keepdims=True)
        chords_vit = librosa.sequence.viterbi_discriminative(probs, self.trans)
        times = librosa.frames_to_time(
            np.arange(len(chords_vit)), sr=sr, hop_length=self.hop_length
        )
        chord_progression = [
            (self.labels[chord], float(time)) for chord, time in zip(chords_vit, times)
        ]
        # Remove consecutive duplicates
        return [
            x for i, x in enumerate(chord_progression)
            if i == 0 or x[0] != chord_progression[i - 1][0]
        ]

    def _check_for_sevenths(self, chroma, chord_progression, sr, window_range, threshold):
        updated_cp = []
        for chord, timestamp in chord_progression:
            if 'N' not in chord:
                chord_root = chord.rstrip('7').rstrip('m')
                is_minor = 'm' in chord and 'maj' not in chord
                root_index = self.labels.index(chord_root) % 12
                seventh_index = (root_index + 10) % 12
                major_seventh_index = (root_index + 11) % 12

                frame_index = librosa.time_to_frames(
                    timestamp, sr=sr, hop_length=self.hop_length
                )
                frame_start = max(0, frame_index - window_range // 2)
                frame_end = min(chroma.shape[1], frame_index + window_range // 2)
                chroma_window = chroma[:, frame_start:frame_end]
                avg_chroma = np.mean(chroma_window, axis=1)

                if avg_chroma[seventh_index] > threshold:
                    if is_minor:
                        chord += '7'
                    elif avg_chroma[major_seventh_index] > threshold:
                        chord += 'maj7'
                    else:
                        chord += '7'
                elif 'dim' in chord:
                    chord += '⦰7' if avg_chroma[seventh_index] > threshold else 'dim7'
            updated_cp.append((chord, timestamp))
        return updated_cp

    def get_beats(self, onset_env, sr):
        # Detect beat frames
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=self.hop_length
        )
        # Convert frames to times
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)

        # If the first beat is significantly after 0.0, insert a beat at 0.0
        if beat_times.size == 0 or beat_times[0] > 0.1:
            beat_times = np.insert(beat_times, 0, 0.0)

        return beat_times
    
    def getTonality(self, song, print_all=False):
        """
        Analyze tonality with detailed probability output
        
        Parameters:
        -----------
        song : str
            Path to the audio file
        print_all : bool
            Whether to print all tonality probabilities
            
        Returns:
        --------
        str
            Detected key (e.g., "D major" or "B minor")
        """

        
        y, sr = librosa.load(song)
        
        # Compute the chroma features using CQT
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Sum chroma features over time to emphasize prominent pitches
        chroma_vector = np.sum(chroma_cq, axis=1)
        chroma_vector /= np.linalg.norm(chroma_vector)
        
        # Modified Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Normalize the profiles
        major_profile /= np.linalg.norm(major_profile)
        minor_profile /= np.linalg.norm(minor_profile)
        
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                    'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        correlations = []
        for i in range(12):
            # Rotate the key profiles
            major_profile_rotated = np.roll(major_profile, i)
            minor_profile_rotated = np.roll(minor_profile, i)
            
            # Compute the correlation with adjusted weights
            major_corr = np.dot(chroma_vector, major_profile_rotated) * 1.1
            minor_corr = np.dot(chroma_vector, minor_profile_rotated) * 1.1  # Increase minor influence
            
            correlations.append({
                'key': key_names[i],
                'mode': 'major',
                'correlation': major_corr
            })
            correlations.append({
                'key': key_names[i],
                'mode': 'minor',
                'correlation': minor_corr
            })
        
        # Sort correlations for better visibility
        sorted_correlations = sorted(correlations, key=lambda x: x['correlation'], reverse=True)
        
        if print_all:
            # Calculate max correlation for normalization
            max_corr = sorted_correlations[0]['correlation']
            
            # Print all correlations and relative probabilities
            for i, corr_data in enumerate(sorted_correlations):
                # Calculate a probability-like value by normalizing to the max
                probability = corr_data['correlation'] / max_corr
                
                # Highlight the top result
                highlight = " ← SELECTED" if i == 0 else ""
                
                # Check for relative major/minor pairs
                rel_note = ""
                if i > 0:  # Skip for the top result
                    if corr_data['mode'] == 'major':
                        # Find relative minor (3 semitones down or 9 up)
                        rel_minor_idx = (key_names.index(corr_data['key']) + 9) % 12
                        rel_minor = f"{key_names[rel_minor_idx]} minor"
                        # Check if it's in top 5
                        for j, top_corr in enumerate(sorted_correlations[:5]):
                            if top_corr['key'] == key_names[rel_minor_idx] and top_corr['mode'] == 'minor':
                                rel_note = f" (relative to {rel_minor}, #{j+1})"
                                break
                    else:  # minor mode
                        # Find relative major (3 semitones up or 9 down)
                        rel_major_idx = (key_names.index(corr_data['key']) + 3) % 12
                        rel_major = f"{key_names[rel_major_idx]} major"
                        # Check if it's in top 5
                        for j, top_corr in enumerate(sorted_correlations[:5]):
                            if top_corr['key'] == key_names[rel_major_idx] and top_corr['mode'] == 'major':
                                rel_note = f" (relative to {rel_major}, #{j+1})"
                                break
                
                # print(f"{i+1:2d}. {corr_data['key']} {corr_data['mode']:5s}: {corr_data['correlation']:.6f} " +
                #     f"(prob: {probability:.2%}){highlight}{rel_note}")
            
            # Add information about bias
            
            # Add note about relative keys
            top_key = sorted_correlations[0]
            if top_key['mode'] == 'major':
                rel_minor_idx = (key_names.index(top_key['key']) + 9) % 12
                rel_minor = f"{key_names[rel_minor_idx]} minor"
                print(f"\nThe relative minor of {top_key['key']} major is {rel_minor}")
            else:  # minor mode
                rel_major_idx = (key_names.index(top_key['key']) + 3) % 12
                rel_major = f"{key_names[rel_major_idx]} major"
                print(f"\nThe relative major of {top_key['key']} minor is {rel_major}")
                
            # Check for close scores between relatives
            for i, corr_data in enumerate(sorted_correlations[:5]):
                if i == 0:
                    continue  # Skip the top result
                
                # If this is the relative key to the top result and scores are close
                if ((corr_data['mode'] == 'major' and 
                    key_names.index(corr_data['key']) == (key_names.index(sorted_correlations[0]['key']) + 3) % 12 and
                    sorted_correlations[0]['mode'] == 'minor') or
                    (corr_data['mode'] == 'minor' and
                    key_names.index(corr_data['key']) == (key_names.index(sorted_correlations[0]['key']) + 9) % 12 and
                    sorted_correlations[0]['mode'] == 'major')):
                    
                    score_diff = sorted_correlations[0]['correlation'] - corr_data['correlation']
                    score_diff_pct = score_diff / sorted_correlations[0]['correlation'] * 100
                    
                    # if score_diff_pct < 5:
                    #     print(f"\nWarning: The relative {corr_data['mode']} key ({corr_data['key']} {corr_data['mode']})")
                    #     print(f"is very close ({score_diff_pct:.1f}% difference) to the selected key.")
                    #     print(f"You may want to verify which is more appropriate for your analysis.")
        
        # Find the best matching key
        best_match = max(correlations, key=lambda x: x['correlation'])
        
        tonality = f"{best_match['key']} {best_match['mode']}"
        
        return tonality

if __name__ == "__main__":
    song_path = "path/to/your/song.mp3"
    extractor = TriadExtractor(hop_length=256)
    chord_changes = extractor.extract_chords(
        song_path,
        threshold=0.5,
        check_on_beat=True
    )
    for chord_change in chord_changes:
        print(f"Chord: {chord_change.chord}, Timestamp: {chord_change.timestamp:.2f}")
