import re
import numpy as np
from collections import Counter, defaultdict

class ChordComparator:
    def __init__(self):
        # Circle of fifths with both sharp and flat representations
        self.circle_of_fifths_sharp = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
        self.circle_of_fifths_flat = ['C', 'G', 'D', 'A', 'E', 'B', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F']
        
        # Enharmonic equivalents
        self.enharmonic_map = {
            'F#': 'Gb', 'Gb': 'F#',
            'C#': 'Db', 'Db': 'C#', 
            'G#': 'Ab', 'Ab': 'G#',
            'D#': 'Eb', 'Eb': 'D#',
            'A#': 'Bb', 'Bb': 'A#'
        }
    
    QUALITY_VOCAB = [
    '', '7', '7#11', '7#9', '7alt', '7b13', '7b9', '7sus', '7sus4', '9', 'aug', 'dim',
    'dim7', 'm', 'm7', 'm7b5', 'm9', 'maj7', 'maj9', 'mmaj7', 'power', 'power9', 'sus4'
    ]
    #--------------------------------------------------------------
    def normalize_chord(self, chord_name):
        """Normalize chord names for comparison"""
        if not chord_name:
            return ""
        
        normalized = re.sub(r'\s+', '', chord_name.strip())
        
        equivalences = {
            'D7susadd9': 'Dsus4',
            'G7/D': 'G7',
            'Bb/D': 'Bb',
        }
        
        return equivalences.get(normalized, normalized)
    #--------------------------------------------------------------
    def get_chord_root(self, chord_name):
        """Extract root note from chord"""
        match = re.match(r'^([A-G][#b]?)', chord_name)
        return match.group(1) if match else None
    #--------------------------------------------------------------
    def get_chord_quality(self, chord_name):
        """Extract chord quality/type using strict vocabulary."""
        if not chord_name:
            return ''
        # Remove spaces, standardize
        normalized = re.sub(r'\s+', '', chord_name.strip())
        m = re.match(r'^([A-G][b#]?)(.*)', normalized)
        if not m:
            return 'unknown'
        root, quality = m.group(1), m.group(2)
        # Direct match
        if quality in self.QUALITY_VOCAB:
            return quality
        return 'unknown'
    #--------------------------------------------------------------
    def chord_similarity(self, chord1, chord2):
        """Calculate similarity between two chords (0.0 to 1.0)"""
        if not chord1 or not chord2:
            return 0.0
            
        if self.normalize_chord(chord1) == self.normalize_chord(chord2):
            return 1.0
        
        root1 = self.get_chord_root(chord1)
        root2 = self.get_chord_root(chord2)
        
        if not root1 or not root2:
            return 0.0
        
        if root1 != root2:
            # Normalize enharmonic equivalents
            norm_root1 = self.enharmonic_map.get(root1, root1)
            norm_root2 = self.enharmonic_map.get(root2, root2)
            
            if norm_root1 == norm_root2:
                return 0.99  # Same root, different spelling
            
            # Check fifth relationship
            try:
                idx1 = self.circle_of_fifths_sharp.index(norm_root1)
                idx2 = self.circle_of_fifths_sharp.index(norm_root2)
                distance = min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))
                if distance == 1:
                    return 0.5
            except ValueError:
                pass
            return 0.0
        
        quality1 = self.get_chord_quality(chord1)
        quality2 = self.get_chord_quality(chord2)
        
        major_family = ['major', 'major7']
        minor_family = ['minor', 'minor7']
        dominant_family = ['dominant7', 'suspended']
        
        families = [major_family, minor_family, dominant_family]
        
        for family in families:
            if quality1 in family and quality2 in family:
                return 0.9
        
        return 0.9
    #--------------------------------------------------------------
    def extract_ireal_chords(self, ireal_data):
        """Extract clean chord sequence from iReal data"""
        ignored_tokens = {'<start>', '<end>', '<pad>', '.', '|', 
                        'Repeat_0', 'Repeat_1', 'Repeat_2', 'Repeat_3', 
                        'Form_A', 'Form_B', 'Form_C', 'Form_D', 
                        'Form_verse', 'Form_intro', 'Form_Coda', 
                        'Form_Segno', '|:', ':|'}

        chords = []
        for chord, offset in zip(ireal_data['chords'], ireal_data['offsets']):
            if chord not in ignored_tokens and self.is_actual_chord(chord):
                chords.append({'chord': chord, 'offset': offset})
        
        return chords
    #--------------------------------------------------------------
    def calculate_chord_weights(self, chord_data, data_type='ireal'):
        """Calculate weights for chords"""
        if data_type == 'ireal':
            chord_names = [item['chord'] for item in chord_data]
            counts = Counter(chord_names)
            total = sum(counts.values())
            return {chord: count/total for chord, count in counts.items()}
        
        elif data_type == 'lastfm':
            weights = defaultdict(float)
            total_weight = 0
            
            for item in chord_data:
                weight = item.get('weight', item.get('duration', 1))
                chord_name = item.get('chord_name', item.get('chord', ''))
                weights[chord_name] += weight
                total_weight += weight
            
            return {chord: weight/total_weight for chord, weight in weights.items()}
    #--------------------------------------------------------------
    def find_best_matches(self, ireal_weights, lastfm_weights):
        """Find best chord matches between datasets"""
        matches = []
        
        for ireal_chord, ireal_weight in ireal_weights.items():
            for lastfm_chord, lastfm_weight in lastfm_weights.items():
                similarity = self.chord_similarity(ireal_chord, lastfm_chord)
                if similarity > 0:
                    matches.append({
                        'ireal_chord': ireal_chord,
                        'lastfm_chord': lastfm_chord,
                        'similarity': similarity,
                        'ireal_weight': ireal_weight,
                        'lastfm_weight': lastfm_weight,
                        'combined_score': similarity * min(ireal_weight, lastfm_weight)
                    })
        
        return sorted(matches, key=lambda x: x['combined_score'], reverse=True)
    
    #--------------------------------------------------------------
    def is_actual_chord(self, chord_name):
        # Exclude known non-chord tokens
        if chord_name.startswith("Form_"):
            return False
        # Add more rules as needed
        return True



    #--------------------------------------------------------------
    def calculate_similarity_metrics(self, ireal_chords, lastfm_chords, match_threshold=0.7):
        ireal_weights = self.calculate_chord_weights(ireal_chords, 'ireal')
        lastfm_weights = self.calculate_chord_weights(lastfm_chords, 'lastfm')
        ireal_vocab = set(ireal_weights.keys())
        lastfm_vocab = set(lastfm_weights.keys())

        n_extracted = len(lastfm_vocab)
        n_matched_extracted = 0
        sum_best_sim = 0
        final_matches = []

        for extracted_chord in lastfm_vocab:
            best_ref_chord = None
            best_similarity = 0
            for ref_chord in ireal_vocab:
                sim = self.chord_similarity(extracted_chord, ref_chord)
                if sim > best_similarity:
                    best_similarity = sim
                    best_ref_chord = ref_chord
            sum_best_sim += best_similarity
            if best_similarity >= match_threshold:
                n_matched_extracted += 1
            final_matches.append({
                'lastfm_chord': extracted_chord,
                'ireal_chord': best_ref_chord,
                'similarity': best_similarity
            })

        precision = n_matched_extracted / n_extracted if n_extracted else 0
        error_percent = 100 * (1 - precision) if n_extracted else 0
        avg_similarity = sum_best_sim / n_extracted if n_extracted else 0

        n_reference = len(ireal_vocab)
        n_matched_reference = 0
        matched_chords = set()
        for ref_chord in ireal_vocab:
            best_similarity = max(
                [self.chord_similarity(ref_chord, extracted_chord) for extracted_chord in lastfm_vocab] or [0]
            )
            if best_similarity >= match_threshold:
                n_matched_reference += 1
                matched_chords.add(ref_chord)

        recall = n_matched_reference / n_reference if n_reference else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        vocabulary_overlap = len(matched_chords) / n_reference if n_reference else 0

        # Composite score if needed
        w_f1 = 0.5
        w_avg_sim = 0.3
        w_precision = 0.2
        overall_score = (w_f1 * f1_score) + (w_avg_sim * avg_similarity) + (w_precision * precision)

        return {
            'avg_similarity': avg_similarity,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'error_percent': error_percent,
            'vocabulary_overlap': vocabulary_overlap,
            'overall_score': overall_score,
            'n_extracted': n_extracted,
            'n_reference': n_reference,
            'ireal_vocab_size': n_reference,
            'lastfm_vocab_size': n_extracted,
            'n_matched_extracted': n_matched_extracted,
            'n_matched_reference': n_matched_reference,
            'final_matches': final_matches
        }


#--------------------------------------------------------------
def debug_chord_matches(ireal_data, lastfm_data, show_all=False):
    """Show detailed chord matching results"""
    comparator = ChordComparator()
    
    # Use the SAME method as the main comparison
    result = comparator.calculate_similarity_metrics(
        comparator.extract_ireal_chords(ireal_data), 
        lastfm_data
    )
    
    ireal_chords = comparator.extract_ireal_chords(ireal_data)
    ireal_weights = comparator.calculate_chord_weights(ireal_chords, 'ireal')
    lastfm_weights = comparator.calculate_chord_weights(lastfm_data, 'lastfm')
    
    print("=== CHORD VOCABULARIES ===")
    print(f"iReal chords: {sorted(ireal_weights.keys())}")
    print(f"LastFM chords: {sorted(lastfm_weights.keys())}")
    
    print("\n=== MATCHING RESULTS ===")
    
    # Use the final_matches from the fixed algorithm
    final_matches = result['final_matches']
    used_lastfm = set()
    
    for match in final_matches:
        used_lastfm.add(match['lastfm_chord'])
        similarity = match['similarity']
        ireal = match['ireal_chord'] if match['ireal_chord'] is not None else 'NO MATCH'
        lastfm = match['lastfm_chord'] if match['lastfm_chord'] is not None else 'NO MATCH'
        status = "✓ GOOD" if similarity >= 0.9 else "⚠ GOOD" if similarity >= 0.7 else "✗ BAD"
        if similarity < 0.9 or show_all:
            print(f"{ireal:12} -> {lastfm:12} | {similarity:.2f} | {status}")

    
    # Show unmatched chords
    matched_ireal = {m['ireal_chord'] for m in final_matches}
    unmatched_ireal = set(ireal_weights.keys()) - matched_ireal
    for chord in sorted(unmatched_ireal):
        print(f"{chord:12} -> {'NO MATCH':12} | 0.00 | ✗ MISSING")
    
    unmatched_lastfm = set(lastfm_weights.keys()) - used_lastfm
    if unmatched_lastfm:
        print(f"\n=== UNMATCHED LASTFM CHORDS ===")
        for chord in sorted(unmatched_lastfm):
            print(f"LastFM has: {chord} (not in iReal)")

#--------------------------------------------------------------
def compare_song_chords(ireal_data, lastfm_data):
    """
    Compare chords between iReal and LastFM data
    
    Args:
        ireal_data: dict with 'chords' and 'offsets' keys
        lastfm_data: list of chord dictionaries
    
    Returns:
        dict with similarity metrics
    """
    comparator = ChordComparator()
    ireal_chords = comparator.extract_ireal_chords(ireal_data)
    return comparator.calculate_similarity_metrics(ireal_chords, lastfm_data)