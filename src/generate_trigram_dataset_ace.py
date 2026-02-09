#!/usr/bin/env python3
"""
Generate trigram dataset from ACE LAB files with Roman numeral conversion.
This script is designed to run in tmux for long-running processing.

Usage:
    python generate_trigram_dataset_ace.py
    
Or in tmux:
    tmux new -s trigrams
    python generate_trigram_dataset_ace.py
    # Detach with Ctrl+B then D
"""

import os
import sys
import json
import logging
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm
import numpy as np
import re
import music21
from datetime import datetime
from joblib import Parallel, delayed

# Setup logging
log_file = f'/workspace/trigram_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# TONALITY EXTRACTION
# ============================================================================

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTE_TO_PC = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4, 'E#': 5, 'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11, 'B#': 0
}
PC_TO_NOTE = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

def get_tonality_from_json(song_path: str, song_id: str) -> tuple:
    """Get tonality from _analysis.json file."""
    analysis_json = os.path.join(song_path, f"{song_id}_analysis.json")
    
    if not os.path.exists(analysis_json):
        return None
    
    try:
        with open(analysis_json, 'r') as f:
            data = json.load(f)
        
        tonality = data.get('tonality')
        if tonality:
            parts = tonality.strip().split()
            if len(parts) >= 2:
                key_root = parts[0]
                mode = parts[1].lower()
                return (key_root, mode)
            elif len(parts) == 1:
                return (parts[0], 'major')
    except Exception as e:
        logger.debug(f"Error reading tonality from {analysis_json}: {e}")
    
    return None

def parse_root_from_label(chord_label: str) -> str:
    """Extract root note from chord label."""
    if ':' in chord_label:
        root = chord_label.split(':')[0]
    else:
        root = chord_label
    return root.strip()

def root_to_pitch_class(root: str) -> int:
    """Convert root note to pitch class (0-11)."""
    return NOTE_TO_PC.get(root, -1)

def estimate_key_from_chords_fallback(chord_labels: list) -> tuple:
    """FALLBACK: Estimate key from chord sequence using Krumhansl-Kessler."""
    if not chord_labels:
        return ('C', 'major')
    
    pc_counts = np.zeros(12, dtype=float)
    for label in chord_labels:
        root = parse_root_from_label(label)
        pc = root_to_pitch_class(root)
        if pc >= 0:
            pc_counts[pc] += 1
    
    total = pc_counts.sum()
    if total == 0:
        return ('C', 'major')
    pc_distribution = pc_counts / total
    
    best_key = None
    best_corr = -2
    
    for pc in range(12):
        rotated_major = np.roll(MAJOR_PROFILE, pc)
        rotated_minor = np.roll(MINOR_PROFILE, pc)
        
        major_corr = np.corrcoef(pc_distribution.astype(float), rotated_major.astype(float))[0, 1]
        minor_corr = np.corrcoef(pc_distribution.astype(float), rotated_minor.astype(float))[0, 1]
        
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = (PC_TO_NOTE[pc], 'major')
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = (PC_TO_NOTE[pc], 'minor')
    
    return best_key if best_key else ('C', 'major')

# ============================================================================
# ROMAN NUMERAL CONVERSION
# ============================================================================

CHORD_TYPE_TO_MUSIC21 = {
    '': '', 'maj': '', 'min': 'm', 'm': 'm', 'min7': 'm7', 'm7': 'm7',
    'maj7': 'maj7', '7': '7', 'dim': 'dim', 'dim7': 'dim7', 'aug': 'aug',
    'm7b5': 'ø7', 'hdim7': 'ø7', '9': '9', 'maj9': 'maj9', 'm9': 'm9',
    'sus4': 'sus4', 'sus2': 'sus2', '7sus4': '7sus4', 'add9': 'add9',
    '6': '6', 'm6': 'm6', '11': '11', '13': '13', 'minmaj7': 'mM7',
}

def convert_lab_notation(chord_label: str) -> tuple:
    """Convert LAB chord notation to music21-compatible format."""
    if not chord_label or chord_label == 'N':
        return None, None, None
    
    bass = None
    if '/' in chord_label:
        parts = chord_label.rsplit('/', 1)
        chord_label = parts[0]
        bass = parts[1]
    
    if ':' in chord_label:
        root, quality = chord_label.split(':', 1)
    else:
        root = chord_label
        quality = ''
    
    original_quality = quality
    root = root.replace('b', '-')
    quality = quality.strip()
    
    # Strip ALL parenthetical extensions
    if '(' in quality and ')' in quality:
        quality = quality[:quality.index('(')]
    
    if 'sus4' in quality and 'b7' in quality and '(' not in quality:
        m21_quality = '7sus4'
    elif 'sus2' in quality and 'b7' in quality and '(' not in quality:
        m21_quality = '7sus2'
    elif quality.startswith('(') and quality.endswith(')'):
        intervals = quality.strip('()').split(',')
        if '1' in intervals and '5' in intervals and '3' not in intervals:
            m21_quality = '5'
        elif len(intervals) == 1 and intervals[0] == '3':
            m21_quality = 'sus2'
        else:
            m21_quality = ''
    else:
        m21_quality = CHORD_TYPE_TO_MUSIC21.get(quality, quality)
        
        if m21_quality == quality and quality:
            if quality.startswith('min'):
                m21_quality = 'm' + quality[3:]
            elif quality.startswith('maj'):
                m21_quality = 'maj' + quality[3:] if len(quality) > 3 else ''
    
    return f"{root}{m21_quality}", original_quality, bass

def separate_roman_numeral(figure: str) -> tuple:
    """Split figure into function and alteration."""
    m = re.match(r"^([b#]*[IViv]+)(.*)$", figure)
    if m:
        return m.group(1), m.group(2) or ''
    return figure, ''

def chord_to_roman_numeral(chord_label: str, key_root: str, key_mode: str) -> str:
    """Convert chord label to Roman numeral notation."""
    try:
        m21_chord, original_quality, bass = convert_lab_notation(chord_label)
        if not m21_chord:
            return None
        
        key_root_m21 = key_root.replace('b', '-')
        k = music21.key.Key(key_root_m21, key_mode)
        c = music21.harmony.ChordSymbol(m21_chord)
        rn = music21.roman.romanNumeralFromChord(c, k)
        
        base, _ = separate_roman_numeral(rn.figure)
        
        if original_quality and 'sus' in original_quality.lower():
            base = base.upper()
        
        if bass:
            return f"{base}/{bass}"
        return base
        
    except Exception as e:
        logger.debug(f"Error converting {chord_label}: {e}")
        return None

def chords_to_functional_with_tonality(chord_labels: list, song_path: str, song_id: str) -> list:
    """Convert chord labels to functional Roman numerals."""
    if not chord_labels:
        return []
    
    tonality = get_tonality_from_json(song_path, song_id)
    
    if tonality:
        key_root, key_mode = tonality
    else:
        key_root, key_mode = estimate_key_from_chords_fallback(chord_labels)
    
    roman_numerals = []
    for label in chord_labels:
        rn = chord_to_roman_numeral(label, key_root, key_mode)
        if rn:
            roman_numerals.append(rn)
    
    return roman_numerals

# ============================================================================
# FILE PARSING
# ============================================================================

def parse_lab_file(lab_path: str) -> list:
    """Parse LAB file and extract chord labels."""
    if not os.path.exists(lab_path):
        return []
    
    chords = []
    try:
        with open(lab_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    chord_label = parts[2]
                    if chord_label != 'N':
                        chords.append(chord_label)
    except Exception as e:
        logger.error(f"Error reading {lab_path}: {e}")
        return []
    
    return chords

# ============================================================================
# N-GRAM GENERATION
# ============================================================================

def make_ngram(chords, n=3):
    """Create n-grams from chord sequence."""
    if len(chords) < n:
        return []
    
    ngrams = []
    
    # Remove consecutive duplicates
    filtered_chords = []
    for i, chord in enumerate(chords):
        if i == 0 or chord != chords[i-1]:
            filtered_chords.append(chord)
    
    # Create n-grams
    for i in range(len(filtered_chords) - n + 1):
        ngram = tuple(filtered_chords[i:i+n])
        lowercase_ngram = [chord.lower() for chord in ngram]
        unique_chords = len(set(lowercase_ngram))
        
        if n == 3:
            if unique_chords == 3:
                ngrams.append(ngram)
        else:
            if unique_chords >= 3:
                ngrams.append(ngram)
    
    return ngrams

# ============================================================================
# GENRE NORMALIZATION
# ============================================================================

target_genres = [
    'rock', 'pop', 'jazz', 'electronic', 'blues', 'metal', 'hiphop', 'country',
    'folk', 'soul', 'punk', 'classical', 'reggae', 'rnb', 'indie', 'funk',
    'dance', 'latin', 'ambient', 'experimental', 'world'
]

explicit_map = {
    'pink': 'pop', 'p!nk': 'pop', 'beatles': 'rock', 'alternative': 'rock',
    'emo': 'punk', 'shoegaze': 'rock', 'acoustic': 'folk', '80s': 'pop',
    '90s': 'rock', 'rap': 'hiphop', 'hip hop': 'hiphop', 'indie rock': 'rock',
}

def normalize_tag(tag):
    """Normalize genre tag."""
    if tag is None:
        return 'other'
    
    t = tag.strip().lower()
    
    if t in explicit_map:
        return explicit_map[t]
    
    if t in target_genres:
        return t
    
    for genre in target_genres:
        if genre in t:
            return genre
    
    return 'other'

def extract_styles(style_json_path):
    """Extract and normalize styles from style JSON file."""
    try:
        with open(style_json_path, "r") as f:
            data = json.load(f)

        original = data.get("original", None)
        normalized_original_styles = []
        
        if isinstance(original, list):
            for style in original:
                normalized_style = normalize_tag(style)
                if normalized_style != 'other':
                    normalized_original_styles.append(normalized_style)
        elif isinstance(original, str):
            styles = [s.strip() for s in original.split(",") if s.strip()]
            for style in styles:
                normalized_style = normalize_tag(style)
                if normalized_style != 'other':
                    normalized_original_styles.append(normalized_style)
        
        genres = data.get("genre_dortmund", {})
        dortmund_genres = []
        for genre, confidence in genres.items():
            dortmund_genres.append({
                'genre': normalize_tag(genre),
                'confidence': float(confidence)
            })
        
        return normalized_original_styles, dortmund_genres
    except Exception as e:
        logger.debug(f"Error extracting styles from {style_json_path}: {e}")
        return [], []

# ============================================================================
# PARALLEL PROCESSING WORKER
# ============================================================================

def process_single_song(song_id, collection_path, n):
    """
    Worker function to process a single song.
    Returns ngrams with metadata or None if failed.
    """
    try:
        song_path = os.path.join(collection_path, song_id)
        style_json = os.path.join(song_path, f"{song_id}_style.json")
        lab_file = os.path.join(song_path, f"{song_id}.lab")
        
        if not os.path.exists(lab_file):
            return None
        
        # Extract styles
        original_styles = []
        dortmund_genres = []
        if os.path.exists(style_json):
            original_styles, dortmund_genres = extract_styles(style_json)
        
        # Parse LAB file
        raw_chords = parse_lab_file(lab_file)
        if not raw_chords:
            return None
        
        # Convert to Roman numerals
        roman_numerals = chords_to_functional_with_tonality(raw_chords, song_path, song_id)
        if not roman_numerals:
            return None
        
        # Generate n-grams
        ngrams = make_ngram(roman_numerals, n)
        
        if not ngrams:
            return None
        
        # Return results
        return {
            'song_id': song_id,
            'ngrams': ngrams,
            'original_styles': original_styles,
            'dortmund_genres': dortmund_genres
        }
        
    except Exception as e:
        return {'error': str(e), 'song_id': song_id}

# ============================================================================
# MAIN DATASET GENERATION
# ============================================================================

def create_ngram_aggregated_dataset_ace(base_path, collections, n=3):
    """Create n-gram aggregated dataset from LAB files."""
    logger.info(f"Creating aggregated {n}-gram dataset from LAB files (ACE format)...")
    logger.info(f"⚡ FAST MODE: Tonality from _analysis.json (no audio loading)")
    
    collection_data = {}
    
    for collection in collections:
        logger.info(f"\nProcessing collection: {collection}")
        collection_path = os.path.join(base_path, collection)
        
        if not os.path.exists(collection_path):
            logger.warning(f"Collection path {collection_path} does not exist")
            continue
            
        ngram_data = defaultdict(lambda: {
            "count": 0,
            "human_styles": Counter(),
            "dortmund_genres": Counter(),
            "song_ids": []
        })
        
        song_ids = [d for d in os.listdir(collection_path) 
                   if os.path.isdir(os.path.join(collection_path, d))]
        
        # TEST MODE: Process only first 20 songs
        # song_ids = song_ids[:20]
        
        logger.info(f"  Processing {len(song_ids)} songs with 20 parallel workers...")
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # Process songs in parallel using joblib
        results = Parallel(n_jobs=20, backend="multiprocessing", verbose=10)(
            delayed(process_single_song)(song_id, collection_path, n) for song_id in song_ids
        )
        
        # Aggregate results
        for result in results:
            if result is None:
                skipped_count += 1
                continue
            
            if 'error' in result:
                error_count += 1
                logger.error(f"Error processing {result['song_id']}: {result['error']}")
                continue
            
            processed_count += 1
            song_id = result['song_id']
            ngrams = result['ngrams']
            original_styles = result['original_styles']
            dortmund_genres = result['dortmund_genres']
            
            # Aggregate data
            for ngram in ngrams:
                ngram_key = tuple(ngram)
                ngram_data[ngram_key]["count"] += 1
                
                if song_id not in ngram_data[ngram_key]["song_ids"]:
                    ngram_data[ngram_key]["song_ids"].append(song_id)
                
                for style in original_styles:
                    ngram_data[ngram_key]["human_styles"][style] += 1
                
                for genre_info in dortmund_genres:
                    genre = genre_info['genre']
                    confidence = genre_info['confidence']
                    ngram_data[ngram_key]["dortmund_genres"][genre] += confidence
        
        collection_data[collection] = dict(ngram_data)
        logger.info(f"  Found {len(ngram_data)} unique {n}-grams in {collection}")
        logger.info(f"  Processed {processed_count} songs, skipped {skipped_count}, errors {error_count}")
    
    return collection_data

def export_ngram_dataset(collection_data, n, output_dir="/workspace/ngram_datasets"):
    """Export n-gram dataset to JSON files."""
    if n == 3:
        output_path = os.path.join(output_dir, "trigrams")
    elif n == 4:
        output_path = os.path.join(output_dir, "tetragrams")
    else:
        output_path = os.path.join(output_dir, f"{n}grams")
    
    os.makedirs(output_path, exist_ok=True)
    
    for collection, ngram_data in collection_data.items():
        export_data = []
        
        for ngram_tuple, stats in ngram_data.items():
            top_human_styles = stats["human_styles"].most_common(10)
            top_dortmund_genres = sorted(
                stats["dortmund_genres"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            export_entry = {
                "ngram": list(ngram_tuple),
                "count": stats["count"],
                "human_styles": top_human_styles,
                "dortmund_genres": top_dortmund_genres,
                "song_ids": stats["song_ids"],
                "num_songs": len(stats["song_ids"])
            }
            export_data.append(export_entry)
        
        export_data.sort(key=lambda x: x["count"], reverse=True)
        
        output_file = os.path.join(output_path, f"{n}gram_dataset_{collection}_ace.json")
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} {n}-grams for {collection} to {output_file}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    logger.info("="*80)
    logger.info("TRIGRAM DATASET GENERATION - ACE LAB FILES")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    BASE_PATH = "/workspace/dataset"
    COLLECTIONS = ["lastfm", "suno", "udio"]
    
    try:
        # Generate trigram dataset
        logger.info("\nStarting trigram dataset generation...")
        trigram_collection_data = create_ngram_aggregated_dataset_ace(BASE_PATH, COLLECTIONS, n=3)
        
        # Export datasets
        logger.info("\n" + "="*80)
        logger.info("EXPORTING TRIGRAM DATASETS")
        logger.info("="*80)
        export_ngram_dataset(trigram_collection_data, n=3)
        
        # Print statistics
        logger.info("\n" + "="*80)
        logger.info("TRIGRAM STATISTICS")
        logger.info("="*80)
        
        for collection, ngram_data in trigram_collection_data.items():
            total_ngrams = sum(stats["count"] for stats in ngram_data.values())
            unique_ngrams = len(ngram_data)
            
            logger.info(f"\n{collection.upper()}:")
            logger.info(f"  Unique trigrams: {unique_ngrams:,}")
            logger.info(f"  Total trigram instances: {total_ngrams:,}")
            
            if ngram_data:
                most_common = max(ngram_data.items(), key=lambda x: x[1]["count"])
                ngram, stats = most_common
                logger.info(f"  Most common trigram: {' - '.join(ngram)} (count: {stats['count']})")
        
        logger.info("\n" + "="*80)
        logger.info("✅ GENERATION COMPLETE!")
        logger.info(f"Log file: {log_file}")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
