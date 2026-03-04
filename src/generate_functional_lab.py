#!/usr/bin/env python3
"""
Generate functional-harmony LAB files using music21.

Key estimation: duration-weighted diatonic template matching on ACE .lab chords.
For each of 24 candidate keys (12 major + 12 minor), compute the fraction of
total chord duration that is diatonic.  The key with the highest weighted
diatonic coverage wins.

Output format (per line):
    <start> <end> <functional>

For slash chords, append bass scale-degree info like:
    i (/III)

Usage:
    python generate_functional_lab.py
    python generate_functional_lab.py --collections suno udio --n-jobs 20
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import music21
from joblib import Parallel, delayed
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DEFAULT_COLLECTIONS = ["lastfm", "suno", "udio"]
DEFAULT_DATASET_BASE = "/workspace/dataset"
DEFAULT_AUDIO_BASE = "/mnt/Aimir_HD"  # audio at {audio_base}/{collection}/audio/{song_id}.mp3
DEFAULT_N_JOBS = 20

# ---------------------------------------------------------------------------
# Diatonic templates: for each mode, the expected (root_pc, quality) of each
# scale degree.  quality is 'maj', 'min', or 'dim'.
# ---------------------------------------------------------------------------
#                       I       ii      iii     IV      V       vi      vii°
_MAJOR_TEMPLATE = [(0,'maj'),(2,'min'),(4,'min'),(5,'maj'),(7,'maj'),(9,'min'),(11,'dim')]
#                       i       ii°     III     iv      v/V     VI      VII
# Includes BOTH v:min (natural) and V:maj (harmonic minor) as diatonic.
_MINOR_TEMPLATE = [(0,'min'),(2,'dim'),(3,'maj'),(5,'min'),(7,'min'),(7,'maj'),(8,'maj'),(10,'maj')]

NOTE_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
}
PC_TO_NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

CHORD_TYPE_TO_MUSIC21 = {
    "": "",
    "maj": "",
    "min": "m",
    "m": "m",
    "min7": "m7",
    "m7": "m7",
    "maj7": "maj7",
    "7": "7",
    "dim": "dim",
    "dim7": "dim7",
    "aug": "aug",
    "m7b5": "ø7",
    "hdim7": "ø7",
    "9": "9",
    "maj9": "maj9",
    "m9": "m9",
    "sus4": "sus4",
    "sus2": "sus2",
    "7sus4": "7sus4",
    "add9": "add9",
    "6": "6",
    "m6": "m6",
    "11": "11",
    "13": "13",
    "minmaj7": "mM7",
}

ROMAN_SCALE_DEGREES = ["I", "II", "III", "IV", "V", "VI", "VII"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def parse_lab_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            start = float(parts[0])
            end = float(parts[1])
            label = parts[2]
            if label == "N":
                continue
            rows.append({"start": start, "end": end, "label": label})
    return rows


def parse_root_from_label(chord_label):
    if ":" in chord_label:
        root = chord_label.split(":")[0]
    else:
        root = chord_label
    return root.strip()


def root_to_pitch_class(root):
    return NOTE_TO_PC.get(root, -1)


def _classify_quality(quality_str):
    """Map a LAB quality string to one of 'maj', 'min', 'dim', 'aug', 'sus', 'other'."""
    q = quality_str.lower().strip()
    if not q or q in ('maj', 'maj7', 'maj9', '7', '9', '11', '13', '6', 'add9',
                       '7sus4', '7sus2', '5'):
        # dominant-7, plain triads, extensions → functionally major
        return 'maj'
    if q in ('sus4', 'sus2'):
        return 'sus'                    # ambiguous — we give partial credit
    if q.startswith('min') or q.startswith('m') or q in ('m7b5', 'hdim7'):
        if 'dim' in q:
            return 'dim'
        return 'min'
    if q in ('dim', 'dim7'):
        return 'dim'
    if q in ('aug',):
        return 'aug'
    return 'other'


def _score_key(lab_rows, tonic_pc, template):
    """Score a candidate key (tonic_pc + template) against the .lab rows.

    For each chord we check whether (root_pc, quality) matches a diatonic
    scale degree.  The returned score is the total duration (seconds) of
    matching chords.  A 'sus' chord gets 0.5 credit if the root is diatonic.

    A 20% bonus is added for tonic-chord duration to break ties between
    relative major/minor keys (which share the same diatonic pitch-class set).
    """
    score = 0.0
    tonic_dur = 0.0
    total_dur = 0.0
    # Build lookup: transposed template → set of (pc, quality)
    diatonic = {((tonic_pc + pc) % 12, q) for pc, q in template}
    diatonic_roots = {(tonic_pc + pc) % 12 for pc, _ in template}
    # Expected tonic quality
    tonic_quality = template[0][1]          # 'maj' for major, 'min' for minor

    for row in lab_rows:
        dur = row["end"] - row["start"]
        if dur <= 0:
            continue
        total_dur += dur

        root_str = parse_root_from_label(row["label"])
        root_pc = root_to_pitch_class(root_str)
        if root_pc < 0:
            continue

        # Determine quality from the label
        label = row["label"]
        if ":" in label:
            raw_q = label.split(":", 1)[1]
            # Strip bass note
            if "/" in raw_q:
                raw_q = raw_q.rsplit("/", 1)[0]
            # Strip parenthetical extensions
            if "(" in raw_q:
                raw_q = raw_q[:raw_q.index("(")]
        else:
            raw_q = ""
        quality = _classify_quality(raw_q)

        if (root_pc, quality) in diatonic:
            score += dur
            # Track tonic chord duration
            if root_pc == tonic_pc and quality == tonic_quality:
                tonic_dur += dur
        elif quality == 'sus' and root_pc in diatonic_roots:
            score += dur * 0.5              # partial credit for sus chords
            if root_pc == tonic_pc:
                tonic_dur += dur * 0.25
        elif quality == 'other' and root_pc in diatonic_roots:
            score += dur * 0.25             # small credit for exotic qualities on diatonic root

    # Tonic bonus: 20% of tonic chord duration added to break relative-key ties
    score += tonic_dur * 0.2

    return score, total_dur


def estimate_key_diatonic(lab_rows):
    """Estimate key via duration-weighted diatonic template matching.

    Tries all 24 keys (12 major + 12 minor).  Returns (root_note, mode, confidence)
    where confidence ∈ [0, 1] is the fraction of total duration that is diatonic.
    """
    if not lab_rows:
        return "C", "major", 0.0

    best_root = "C"
    best_mode = "major"
    best_score = -1.0
    total_dur = 0.0

    for pc in range(12):
        for template, mode in [(_MAJOR_TEMPLATE, "major"), (_MINOR_TEMPLATE, "minor")]:
            sc, td = _score_key(lab_rows, pc, template)
            if td > 0:
                total_dur = td
            if sc > best_score:
                best_score = sc
                best_root = PC_TO_NOTE[pc]
                best_mode = mode

    conf = best_score / total_dur if total_dur > 0 else 0.0
    return best_root, best_mode, conf


def convert_lab_notation(chord_label):
    if not chord_label or chord_label == "N":
        return None, None, None
    bass = None
    if "/" in chord_label:
        parts = chord_label.rsplit("/", 1)
        chord_label = parts[0]
        bass = parts[1]
    if ":" in chord_label:
        root, quality = chord_label.split(":", 1)
    else:
        root = chord_label
        quality = ""
    original_quality = quality
    root = root.replace("b", "-")
    quality = quality.strip()
    if "(" in quality and ")" in quality:
        quality = quality[:quality.index("(")]
    if "sus4" in quality and "b7" in quality and "(" not in quality:
        m21_quality = "7sus4"
    elif "sus2" in quality and "b7" in quality and "(" not in quality:
        m21_quality = "7sus2"
    else:
        m21_quality = CHORD_TYPE_TO_MUSIC21.get(quality, quality)
        if m21_quality == quality and quality:
            if quality.startswith("min"):
                m21_quality = "m" + quality[3:]
            elif quality.startswith("maj"):
                m21_quality = "maj" + quality[3:] if len(quality) > 3 else ""
    return f"{root}{m21_quality}", original_quality, bass


def separate_roman_numeral(figure):
    m = re.match(r"^([b#]*[IViv]+)(.*)$", figure)
    if m:
        return m.group(1), m.group(2) or ""
    return figure, ""


def chord_to_roman_numeral(chord_label, key_root, key_mode):
    try:
        m21_chord, original_quality, bass = convert_lab_notation(chord_label)
        if not m21_chord:
            return None, bass
        key_root_m21 = key_root.replace("b", "-")
        k = music21.key.Key(key_root_m21, key_mode)
        c = music21.harmony.ChordSymbol(m21_chord)
        rn = music21.roman.romanNumeralFromChord(c, k)
        base, _ = separate_roman_numeral(rn.figure)
        if original_quality and "sus" in original_quality.lower():
            base = base.upper()
        return base, bass
    except Exception:
        return None, None


def bass_to_degree(bass_note, key_root, key_mode):
    if not bass_note:
        return None
    try:
        key_root_m21 = key_root.replace("b", "-")
        key = music21.key.Key(key_root_m21, key_mode)
        pitch = music21.pitch.Pitch(bass_note.replace("b", "-"))
        degree = key.getScaleDegreeFromPitch(pitch)
        if degree is None:
            return None
        return ROMAN_SCALE_DEGREES[degree - 1]
    except Exception:
        return None


def format_functional_label(rn, bass_degree):
    if rn is None:
        return None
    if bass_degree:
        return f"{rn} (/{bass_degree})"
    return rn


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------

def process_song(song_id, collection, dataset_base, audio_base, output_suffix):
    song_dir = Path(dataset_base) / collection / song_id
    lab_path = song_dir / f"{song_id}.lab"
    output_path = song_dir / f"{song_id}{output_suffix}"

    if not lab_path.exists():
        return {"status": "skipped", "song_id": song_id, "reason": "missing_lab"}

    lab_rows = parse_lab_file(lab_path)
    if not lab_rows:
        return {"status": "skipped", "song_id": song_id, "reason": "empty_lab"}

    # --- Key estimation: diatonic template matching on .lab chords ---
    key_root, key_mode, key_conf = estimate_key_diatonic(lab_rows)

    lines = []
    for row in lab_rows:
        rn, bass = chord_to_roman_numeral(row["label"], key_root, key_mode)
        bass_degree = bass_to_degree(bass, key_root, key_mode)
        func = format_functional_label(rn, bass_degree)
        if not func:
            continue
        lines.append(f"{row['start']:.3f} {row['end']:.3f} {func}\n")

    if not lines:
        return {"status": "skipped", "song_id": song_id, "reason": "no_functional"}

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return {
        "status": "ok",
        "song_id": song_id,
        "key": f"{key_root} {key_mode}",
        "confidence": key_conf,
        "output": str(output_path),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate functional harmony LAB files")
    parser.add_argument("--collections", nargs="+", default=DEFAULT_COLLECTIONS)
    parser.add_argument("--dataset-base", default=DEFAULT_DATASET_BASE)
    parser.add_argument("--audio-base", default=DEFAULT_AUDIO_BASE)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--output-suffix", default="_functional_2.lab")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N songs (0 = no limit)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    jobs = []
    for collection in args.collections:
        collection_dir = Path(args.dataset_base) / collection
        if not collection_dir.exists():
            logging.warning("Missing collection dir: %s", collection_dir)
            continue
        for d in collection_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                jobs.append((d.name, collection))

    logging.info("Found %d songs", len(jobs))

    if args.limit and args.limit > 0:
        jobs = jobs[: args.limit]
        logging.info("Limiting to %d songs", len(jobs))

    results = []
    ok = 0
    skipped = 0
    with tqdm(total=len(jobs), desc="Processing songs", unit="song") as pbar:
        for result in Parallel(n_jobs=args.n_jobs, backend="loky", return_as="generator")(
            delayed(process_song)(song_id, collection, args.dataset_base, args.audio_base, args.output_suffix)
            for song_id, collection in jobs
        ):
            results.append(result)
            if result["status"] == "ok":
                ok += 1
            else:
                skipped += 1
            pbar.set_postfix(ok=ok, skipped=skipped)
            pbar.update(1)

    logging.info("Done. ok=%d skipped=%d", ok, skipped)


if __name__ == "__main__":
    main()
