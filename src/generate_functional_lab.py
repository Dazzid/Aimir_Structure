#!/usr/bin/env python3
"""
Generate functional-harmony LAB files using music21 with audio HPCP+K-K tonality.

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

import librosa
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

# K-K profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

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


def match_key_profile(pc_distribution):
    best_key = None
    best_corr = -2
    for pc in range(12):
        rotated_major = np.roll(MAJOR_PROFILE, pc)
        rotated_minor = np.roll(MINOR_PROFILE, pc)
        major_corr = np.corrcoef(pc_distribution.astype(float), rotated_major.astype(float))[0, 1]
        minor_corr = np.corrcoef(pc_distribution.astype(float), rotated_minor.astype(float))[0, 1]
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = (PC_TO_NOTE[pc], "major")
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = (PC_TO_NOTE[pc], "minor")
    if not best_key:
        return "C", "major", 0.0
    return best_key[0], best_key[1], float(best_corr)


def estimate_key_from_audio_hpcp(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    if chroma_mean.sum() == 0:
        return "C", "major", 0.0
    chroma_mean = chroma_mean / chroma_mean.sum()
    return match_key_profile(chroma_mean)


def estimate_key_from_chords(chord_labels):
    if not chord_labels:
        return "C", "major", 0.0
    pc_counts = np.zeros(12, dtype=float)
    for label in chord_labels:
        root = parse_root_from_label(label)
        pc = root_to_pitch_class(root)
        if pc >= 0:
            pc_counts[pc] += 1
    total = pc_counts.sum()
    if total == 0:
        return "C", "major", 0.0
    pc_distribution = pc_counts / total
    return match_key_profile(pc_distribution)


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

    audio_path = Path(audio_base) / collection / "audio" / f"{song_id}.mp3"
    key_root = None
    key_mode = None
    key_conf = None

    if audio_path.exists():
        try:
            key_root, key_mode, key_conf = estimate_key_from_audio_hpcp(str(audio_path))
        except Exception:
            key_root = None
            key_mode = None

    if not key_root:
        key_root, key_mode, key_conf = estimate_key_from_chords([r["label"] for r in lab_rows])

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
    parser.add_argument("--output-suffix", default="_functional.lab")
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
