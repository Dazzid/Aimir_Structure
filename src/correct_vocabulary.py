#!/usr/bin/env python3

import os
import json
import re
from tqdm import tqdm
from joblib import Parallel, delayed

def clean_chord_quality(quality):
    mapping = {
        '5': 'power',
        '59': 'power9',
        '7': '7',
        '7#11': '7#11',
        '7#9': '7#9',
        '7alt': '7alt',
        '7b13': '7b13',
        '7b9': '7b9',
        '7sus': '7sus',
        '7sus4': '7sus4',
        '9': '9',
        'aug': 'aug',
        'b5': 'm7b5',
        'b59': 'm7b5',
        'b5b9': 'dim7',
        'b9': '7b9',
        'dim': 'dim',
        'dim7': 'dim7',
        'm': 'm',
        'm7': 'm7',
        'm79': 'm7',
        'm7b5': 'm7b5',
        'm9': 'm9',
        'maj7': 'maj7',
        'maj7b9': 'maj7',
        'maj9': 'maj9',
        'mb9': '7b9',
        'mmaj7': 'mmaj7',
        'mmaj7b9': 'mmaj7',
        'sus4': 'sus4',
        '': '',
        '5b9': '7b9',
    }
    return mapping.get(quality, quality)

def split_root_quality(chord_name):
    m = re.match(r"([A-G][b#]?)(.*)", chord_name)
    if not m:
        return chord_name, ""
    return m.group(1), m.group(2)

def process_file(in_path, fileType="_analysis.json"):
    id_part = os.path.basename(in_path)[:-len(fileType)]
    out_path = os.path.join(os.path.dirname(in_path), f"{id_part}_vocab_corrected.json")
    with open(in_path, "r") as f:
        data = json.load(f)
    if "chords" in data:
        for chord in data["chords"]:
            chord_name = chord.get("chord_name", "")
            root_name, quality = split_root_quality(chord_name)
            corrected_quality = clean_chord_quality(quality)
            new_chord_name = root_name + corrected_quality
            chord["chord_name"] = new_chord_name
            if "chord_type" in chord:
                chord["chord_type"] = corrected_quality
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_all_collections(base_dir="/workspace/dataset_corrected", num_jobs=20):
    collections = ["lastfm", "suno", "udio"]
    input_files = []
    fileType = "_analysis.json"

    for collection in collections:
        collection_dir = os.path.join(base_dir, collection)
        for root, dirs, files in os.walk(collection_dir):
            for fname in files:
                if fname.endswith(fileType):
                    input_files.append(os.path.join(root, fname))

    # tqdm + Parallel for nice progress bar
    Parallel(n_jobs=num_jobs)(
        delayed(process_file)(in_path, fileType) for in_path in tqdm(input_files, desc="Normalizing chord vocabularies")
    )

if __name__ == "__main__":
    process_all_collections()
