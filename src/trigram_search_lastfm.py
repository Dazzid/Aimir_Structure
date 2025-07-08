import json
import re
from collections import Counter
from pathlib import Path

def normalize_function_label(label):
    match = re.match(r"([b#♭♯]*[IViv]+)", label)
    if match:
        return match.group(1)
    return label

def normalize_trigram(function_trigram):
    return tuple(normalize_function_label(x) for x in function_trigram)

def match_score(query, candidate):
    """Number of positions matching exactly."""
    return sum(q == c for q, c in zip(query, candidate))

def query_lastfm_trigram_usage(query_trigram, 
                               trigram_path="../dataset/trigrams/trigram_dataset_lastfm.json",
                               top_n=10):
    def normalize(label):
        match = re.match(r"([b#♭♯]*[IViv]+)", label)
        if match:
            return match.group(1).upper()
        return label

    # Normalize query to uppercase degrees
    query_norm = [normalize(x) for x in query_trigram]

    # --- LASTFM TRIGRAM DATASET ---
    trigram_path = Path(trigram_path)
    with trigram_path.open(encoding="utf-8") as f:
        trigram_data = json.load(f)

    found_entry = None
    best_score = -1
    best_entry = None
    best_entry_count = 0

    for entry in trigram_data:
        entry_norm = [normalize_function_label(x) for x in entry['function']]
        if entry_norm == query_trigram:
            found_entry = entry
            break
        score = match_score(query_trigram, entry_norm)
        if score > best_score or (score == best_score and entry.get("count", 0) > best_entry_count):
            best_score = score
            best_entry = entry
            best_entry_count = entry.get("count", 0)

    if found_entry:
        result_func = {
            "count": found_entry['count'],
            "styles": found_entry['styles'][:top_n],
            "genres": found_entry['genres'][:top_n],
            "matched": "exact",
            "function": found_entry['function'],
            "song_ids": found_entry['song_ids'][:top_n]
        }
    elif best_score > 0:  # if any match at all
        result_func = {
            "count": best_entry['count'],
            "styles": best_entry['styles'][:top_n],
            "genres": best_entry['genres'][:top_n],
            "matched": f"partial ({best_score}/3)",
            "function": best_entry['function'],
            "song_ids": best_entry['song_ids'][:top_n]
        }
    else:
        result_func = None

    return result_func
