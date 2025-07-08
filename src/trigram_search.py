import json
import re
from collections import Counter
from pathlib import Path

def normalize_function_label(label):
    match = re.match(r"([b#♭♯]*[IViv]+)", label)
    if match:
        return match.group(1)  # Do NOT upper!
    return label

def normalize_trigram(function_trigram):
    return tuple(normalize_function_label(x) for x in function_trigram)

def match_score(query, candidate):
    """Number of positions matching exactly."""
    return sum(q == c for q, c in zip(query, candidate))


def query_trigram_usage(query_trigram, 
                        func_path="/workspace/dataset/trigrams/functional_trigrams.json",
                        trigram_path="/workspace/dataset/trigrams/trigram_dataset.json",
                        top_n=10):
    import json
    from collections import Counter
    from pathlib import Path

    def normalize(label):
        match = re.match(r"([b#♭♯]*[IViv]+)", label)
        if match:
            return match.group(1).upper()
        return label

    # Normalize query to uppercase degrees
    query_norm = [normalize(x) for x in query_trigram]

    # --- FUNCTIONAL_TRIGRAMS ---
    func_path = Path(func_path)
    with func_path.open(encoding="utf-8") as f:
        functional_trigrams = json.load(f)

    found_func = None
    best_score = -1
    best_entry = None
    best_entry_count = 0

    for entry in functional_trigrams:
        entry_norm = [normalize_function_label(x) for x in entry['function']]
        if entry_norm == query_trigram:
            found_func = entry
            break
        score = match_score(query_trigram, entry_norm)
        if score > best_score or (score == best_score and entry.get("count", 0) > best_entry_count):
            best_score = score
            best_entry = entry
            best_entry_count = entry.get("count", 0)

    if found_func:
        result_func = {
            "count": found_func['count'],
            "styles": found_func['styles'][:top_n],
            "matched": "exact",
            "function": found_func['function'],
        }
    elif best_score > 0:  # if any match at all
        result_func = {
            "count": best_entry['count'],
            "styles": best_entry['styles'][:top_n],
            "matched": f"partial ({best_score}/3)",
            "function": best_entry['function'],
        }
    else:
        result_func = None


    # --- TRIGRAM_DATASET ---
    trigram_path = Path(trigram_path)
    with trigram_path.open(encoding="utf-8") as f:
        trigram_data = json.load(f)

    style_counter = Counter()
    composer_counter = Counter()
    total_count = 0

    for entry in trigram_data:
        entry_norm = [normalize(x) for x in entry['function']]
        if entry_norm == query_norm:
            count = entry['count']
            total_count += count
            for style in entry.get('styles', []):
                style_counter[style] += count
            for composer in entry.get('composers', []):
                composer_counter[composer] += count

    result_trigram = {
        "total_count": total_count,
        "styles": style_counter.most_common(top_n),
        "composers": composer_counter.most_common(top_n)
    }

    return {
        "functional_trigrams": result_func,
        "trigram_dataset": result_trigram
    }

