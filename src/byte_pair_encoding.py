import os
import json
from collections import Counter, defaultdict
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

INIT_TOKEN = "<init>"
END_TOKEN = "<end>"

def load_sequences(collection_path):
    sequences = []
    for root, _, files in os.walk(collection_path):
        for file in files:
            if file.endswith("_analysis.json"):
                with open(os.path.join(root, file)) as f:
                    data = json.load(f)
                    if "bound_segments" in data:
                        raw = [str(x) for x in data["bound_segments"]]
                        seq = tuple([INIT_TOKEN] + raw + [END_TOKEN])
                        sequences.append(seq)
    return sequences

def load_chords(collection_path):
    sequences = []
    for root, _, files in os.walk(collection_path):
        for file in files:
            if file.endswith("_analysis.json"):
                with open(os.path.join(root, file)) as f:
                    data = json.load(f)
                    if "chords" in data:
                        chords = data["chords"]
                        raw = [str(chord['functional_harmony']['functional']) for chord in chords]
                        seq = tuple([INIT_TOKEN] + raw + [END_TOKEN])
                        sequences.append(seq)
    return sequences

def get_pair_frequencies(sequences):
    pair_freq = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            pair_freq[pair] += 1
    return pair_freq

def merge_pair(sequences, pair_to_merge, new_token_id):
    merged_sequences = []
    for seq in sequences:
        merged_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i+1]) == pair_to_merge:
                merged_seq.append(new_token_id)
                i += 2
            else:
                merged_seq.append(seq[i])
                i += 1
        merged_sequences.append(tuple(merged_seq))
    return merged_sequences

def expand_token(token, merge_rules):
    if token not in merge_rules:
        return [token]
    else:
        a, b = merge_rules[token]
        return expand_token(a, merge_rules) + expand_token(b, merge_rules)

def byte_pair_encoding(sequences, num_merges=50):
    merge_rules = {}
    merge_history = []
    token_index = 0

    for _ in range(num_merges):
        pair_freq = get_pair_frequencies(sequences)
        if not pair_freq:
            break
        most_common_pair, freq = pair_freq.most_common(1)[0]
        new_token = f"MERGE_{token_index}"
        token_index += 1
        merge_history.append((most_common_pair, freq))
        sequences = merge_pair(sequences, most_common_pair, new_token)
        merge_rules[new_token] = most_common_pair

    return merge_rules, merge_history

def extract_top_patterns(merge_rules, merge_history, num=20):
    patterns = []
    for (a, b), freq in merge_history[:num]:
        token_id = max((k for k, v in merge_rules.items() if v == (a, b)), default=None)
        if token_id is not None:
            expanded = expand_token(token_id, merge_rules)
            patterns.append((tuple(expanded), freq))
    return patterns

def print_patterns(merge_rules, merge_history, num=10):
    print("\nTop Merged Patterns (in original tokens):")
    for i, ((a, b), freq) in enumerate(merge_history[:num]):
        token_id = max((k for k, v in merge_rules.items() if v == (a, b)), default=None)
        if token_id is not None:
            expanded = expand_token(token_id, merge_rules)
            print(f"{i+1:2d}. Pattern: {expanded}, Frequency: {freq}")

def save_bpe_results(save_path, merge_rules, merge_history):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    serializable_rules = [[token_id, list(pair)] for token_id, pair in merge_rules.items()]
    serializable_history = [[list(pair), freq] for pair, freq in merge_history]
    with open(save_path, "w") as f:
        json.dump({"merge_rules": serializable_rules, "merge_history": serializable_history}, f)

def load_bpe_results(load_path):
    with open(load_path, "r") as f:
        data = json.load(f)
    merge_rules = {token_id: tuple(pair) for token_id, pair in data["merge_rules"]}
    merge_history = [(tuple(pair), freq) for pair, freq in data["merge_history"]]
    return merge_rules, merge_history

def plot_pattern_heatmap(collection_pattern_dict, analysis_type="chords", image_save_dir="./"):
    os.makedirs(image_save_dir, exist_ok=True)
    all_patterns = set()
    for patterns in collection_pattern_dict.values():
        for pattern, _ in patterns:
            all_patterns.add(pattern)

    pattern_list = sorted(all_patterns, key=lambda p: str(p))
    collection_list = list(collection_pattern_dict.keys())
    freq_matrix = np.zeros((len(pattern_list), len(collection_list)))

    for i, pattern in enumerate(pattern_list):
        for j, collection in enumerate(collection_list):
            match = [freq for p, freq in collection_pattern_dict[collection] if p == pattern]
            freq_matrix[i, j] = match[0] if match else 0

    plt.figure(figsize=(10, 14))
    sns.heatmap(freq_matrix,
                xticklabels=collection_list,
                yticklabels=[str(p) for p in pattern_list],
                cmap="viridis",
                linewidths=0.5,
                annot=True,
                fmt='.0f',
                cbar=False)
    plt.xlabel("Collection", fontsize=18, fontweight='bold')
    plt.ylabel("Pattern", fontsize=18, fontweight='bold')
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.tight_layout()
    png_path = os.path.join(image_save_dir, f"pattern_heatmap_{analysis_type}.png")
    pdf_path = os.path.join(image_save_dir, f"pattern_heatmap_{analysis_type}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

def main(analysis_type="bound_segments", use_cache=True, image_save_dir="./images"):
    collection_patterns = {}

    for collection in ["lastfm", "suno", "udio"]:
        print(f"Processing collection: {collection} for {analysis_type} analysis")
        collection_path = f"../dataset/{collection}"
        bpe_cache_path = os.path.join(f"../dataset/bpe/{collection}_{analysis_type}.json")

        if use_cache and os.path.exists(bpe_cache_path):
            merge_rules, merge_history = load_bpe_results(bpe_cache_path)
            print(f"Loaded BPE results from cache: {bpe_cache_path}")
        else:
            if analysis_type == "chords":
                sequences = load_chords(collection_path)
            elif analysis_type == "bound_segments":
                sequences = load_sequences(collection_path)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")

            diversity = [len(set(seq)) / len(seq) for seq in sequences if len(seq) > 0]
            if diversity:
                print(f"Mean unique tokens ratio: {sum(diversity) / len(diversity):.4f}, Std: {np.std(diversity):.4f}")

            num_merges = 2000 if analysis_type == "chords" else 500
            merge_rules, merge_history = byte_pair_encoding(sequences, num_merges=num_merges)

            if use_cache:
                save_bpe_results(bpe_cache_path, merge_rules, merge_history)
                print(f"Saved BPE results to: {bpe_cache_path}")

        print_patterns(merge_rules, merge_history, num=20)
        top_patterns = extract_top_patterns(merge_rules, merge_history, num=20)
        collection_patterns[collection] = top_patterns
        print("---" * 10)

    plot_pattern_heatmap(collection_patterns, analysis_type=analysis_type, image_save_dir=image_save_dir)

if __name__ == "__main__":
    main(analysis_type="bound_segments", use_cache=True, image_save_dir="../samples/")
