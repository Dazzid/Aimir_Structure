#!/usr/bin/env python3
"""
Parallel normalization process for chord analysis data.
Processes JSON files from suno, udio, and lastfm collections using 20 workers.

Usage: taskset -c 0-19 python normalization_process.py
"""

import os
import json
import glob
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

# Import the normalization functions
from normalization_and_weights import compute_chord_weights, merge_isolated_chords
from JSON_parser import get_JSON_files


def process_single_file(file_path):
    """
    Process a single _vocab_corrected JSON file and save the normalized result.
    
    Args:
        file_path (str): Path to the *_vocab_corrected.json file
    """
    # Extract the ID from the file path
    file_path = Path(file_path)
    folder_path = file_path.parent
    file_name = file_path.stem  # Remove .json extension
    id_part = file_name.replace('_vocab_corrected', '')
    
    # Define output path
    output_path = folder_path / f"{id_part}_normalized.json"
    
    # Step 1: Load and process the JSON file
    chords_data = get_JSON_files(str(file_path))
    
    # Step 2: Compute chord weights
    weighted_chords = compute_chord_weights(chords_data)
    
    # Step 3: Merge isolated chords
    processed_chords = merge_isolated_chords(weighted_chords)
    
    # Step 4: Create the complete output structure preserving original data
    output_data = {
        "sr": chords_data[0].get("sr", 22050) if chords_data else 22050,  # Preserve sample rate
        "chords": processed_chords
    }
    
    # If the original file had additional fields, preserve them
    original_data = {}
    with open(str(file_path), 'r') as f:
        original_data = json.load(f)
    
    # Copy any additional fields from original data
    for key, value in original_data.items():
        if key not in output_data:
            output_data[key] = value
    
    # Save the normalized result
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)


def find_all_analysis_files():
    """
    Find all *_analysis.json files in the dataset collections.
    
    Returns:
        list: List of file paths to process
    """
    collections = ["lastfm", "suno", "udio"]
    all_files = []
    
    for collection in collections:
        dataset_path = f'/workspace/dataset_corrected/{collection}'
        # Find all *_analysis.json files in subdirectories
        pattern = os.path.join(dataset_path, '*', '*_vocab_corrected.json')
        files = glob.glob(pattern)
        all_files.extend(files)
    
    return all_files


def main():
    """
    Main function to run the parallel normalization process.
    """
    # Find all files to process
    files_to_process = find_all_analysis_files()
    
    # Process files in parallel with progress bar
    num_jobs = 20  # Number of parallel workers
    Parallel(n_jobs=num_jobs)(
        delayed(process_single_file)(file_path) 
        for file_path in tqdm(files_to_process)
    )
    print("Process completed successfully!")


if __name__ == "__main__":
    main()