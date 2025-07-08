#!/usr/bin/env python3
"""
Audio source separation batch processor for multiple collections.
Run in tmux with: python process_audio.py
"""

import os
import glob
import torch
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from source_separation import process_audio_file, process_folder_batch

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"source_separation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def list_mp3_files(directory, limit=None):
    """List all .mp3 files (case-insensitive) in directory and its subdirectories."""
    logger.info(f"Scanning for MP3 files in {directory}...")
    
    try:
        # Get all files recursively
        paths = glob.glob(os.path.join(directory, "**", "*"), recursive=True)
        
        # Filter for files that are mp3s regardless of case
        mp3_files = [f for f in paths if os.path.isfile(f) and f.lower().endswith('.mp3')]
        
        if limit is not None and len(mp3_files) > limit:
            logger.info(f"Limiting to {limit} files from {len(mp3_files)} total")
            return mp3_files[:limit]
        
        logger.info(f"Found {len(mp3_files)} MP3 files")
        return mp3_files
    except Exception as e:
        logger.error(f"Error scanning for MP3 files: {e}")
        return []

def check_gpu():
    """Check for GPU availability and print info"""
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("No GPU available, using CPU")

def process_collections(base_dir, datasets, processing_mode, model_name, limit=None):
    """Process audio files from multiple collections"""
    logger.info(f"Starting audio processing with mode {processing_mode}")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Model: {model_name}")
    
    # Print information about each collection
    logger.info("Found the following collections:")
    for dataset in datasets:
        audio_dir = os.path.join(base_dir, dataset, "audio")
        mp3_files = list_mp3_files(audio_dir, limit=limit)
        
        if limit:
            logger.info(f" - {dataset}: Processing {len(mp3_files)} audio files (limited)")
        else:
            logger.info(f" - {dataset}: {len(mp3_files)} audio files")
    
    # Process files for each collection
    start_time = time.time()
    for dataset in datasets:
        dataset_start = time.time()
        logger.info(f"\nProcessing collection: {dataset}")
        
        audio_dir = os.path.join(base_dir, dataset, "audio")
        audio_files = list_mp3_files(audio_dir, limit=limit)  # Fixed: was list_audio_files
        
        if len(audio_files) == 0:
            logger.warning(f"No audio files found in {audio_dir}")
            continue
        
        try:
            if processing_mode == 1 and audio_files:
                # Process a single file from this collection
                file_index = 0
                selected_file = audio_files[file_index]
                file_id = Path(selected_file).stem
                
                logger.info(f"Processing file: {file_id} from {dataset}")
                
                # Process the file
                process_audio_file(selected_file, model_name=model_name)
                
                # Check results
                output_audio_dir = os.path.join(base_dir, dataset, "segmented", file_id, "audio")
                if os.path.exists(output_audio_dir):
                    logger.info(f"Results saved to: {output_audio_dir}")
                    
            elif processing_mode == 2:
                # Process all files in this collection in batch
                logger.info(f"Processing all files in {dataset} with batch mode")
                process_folder_batch(dataset, base_path=base_dir, model_name=model_name)
        except Exception as e:
            logger.error(f"Error processing dataset {dataset}: {e}")
            logger.error("Continuing with next dataset...")
        
        dataset_duration = time.time() - dataset_start
        logger.info(f"Finished processing {dataset} in {dataset_duration:.2f} seconds")
    
    total_duration = time.time() - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info("\nProcessing complete for all collections!")
    logger.info(f"Total processing time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio source separation batch processor")
    parser.add_argument("--base-dir", type=str, default="/mnt/Aimir_HD/",
                        help="Base directory containing the collections")
    parser.add_argument("--datasets", type=str, nargs="+", default=["suno", "udio", "lastfm"],
                        help="Dataset names to process")
    parser.add_argument("--mode", type=int, default=2, choices=[1, 2],
                        help="Processing mode: 1 for single file, 2 for batch")
    parser.add_argument("--model", type=str, default="htdemucs",
                        help="Demucs model name to use")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of files to process per dataset (for testing)")
    
    args = parser.parse_args()
    
    # Check GPU availability
    check_gpu()
    
    # Process the collections
    process_collections(
        base_dir=args.base_dir,
        datasets=args.datasets,
        processing_mode=args.mode,
        model_name=args.model,
        limit=args.limit
    )