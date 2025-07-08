#!/usr/bin/env python3
"""
Audio source separation and MIDI extraction processor with cleanup.
This script:
1. Separates audio sources from MP3 files
2. Extracts MIDI from the separated audio files
3. Converts vocals.wav to vocals.mp3 and deletes other audio files to save disk space

Run with: python process.py [options] 
Or with: python process.py datasets=suno
"""

import os
import glob
import torch
import time
import logging
import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from source_separation import process_audio_file
from extract_full_midi import extract_full_song

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"audio_midi_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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

def extract_midi_from_audio(song_path, verbose=False):
    """
    Extract MIDI from separated audio files in a song's audio directory.
    
    Args:
        song_path: Path to the song directory containing audio folder
        verbose: Whether to print detailed information
    
    Returns:
        bool: True if successful, False otherwise
    """
    audio_folder = os.path.join(song_path, "audio")
    midi_folder = os.path.join(song_path, "midi")
    
    # Skip if no audio folder
    if not os.path.isdir(audio_folder):
        logger.error(f"Audio folder not found in {song_path}")
        return False
    
    # Create midi folder if it doesn't exist
    os.makedirs(midi_folder, exist_ok=True)
    
    success = True
    # Iterate over all .wav files in the audio folder
    for file in os.listdir(audio_folder):
        if file.lower().endswith(".wav"):
            audio_path = os.path.join(audio_folder, file)
            base_name = os.path.splitext(file)[0]
            output_midi_path = os.path.join(midi_folder, f"{base_name}.mid")
            
            try:
                # Extract MIDI from the audio file
                if verbose:
                    logger.info(f"Extracting MIDI from {audio_path} to {output_midi_path}")
                extract_full_song(audio_path, output_midi_path, verbose=verbose)
            except Exception as e:
                logger.error(f"Error extracting MIDI from {audio_path}: {e}")
                success = False
    
    if success:
        logger.info(f"MIDI extraction complete for {song_path}")
    else:
        logger.warning(f"MIDI extraction had issues for {song_path}")
    
    return success

def convert_wav_to_mp3(wav_path, mp3_path):
    """
    Convert a WAV file to MP3 using LAME.
    
    Args:
        wav_path: Path to the WAV file
        mp3_path: Path where the MP3 file will be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Converting {wav_path} to MP3 using LAME...")
        
        # Use LAME for conversion with high quality settings
        subprocess.run([
            'lame', '-V0', '--vbr-new', wav_path, mp3_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info(f"Successfully converted to MP3 using LAME")
        return True
            
    except Exception as e:
        logger.error(f"Error converting WAV to MP3: {e}")
        return False

def cleanup_audio_folder(song_path, keep_vocals=True):
    """
    Clean up the audio folder to save disk space.
    If keep_vocals is True, converts vocals.wav to vocals.mp3 and deletes other WAV files.
    If keep_vocals is False, deletes the entire audio folder.
    
    Args:
        song_path: Path to the song directory containing audio folder
        keep_vocals: Whether to keep vocals (converted to MP3) or delete all audio
    
    Returns:
        bool: True if successful, False otherwise
    """
    audio_folder = os.path.join(song_path, "audio")
    
    if not os.path.exists(audio_folder):
        logger.warning(f"Audio folder not found in {song_path}, nothing to clean up")
        return True
    
    try:
        if keep_vocals:
            # Look for vocals.wav
            vocals_wav_path = os.path.join(audio_folder, "vocals.wav")
            vocals_mp3_path = os.path.join(audio_folder, "vocals.mp3")
            
            # Convert vocals.wav to MP3 if it exists
            vocals_converted = False
            if os.path.exists(vocals_wav_path):
                logger.info(f"Converting vocals.wav to MP3...")
                
                # Try to convert vocals.wav to MP3
                if convert_wav_to_mp3(vocals_wav_path, vocals_mp3_path):
                    # Check if the MP3 file was created successfully
                    if os.path.exists(vocals_mp3_path) and os.path.getsize(vocals_mp3_path) > 0:
                        logger.info(f"Successfully converted vocals to MP3")
                        vocals_converted = True
                        
                        # Delete the original vocals.wav since we now have an MP3 version
                        os.remove(vocals_wav_path)
                        logger.info(f"Deleted original vocals.wav")
                    else:
                        logger.warning(f"MP3 conversion failed - MP3 file not created or empty")
                else:
                    logger.warning(f"Failed to convert vocals.wav to MP3, will keep original WAV")
            
            # Delete all WAV files (except vocals.wav if conversion failed)
            for file in os.listdir(audio_folder):
                if file.lower().endswith(".wav"):
                    # Skip vocals.wav if conversion failed
                    if file == "vocals.wav" and not vocals_converted:
                        logger.info(f"Keeping {file} since MP3 conversion failed")
                        continue
                    
                    try:
                        os.remove(os.path.join(audio_folder, file))
                        logger.info(f"Deleted {file}")
                    except Exception as e:
                        logger.error(f"Error deleting {file}: {e}")
            
            # Verify what we have in the folder after cleanup
            remaining_files = os.listdir(audio_folder)
            logger.info(f"After cleanup, audio folder contains: {remaining_files}")
            
            return True
        else:
            # Remove the entire audio folder
            shutil.rmtree(audio_folder)
            logger.info(f"Successfully deleted audio folder: {audio_folder}")
            return True
            
    except Exception as e:
        logger.error(f"Error cleaning up audio folder {audio_folder}: {e}")
        return False

def process_single_file(audio_path, model_name="htdemucs", extract_midi=True, 
                   cleanup=True, keep_vocals=True, verbose=False):
    """
    Process a single MP3 file through the full pipeline:
    1. Separate audio sources
    2. Extract MIDI files from separated sources
    3. Clean up audio files to save space
    
    Args:
        audio_path: Path to the MP3 file
        model_name: Name of the source separation model to use
        extract_midi: Whether to extract MIDI files
        cleanup: Whether to delete audio files after MIDI extraction
        keep_vocals: Whether to keep vocals (converted to MP3) or delete all audio
        verbose: Whether to print detailed information
    
    Returns:
        bool: True if all steps succeeded, False otherwise
    """
    try:
        # Extract file ID and dataset from path
        file_id = Path(audio_path).stem
        
        # Determine dataset from path
        path_str = str(audio_path).lower()
        if "/suno/" in path_str:
            dataset = "suno"
        elif "/udio/" in path_str:
            dataset = "udio"
        elif "/lastfm/" in path_str:
            dataset = "lastfm"
        else:
            logger.error(f"Could not determine dataset from path: {audio_path}")
            return False
        
        # Setup directories
        base_dir = "/mnt/Aimir_HD"
        segmented_dir = os.path.join(base_dir, dataset, "segmented")
        song_path = os.path.join(segmented_dir, file_id)
        
        # Check if the folder already exists - if so, skip this file
        if os.path.exists(song_path):
            logger.info(f"Skipping {file_id} - folder already exists at {song_path}")
            return True
            
        logger.info(f"Processing {file_id} from {dataset}")
        
        # Step 1: Separate audio sources
        success = process_audio_file(audio_path, model_name=model_name)
        if not success:
            # If source separation fails, log the error and skip this file
            logger.error(f"Source separation failed for {audio_path}. Aborting.")
            sys.exit(1)
        
        # Step 2: Extract MIDI (if enabled)
        if extract_midi:
            midi_success = extract_midi_from_audio(song_path, verbose=verbose)
            if not midi_success:
                logger.error(f"MIDI extraction failed for {audio_path}, aborting.")
                sys.exit(1)
        
        # Step 3: Clean up audio (if enabled)
        if cleanup and (not extract_midi or midi_success):
            cleanup_success = cleanup_audio_folder(song_path, keep_vocals=keep_vocals)
            if not cleanup_success:
                logger.error(f"Cleanup failed for {audio_path}, aborting.")
                sys.exit(1)
        
        return True
    
    except Exception as e:
        import traceback
        logger.error(f"Fatal error processing {audio_path}: {e}")
        logger.error(traceback.format_exc())
        # Immediately stop the entire run
        sys.exit(1)

def process_folder_batch(collection_name, base_path="/mnt/Aimir_HD", model_name="htdemucs", 
                        extract_midi=True, cleanup=True, keep_vocals=True, verbose=False, limit=None):
    """
    Process all audio files in a specific collection through the full pipeline.
    
    Args:
        collection_name: Name of the collection (suno, udio, lastfm)
        base_path: Base path to collections
        model_name: Name of the source separation model to use
        extract_midi: Whether to extract MIDI files
        cleanup: Whether to delete audio files after MIDI extraction
        keep_vocals: Whether to keep vocals (converted to MP3) or delete all audio
        verbose: Whether to print detailed information
        limit: Maximum number of files to process (None for all)
    """
    # Construct the input audio folder path
    input_folder = os.path.join(base_path, collection_name, "audio")
    
    # Find all MP3 files
    files = list_mp3_files(input_folder, limit=limit)
    
    logger.info(f"Found {len(files)} audio files to process in {input_folder}")
    
    # Track success/failure
    success_count = 0
    failed_files = []
    
    # Process each file
    start_time = time.time()
    for i, file_path in enumerate(files):
        file_start_time = time.time()
        logger.info(f"\n[{i+1}/{len(files)}] Processing {file_path}")
        
        # Get file_id to check if folder already exists
        file_id = Path(file_path).stem
        
        # Determine dataset from path
        path_str = str(file_path).lower()
        if "/suno/" in path_str:
            dataset = "suno"
        elif "/udio/" in path_str:
            dataset = "udio"
        elif "/lastfm/" in path_str:
            dataset = "lastfm"
        else:
            logger.error(f"Could not determine dataset from path: {file_path}")
            failed_files.append(file_path)
            continue
            
        # Check if this song folder already exists
        song_path = os.path.join(base_path, dataset, "segmented", file_id)
        if os.path.exists(song_path):
            logger.info(f"Skipping {file_id} - folder already exists at {song_path}")
            success_count += 1
            continue
        
        if process_single_file(file_path, model_name, extract_midi, cleanup, keep_vocals, verbose):
            success_count += 1
        else:
            failed_files.append(file_path)
        
        file_duration = time.time() - file_start_time
        logger.info(f"File processed in {file_duration:.2f} seconds")
        
        # Estimate remaining time
        if i > 0:
            avg_time_per_file = (time.time() - start_time) / (i + 1)
            remaining_files = len(files) - (i + 1)
            est_remaining_time = avg_time_per_file * remaining_files
            
            hours, remainder = divmod(est_remaining_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            logger.info(f"Estimated remaining time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Report results
    total_duration = time.time() - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"\nProcessing complete: {success_count}/{len(files)} files processed successfully")
    logger.info(f"Total processing time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    if failed_files:
        logger.error(f"Failed files ({len(failed_files)}):")
        for file in failed_files:
            logger.error(f"  - {file}")

def process_collections(base_dir, datasets, model_name, extract_midi=True, cleanup=True, 
                    keep_vocals=True, verbose=False, limit=None):
    """Process multiple collections"""
    logger.info(f"Starting audio+MIDI processing pipeline")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Extract MIDI: {extract_midi}")
    logger.info(f"Cleanup audio: {cleanup}")
    logger.info(f"Keep vocals (convert to MP3): {keep_vocals}")
    
    # Process each collection
    start_time = time.time()
    for dataset in datasets:
        dataset_start = time.time()
        logger.info(f"\nProcessing collection: {dataset}")
        
        try:
            process_folder_batch(
                dataset, 
                base_path=base_dir, 
                model_name=model_name,
                extract_midi=extract_midi,
                cleanup=cleanup,
                keep_vocals=keep_vocals,
                verbose=verbose,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Error processing dataset {dataset}: {e}")
            logger.error("Continuing with next dataset...")
        
        dataset_duration = time.time() - dataset_start
        hours, remainder = divmod(dataset_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"Finished processing {dataset} in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    total_duration = time.time() - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info("\nProcessing complete for all collections!")
    logger.info(f"Total processing time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

def process_existing_audio_folders(base_dir, datasets=None, extract_midi=True, cleanup=True, 
                            keep_vocals=True, verbose=False):
    """
    Special mode to process existing audio folders that have already been created.
    Useful to extract MIDI and/or clean up after a previous run that only did audio separation.
    
    Args:
        base_dir: Base directory where collections are stored
        datasets: List of datasets to process (None for all)
        extract_midi: Whether to extract MIDI
        cleanup: Whether to clean up audio after MIDI extraction
        keep_vocals: Whether to keep vocals (converted to MP3) or delete all audio
        verbose: Whether to print detailed output
    """
    if datasets is None:
        datasets = ["suno", "udio", "lastfm"]
    
    logger.info(f"Processing existing audio folders for MIDI extraction and cleanup")
    logger.info(f"Extract MIDI: {extract_midi}, Cleanup: {cleanup}, Keep vocals (convert to MP3): {keep_vocals}")
    
    for dataset in datasets:
        segmented_dir = os.path.join(base_dir, dataset, "segmented")
        if not os.path.isdir(segmented_dir):
            logger.warning(f"Segmented folder not found in {dataset}, skipping")
            continue
            
        logger.info(f"Processing existing audio folders in {dataset}")
        
        # Get all song folders in the segmented directory
        song_folders = [f for f in os.listdir(segmented_dir) if os.path.isdir(os.path.join(segmented_dir, f))]
        logger.info(f"Found {len(song_folders)} song folders in {dataset}")
        
        success_count = 0
        failed_folders = []
        
        # Process each song folder
        for i, folder_name in enumerate(song_folders):
            song_path = os.path.join(segmented_dir, folder_name)
            audio_folder = os.path.join(song_path, "audio")
            
            # Skip if no audio folder or if already processed (no audio folder but has midi folder)
            if not os.path.isdir(audio_folder):
                midi_folder = os.path.join(song_path, "midi")
                if os.path.isdir(midi_folder):
                    # Already processed successfully
                    logger.debug(f"Skipping {folder_name}: already processed (no audio folder, but has midi)")
                    success_count += 1
                    continue
                else:
                    # Neither audio nor midi folder - something is wrong
                    logger.warning(f"Skipping {folder_name}: no audio or midi folder found")
                    failed_folders.append(folder_name)
                    continue
            
            logger.info(f"[{i+1}/{len(song_folders)}] Processing {folder_name}")
            
            success = True
            # Extract MIDI if enabled
            if extract_midi:
                midi_success = extract_midi_from_audio(song_path, verbose=verbose)
                if not midi_success:
                    logger.warning(f"MIDI extraction had issues for {folder_name}")
                    success = False
            
            # Clean up audio if enabled and MIDI extraction was successful
            if cleanup and (not extract_midi or success):
                cleanup_success = cleanup_audio_folder(song_path, keep_vocals=keep_vocals)
                if not cleanup_success:
                    logger.warning(f"Cleanup failed for {folder_name}")
                    success = False
            
            if success:
                success_count += 1
            else:
                failed_folders.append(folder_name)
        
        # Report results for this dataset
        logger.info(f"Processing complete for {dataset}: {success_count}/{len(song_folders)} folders processed successfully")
        
        if failed_folders:
            logger.warning(f"Failed folders in {dataset} ({len(failed_folders)}):")
            for folder in failed_folders[:10]:  # Only show first 10 to avoid log spam
                logger.warning(f"  - {folder}")
            if len(failed_folders) > 10:
                logger.warning(f"  ... and {len(failed_folders) - 10} more")

if __name__ == "__main__":
    # Handle the custom "datasets=suno" format
    for i, arg in enumerate(sys.argv):
        if '=' in arg and not arg.startswith('--'):
            try:
                key, value = arg.split('=', 1)
                sys.argv[i] = f"--{key}"
                sys.argv.insert(i+1, value)
            except Exception as e:
                print(f"Error parsing custom argument {arg}: {e}")

    parser = argparse.ArgumentParser(description="Audio source separation and MIDI extraction with cleanup")
    parser.add_argument("--base-dir", type=str, default="/mnt/Aimir_HD",
                        help="Base directory containing the collections")
    parser.add_argument("--datasets", type=str, nargs="+", default=["suno", "udio", "lastfm"],
                        help="Dataset names to process")
    parser.add_argument("--model", type=str, default="htdemucs",
                        help="Demucs model name to use for source separation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of files to process per dataset (for testing)")
    parser.add_argument("--no-midi", action="store_true",
                        help="Skip MIDI extraction (source separation only)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep audio files after processing (don't delete)")
    parser.add_argument("--delete-all-audio", action="store_true",
                        help="Delete all audio including vocals (by default vocals are converted to MP3)")
    parser.add_argument("--existing-only", action="store_true",
                        help="Process only existing audio folders (skip source separation)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed processing information")
    
    args = parser.parse_args()
    
    # Check GPU availability
    check_gpu()
    
    # Process existing audio folders only?
    if args.existing_only:
        process_existing_audio_folders(
            base_dir=args.base_dir,
            datasets=args.datasets,
            extract_midi=not args.no_midi,
            cleanup=not args.no_cleanup,
            keep_vocals=not args.delete_all_audio,
            verbose=args.verbose
        )
    else:
        # Process collections from scratch
        process_collections(
            base_dir=args.base_dir,
            datasets=args.datasets,
            model_name=args.model,
            extract_midi=not args.no_midi,
            cleanup=not args.no_cleanup,
            keep_vocals=not args.delete_all_audio,
            verbose=args.verbose,
            limit=args.limit
        )