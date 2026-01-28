"""
ACE Chord Extractor - Parallel chord extraction using ACE model.

Extracts chords from audio files across suno, udio, and lastfm collections.
Uses 8 GPUs with 1 worker per GPU to avoid OOM errors.

Usage:
    python ACE_chord_extractor.py --test       # Test with 20 files
    python ACE_chord_extractor.py              # Full run on all files
"""

import os
import sys
import time
import random
import argparse
import gc
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

# Add consonance-ACE to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "consonance-ACE"))

from ACE.inference import run_inference


# Configuration
COLLECTIONS = ["suno", "udio", "lastfm"]
DATASET_BASE = Path(__file__).parent.parent / "dataset"
AUDIO_BASE = Path("/mnt/Aimir_HD")
CHECKPOINT = Path(__file__).parent.parent / "consonance-ACE/ACE/checkpoints/conformer_decomposed_smooth.ckpt"
VOCAB_PATH = Path(__file__).parent.parent / "consonance-ACE/ACE/chords_vocab.joblib"

# Available GPUs
AVAILABLE_GPUS = list(range(8))  # GPUs 0-7

N_JOBS = len(AVAILABLE_GPUS)  # 1 worker per GPU
CHORD_MIN_DURATION = 0.5


def process_single_song(collection: str, song_id: str, gpu_id: int = 0) -> dict:
    """
    Process a single song: extract chords and save .lab file.
    
    Args:
        collection: Collection name (suno, udio, lastfm)
        song_id: Unique identifier for the song
        gpu_id: GPU to use for this worker
        
    Returns:
        dict with status and message
    """
    import torch
    
    # Set GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Paths
    audio_path = AUDIO_BASE / collection / "audio" / f"{song_id}.mp3"
    output_dir = DATASET_BASE / collection / song_id
    output_lab = output_dir / f"{song_id}.lab"
    
    # Skip if already processed
    if output_lab.exists():
        return {"status": "skipped", "song_id": song_id, "message": "Already exists"}
    
    # Check audio exists
    if not audio_path.exists():
        return {"status": "error", "song_id": song_id, "message": f"Audio not found: {audio_path}"}
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        run_inference(
            audio_path=audio_path,
            checkpoint=CHECKPOINT,
            vocab_path=VOCAB_PATH,
            out_lab=output_lab,
            chord_min_duration=CHORD_MIN_DURATION,
            model_name="conformer_decomposed",
            threshold=0.5,
            chunk_dur=20.0,
        )
        return {"status": "success", "song_id": song_id, "message": f"Saved to {output_lab}"}
    except Exception as e:
        return {"status": "error", "song_id": song_id, "message": str(e)}
    finally:
        # Clean up GPU memory after each song
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def get_all_jobs(collections: list[str]) -> list[tuple[str, str]]:
    """
    Collect all (collection, song_id) pairs from dataset folders.
    
    Returns:
        List of (collection, song_id) tuples
    """
    jobs = []
    for collection in collections:
        collection_dir = DATASET_BASE / collection
        if not collection_dir.exists():
            print(f"⚠️ Collection directory not found: {collection_dir}")
            continue
            
        # Get all folder names (song IDs)
        song_ids = [
            d.name for d in collection_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ]
        print(f"📁 Found {len(song_ids)} songs in {collection}")
        
        for song_id in song_ids:
            jobs.append((collection, song_id))
    
    return jobs


def main():
    parser = argparse.ArgumentParser(description="ACE Chord Extractor - Parallel processing")
    parser.add_argument("--test", action="store_true", help="Test mode: process only 20 files")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS, help="Number of parallel workers")
    parser.add_argument("--collections", nargs="+", default=COLLECTIONS, help="Collections to process")
    args = parser.parse_args()
    
    random.seed(42)
    
    # Collect all jobs
    print("🔍 Scanning dataset folders...")
    jobs = get_all_jobs(args.collections)
    random.shuffle(jobs)  # Randomize for better GPU distribution
    
    # Filter out already processed
    jobs_to_process = []
    for collection, song_id in jobs:
        output_lab = DATASET_BASE / collection / song_id / f"{song_id}.lab"
        if not output_lab.exists():
            jobs_to_process.append((collection, song_id))
    
    print(f"📊 Total songs: {len(jobs)}, Already processed: {len(jobs) - len(jobs_to_process)}, To process: {len(jobs_to_process)}")
    
    # Test mode: limit to 20 files
    if args.test:
        jobs_to_process = jobs_to_process[:20]
        print(f"🧪 TEST MODE: Processing only {len(jobs_to_process)} files")
    
    if not jobs_to_process:
        print("✅ All files already processed!")
        return
    
    # Process in parallel
    print(f"🚀 Starting parallel processing with {args.n_jobs} workers on GPUs {AVAILABLE_GPUS}...")
    start_time = time.time()
    
    results = list(tqdm(
        Parallel(n_jobs=args.n_jobs, backend="loky", return_as="generator")(
            delayed(process_single_song)(collection, song_id, AVAILABLE_GPUS[i % len(AVAILABLE_GPUS)]) 
            for i, (collection, song_id) in enumerate(jobs_to_process)
        ),
        total=len(jobs_to_process),
        desc="Extracting chords"
    ))
    
    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    
    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    
    print(f"\n{'='*60}")
    print(f"✅ Completed: {success}")
    print(f"⏭️ Skipped: {skipped}")
    print(f"❌ Errors: {errors}")
    print(f"⏱️ Total time: {int(minutes)} min {seconds:.1f} sec")
    print(f"📈 Rate: {len(jobs_to_process) / elapsed:.2f} songs/sec")
    
    # Print errors if any
    if errors > 0:
        print(f"\n❌ Error details:")
        for r in results:
            if r["status"] == "error":
                print(f"  - {r['song_id']}: {r['message']}")


if __name__ == "__main__":
    main()
