import os
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from get_music_form import process_song
import random

def main():
    collections = ["lastfm", "suno", "udio"]
    n_jobs = 20  # Number of parallel workers
    random.seed(42)  # For reproducibility
    dataset_size = 20010  # Number of songs to process per collection
    jobs = []
    for collection in collections:
        
        # audio_dir = f"../samples/{collection}_samples/"
        audio_dir = f"/mnt/Aimir_HD/{collection}/"
        segmented_dir = os.path.join(audio_dir, "segmented")
        assert os.path.isdir(segmented_dir), f"Segmented dir not found for {collection}: {segmented_dir}"
        # Get list of folder names (song IDs) inside 'segmented'
        song_ids = [d for d in os.listdir(segmented_dir) if os.path.isdir(os.path.join(segmented_dir, d))]
        song_ids = song_ids[:dataset_size]
        random.shuffle(song_ids) # Randomized order for parallelization
        
        for song_id in song_ids:
            jobs.append((collection, song_id, audio_dir))

    start_time = time.time()
    results = list(tqdm(
        Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(process_song)(col, song_id, audio_dir) for col, song_id, audio_dir in jobs
        ),
        total=len(jobs),
        desc="Processing all songs"
    ))
    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    print(f"\nTotal elapsed time: {int(minutes)} min {seconds:.1f} sec for {len(jobs)} songs ({n_jobs} parallel workers)")

if __name__ == "__main__":
    main()
