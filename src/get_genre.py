import os
import requests

# Limit visible GPUs to GPU 0 and 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import subprocess
import numpy as np
from essentia.standard import TensorflowPredictMusiCNN, MonoLoader
import json

dataset_path = "/home/dalmazzo/Aimir_Structure/dataset/"
save_path = dataset_path
audio_dir = "/mnt/Aimir_HD/"
collections = ["lastfm", "suno", "udio"]
# Our models take audio streams at 16kHz
sr = 16000
patch_hop_size = 0  # No overlap for efficiency
batch_size = 256
model_type = "musicnn-msd-2"
model_name = "genre_dortmund"
labels = [
            "alternative",
            "blues",
            "electronic",
            "folkcountry",
            "funksoulrnb",
            "jazz",
            "pop",
            "raphiphop",
            "rock",
        ]
frame_size = 512
hop_size = 256
patch_size = 187
nbands = 96

with open(os.path.join(os.path.dirname(__file__), "../../lastfm_key.json")) as f:
    api_key_data = json.load(f)
api_key = api_key_data["api_key"]

def get_lastfm_track_tags(api_key, artist, track, limit=10):
    url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "track.gettoptags",
        "artist": artist,
        "track": track,
        "api_key": api_key,
        "format": "json"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        return []

    data = response.json()
    
    # Check if the expected data is present
    if "toptags" in data and "tag" in data["toptags"]:
        tags = data["toptags"]["tag"]
        return [tag["name"] for tag in tags[:limit]]
    else:
        print("No tags found or malformed response.")
        return []

def download_genre_dortmund_model():
    url = "https://essentia.upf.edu/models/classifiers/genre_dortmund/genre_dortmund-musicnn-msd-2.pb"
    models_dir = "/home/lcros/MusicAnalysis/models"
    os.makedirs(models_dir, exist_ok=True)
    output_path = os.path.join(models_dir, "genre_dortmund-musicnn-msd-2.pb")
    if not os.path.exists(output_path):
        print("Downloading genre_dortmund model...")
        subprocess.run([
            "wget", "-nc", url, "-P", models_dir
        ], check=True)
    else:
        print("Model already exists at", output_path)

def load_genre_model():
    modelFilename = f"./models/{model_name}-{model_type}.pb"
    return TensorflowPredictMusiCNN(
        graphFilename=modelFilename,
        patchHopSize=patch_hop_size,
        patchSize=patch_size,
        batchSize=batch_size
    )

def analyse(collection, song_id, model):
    audio_path = os.path.join(audio_dir, collection, "audio", f"{song_id}.mp3")
    # Instantiate a MonoLoader and run it in the same line
    audio = MonoLoader(filename=audio_path, sampleRate=sr)()
    out_file = os.path.join(save_path, collection, song_id, f"{song_id}_style.json")
    # if out_file already exists, load it and only get the "original" tags
    if os.path.exists(out_file):
        model_results = json.load(open(out_file, "r"))
    else:
        activations = model(audio)
        model_results = {model_name: {}}
        for label, activation in zip(labels, np.mean(activations, axis=0)):
            model_results[model_name][label] = str(activation)
    
    metadata = json.load(open(os.path.join(audio_dir, collection, "metadata", f"{song_id}.json"), "r"))
    if collection == "lastfm":
        # Fetch tags from Last.fm
        artist = metadata.get("artist", "Unknown Artist")
        track = metadata.get("name", "Unknown Track")
        tags = get_lastfm_track_tags(api_key, artist, track)
        model_results["original"] = tags
    else:
        model_results["original"] = metadata.get("tags", [])

    # Save results to JSON file
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(model_results, f, indent=4)

def main():
    download_genre_dortmund_model()
    model = load_genre_model()

    for collection in collections:
        collection_path = os.path.join(audio_dir, collection, "audio")
        if not os.path.exists(collection_path):
            print(f"Collection {collection} does not exist at {collection_path}. Skipping...")
            continue
    
        # song ids are the ones found in dataset_path
        collection_dataset_path = os.path.join(dataset_path, collection)
        if not os.path.exists(collection_dataset_path):
            print(f"Creating dataset path for {collection} at {collection_dataset_path}...")
            os.makedirs(collection_dataset_path, exist_ok=True)
        print(f"Processing collection: {collection}")
        
        for song_id in os.listdir(collection_dataset_path):
            song_path = os.path.join(collection_dataset_path, song_id)
            if not os.path.isdir(song_path):
                print(f"Skipping {song_id}, not a directory.")
                continue
            
            print(f"Processing song: {song_id}")
            try:
                analyse(collection, song_id, model)
            except Exception as e:
                print(f"Error processing {song_id}: {e}")

if __name__ == "__main__":
    main()
