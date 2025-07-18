import os
import json
import numpy as np
from beat_tracking_dynamic import run_time_varying_tempo, analyze_chords_from_beat_results, correct_enharmonics
from triadExtractor import TriadExtractor
from transposition import Transposition
from mixed_techniques_chord_analyzer import correct_enharmonics as correct_enharmonics_audio
import chordAnalyzer as ca
import formExtractor as fem
from tqdm import tqdm

# Custom encoder for saving numpy and other data types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

# Process one song
def process_song(collection, song_id, base_path):
    print(f"PID {os.getpid()} processing {collection}/{song_id}")

    out_dir = os.path.join("../dataset_2", collection, song_id)
    out_json = os.path.join(out_dir, f"{song_id}_analysis.json")

    if os.path.isfile(out_json):
        print(f"✅ Skipping {collection}/{song_id} — output exists before dir lock.")
        return

    try:
        os.makedirs(out_dir, exist_ok=False)
        print(f"🔒 Claimed directory for {collection}/{song_id}")
    except FileExistsError:
        print(f"🔒 Directory already exists for {collection}/{song_id}, skipping (busy or done).")
        return

    if os.path.isfile(out_json):
        print(f"✅ Skipping {collection}/{song_id} — output exists after dir lock.")
        return

    try:
        print(f"[{song_id}] Checking audio and midi files")
        audio_path = os.path.join(base_path, "audio", f"{song_id}.mp3")
        midi_path = os.path.join(base_path, "segmented", song_id, "midi")
        bass_midi = os.path.join(midi_path, "bass.mid")
        harmony_midi = os.path.join(midi_path, "harmony.mid")

        assert os.path.isfile(bass_midi), f"❌ FATAL: Bass MIDI not found: {bass_midi}"
        assert os.path.isfile(harmony_midi), f"❌ FATAL: Harmony MIDI not found: {harmony_midi}"

        midi_paths = {'bass': bass_midi, 'harmony': harmony_midi}

        # This is usually where your error occurs:
        results = run_time_varying_tempo(audio_path, midi_paths)

        chord_results = analyze_chords_from_beat_results(results, beats_per_row=32)
        transposer = Transposition()
        triads = TriadExtractor(hop_length=1024)
        tonality = triads.getTonality(audio_path)
        tonality, alterations, scale = transposer.get_alterations_scales(tonality)
        chord_results = correct_enharmonics(chord_results, tonality, alterations, scale)
        functional_chords = ca.functional_chords(chord_results, tonality)
        beats = results['beats']
        formData = fem.formExtractor()
        data_dict = formData.optimize_song_structure(audio_path, functional_chords, beats, K=4, min_duration=2.5)

        with open(out_json, "w") as f:
            json.dump(data_dict, f, cls=CustomJSONEncoder)
        print(f"✅ Saved: {out_json}")

    except Exception as e:
        print(f"❌ Error for {collection}/{song_id}: {e}")
        # Clean up only if no valid output exists
        try:
            if os.path.isdir(out_dir):
                if not (os.path.isfile(out_json) and os.path.getsize(out_json) > 0):
                    import shutil
                    shutil.rmtree(out_dir)
                    print(f"🗑️  Removed incomplete folder for {collection}/{song_id}")
        except Exception as cleanup_e:
            print(f"Cleanup failed for {collection}/{song_id}: {cleanup_e}")
        return



