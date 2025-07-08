import os
import librosa
import soundfile as sf
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm

audio_dir = '/mnt/Aimir_HD/suno/audio'
file_list = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]

def check_file(fname):
    fpath = os.path.join(audio_dir, fname)
    result = {"file": fname}
    try:
        # Try loading with librosa
        y, sr = librosa.load(fpath, sr=None, mono=False)
        result["librosa_loaded"] = True
        result["sr"] = sr
        result["shape"] = y.shape
        result["duration"] = y.shape[-1] / sr if len(y.shape) == 1 else y.shape[-1] / sr
        result["channels"] = 1 if len(y.shape) == 1 else y.shape[0]
        result["min"] = float(y.min())
        result["max"] = float(y.max())
    except Exception as e:
        result["librosa_loaded"] = False
        result["error"] = str(e)
        result["sr"] = None
        result["shape"] = None
        result["duration"] = None
        result["channels"] = None

    try:
        f = sf.SoundFile(fpath)
        result["soundfile_loaded"] = True
        result["sf_channels"] = f.channels
        result["sf_sr"] = f.samplerate
        result["sf_frames"] = f.frames
        result["sf_duration"] = f.frames / f.samplerate
        f.close()
    except Exception as e:
        result["soundfile_loaded"] = False
        result["sf_error"] = str(e)
    
    return result


def main():
    results = []
    with parallel_backend('loky', n_jobs=20):  # 'loky' is default; n_jobs=20 for 20 processes
        results = Parallel()(
            delayed(check_file)(fname) for fname in tqdm(file_list)
        )

    df = pd.DataFrame(results)
    df.to_csv("audio_integrity_check_results.csv", index=False)
    df.head()

if __name__ == "__main__":
    main()