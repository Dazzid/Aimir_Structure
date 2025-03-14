import os
import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import gc
from torch.cuda.amp import autocast

# Configuration Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SONG_PATH = "/workspace/src/test_audio/Djavan - Azul (Ao Vivo).wav"
OUTPUT_DIR = "/workspace/src/test_audio/segmented"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utility Functions
def load_audio(filepath: str):
    print(f"Loading audio from {filepath}...")
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate

def normalize_waveform(waveform: torch.Tensor):
    ref = waveform.mean(0)
    return (waveform - ref.mean()) / ref.std(), ref

def denormalize_waveform(waveform: torch.Tensor, ref: torch.Tensor):
    return waveform * ref.std() + ref.mean()

def separate_sources_in_batches(model, waveform, segment_size=10, overlap=0.1, device=DEVICE):
    """
    Process long audio files by splitting them into segments with overlap.
    
    Args:
        model: The Demucs model
        waveform: The input audio waveform
        segment_size: Size of each segment in seconds
        overlap: Overlap between segments as a fraction of segment_size
        device: Device to process on
        
    Returns:
        Separated sources
    """
    # Get sample rate and calculate segment length in samples
    audio_length = waveform.shape[-1]
    sample_rate = 44100  # Demucs expects 44.1kHz
    segment_length = int(segment_size * sample_rate)
    overlap_length = int(segment_length * overlap)
    
    # Initialize list to store separated segments
    separated_segments = None
    
    # Calculate total segments and print progress info
    total_segments = (audio_length - overlap_length) // (segment_length - overlap_length) + 1
    print(f"Processing audio in {total_segments} segments...")
    
    for i, start in enumerate(range(0, audio_length, segment_length - overlap_length)):
        end = min(start + segment_length, audio_length)
        segment_waveform = waveform[:, start:end].to(device)
        
        # Add padding if needed
        if segment_waveform.shape[-1] < segment_length:
            padded = torch.zeros(segment_waveform.shape[0], segment_length, device=device)
            padded[:, :segment_waveform.shape[-1]] = segment_waveform
            segment_waveform = padded
        
        print(f"Processing segment {i+1}/{total_segments} ({start/sample_rate:.1f}s - {end/sample_rate:.1f}s)")
        
        with torch.no_grad(), autocast():
            # Use the model directly - this was the issue in the previous code
            segment_sources = model(segment_waveform[None])[0]
        
        # Move results to CPU to save GPU memory
        segment_sources = [s.cpu() for s in segment_sources]
        
        # If this is the first segment, initialize our separated_segments list
        if separated_segments is None:
            # Create a list with an empty tensor for each source
            separated_segments = [torch.zeros(1, 0) for _ in range(len(segment_sources))]
        
        # Process each segment
        for src_idx, src in enumerate(segment_sources):
            current_length = separated_segments[src_idx].shape[-1]
            
            # If this is not the first segment and there is overlap
            if current_length > 0 and overlap > 0:
                # Calculate overlap region
                overlap_start = current_length - overlap_length
                
                # Create crossfade weights
                fade_in = torch.linspace(0, 1, overlap_length)
                fade_out = 1 - fade_in
                
                # Apply crossfade in the overlap region
                overlap_end = current_length
                
                # Ensure we don't go out of bounds
                actual_overlap = min(overlap_length, current_length, src.shape[-1])
                
                if actual_overlap > 0:
                    # Get the overlapping parts
                    old_part = separated_segments[src_idx][..., -actual_overlap:]
                    new_part = src[..., :actual_overlap]
                    
                    # Apply crossfade
                    crossfaded = (
                        fade_out[-actual_overlap:].view(1, -1) * old_part +
                        fade_in[:actual_overlap].view(1, -1) * new_part
                    )
                    
                    # Replace overlap region with crossfaded version
                    separated_segments[src_idx][..., -actual_overlap:] = crossfaded
                    
                    # Append the non-overlap part
                    if src.shape[-1] > actual_overlap:
                        separated_segments[src_idx] = torch.cat([
                            separated_segments[src_idx], 
                            src[..., actual_overlap:min(end-start, src.shape[-1])]
                        ], dim=-1)
                else:
                    # Just append if there's no actual overlap
                    separated_segments[src_idx] = torch.cat([
                        separated_segments[src_idx], 
                        src[..., :min(end-start, src.shape[-1])]
                    ], dim=-1)
            else:
                # First segment, just store as is (trimmed to actual size)
                separated_segments[src_idx] = src[..., :min(end-start, src.shape[-1])]
        
        # Clean up to save memory
        del segment_waveform, segment_sources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return separated_segments

# Main Workflow
def run_source_separation():
    try:
        # Step 1: Load the audio data
        print("Loading audio...")
        waveform, sample_rate = load_audio(SONG_PATH)
        
        # Check if we need to resample (Demucs expects 44.1kHz)
        if sample_rate != 44100:
            print(f"Resampling from {sample_rate}Hz to 44100Hz...")
            waveform = torchaudio.functional.resample(waveform, sample_rate, 44100)
            sample_rate = 44100
        
        print(f"Audio shape: {waveform.shape}, Duration: {waveform.shape[-1]/sample_rate:.2f}s")

        # Step 2: Normalize the waveform
        print("Normalizing audio...")
        normalized_waveform, ref = normalize_waveform(waveform)

        # Step 3: Load the pre-trained Demucs model
        print(f"Loading pre-trained model on {DEVICE}...")
        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        model = bundle.get_model().to(DEVICE)
        model.eval()

        # Calculate appropriate segment size based on audio length
        audio_duration = waveform.shape[-1] / sample_rate
        segment_size = min(30, audio_duration / 2)  # Use smaller segments for shorter files
        
        # Step 4: Separate the sources in batches
        print(f"Separating sources with segment size of {segment_size}s...")
        # Demucs 4-stem model outputs: drums, bass, other, vocals
        sources = separate_sources_in_batches(
            model, 
            normalized_waveform, 
            segment_size=segment_size, 
            overlap=0.2,  # 20% overlap for smoother transitions
            device=DEVICE
        )
        
        # Denormalize back to original scale
        sources = [denormalize_waveform(s, ref) for s in sources]

        # Step 5: Save the separated tracks
        print("Saving separated tracks...")
        source_names = ["drums", "bass", "other", "vocals"]
        
        for name, source in zip(source_names, sources):
            # If name is "other", rename to "harmony" as per requirements
            output_name = "harmony" if name == "other" else name
            output_path = os.path.join(OUTPUT_DIR, f"{output_name}.wav")
            torchaudio.save(output_path, source, sample_rate)
            print(f"Saved {output_name} to {output_path}")

        print("Source separation complete!")
        
    except Exception as e:
        import traceback
        print(f"Error during source separation: {e}")
        traceback.print_exc()
        print("\nTrying alternative approach...")
        run_demucs_cli()

def run_demucs_cli():
    """
    Fallback method: Run Demucs through command line if the Python API fails
    """
    import subprocess
    import sys
    
    print("Attempting to use Demucs command-line interface...")
    
    # Check if demucs is installed
    try:
        subprocess.run(["demucs", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        print("Demucs not found. Installing demucs...")
        subprocess.run([sys.executable, "-m", "pip", "install", "demucs"], check=True)
    
    # Process the audio file with Demucs
    cmd = [
        "demucs", 
        "-n", "htdemucs",  # Use the latest model
        "-o", OUTPUT_DIR,
        SONG_PATH
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    
    if result.returncode == 0:
        print("Demucs separation successful!")
        
        # Organize files - rename "other" to "harmony"
        base_name = os.path.basename(SONG_PATH).replace(".wav", "")
        other_path = os.path.join(OUTPUT_DIR, "htdemucs", base_name, "other.wav")
        harmony_path = os.path.join(OUTPUT_DIR, "harmony.wav")
        
        # Copy all files to the root of OUTPUT_DIR
        for stem in ["vocals", "drums", "bass"]:
            src = os.path.join(OUTPUT_DIR, "htdemucs", base_name, f"{stem}.wav")
            dst = os.path.join(OUTPUT_DIR, f"{stem}.wav")
            if os.path.exists(src):
                os.replace(src, dst)
                print(f"Moved {stem} to {dst}")
        
        # Handle the "other" to "harmony" rename
        if os.path.exists(other_path):
            os.replace(other_path, harmony_path)
            print(f"Renamed and moved 'other' to 'harmony' at {harmony_path}")
        
        print("All files organized successfully!")
    else:
        print("Demucs command failed. Please check your installation and file paths.")

# Run the workflow
if __name__ == "__main__":
    run_source_separation()