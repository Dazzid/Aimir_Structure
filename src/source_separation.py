import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
import gc
from torch.amp import autocast
import argparse
import glob
import sys

#https://pytorch.org/audio/main/tutorials/hybrid_demucs_tutorial.html

# Configuration Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure output directory exists
def ensure_dir_exists(dir_path):
    """Create directory if it doesn't exist"""
    os.makedirs(dir_path, exist_ok=True)

# Utility Functions
def load_audio(filepath: str):
    """Load audio file with proper error handling"""
    print(f"Loading audio from {filepath}...")
    try:
        waveform, sample_rate = torchaudio.load(filepath)
        return waveform, sample_rate
    except Exception as e:
        print(f"Error loading audio: {e}")
        raise

def normalize_waveform(waveform: torch.Tensor, target_peak=0.9):
    """Normalize waveform with proper gain control"""
    # First center the waveform
    ref = waveform.mean(0)
    centered = waveform - ref.mean()
    
    # Then normalize to target peak
    current_peak = torch.max(torch.abs(centered))
    if current_peak > 0:
        gain = target_peak / current_peak
        normalized = centered * gain
        print(f"Normalized audio: peak from {current_peak:.4f} to {target_peak:.4f}")
        return normalized, current_peak
    else:
        print("Warning: Audio seems to be silent, no normalization applied")
        return centered, 1.0

def separate_sources_in_batches(model, waveform, segment_size=10, overlap=0.1, device=DEVICE, sample_rate=44100):
    """
    Process long audio files by splitting them into segments with overlap.
    Uses mixed precision for faster GPU processing.
    
    Args:
        model: The Demucs model
        waveform: The input audio waveform
        segment_size: Size of each segment in seconds
        overlap: Overlap between segments as a fraction of segment_size
        device: Device to process on
        sample_rate: Audio sample rate
        
    Returns:
        Separated sources
    """
    # Import here to avoid circular imports
    from demucs.apply import apply_model
    
    # Ensure the input is in the right format
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension for mono
    
    # Get audio dimensions
    audio_length = waveform.shape[-1]
    segment_length = int(segment_size * sample_rate)
    overlap_length = int(segment_length * overlap)
    
    # Initialize tensor to store separated segments
    separated_segments = None
    
    # Calculate total segments and print progress info
    total_segments = (audio_length - overlap_length) // (segment_length - overlap_length) + 1
    print(f"Processing audio in {total_segments} segments with {overlap*100:.0f}% overlap...")
    
    # Process in segments
    for i, start in enumerate(range(0, audio_length, segment_length - overlap_length)):
        end = min(start + segment_length, audio_length)
        segment_waveform = waveform[:, start:end].to(device)
        
        # Add padding if needed
        if segment_waveform.shape[-1] < segment_length:
            padded = torch.zeros(segment_waveform.shape[0], segment_length, device=device)
            padded[:, :segment_waveform.shape[-1]] = segment_waveform
            segment_waveform = padded
        
        print(f"Processing segment {i+1}/{total_segments} ({start/sample_rate:.1f}s - {end/sample_rate:.1f}s)")
        
        # Use mixed precision for faster GPU computation
        with torch.no_grad(), autocast('cuda', enabled=device.type == 'cuda'):
            # Apply the model using apply_model function instead of calling separate()
            segment_sources = apply_model(model, segment_waveform.unsqueeze(0))[0]
            
        # Move results to CPU to save GPU memory
        segment_sources = segment_sources.cpu()
        
        # Initialize output tensors if first segment
        if separated_segments is None:
            # Create a list with an empty tensor for each source
            separated_segments = [torch.zeros(segment_sources.shape[1], 0) for _ in range(segment_sources.shape[0])]
        
        # Process each source
        for src_idx in range(segment_sources.shape[0]):
            src = segment_sources[src_idx]  # Get source
            current_length = separated_segments[src_idx].shape[-1]
            
            # For the first segment, or if no overlap
            if current_length == 0 or overlap == 0:
                separated_segments[src_idx] = src[..., :end-start]
            else:
                # Calculate overlap region
                overlap_length = min(current_length, int(segment_length * overlap))
                
                # Create crossfade weights
                fade_in = torch.linspace(0, 1, overlap_length, device=src.device)
                fade_out = 1 - fade_in
                
                # Get the end of the previous segment and start of the current one
                prev_end = separated_segments[src_idx][..., -overlap_length:]
                curr_start = src[..., :overlap_length]
                
                # Apply crossfade
                crossfaded = fade_out.view(1, -1) * prev_end + fade_in.view(1, -1) * curr_start
                
                # Replace the overlap region
                separated_segments[src_idx][..., -overlap_length:] = crossfaded
                
                # Append the non-overlapping part
                if end-start > overlap_length:
                    separated_segments[src_idx] = torch.cat([
                        separated_segments[src_idx], 
                        src[..., overlap_length:end-start]
                    ], dim=-1)
        
        # Clean up to save memory
        del segment_waveform, segment_sources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return separated_segments

def get_dataset_from_path(audio_path):
    """
    Determine which dataset the audio file belongs to based on path.
    
    Args:
        audio_path: Full path to the audio file
        
    Returns:
        Dataset name (suno, udio, or lastfm)
    """
    path_str = str(audio_path).lower()
    
    # Check for each dataset name in the path string
    if "/suno/" in path_str:
        return "suno"
    elif "/udio/" in path_str:
        return "udio"
    elif "/lastfm/" in path_str:
        return "lastfm"
    
    # If no match found, raise an error to prevent incorrect processing
    raise ValueError(f"Could not determine dataset from path: {audio_path}")

def process_audio_file(audio_path, model_name="htdemucs"):
    """Process a single audio file and save stems to the correct folder structure"""
    # Extract file ID and dataset from path
    file_id = Path(audio_path).stem
    dataset = get_dataset_from_path(audio_path)
    
    # Setup output directories
    samples_dir = "/mnt/Aimir_HD"  # Base directory
    segmented_dir = os.path.join(samples_dir, dataset, "segmented", file_id)
    audio_output_dir = os.path.join(segmented_dir, "audio")
    
    # Skip if folder already exists
    if os.path.exists(audio_output_dir):
        print(f"Output directory already exists for {file_id}, skipping")
        return True

    print(f"Processing audio file: {audio_path}")
    
    # Step 1: Load the audio data (catch failures)
    try:
        waveform, sample_rate = load_audio(audio_path)
    except Exception as e:
        print(f"Warning: cannot load '{audio_path}': {e}. Skipping this file.")
        sys.exit(1)

    # Now that load succeeded, create output directory
    ensure_dir_exists(audio_output_dir)
    print(f"Output directory: {audio_output_dir}")

    print(f"Audio shape: {waveform.shape}, Duration: {waveform.shape[-1]/sample_rate:.2f}s, Sample rate: {sample_rate}Hz")
    
    # Check if mono audio (1 channel) and convert to stereo (2 channels)
    if waveform.shape[0] == 1:
        print("Detected mono audio, converting to stereo...")
        waveform = waveform.repeat(2, 1)  # Duplicate channel
        print(f"Converted to stereo. New shape: {waveform.shape}")
    
    # Step 2: Load the pre-trained Demucs model
    print(f"Loading pre-trained model '{model_name}' on {DEVICE}...")
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    model = get_model(model_name)
    model.to(DEVICE)
    model.eval()
    
    # Resample if needed
    if sample_rate != model.samplerate:
        print(f"Resampling from {sample_rate}Hz to {model.samplerate}Hz...")
        waveform = torchaudio.functional.resample(waveform, sample_rate, model.samplerate)
        sample_rate = model.samplerate
    
    # Step 3: Normalize the waveform
    print("Normalizing audio...")
    normalized_waveform, peak = normalize_waveform(waveform)
    
    # Determine segment size
    audio_duration = waveform.shape[-1] / sample_rate
    segment_size = min(30, max(10, audio_duration / 5))
    
    # Step 4: Separate the sources
    print(f"Separating sources with segment size of {segment_size}s...")
    try:
        sources = separate_sources_in_batches(
            model,
            normalized_waveform,
            segment_size=segment_size,
            overlap=0.2,
            device=DEVICE,
            sample_rate=sample_rate
        )
    except Exception as e:
        import traceback
        print(f"Error during source separation for {audio_path}: {e}")
        traceback.print_exc()
        return False
    
    # Step 5: Save the separated stems
    print(f"Saving separated tracks to {audio_output_dir}...")
    source_names = model.sources
    for name, source in zip(source_names, sources):
        output_name = "harmony" if name == "other" else name
        output_path = os.path.join(audio_output_dir, f"{output_name}.wav")
        
        # Normalize each stem
        stem_peak = torch.max(torch.abs(source))
        if stem_peak > 0:
            normalized_source = source * (0.9 / stem_peak)
            print(f"Normalized {output_name} stem: peak {stem_peak:.4f} → 0.9")
        else:
            normalized_source = source
            print(f"Warning: {output_name} stem appears silent")
        
        torchaudio.save(output_path, normalized_source, sample_rate)
        print(f"Saved {output_name} to {output_path}")
    
    print(f"Source separation complete for {file_id}!")
    return True

def process_folder_batch(collection_name, base_path="/workspace/samples", file_pattern="*.mp3", model_name="htdemucs"):
    """Process all audio files in a specific collection folder"""
    # Construct the input audio folder path
    input_folder = os.path.join(base_path, collection_name, "audio")
    
    # Find all matching audio files
    files = glob.glob(os.path.join(input_folder, file_pattern))
    
    # print(f"Found {len(files)} audio files to process in {input_folder}")
    
    # Track success/failure
    success_count = 0
    failed_files = []
    
    # Process each file
    for i, file_path in enumerate(files):
        # print(f"\n[{i+1}/{len(files)}] Processing {file_path}")
        
        if process_audio_file(file_path, model_name):
            success_count += 1
        else:
            failed_files.append(file_path)
    
    # Report results
    # print(f"\nProcessing complete: {success_count}/{len(files)} files processed successfully")
    
    if failed_files:
        print(f"Failed files ({len(failed_files)}):")
        for file in failed_files:
            print(f"  - {file}")

# CLI for direct usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio source separation using Demucs")
    parser.add_argument("--input", type=str, help="Input audio file or folder")
    parser.add_argument("--collection", type=str, choices=["suno_samples", "udio_samples", "lastfm_samples"], 
                        help="Collection to process (if processing an entire collection)")
    parser.add_argument("--model", type=str, default="htdemucs", help="Demucs model name")
    parser.add_argument("--batch", action="store_true", help="Process all audio files in the collection")
    
    args = parser.parse_args()
    
    if args.batch and args.collection:
        process_folder_batch(args.collection, model_name=args.model)
    elif args.input:
        if os.path.isfile(args.input):
            process_audio_file(args.input, model_name=args.model)
        else:
            print(f"Error: {args.input} is not a file")
    else:
        print("Error: Must provide either --input or --collection with --batch")