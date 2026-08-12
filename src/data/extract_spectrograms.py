import os
import glob
import torch
import yaml
import concurrent.futures
from src.utils.logger import get_logger
from src.utils.audio_utils import load_and_preprocess_audio

logger = get_logger("spectrogram_extractor")

def load_config(config_path="config.yaml"):
    """Loads project settings from the YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def process_single_file(file_path, output_dir, target_sr):
    """Helper function to process a single file on a separate CPU core."""
    filename = os.path.basename(file_path)
    for ext in [".wav", ".mp3", ".m4a", ".ogg", ".flac"]:
        filename = filename.replace(ext, ".pt")
        
    out_path = os.path.join(output_dir, filename)

    # Skip if already processed
    if os.path.exists(out_path):
        return True

    # Use our centralized audio utility!
    spec_tensor = load_and_preprocess_audio(file_path, target_sr=target_sr)
    
    if spec_tensor is not None:
        torch.save(spec_tensor, out_path)
        return True
        
    return False

def process_directory(input_dir, output_dir, target_sr):
    """Processes all audio files in a directory concurrently."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Grab all supported audio formats cleanly
    audio_files = []
    for ext in ("*.wav", "*.mp3", "*.m4a", "*.ogg", "*.flac"):
        audio_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return

    logger.info(f"Extracting spectrograms for {len(audio_files)} files in {input_dir}...")

    # MULTIPROCESSING: Process multiple audio files simultaneously
    success_count = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_file, path, output_dir, target_sr)
            for path in audio_files
        ]
        
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                success_count += 1

    logger.info(f"Finished {input_dir}. {success_count}/{len(audio_files)} tensors saved to {output_dir}")

if __name__ == "__main__":
    logger.info("=== Starting Universal Mel Spectrogram Extraction Pipeline ===")
    
    config = load_config()
    sr = config["audio"]["sample_rate"]

    directories_to_process = [
        # 1. Clean Balanced Data
        ("data/clean_balanced/fake_tamil", "data/spectrograms/clean_balanced/fake_tamil"),
        ("data/clean_balanced/fake_sinhala", "data/spectrograms/clean_balanced/fake_sinhala"),
        ("data/clean_balanced/real_tamil", "data/spectrograms/clean_balanced/real_tamil"),
        ("data/clean_balanced/real_sinhala", "data/spectrograms/clean_balanced/real_sinhala"),
        
        # 2. Compressed Opus Data (Mild & Heavy)
        ("data/compressed/fake_tamil_mild", "data/spectrograms/compressed/fake_tamil_mild"),
        ("data/compressed/fake_sinhala_mild", "data/spectrograms/compressed/fake_sinhala_mild"),
        ("data/compressed/fake_tamil_heavy", "data/spectrograms/compressed/fake_tamil_heavy"),
        ("data/compressed/fake_sinhala_heavy", "data/spectrograms/compressed/fake_sinhala_heavy"),
        ("data/compressed/real_tamil_mild", "data/spectrograms/compressed/real_tamil_mild"),
        ("data/compressed/real_sinhala_mild", "data/spectrograms/compressed/real_sinhala_mild"),
        ("data/compressed/real_tamil_heavy", "data/spectrograms/compressed/real_tamil_heavy"),
        ("data/compressed/real_sinhala_heavy", "data/spectrograms/compressed/real_sinhala_heavy"),
        
        # 3. Universal Augmented Data (Native MP3 & M4A)
        ("data/augmented/fake_tamil_mp3", "data/spectrograms/augmented/fake_tamil_mp3"),
        ("data/augmented/fake_tamil_m4a", "data/spectrograms/augmented/fake_tamil_m4a"),
        ("data/augmented/fake_sinhala_mp3", "data/spectrograms/augmented/fake_sinhala_mp3"),
        ("data/augmented/fake_sinhala_m4a", "data/spectrograms/augmented/fake_sinhala_m4a"),
        ("data/augmented/real_tamil_mp3", "data/spectrograms/augmented/real_tamil_mp3"),
        ("data/augmented/real_tamil_m4a", "data/spectrograms/augmented/real_tamil_m4a"),
        ("data/augmented/real_sinhala_mp3", "data/spectrograms/augmented/real_sinhala_mp3"),
        ("data/augmented/real_sinhala_m4a", "data/spectrograms/augmented/real_sinhala_m4a")
    ]

    for in_dir, out_dir in directories_to_process:
        process_directory(in_dir, out_dir, target_sr=sr)

    logger.info("=== Universal Spectrogram Extraction Complete ===")