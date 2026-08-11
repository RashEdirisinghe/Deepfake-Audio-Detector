import os
import random
import glob
import shutil
from pydub import AudioSegment
from src.data.compress_audio import compress_audio
from src.utils.logger import get_logger

logger = get_logger("balance_and_augment")

def save_native_compressed(input_path, output_path, format_type, bitrate):
    """Saves a true .mp3 or .m4a file for web app testing and dataset augmentation."""
    try:
        audio = AudioSegment.from_file(input_path)
        codec_map = {"mp3": "libmp3lame", "m4a": "aac"}
        container_map = {"mp3": "mp3", "m4a": "ipod"} # 'ipod' container creates standard .m4a AAC files in pydub
        
        audio.export(
            output_path, 
            format=container_map[format_type], 
            bitrate=bitrate, 
            codec=codec_map[format_type]
        )
    except Exception as e:
        logger.error(f"Failed to export native {format_type} for {input_path}: {e}")

def main():
    logger.info("=== Starting Universal Dataset Balancing & Augmentation ===")
    
    # 1. Define all 4 core source directories
    directories = {
        "real_sinhala": "data/real/sinhala",
        "real_tamil": "data/real/tamil",
        "fake_sinhala": "data/fake/sinhala",
        "fake_tamil": "data/fake/tamil"
    }

    file_lists = {}
    counts = {}

    # 2. Count files to find the bottleneck
    for name, path in directories.items():
        wavs = glob.glob(os.path.join(path, "*.wav"))
        file_lists[name] = wavs
        counts[name] = len(wavs)
        logger.info(f"Found {len(wavs)} files in {name}")

    # Find the lowest number of files, capped at 2000 to save CPU time
    min_count = min(counts.values())
    target_count = min(min_count, 2000)
    
    logger.info(f"PERFECT BALANCE TARGET: Exactly {target_count} files per category.")

    # 3. Process each category
    for name, all_wavs in file_lists.items():
        random.seed(42)
        random.shuffle(all_wavs)
        selected_wavs = all_wavs[:target_count]

        # Setup output directories
        clean_dir = os.path.join("data/clean_balanced", name)
        mild_dir = os.path.join("data/compressed", f"{name}_mild")
        heavy_dir = os.path.join("data/compressed", f"{name}_heavy")
        mp3_dir = os.path.join("data/augmented", f"{name}_mp3")
        m4a_dir = os.path.join("data/augmented", f"{name}_m4a")

        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(mild_dir, exist_ok=True)
        os.makedirs(heavy_dir, exist_ok=True)
        os.makedirs(mp3_dir, exist_ok=True)
        os.makedirs(m4a_dir, exist_ok=True)

        logger.info(f"Processing {target_count} files for {name}...")

        for i, file_path in enumerate(selected_wavs, 1):
            filename = os.path.basename(file_path)
            filename_no_ext = os.path.splitext(filename)[0]

            # A. Save the Balanced Clean file
            clean_dest = os.path.join(clean_dir, filename)
            if not os.path.exists(clean_dest):
                shutil.copy(file_path, clean_dest)

            # B. Opus Mild Compression (Decoded back to WAV using your function)
            mild_path = os.path.join(mild_dir, filename)
            if not os.path.exists(mild_path):
                compress_audio(file_path, mild_path, bitrate="64k", format="ogg", codec="libopus")

            # C. Opus Heavy Compression (Decoded back to WAV using your function)
            heavy_path = os.path.join(heavy_dir, filename)
            if not os.path.exists(heavy_path):
                compress_audio(file_path, heavy_path, bitrate="16k", format="ogg", codec="libopus")

            # D. Native MP3 file (Saved as actual .mp3 for universal model training)
            mp3_path = os.path.join(mp3_dir, f"{filename_no_ext}.mp3")
            if not os.path.exists(mp3_path):
                save_native_compressed(file_path, mp3_path, format_type="mp3", bitrate="64k")

            # E. Native M4A/AAC file (Saved as actual .m4a for web app testing)
            m4a_path = os.path.join(m4a_dir, f"{filename_no_ext}.m4a")
            if not os.path.exists(m4a_path):
                save_native_compressed(file_path, m4a_path, format_type="m4a", bitrate="64k")

            if i % 100 == 0:
                logger.info(f"   -> Completed {i}/{target_count} for {name}")

    logger.info("=== Universal Balancing and Augmentation Complete! ===")

if __name__ == "__main__":
    main()