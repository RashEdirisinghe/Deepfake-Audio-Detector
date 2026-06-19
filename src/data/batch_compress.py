import os
import glob
from src.data.compress_audio import compress_audio
from src.utils.logger import get_logger

logger = get_logger("batch_compression")


def process_directory(input_dir: str, output_dir: str, compression_level: str):
    os.makedirs(output_dir, exist_ok=True)

    # Grab all wav files in the input directory
    audio_files = glob.glob(os.path.join(input_dir, "*.wav"))

    if not audio_files:
        logger.warning(f"No WAV files found in {input_dir}")
        return

    logger.info(f"Starting compression for {len(audio_files)} files in {input_dir} at {compression_level} bitrate...")

    for file_path in audio_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)

        # Apply compression
        compress_audio(file_path, output_path, bitrate=compression_level)

    logger.info(f"Finished compressing {input_dir}")


def run_batch_compression():
    # Define directories
    directories_to_compress = [
        ("data/fake/tamil", "data/compressed/fake_tamil_mild", "64k"),
        ("data/fake/sinhala", "data/compressed/fake_sinhala_mild", "64k"),
        ("data/fake/tamil", "data/compressed/fake_tamil_heavy", "16k"),
        ("data/fake/sinhala", "data/compressed/fake_sinhala_heavy", "16k")
    ]

    for in_dir, out_dir, bitrate in directories_to_compress:
        process_directory(in_dir, out_dir, bitrate)


if __name__ == "__main__":
    run_batch_compression()