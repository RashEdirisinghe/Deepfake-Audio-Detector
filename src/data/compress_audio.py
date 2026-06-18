import os
from pydub import AudioSegment
from src.utils.logger import get_logger

logger = get_logger("compression_engine")


def compress_audio(input_path: str, output_path: str, bitrate: str = "32k", format: str = "ogg",
                   codec: str = "libopus"):
    """
    Simulates real-world transmission compression (like WhatsApp/Telegram).
    Compresses the audio and exports it back as a WAV file for model ingestion.
    """
    try:
        # Load the original clean audio
        audio = AudioSegment.from_file(input_path)

        # Create a temporary compressed file (e.g., Opus/OGG)
        temp_compressed_path = output_path.replace(".wav", f"_temp.{format}")

        # Export with lossy compression
        audio.export(temp_compressed_path, format=format, bitrate=bitrate, codec=codec)

        # Load the compressed file and export back to WAV (Decoding)
        compressed_audio = AudioSegment.from_file(temp_compressed_path)
        compressed_audio.export(output_path, format="wav")

        # Clean up the temporary file
        os.remove(temp_compressed_path)

    except Exception as e:
        logger.error(f"Failed to compress {input_path}: {e}")


if __name__ == "__main__":
    # Test block
    logger.info("Compression utility loaded successfully.")