import os
import pandas as pd
from pydub import AudioSegment
from src.utils.logger import get_logger

logger = get_logger("data_prep")


def process_raw_directory(raw_dir, audio_out_dir, csv_out_path):
    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_out_path), exist_ok=True)

    sentences = []
    audio_count = 0

    logger.info(f"Crawling through {raw_dir} for audio and text...")

    # os.walk automatically digs through all nested folders!
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            file_path = os.path.join(root, file)

            # 1. Handle Audio Files (Convert FLAC/MP3 to WAV)
            if file.endswith((".flac", ".mp3", ".wav")):
                try:
                    # Load the audio and export as 16kHz mono WAV
                    audio = AudioSegment.from_file(file_path)
                    audio = audio.set_frame_rate(16000).set_channels(1)

                    # Save it directly to our clean data/real/ folder
                    new_filename = f"real_{audio_count:05d}.wav"
                    out_path = os.path.join(audio_out_dir, new_filename)
                    audio.export(out_path, format="wav")
                    audio_count += 1
                except Exception as e:
                    logger.warning(f"Could not process audio {file}: {e}")

            # 2. Handle Text Files (Extract sentences for our TTS)
            elif file.endswith(".tsv"):
                try:
                    df = pd.read_csv(file_path, sep='\t')
                    if 'sentence' in df.columns:
                        sentences.extend(df['sentence'].dropna().tolist())
                except Exception as e:
                    logger.warning(f"Could not read TSV {file}: {e}")

            elif file.endswith(".txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            # Assuming basic text files, grab lines that aren't empty
                            clean_line = line.strip()
                            if clean_line:
                                sentences.append(clean_line)
                except Exception as e:
                    logger.warning(f"Could not read TXT {file}: {e}")

    # Remove duplicates and save to the clean CSV our generator expects
    unique_sentences = list(set(sentences))
    if unique_sentences:
        pd.DataFrame({'sentence': unique_sentences}).to_csv(csv_out_path, index=False)
        logger.info(f"Saved {len(unique_sentences)} unique sentences to {csv_out_path}")

    logger.info(f"Successfully converted and moved {audio_count} audio files to {audio_out_dir}")


if __name__ == "__main__":
    logger.info("=== Starting Raw Data Preparation ===")

    # Process Tamil
    process_raw_directory(
        raw_dir="data/raw_downloads/tamil",
        audio_out_dir="data/real/tamil",
        csv_out_path="data/real/tamil_transcripts.csv"
    )

    # Process Sinhala
    process_raw_directory(
        raw_dir="data/raw_downloads/sinhala",
        audio_out_dir="data/real/sinhala",
        csv_out_path="data/real/sinhala_transcripts.csv"
    )

    logger.info("=== Data Preparation Complete ===")