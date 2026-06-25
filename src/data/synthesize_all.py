import os
import csv
import pandas as pd
from src.data.generate_tamil_synthetic import generate_tamil_audio
from src.data.generate_sinhala_synthetic import generate_sinhala_audio
from src.utils.logger import get_logger

logger = get_logger("batch_synthesizer")


def write_manifest(rows, manifest_path):
    """Writes the generated metadata to a CSV manifest."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_path", "language", "text", "label", "generator", "source", "compression_level"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_language_rows(output_dir, language, texts, generator_name, generated_files):
    """Formats the metadata ONLY for files that actually generated successfully."""
    rows = []
    prefix = "fake_tam" if language == "Tamil" else "fake_sin"

    for idx, text in enumerate(texts, start=1):
        expected_path = os.path.join(output_dir, f"{prefix}_{idx:04d}.wav")

        # EDGE CASE SOLVED: Check against the list of successful files!
        if expected_path in generated_files:
            rows.append(
                {
                    "file_path": expected_path,
                    "language": language,
                    "text": text,
                    "label": "fake",
                    "generator": generator_name,
                    "source": "synthetic_tts",
                    "compression_level": "clean",
                }
            )
        else:
            logger.warning(f"Skipping manifest entry for {expected_path} (Generation failed)")

    return rows


def generate_from_csv(csv_path, language, output_dir, generator_name, sample_limit=2000):
    """Reads real transcripts, generates fake audio, and returns metadata rows."""
    if not os.path.exists(csv_path):
        logger.warning(f"Manifest not found at {csv_path}. Waiting for real dataset downloads!")
        return []

    try:
        df = pd.read_csv(csv_path)
        if 'sentence' not in df.columns:
            logger.error(f"Could not find 'sentence' column in {csv_path}")
            return []

        sentences = df['sentence'].dropna().tolist()[:sample_limit]
        logger.info(f"Loaded {len(sentences)} unique text prompts for {language}...")

        os.makedirs(output_dir, exist_ok=True)

        # Capture the successful files returned by your optimized scripts
        if language.lower() == "tamil":
            successful_files = generate_tamil_audio(sentences, output_dir=output_dir)
        elif language.lower() == "sinhala":
            successful_files = generate_sinhala_audio(sentences, output_dir=output_dir)
        else:
            successful_files = []

        # Pass the successful files into the row builder
        return build_language_rows(output_dir, language.capitalize(), sentences, generator_name, successful_files)

    except Exception as e:
        logger.error(f"Failed to process dataset manifest: {e}")
        return []


if __name__ == "__main__":
    logger.info("=== Starting Mass Synthetic Audio Generation & Indexing ===")

    tamil_csv_path = "data/real/tamil_transcripts.csv"
    sinhala_csv_path = "data/real/sinhala_transcripts.csv"
    manifest_path = "data/metadata/synthetic_manifest.csv"

    all_rows = []

    # Generate Tamil
    tamil_rows = generate_from_csv(tamil_csv_path, "tamil", "data/fake/tamil", "facebook/mms-tts-tam", 2000)
    all_rows.extend(tamil_rows)

    # Generate Sinhala
    sinhala_rows = generate_from_csv(sinhala_csv_path, "sinhala", "data/fake/sinhala", "gTTS-si", 2000)
    all_rows.extend(sinhala_rows)

    # Save the manifest
    if all_rows:
        write_manifest(all_rows, manifest_path)
        logger.info(f"Manifest saved to {manifest_path} with {len(all_rows)} verified records.")

    logger.info("=== Mass Generation Process Completed ===")