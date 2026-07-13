import os
import random
import glob
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.utils.logger import get_logger

logger = get_logger("master_manifest")

def scan_audio_files(directory, label, compression_level):
    """Scans a directory for WAV files and assigns a group_id based on base filename."""
    records = []
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return records

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                
                if "sinhala" in file_path.lower() or "sin" in file.lower():
                    language = "Sinhala"
                elif "tamil" in file_path.lower() or "tam" in file.lower():
                    language = "Tamil"
                else:
                    language = "Unknown"

                # Group ID is the base filename without extensions/compression suffixes
                base_id = os.path.splitext(file)[0]

                records.append({
                    "file_path": file_path,
                    "language": language,
                    "label": label,
                    "compression_level": compression_level,
                    "group_id": base_id
                })
    return records

def build_master_manifests():
    logger.info("=== Scanning Audio Repositories for Master Manifest ===")
    all_records = []

    # 1. Real Audio (Clean - Strictly Sampled)
    real_tamil_clean = scan_audio_files("data/real/tamil", label="real", compression_level="clean")
    all_records.extend(random.sample(real_tamil_clean, 2000) if len(real_tamil_clean) > 2000 else real_tamil_clean)
    
    real_sinhala_clean = scan_audio_files("data/real/sinhala", label="real", compression_level="clean")
    all_records.extend(random.sample(real_sinhala_clean, 2000) if len(real_sinhala_clean) > 2000 else real_sinhala_clean)

    # 1b. Real Audio (Compressed)
    all_records.extend(scan_audio_files("data/compressed/real_tamil_mild", label="real", compression_level="mild"))
    all_records.extend(scan_audio_files("data/compressed/real_sinhala_mild", label="real", compression_level="mild"))
    all_records.extend(scan_audio_files("data/compressed/real_tamil_heavy", label="real", compression_level="heavy"))
    all_records.extend(scan_audio_files("data/compressed/real_sinhala_heavy", label="real", compression_level="heavy"))

    # 2. Fake Audio (Clean)
    all_records.extend(scan_audio_files("data/fake/tamil", label="fake", compression_level="clean"))
    all_records.extend(scan_audio_files("data/fake/sinhala", label="fake", compression_level="clean"))

    # 3. Compressed Fake Audio (Mild & Heavy)
    all_records.extend(scan_audio_files("data/compressed/fake_tamil_mild", label="fake", compression_level="mild"))
    all_records.extend(scan_audio_files("data/compressed/fake_sinhala_mild", label="fake", compression_level="mild"))
    all_records.extend(scan_audio_files("data/compressed/fake_tamil_heavy", label="fake", compression_level="heavy"))
    all_records.extend(scan_audio_files("data/compressed/fake_sinhala_heavy", label="fake", compression_level="heavy"))

    df = pd.DataFrame(all_records)
    if df.empty:
        logger.error("No audio records found across directories!")
        return

    logger.info(f"Total audio samples indexed: {len(df)}")

    # GROUP-BASED SPLIT: Ensures clean & compressed variants of the same file stay together!
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['group_id']))

    train_df = df.iloc[train_idx].drop(columns=['group_id'])
    val_df = df.iloc[val_idx].drop(columns=['group_id'])

    os.makedirs("data/metadata", exist_ok=True)
    train_path = "data/metadata/train_manifest.csv"
    val_path = "data/metadata/val_manifest.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logger.info(f"Strict Group Train split ({len(train_df)} samples) saved to {train_path}")
    logger.info(f"Strict Group Validation split ({len(val_df)} samples) saved to {val_path}")

if __name__ == "__main__":
    build_master_manifests()