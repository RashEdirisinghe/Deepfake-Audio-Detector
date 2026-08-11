import os
import glob
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.utils.logger import get_logger

logger = get_logger("master_manifest")

def scan_audio_files(directory, label, compression_level):
    """Scans a directory for audio files (.wav, .mp3, .m4a) and assigns a group_id based on base filename."""
    records = []
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return records

    extensions = ("*.wav", "*.mp3", "*.m4a", "*.ogg", "*.flac")
    for ext in extensions:
        files = glob.glob(os.path.join(directory, "**", ext), recursive=True)
        for file_path in files:
            file_path = file_path.replace("\\", "/")
            filename = os.path.basename(file_path)
            
            if "sinhala" in file_path.lower() or "sin" in filename.lower():
                language = "Sinhala"
            elif "tamil" in file_path.lower() or "tam" in filename.lower():
                language = "Tamil"
            else:
                language = "Unknown"

            # Group ID is the base filename without extensions/compression suffixes
            # This guarantees Clean, Mild, Heavy, MP3, and M4A versions of the SAME file stay in the SAME split
            base_id = os.path.splitext(filename)[0]

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

    # 1. Clean Audio (Strictly using our new balanced clean_balanced folders)
    all_records.extend(scan_audio_files("data/clean_balanced/real_tamil", label="real", compression_level="clean"))
    all_records.extend(scan_audio_files("data/clean_balanced/real_sinhala", label="real", compression_level="clean"))
    all_records.extend(scan_audio_files("data/clean_balanced/fake_tamil", label="fake", compression_level="clean"))
    all_records.extend(scan_audio_files("data/clean_balanced/fake_sinhala", label="fake", compression_level="clean"))

    # 1b. Opus Compressed Audio (Mild & Heavy)
    all_records.extend(scan_audio_files("data/compressed/real_tamil_mild", label="real", compression_level="mild"))
    all_records.extend(scan_audio_files("data/compressed/real_sinhala_mild", label="real", compression_level="mild"))
    all_records.extend(scan_audio_files("data/compressed/real_tamil_heavy", label="real", compression_level="heavy"))
    all_records.extend(scan_audio_files("data/compressed/real_sinhala_heavy", label="real", compression_level="heavy"))

    all_records.extend(scan_audio_files("data/compressed/fake_tamil_mild", label="fake", compression_level="mild"))
    all_records.extend(scan_audio_files("data/compressed/fake_sinhala_mild", label="fake", compression_level="mild"))
    all_records.extend(scan_audio_files("data/compressed/fake_tamil_heavy", label="fake", compression_level="heavy"))
    all_records.extend(scan_audio_files("data/compressed/fake_sinhala_heavy", label="fake", compression_level="heavy"))

    # 2. Universal Augmented Folders (Native MP3 & M4A)
    all_records.extend(scan_audio_files("data/augmented/real_tamil_mp3", label="real", compression_level="mild"))
    all_records.extend(scan_audio_files("data/augmented/real_tamil_m4a", label="real", compression_level="mild"))
    all_records.extend(scan_audio_files("data/augmented/real_sinhala_mp3", label="real", compression_level="mild"))
    all_records.extend(scan_audio_files("data/augmented/real_sinhala_m4a", label="real", compression_level="mild"))

    all_records.extend(scan_audio_files("data/augmented/fake_tamil_mp3", label="fake", compression_level="mild"))
    all_records.extend(scan_audio_files("data/augmented/fake_tamil_m4a", label="fake", compression_level="mild"))
    all_records.extend(scan_audio_files("data/augmented/fake_sinhala_mp3", label="fake", compression_level="mild"))
    all_records.extend(scan_audio_files("data/augmented/fake_sinhala_m4a", label="fake", compression_level="mild"))

    df = pd.DataFrame(all_records)
    if df.empty:
        logger.error("No audio records found across directories!")
        return

    logger.info(f"Total audio samples indexed: {len(df)}")

    # Create a composite stratification target
    df['stratify_target'] = df['label'] + "_" + df['language']

    # n_splits=5 perfectly creates an 80% Train / 20% Val split
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    train_idx, val_idx = next(sgkf.split(df, y=df['stratify_target'], groups=df['group_id']))

    train_df = df.iloc[train_idx].drop(columns=['group_id', 'stratify_target'])
    val_df = df.iloc[val_idx].drop(columns=['group_id', 'stratify_target'])

    os.makedirs("data/metadata", exist_ok=True)
    train_path = "data/metadata/train_manifest.csv"
    val_path = "data/metadata/val_manifest.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logger.info(f"Stratified Group Train split ({len(train_df)} samples) saved to {train_path}")
    logger.info(f"Stratified Group Validation split ({len(val_df)} samples) saved to {val_path}")

if __name__ == "__main__":
    build_master_manifests()