import os
import glob
import torch
import torchaudio
from src.utils.logger import get_logger

logger = get_logger("spectrogram_extractor")


def audio_to_mel_spectrogram(audio_path, target_sr=16000, n_mels=128, n_fft=2048, hop_length=512):
    """Loads an audio file and converts it to a log-Mel spectrogram tensor."""
    try:
        waveform, sr = torchaudio.load(audio_path)

        # Resample if the audio isn't 16kHz
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        # Convert stereo to mono (ResNets expect a 1-channel input)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Generate the Mel Spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        mel_spec = mel_transform(waveform)

        # Convert to Log-Mel (Decibel scale) for better model stability
        db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        log_mel_spec = db_transform(mel_spec)

        return log_mel_spec

    except Exception as e:
        logger.error(f"Failed to process {audio_path}: {e}")
        return None


def process_directory(input_dir, output_dir):
    """Processes all WAV files in a directory and saves them as PyTorch tensors."""
    os.makedirs(output_dir, exist_ok=True)
    audio_files = glob.glob(os.path.join(input_dir, "*.wav"))

    if not audio_files:
        logger.warning(f"No WAV files found in {input_dir}")
        return

    logger.info(f"Extracting spectrograms for {len(audio_files)} files in {input_dir}...")

    for file_path in audio_files:
        # Change the extension from .wav to .pt
        filename = os.path.basename(file_path).replace(".wav", ".pt")
        out_path = os.path.join(output_dir, filename)

        # Skip if already processed (great for if the script crashes and you have to restart)
        if os.path.exists(out_path):
            continue

        spec_tensor = audio_to_mel_spectrogram(file_path)
        if spec_tensor is not None:
            torch.save(spec_tensor, out_path)

    logger.info(f"Finished processing {input_dir}. Tensors saved to {output_dir}")


if __name__ == "__main__":
    logger.info("=== Starting Mel Spectrogram Extraction Pipeline ===")

    # We will expand this list later once the real data and compressed data are ready!
    directories_to_process = [
        ("data/fake/tamil", "data/spectrograms/fake/tamil"),
        ("data/fake/sinhala", "data/spectrograms/fake/sinhala")
    ]

    for in_dir, out_dir in directories_to_process:
        process_directory(in_dir, out_dir)

    logger.info("=== Spectrogram Extraction Complete ===")