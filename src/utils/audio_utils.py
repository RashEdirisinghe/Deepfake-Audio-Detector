import torchaudio
import torch


def load_and_resample(file_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Loads an audio file and resamples it to target sampling rate."""
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert stereo to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform


def get_audio_duration(file_path: str) -> float:
    """Returns duration of an audio file in seconds."""
    info = torchaudio.info(file_path)
    return info.num_frames / info.sample_rate