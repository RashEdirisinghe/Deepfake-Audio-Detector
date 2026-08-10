import torch
import torchaudio
import torch.nn.functional as F
import librosa
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("audio_utils")

def load_and_preprocess_audio(audio_path, target_sr: int = 16000, max_frames: int = 128) -> torch.Tensor:
    """
    Centralized audio loader. Handles safe librosa loading (supports mp3, m4a, wav), 
    VAD trimming, noise floor equalization, mono-conversion, Mel generation, 
    and Z-score normalization.
    """
    try:
        # 1. Load with librosa (This natively handles all major audio formats!)
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        if len(y) == 0:
            logger.error(f"Audio file is empty: {audio_path}")
            return None

        # --- THE FIX 1: Voice Activity Detection (VAD) ---
        # Trim leading and trailing silence (anything quieter than 30dB below peak)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        if len(y_trimmed) > target_sr * 0.5:
            # Ensure at least 0.5s remains
            y = y_trimmed

        # --- THE FIX 2: Universal Noise Floor ---
        # Inject a microscopic layer of white noise to eliminate 0.00dB digital silence.
        max_amp = np.amax(np.abs(y))
        if max_amp == 0:
            max_amp = 1.0  # SAFETY FIX: Prevent scaling by 0 if file is pure silence
            
        noise_amp = 0.005 * max_amp
        y = y + noise_amp * np.random.normal(size=y.shape)

        # 2. Convert to tensor
        waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        # 3. Convert stereo to mono if needed (redundant but safe)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 4. Generate log-Mel Spectrogram using torchaudio (Preserves your training math!)
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr, n_fft=2048, hop_length=512, n_mels=128
        )(waveform)
        log_mel_spec = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(mel_spec)

        # 5. Pad or Truncate to max_frames (128)
        current_frames = log_mel_spec.shape[2]
        if current_frames < max_frames:
            log_mel_spec = F.pad(log_mel_spec, (0, max_frames - current_frames))
        elif current_frames > max_frames:
            log_mel_spec = log_mel_spec[:, :, :max_frames]

        # 6. Z-Score Normalization
        mean = log_mel_spec.mean()
        std = log_mel_spec.std()
        log_mel_spec = (log_mel_spec - mean) / (std + 1e-7)

        return log_mel_spec

    except Exception as e:
        logger.error(f"Failed to process {audio_path}: {e}")
        return None