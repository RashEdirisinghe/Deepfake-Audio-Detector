import os
import torch
import torch.nn.functional as F
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader
from src.utils.logger import get_logger

logger = get_logger("spectrogram_dataset")

class DeepfakeSpectrogramDataset(Dataset):
    def __init__(self, manifest_path, max_time_frames=128):
        """
        Custom PyTorch Dataset for Multi-Task Audio Deepfake Detection.
        """
        if not os.path.exists(manifest_path):
            logger.error(f"Manifest not found at {manifest_path}. Creating empty dataset.")
            self.manifest = pd.DataFrame()
        else:
            self.manifest = pd.read_csv(manifest_path)
        
        self.max_time_frames = max_time_frames
        self.label_map = {"real": 1, "fake": 0}
        self.compression_map = {"clean": 0, "mild": 1, "heavy": 2}
        logger.info(f"Initialized dataset with {len(self.manifest)} records.")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        
        # 1. Parse Labels
        binary_label = self.label_map.get(str(row.get('label', 'fake')).lower(), 0)
        comp_level = str(row.get('compression_level', 'clean')).lower()
        compression_label = self.compression_map.get(comp_level, 0)
        
        # 2. Reconstruct Tensor Path based on actual extraction logic
        wav_path = str(row['file_path']).replace("\\", "/")
        tensor_path = wav_path.replace("data/compressed", "data/spectrograms/compressed") \
                              .replace("data/real", "data/spectrograms/real") \
                              .replace("data/fake", "data/spectrograms/fake") \
                              .replace(".wav", ".pt")
        
        # 3. Load and Pad/Truncate Tensor Safely
        try:
            spectrogram = torch.load(tensor_path, weights_only=True)
            
            # ResNets require identical sizes in a batch [Channels, Mels, Time]
            current_frames = spectrogram.shape[2]
            
            if current_frames < self.max_time_frames:
                # Pad with zeros on the time axis (right side)
                pad_amount = self.max_time_frames - current_frames
                spectrogram = F.pad(spectrogram, (0, pad_amount))
            elif current_frames > self.max_time_frames:
                # Truncate time axis to max_time_frames
                spectrogram = spectrogram[:, :, :self.max_time_frames]
                
        except Exception as e:
            # If a tensor is missing, return a dummy tensor so training doesn't crash
            spectrogram = torch.zeros((1, 128, self.max_time_frames))
        
        # --- THE FINAL ANTI-CHEAT: Z-SCORE NORMALIZATION ---
        # TTS is digitally loud. Microphones are quiet. 
        # This forces all audio to have the exact same average "volume"
        mean = spectrogram.mean()
        std = spectrogram.std()
        spectrogram = (spectrogram - mean) / (std + 1e-7)  # 1e-7 prevents divide-by-zero
        # ---------------------------------------------------

        # --- THE FIX 2.0: SPECAUGMENT & NOISE ---
        # 1. Add background static (Now relative to standard volume!)
        noise = torch.randn(spectrogram.size()) * 0.1
        spectrogram = spectrogram + noise
        
        # 2. SpecAugment: Randomly mask out frequencies and time chunks
        if torch.rand(1).item() < 0.7:  # 70% chance to apply
            freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
            time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)
            spectrogram = freq_mask(spectrogram)
            spectrogram = time_mask(spectrogram)
        # ----------------------------------------
        
        # 4. Format Targets for PyTorch
        target_binary = torch.tensor([binary_label], dtype=torch.float32)
        target_compression = torch.tensor(compression_label, dtype=torch.long)
        
        return spectrogram, target_binary, target_compression

def get_dataloader(manifest_path, spectrogram_base_dir, batch_size=16, shuffle=True):
    dataset = DeepfakeSpectrogramDataset(manifest_path)
    # Set num_workers=0 to prevent multiprocessing crashes on Windows
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)