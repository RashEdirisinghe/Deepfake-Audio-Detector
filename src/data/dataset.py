import os
import torch
import torch.nn.functional as F
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader
from src.utils.logger import get_logger

logger = get_logger("spectrogram_dataset")

class DeepfakeSpectrogramDataset(Dataset):
    def __init__(self, manifest_path, is_training=False, max_time_frames=128):
        """
        Custom PyTorch Dataset for Multi-Task Audio Deepfake Detection.
        """
        if not os.path.exists(manifest_path):
            logger.error(f"Manifest not found at {manifest_path}. Creating empty dataset.")
            self.manifest = pd.DataFrame()
        else:
            self.manifest = pd.read_csv(manifest_path)
        
        self.max_time_frames = max_time_frames
        self.is_training = is_training
        self.label_map = {"real": 1, "fake": 0}
        self.compression_map = {"clean": 0, "mild": 1, "heavy": 2}
        
        mode = "TRAINING" if is_training else "VALIDATION"
        logger.info(f"Initialized {mode} dataset with {len(self.manifest)} records.")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        
        # 1. Parse Labels
        binary_label = self.label_map.get(str(row.get('label', 'fake')).lower(), 0)
        comp_level = str(row.get('compression_level', 'clean')).lower()
        compression_label = self.compression_map.get(comp_level, 0)
        
        # 2. Reconstruct Tensor Path based on universal extraction logic
        wav_path = str(row['file_path']).replace("\\", "/")
        
        # Map all source root folders to their respective spectrogram directories
        tensor_path = wav_path.replace("data/clean_balanced", "data/spectrograms/clean_balanced") \
                              .replace("data/compressed", "data/spectrograms/compressed") \
                              .replace("data/augmented", "data/spectrograms/augmented") \
                              .replace("data/real", "data/spectrograms/real") \
                              .replace("data/fake", "data/spectrograms/fake")
        
        # Safely swap out any audio extension (.wav, .mp3, .m4a, etc.) for .pt
        for ext in [".wav", ".mp3", ".m4a", ".ogg", ".flac"]:
            if tensor_path.endswith(ext):
                tensor_path = tensor_path[:-len(ext)] + ".pt"
                break
        
        # 3. Load and Pad/Truncate Tensor Safely
        try:
            spectrogram = torch.load(tensor_path, weights_only=True)
            
            current_frames = spectrogram.shape[2]
            
            # Safety net: pad/truncate if not already 128
            if current_frames < self.max_time_frames:
                pad_amount = self.max_time_frames - current_frames
                spectrogram = F.pad(spectrogram, (0, pad_amount))
            elif current_frames > self.max_time_frames:
                spectrogram = spectrogram[:, :, :self.max_time_frames]
                
        except Exception as e:
            # Fallback zero tensor if file is missing or corrupted
            spectrogram = torch.zeros((1, 128, self.max_time_frames))
        
        # Safety net: Z-Score Normalization
        mean = spectrogram.mean()
        std = spectrogram.std()
        spectrogram = (spectrogram - mean) / (std + 1e-7) 

        # Only apply noise and SpecAugment during TRAINING!
        if self.is_training:
            # 1. Add background static 
            noise = torch.randn(spectrogram.size()) * 0.1
            spectrogram = spectrogram + noise
            
            # 2. SpecAugment: Randomly mask out frequencies and time chunks
            if torch.rand(1).item() < 0.7:  
                freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
                time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)
                spectrogram = freq_mask(spectrogram)
                spectrogram = time_mask(spectrogram)
        
        # 4. Format Targets for PyTorch
        target_binary = torch.tensor(binary_label, dtype=torch.float32)
        target_compression = torch.tensor(compression_label, dtype=torch.long)
        
        return spectrogram, target_binary, target_compression

def get_dataloader(manifest_path, spectrogram_base_dir, batch_size=16, shuffle=True):
    dataset = DeepfakeSpectrogramDataset(manifest_path, is_training=shuffle)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)