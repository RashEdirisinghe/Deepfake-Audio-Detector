import os
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.utils.logger import get_logger

logger = get_logger("spectrogram_dataset")


class DeepfakeSpectrogramDataset(Dataset):
    def __init__(self, manifest_path, spectrogram_base_dir, target_length=128):
        """
        Custom PyTorch Dataset for Multi-Task Audio Deepfake Detection.

        Args:
            manifest_path (str): Path to the CSV manifest.
            spectrogram_base_dir (str): Root directory where .pt files are stored.
            target_length (int): Fixed width/length for spectrogram tensors.
        """
        if not os.path.exists(manifest_path):
            logger.error(f"Manifest not found at {manifest_path}. Creating empty dataset.")
            self.manifest = pd.DataFrame()
        else:
            self.manifest = pd.read_csv(manifest_path)

        self.spectrogram_base_dir = spectrogram_base_dir
        self.target_length = target_length

        # Label mappings based on project scope
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

        # 2. Reconstruct Tensor Path
        # The manifest points to a .wav, but we need the .pt file!
        wav_name = os.path.basename(row['file_path'])
        pt_name = wav_name.replace(".wav", ".pt")

        folder_type = "fake" if binary_label == 0 else "real"
        lang_folder = str(row.get('language', 'unknown')).lower()

        tensor_path = os.path.join(self.spectrogram_base_dir, folder_type, lang_folder, pt_name)

        # 3. Load Tensor Safely & Normalize Size
        try:
            spectrogram = torch.load(tensor_path, weights_only=True)

            # Ensure 3D tensor shape [1, 128, length]
            if spectrogram.dim() == 2:
                spectrogram = spectrogram.unsqueeze(0)

            # Pad or Crop time dimension (Width) so every tensor is fixed to target_length
            c, h, w = spectrogram.shape
            if w < self.target_length:
                spectrogram = F.pad(spectrogram, (0, self.target_length - w))
            elif w > self.target_length:
                spectrogram = spectrogram[:, :, :self.target_length]

        except Exception as e:
            # If a tensor is missing, return a dummy tensor so the training loop doesn't crash
            spectrogram = torch.zeros((1, 128, self.target_length))

        # 4. Format Targets
        target_binary = torch.tensor([binary_label], dtype=torch.float32)
        target_compression = torch.tensor(compression_label, dtype=torch.long)
        
        # --- THE FIX 2.0: SPECAUGMENT & NOISE ---
        import torchaudio
        
        # 1. Add background static
        noise = torch.randn(spectrogram.size()) * 0.05
        spectrogram = spectrogram + noise
        
        # 2. SpecAugment: Randomly mask out frequencies and time chunks
        # This prevents the model from memorizing specific microphone or TTS frequencies
        if torch.rand(1).item() < 0.7:  # 70% chance to apply
            freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
            time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)
            spectrogram = freq_mask(spectrogram)
            spectrogram = time_mask(spectrogram)
        # ----------------------------------------
        
        return spectrogram, target_binary, target_compression


def get_dataloader(manifest_path, spectrogram_base_dir, batch_size=16, shuffle=True):
    dataset = DeepfakeSpectrogramDataset(manifest_path, spectrogram_base_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


if __name__ == "__main__":
    # Test script initialization
    logger.info("Dataset and DataLoader module built successfully!")