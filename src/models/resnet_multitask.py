import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class AudioDeepfakeResNet(nn.Module):
    def __init__(self, num_compression_classes=3):
        """
        Multi-task ResNet-18 for Audio Deepfake Detection and Compression Provenance.
        """
        super(AudioDeepfakeResNet, self).__init__()

        # Load a pretrained ResNet-18 backbone
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Adapt the first convolutional layer for 1-channel input (Mel Spectrogram)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )

        # Initialize the new 1-channel weights by averaging the original 3-channel RGB weights
        with torch.no_grad():
            self.backbone.conv1.weight.copy_(original_conv1.weight.mean(dim=1, keepdim=True))

        # Extract the number of features coming out of the ResNet backbone (512 for ResNet18)
        num_features = self.backbone.fc.in_features

        # Remove the original classification layer
        self.backbone.fc = nn.Identity()

        # Head 1: Binary Classification (Real vs. Fake)
        self.real_fake_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Outputs a single logit for binary classification
        )

        # Head 2: Multi-class Classification (Compression Provenance: Clean, Mild, Heavy)
        self.compression_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_compression_classes)
        )

    def forward(self, x):
        # Pass the spectrogram through the ResNet backbone
        features = self.backbone(x)

        # Pass the extracted features through both heads simultaneously
        real_fake_logits = self.real_fake_head(features)
        compression_logits = self.compression_head(features)

        return real_fake_logits, compression_logits


if __name__ == "__main__":
    # Quick sanity check to ensure the architecture works
    model = AudioDeepfakeResNet()
    dummy_spectrogram = torch.randn(16, 1, 128, 256)  # (Batch Size, Channels, Mels, Time)

    rf_out, comp_out = model(dummy_spectrogram)
    print(f"Real/Fake Output Shape: {rf_out.shape}")  # Expected: [16, 1]
    print(f"Compression Output Shape: {comp_out.shape}")  # Expected: [16, 3]
    print("Model architecture built successfully!")