import torch
import torch.nn as nn
import torch.optim as optim
from src.models.resnet_multitask import AudioDeepfakeResNet
from src.data.dataset import get_dataloader
from src.utils.logger import get_logger

logger = get_logger("training_loop")


def train_model(manifest_path, spectrogram_dir, epochs=10, batch_size=16, learning_rate=1e-4):
    # 1. Setup Device (Use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training on device: {device}")

    # 2. Initialize Model, Data, and Optimizer
    model = AudioDeepfakeResNet(num_compression_classes=3).to(device)
    dataloader = get_dataloader(manifest_path, spectrogram_dir, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Define the Multi-Task Loss Functions
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_multi = nn.CrossEntropyLoss()

    logger.info(f"Beginning training for {epochs} epochs...")

    # 4. The Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (spectrograms, target_binary, target_comp) in enumerate(dataloader):
            # Move data to GPU/CPU
            spectrograms = spectrograms.to(device)
            target_binary = target_binary.to(device)
            target_comp = target_comp.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward Pass
            out_binary, out_comp = model(spectrograms)

            # Calculate Losses
            loss_binary = criterion_binary(out_binary, target_binary)
            loss_comp = criterion_multi(out_comp, target_comp)

            # Combine the losses (you can add weights here later if one task is harder)
            total_loss = loss_binary + loss_comp

            # Backward Pass and Optimize
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] - Loss: {total_loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        logger.info(f"--- Epoch {epoch + 1} Completed | Average Loss: {epoch_loss:.4f} ---")

    # 5. Save the trained weights
    torch.save(model.state_dict(), "weights/deepfake_resnet_latest.pth")
    logger.info("Training complete. Model weights saved to weights/deepfake_resnet_latest.pth")


if __name__ == "__main__":
    # Example usage (will run properly once datasets are fully prepared!)
    train_model(
        manifest_path="data/metadata/synthetic_manifest.csv",
        spectrogram_dir="data/spectrograms",
        epochs=5
    )