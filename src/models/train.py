import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from src.models.resnet_multitask import AudioDeepfakeResNet
from src.data.dataset import get_dataloader
from src.utils.logger import get_logger

logger = get_logger("training_loop")


def plot_metrics(history, epochs, output_dir="plots"):
    """Generates and saves training metric graphs using matplotlib."""
    os.makedirs(output_dir, exist_ok=True)
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))

    # Graph 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='red', marker='o')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Graph 2: Validation Accuracy & F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['val_acc'], label='Val Accuracy', color='blue', marker='o')
    plt.plot(epochs_range, history['val_f1'], label='Val F1 Score', color='green', marker='s')
    plt.title('Validation Metrics over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()

    plot_path = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"--> Training metrics graph successfully saved to {plot_path}")


def train_and_validate(train_manifest, val_manifest, spectrogram_dir, epochs=10, batch_size=16, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training on device: {device}")

    model = AudioDeepfakeResNet(num_compression_classes=3).to(device)
    train_loader = get_dataloader(train_manifest, spectrogram_dir, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_manifest, spectrogram_dir, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_multi = nn.CrossEntropyLoss()

    best_f1 = 0.0

    # Dictionary to track metrics for our matplotlib graphs
    history = {"train_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(epochs):
        # =======================
        #      TRAINING PHASE
        # =======================
        model.train()
        running_loss = 0.0

        for batch_idx, (spectrograms, target_binary, target_comp) in enumerate(train_loader):
            spectrograms, target_binary, target_comp = spectrograms.to(device), target_binary.to(
                device), target_comp.to(device)

            optimizer.zero_grad()
            out_binary, out_comp = model(spectrograms)

            loss = criterion_binary(out_binary, target_binary) + criterion_multi(out_comp, target_comp)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        history["train_loss"].append(epoch_loss)

        # =======================
        #     VALIDATION PHASE
        # =======================
        model.eval()
        all_rf_preds, all_rf_targets = [], []

        with torch.no_grad():
            for spectrograms, target_binary, target_comp in val_loader:
                spectrograms, target_binary, target_comp = spectrograms.to(device), target_binary.to(
                    device), target_comp.to(device)

                out_binary, _ = model(spectrograms)

                rf_preds = torch.sigmoid(out_binary).round().cpu().numpy()
                all_rf_preds.extend(rf_preds)
                all_rf_targets.extend(target_binary.cpu().numpy())

        val_acc = accuracy_score(all_rf_targets, all_rf_preds)
        val_f1 = f1_score(all_rf_targets, all_rf_preds, zero_division=0)

        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            # Ensure the weights directory exists before saving
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/best_deepfake_resnet.pth")
            logger.info("--> New best model saved!")

    # Generate the graphs once training is complete!
    plot_metrics(history, epochs)
    logger.info("Training complete.")


if __name__ == "__main__":
    logger.info("Training script ready to run.")