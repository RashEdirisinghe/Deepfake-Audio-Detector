import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score, f1_score
from src.models.resnet_multitask import AudioDeepfakeResNet
from src.data.dataset import get_dataloader
from src.utils.logger import get_logger

logger = get_logger("training_loop")


def train_and_validate(train_manifest, val_manifest, spectrogram_dir, epochs=10, batch_size=16, learning_rate=1e-4):
    # 1. Initialize Weights & Biases
    wandb.init(
        project="Compression-Aware-Deepfake-Detection",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "architecture": "ResNet-18 Multitask",
            "dataset": "Sinhala & Tamil Audio"
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training on device: {device}")

    model = AudioDeepfakeResNet(num_compression_classes=3).to(device)
    train_loader = get_dataloader(train_manifest, spectrogram_dir, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_manifest, spectrogram_dir, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_multi = nn.CrossEntropyLoss()

    best_f1 = 0.0

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

            # Log batch-level loss to wandb
            wandb.log({"batch_loss": loss.item()})

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
        epoch_loss = running_loss / len(train_loader)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Log epoch-level metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_accuracy": val_acc,
            "val_f1_score": val_f1
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "weights/best_deepfake_resnet.pth")
            logger.info("--> New best model saved!")

    # Close the wandb run
    wandb.finish()
    logger.info("Training complete.")


if __name__ == "__main__":
    logger.info("Training script ready to run.")