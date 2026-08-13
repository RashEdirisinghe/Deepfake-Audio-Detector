import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from src.models.resnet_multitask import AudioDeepfakeResNet
from src.data.dataset import get_dataloader
from src.utils.logger import get_logger

logger = get_logger("training_loop")

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --- NEW METRIC: Expected Calibration Error (ECE) ---
def calculate_ece(probs, labels, n_bins=10):
    """Calculates the Expected Calibration Error to see if confidence scores are trustworthy."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

# --- NEW METRIC: Equal Error Rate (EER) ---
def calculate_eer(labels, probs):
    """Calculates the Equal Error Rate, a standard forensic biometric metric."""
    fpr, tpr, _ = roc_curve(labels, probs)
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    return eer


def plot_metrics(history, epochs, output_dir="plots"):
    """Generates and saves training metric graphs using matplotlib."""
    os.makedirs(output_dir, exist_ok=True)
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(18, 5))

    # Graph 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='red', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.grid(True)
    plt.legend()

    # Graph 2: Accuracy & F1
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['val_acc'], label='Deepfake Acc', color='blue', marker='o')
    plt.plot(epochs_range, history['val_comp_acc'], label='Compression Acc', color='purple', marker='^')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.grid(True)
    plt.legend()
    
    # Graph 3: Forensic Metrics (EER & ECE)
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['val_eer'], label='EER (Lower is better)', color='orange', marker='s')
    plt.plot(epochs_range, history['val_ece'], label='ECE (Lower is better)', color='green', marker='x')
    plt.title('Forensic & Calibration Metrics')
    plt.xlabel('Epochs')
    plt.grid(True)
    plt.legend()

    plot_path = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"--> Training metrics graph successfully saved to {plot_path}")


def train_and_validate(train_manifest, val_manifest, spectrogram_dir, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training on device: {device}")

    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']

    model = AudioDeepfakeResNet(num_compression_classes=3).to(device)
    train_loader = get_dataloader(train_manifest, spectrogram_dir, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_manifest, spectrogram_dir, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Binary loss for Deepfake detection, CrossEntropy for Compression
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_multi = nn.CrossEntropyLoss()

    best_f1 = 0.0

    history = {"train_loss": [], "val_acc": [], "val_f1": [], "val_comp_acc": [], "val_eer": [], "val_ece": []}

    for epoch in range(epochs):
        # =======================
        #        TRAINING 
        # =======================
        model.train()
        running_loss = 0.0

        for spectrograms, target_binary, target_comp in train_loader:
            spectrograms = spectrograms.to(device)
            target_binary = target_binary.to(device).float().unsqueeze(1) # Fix shape for BCE
            target_comp = target_comp.to(device)

            optimizer.zero_grad()
            out_binary, out_comp = model(spectrograms)

            # THE FIX: Multi-task Loss Weighting (0.5 for compression)
            loss_binary = criterion_binary(out_binary, target_binary)
            loss_comp = criterion_multi(out_comp, target_comp)
            loss = loss_binary + (0.5 * loss_comp) 
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        history["train_loss"].append(epoch_loss)

        # =======================
        #       VALIDATION 
        # =======================
        model.eval()
        all_rf_probs, all_rf_targets = [], []
        all_comp_preds, all_comp_targets = [], []

        with torch.no_grad():
            for spectrograms, target_binary, target_comp in val_loader:
                spectrograms = spectrograms.to(device)
                
                out_binary, out_comp = model(spectrograms)

                # Real/Fake Probabilities
                rf_probs = torch.sigmoid(out_binary).cpu().numpy().flatten()
                all_rf_probs.extend(rf_probs)
                all_rf_targets.extend(target_binary.numpy())
                
                # Compression Predictions
                comp_preds = torch.argmax(out_comp, dim=1).cpu().numpy()
                all_comp_preds.extend(comp_preds)
                all_comp_targets.extend(target_comp.numpy())

        # Convert to numpy arrays for sklearn
        all_rf_targets = np.array(all_rf_targets)
        all_rf_probs = np.array(all_rf_probs)
        all_rf_preds = (all_rf_probs >= 0.5).astype(int)

        # Calculate all metrics
        val_acc = accuracy_score(all_rf_targets, all_rf_preds)
        val_f1 = f1_score(all_rf_targets, all_rf_preds, zero_division=0)
        val_comp_acc = accuracy_score(all_comp_targets, all_comp_preds)
        
        # Calculate specialized forensic metrics
        val_eer = calculate_eer(all_rf_targets, all_rf_probs)
        val_ece = calculate_ece(all_rf_probs, all_rf_targets)

        # Update History
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_comp_acc"].append(val_comp_acc)
        history["val_eer"].append(val_eer)
        history["val_ece"].append(val_ece)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | "
            f"DF Acc: {val_acc:.4f} | Comp Acc: {val_comp_acc:.4f} | EER: {val_eer:.4f} | ECE: {val_ece:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/best_deepfake_resnet.pth")
            logger.info("--> New best model saved!")

    plot_metrics(history, epochs)
    logger.info("Training complete.")


if __name__ == "__main__":
    logger.info("=== Starting Multi-Task Deepfake Detector Training ===")
    config = load_config()
    
    train_and_validate(
        train_manifest="data/metadata/train_manifest.csv",  
        val_manifest="data/metadata/val_manifest.csv",    
        spectrogram_dir="data/spectrograms",
        config=config
    )