# Cadence: Compression-Aware Audio Deepfake Forensics

Cadence is an end-to-end, explainable audio deepfake detection system optimized for low-resource South Asian languages (Sinhala and Tamil). 

Unlike standard deepfake detectors that fail on compressed audio or cheat using background noise, Cadence utilizes a multi-task ResNet-18 architecture to simultaneously classify audio authenticity and compression provenance (Clean, Mild, Heavy). It includes active artifact regularization (VAD and noise-floor injection) and provides visual forensic proof via Grad-CAM heatmaps.

## 🚀 Key Features
* **Universal Format Support:** Natively ingests and evaluates `.wav`, `.mp3`, and `.m4a` (AAC) files.
* **Shortcut Mitigation:** Automated Voice Activity Detection (VAD) and universal white-noise injection prevent the model from exploiting digital silence.
* **Multi-Task ResNet-18:** Simultaneously predicts Real/Fake probabilities and compression states.
* **Explainable AI (XAI):** Generates thread-safe, frequency-aligned Grad-CAM heatmaps.
* **Forensic Dashboard:** Interactive Flask web UI with dynamic text summaries and UUID-isolated sessions.

## 📂 Project Structure
* `data/`: Raw audio, balanced subsets, augmented files, and spectrogram `.pt` tensors (Ignored by Git).
* `src/data/`: Scripts for dataset balancing, FFmpeg codec augmentation, manifest generation, and tensor extraction.
* `src/models/`: The PyTorch ResNet-18 architecture, training loops, and Grad-CAM explainer.
* `src/utils/`: Centralized audio processing (`audio_utils.py`) and logging utilities.
* `src/app.py`: The Flask WSGI backend server.
* `static/` & `templates/`: Frontend HTML, CSS, and JS for the web dashboard.
* `weights/`: Saved model checkpoints (`best_deepfake_resnet.pth`).
* `config.yaml`: Centralized configuration for hyperparameters and paths.

## ⚙️ Setup Instructions

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt