# Compression-Aware Audio Deepfake Detection

This project detects synthetic speech in low-resource languages (Sinhala and Tamil) under realistic compression conditions.

## Project Structure
* `data/`: Raw and processed audio datasets (Ignored by Git).
* `src/data/`: Data loading and synthesis scripts.
* `src/utils/`: Shared utilities for logging and audio I/O.
* `weights/`: Model checkpoints.
* `config.yaml`: Centralized configuration file.

## Setup Instructions

1. Install requirements:
   ```bash
   pip install -r requirements.txt