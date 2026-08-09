import os
import torch
import torchaudio
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template

from src.models.resnet_multitask import AudioDeepfakeResNet
from src.models.explain import generate_heatmap
from src.utils.logger import get_logger

logger = get_logger("flask_app")

# Assuming app.py is inside the src/ folder:
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Ensure static directories exist so the app doesn't crash on the first upload
os.makedirs("static/heatmaps", exist_ok=True)

# 1. Load the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioDeepfakeResNet(num_compression_classes=3).to(device)
weights_path = "weights/best_deepfake_resnet.pth"

# We will load the weights ONLY if they exist
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    logger.info(f"Loaded model weights from {weights_path}")
else:
    logger.warning(f"Weights not found at {weights_path}. Model will use random weights!")

model.eval()


# --- HTML ROUTES ---

@app.route('/')
@app.route('/index.html')
def index():
    """Serves the Home page."""
    return render_template('index.html')


@app.route('/about.html')
def about():
    """Serves the About page."""
    return render_template('about.html')


@app.route('/detect.html')
def detect():
    """Serves the Detector tool page."""
    return render_template('detect.html')


# --- API / INFERENCE ROUTE ---

@app.route('/predict', methods=['POST'])
def predict():
    """Receives audio, runs inference, and generates heatmap."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file temporarily
    temp_audio_path = "static/temp_upload.wav"
    file.save(temp_audio_path)

    try:
        # 1. Process audio to spectrogram (Must match dataset.py exactly!)
        waveform, sr = torchaudio.load(temp_audio_path)

        # Resample to 16kHz
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Generate Mel Spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=2048, hop_length=512, n_mels=128
        )(waveform)
        log_mel_spec = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(mel_spec)

        # Pad or truncate to 128 time frames
        current_frames = log_mel_spec.shape[2]
        if current_frames < 128:
            log_mel_spec = F.pad(log_mel_spec, (0, 128 - current_frames))
        elif current_frames > 128:
            log_mel_spec = log_mel_spec[:, :, :128]

        # --- INFERENCE FIX: Z-Score Normalization ---
        mean = log_mel_spec.mean()
        std = log_mel_spec.std()
        log_mel_spec = (log_mel_spec - mean) / (std + 1e-7)

        # Add batch dimension: [1, Channels, Mels, Time]
        input_tensor = log_mel_spec.unsqueeze(0).to(device)

        # 2. Run Inference on BOTH heads
        with torch.no_grad():
            rf_logits, comp_logits = model(input_tensor)

            # Real/Fake Prediction
            probability = torch.sigmoid(rf_logits).item()
            prediction = "Real" if probability >= 0.5 else "Fake"
            confidence = round((probability if prediction == "Real" else (1 - probability)) * 100, 2)

            # Compression Provenance Prediction
            comp_class = torch.argmax(comp_logits, dim=1).item()
            comp_labels = {0: "Clean (Uncompressed)", 1: "Mild Compression", 2: "Heavy Compression"}
            provenance = comp_labels.get(comp_class, "Unknown")

        # 3. Generate Heatmap
        heatmap_path = "static/heatmaps/latest_heatmap.png"
        generate_heatmap(weights_path, temp_audio_path, heatmap_path)

        # Clean up temp audio
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        # 4. Return Data to the detect.js frontend
        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "heatmap_url": "/" + heatmap_path,
            "compression_provenance": provenance
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Deepfake Detector Web Interface at http://127.0.0.1:5000")
    app.run(debug=True)