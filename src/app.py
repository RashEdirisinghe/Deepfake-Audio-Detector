import os
import torch
import torchaudio
from flask import Flask, request, jsonify, render_template
from src.models.resnet_multitask import AudioDeepfakeResNet
from src.models.explain import generate_heatmap

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# 1. Load the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioDeepfakeResNet(num_compression_classes=3).to(device)
weights_path = "weights/best_deepfake_resnet.pth"

# We will load the weights ONLY if they exist (so the app doesn't crash before training is done)
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()


@app.route('/')
def index():
    """Serves the HTML website."""
    return render_template('index.html')


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
        # Process audio to spectrogram
        waveform, sr = torchaudio.load(temp_audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=128)(
            waveform)
        log_mel_spec = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(mel_spec)

        # Pad or truncate to 128 time frames (just like our dataset class!)
        import torch.nn.functional as F
        current_frames = log_mel_spec.shape[2]
        if current_frames < 128:
            log_mel_spec = F.pad(log_mel_spec, (0, 128 - current_frames))
        elif current_frames > 128:
            log_mel_spec = log_mel_spec[:, :, :128]

        input_tensor = log_mel_spec.unsqueeze(0).to(device)

        # Run Inference
        with torch.no_grad():
            rf_logits, _ = model(input_tensor)
            probability = torch.sigmoid(rf_logits).item()

            # Since Real is 1 and Fake is 0:
            prediction = "Real" if probability >= 0.5 else "Fake"
            confidence = round((probability if prediction == "Real" else (1 - probability)) * 100, 2)

        # Generate Heatmap
        heatmap_path = "static/heatmaps/latest_heatmap.png"
        generate_heatmap(weights_path, temp_audio_path, heatmap_path)

        # Clean up temp audio
        os.remove(temp_audio_path)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "heatmap_url": "/" + heatmap_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Deepfake Detector Web Interface at http://127.0.0.1:5000")
    app.run(debug=True)
