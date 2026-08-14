import os
import uuid
import torch
import webbrowser
from threading import Timer
from flask import Flask, request, jsonify, render_template
from src.models.resnet_multitask import AudioDeepfakeResNet
from src.models.explain import generate_heatmap
from src.utils.logger import get_logger
from src.utils.audio_utils import load_and_preprocess_audio

logger = get_logger("flask_app")

# Assuming app.py is inside the src/ folder:
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# --- NEW: Define allowed audio formats ---
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'flac', 'aac', 'opus', 'webm', 'wma', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# -----------------------------------------

# Ensure static directories exist so the app doesn't crash on the first upload
os.makedirs("static/heatmaps", exist_ok=True)
os.makedirs("static/temp", exist_ok=True)

# 1. Load the Model (Only ONCE!)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioDeepfakeResNet(num_compression_classes=3).to(device)
weights_path = "weights/best_deepfake_resnet.pth"

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    logger.info(f"Loaded model weights from {weights_path}")
else:
    logger.warning(f"Weights not found at {weights_path}. Model will use random weights for now!")

model.eval()

# --- HTML ROUTES ---
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/detect.html')
def detect():
    return render_template('detect.html')


# --- API / INFERENCE ROUTE ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # --- NEW: Check if the file extension is allowed ---
    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    # ---------------------------------------------------

    # THE FIX 1: Generate unique IDs for every request to prevent race conditions!
    request_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename)[1].lower() or ".wav"
    temp_audio_path = f"static/temp/upload_{request_id}{ext}"
    file.save(temp_audio_path)

    try:
        # THE FIX 2: Use our centralized audio utility
        log_mel_spec = load_and_preprocess_audio(temp_audio_path, target_sr=16000, max_frames=128)
        
        if log_mel_spec is None:
            return jsonify({"error": "Failed to process audio file"}), 500
            
        # Add batch dimension: [1, 1, Mels, Time]
        input_tensor = log_mel_spec.unsqueeze(0).to(device)

        # 2. Run Inference on BOTH heads
        with torch.no_grad():
            rf_logits, comp_logits = model(input_tensor)

            # Real/Fake Prediction
            probability = torch.sigmoid(rf_logits).item()
            is_real = probability >= 0.5
            prediction = "Real" if is_real else "Fake"
            confidence = round((probability if is_real else (1 - probability)) * 100, 2)

            # Compression Provenance Prediction
            comp_class = torch.argmax(comp_logits, dim=1).item()
            comp_labels = {0: "Clean (Uncompressed)", 1: "Mild Compression", 2: "Heavy Compression"}
            provenance = comp_labels.get(comp_class, "Unknown")

        # 3. Generate Heatmap 
        # THE FIX 3: Pass the ALREADY LOADED model, saving tons of RAM!
        heatmap_path = f"static/heatmaps/heatmap_{request_id}.png"
        target_class = 1 if is_real else 0 # Explain the decision it actually made!
        
        generate_heatmap(model, temp_audio_path, heatmap_path, target_class=target_class)

        # Clean up temp audio (but keep the heatmap to display it)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        # 4. Return Data to the frontend
        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "heatmap_url": "/" + heatmap_path,
            "compression_provenance": provenance
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000')

if __name__ == '__main__':
    print("🚀 Starting Deepfake Detector Web Interface at http://127.0.0.1:5000")

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        Timer(1.25, open_browser).start()

    app.run(debug=False) 