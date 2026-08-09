import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.models.resnet_multitask import AudioDeepfakeResNet
from src.utils.logger import get_logger

logger = get_logger("grad_cam_explainer")


def generate_heatmap(model_path, audio_path, output_path):
    """Generates a Grad-CAM heatmap for a given audio file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the trained model
    model = AudioDeepfakeResNet(num_compression_classes=3).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        logger.error(f"Model weights not found at {model_path}")
        return

    # 2. Process the input audio into a spectrogram
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=128)(
            waveform)
        log_mel_spec = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(mel_spec)

        # Add batch dimension: [1, 1, 128, Time]
        input_tensor = log_mel_spec.unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Failed to process audio {audio_path}: {e}")
        return

    # 3. Setup Grad-CAM
    # We target the last convolutional layer of the ResNet backbone
    target_layers = [model.backbone.layer4[-1]]

    # Initialize CAM (using the real/fake head for the explanation)
    cam = GradCAM(model=model, target_layers=target_layers)

    # 4. Generate the Heatmap
    # We ask Grad-CAM to explain the 'fake' class (which is index 0 in our binary setup)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    # Normalize the original spectrogram to act as the background image [0, 1]
    spec_image = input_tensor.squeeze().cpu().numpy()
    spec_image = (spec_image - spec_image.min()) / (spec_image.max() - spec_image.min())
    # Convert 1-channel to 3-channel RGB for visualization
    spec_image_rgb = np.stack([spec_image] * 3, axis=-1)

    # Overlay the heatmap
    visualization = show_cam_on_image(spec_image_rgb, grayscale_cam, use_rgb=True)

    # 5. Save the output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.imshow(visualization)
    plt.title(f"Grad-CAM Heatmap: {os.path.basename(audio_path)}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Heatmap successfully saved to {output_path}")


if __name__ == "__main__":
    logger.info("=== Generating Grad-CAM Heatmaps ===")
    
    # Path to the weights your training script is currently saving!
    model_weights = "weights/best_deepfake_resnet.pth"
    
    # Let's test it on one Real file and one Fake file
    # (Note: Check your data/real/tamil/ folder and replace 'real_00000.wav' with a filename that actually exists there!)
    test_files = [
        ("data/real/tamil/real_00000.wav", "plots/heatmap_real_tamil.png"),
        ("data/fake/tamil/fake_tam_0001.wav", "plots/heatmap_fake_tamil.png"),
        ("data/compressed/fake_tamil_heavy/fake_tam_0001.wav", "plots/heatmap_fake_tamil_heavy.png")
    ]
    
    for audio_in, image_out in test_files:
        if os.path.exists(audio_in):
            logger.info(f"Processing heatmap for {audio_in}...")
            generate_heatmap(model_weights, audio_in, image_out)
        else:
            logger.warning(f"Could not find {audio_in}. Please check the filename!")
            
    logger.info("=== Heatmap Generation Complete ===")