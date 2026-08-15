import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before pyplot import to prevent GUI crashes in Flask!
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.models.resnet_multitask import AudioDeepfakeResNet
from src.utils.logger import get_logger
from src.utils.audio_utils import load_and_preprocess_audio

logger = get_logger("grad_cam_explainer")

# --- Convert 1-logit output into 2-class logits [Fake, Real] ---
class BinaryHeadWrapper(torch.nn.Module):
    """Wraps the single logit into [Fake_Score, Real_Score] so Grad-CAM
    can properly target Class 0 (Fake) or Class 1 (Real).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        rf_logits, _ = self.model(x)
        # Index 0 = Fake (-rf_logits), Index 1 = Real (+rf_logits)
        return torch.cat([-rf_logits, rf_logits], dim=1)

def generate_heatmap(model_or_path, audio_path, output_path, target_class=0):
    """Generates a Grad-CAM heatmap for a given audio file.
    target_class: 0 for Fake, 1 for Real.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- THE FIX 1: Prevent double-loading in Flask ---
    # If a string is passed, load the model. If a model is passed, just use it!
    if isinstance(model_or_path, str):
        base_model = AudioDeepfakeResNet(num_compression_classes=3).to(device)
        if os.path.exists(model_or_path):
            base_model.load_state_dict(torch.load(model_or_path, map_location=device))
            base_model.eval()
        else:
            logger.error(f"Model weights not found at {model_or_path}")
            return
    else:
        base_model = model_or_path
        
    wrapped_model = BinaryHeadWrapper(base_model)

    try:
        # --- THE FIX 2: Use the DRY utility function ---
        log_mel_spec = load_and_preprocess_audio(audio_path, target_sr=16000, max_frames=128)
        if log_mel_spec is None:
            return
            
        input_tensor = log_mel_spec.unsqueeze(0).to(device) # Add batch dimension [1, 1, 128, 128]
        
    except Exception as e:
        logger.error(f"Failed to process audio {audio_path}: {e}")
        return

    target_layers = [base_model.backbone.layer4[-1]]
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
    
    # Target Class 0 (Fake) or Class 1 (Real)
    targets = [ClassifierOutputTarget(target_class)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    spec_image = input_tensor.squeeze().cpu().numpy()
    spec_image = (spec_image - spec_image.min()) / (spec_image.max() - spec_image.min() + 1e-7)
    spec_image_rgb = np.stack([spec_image]*3, axis=-1)
    
    visualization = show_cam_on_image(spec_image_rgb, grayscale_cam, use_rgb=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # --- THE FIX 3: Thread-safe Object-Oriented Matplotlib ---
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot origin='lower' so low frequencies are at the bottom
    ax.imshow(visualization, origin='lower')
    ax.set_title(f"Grad-CAM Heatmap (Target Class: {'Real' if target_class == 1 else 'Fake'})")
    ax.axis('off')
    
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig) # Safely close the specific figure instance to free memory
    
    logger.info(f"Heatmap successfully saved to {output_path}")

if __name__ == "__main__":
    logger.info("=== Generating Grad-CAM Heatmaps ===")
    model_weights = "weights/best_deepfake_resnet.pth"
    
    test_files = [
        ("data/real/tamil/real_00000.wav", "plots/heatmap_real_tamil.png", 1),
        ("data/fake/tamil/fake_tam_0001.wav", "plots/heatmap_fake_tamil.png", 0),
        ("data/compressed/fake_tamil_heavy/fake_tam_0001.wav", "plots/heatmap_fake_tamil_heavy.png", 0)
    ]
    
    for audio_in, image_out, target_cls in test_files:
        if os.path.exists(audio_in):
            logger.info(f"Processing heatmap for {audio_in} (Target: {target_cls})...")
            generate_heatmap(model_weights, audio_in, image_out, target_class=target_cls)
        else:
            logger.warning(f"Could not find {audio_in}. Please check the filename!")
            
    logger.info("=== Heatmap Generation Complete ===")