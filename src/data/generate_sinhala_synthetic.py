import os
import torch
import scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer
from src.utils.logger import get_logger

logger = get_logger("sinhala_tts")


def generate_sinhala_audio(text_samples, output_dir="data/fake/sinhala"):
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading Meta MMS Sinhala TTS model...")
    model_name = "facebook/mms-tts-sin"
    model = VitsModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sample_rate = model.config.sampling_rate

    for idx, text in enumerate(text_samples):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform

        file_path = os.path.join(output_dir, f"fake_sin_{idx + 1:04d}.wav")
        scipy.io.wavfile.write(file_path, rate=sample_rate, data=output[0].cpu().numpy())
        logger.info(f"Generated synthetic Sinhala audio: {file_path}")


if __name__ == "__main__":
    # Test sentences in Sinhala
    sample_texts = [
        "ආයුබෝවන්, මෙය පරීක්ෂණ සටහනකි.",
        "කෘතිම බුද්ධිය මගින් ශ්‍රව්‍ය උත්පාදනය කිරීම."
    ]
    generate_sinhala_audio(sample_texts)