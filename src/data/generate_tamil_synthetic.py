import os
import torch
import scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer
from src.utils.logger import get_logger

logger = get_logger("tamil_tts")


def generate_tamil_audio(text_samples, output_dir="data/fake/tamil"):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Loading Meta MMS Tamil TTS model...")
    model_name = "facebook/mms-tts-tam"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VitsModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    sample_rate = model.config.sampling_rate

    generated_files = []
    for idx, text in enumerate(text_samples, start=1):
        try:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(**inputs).waveform

            file_path = os.path.join(output_dir, f"fake_tam_{idx:04d}.wav")
            scipy.io.wavfile.write(file_path, rate=sample_rate, data=output[0].cpu().numpy())
            generated_files.append(file_path)
            logger.info(f"Generated synthetic Tamil audio: {file_path}")
        except Exception as e:
            logger.error(f"Failed to generate Tamil audio for sample {idx}: {e}")

    return generated_files


if __name__ == "__main__":
    sample_texts = [
        "வணக்கம், இது ஒரு சோதனை.",
        "செயற்கை நுண்ணறிவு தொழில்நுட்பம் வேகமாக வளர்ந்து வருகிறது."
    ]
    generate_tamil_audio(sample_texts)