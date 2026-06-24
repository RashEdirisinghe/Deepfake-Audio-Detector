import os
from gtts import gTTS
from pydub import AudioSegment
from src.utils.logger import get_logger

logger = get_logger("sinhala_tts")

def generate_sinhala_audio(text_samples, output_dir="data/fake/sinhala"):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Using Google TTS for Sinhala synthetic generation...")

    generated_files = []

    for idx, text in enumerate(text_samples, start=1):
        temp_mp3 = os.path.join(output_dir, f"temp_{idx:04d}.mp3")
        file_path = os.path.join(output_dir, f"fake_sin_{idx:04d}.wav")

        try:
            tts = gTTS(text=text, lang="si")
            tts.save(temp_mp3)

            audio = AudioSegment.from_mp3(temp_mp3)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(file_path, format="wav")

            generated_files.append(file_path)
            logger.info(f"Generated Sinhala synthetic audio: {file_path}")

        except Exception as e:
            logger.error(f"Failed to generate Sinhala audio for sample {idx}: {e}")

        finally:
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)

    return generated_files


if __name__ == "__main__":
    sample_texts = [
        "ආයුබෝවන්, මෙය පරීක්ෂණ සටහනකි.",
        "කෘතිම බුද්ධිය මගින් ශ්‍රව්‍ය උත්පාදනය කිරීම."
    ]
    generate_sinhala_audio(sample_texts)