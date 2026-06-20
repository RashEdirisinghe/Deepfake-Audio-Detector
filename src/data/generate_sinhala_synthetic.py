import os
from gtts import gTTS
from pydub import AudioSegment
from src.utils.logger import get_logger

logger = get_logger("sinhala_tts")


def generate_sinhala_audio(text_samples, output_dir="data/fake/sinhala"):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Using Google TTS for Sinhala synthetic generation...")

    for idx, text in enumerate(text_samples):
        temp_mp3 = os.path.join(output_dir, f"temp_{idx}.mp3")
        file_path = os.path.join(output_dir, f"fake_sin_{idx + 1:04d}.wav")

        # 1. Generate audio in Sinhala natively
        tts = gTTS(text=text, lang='si')
        tts.save(temp_mp3)

        # 2. Convert the MP3 to a 16kHz mono WAV file
        audio = AudioSegment.from_mp3(temp_mp3)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(file_path, format="wav")

        # 3. Clean up the temp file
        os.remove(temp_mp3)

        logger.info(f"Generated true Sinhala synthetic audio: {file_path}")


if __name__ == "__main__":
    sample_texts = [
        "ආයුබෝවන්, මෙය පරීක්ෂණ සටහනකි.",
        "කෘතිම බුද්ධිය මගින් ශ්‍රව්‍ය උත්පාදනය කිරීම."
    ]
    generate_sinhala_audio(sample_texts)