from src.data.generate_tamil_fakes import generate_tamil_audio
from src.data.generate_sinhala_fakes import generate_sinhala_audio
from src.utils.logger import get_logger

logger = get_logger("batch_synthesizer")


def run_synthetic_pipeline():
    logger.info("=== Starting Synthetic Audio Generation Pipeline ===")

    tamil_prompts = [
        "வணக்கம், இது ஒரு சோதனை.",
        "செயற்கை நுண்ணறிவு தொழில்நுட்பம் வேகமாக வளர்ந்து வருகிறது."
    ]
    sinhala_prompts = [
        "ආයුබෝවන්, මෙය පරීක්ෂණ සටහනකි.",
        "කෘතිම බුද්ධිය මගින් ශ්‍රව්‍ය උත්පාදනය කිරීම."
    ]

    generate_tamil_audio(tamil_prompts)
    generate_sinhala_audio(sinhala_prompts)

    logger.info("=== Synthetic Generation Complete ===")


if __name__ == "__main__":
    run_synthetic_pipeline()