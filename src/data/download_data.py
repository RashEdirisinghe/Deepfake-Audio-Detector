import os
import yaml
from src.utils.logger import get_logger

logger = get_logger("data_downloader")

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_dataset_stub(language_code: str):
    logger.info(f"Initializing download script for language code: '{language_code}'...")
    # Dataset download and extraction logic will go here

if __name__ == "__main__":
    config = load_config()
    for lang in config["project"]["languages"]:
        download_dataset_stub(lang)