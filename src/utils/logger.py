import logging
import sys
import os

def get_logger(name="deepfake_detector", log_file="logs/project_run.log"):
    """Creates a logger that outputs to both the console and a file."""
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 1. Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 2. File Handler (NEW)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger