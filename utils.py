# src/utils/logging_utils.py
import logging
import sys
import os
from PIL import Image
import yaml
import logging

def setup_logger(level="INFO"):
    """Sets up the root logger."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
            # Optionally add FileHandler here if needed
        ]
    )
    # Suppress excessive logging from underlying libraries if needed
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

def get_logger(name):
    """Gets a logger instance."""
    return logging.getLogger(name)

logger = logging.getLogger(__name__)

def load_image(path: str) -> Image.Image | None:
    """
    Load an image from a local path. Handles potential errors.
    """
    try:
        if not os.path.exists(path):
            logger.error(f"Image path does not exist: {path}")
            return None
        img = Image.open(path).convert("RGB")
        logger.debug(f"Loaded image from: {path}")
        return img
    except Exception as e:
        logger.error(f"Error loading image {path}: {e}")
        return None

def load_config(config_path: str) -> dict:
    """Loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        raise # Re-raise the exception as config is critical

# Add other general utilities here as needed