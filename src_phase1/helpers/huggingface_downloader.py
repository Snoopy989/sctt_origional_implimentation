"""
Generic Hugging Face Model Downloader
======================================

Downloads models from Hugging Face Hub based on .env configuration.

Requirements:
pip install transformers torch huggingface_hub python-dotenv

.env Configuration:
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
HUGGINGFACE_TOKEN=your_token_here  (or HF_TOKEN)
"""

import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_config():
    """Load configuration from .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logging.warning("python-dotenv not installed, using environment variables only")

    config = {
        'model_name': os.getenv("MODEL_NAME"),
        'token': os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    }
    
    return config

def download_model(model_name=None, save_directory="./downloaded_models", auth_token=None):
    """
    Download a model from Hugging Face Hub.

    Args:
        model_name (str): HuggingFace model identifier (e.g., 'meta-llama/Llama-2-7b-chat-hf')
                         If None, reads from MODEL_NAME env variable
        save_directory (str): Base directory to save the model
        auth_token (str): Hugging Face authentication token (required for gated models)

    Returns:
        str: Path to downloaded model directory, or None if failed
    """
    # Load config if not provided
    if model_name is None or auth_token is None:
        config = load_config()
        model_name = model_name or config['model_name']
        auth_token = auth_token or config['token']
    
    if not model_name:
        raise ValueError("MODEL_NAME not specified. Set in .env file or pass as argument")
    
    if not auth_token:
        logging.warning("No authentication token provided. This may fail for gated models.")

    try:
        from huggingface_hub import snapshot_download

        # Create model-specific subdirectory name from model identifier
        model_dirname = model_name.replace('/', '--')
        save_path = Path(save_directory) / model_dirname
        save_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"Downloading {model_name} to: {save_path}")

        # Download the model
        downloaded_path = snapshot_download(
            repo_id=model_name,
            local_dir=str(save_path),
            local_dir_use_symlinks=False,
            token=auth_token
        )

        logging.info(f"Model downloaded successfully to: {downloaded_path}")
        return str(save_path)

    except Exception as e:
        logging.error(f"Error downloading model: {e}")
        return None

if __name__ == "__main__":
    config = load_config()

    if not config['model_name']:
        logging.error("MODEL_NAME not found in .env file")
        logging.info("Add to .env: MODEL_NAME=meta-llama/Llama-2-7b-chat-hf")
        exit(1)
    
    if not config['token']:
        logging.warning("No HuggingFace token found - may fail for gated models")
        logging.info("Get token from: https://huggingface.co/settings/tokens")
        logging.info("Add to .env: HUGGINGFACE_TOKEN=your_token_here")
    
    logging.info(f"Model to download: {config['model_name']}")
    download_model()