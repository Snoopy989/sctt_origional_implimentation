"""
PEFT Adapter Downloader
========================
Downloads fine-tuned LoRA adapters from Hugging Face Hub into
the local adapters/ directory.

Usage (from project root, with venv active):
    python src_phase1/helpers/download_adapters.py

HF token is read from .env (HF_TOKEN or HUGGINGFACE_TOKEN).
"""

import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add custom SUCCESS level between INFO and WARNING
SUCCESS = 25
logging.addLevelName(SUCCESS, 'SUCCESS')

def log_success(msg, *args, **kwargs):
    logging.log(SUCCESS, msg, *args, **kwargs)

# ── Adapters to download ──────────────────────────────────────────────────────
ADAPTERS = [
    "PhillipGre/llama2-13b-sctt-regression",
    "PhillipGre/Llama-2-7b_sctt_regression",
]

# Local folder that will hold all adapters (mirrors downloaded_models/ convention)
ADAPTERS_DIR = Path(__file__).resolve().parents[2] / "downloaded_adapters"


def get_token():
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except ImportError:
        pass
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def download_adapter(repo_id: str, token: str | None) -> str | None:
    from huggingface_hub import snapshot_download

    local_dir = ADAPTERS_DIR / repo_id.replace("/", "--")
    local_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Downloading adapter: {repo_id}")
    logging.info(f"Saving to: {local_dir}")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=token,
            )
            log_success(f"Download complete: {local_dir}")
            return str(local_dir)
        except KeyboardInterrupt:
            logging.warning("Download interrupted. Re-run to resume (download is resumable).")
            return None
        except Exception as e:
            if attempt < max_retries:
                logging.warning(f"Attempt {attempt}/{max_retries} failed: {e}. Retrying...")
            else:
                logging.error(f"Failed after {max_retries} attempts: {e}")
                return None


if __name__ == "__main__":
    token = get_token()
    if not token:
        logging.warning("No HF token found in .env. Private or gated repos will fail.")

    results = {}
    for repo_id in ADAPTERS:
        path = download_adapter(repo_id, token)
        results[repo_id] = path

    logging.info("ADAPTER DOWNLOAD SUMMARY")
    for repo_id, path in results.items():
        if path:
            log_success(f"{repo_id} -> {path}")
        else:
            logging.error(f"{repo_id} -> FAILED")
    logging.info(f"Adapters saved under: {ADAPTERS_DIR}")
    logging.info("Update ADAPTER_PATH in .env to use a local path, e.g.:")
    for repo_id in ADAPTERS:
        logging.info(f"  ADAPTER_PATH=./downloaded_adapters/{repo_id.replace('/', '--')}")
