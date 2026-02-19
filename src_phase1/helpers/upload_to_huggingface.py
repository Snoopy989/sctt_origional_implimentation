import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_REPO_ID = "PhillipGre/Llama-2-7b_sctt_regression"
DEFAULT_OLDGOOD_DIR = "Llama-2-7b_sctt_regression_oldgood/sctt_results_LORA_10_epochs_Llama-2-7b-chat-hf"
DEFAULT_BEST_CHECKPOINT_DIR = "sctt_results_LORA_1_epochs_Llama-2-7b-chat-hf/checkpoint-748"

REQUIRED_INFERENCE_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
]

OPTIONAL_INFERENCE_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "README.md",
]

OPTIONAL_TRAINING_FILES = [
    "training_args.bin",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
]


def load_token() -> str:
    load_dotenv()
    token = os.getenv("HF_TOKEN_WRITE")
    if not token:
        raise ValueError(
            "No Hugging Face token found. Set HF_TOKEN or HUGGINGFACE_TOKEN in your environment/.env"
        )
    return token


def copy_with_fallback(filename: str, preferred_dir: Path, fallback_dir: Path, out_dir: Path, required: bool) -> str:
    preferred_path = preferred_dir / filename
    fallback_path = fallback_dir / filename
    destination = out_dir / filename

    if preferred_path.exists():
        shutil.copy2(preferred_path, destination)
        return "preferred"
    if fallback_path.exists():
        shutil.copy2(fallback_path, destination)
        return "fallback"

    if required:
        raise FileNotFoundError(
            f"Required file missing in both locations: {filename}\n"
            f" - preferred: {preferred_path}\n"
            f" - fallback:  {fallback_path}"
        )
    return "missing"


def build_upload_bundle(preferred_dir: Path, fallback_dir: Path, include_training_artifacts: bool) -> Path:
    if not preferred_dir.exists():
        raise FileNotFoundError(f"Preferred checkpoint directory not found: {preferred_dir}")
    if not fallback_dir.exists():
        raise FileNotFoundError(f"Fallback model directory not found: {fallback_dir}")

    bundle_dir = Path(tempfile.mkdtemp(prefix="hf_upload_bundle_"))
    manifest = {
        "preferred_dir": str(preferred_dir.resolve()),
        "fallback_dir": str(fallback_dir.resolve()),
        "files": {},
    }

    for filename in REQUIRED_INFERENCE_FILES:
        source_used = copy_with_fallback(
            filename=filename,
            preferred_dir=preferred_dir,
            fallback_dir=fallback_dir,
            out_dir=bundle_dir,
            required=True,
        )
        manifest["files"][filename] = source_used

    for filename in OPTIONAL_INFERENCE_FILES:
        source_used = copy_with_fallback(
            filename=filename,
            preferred_dir=preferred_dir,
            fallback_dir=fallback_dir,
            out_dir=bundle_dir,
            required=False,
        )
        manifest["files"][filename] = source_used

    if include_training_artifacts:
        training_out_dir = bundle_dir / "training_artifacts" / preferred_dir.name
        training_out_dir.mkdir(parents=True, exist_ok=True)
        for filename in OPTIONAL_TRAINING_FILES:
            candidate = preferred_dir / filename
            if candidate.exists():
                shutil.copy2(candidate, training_out_dir / filename)
                manifest["files"][f"training_artifacts/{preferred_dir.name}/{filename}"] = "preferred"

    manifest_path = bundle_dir / "upload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return bundle_dir


def upload_bundle(bundle_dir: Path, repo_id: str, token: str, commit_message: str, private: bool) -> None:
    api = HfApi(token=token)

    create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(bundle_dir),
        repo_id=repo_id,
        commit_message=commit_message,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the 7B SCTT LoRA model to Hugging Face with preferred+fallback file sourcing."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Target Hugging Face model repo ID")
    parser.add_argument(
        "--preferred-dir",
        default=DEFAULT_BEST_CHECKPOINT_DIR,
        help="Preferred source directory (best checkpoint)",
    )
    parser.add_argument(
        "--fallback-dir",
        default=DEFAULT_OLDGOOD_DIR,
        help="Fallback source directory (old known-good model)",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Commit message for Hugging Face upload (if omitted, you will be prompted)",
    )
    parser.add_argument("--private", action="store_true", help="Create/use a private Hugging Face repo")
    parser.add_argument(
        "--include-training-artifacts",
        action="store_true",
        help="Also upload trainer artifacts (optimizer/scheduler/state) under training_artifacts/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and print bundle contents without uploading",
    )
    return parser.parse_args()


def resolve_commit_message(cli_message: str | None) -> str:
    if cli_message and cli_message.strip():
        return cli_message.strip()

    message = input("Enter Hugging Face commit message: ").strip()
    if not message:
        raise ValueError("Commit message is required. Pass --commit-message or enter one when prompted.")
    return message


def main() -> None:
    args = parse_args()

    preferred_dir = Path(args.preferred_dir)
    fallback_dir = Path(args.fallback_dir)

    bundle_dir = build_upload_bundle(
        preferred_dir=preferred_dir,
        fallback_dir=fallback_dir,
        include_training_artifacts=args.include_training_artifacts,
    )

    file_count = sum(1 for p in bundle_dir.rglob("*") if p.is_file())
    logger.info("Prepared upload bundle at %s (%d files)", bundle_dir, file_count)
    logger.info("Target repo: %s", args.repo_id)

    if args.dry_run:
        logger.info("Dry-run enabled. No upload performed.")
        return

    commit_message = resolve_commit_message(args.commit_message)
    token = load_token()
    upload_bundle(
        bundle_dir=bundle_dir,
        repo_id=args.repo_id,
        token=token,
        commit_message=commit_message,
        private=args.private,
    )

    logger.info("Upload complete.")
    logger.info("Model URL: https://huggingface.co/%s", args.repo_id)
    logger.info(
        "Pull later with snapshot_download(repo_id='%s', local_dir='downloaded_models/%s')",
        args.repo_id,
        args.repo_id.split('/')[-1],
    )


if __name__ == "__main__":
    main()
