"""
Download TalkVerse checkpoint from Hugging Face Hub.

This script downloads the S2V-5B model checkpoint required for inference.
The downloaded checkpoint will be used as the lora_ckpt parameter in generate.py.

Usage:
    # Download to default location (./ckpts/talkverse/)
    python download_checkpoint.py

    # Download to custom location
    python download_checkpoint.py --output_dir /path/to/custom/dir

    # Force re-download even if checkpoint exists
    python download_checkpoint.py --force

As a module:
    from download_checkpoint import download_talkverse_checkpoint
    checkpoint_path = download_talkverse_checkpoint()
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

# Hugging Face repository configuration
# TODO: Update these values before open-sourcing
HF_REPO_ID = "Snap-Research/TalkVerse-S2V-5B"  # Replace with your actual repo ID
HF_CHECKPOINT_FILENAME = "talkverse_s2v_5b.safetensors"
HF_CONFIG_FILENAME = "config.json"

# Default local directory for checkpoints
DEFAULT_CHECKPOINT_DIR = "ckpts/TalkVerse-S2V-5B"


def download_talkverse_checkpoint(
    output_dir: str = None,
    repo_id: str = None,
    filename: str = None,
    force: bool = False,
    token: str = None,
    download_config: bool = True,
) -> str:
    """
    Download the TalkVerse checkpoint from Hugging Face Hub.

    Args:
        output_dir: Directory to save the checkpoint. Defaults to ./ckpts/talkverse/
        repo_id: Hugging Face repository ID. Defaults to HF_REPO_ID.
        filename: Checkpoint filename in the repository. Defaults to HF_CHECKPOINT_FILENAME.
        force: If True, re-download even if the file exists locally.
        token: Hugging Face token for private repositories (optional).
        download_config: If True, also download config.json.

    Returns:
        str: Path to the downloaded checkpoint file.

    Raises:
        ImportError: If huggingface_hub is not installed.
        Exception: If download fails.
    """
    try:
        from huggingface_hub import hf_hub_download, HfFolder
    except ImportError:
        raise ImportError(
            "huggingface_hub is required but not installed. "
            "Please install it with: pip install huggingface_hub"
        )

    # Use defaults if not specified
    repo_id = repo_id or HF_REPO_ID
    filename = filename or HF_CHECKPOINT_FILENAME
    output_dir = output_dir or DEFAULT_CHECKPOINT_DIR

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if checkpoint already exists
    local_file = output_path / filename
    if local_file.exists() and not force:
        logging.info(f"Checkpoint already exists at {local_file}")
        logging.info("Use --force to re-download.")
        return str(local_file)

    logging.info(f"Downloading checkpoint from Hugging Face Hub...")
    logging.info(f"  Repository: {repo_id}")
    logging.info(f"  Filename: {filename}")
    logging.info(f"  Output directory: {output_path.absolute()}")

    try:
        # Download the checkpoint
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(output_path),
            local_dir_use_symlinks=False,
            token=token,
            force_download=force,
        )

        logging.info(f"✅ Successfully downloaded checkpoint to: {downloaded_path}")

        # Also download config.json if requested
        if download_config:
            try:
                config_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=HF_CONFIG_FILENAME,
                    local_dir=str(output_path),
                    local_dir_use_symlinks=False,
                    token=token,
                    force_download=force,
                )
                logging.info(f"✅ Downloaded config to: {config_path}")
            except Exception as e:
                logging.warning(f"Could not download config.json: {e}")

        return downloaded_path

    except Exception as e:
        logging.error(f"❌ Failed to download checkpoint: {e}")
        raise


def get_checkpoint_path(output_dir: str = None) -> str:
    """
    Get the path to the TalkVerse checkpoint, downloading if necessary.

    This is a convenience function that downloads the checkpoint if it doesn't
    exist locally and returns the path.

    Args:
        output_dir: Directory where the checkpoint is stored.

    Returns:
        str: Path to the checkpoint file.
    """
    output_dir = output_dir or DEFAULT_CHECKPOINT_DIR
    local_file = Path(output_dir) / HF_CHECKPOINT_FILENAME

    if not local_file.exists():
        logging.info("Checkpoint not found locally. Downloading...")
        return download_talkverse_checkpoint(output_dir=output_dir)

    return str(local_file)


def load_config(output_dir: str = None) -> dict:
    """
    Load the model configuration from config.json.

    Args:
        output_dir: Directory where the config is stored.

    Returns:
        dict: Model configuration dictionary.
    """
    import json
    
    output_dir = output_dir or DEFAULT_CHECKPOINT_DIR
    config_file = Path(output_dir) / HF_CONFIG_FILENAME

    if not config_file.exists():
        logging.warning(f"Config file not found at {config_file}")
        return {}

    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Download TalkVerse checkpoint from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default location
  python download_checkpoint.py

  # Download to custom directory
  python download_checkpoint.py --output_dir /path/to/checkpoints

  # Force re-download
  python download_checkpoint.py --force

  # Use with generate.py
  python download_checkpoint.py
  python generate.py --task s2v-5B --lora_ckpt ckpts/talkverse/talkverse_s2v_5b.safetensors ...
        """
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Directory to save the checkpoint (default: {DEFAULT_CHECKPOINT_DIR})"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=HF_REPO_ID,
        help=f"Hugging Face repository ID (default: {HF_REPO_ID})"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=HF_CHECKPOINT_FILENAME,
        help=f"Checkpoint filename in the repository (default: {HF_CHECKPOINT_FILENAME})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if checkpoint exists locally"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token for private repositories (optional). "
             "Can also be set via HF_TOKEN environment variable."
    )

    args = parser.parse_args()

    # Get token from environment if not provided
    token = args.token or os.environ.get("HF_TOKEN")

    try:
        checkpoint_path = download_talkverse_checkpoint(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            filename=args.filename,
            force=args.force,
            token=token,
        )

        print(f"\nCheckpoint path: {checkpoint_path}")
        print(f"\nTo use with generate.py, run:")
        print(f"  python generate.py --task s2v-5B --lora_ckpt {checkpoint_path} ...")

    except Exception as e:
        logging.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

