"""Submit batch experiments to provider batch APIs for 50% discount.

Usage:
    python scripts/submit_batch.py experiments/exp_batch.yaml

Requirements:
    - Experiment config must have exactly ONE provider and ONE model
    - All scales × repeats will be submitted in a single batch
    - Batch ID and metadata saved to experiment folder for later processing
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from loguru import logger

from legal_attitudes.batch import (
    create_anthropic_batch,
    create_google_batch,
    create_openai_batch,
)
from legal_attitudes.config import BatchConfig
from legal_attitudes.utils import RESULTS_DIR, ROOT, setup_logging


def save_batch_metadata(cfg: BatchConfig, batch_info: dict, config_path: Path):
    """Save batch metadata to experiment folder."""
    experiment_dir = RESULTS_DIR / cfg.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg.models[0]

    metadata = {
        "batch_id": batch_info["batch_id"],
        "provider": model_cfg.provider,
        "model": model_cfg.name,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": batch_info["status"],
        "num_requests": batch_info["num_requests"],
        "num_scales": len(cfg.prompts),
        "num_repeats": cfg.repeats,
        "config_path": str(config_path),
        "experiment_name": cfg.experiment_name,
    }

    # Add provider-specific fields
    if "input_file_id" in batch_info:
        metadata["input_file_id"] = batch_info["input_file_id"]
    if "file_name" in batch_info:
        metadata["file_name"] = batch_info["file_name"]

    metadata_path = experiment_dir / "batch_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info(f"Saved batch metadata to {metadata_path}")
    return metadata_path


def main(config_path: str):
    """Load config, create batch, submit, and save metadata."""
    cfg_path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = BatchConfig(**raw)

    # Validate single provider/model
    if len(cfg.models) != 1:
        raise ValueError(
            f"Batch experiments require exactly ONE model. "
            f"Found {len(cfg.models)} models in config."
        )

    model_cfg = cfg.models[0]
    provider = model_cfg.provider

    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Provider: {provider}")
    logger.info(f"Model: {model_cfg.name}")
    logger.info(f"Scales: {len(cfg.prompts)}")
    logger.info(f"Repeats: {cfg.repeats}")
    logger.info(f"Total requests: {len(cfg.prompts) * cfg.repeats}")

    # Load all prompts
    prompt_texts = {}
    for prompt_cfg in cfg.prompts:
        prompt_path = ROOT / prompt_cfg.path
        prompt_texts[prompt_path.stem] = prompt_path.read_text()

    # Setup batch input path
    experiment_dir = RESULTS_DIR / cfg.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    batch_input_path = experiment_dir / "batch_input.jsonl"

    # Create and submit batch based on provider
    if provider == "openai":
        batch_info = create_openai_batch(cfg, prompt_texts, batch_input_path)
    elif provider == "anthropic":
        batch_info = create_anthropic_batch(cfg, prompt_texts, batch_input_path)
    elif provider == "google":
        batch_info = create_google_batch(cfg, prompt_texts, batch_input_path)
    else:
        raise ValueError(f"Unsupported provider for batch: {provider}")

    # Save metadata
    metadata_path = save_batch_metadata(cfg, batch_info, cfg_path)

    logger.info("Batch submitted successfully!")
    logger.info(f"Batch ID: {batch_info['batch_id']}")
    logger.info(f"Status: {batch_info['status']}")
    logger.info(f"Metadata: {metadata_path}")
    logger.info("Next steps:")
    logger.info("1. Wait for batch to complete (check provider dashboard)")
    logger.info(f"2. Run: python scripts/process_batch_results.py {cfg_path}")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Submit batch experiment to provider batch API"
    )
    parser.add_argument("config", help="Path to experiment YAML config")
    args = parser.parse_args()

    try:
        main(args.config)
    except Exception as exc:
        logger.error(f"Failed to submit batch: {exc}")
        raise
