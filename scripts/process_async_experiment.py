"""Process async experiment results and create a tidy dataframe.

Usage:
    python scripts/process_async_experiment.py experiments/exp1.1.1_tyler_t=1.yaml

This script:
1. Loads the experiment config to find the experiment folder
2. Reads all result JSON files from provider_model subdirectories
3. Creates a tidy dataframe with columns: provider, model, scale, repeat, item, response
4. Saves as CSV in the experiment folder
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from legal_attitudes.config import BatchConfig
from legal_attitudes.utils import RESULTS_DIR, setup_logging


def parse_model_dir_name(dir_name: str) -> tuple[str, str]:
    """Parse provider_model directory name.

    Args:
        dir_name: e.g., "openrouter_deepseek_deepseek-v3.2"

    Returns:
        Tuple of (provider, model) where model has _ replaced back to /
    """
    parts = dir_name.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid model directory name: {dir_name}")

    provider = parts[0]
    # Model may have _ that were sanitized from / and :
    # We'll just keep it as-is since we don't know the original format
    model = parts[1]

    return provider, model


def create_results_dataframe(experiment_dir: Path) -> pd.DataFrame:
    """Create a tidy dataframe from all async experiment result files.

    Args:
        experiment_dir: Path to experiment folder (e.g., results/exp1.1.1_tyler_t=1)

    Returns:
        DataFrame with columns: provider, model, scale, repeat, item, response
    """
    rows = []

    # Find all provider_model subdirectories
    model_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not model_dirs:
        logger.warning(f"No model directories found in {experiment_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(model_dirs)} model directories")

    for model_dir in sorted(model_dirs):
        # Parse directory name to get provider and model
        try:
            provider, model = parse_model_dir_name(model_dir.name)
        except ValueError as e:
            logger.warning(f"Skipping directory: {e}")
            continue

        logger.info(f"Processing {model_dir.name} (provider={provider}, model={model})")

        # Find all result JSON files
        result_files = list(model_dir.glob("*_repeat_*.json"))
        logger.info(f"  Found {len(result_files)} result files")

        for result_file in sorted(result_files):
            # Parse filename: {scale_name}_repeat_{###}.json
            filename = result_file.stem
            parts = filename.rsplit("_repeat_", 1)
            if len(parts) != 2:
                logger.warning(f"Skipping file with unexpected name: {result_file}")
                continue

            scale_name = parts[0]
            repeat_num = int(parts[1])

            # Load result
            try:
                result = json.loads(result_file.read_text())

                # Parse JSON response
                json_field = result.get("json")
                if not json_field:
                    logger.warning(f"No json field in {result_file}")
                    continue

                # Handle case where json might already be a dict or a string
                if isinstance(json_field, dict):
                    parsed_json = json_field
                elif isinstance(json_field, str):
                    parsed_json = json.loads(json_field)
                else:
                    logger.warning(f"Unexpected json type in {result_file.name}: {type(json_field)}")
                    continue

                # Extract each question response
                for item_num, response_value in parsed_json.items():
                    rows.append({
                        "provider": provider,
                        "model": model,
                        "scale": scale_name,
                        "repeat": repeat_num,
                        "item": item_num,
                        "response": response_value,
                    })
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in {result_file.name} (likely placeholder/refusal) - skipping")
                continue
            except Exception as e:
                logger.error(f"Error processing {result_file.name}: {e}")
                continue

    df = pd.DataFrame(rows)

    if not df.empty:
        logger.info(f"Created dataframe with {len(df)} rows")
        logger.info(f"  Providers: {df['provider'].unique().tolist()}")
        logger.info(f"  Models: {df['model'].unique().tolist()}")
        logger.info(f"  Scales: {df['scale'].unique().tolist()}")
        logger.info(f"  Repeats: {df['repeat'].min()}-{df['repeat'].max()}")
    else:
        logger.warning("Created empty dataframe")

    return df


def main(config_path: Path):
    """Main processing logic."""

    # Load config
    raw = yaml.safe_load(config_path.read_text())
    cfg = BatchConfig(**raw)

    # Derive experiment directory from config
    experiment_dir = RESULTS_DIR / cfg.experiment_name

    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_dir}\n"
            f"Have you run this experiment yet?"
        )

    logger.info(f"Config: {config_path}")
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Experiment dir: {experiment_dir}")

    # Create dataframe
    logger.info("Creating results dataframe...")
    df = create_results_dataframe(experiment_dir)

    if df.empty:
        logger.warning("No results to save")
        return

    # Save CSV
    csv_path = experiment_dir / f"{cfg.experiment_name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results CSV to: {csv_path}")

    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Total responses: {len(df)}")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Process async experiment results and create CSV"
    )
    parser.add_argument(
        "config",
        help="Path to experiment config file (e.g., experiments/exp1.1.1_tyler_t=1.yaml)"
    )
    args = parser.parse_args()

    try:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        main(config_path)
    except Exception as exc:
        logger.error(f"Failed to process experiment: {exc}")
        raise
