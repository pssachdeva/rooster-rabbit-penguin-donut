"""Process completed batch results from provider batch APIs.

Usage:
    python scripts/process_batch_results.py experiments/exp1.2.0_batch_openai.yaml

    # Check status without downloading
    python scripts/process_batch_results.py experiments/exp1.2.0_batch_openai.yaml --status-only

    # Force re-download even if results exist
    python scripts/process_batch_results.py experiments/exp1.2.0_batch_openai.yaml --force

This script:
1. Loads the experiment config to find the experiment folder
2. Reads batch_metadata.json from the experiment folder
3. Checks batch status with the provider
4. Downloads results if batch is complete
5. Parses and saves individual result files
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from anthropic import Anthropic
from google import genai
from loguru import logger
from openai import OpenAI

from legal_attitudes.api import extract_json, make_refusal_response
from legal_attitudes.config import BatchConfig
from legal_attitudes.schemas import get_schema
from legal_attitudes.utils import RESULTS_DIR, setup_logging


def load_batch_metadata(experiment_dir: Path) -> dict:
    """Load batch metadata from experiment folder."""
    metadata_path = experiment_dir / "batch_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Batch metadata not found: {metadata_path}\n"
            f"Have you submitted this batch yet?"
        )
    return json.loads(metadata_path.read_text())


def check_openai_batch_status(batch_id: str) -> dict:
    """Check OpenAI batch status and return batch object."""
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    return {
        "status": batch.status,
        "request_counts": {
            "total": batch.request_counts.total,
            "completed": batch.request_counts.completed,
            "failed": batch.request_counts.failed,
        },
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
    }


def download_openai_results(batch_id: str, output_dir: Path) -> Path:
    """Download OpenAI batch results to a file."""
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        raise RuntimeError(f"Batch not completed yet. Status: {batch.status}")

    if not batch.output_file_id:
        raise RuntimeError("Batch completed but no output file available")

    # Download output file
    file_content = client.files.content(batch.output_file_id)
    output_path = output_dir / "batch_output.jsonl"
    output_path.write_bytes(file_content.content)

    logger.info(f"Downloaded batch output to {output_path}")

    # Download error file if exists
    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id)
        error_path = output_dir / "batch_errors.jsonl"
        error_path.write_bytes(error_content.content)
        logger.warning(f"Downloaded batch errors to {error_path}")

    return output_path


def check_anthropic_batch_status(batch_id: str) -> dict:
    """Check Anthropic batch status and return batch object."""
    client = Anthropic()
    batch = client.messages.batches.retrieve(batch_id)

    return {
        "status": batch.processing_status,
        "request_counts": {
            "total": batch.request_counts.processing + batch.request_counts.succeeded + batch.request_counts.errored + batch.request_counts.canceled + batch.request_counts.expired,
            "completed": batch.request_counts.succeeded,
            "failed": batch.request_counts.errored,
        },
        "results_url": batch.results_url if hasattr(batch, 'results_url') else None,
    }


def download_anthropic_results(batch_id: str, output_dir: Path) -> Path:
    """Download Anthropic batch results to a file."""
    client = Anthropic()
    batch = client.messages.batches.retrieve(batch_id)

    if batch.processing_status != "ended":
        raise RuntimeError(f"Batch not completed yet. Status: {batch.processing_status}")

    # Get all results
    results = []
    for result in client.messages.batches.results(batch_id):
        results.append(result.model_dump())

    # Save to JSONL
    output_path = output_dir / "batch_output.jsonl"
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info(f"Downloaded {len(results)} batch results to {output_path}")
    return output_path


def check_google_batch_status(batch_id: str) -> dict:
    """Check Google/Gemini batch status and return batch object."""
    import os

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable 'GEMINI_API_KEY' must be set")

    client = genai.Client(api_key=api_key)
    batch = client.batches.get(name=batch_id)

    # Gemini batch state can be: STATE_UNSPECIFIED, PROCESSING, COMPLETED, FAILED, CANCELLED
    return {
        "status": batch.state.name if hasattr(batch, 'state') else "UNKNOWN",
        "request_counts": {
            "total": getattr(batch, 'total_count', 0),
            "completed": getattr(batch, 'completed_count', 0),
            "failed": getattr(batch, 'failed_count', 0),
        },
        "output_uri": getattr(batch, 'output_uri', None),
    }


def download_google_results(batch_id: str, output_dir: Path) -> Path:
    """Download Google/Gemini batch results to a file."""
    import os

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable 'GEMINI_API_KEY' must be set")

    client = genai.Client(api_key=api_key)
    batch = client.batches.get(name=batch_id)

    if not hasattr(batch, 'state') or batch.state.name != "COMPLETED":
        status = batch.state.name if hasattr(batch, 'state') else "UNKNOWN"
        raise RuntimeError(f"Batch not completed yet. Status: {status}")

    # Download the output file
    if not hasattr(batch, 'output_uri') or not batch.output_uri:
        raise RuntimeError("Batch completed but no output URI available")

    # The output_uri is a GCS URI, need to download the file
    output_file = client.files.get(name=batch.output_uri)

    # Download file content
    output_path = output_dir / "batch_output.jsonl"
    # Note: This may need adjustment based on actual Gemini API
    # You might need to use the files.download() method or similar
    with open(output_path, "wb") as f:
        # This is a placeholder - actual implementation may vary
        f.write(output_file.read())

    logger.info(f"Downloaded batch output to {output_path}")
    return output_path


def parse_and_save_results(
    output_path: Path,
    experiment_dir: Path,
    provider: str,
    model: str,
    cfg: BatchConfig,
    force: bool = False
):
    """Parse batch output JSONL and save individual result files."""
    # Build map of scale name -> schema
    schema_map = {p.path.stem: p.schema_name for p in cfg.prompts}
    prompt_path_map = {p.path.stem: str(p.path) for p in cfg.prompts}

    # Create output directory
    safe_model = model.replace("/", "_").replace(":", "_")
    model_dir = experiment_dir / f"{provider}_{safe_model}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Parse results
    results_parsed = 0
    results_skipped = 0
    results_failed = 0

    with open(output_path) as f:
        for line in f:
            result = json.loads(line)

            # Extract custom_id to determine scale and repeat
            if provider == "openai":
                custom_id = result.get("custom_id")
                response = result.get("response", {})
                if response.get("status_code") != 200:
                    logger.error(f"Failed request: {custom_id} - {response.get('body', {}).get('error')}")
                    results_failed += 1
                    continue
                raw_text = response["body"]["choices"][0]["message"]["content"]

            elif provider == "anthropic":
                custom_id = result.get("custom_id")
                if result.get("result", {}).get("type") == "error":
                    logger.error(f"Failed request: {custom_id} - {result['result']['error']}")
                    results_failed += 1
                    continue
                raw_text = result["result"]["message"]["content"][0]["text"]

            elif provider == "google":
                custom_id = result.get("key")
                response = result.get("response")
                if not response or "error" in response:
                    logger.error(f"Failed request: {custom_id} - {response.get('error')}")
                    results_failed += 1
                    continue
                # Extract text from Gemini response
                raw_text = response["candidates"][0]["content"]["parts"][0]["text"]

            else:
                raise ValueError(f"Unknown provider: {provider}")

            # Parse custom_id: "{scale_name}_repeat_{###}"
            parts = custom_id.rsplit("_repeat_", 1)
            if len(parts) != 2:
                logger.error(f"Invalid custom_id format: {custom_id}")
                results_failed += 1
                continue

            scale_name = parts[0]
            repeat_num = int(parts[1])

            # Check if already exists
            output_file = model_dir / f"{scale_name}_repeat_{repeat_num:03d}.json"
            if output_file.exists() and not force:
                results_skipped += 1
                continue

            # Get schema
            schema_name = schema_map.get(scale_name)
            if not schema_name:
                logger.error(f"Unknown scale: {scale_name}")
                results_failed += 1
                continue

            schema_cls = get_schema(schema_name)

            # Extract JSON
            extracted = extract_json(raw_text)
            if extracted is None:
                logger.warning(f"Failed to extract JSON from {custom_id}, marking as refusal")
                json_out = make_refusal_response(schema_cls)
            else:
                json_out = extracted

            # Build result object
            result_obj = {
                "run_index": repeat_num - 1,
                "prompt": prompt_path_map.get(scale_name, f"prompts/{scale_name}.txt"),
                "provider": provider,
                "model": model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw": raw_text,
                "json": json_out,
            }

            # Save
            output_file.write_text(json.dumps(result_obj, indent=2))
            results_parsed += 1

    logger.info(f"Results: {results_parsed} saved, {results_skipped} skipped, {results_failed} failed")


def create_results_dataframe(experiment_dir: Path, provider: str, model: str) -> pd.DataFrame:
    """Create a tidy dataframe from all result files.

    Returns:
        DataFrame with columns: provider, model, scale, repeat, item, response
    """
    safe_model = model.replace("/", "_").replace(":", "_")
    model_dir = experiment_dir / f"{provider}_{safe_model}"

    if not model_dir.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return pd.DataFrame()

    rows = []

    # Iterate through all result JSON files
    for result_file in sorted(model_dir.glob("*_repeat_*.json")):
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
            json_field = result["json"]

            # Handle case where json might already be a dict or a string
            if isinstance(json_field, dict):
                parsed_json = json_field
            elif isinstance(json_field, str):
                parsed_json = json.loads(json_field)
            else:
                logger.warning(f"Unexpected json type in {result_file}: {type(json_field)}")
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
            logger.error(f"Error processing {result_file}: {e}")
            continue

    df = pd.DataFrame(rows)
    logger.info(f"Created dataframe with {len(df)} rows ({len(rows)} responses)")
    return df


def main(config_path: Path, status_only: bool = False, force: bool = False):
    """Main processing logic."""

    # Load config
    raw = yaml.safe_load(config_path.read_text())
    cfg = BatchConfig(**raw)

    # Derive experiment directory from config
    experiment_dir = RESULTS_DIR / cfg.experiment_name

    # Load metadata
    metadata = load_batch_metadata(experiment_dir)
    batch_id = metadata["batch_id"]
    provider = metadata["provider"]
    model = metadata["model"]

    logger.info(f"Config: {config_path}")
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Experiment dir: {experiment_dir}")
    logger.info(f"Provider: {provider}")
    logger.info(f"Model: {model}")
    logger.info(f"Batch ID: {batch_id}")

    # Check status
    if provider == "openai":
        status_info = check_openai_batch_status(batch_id)
    elif provider == "anthropic":
        status_info = check_anthropic_batch_status(batch_id)
    elif provider == "google":
        status_info = check_google_batch_status(batch_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    logger.info(f"Status: {status_info['status']}")
    logger.info(f"Request counts: {status_info['request_counts']}")

    if status_only:
        logger.info("Status check complete (--status-only flag set)")
        return

    # Check if completed
    completed_statuses = {
        "openai": "completed",
        "anthropic": "ended",
        "google": "COMPLETED",
    }

    if status_info["status"] != completed_statuses.get(provider):
        logger.warning(f"Batch not yet completed. Current status: {status_info['status']}")
        logger.info("Run with --status-only to check status without downloading")
        return

    # Download results
    logger.info("Downloading batch results...")
    if provider == "openai":
        output_path = download_openai_results(batch_id, experiment_dir)
    elif provider == "anthropic":
        output_path = download_anthropic_results(batch_id, experiment_dir)
    elif provider == "google":
        output_path = download_google_results(batch_id, experiment_dir)

    # Parse and save individual results
    logger.info("Parsing and saving individual results...")

    parse_and_save_results(
        output_path=output_path,
        experiment_dir=experiment_dir,
        provider=provider,
        model=model,
        cfg=cfg,
        force=force,
    )

    # Create CSV dataframe
    logger.info("Creating results dataframe...")
    df = create_results_dataframe(experiment_dir, provider, model)

    if not df.empty:
        csv_path = experiment_dir / f"{cfg.experiment_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results CSV to: {csv_path}")

    logger.info("=" * 60)
    logger.info("Batch processing complete!")
    logger.info(f"Results saved to: {experiment_dir}")
    if not df.empty:
        logger.info(f"CSV: {csv_path}")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Process completed batch results from provider batch API"
    )
    parser.add_argument(
        "config",
        help="Path to experiment config file (e.g., experiments/exp1.2.0_batch_openai.yaml)"
    )
    parser.add_argument("--status-only", action="store_true", help="Only check status, don't download results")
    parser.add_argument("--force", action="store_true", help="Force re-download and overwrite existing results")
    args = parser.parse_args()

    try:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        main(config_path, args.status_only, args.force)
    except Exception as exc:
        logger.error(f"Failed to process batch: {exc}")
        raise
