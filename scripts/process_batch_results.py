"""Process completed batch results from provider batch APIs.

Usage:
    python scripts/process_batch_results.py experiments/exp1.2.0_batch_openai.yaml

    # Check status without downloading
    python scripts/process_batch_results.py experiments/exp1.2.0_batch_openai.yaml --status-only

    # Force re-download even if results exist
    python scripts/process_batch_results.py experiments/exp1.2.0_batch_openai.yaml --force

    # Concatenate results to an existing file (or create it)
    python scripts/process_batch_results.py experiments/exp1.2.0_batch_openai.yaml --file data/combined.csv

This script:
1. Loads the experiment config to find the experiment folder
2. Reads batch_metadata.json from the experiment folder
3. Checks batch status with the provider
4. Downloads results if batch is complete
5. Parses and saves individual result files
6. Outputs CSV with columns: provider, model, temperature, scale, repeat, item, response, raw_output
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
    import requests

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable 'GEMINI_API_KEY' must be set")

    client = genai.Client(api_key=api_key)
    batch = client.batches.get(name=batch_id)

    logger.info(f"Batch object: {batch}")

    if not hasattr(batch, 'state') or batch.state.name != "JOB_STATE_SUCCEEDED":
        status = batch.state.name if hasattr(batch, 'state') else "UNKNOWN"
        raise RuntimeError(f"Batch not completed yet. Status: {status}")

    # Get the destination file name from batch.dest.file_name
    dest = getattr(batch, 'dest', None)
    if not dest:
        raise RuntimeError(f"Batch completed but no dest available. Batch: {batch}")

    # dest is a BatchJobDestination object with file_name attribute
    file_name = getattr(dest, 'file_name', None)
    if not file_name:
        raise RuntimeError(f"Batch dest has no file_name. dest: {dest}")

    logger.info(f"Downloading results from: {file_name}")

    output_path = output_dir / "batch_output.jsonl"

    # Workaround for Google bug #1759: Batch API generates file IDs > 40 chars
    # which the Files API rejects. Use direct HTTP download instead.
    # See: https://github.com/googleapis/python-genai/issues/1759
    try:
        # Construct direct download URL
        # file_name is like "files/batch-xxx", we need just the ID part
        file_id = file_name.replace("files/", "")
        download_url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}:download?alt=media&key={api_key}"

        logger.info(f"Downloading via direct URL (workaround for bug #1759)")
        response = requests.get(download_url)
        response.raise_for_status()

        # Write content to file
        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded batch output to {output_path} ({len(response.content)} bytes)")
        return output_path

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error downloading file: {e}")
        logger.error(f"Response: {e.response.text if e.response else 'No response'}")
        raise RuntimeError(
            f"Could not download batch results file. File: {file_name}\n"
            f"Error: {e}\n"
            f"Batch object: {batch}"
        )
    except Exception as e:
        logger.error(f"Failed to download file {file_name}: {e}")
        raise RuntimeError(
            f"Could not download batch results file. File: {file_name}\n"
            f"Error: {e}\n"
            f"Batch object: {batch}"
        )


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
                    logger.error(f"Failed request: {custom_id} - {response.get('error') if response else 'no response'}")
                    results_failed += 1
                    continue

                # Extract text from Gemini response - handle different response structures
                try:
                    candidates = response.get("candidates", [])
                    if not candidates:
                        logger.error(f"No candidates in response for {custom_id}: {response}")
                        results_failed += 1
                        continue

                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])

                    if parts:
                        raw_text = parts[0].get("text", "")
                    elif "text" in content:
                        # Alternative structure: content.text directly
                        raw_text = content["text"]
                    else:
                        logger.error(f"Could not extract text from response for {custom_id}: {response}")
                        results_failed += 1
                        continue

                except (KeyError, IndexError, TypeError) as e:
                    logger.error(f"Error parsing Google response for {custom_id}: {e}")
                    logger.debug(f"Response structure: {response}")
                    results_failed += 1
                    continue

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


def create_results_dataframe(
    experiment_dir: Path, provider: str, model: str, temperature: float
) -> pd.DataFrame:
    """Create a tidy dataframe from all result files.

    Returns:
        DataFrame with columns: provider, model, temperature, scale, repeat, item, response, raw_output
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
            raw_output = result.get("raw", "")

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
                    "temperature": temperature,
                    "scale": scale_name,
                    "repeat": repeat_num,
                    "item": item_num,
                    "response": response_value,
                    "raw_output": raw_output,
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


def process_single_batch(
    batch_entry: dict,
    experiment_dir: Path,
    cfg: BatchConfig,
    status_only: bool,
    force: bool,
) -> pd.DataFrame | None:
    """Process a single batch entry.

    Returns DataFrame of results, or None if batch not ready or status_only.
    """
    batch_id = batch_entry["batch_id"]
    provider = batch_entry["provider"]
    model = batch_entry["model"]
    safe_model = model.replace("/", "_").replace(":", "_")

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
        return None

    # Check if completed
    completed_statuses = {
        "openai": "completed",
        "anthropic": "ended",
        "google": "JOB_STATE_SUCCEEDED",
    }

    if status_info["status"] != completed_statuses.get(provider):
        logger.warning(f"Batch not yet completed. Current status: {status_info['status']}")
        return None

    # Download results
    logger.info("Downloading batch results...")
    if provider == "openai":
        output_path = download_openai_results(batch_id, experiment_dir)
    elif provider == "anthropic":
        output_path = download_anthropic_results(batch_id, experiment_dir)
    elif provider == "google":
        output_path = download_google_results(batch_id, experiment_dir)

    # Rename output file to be model-specific
    model_output_path = experiment_dir / f"batch_output_{provider}_{safe_model}.jsonl"
    if output_path != model_output_path:
        output_path.rename(model_output_path)
        output_path = model_output_path

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

    # Create dataframe for this model
    logger.info("Creating results dataframe...")
    df = create_results_dataframe(experiment_dir, provider, model, cfg.temperature)

    return df


def main(
    config_path: Path,
    status_only: bool = False,
    force: bool = False,
    output_file: Path | None = None,
):
    """Main processing logic for multiple batches."""

    # Load config
    raw = yaml.safe_load(config_path.read_text())
    cfg = BatchConfig(**raw)

    # Derive experiment directory from config
    experiment_dir = RESULTS_DIR / cfg.experiment_name

    # Load metadata
    metadata = load_batch_metadata(experiment_dir)
    batches = metadata["batches"]

    logger.info(f"Config: {config_path}")
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Experiment dir: {experiment_dir}")
    logger.info(f"Number of batches: {len(batches)}")

    # Process each batch
    all_dfs = []
    completed_count = 0
    pending_count = 0

    for i, batch_entry in enumerate(batches, 1):
        logger.info("=" * 60)
        logger.info(f"Processing batch {i}/{len(batches)}")

        df = process_single_batch(
            batch_entry=batch_entry,
            experiment_dir=experiment_dir,
            cfg=cfg,
            status_only=status_only,
            force=force,
        )

        if df is not None and not df.empty:
            all_dfs.append(df)
            completed_count += 1
        elif not status_only:
            pending_count += 1

    if status_only:
        logger.info("=" * 60)
        logger.info("Status check complete (--status-only flag set)")
        return

    # Combine all results into a single CSV
    logger.info("=" * 60)
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Save to experiment directory
        csv_path = experiment_dir / f"{cfg.experiment_name}.csv"
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"Saved combined results CSV to: {csv_path}")
        logger.info(f"Total rows: {len(combined_df)}")

        # If --file specified, concatenate to that file
        if output_file:
            if output_file.exists():
                existing_df = pd.read_csv(output_file)
                merged_df = pd.concat([existing_df, combined_df], ignore_index=True)
                merged_df.to_csv(output_file, index=False)
                logger.info(f"Concatenated results to existing file: {output_file}")
                logger.info(f"Previous rows: {len(existing_df)}, Added: {len(combined_df)}, Total: {len(merged_df)}")
            else:
                combined_df.to_csv(output_file, index=False)
                logger.info(f"Created new output file: {output_file}")
                logger.info(f"Total rows: {len(combined_df)}")
    else:
        logger.warning("No results to save - all batches may still be pending")

    logger.info("=" * 60)
    logger.info(f"Batch processing complete!")
    logger.info(f"Completed: {completed_count}/{len(batches)} batches")
    if pending_count > 0:
        logger.warning(f"Pending: {pending_count} batches still processing")
    logger.info(f"Results saved to: {experiment_dir}")


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
    parser.add_argument(
        "--file",
        help="Optional output file to concatenate results to (creates if doesn't exist)"
    )
    args = parser.parse_args()

    try:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        output_file = None
        if args.file:
            output_file = Path(args.file).expanduser().resolve()

        main(config_path, args.status_only, args.force, output_file)
    except Exception as exc:
        logger.error(f"Failed to process batch: {exc}")
        raise
