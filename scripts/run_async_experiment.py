"""Run repeated queries for models without native batch APIs (e.g., OpenRouter).

Usage:
    python scripts/run_async_experiment.py experiments/exp.yaml
"""
import argparse
import json
import warnings
from pathlib import Path

# Suppress Pydantic serialization warnings from LiteLLM response objects
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

import litellm
import yaml
from loguru import logger
from tqdm import tqdm

from legal_attitudes.batch import run_repeats
from legal_attitudes.utils import ROOT, RESULTS_DIR, setup_logging

litellm.suppress_debug_info = True

def main(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    experiment_name = cfg["experiment_name"]
    temperature = cfg.get("temperature", 1.0)
    max_tokens = cfg.get("max_completion_tokens", 500)
    repeats = cfg.get("repeats", 50)
    # Rate limit handling settings
    chunk_size = cfg.get("concurrency", 5)
    max_retries = cfg.get("max_retries", 10)
    initial_backoff = cfg.get("initial_backoff", 5.0)

    # Build list of (prompt, model) pairs for progress bar
    tasks = [
        (prompt_cfg, model_cfg)
        for model_cfg in cfg["models"]
        for prompt_cfg in cfg["prompts"]
    ]

    pbar = tqdm(tasks, desc="Starting...")
    for prompt_cfg, model_cfg in pbar:
        prompt_path = ROOT / prompt_cfg["path"]
        prompt_text = prompt_path.read_text()
        schema_name = prompt_cfg["schema_name"]
        scale_name = prompt_path.stem

        provider = model_cfg["provider"]
        model_name = model_cfg["name"]
        litellm_model = f"{provider}/{model_name}"

        # Update progress bar description
        pbar.set_description(f"{scale_name} | {model_name}")

        # Save each run: experiment/model/{scale}_repeat_{###}.json
        safe_model = model_name.replace("/", "_").replace(":", "_")
        model_dir = RESULTS_DIR / experiment_name / f"{provider}_{safe_model}"

        missing_run_ids = [
            run_id
            for run_id in range(1, repeats + 1)
            if not (model_dir / f"{scale_name}_repeat_{run_id:03d}.json").exists()
        ]
        if not missing_run_ids:
            logger.info(f"Skip {scale_name} | {model_name}: all {repeats} runs present")
            continue
        if len(missing_run_ids) == repeats:
            logger.info(f"Run full {scale_name} | {model_name}: {repeats} runs")
        else:
            logger.info(
                f"Fill {scale_name} | {model_name}: {len(missing_run_ids)} missing "
                f"(ids {missing_run_ids})"
            )

        results = run_repeats(
            prompt_text=prompt_text,
            model=litellm_model,
            schema_name=schema_name,
            repeats=len(missing_run_ids),
            temperature=temperature,
            max_tokens=max_tokens,
            chunk_size=chunk_size,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        for r, run_id in zip(results, missing_run_ids):
            r["run_index"] = run_id - 1
            out_file = model_dir / f"{scale_name}_repeat_{run_id:03d}.json"
            out_file.write_text(json.dumps(r, indent=2))

    logger.info(f"Done. Saved {len(tasks)} × {repeats} runs.")


if __name__ == "__main__":
    setup_logging(use_tqdm=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment YAML")
    args = parser.parse_args()
    main(args.config)
