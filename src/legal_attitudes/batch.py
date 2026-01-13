"""Helpers for running repeated model queries via LiteLLM batch_completion and provider batch APIs."""
import json
from pathlib import Path

from anthropic import Anthropic
from google import genai
from litellm import batch_completion
from loguru import logger
from openai import OpenAI

from legal_attitudes.api import extract_json, make_refusal_response
from legal_attitudes.config import BatchConfig
from legal_attitudes.schemas import get_schema


def run_repeats(
    prompt_text: str,
    model: str,
    schema_name: str,
    repeats: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 500,
) -> list[dict]:
    """Run the same prompt N times using LiteLLM batch_completion.

    Args:
        prompt_text: The prompt to send.
        model: LiteLLM model string (e.g., "openrouter/qwen/qwen3-max").
        schema_name: Name of the Pydantic schema for parsing/refusal.
        repeats: Number of times to run.
        temperature: Sampling temperature.
        max_tokens: Max tokens in response.

    Returns:
        List of dicts with run_index, raw, json keys.
    """
    # Look up the Pydantic schema for JSON extraction and refusal handling
    schema_cls = get_schema(schema_name)

    # Build N identical message lists—one per repeat
    messages = [[{"role": "user", "content": prompt_text}] for _ in range(repeats)]

    # batch_completion sends all requests concurrently and handles retries
    responses = batch_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Process each response: extract JSON or mark as refusal
    results = []
    for i, resp in enumerate(responses):
        if isinstance(resp, Exception):
            print(f"Error: {resp}")
            break
        # Get raw text from response
        raw = resp.choices[0].message.content
        # Try to extract JSON; if missing/malformed, treat as refusal
        extracted = extract_json(raw)
        json_out = extracted if extracted else make_refusal_response(schema_cls)
        results.append({"run_index": i, "raw": raw, "json": json_out})

    return results


# ---------------------------------------------------------------------------
# Provider Batch API Submission Functions
# ---------------------------------------------------------------------------


def create_openai_batch(
    cfg: BatchConfig,
    prompt_texts: dict[str, str],
    batch_input_path: Path
) -> dict:
    """Create and submit OpenAI batch request.

    Args:
        cfg: Batch configuration
        prompt_texts: Dict mapping prompt stem names to prompt text
        batch_input_path: Path where to save the JSONL batch input file

    Returns:
        dict with batch_id, input_file_id, status, and num_requests
    """
    client = OpenAI()
    model_cfg = cfg.models[0]

    # Build JSONL batch requests
    requests = []
    for repeat_idx in range(cfg.repeats):
        for prompt_cfg in cfg.prompts:
            prompt_name = prompt_cfg.path.stem
            prompt_text = prompt_texts[prompt_name]
            schema_cls = get_schema(prompt_cfg.schema_name)

            custom_id = f"{prompt_name}_repeat_{repeat_idx:03d}"

            # Build request body
            body = {
                "model": model_cfg.name,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_completion_tokens": cfg.max_completion_tokens,
            }

            # Add temperature if model supports it
            from legal_attitudes.api import OPENAI_NO_TEMP_MODELS
            if model_cfg.name not in OPENAI_NO_TEMP_MODELS:
                body["temperature"] = cfg.temperature

            if cfg.seed is not None:
                body["seed"] = cfg.seed

            # Add structured output if requested
            if cfg.use_structured_output:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_cls.__name__,
                        "schema": schema_cls.model_json_schema(),
                        "strict": True,
                    },
                }

            requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            })

    # Write JSONL file
    with open(batch_input_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    logger.info(f"Created batch input file with {len(requests)} requests: {batch_input_path}")

    # Upload file to OpenAI
    with open(batch_input_path, "rb") as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )

    logger.info(f"Uploaded batch file: {batch_input_file.id}")

    # Create batch
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "experiment_name": cfg.experiment_name,
            "num_scales": str(len(cfg.prompts)),
            "num_repeats": str(cfg.repeats),
        }
    )

    logger.info(f"Created OpenAI batch: {batch.id}")

    return {
        "batch_id": batch.id,
        "input_file_id": batch_input_file.id,
        "status": batch.status,
        "num_requests": len(requests),
    }


def create_anthropic_batch(
    cfg: BatchConfig,
    prompt_texts: dict[str, str],
    batch_input_path: Path
) -> dict:
    """Create and submit Anthropic batch request.

    Args:
        cfg: Batch configuration
        prompt_texts: Dict mapping prompt stem names to prompt text
        batch_input_path: Path where to save the JSONL batch input file (for reference)

    Returns:
        dict with batch_id, status, and num_requests
    """
    client = Anthropic()
    model_cfg = cfg.models[0]

    # Build batch requests
    requests = []
    for repeat_idx in range(cfg.repeats):
        for prompt_cfg in cfg.prompts:
            prompt_name = prompt_cfg.path.stem
            prompt_text = prompt_texts[prompt_name]

            custom_id = f"{prompt_name}_repeat_{repeat_idx:03d}"

            requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": model_cfg.name,
                    "max_tokens": cfg.max_completion_tokens,
                    "temperature": cfg.temperature,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt_text}]
                        }
                    ],
                }
            })

    logger.info(f"Submitting Anthropic batch with {len(requests)} requests")

    # Create batch (Anthropic accepts requests directly, no file upload needed)
    batch = client.messages.batches.create(requests=requests)

    logger.info(f"Created Anthropic batch: {batch.id}")

    # Save requests locally for reference
    with open(batch_input_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    return {
        "batch_id": batch.id,
        "status": batch.processing_status,
        "num_requests": len(requests),
    }


def create_google_batch(
    cfg: BatchConfig,
    prompt_texts: dict[str, str],
    batch_input_path: Path
) -> dict:
    """Create and submit Google (Gemini) batch request.

    Args:
        cfg: Batch configuration
        prompt_texts: Dict mapping prompt stem names to prompt text
        batch_input_path: Path where to save the batch input file

    Returns:
        dict with batch_id, status, file_name, and num_requests
    """
    import os
    from google import genai
    from google.genai import types as gemini_types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Environment variable 'GEMINI_API_KEY' must be set for Gemini batching."
        )

    client = genai.Client(api_key=api_key)
    model_cfg = cfg.models[0]

    # Build JSONL batch requests
    requests = []
    for repeat_idx in range(cfg.repeats):
        for prompt_cfg in cfg.prompts:
            prompt_name = prompt_cfg.path.stem
            prompt_text = prompt_texts[prompt_name]
            schema_cls = get_schema(prompt_cfg.schema_name)

            custom_id = f"{prompt_name}_repeat_{repeat_idx:03d}"

            # Build Gemini batch request format
            request = {
                "key": custom_id,
                "request": {
                    "contents": [
                        {"parts": [{"text": prompt_text}], "role": "user"}
                    ],
                    "generation_config": {
                        "max_output_tokens": cfg.max_completion_tokens,
                        "temperature": cfg.temperature,
                    },
                },
            }

            # Add structured output if requested
            if cfg.use_structured_output:
                request["request"]["generation_config"]["response_mime_type"] = "application/json"
                request["request"]["generation_config"]["response_schema"] = schema_cls

            requests.append(request)

    # Write JSONL file
    with open(batch_input_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    logger.info(f"Created batch input file with {len(requests)} requests: {batch_input_path}")

    # Upload file to Gemini
    upload = client.files.upload(
        file=str(batch_input_path),
        config=gemini_types.UploadFileConfig(
            display_name=f"{cfg.experiment_name}_batch_input",
            mime_type="application/jsonl",
        ),
    )

    logger.info(f"Uploaded batch file: {upload.name}")

    # Create batch job
    batch_job = client.batches.create(
        model=model_cfg.name,
        src=upload.name,
    )

    logger.info(f"Created Gemini batch: {batch_job.name}")

    return {
        "batch_id": batch_job.name,
        "file_name": upload.name,
        "status": batch_job.state.name if hasattr(batch_job, 'state') else "SUBMITTED",
        "num_requests": len(requests),
    }
