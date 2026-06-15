# code/utils/cluster/cluster_model_interface.py
"""Model interface for local vLLM and OpenAI-compatible API backends."""

from __future__ import annotations

import asyncio
import datetime
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

# Import local models lazily so task code can still be inspected in
# environments where the cluster runtime files are not installed.
try:
    from .local_models_cluster import (
        MODEL_PATHS,
        batch_query_local_model_vllm,
        query_local_model_vllm,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised in review bundles
    MODEL_PATHS = {}

    async def query_local_model_vllm(*args, **kwargs):
        raise RuntimeError(
            "Cluster local model support is unavailable because local_models_cluster.py "
            "is not installed in this environment."
        )

    async def batch_query_local_model_vllm(*args, **kwargs):
        raise RuntimeError(
            "Cluster local model support is unavailable because local_models_cluster.py "
            "is not installed in this environment."
        )


PromptInput = Union[str, Sequence[Mapping[str, Any]]]

# Local vLLM model configurations. API model names are passed through directly.
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {}
for model_name in MODEL_PATHS.keys():
    MODEL_CONFIGS[model_name] = {
        "provider": "local",
        "model_id": model_name,
        "max_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.9,
    }


def get_inference_backend(backend_override: Optional[str] = None) -> str:
    """Resolve and validate the inference backend."""
    backend = (backend_override or os.environ.get("ROR_INFERENCE_BACKEND", "vllm")).strip().lower()
    if backend == "openai":
        backend = "api"
    if backend not in {"vllm", "api"}:
        raise ValueError(
            f"Inference backend '{backend}' is not supported. "
            "Use 'vllm' for local/cluster runs or 'api' for OpenAI-compatible endpoints."
        )
    return backend


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return max(1, int(default))
    try:
        value = int(raw)
        if value < 1:
            raise ValueError
        return value
    except (TypeError, ValueError):
        if not hasattr(_parse_int_env, "_invalid_value_warning_printed"):
            print(f"[inference] Ignoring invalid {name}={raw!r}; using {default}.")
            _parse_int_env._invalid_value_warning_printed = True
        return max(1, int(default))


def _parse_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        if not hasattr(_parse_float_env, "_invalid_value_warning_printed"):
            print(f"[inference] Ignoring invalid {name}={raw!r}; using {default}.")
            _parse_float_env._invalid_value_warning_printed = True
        return float(default)


def get_api_max_concurrent(default: int = 8) -> int:
    """Return the configured API concurrency limit."""
    return _parse_int_env("ROR_API_MAX_CONCURRENT", default)


def get_vllm_batch_size(default: int = 16) -> int:
    """Return the task-level batch size.

    The function name is retained for older task code. In API mode the value
    controls API concurrency; in vLLM mode it controls vLLM prompt batch size.
    """
    if get_inference_backend() == "api":
        return get_api_max_concurrent(default)
    return _parse_int_env("ROR_VLLM_BATCH_SIZE", default)


def _json_env(name: str, default: Any) -> Any:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc}") from exc


def _api_messages(prompt: PromptInput) -> List[Mapping[str, Any]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    messages = list(prompt)
    if not messages:
        raise ValueError("API prompt messages cannot be empty.")
    return messages


def _api_token_kwargs(max_tokens: Optional[int]) -> Dict[str, int]:
    token_param = os.environ.get("ROR_API_TOKEN_PARAM", "max_tokens").strip()
    if not token_param or token_param == "none" or max_tokens is None:
        return {}
    if token_param not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError(
            "ROR_API_TOKEN_PARAM must be one of max_tokens, max_completion_tokens, or none."
        )
    return {token_param: int(max_tokens)}


def _api_client():
    try:
        from openai import AsyncOpenAI
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("The API backend requires the openai package. Install openai>=1.0.") from exc

    key_env = os.environ.get("ROR_API_KEY_ENV", "OPENAI_API_KEY")
    api_key = os.environ.get(key_env)
    if not api_key:
        raise RuntimeError(f"API backend requires an API key in ${key_env}.")

    base_url = os.environ.get("ROR_API_BASE_URL")
    if base_url:
        return AsyncOpenAI(api_key=api_key, base_url=base_url)
    return AsyncOpenAI(api_key=api_key)


async def _query_api_with_client(
    client: Any,
    prompt: PromptInput,
    model_name: str,
    max_tokens: Optional[int],
) -> str:
    request: Dict[str, Any] = {
        "model": model_name,
        "messages": _api_messages(prompt),
        "temperature": _parse_float_env("ROR_API_TEMPERATURE", 0.1),
        "top_p": _parse_float_env("ROR_API_TOP_P", 0.9),
        **_api_token_kwargs(max_tokens),
    }

    extra_body = _json_env("ROR_API_EXTRA_BODY_JSON", None)
    extra_headers = _json_env("ROR_API_EXTRA_HEADERS_JSON", None)
    if extra_body:
        request["extra_body"] = extra_body
    if extra_headers:
        request["extra_headers"] = extra_headers

    response = await client.chat.completions.create(**request)
    if not response.choices:
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(getattr(item, "text", item)))
        return "".join(parts)
    return content or ""


def get_available_local_models() -> List[str]:
    """Get list of available local models on the target cluster."""
    available = []
    for model_name, model_path in MODEL_PATHS.items():
        if os.path.exists(model_path):
            available.append(model_name)
    return available


async def query_model(
    prompt: PromptInput,
    model_name: str = "qwen3-14b",
    max_retries: int = 3,
    retry_delay: int = 5,
    max_tokens: Optional[int] = None,
    quantization: str = None,
    double_quant: bool = None,
    compute_dtype: str = None,
    backend: Optional[str] = None,
) -> str:
    """Query one prompt through the configured backend."""
    inference_backend = get_inference_backend(backend)

    if inference_backend == "api":
        client = _api_client()
        attempts = 0
        try:
            while attempts < max_retries:
                try:
                    return await _query_api_with_client(client, prompt, model_name, max_tokens)
                except Exception as exc:  # noqa: BLE001
                    attempts += 1
                    print(f"[API retry {attempts}/{max_retries}] Error querying model: {exc}")
                    if attempts == max_retries:
                        raise Exception(f"Failed to query API model after {max_retries} attempts: {exc}") from exc
                    await asyncio.sleep(retry_delay)
        finally:
            await client.close()

    if model_name not in MODEL_CONFIGS:
        available = get_available_local_models()
        raise ValueError(f"Invalid model name: {model_name}. Available models: {', '.join(available)}")

    config = MODEL_CONFIGS[model_name]
    max_tokens_to_use = max_tokens if max_tokens is not None else config["max_tokens"]

    # Keep compatibility with old caller signatures while enforcing vLLM execution.
    if any(v not in (None, False) for v in [quantization, double_quant, compute_dtype]):
        if not hasattr(query_model, "_legacy_quant_warning_printed"):
            print(
                "[vLLM] Ignoring legacy transformers quantization flags "
                "(quantization/double_quant/compute_dtype)."
            )
            query_model._legacy_quant_warning_printed = True

    available_models = get_available_local_models()
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not downloaded. Available models: {', '.join(available_models)}")

    attempts = 0
    while attempts < max_retries:
        try:
            return await query_local_model_vllm(
                prompt=prompt,
                model_name=config["model_id"],
                max_tokens=max_tokens_to_use,
                temperature=config["temperature"],
                top_p=config["top_p"],
            )
        except Exception as exc:  # noqa: BLE001
            if "vllm is not available in this environment" in str(exc).lower() or "no module named 'vllm'" in str(exc).lower():
                raise RuntimeError(str(exc)) from exc
            attempts += 1
            print(f"[vLLM retry {attempts}/{max_retries}] Error querying model: {exc}")
            if attempts == max_retries:
                raise Exception(f"Failed to query model after {max_retries} attempts: {exc}") from exc
            await asyncio.sleep(retry_delay)

    raise Exception("Unexpected error in query_model")


def _expand_model_names(model_names: Union[str, List[str], None], count: int) -> List[str]:
    if model_names is None:
        return ["qwen3-14b"] * count
    if isinstance(model_names, str):
        return [model_names] * count
    if len(model_names) == 1:
        return list(model_names) * count
    if len(model_names) != count:
        raise ValueError(f"Expected {count} model names, got {len(model_names)}.")
    return list(model_names)


async def batch_query_model(
    prompts: List[PromptInput],
    model_names: Union[str, List[str], None] = None,
    max_concurrent: int = 1,
    max_tokens: Optional[int] = None,
    backend: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> List[str]:
    """Query multiple prompts through the configured backend."""
    expanded_model_names = _expand_model_names(model_names, len(prompts))
    inference_backend = get_inference_backend(backend)

    if inference_backend == "api":
        client = _api_client()
        semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

        async def _worker(index: int, prompt: PromptInput, model_name: str) -> str:
            attempts = 0
            while attempts < max_retries:
                try:
                    async with semaphore:
                        return await _query_api_with_client(client, prompt, model_name, max_tokens)
                except Exception as exc:  # noqa: BLE001
                    attempts += 1
                    if attempts < max_retries:
                        print(f"[API batch retry {attempts}/{max_retries}] item {index} failed: {exc}")
                        await asyncio.sleep(retry_delay)
                    else:
                        return f"[QUERY_ERROR] {exc}"
            return "[QUERY_ERROR] unexpected API retry state"

        try:
            return await asyncio.gather(
                *[
                    _worker(index, prompt, model_name)
                    for index, (prompt, model_name) in enumerate(zip(prompts, expanded_model_names))
                ]
            )
        finally:
            await client.close()

    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    grouped_prompts: Dict[str, List[PromptInput]] = defaultdict(list)
    for idx, (prompt, model_name) in enumerate(zip(prompts, expanded_model_names)):
        grouped_indices[model_name].append(idx)
        grouped_prompts[model_name].append(prompt)

    outputs: List[str] = [""] * len(prompts)
    for model_name, prompts_for_model in grouped_prompts.items():
        config = MODEL_CONFIGS.get(model_name)
        if not config:
            raise ValueError(f"Invalid model name: {model_name}")
        max_tokens_for_model = max_tokens if max_tokens is not None else config["max_tokens"]
        responses = await batch_query_local_model_vllm(
            prompts=prompts_for_model,
            model_name=config["model_id"],
            max_tokens=max_tokens_for_model,
            temperature=config["temperature"],
            top_p=config["top_p"],
        )
        for idx, response in zip(grouped_indices[model_name], responses):
            outputs[idx] = response
    return outputs


async def query_prompts_batched(
    prompts: List[PromptInput],
    model_name: str,
    max_tokens: Optional[int] = None,
    backend: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> List[str]:
    """Query many prompts with backend-appropriate batching/concurrency."""
    if not prompts:
        return []

    inference_backend = get_inference_backend(backend)
    resolved_batch_size = max(1, int(batch_size)) if batch_size is not None else get_vllm_batch_size()

    if inference_backend == "api":
        return await batch_query_model(
            prompts=prompts,
            model_names=model_name,
            max_concurrent=resolved_batch_size,
            max_tokens=max_tokens,
            backend="api",
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    outputs: List[str] = [""] * len(prompts)
    for start in range(0, len(prompts), resolved_batch_size):
        end = min(start + resolved_batch_size, len(prompts))
        chunk_prompts = prompts[start:end]

        attempts = 0
        chunk_done = False
        while attempts < max_retries:
            try:
                chunk_responses = await batch_query_model(
                    prompts=chunk_prompts,
                    model_names=model_name,
                    max_tokens=max_tokens,
                    backend="vllm",
                )
                if len(chunk_responses) != len(chunk_prompts):
                    raise RuntimeError(
                        f"Batch size mismatch for chunk [{start}:{end}] "
                        f"(got {len(chunk_responses)} responses for {len(chunk_prompts)} prompts)."
                    )
                outputs[start:end] = chunk_responses
                chunk_done = True
                break
            except Exception as exc:  # noqa: BLE001
                attempts += 1
                if attempts < max_retries:
                    print(f"[vLLM batch retry {attempts}/{max_retries}] chunk {start}:{end} failed: {exc}")
                    await asyncio.sleep(retry_delay)

        if chunk_done:
            continue

        print(
            f"[vLLM batch fallback] chunk {start}:{end} failed after {max_retries} retries; "
            "falling back to single-prompt queries."
        )
        for offset, prompt in enumerate(chunk_prompts):
            target_idx = start + offset
            try:
                outputs[target_idx] = await query_model(
                    prompt=prompt,
                    model_name=model_name,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    max_tokens=max_tokens,
                    backend="vllm",
                )
            except Exception as exc:  # noqa: BLE001
                outputs[target_idx] = f"[QUERY_ERROR] {exc}"

    return outputs


def save_model_response(
    response: Union[str, List[str]],
    output_dir: str,
    prompt: Union[str, List[str], None] = None,
    model_name: Union[str, List[str], None] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save model response and metadata to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"response_{timestamp}.json")

    responses = [response] if isinstance(response, str) else response
    prompts = [prompt] if isinstance(prompt, str) else prompt if prompt else None
    model_names = [model_name] if isinstance(model_name, str) else model_name if model_name else None

    data = {
        "responses": responses,
        "timestamp": timestamp,
    }

    if prompts:
        data["prompts"] = prompts
    if model_names:
        data["model_names"] = model_names
    if metadata:
        data["metadata"] = metadata

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


def list_cluster_models():
    """List registered local models and show API backend usage."""
    print("=== Available Local Models ===\n")
    print("Local Models (downloaded to the target cluster):")
    available = get_available_local_models()

    for model_name, model_path in sorted(MODEL_PATHS.items()):
        exists = os.path.exists(model_path)
        if model_name in MODEL_CONFIGS:
            status = "AVAILABLE" if exists else "MISSING"
            print(f"  {status} {model_name}")

    print(f"\nTotal available: {len(available)} models")
    print("\nLocal/cluster example:")
    print("  python code/main.py --inference_backend vllm --model qwen3-14b --task tapatan")
    print("\nOpenAI-compatible API example:")
    print("  export OPENAI_API_KEY=...")
    print("  python code/main.py --inference_backend api --model gpt-5.4 --api_token_param max_completion_tokens --task tapatan")
