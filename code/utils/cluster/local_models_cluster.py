"""Local model registry for cluster vLLM runs.

Set ROR_MODEL_ROOT to the directory that contains the downloaded model
folders. The default placeholder keeps the submission anonymous.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

PromptInput = Union[str, Sequence[Mapping[str, Any]]]

MODEL_ROOT = Path(os.environ.get("ROR_MODEL_ROOT", "/path/to/downloaded/models")).expanduser()


def _model_path(*parts: str) -> str:
    return str(MODEL_ROOT.joinpath(*parts))


# Final-paper open-weight models supported by the cluster/vLLM runner.
MODEL_PATHS = {
    # Qwen
    "qwen2.5-32b-instruct": _model_path("qwen", "instruction-tuned", "qwen25-32b-instruct"),
    "qwen2.5-32b": _model_path("qwen", "base", "qwen25-32b-base"),
    "qwen3-14b": _model_path("qwen", "qwen3-14b"),
    "qwen3-14b-base": _model_path("qwen", "qwen3-14b-base"),
    "qwen3-8b": _model_path("qwen", "qwen3-8b"),
    "qwen3-8b-base": _model_path("qwen", "qwen3-8b-base"),
    # Google Gemma
    "gemma-27b-instruct": _model_path("google", "gemma-3-27b-it"),
    "gemma-27b-base": _model_path("google", "gemma-3-27b-pt"),
    "gemma-12b-instruct": _model_path("google", "gemma-3-12b-it"),
    "gemma-12b-base": _model_path("google", "gemma-3-12b-pt"),
    # AllenAI OLMo
    "olmo-3.1-32b-instruct": _model_path("allenai", "olmo-3.1-32b-instruct"),
    "olmo-3-1125-32b": _model_path("allenai", "olmo-3-1125-32b"),
    "olmo-3-7b-instruct": _model_path("allenai", "olmo-3-7b-instruct"),
    "olmo-3-1025-7b": _model_path("allenai", "olmo-3-1025-7b"),
}


async def query_local_model_vllm(
    prompt: PromptInput,
    model_name: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    """Query a local model through the shared vLLM engine."""
    from .vllm_singleton import async_generate_response, resolve_vllm_config

    model_path = MODEL_PATHS.get(model_name)
    if not model_path:
        raise ValueError(f"Model {model_name} not found in MODEL_PATHS")
    if not os.path.exists(model_path):
        raise ValueError(
            f"Model path {model_path} does not exist. "
            "Set ROR_MODEL_ROOT to the cluster directory containing downloaded models."
        )

    response = await async_generate_response(
        prompt=prompt,
        model_path=model_path,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        engine_config=resolve_vllm_config(),
    )
    if not response:
        raise RuntimeError("Model returned empty response")
    return response


async def batch_query_local_model_vllm(
    prompts: Sequence[PromptInput],
    model_name: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> list[str]:
    """Batch query a single model with vLLM continuous batching."""
    from .vllm_singleton import async_generate_batch_responses, resolve_vllm_config

    model_path = MODEL_PATHS.get(model_name)
    if not model_path:
        raise ValueError(f"Model {model_name} not found in MODEL_PATHS")
    if not os.path.exists(model_path):
        raise ValueError(
            f"Model path {model_path} does not exist. "
            "Set ROR_MODEL_ROOT to the cluster directory containing downloaded models."
        )

    responses = await async_generate_batch_responses(
        prompts=prompts,
        model_path=model_path,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        engine_config=resolve_vllm_config(),
    )
    if len(responses) != len(prompts):
        raise RuntimeError(
            f"vLLM batch output size mismatch: got {len(responses)}, expected {len(prompts)}"
        )
    return responses


def list_available_models() -> list[str]:
    """List models that are present under ROR_MODEL_ROOT."""
    return [name for name, path in MODEL_PATHS.items() if os.path.exists(path)]


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Return basic path and size information for a registered model."""
    if model_name not in MODEL_PATHS:
        return {"error": "Model not found"}

    model_path = MODEL_PATHS[model_name]
    info: Dict[str, Any] = {
        "name": model_name,
        "path": model_path,
        "exists": os.path.exists(model_path),
    }
    if info["exists"]:
        try:
            total_size = 0
            for dirpath, _, filenames in os.walk(model_path):
                for filename in filenames:
                    total_size += os.path.getsize(os.path.join(dirpath, filename))
            info["size_gb"] = total_size / (1024**3)
        except OSError:
            info["size_gb"] = "unknown"
    return info
