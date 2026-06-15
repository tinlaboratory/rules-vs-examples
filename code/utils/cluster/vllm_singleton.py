# code/utils/cluster/vllm_singleton.py
# Singleton vLLM engine loader for cluster inference.

from __future__ import annotations

import asyncio
import gc
import json
import os
from dataclasses import dataclass
from threading import Lock
from typing import Any, Mapping, Optional, Sequence, Union


@dataclass(frozen=True)
class VLLMEngineConfig:
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    enforce_eager: bool = False
    quantization: Optional[str] = None
    swap_space: float = 4.0
    cpu_offload_gb: float = 0.0
    trust_remote_code: bool = True


_engine = None
_engine_model_name: Optional[str] = None
_engine_model_path: Optional[str] = None
_engine_config: Optional[VLLMEngineConfig] = None
_engine_lock = Lock()
_warn_once_cache: set[str] = set()

PromptInput = Union[str, Sequence[Mapping[str, Any]]]


def _warn_once(key: str, message: str) -> None:
    if key in _warn_once_cache:
        return
    _warn_once_cache.add(key)
    print(message)


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    return int(value)


def _parse_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    return float(value)


def resolve_vllm_config(
    tensor_parallel_size: Optional[int] = None,
    dtype: Optional[str] = None,
    gpu_memory_utilization: Optional[float] = None,
    max_model_len: Optional[int] = None,
    enforce_eager: Optional[bool] = None,
    quantization: Optional[str] = None,
    swap_space: Optional[float] = None,
    cpu_offload_gb: Optional[float] = None,
    trust_remote_code: Optional[bool] = None,
) -> VLLMEngineConfig:
    """Resolve vLLM config from explicit overrides, then environment, then defaults."""
    env_tp_default = _parse_int(
        os.environ.get("ROR_VLLM_TENSOR_PARALLEL_SIZE") or os.environ.get("ROR_NUM_GPUS"),
        1,
    )
    resolved_tp = tensor_parallel_size if tensor_parallel_size is not None else env_tp_default
    resolved_tp = max(1, int(resolved_tp))

    resolved_dtype = dtype if dtype is not None else os.environ.get("ROR_VLLM_DTYPE", "auto")
    resolved_gpu_util = (
        gpu_memory_utilization
        if gpu_memory_utilization is not None
        else _parse_float(os.environ.get("ROR_VLLM_GPU_MEMORY_UTILIZATION"), 0.9)
    )
    resolved_max_len = (
        max_model_len
        if max_model_len is not None
        else (
            int(os.environ["ROR_VLLM_MAX_MODEL_LEN"])
            if os.environ.get("ROR_VLLM_MAX_MODEL_LEN")
            else None
        )
    )
    resolved_enforce_eager = (
        enforce_eager
        if enforce_eager is not None
        else _parse_bool(os.environ.get("ROR_VLLM_ENFORCE_EAGER"), False)
    )
    resolved_quant = (
        quantization
        if quantization is not None
        else (os.environ.get("ROR_VLLM_QUANTIZATION") or None)
    )
    resolved_swap_space = (
        swap_space
        if swap_space is not None
        else _parse_float(os.environ.get("ROR_VLLM_SWAP_SPACE"), 4.0)
    )
    resolved_cpu_offload_gb = (
        cpu_offload_gb
        if cpu_offload_gb is not None
        else _parse_float(os.environ.get("ROR_VLLM_CPU_OFFLOAD_GB"), 0.0)
    )
    resolved_trust_remote_code = (
        trust_remote_code
        if trust_remote_code is not None
        else _parse_bool(os.environ.get("ROR_VLLM_TRUST_REMOTE_CODE"), True)
    )

    return VLLMEngineConfig(
        tensor_parallel_size=resolved_tp,
        dtype=resolved_dtype,
        gpu_memory_utilization=resolved_gpu_util,
        max_model_len=resolved_max_len,
        enforce_eager=resolved_enforce_eager,
        quantization=resolved_quant,
        swap_space=resolved_swap_space,
        cpu_offload_gb=resolved_cpu_offload_gb,
        trust_remote_code=resolved_trust_remote_code,
    )


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_engine(model_path: str, config: VLLMEngineConfig):
    try:
        from vllm import LLM
    except Exception as exc:
        raise RuntimeError(
            "vLLM backend is requested but vllm is not available in this environment. "
            "Install it with `pip install vllm` on the target cluster."
        ) from exc

    kwargs = {
        "model": model_path,
        "tensor_parallel_size": config.tensor_parallel_size,
        "dtype": config.dtype,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "enforce_eager": config.enforce_eager,
        "trust_remote_code": config.trust_remote_code,
        "swap_space": config.swap_space,
        "cpu_offload_gb": config.cpu_offload_gb,
    }
    if config.max_model_len is not None:
        kwargs["max_model_len"] = config.max_model_len
    if config.quantization:
        kwargs["quantization"] = config.quantization

    return LLM(**kwargs)


def _get_engine(model_path: str, model_name: str, config: VLLMEngineConfig):
    global _engine, _engine_model_name, _engine_model_path, _engine_config

    with _engine_lock:
        if (
            _engine is not None
            and _engine_model_name == model_name
            and _engine_model_path == model_path
            and _engine_config == config
        ):
            return _engine

        if _engine is not None:
            print(f"[vLLM] Replacing loaded engine: {_engine_model_name} -> {model_name}")
            _engine = None
            _engine_model_name = None
            _engine_model_path = None
            _engine_config = None
            gc.collect()
            _clear_cuda_cache()

        print(
            "[vLLM] Loading engine "
            f"(model={model_name}, tp={config.tensor_parallel_size}, dtype={config.dtype})..."
        )
        _engine = _load_engine(model_path=model_path, config=config)
        _engine_model_name = model_name
        _engine_model_path = model_path
        _engine_config = config
        return _engine


def _coerce_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _parse_stop_sequences(stop: Optional[Sequence[str]] = None) -> Optional[list[str]]:
    if stop is not None:
        if isinstance(stop, str):
            value = _coerce_to_str(stop)
            return [value] if value != "" else None
        cleaned: list[str] = []
        for item in stop:
            value = _coerce_to_str(item)
            if value != "":
                cleaned.append(value)
        return cleaned or None

    raw = os.environ.get("ROR_VLLM_STOP")
    if raw is None or raw == "":
        return None

    parsed: list[str]
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, str):
            parsed = [decoded]
        elif isinstance(decoded, list):
            parsed = [_coerce_to_str(item) for item in decoded if _coerce_to_str(item) != ""]
        else:
            raise ValueError("expected JSON string or list of strings")
    except Exception:
        if "||" in raw:
            parsed = [part for part in raw.split("||") if part != ""]
        else:
            parsed = [raw]

    return parsed or None


def _is_chat_messages(prompt: Any) -> bool:
    if isinstance(prompt, str):
        return False
    if not isinstance(prompt, Sequence):
        return False
    if len(prompt) == 0:
        return False
    first = prompt[0]
    return isinstance(first, Mapping) and ("role" in first) and ("content" in first)


def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
                continue
            if isinstance(item, Mapping):
                text = item.get("text")
                if text:
                    pieces.append(_coerce_to_str(text))
                    continue
                data = item.get("content")
                if data:
                    pieces.append(_coerce_to_str(data))
                    continue
            pieces.append(_coerce_to_str(item))
        return "\n".join(piece for piece in pieces if piece)
    return _coerce_to_str(content)


def _fallback_chat_to_text(
    messages: Sequence[Mapping[str, Any]],
    add_generation_prompt: bool,
) -> str:
    lines: list[str] = []
    for message in messages:
        role = _coerce_to_str(message.get("role", "user")).strip().lower() or "user"
        content = _normalize_message_content(message.get("content", ""))
        lines.append(f"{role}: {content}")
    if add_generation_prompt:
        lines.append("assistant:")
    return "\n".join(lines)


def _resolve_tokenizer_for_chat(engine):
    tokenizer = None
    try:
        tokenizer = engine.get_tokenizer()
    except Exception as exc:
        _warn_once(
            "tokenizer_load_failure",
            f"[vLLM] Could not get tokenizer from engine for chat template formatting: {exc}",
        )
        return None

    if tokenizer is None:
        return None
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer
    nested = getattr(tokenizer, "tokenizer", None)
    if nested is not None and hasattr(nested, "apply_chat_template"):
        return nested
    return None


def _parse_chat_add_generation_prompt(default: bool = True) -> bool:
    return _parse_bool(os.environ.get("ROR_VLLM_CHAT_ADD_GENERATION_PROMPT"), default)


def _is_base_model(model_name: Optional[str]) -> bool:
    normalized = (model_name or "").strip().lower()
    if not normalized:
        return False

    if "base" in normalized:
        return True

    base_only_names = {
        "olmo-3-1025-7b",
    }
    return normalized in base_only_names


def _resolve_text_prompts_as_chat(model_name: Optional[str]) -> bool:
    raw = os.environ.get("ROR_VLLM_TEXT_PROMPTS_AS_CHAT")
    if raw is not None and raw.strip() != "":
        return _parse_bool(raw, default=False)

    # Auto mode: use chat templates for instruction-tuned models, keep known base models raw.
    return not _is_base_model(model_name)


def _is_qwen3_model(model_name: Optional[str]) -> bool:
    normalized = (model_name or "").strip().lower()
    return "qwen3" in normalized


def _resolve_chat_enable_thinking(model_name: Optional[str]) -> Optional[bool]:
    raw = os.environ.get("ROR_VLLM_ENABLE_THINKING")
    if raw is not None and raw.strip() != "":
        return _parse_bool(raw, default=False)

    # Default Qwen3 chat-template rendering to non-thinking mode.
    if _is_qwen3_model(model_name):
        return False
    return None


def _normalize_prompt_input(prompt: PromptInput, engine, model_name: Optional[str] = None) -> str:
    messages: Optional[list[Mapping[str, Any]]] = None

    if _is_chat_messages(prompt):
        messages = list(prompt)
    else:
        prompt_text = _coerce_to_str(prompt)
        if not _resolve_text_prompts_as_chat(model_name=model_name):
            return prompt_text
        messages = [{"role": "user", "content": prompt_text}]
        _warn_once(
            f"text_prompts_wrapped_as_chat::{(model_name or '').strip().lower()}",
            f"[vLLM] Wrapping text prompts with default chat template for model={model_name or 'unknown'}. "
            "Set ROR_VLLM_TEXT_PROMPTS_AS_CHAT=false to disable.",
        )

    add_generation_prompt = _parse_chat_add_generation_prompt(default=True)
    enable_thinking = _resolve_chat_enable_thinking(model_name=model_name)
    if enable_thinking is False and _is_qwen3_model(model_name):
        _warn_once(
            "qwen3_non_thinking_default",
            "[vLLM] Qwen3 chat prompts default to non-thinking mode "
            "(enable_thinking=False). Set ROR_VLLM_ENABLE_THINKING=true to override.",
        )

    tokenizer = _resolve_tokenizer_for_chat(engine)
    if tokenizer is not None:
        try:
            kwargs = {}
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            if isinstance(rendered, str) and rendered:
                return rendered
        except Exception as exc:
            _warn_once(
                "chat_template_failure",
                f"[vLLM] Chat template rendering failed; using fallback prompt formatting: {exc}",
            )
    else:
        _warn_once(
            "chat_template_missing",
            "[vLLM] Tokenizer has no chat template support; using fallback prompt formatting.",
        )

    return _fallback_chat_to_text(messages, add_generation_prompt=add_generation_prompt)


def _build_sampling_params(
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: Optional[Sequence[str]] = None,
):
    from vllm import SamplingParams

    temp = max(0.0, float(temperature))
    kwargs = {
        "max_tokens": int(max_tokens),
        "temperature": temp,
        "top_p": float(top_p),
    }
    stop_sequences = _parse_stop_sequences(stop=stop)
    if stop_sequences:
        kwargs["stop"] = stop_sequences
    return SamplingParams(
        **kwargs,
    )


def generate_batch_responses(
    prompts: Sequence[PromptInput],
    model_path: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    engine_config: Optional[VLLMEngineConfig] = None,
    stop: Optional[Sequence[str]] = None,
) -> list[str]:
    if not prompts:
        return []

    cfg = engine_config or resolve_vllm_config()
    engine = _get_engine(model_path=model_path, model_name=model_name, config=cfg)
    normalized_prompts = [
        _normalize_prompt_input(prompt, engine=engine, model_name=model_name)
        for prompt in prompts
    ]
    sampling_params = _build_sampling_params(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )

    outputs = engine.generate(normalized_prompts, sampling_params=sampling_params, use_tqdm=False)
    responses: list[str] = []
    for output in outputs:
        if not output.outputs:
            responses.append("")
            continue
        text = output.outputs[0].text or ""
        responses.append(text.strip())
    return responses


def generate_response(
    prompt: PromptInput,
    model_path: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    engine_config: Optional[VLLMEngineConfig] = None,
    stop: Optional[Sequence[str]] = None,
) -> str:
    responses = generate_batch_responses(
        prompts=[prompt],
        model_path=model_path,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        engine_config=engine_config,
        stop=stop,
    )
    return responses[0] if responses else ""


async def async_generate_response(
    prompt: PromptInput,
    model_path: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    engine_config: Optional[VLLMEngineConfig] = None,
    stop: Optional[Sequence[str]] = None,
) -> str:
    return await asyncio.to_thread(
        generate_response,
        prompt=prompt,
        model_path=model_path,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        engine_config=engine_config,
        stop=stop,
    )


async def async_generate_batch_responses(
    prompts: Sequence[PromptInput],
    model_path: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    engine_config: Optional[VLLMEngineConfig] = None,
    stop: Optional[Sequence[str]] = None,
) -> list[str]:
    return await asyncio.to_thread(
        generate_batch_responses,
        prompts=list(prompts),
        model_path=model_path,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        engine_config=engine_config,
        stop=stop,
    )


def unload_engine() -> None:
    global _engine, _engine_model_name, _engine_model_path, _engine_config

    with _engine_lock:
        if _engine is None:
            return
        print(f"[vLLM] Unloading engine: {_engine_model_name}")
        _engine = None
        _engine_model_name = None
        _engine_model_path = None
        _engine_config = None
        gc.collect()
        _clear_cuda_cache()
