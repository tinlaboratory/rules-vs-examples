"""Experiment entrypoint for the final Rules-vs-Examples task suite."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import List, Optional

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from utils.cluster.cache_env import configure_cluster_cache_env
from utils.cluster.cluster_model_interface import get_available_local_models, list_cluster_models

FINAL_TASKS = {
    "set_game",
    "operator_func",
    "tapatan",
    "noun_class_agreement",
    "lexical_category_inference",
}

DEFAULT_MODEL = "qwen3-14b"
DEFAULT_NUM_TEST_SAMPLES = {
    "set_game": 100,
    "operator_func": 100,
    "tapatan": 100,
    "noun_class_agreement": 100,
    "lexical_category_inference": 600,
}
DEFAULT_MAX_TOKENS = {
    "set_game": 50,
    "operator_func": 50,
    "tapatan": 4,
    "noun_class_agreement": 8,
    "lexical_category_inference": 2,
}


def _parse_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_stop_sequences(stop_json: Optional[str], stop_values: Optional[List[str]]) -> List[str]:
    stops: List[str] = []
    if stop_json:
        parsed = json.loads(stop_json)
        if isinstance(parsed, str):
            stops.append(parsed)
        elif isinstance(parsed, list):
            stops.extend(str(item) for item in parsed if item is not None and str(item) != "")
        else:
            raise ValueError("--vllm_stop_json must be a JSON string or list of strings.")
    if stop_values:
        stops.extend(str(item) for item in stop_values if item is not None and str(item) != "")
    return stops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final task experiments with vLLM or an OpenAI-compatible API.")
    parser.add_argument(
        "--task",
        required=False,
        default="tapatan",
        choices=sorted(FINAL_TASKS | {"all_final_tasks"}),
        help="Final-paper task to run.",
    )
    parser.add_argument("--mode", default="rules", choices=["rules", "examples", "combined", "all"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num_test_samples", "--num-test-samples", type=int, default=None)
    parser.add_argument("--num_demos", "--num-demos", type=int, default=None)
    parser.add_argument("--max_tokens", "--max-tokens", type=int, default=None)
    parser.add_argument("--output_dir", "--output-dir", default="experiment_outputs")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--difficulty", default="1", help="Difficulty for set_game/operator_func.")
    parser.add_argument(
        "--tapatan_difficulty",
        "--tapatan-difficulty",
        default="easy",
        choices=["1", "2", "3", "easy", "medium", "hard", "d1", "d2", "d3"],
    )
    parser.add_argument(
        "--tapatan_input_format",
        "--tapatan-input-format",
        default="move_sequence",
        choices=["move_sequence", "final_board_state", "moves", "board"],
    )
    parser.add_argument(
        "--noun_class_difficulty",
        "--noun-class-difficulty",
        default="1",
        choices=["1", "2", "3", "easy", "medium", "hard"],
    )
    parser.add_argument(
        "--lexical_category_inference_difficulty",
        "--lexical-category-inference-difficulty",
        default="all",
        choices=["all", "1", "2", "3", "easy", "medium", "hard", "d1", "d2", "d3"],
    )
    parser.add_argument(
        "--lexical_category_inference_logic_condition",
        "--lexical-category-inference-logic-condition",
        default="all",
        choices=["shared", "either", "both", "all"],
    )

    parser.add_argument("--inference_backend", "--inference-backend", default="vllm", choices=["vllm", "api", "openai"])
    parser.add_argument("--api_base_url", "--api-base-url", default=None)
    parser.add_argument("--api_key_env", "--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api_max_concurrent", "--api-max-concurrent", type=int, default=8)
    parser.add_argument("--api_temperature", "--api-temperature", type=float, default=0.1)
    parser.add_argument("--api_top_p", "--api-top-p", type=float, default=0.9)
    parser.add_argument("--api_extra_body_json", "--api-extra-body-json", default=None)
    parser.add_argument("--api_extra_headers_json", "--api-extra-headers-json", default=None)
    parser.add_argument(
        "--api_token_param",
        "--api-token-param",
        default="max_tokens",
        choices=["max_tokens", "max_completion_tokens", "none"],
        help="Token-limit parameter to send to the API. Use 'none' to omit token limits.",
    )
    parser.add_argument("--vllm_tensor_parallel_size", "--vllm-tensor-parallel-size", type=int, default=None)
    parser.add_argument(
        "--vllm_dtype",
        "--vllm-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--vllm_gpu_memory_utilization", "--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", "--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--vllm_enforce_eager", "--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm_quantization", "--vllm-quantization", default=None)
    parser.add_argument("--vllm_swap_space", "--vllm-swap-space", type=float, default=4.0)
    parser.add_argument("--vllm_cpu_offload_gb", "--vllm-cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--vllm_batch_size", "--vllm-batch-size", type=int, default=16)
    parser.add_argument("--vllm_stop", "--vllm-stop", action="append", default=None)
    parser.add_argument("--vllm_stop_json", "--vllm-stop-json", default=None)
    parser.add_argument("--num_gpus", "--num-gpus", type=int, default=None)

    parser.add_argument("--cluster_cache_root", "--cluster-cache-root", default=None)
    parser.add_argument("--cluster_use_tmpdir_cache", "--cluster-use-tmpdir-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_every", "--checkpoint-every", type=int, default=50)
    parser.add_argument("--list_models", "--list-models", action="store_true")
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    cache_root = args.cluster_cache_root or os.environ.get("ROR_CLUSTER_CACHE_ROOT")
    use_tmpdir_cache = args.cluster_use_tmpdir_cache or _parse_env_bool("ROR_CLUSTER_USE_TMPDIR_CACHE", False)
    configured_cache_env = configure_cluster_cache_env(
        cache_root=cache_root,
        use_tmpdir_if_available=use_tmpdir_cache,
    )
    if configured_cache_env:
        print(f"[cluster cache] Using cache root: {configured_cache_env['ROR_CLUSTER_CACHE_ROOT']}")

    inference_backend = "api" if args.inference_backend == "openai" else args.inference_backend
    os.environ["ROR_INFERENCE_BACKEND"] = inference_backend
    os.environ["ROR_RESUME"] = "true" if args.resume else "false"
    os.environ["ROR_CHECKPOINT_EVERY"] = str(max(1, int(args.checkpoint_every)))

    if inference_backend == "api":
        os.environ["ROR_API_KEY_ENV"] = args.api_key_env
        os.environ["ROR_API_MAX_CONCURRENT"] = str(max(1, int(args.api_max_concurrent)))
        os.environ["ROR_API_TEMPERATURE"] = str(args.api_temperature)
        os.environ["ROR_API_TOP_P"] = str(args.api_top_p)
        os.environ["ROR_API_TOKEN_PARAM"] = args.api_token_param
        if args.api_base_url:
            os.environ["ROR_API_BASE_URL"] = args.api_base_url
        else:
            os.environ.pop("ROR_API_BASE_URL", None)
        if args.api_extra_body_json:
            json.loads(args.api_extra_body_json)
            os.environ["ROR_API_EXTRA_BODY_JSON"] = args.api_extra_body_json
        else:
            os.environ.pop("ROR_API_EXTRA_BODY_JSON", None)
        if args.api_extra_headers_json:
            json.loads(args.api_extra_headers_json)
            os.environ["ROR_API_EXTRA_HEADERS_JSON"] = args.api_extra_headers_json
        else:
            os.environ.pop("ROR_API_EXTRA_HEADERS_JSON", None)
        return

    os.environ["ROR_VLLM_TENSOR_PARALLEL_SIZE"] = str(args.vllm_tensor_parallel_size or args.num_gpus or 1)
    os.environ["ROR_VLLM_DTYPE"] = args.vllm_dtype
    os.environ["ROR_VLLM_GPU_MEMORY_UTILIZATION"] = str(args.vllm_gpu_memory_utilization)
    os.environ["ROR_VLLM_ENFORCE_EAGER"] = "true" if args.vllm_enforce_eager else "false"
    os.environ["ROR_VLLM_SWAP_SPACE"] = str(args.vllm_swap_space)
    os.environ["ROR_VLLM_CPU_OFFLOAD_GB"] = str(args.vllm_cpu_offload_gb)
    os.environ["ROR_VLLM_BATCH_SIZE"] = str(max(1, int(args.vllm_batch_size)))
    os.environ["ROR_VLLM_TRUST_REMOTE_CODE"] = "true"

    if args.vllm_max_model_len is not None:
        os.environ["ROR_VLLM_MAX_MODEL_LEN"] = str(args.vllm_max_model_len)
    else:
        os.environ.pop("ROR_VLLM_MAX_MODEL_LEN", None)
    if args.vllm_quantization:
        os.environ["ROR_VLLM_QUANTIZATION"] = args.vllm_quantization
    else:
        os.environ.pop("ROR_VLLM_QUANTIZATION", None)
    if args.vllm_stop_json or args.vllm_stop:
        os.environ["ROR_VLLM_STOP"] = json.dumps(_resolve_stop_sequences(args.vllm_stop_json, args.vllm_stop))
    else:
        os.environ.pop("ROR_VLLM_STOP", None)


def _modes(args: argparse.Namespace) -> List[str]:
    return ["rules", "examples", "combined"] if args.mode == "all" else [args.mode]


async def run_one_task(task: str, args: argparse.Namespace) -> None:
    num_test_samples = DEFAULT_NUM_TEST_SAMPLES[task] if args.num_test_samples is None else args.num_test_samples
    max_tokens = DEFAULT_MAX_TOKENS[task] if args.max_tokens is None else args.max_tokens
    if args.inference_backend in {"api", "openai"} and args.api_token_param == "none":
        max_tokens = None

    for mode in _modes(args):
        if task == "set_game":
            from tasks.set_game import run_set_game_experiment

            await run_set_game_experiment(
                difficulty=int(args.difficulty),
                mode=mode,
                num_test_samples=num_test_samples,
                num_demos=args.num_demos if args.num_demos is not None else 20,
                output_dir=args.output_dir,
                model_name=args.model,
                seed=args.seed,
                max_tokens=max_tokens,
            )
        elif task == "operator_func":
            from tasks.operator_func import run_operator_func_experiment

            await run_operator_func_experiment(
                difficulty=int(args.difficulty),
                mode=mode,
                num_test_samples=num_test_samples,
                num_demos=args.num_demos if args.num_demos is not None else 1,
                output_dir=args.output_dir,
                model_name=args.model,
                seed=args.seed,
                max_tokens=max_tokens,
            )
        elif task == "tapatan":
            from tasks.tapatan import run_tapatan_experiment

            await run_tapatan_experiment(
                difficulty=args.tapatan_difficulty,
                mode=mode,
                num_test_samples=num_test_samples,
                num_demos=args.num_demos,
                output_dir=args.output_dir,
                model_name=args.model,
                seed=args.seed,
                max_tokens=max_tokens,
                input_format=args.tapatan_input_format,
            )
        elif task == "noun_class_agreement":
            from tasks.noun_class_agreement import run_noun_class_agreement_experiment

            await run_noun_class_agreement_experiment(
                difficulty=args.noun_class_difficulty,
                mode=mode,
                num_test_samples=num_test_samples,
                num_demos=args.num_demos,
                output_dir=args.output_dir,
                model_name=args.model,
                seed=args.seed,
                max_tokens=max_tokens,
            )
        elif task == "lexical_category_inference":
            from tasks.lexical_category_inference import get_design_cell_plan, run_lexical_category_inference_experiment

            cell_plan = get_design_cell_plan(
                args.lexical_category_inference_difficulty,
                args.lexical_category_inference_logic_condition,
            )
            for index, (logic, level, _) in enumerate(cell_plan, start=1):
                print(f"--- [{index}/{len(cell_plan)}] lexical_category_inference cell={logic}/{level} ---")
                await run_lexical_category_inference_experiment(
                    difficulty=level,
                    logic_condition=logic,
                    mode=mode,
                    num_test_samples=num_test_samples,
                    num_demos=args.num_demos,
                    output_dir=args.output_dir,
                    model_name=args.model,
                    seed=args.seed,
                    max_tokens=max_tokens,
                )
        else:
            raise ValueError(f"Unknown task: {task}")


async def main() -> None:
    args = parse_args()
    if args.list_models:
        list_cluster_models()
        return

    inference_backend = "api" if args.inference_backend == "openai" else args.inference_backend
    available = get_available_local_models()
    if inference_backend == "vllm" and available and args.model not in available:
        print(f"Warning: model '{args.model}' is registered but not present under ROR_MODEL_ROOT.")
        print(f"Detected local models: {', '.join(sorted(available))}")

    configure_runtime(args)

    tasks = ["set_game", "operator_func", "tapatan", "noun_class_agreement", "lexical_category_inference"]
    if args.task != "all_final_tasks":
        tasks = [args.task]
    for task in tasks:
        print(f"=== Running {task} with {args.model} ===")
        await run_one_task(task, args)


if __name__ == "__main__":
    asyncio.run(main())
