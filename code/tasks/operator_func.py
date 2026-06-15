# code/tasks/operator_func.py

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from utils.cluster.cluster_model_interface import query_prompts_batched, get_vllm_batch_size
from utils.dataset_io import get_data_root, load_records
from utils.cluster.checkpointing import (
    is_resume_enabled,
    get_checkpoint_every,
    get_snapshot_every,
    load_checkpoint,
    save_checkpoint,
    save_result_snapshot,
    clear_checkpoint,
)
from prompts.operator_func_prompts import (
    get_rule_based_prompt,
    get_example_based_prompt,
    get_combined_prompt,
)

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_ROOT = get_data_root(WORKSPACE_ROOT)
OPERATOR_FUNCTION_DATA_DIR = os.path.join(DATA_ROOT, "operator_function")
OPERATOR_FUNC_DATA_DIR = os.path.join(DATA_ROOT, "op_func")
DIFFICULTY_NAME = {1: "easy", 2: "medium", 3: "hard"}

def operator_func_base(input):
    """ mul -> plus """
    x,y,z = input
    output = (x * y) + z
    return output

def operator_func_lv2(input):
    """ minus -> mul -> plus """
    x,y,z,a = input
    output = ((x-y)*z) + a
    return output

def operator_func_lv3(input):
    """ plus -> mul -> minus -> div """
    x,y,z,a,b = input
    output = (((x+y)*z)-a)/b
    return output

def extract_outcome(text: str) -> Any:
    """Extract the predicted answer from model response."""
    if not isinstance(text, str):
        return "unknown"

    norm_text = re.sub(r"\s+", " ", text.strip())

    boxed_match = re.search(r"\\boxed\{(.+?)\}", norm_text)
    if not boxed_match:
        return "Not boxed"

    boxed_content = boxed_match.group(1).strip()

    try:
        return int(boxed_content)
    except ValueError:
        pass

    return boxed_content if boxed_content else "unknown"


def check_outcome_validity(cleaned_response: Any, answer: Any) -> Tuple[bool, str]:
    """Check if the model's response is valid and correct."""
    if cleaned_response in ("unknown", "Not boxed"):
        return False, "Problem with raw response"
    if cleaned_response == answer:
        return True, "Correct"
    return False, "The answer is not correct"


def examples_sampling(train_samples: List[Any], num_examples: int, rng: random.Random) -> List[Any]:
    """Sample num_examples examples from training data."""
    if not train_samples:
        return []
    if len(train_samples) <= num_examples:
        print(f"Warning: Requested {num_examples} examples but only {len(train_samples)} available")
        return list(train_samples)
    return rng.sample(train_samples, num_examples)


def load_operator_func_split(difficulty: int, split: str) -> List[Any]:
    """Load operator func data from a specific split."""
    candidates = [
        os.path.join(
            OPERATOR_FUNCTION_DATA_DIR,
            DIFFICULTY_NAME[difficulty],
            f"{split}.jsonl",
        ),
        os.path.join(
            OPERATOR_FUNCTION_DATA_DIR,
            split,
            f"operator_function_{split}_{difficulty}.json",
        ),
        os.path.join(OPERATOR_FUNC_DATA_DIR, split, f"op_func_{split}_{difficulty}.json"),
    ]

    path = next((candidate for candidate in candidates if os.path.exists(candidate)), None)
    if path is None:
        raise FileNotFoundError(
            "Operator Function data file not found. Tried: "
            + ", ".join(candidates)
        )

    records = load_records(path)

    print(f"Loaded {len(records)} {split} cases from {path}")
    return records


def load_operator_func_data(
    difficulty: int,
    num_test_samples: int,
    num_examples: int,
    mode: str,
    seed: int,
) -> Tuple[List[Any], List[Any]]:
    """Load operator func data with train and test splits."""
    rng = random.Random(seed)

    train_examples: List[Any] = []
    if mode in {"examples", "combined"} and num_examples > 0:
        train_examples = load_operator_func_split(difficulty, "train")

    test_pool = load_operator_func_split(difficulty, "test")
    if not test_pool:
        raise ValueError(f"No operator func test examples for difficulty {difficulty}.")

    if num_test_samples >= len(test_pool):
        selected = list(test_pool)
        rng.shuffle(selected)
    else:
        selected = rng.sample(test_pool, num_test_samples)

    return train_examples, selected

async def run_operator_func_experiment(
    difficulty: int,
    mode: str,
    num_test_samples: int,
    output_dir: str,
    model_name: str,
    num_demos: int = 1,
    seed: int = 123,
    max_tokens: int = 50,
    query_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    query_batch_fn: Optional[Callable[[List[str]], Awaitable[List[str]]]] = None,
    query_batch_size: Optional[int] = None,
    max_concurrent: int = 1,
) -> Dict[str, Any]:
    """Run the operator function experiment."""
    if difficulty not in {1, 2, 3}:
        raise ValueError(f"Unsupported difficulty '{difficulty}'. Choose from 1, 2, 3.")
    if mode not in {"rules", "examples", "combined"}:
        raise ValueError(f"Unsupported mode '{mode}'. Choose from rules/examples/combined.")

    rng = random.Random(seed)

    train_examples, selected_tests = load_operator_func_data(
        difficulty=difficulty,
        num_test_samples=num_test_samples,
        num_examples=num_demos,
        mode=mode,
        seed=seed,
    )

    safe_model_name = model_name.replace("/", "_")
    task_output_dir = os.path.join(output_dir, "operator_func", f"difficulty_{difficulty}", mode, safe_model_name)
    os.makedirs(task_output_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        task_output_dir, f"operator_func_{difficulty}_{mode}_{safe_model_name}_checkpoint.json"
    )

    resume_enabled = is_resume_enabled()
    checkpoint_every = get_checkpoint_every()
    snapshot_every = get_snapshot_every(checkpoint_every)
    snapshot_path = os.path.join(
        task_output_dir, f"operator_func_{difficulty}_{mode}_{safe_model_name}_latest.json"
    )

    run_signature = {
        "task": "operator_func",
        "difficulty": difficulty,
        "mode": mode,
        "model_name": model_name,
        "num_test_samples": len(selected_tests),
        "num_examples": num_demos,
        "seed": seed,
        "max_tokens": max_tokens,
        "query_batch_size": (
            max(1, int(query_batch_size))
            if query_batch_fn is not None and query_batch_size is not None
            else None
        ),
    }

    requests: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    start_index = 0

    checkpoint_state = load_checkpoint(checkpoint_path) if resume_enabled else None
    if checkpoint_state and checkpoint_state.get("run_signature") == run_signature:
        requests = list(checkpoint_state.get("requests", []))
        results = list(checkpoint_state.get("results", []))
        start_index = int(checkpoint_state.get("next_index", len(results)))
        start_index = max(0, min(start_index, len(requests)))
        print(
            f"[checkpoint] Resuming Operator Func from {start_index}/{len(requests)} "
            f"using {checkpoint_path}"
        )
    else:
        for idx, sample in enumerate(selected_tests):
            answer = sample["answer"]
            input_val = sample["input"]

            demos: List[Any] = []
            if mode in {"examples", "combined"}:
                demos = examples_sampling(train_examples, num_demos, rng)

            if mode == "rules":
                prompt = get_rule_based_prompt(difficulty, input_val)
            elif mode == "examples":
                prompt = get_example_based_prompt(difficulty, input_val, demos)
            else:
                prompt = get_combined_prompt(difficulty, input_val, demos)

            requests.append(
                {
                    "sample_id": idx,
                    "problem": "",
                    "input": input_val,
                    "expected": answer,
                    "demos": demos,
                    "prompt": prompt,
                }
            )

        if resume_enabled:
            save_checkpoint(
                checkpoint_path,
                {
                    "run_signature": run_signature,
                    "requests": requests,
                    "results": results,
                    "next_index": 0,
                },
            )

    use_tqdm = tqdm is not None
    pbar = (
        tqdm(total=len(requests), initial=start_index, desc=f"operator_func {mode}", unit="sample")
        if use_tqdm
        else None
    )

    def _finalize_one(idx: int, request: Dict[str, Any], raw_response: str) -> None:
        predicted = extract_outcome(raw_response)
        validity, validation_message = check_outcome_validity(predicted, request["expected"])

        result_entry = {
            "sample_id": request["sample_id"],
            "difficulty": difficulty,
            "problem": request["problem"],
            "input": request["input"],
            "expected_result": request["expected"],
            "predicted_result": predicted,
            "prompt": request["prompt"],
            "raw_response": raw_response,
            "demos_used": request["demos"] if mode in {"examples", "combined"} else [],
            "is_correct": validity,
            "validation_message": validation_message,
        }
        results.append(result_entry)

        message = f"[{idx}/{len(requests)}] {validation_message}"
        if use_tqdm and pbar is not None:
            tqdm.write(message)
            pbar.update(1)
        else:
            print(message)

    def _persist_progress() -> None:
        processed = len(results)
        if processed % snapshot_every != 0 and processed < len(requests):
            return

        correct_count = sum(1 for r in results if r.get("is_correct", False))
        accuracy = correct_count / processed if processed > 0 else 0

        snapshot_payload = {
            "task": "operator_func",
            "difficulty": difficulty,
            "mode": mode,
            "model": model_name,
            "num_test_samples": len(requests),
            "num_examples": num_demos if mode in {"examples", "combined"} else 0,
            "status": "in_progress",
            "processed": processed,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "results": results,
        }
        save_result_snapshot(snapshot_path, snapshot_payload)

        if resume_enabled:
            save_checkpoint(
                checkpoint_path,
                {
                    "run_signature": run_signature,
                    "requests": requests,
                    "results": results,
                    "next_index": processed,
                },
            )

    if query_fn is None and query_batch_fn is None:
        batch_size = get_vllm_batch_size()
        for chunk_start in range(start_index, len(requests), batch_size):
            chunk_end = min(chunk_start + batch_size, len(requests))
            chunk_requests = requests[chunk_start:chunk_end]
            chunk_prompts = [r["prompt"] for r in chunk_requests]
            chunk_responses = await query_prompts_batched(
                prompts=chunk_prompts,
                model_name=model_name,
                max_tokens=max_tokens,
                batch_size=len(chunk_requests),
                max_retries=3,
                retry_delay=5,
            )
            for offset, (request, raw_response) in enumerate(zip(chunk_requests, chunk_responses), start=1):
                idx = chunk_start + offset
                _finalize_one(idx, request, raw_response)
            _persist_progress()

    elif query_batch_fn is not None:
        resolved_batch_size = max(1, int(query_batch_size or 1))
        for chunk_start in range(start_index, len(requests), resolved_batch_size):
            chunk_end = min(chunk_start + resolved_batch_size, len(requests))
            chunk_requests = requests[chunk_start:chunk_end]
            chunk_prompts = [r["prompt"] for r in chunk_requests]

            try:
                chunk_responses = await query_batch_fn(chunk_prompts)
                if len(chunk_responses) != len(chunk_requests):
                    raise RuntimeError(
                        f"Batched query size mismatch (got {len(chunk_responses)} "
                        f"for {len(chunk_requests)} prompts)."
                    )
            except Exception as batch_exc:
                print(f"[query_batch_fn fallback] chunk {chunk_start}:{chunk_end} failed: {batch_exc}")
                chunk_responses = []
                for request in chunk_requests:
                    if query_fn is None:
                        chunk_responses.append(str(batch_exc))
                        continue
                    try:
                        raw_response = await query_fn(request["prompt"])
                    except Exception as single_exc:
                        raw_response = str(single_exc)
                    chunk_responses.append(raw_response)

            for offset, (request, raw_response) in enumerate(zip(chunk_requests, chunk_responses), start=1):
                idx = chunk_start + offset
                _finalize_one(idx, request, raw_response)
            _persist_progress()

    else:
        if max_concurrent > 1:
            async def _query_one(prompt: str) -> str:
                try:
                    return await query_fn(prompt)
                except Exception as exc:
                    return str(exc)

            for chunk_start in range(start_index, len(requests), max_concurrent):
                chunk_end = min(chunk_start + max_concurrent, len(requests))
                chunk_requests = requests[chunk_start:chunk_end]
                chunk_responses = await asyncio.gather(
                    *[_query_one(r["prompt"]) for r in chunk_requests]
                )
                for offset, (request, raw_response) in enumerate(
                    zip(chunk_requests, chunk_responses), start=1,
                ):
                    _finalize_one(chunk_start + offset, request, raw_response)
                _persist_progress()
        else:
            for idx, request in enumerate(requests[start_index:], start=start_index + 1):
                try:
                    raw_response = await query_fn(request["prompt"])
                except Exception as exc:
                    raw_response = str(exc)
                _finalize_one(idx, request, raw_response)
                _persist_progress()

    if pbar is not None:
        pbar.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"operator_func_{difficulty}_{mode}_{safe_model_name}_{timestamp}.json"
    output_path = os.path.join(task_output_dir, filename)

    correct_count = sum(1 for r in results if r.get("is_correct", False))
    accuracy = correct_count / len(results) if results else 0

    payload = {
        "task": "operator_func",
        "difficulty": difficulty,
        "mode": mode,
        "model": model_name,
        "num_test_samples": len(selected_tests),
        "num_examples": num_demos if mode in {"examples", "combined"} else 0,
        "timestamp": timestamp,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": len(results),
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    save_result_snapshot(
        snapshot_path,
        {
            **payload,
            "status": "completed",
            "processed": len(results),
        },
    )

    if resume_enabled:
        clear_checkpoint(checkpoint_path)

    print("\n=== Operator Function Experiment Summary ===")
    print(f" Model: {model_name}")
    print(f" Difficulty: {difficulty}")
    print(f" Mode: {mode}")
    print(f" Accuracy: {correct_count}/{len(results)} = {accuracy:.2%}")
    print(f" Results saved to: {output_path}")

    error_counts: Dict[str, int] = {}
    for r in results:
        if not r.get("is_correct", False):
            msg = r.get("validation_message", "Unknown error")
            error_counts[msg] = error_counts.get(msg, 0) + 1

    if error_counts:
        print("\n Error breakdown:")
        for msg, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {msg}: {count}")

    print("============================================\n")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Operator Function experiments.")
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--mode", type=str, default="rules", choices=["rules", "examples", "combined"])
    parser.add_argument("--num-test-samples", type=int, default=50)
    parser.add_argument("--num-demos", type=int, default=1)
    parser.add_argument("--model", type=str, default="qwen3-14b")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=os.path.join(WORKSPACE_ROOT, "experiment_outputs"))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    asyncio.run(
        run_operator_func_experiment(
            difficulty=args.difficulty,
            mode=args.mode,
            num_test_samples=args.num_test_samples,
            num_demos=args.num_demos,
            output_dir=args.output_dir,
            model_name=args.model,
            seed=args.seed,
            max_tokens=args.max_tokens,
        )
    )
