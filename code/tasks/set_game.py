from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from datetime import datetime
from glob import glob
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from utils.cluster.cluster_model_interface import query_prompts_batched, get_vllm_batch_size
from utils.dataset_io import get_data_root, load_records, select_latest_file
from utils.cluster.checkpointing import (
    is_resume_enabled,
    get_checkpoint_every,
    get_snapshot_every,
    load_checkpoint,
    save_checkpoint,
    save_result_snapshot,
    clear_checkpoint,
)

from prompts.set_game_prompts import (
    get_rule_based_prompt,
    get_example_based_prompt,
    get_combined_prompt,
)

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_ROOT = get_data_root(WORKSPACE_ROOT)
SET_GAME_DATA_DIR = os.path.join(DATA_ROOT, "set_game")
DIFFICULTY_NAME = {1: "easy", 2: "medium", 3: "hard"}


def all_same(vals):
    return len(set(vals)) == 1


def all_diff(vals):
    return len(set(vals)) == 3


def is_set_base(cards):
    """Difficulty 1: animal, biome"""
    animal, biome = zip(*cards)
    for attr in [animal, biome]:
        if not (all_same(attr) or all_diff(attr)):
            return False
    return True


def is_set_lv2(cards):
    """Difficulty 2: animal, biome, food"""
    animal, biome, food = zip(*cards)
    for attr in [animal, biome, food]:
        if not (all_same(attr) or all_diff(attr)):
            return False
    return True


def is_set_lv3(cards):
    """Difficulty 3: number, animal, biome, food"""
    number, animal, biome, food = zip(*cards)
    
    # number: exactly 2 unique values (not all_same, not all_diff)
    if len(set(number)) != 2:
        return False

    # remaining attributes: each must be all_same or all_diff
    for attr in [animal, biome, food]:
        if not (all_same(attr) or all_diff(attr)):
            return False
    
    return True


def extract_outcome(text: str, difficulty: int) -> Any:
    """Extract the predicted set from model response."""
    if not isinstance(text, str):
        return "unknown"
    
    #also parse the case like
    #\boxed{First card: rabbit|plain\nSecond card: wolf|desert\nThird card: horse|hill}
    #First card: rabbit|plain\nSecond card: wolf|desert\nThird card: horse|hill
    
    norm_text = re.sub(r'\s+', ' ', text.strip())
    expected_parts = {1: 2, 2: 3, 3: 4}
    if difficulty not in expected_parts:
        return "unknown"
    
    boxed_match = re.search(r"\\boxed\{(.+?)\}", norm_text)
    if not boxed_match:
        return "Not boxed"
    
    boxed_content = boxed_match.group(1)
    
    matches = re.findall(r"\(([^()]+)\)", boxed_content)
    if len(matches) != 3:
        return "unknown"

    cleaned_cards = []

    for m in matches:
        parts = [p.strip() for p in m.split("|")]

        if len(parts) != expected_parts[difficulty]:
            return "unknown"

        try:
            if difficulty == 1:
                animal, biome = parts
                cleaned_cards.append((animal, biome))

            elif difficulty == 2:
                animal, biome, food = parts
                cleaned_cards.append((animal, biome, food))

            elif difficulty == 3:
                number = int(parts[0])
                animal, biome, food = parts[1:]
                cleaned_cards.append((number, animal, biome, food))

        except (ValueError, IndexError):
            return "unknown"

    return cleaned_cards


def check_outcome_validity(cleaned_response: Any, difficulty: int, problem: List, answer: List) -> Tuple[bool, str]:
    """Check if the model's response is valid and correct."""
    if cleaned_response == "unknown":
        return False, "Problem with raw response"
    
    if len(cleaned_response) != 3:
        return False, "Model did not return exactly 3 cards"
    
    # Convert problem cards to tuples for comparison
    problem_tuples = []
    for card in problem:
        if isinstance(card, (list, tuple)):
            if difficulty == 1:
                card_tuple = (card[0], card[1])
            elif difficulty == 2:
                card_tuple = (card[0], card[1], card[2])
            elif difficulty == 3:
                card_tuple = (int(card[0]), card[1], card[2], card[3])
            problem_tuples.append(card_tuple)
    
    # Check if all predicted cards exist in the board
    c1, c2, c3 = cleaned_response
    for c in (c1, c2, c3):
        if c not in problem_tuples:
            return False, f"Card {c} does not exist in the board"
    
    cards = (c1, c2, c3)
    
    # Validate if it's a correct set
    if difficulty == 1:
        valid = is_set_base(cards)
    elif difficulty == 2:
        valid = is_set_lv2(cards)
    elif difficulty == 3:
        valid = is_set_lv3(cards)
    else:
        return False, "Invalid difficulty level"
    
    if not valid:
        return False, "Chosen cards do not form a valid set"

    return True, "Correct set"


def _select_latest_numbered_json(pattern: str) -> Optional[str]:
    """Select the latest numbered JSON file matching a pattern."""
    matches = glob(pattern)
    if not matches:
        return None

    def extract_n(path: str) -> int:
        base = os.path.basename(path)
        m = re.search(r"_(\d+)\.json$", base)
        return int(m.group(1)) if m else -1

    return max(matches, key=extract_n)


def load_set_game_split(difficulty: int, split: str) -> List[Any]:
    """Load set game data from a specific split."""
    base_dir = os.path.join(SET_GAME_DATA_DIR, split)
    hf_path = os.path.join(SET_GAME_DATA_DIR, DIFFICULTY_NAME[difficulty], f"{split}.jsonl")
    pattern = os.path.join(base_dir, f"set_game_{split}_{difficulty}.json")

    # First try exact match
    if os.path.exists(hf_path):
        return load_records(hf_path)
    if os.path.exists(pattern):
        return load_records(pattern)

    # Try with wildcard for numbered versions
    pattern = os.path.join(base_dir, f"set_game_{split}_{difficulty}_*.json")
    path = select_latest_file([pattern])
    
    if not path:
        raise FileNotFoundError(f"No Set Game data found for difficulty {difficulty}/{split} (pattern: {pattern})")

    return load_records(path)


def load_set_game_data(
    difficulty: int,
    num_test_samples: int,
    num_demos: int,
    mode: str,
    seed: int,
) -> Tuple[List[Any], List[Any]]:
    """Load set game data with train and test splits."""
    rng = random.Random(seed)

    train_examples: List[Any] = []
    if mode in {"examples", "combined"} and num_demos > 0:
        train_examples = load_set_game_split(difficulty, "train")

    test_pool = load_set_game_split(difficulty, "test")
    if not test_pool:
        raise ValueError(f"No Set Game test examples for difficulty {difficulty}.")

    if num_test_samples >= len(test_pool):
        selected = list(test_pool)
        rng.shuffle(selected)
    else:
        selected = rng.sample(test_pool, num_test_samples)

    return train_examples, selected


def sample_balanced_demos(
    train: Sequence[Any],
    k: int,
    rng: random.Random,
) -> List[Any]:
    """Sample k examples from each category."""
    if k <= 0 or not train:
        return []

    # Group by category
    category_buckets: Dict[str, List[Any]] = defaultdict(list)
    for example in train:
        cat = example.get("valid_set_category", "unknown")
        category_buckets[cat].append(example)

    demos = []
    for cat, examples in category_buckets.items():
        if k >= len(examples):
            sampled = list(examples)
        else:
            sampled = rng.sample(examples, k)
        demos.extend(sampled)

    rng.shuffle(demos)
    return demos


async def run_set_game_experiment(
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
) -> Dict[str, Any]:
    """Run the set game experiment."""
    if difficulty not in {1, 2, 3}:
        raise ValueError(f"Unsupported difficulty '{difficulty}'. Choose from 1, 2, 3.")
    if mode not in {"rules", "examples", "combined"}:
        raise ValueError(f"Unsupported mode '{mode}'. Choose from rules/examples/combined.")

    rng = random.Random(seed)

    train_examples, selected_tests = load_set_game_data(
        difficulty=difficulty,
        num_test_samples=num_test_samples,
        num_demos=num_demos,
        mode=mode,
        seed=seed,
    )

    safe_model_name = model_name.replace("/", "_")
    task_output_dir = os.path.join(output_dir, "set_game", f"difficulty_{difficulty}", mode, safe_model_name)
    os.makedirs(task_output_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(
        task_output_dir, f"set_game_{difficulty}_{mode}_{safe_model_name}_checkpoint.json"
    )
    
    resume_enabled = is_resume_enabled()
    checkpoint_every = get_checkpoint_every()
    snapshot_every = get_snapshot_every(checkpoint_every)
    snapshot_path = os.path.join(
        task_output_dir, f"set_game_{difficulty}_{mode}_{safe_model_name}_latest.json"
    )
    
    run_signature = {
        "task": "set_game",
        "difficulty": difficulty,
        "mode": mode,
        "model_name": model_name,
        "num_test_samples": len(selected_tests),
        "num_demos": num_demos,
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
            f"[checkpoint] Resuming Set Game from {start_index}/{len(requests)} "
            f"using {checkpoint_path}"
        )
    else:
        for idx, sample in enumerate(selected_tests):
            answer = sample["valid_set"]
            board = list(sample["board"])
            random.shuffle(board)
            problem = board
            
            demos: List[Any] = []
            if mode in {"examples", "combined"}:
                demos = sample_balanced_demos(train_examples, num_demos, rng)

            if mode == "rules":
                prompt = get_rule_based_prompt(difficulty, problem)
            elif mode == "examples":
                prompt = get_example_based_prompt(difficulty, problem, demos)
            else:
                prompt = get_combined_prompt(difficulty, problem, demos)

            requests.append(
                {
                    "sample_id": idx,
                    "problem": problem,
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
        tqdm(total=len(requests), initial=start_index, desc=f"set_game {mode}", unit="sample")
        if use_tqdm
        else None
    )

    def _finalize_one(idx: int, request: Dict[str, Any], raw_response: str) -> None:
        predicted = extract_outcome(raw_response, difficulty)
        validity, validation_message = check_outcome_validity(
            predicted, difficulty, request["problem"], request["expected"]
        )
        
        result_entry = {
            "sample_id": request["sample_id"],
            "difficulty": difficulty,
            "problem": request["problem"],
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
            "task": "set_game",
            "difficulty": difficulty,
            "mode": mode,
            "model": model_name,
            "num_test_samples": len(requests),
            "num_demos": num_demos if mode in {"examples", "combined"} else 0,
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
            except Exception as batch_exc:  # noqa: BLE001
                print(
                    f"[query_batch_fn fallback] chunk {chunk_start}:{chunk_end} failed: {batch_exc}"
                )
                chunk_responses = []
                for request in chunk_requests:
                    if query_fn is None:
                        chunk_responses.append(str(batch_exc))
                        continue
                    try:
                        raw_response = await query_fn(request["prompt"])
                    except Exception as single_exc:  # noqa: BLE001
                        raw_response = str(single_exc)
                    chunk_responses.append(raw_response)

            for offset, (request, raw_response) in enumerate(
                zip(chunk_requests, chunk_responses),
                start=1,
            ):
                idx = chunk_start + offset
                _finalize_one(idx, request, raw_response)
            _persist_progress()
    else:
        for idx, request in enumerate(requests[start_index:], start=start_index + 1):
            try:
                raw_response = await query_fn(request["prompt"])
            except Exception as exc:  # noqa: BLE001
                raw_response = str(exc)
            _finalize_one(idx, request, raw_response)
            _persist_progress()

    if pbar is not None:
        pbar.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"set_game_{difficulty}_{mode}_{safe_model_name}_{timestamp}.json"
    output_path = os.path.join(task_output_dir, filename)

    correct_count = sum(1 for r in results if r.get("is_correct", False))
    accuracy = correct_count / len(results) if results else 0

    payload = {
        "task": "set_game",
        "difficulty": difficulty,
        "mode": mode,
        "model": model_name,
        "num_test_samples": len(selected_tests),
        "num_demos": num_demos if mode in {"examples", "combined"} else 0,
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

    print("\n=== Set Game Experiment Summary ===")
    print(f" Model: {model_name}")
    print(f" Difficulty: {difficulty}")
    print(f" Mode: {mode}")
    print(f" Accuracy: {correct_count}/{len(results)} = {accuracy:.2%}")
    print(f" Results saved to: {output_path}")
    
    # Print breakdown of validation messages
    error_counts = {}
    for r in results:
        if not r.get("is_correct", False):
            msg = r.get("validation_message", "Unknown error")
            error_counts[msg] = error_counts.get(msg, 0) + 1
    
    if error_counts:
        print("\n Error breakdown:")
        for msg, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {msg}: {count}")
    
    print("==================================\n")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Set Game experiments.")
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--mode", type=str, default="rules", choices=["rules", "examples", "combined"])
    parser.add_argument("--num-test-samples", type=int, default=50)
    parser.add_argument("--num-demos", type=int, default=20)
    parser.add_argument("--model", type=str, default="qwen3-14b")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=os.path.join(WORKSPACE_ROOT, "experiment_outputs"))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    asyncio.run(
        run_set_game_experiment(
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
