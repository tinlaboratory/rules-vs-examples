from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from datetime import datetime
from glob import glob
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from utils.cluster.cluster_model_interface import query_prompts_batched, get_vllm_batch_size
from utils.cluster.checkpointing import (
    is_resume_enabled,
    get_checkpoint_every,
    get_snapshot_every,
    load_checkpoint,
    save_checkpoint,
    save_result_snapshot,
    clear_checkpoint,
)
from tasks.tapatan_core import (
    TapatanConfig,
    get_tapatan_config,
    parse_turn,
    split_moves,
    new_game,
    apply_turn,
    has_line,
    find_line_direction,
    max_line_length,
)
from prompts.tapatan_prompts import (
    get_rule_based_prompt,
    get_example_based_prompt,
    get_combined_prompt,
    get_board_rule_based_prompt,
    get_board_example_based_prompt,
    get_board_combined_prompt,
)
from utils.dataset_io import get_data_root, load_records, select_latest_file

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_ROOT = get_data_root(WORKSPACE_ROOT)
TAPATAN_DATA_DIR = os.path.join(DATA_ROOT, "tapatan")

VALID_OUTCOMES = ["A win", "B win", "continue"]


# Difficulty-specific demo counts derived from scenario-atom coverage:
#   Easy  (3×3, k=3): 12 demos  (4 per outcome)
#   Medium(5×5, k=4): 18 demos  (6 per outcome)
#   Hard  (7×7, k=5): 24 demos  (8 per outcome)
_DIFFICULTY_NUM_DEMOS = {"easy": 12, "medium": 18, "hard": 24}

_DIFFICULTY_ALIASES = {
    "1": "easy", "easy": "easy", "d1": "easy",
    "2": "medium", "medium": "medium", "d2": "medium",
    "3": "hard", "hard": "hard", "d3": "hard",
}


def normalize_tapatan_difficulty(difficulty: int | str) -> str:
    """Map numeric/string difficulty to canonical 'easy'/'medium'/'hard'."""
    key = str(difficulty).strip().lower()
    if key not in _DIFFICULTY_ALIASES:
        raise ValueError(
            f"Unknown tapatan difficulty '{difficulty}'. "
            f"Valid values: {sorted(_DIFFICULTY_ALIASES.keys())}"
        )
    return _DIFFICULTY_ALIASES[key]


def get_default_num_demos(difficulty: str) -> int:
    """Return the principled default demo count for *difficulty*."""
    return _DIFFICULTY_NUM_DEMOS[difficulty]



def _select_latest_numbered_json(pattern: str) -> Optional[str]:
    matches = glob(pattern)
    if not matches:
        return None

    def extract_n(path: str) -> int:
        base = os.path.basename(path)
        m = re.search(r"_n(\d+)\.json$", base)
        return int(m.group(1)) if m else -1

    return max(matches, key=extract_n)


def _normalize_tapatan_input_format(input_format: str) -> str:
    key = str(input_format or "move_sequence").strip().lower().replace("-", "_")
    aliases = {
        "moves": "move_sequence",
        "move": "move_sequence",
        "move_sequence": "move_sequence",
        "sequence": "move_sequence",
        "board": "final_board_state",
        "final_board": "final_board_state",
        "final_board_state": "final_board_state",
    }
    if key not in aliases:
        raise ValueError("tapatan input_format must be move_sequence or final_board_state.")
    return aliases[key]


def load_tapatan_split(
    difficulty: str,
    split: str,
    input_format: str = "move_sequence",
) -> List[Dict[str, Any]]:
    input_format = _normalize_tapatan_input_format(input_format)
    hf_path = os.path.join(TAPATAN_DATA_DIR, input_format, difficulty, f"{split}.jsonl")
    base_dir = os.path.join(TAPATAN_DATA_DIR, difficulty, split)
    patterns = [
        hf_path,
        os.path.join(base_dir, f"tapatan_{split}_n*.json"),
    ]
    path = select_latest_file(patterns)
    if not path:
        raise FileNotFoundError(
            f"No Tapatan data found for input_format={input_format} {difficulty}/{split}. "
            f"Tried: {', '.join(patterns)}"
        )
    return load_records(path)


def load_tapatan_data(
    difficulty: str,
    num_test_samples: int,
    num_demos: Optional[int],
    mode: str,
    seed: int,
    input_format: str = "move_sequence",
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    rng = random.Random(seed)
    input_format = _normalize_tapatan_input_format(input_format)

    train_examples: List[Dict[str, Any]] = []
    if mode in {"examples", "combined"} and (num_demos is None or num_demos > 0):
        train_examples = load_tapatan_split(difficulty, "train", input_format=input_format)

    test_pool = load_tapatan_split(difficulty, "test", input_format=input_format)
    if not test_pool:
        raise ValueError(f"No Tapatan test examples for difficulty {difficulty}.")

    if num_test_samples >= len(test_pool):
        selected = list(test_pool)
        rng.shuffle(selected)
    else:
        selected = rng.sample(test_pool, num_test_samples)

    return train_examples, selected


def _analyze_demo_atoms(
    example: Dict[str, Any], config: TapatanConfig
) -> set[str]:
    """Return a set of scenario-atom tags for *example*.

    Atoms are used by ``sample_balanced_demos`` to guarantee coverage of
    critical scenario types (diagonal wins, near-misses, phase-based wins).
    """
    atoms: set[str] = set()
    moves_str = example.get("moves") or ""
    result = example.get("result") or "unknown"

    try:
        turns = split_moves(moves_str)
        move_types = [parse_turn(t)[1] for t in turns]
    except Exception:  # noqa: BLE001
        return atoms

    has_movement = any(m == "move" for m in move_types)

    # Replay game to detect win direction and near-misses.
    state = new_game(config)
    found_near_miss = False
    for turn in turns:
        apply_turn(state, turn)
        mover = "A" if state.to_move == "B" else "B"
        loser = "B" if mover == "A" else "A"

        # Check if loser had a near-miss (k-1 in a row) at any point.
        if not found_near_miss:
            ll = max_line_length(state, loser)
            if ll == config.line_length - 1:
                found_near_miss = True
                atoms.add("near_miss")

        if has_line(state, mover):
            direction = find_line_direction(state, mover)
            if direction and direction in ((1, 1), (1, -1)):
                atoms.add("diagonal_win")
            else:
                atoms.add("non_diagonal_win")
            if has_movement:
                atoms.add("movement_win")
            else:
                atoms.add("placement_win")
            break
    else:
        # Game ended without a win -> continue.
        if has_movement:
            atoms.add("movement_continue")
        else:
            atoms.add("placement_continue")

    return atoms


# Required atoms that the demo set should try to cover.
_REQUIRED_ATOMS = frozenset(
    {
        "diagonal_win",
        "non_diagonal_win",
        "placement_win",
        "movement_win",
        "near_miss",
    }
)


def _demo_category(example: Dict[str, Any]) -> str:
    moves = example.get("moves") or ""
    result = example.get("result") or "unknown"
    try:
        turns = split_moves(moves)
        move_types = [parse_turn(turn)[1] for turn in turns]
    except Exception:  # noqa: BLE001
        return "unknown"

    move_actions = sum(1 for m in move_types if m == "move")
    if result == "continue":
        if move_actions == 0:
            return "continue_place"
        return f"continue_move_{move_actions}"

    if move_types and move_types[-1] == "move":
        pre_moves = sum(1 for m in move_types[:-1] if m == "move")
        return f"{result}_move_{pre_moves}"
    return f"{result}_place"


def sample_balanced_demos(
    train: Sequence[Dict[str, Any]],
    k: int,
    rng: random.Random,
    config: Optional[TapatanConfig] = None,
) -> List[Dict[str, Any]]:
    """Select *k* demos from *train* with atom-coverage + outcome balancing.

    Phase 0 – atom coverage (requires *config*): greedily pick examples
    that cover required scenario atoms (diagonal wins, near-misses, etc.).
    Phase 1 – category coverage: ensure all demo categories are represented.
    Phase 2 – outcome balancing: fill remaining slots keeping A win / B win /
    continue counts as even as possible.
    """
    if k <= 0 or not train:
        return []

    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()

    # --- Phase 0: atom coverage (outcome-aware) ---
    if config is not None:
        atom_map: Dict[str, set[str]] = {}
        for ex in train:
            eid = str(ex.get("id", id(ex)))
            atom_map[eid] = _analyze_demo_atoms(ex, config)

        uncovered = set(_REQUIRED_ATOMS)
        pool = list(train)
        rng.shuffle(pool)

        # Track outcomes during atom selection to break ties toward balance.
        phase0_counts = {label: 0 for label in VALID_OUTCOMES}

        while uncovered and len(selected) < k:
            best_score = 0
            candidates: List[Dict[str, Any]] = []
            for ex in pool:
                eid = str(ex.get("id", id(ex)))
                if eid in selected_ids:
                    continue
                score = len(atom_map.get(eid, set()) & uncovered)
                if score > best_score:
                    best_score = score
                    candidates = [ex]
                elif score == best_score and score > 0:
                    candidates.append(ex)
            if not candidates:
                break
            # Among equal-atom-score candidates, prefer under-represented outcome.
            min_outcome_count = min(
                phase0_counts.get(c.get("result", ""), 0) for c in candidates
            )
            balanced_candidates = [
                c for c in candidates
                if phase0_counts.get(c.get("result", ""), 0) == min_outcome_count
            ]
            chosen = rng.choice(balanced_candidates if balanced_candidates else candidates)
            selected.append(chosen)
            eid = str(chosen.get("id", id(chosen)))
            selected_ids.add(eid)
            uncovered -= atom_map.get(eid, set())
            result = chosen.get("result", "")
            if result in phase0_counts:
                phase0_counts[result] += 1

    # --- Phase 1 & 2: category coverage then outcome balancing ---
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for ex in train:
        eid = str(ex.get("id", id(ex)))
        if eid in selected_ids:
            continue
        key = _demo_category(ex)
        buckets.setdefault(key, []).append(ex)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    categories = list(buckets.keys())
    categories.sort()

    remaining = k - len(selected)

    # Phase 1: one example per category.
    if remaining > 0:
        if remaining >= len(categories):
            for category in categories:
                if buckets[category]:
                    ex = buckets[category].pop(0)
                    selected.append(ex)
                    selected_ids.add(str(ex.get("id", id(ex))))
        else:
            for category in rng.sample(categories, remaining):
                if buckets[category]:
                    ex = buckets[category].pop(0)
                    selected.append(ex)
                    selected_ids.add(str(ex.get("id", id(ex))))

    # Phase 2: outcome-balanced fill.
    remaining = k - len(selected)
    if remaining > 0:
        def _category_outcome(category: str) -> str:
            if category.startswith("A win"):
                return "A win"
            if category.startswith("B win"):
                return "B win"
            if category.startswith("continue"):
                return "continue"
            return "unknown"

        outcome_counts = {label: 0 for label in VALID_OUTCOMES}
        for ex in selected:
            label = ex.get("result")
            if label in outcome_counts:
                outcome_counts[label] += 1

        outcome_categories: Dict[str, List[str]] = {label: [] for label in VALID_OUTCOMES}
        for category, bucket in buckets.items():
            if not bucket:
                continue
            label = _category_outcome(category)
            if label in outcome_categories:
                outcome_categories[label].append(category)

        for cat_list in outcome_categories.values():
            rng.shuffle(cat_list)

        while remaining > 0:
            available = [label for label in VALID_OUTCOMES if outcome_categories[label]]
            if not available:
                break
            min_count = min(outcome_counts[label] for label in available)
            candidates = [label for label in available if outcome_counts[label] == min_count]
            label = rng.choice(candidates)
            categories_for_label = outcome_categories[label]
            if not categories_for_label:
                outcome_categories[label] = []
                continue
            category = rng.choice(categories_for_label)
            bucket = buckets.get(category, [])
            if not bucket:
                outcome_categories[label] = [c for c in categories_for_label if buckets.get(c)]
                continue
            selected.append(bucket.pop(0))
            outcome_counts[label] += 1
            remaining -= 1
            if not bucket:
                outcome_categories[label] = [c for c in categories_for_label if buckets.get(c)]

    # --- Post-processing: enforce strict outcome balance ---
    # Each outcome gets exactly k // 3 slots; remainder distributed randomly.
    base_quota = k // 3
    remainder = k % 3
    quotas = {label: base_quota for label in VALID_OUTCOMES}
    if remainder:
        for label in rng.sample(VALID_OUTCOMES, remainder):
            quotas[label] += 1

    # Group selected demos by outcome, trim each to its quota.
    by_outcome: Dict[str, List[Dict[str, Any]]] = {label: [] for label in VALID_OUTCOMES}
    for ex in selected:
        r = ex.get("result", "")
        if r in by_outcome:
            by_outcome[r].append(ex)

    used_ids: set[str] = set()
    balanced: List[Dict[str, Any]] = []
    for label in VALID_OUTCOMES:
        group = by_outcome[label][: quotas[label]]
        balanced.extend(group)
        for ex in group:
            used_ids.add(str(ex.get("id", id(ex))))

    # Fill any under-represented outcomes from unused training examples.
    for label in VALID_OUTCOMES:
        deficit = quotas[label] - len(by_outcome[label][: quotas[label]])
        if deficit <= 0:
            continue
        unused = [
            ex for ex in train
            if str(ex.get("id", id(ex))) not in used_ids and ex.get("result") == label
        ]
        rng.shuffle(unused)
        for ex in unused[:deficit]:
            balanced.append(ex)
            used_ids.add(str(ex.get("id", id(ex))))

    rng.shuffle(balanced)
    return balanced[:k]


async def run_tapatan_experiment(
    difficulty: str,
    mode: str,
    num_test_samples: int,
    num_demos: Optional[int],
    output_dir: str,
    model_name: str,
    seed: int = 123,
    max_tokens: int = 4,
    query_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    query_batch_fn: Optional[Callable[[List[str]], Awaitable[List[str]]]] = None,
    query_batch_size: Optional[int] = None,
    max_concurrent: int = 1,
    input_format: str = "move_sequence",
) -> Dict[str, Any]:
    difficulty = normalize_tapatan_difficulty(difficulty)
    input_format = _normalize_tapatan_input_format(input_format)
    if mode not in {"rules", "examples", "combined"}:
        raise ValueError(f"Unsupported mode '{mode}'. Choose from rules/examples/combined.")

    if num_demos is None:
        num_demos = get_default_num_demos(difficulty)

    rng = random.Random(seed)
    config = get_tapatan_config(difficulty)

    train_examples, selected_tests = load_tapatan_data(
        difficulty=difficulty,
        num_test_samples=num_test_samples,
        num_demos=num_demos,
        mode=mode,
        seed=seed,
        input_format=input_format,
    )

    safe_model_name = model_name.replace("/", "_")
    task_output_dir = os.path.join(output_dir, "tapatan", input_format, difficulty, mode, safe_model_name)
    os.makedirs(task_output_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        task_output_dir, f"tapatan_{input_format}_{difficulty}_{mode}_{safe_model_name}_checkpoint.json"
    )
    resume_enabled = is_resume_enabled()
    checkpoint_every = get_checkpoint_every()
    snapshot_every = get_snapshot_every(checkpoint_every)
    snapshot_path = os.path.join(
        task_output_dir, f"tapatan_{input_format}_{difficulty}_{mode}_{safe_model_name}_latest.json"
    )
    run_signature = {
        "task": "tapatan",
        "difficulty": difficulty,
        "mode": mode,
        "input_format": input_format,
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
            f"[checkpoint] Resuming Tapatan from {start_index}/{len(requests)} "
            f"using {checkpoint_path}"
        )
    else:
        for example in selected_tests:
            moves = example.get("moves", "")
            board_state = example.get("board_state") or example.get("board") or ""
            demos: List[Dict[str, Any]] = []
            if mode in {"examples", "combined"}:
                demos = sample_balanced_demos(train_examples, num_demos, rng, config=config)

            if input_format == "final_board_state":
                if mode == "rules":
                    prompt = get_board_rule_based_prompt(config, board_state)
                elif mode == "examples":
                    prompt = get_board_example_based_prompt(demos, board_state)
                else:
                    prompt = get_board_combined_prompt(config, demos, board_state)
            else:
                if mode == "rules":
                    prompt = get_rule_based_prompt(config, moves)
                elif mode == "examples":
                    prompt = get_example_based_prompt(demos, moves)
                else:
                    prompt = get_combined_prompt(config, demos, moves)

            requests.append(
                {
                    "example": example,
                    "moves": moves,
                    "board_state": board_state,
                    "expected": example.get("result", ""),
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
        tqdm(total=len(requests), initial=start_index, desc=f"tapatan {mode}", unit="sample")
        if use_tqdm
        else None
    )

    def _finalize_one(idx: int, request: Dict[str, Any], raw_response) -> None:
        # Handle dict response (with metadata) vs plain string
        if isinstance(raw_response, dict):
            content = raw_response.get("content", "")
            finish_reason = raw_response.get("finish_reason")
        else:
            content = raw_response
            finish_reason = None

        example = request["example"]
        result_entry = {
            "id": example.get("id"),
            "difficulty": difficulty,
            "board_size": example.get("board_size", config.board_size),
            "pieces_per_player": example.get("pieces_per_player", config.pieces_per_player),
            "line_length": example.get("line_length", config.line_length),
            "input_format": input_format,
            "test_moves": request["moves"],
            "test_board_state": request["board_state"],
            "expected_result": request["expected"],
            "prompt": request["prompt"],
            "raw_response": content,
            "demos_used": request["demos"] if mode in {"examples", "combined"} else [],
        }
        if finish_reason is not None:
            result_entry["finish_reason"] = finish_reason
        results.append(result_entry)

        if use_tqdm and pbar is not None:
            pbar.update(1)
        else:
            print(f"[{idx}/{len(requests)}] Response={content[:50] if content else 'N/A'}")

    def _persist_progress() -> None:
        processed = len(results)
        if processed % snapshot_every != 0 and processed < len(requests):
            return
        snapshot_payload = {
            "task": "tapatan",
            "difficulty": difficulty,
            "mode": mode,
            "model": model_name,
            "num_test_samples": len(requests),
            "num_demos": num_demos if mode in {"examples", "combined"} else 0,
            "status": "in_progress",
            "processed": processed,
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
        if max_concurrent > 1:
            semaphore = asyncio.Semaphore(max_concurrent)
            pending_results: list = []

            async def _query_and_finalize(idx: int, request: Dict[str, Any]) -> None:
                async with semaphore:
                    try:
                        raw_response = await query_fn(request["prompt"])
                    except Exception as exc:  # noqa: BLE001
                        raw_response = str(exc)
                _finalize_one(idx, request, raw_response)

            tasks_list = [
                asyncio.create_task(_query_and_finalize(idx, req))
                for idx, req in enumerate(requests[start_index:], start=start_index + 1)
            ]
            for coro in asyncio.as_completed(tasks_list):
                await coro
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
    demos_tag = f"demos{num_demos}" if mode in {"examples", "combined"} else "demos0"
    filename = f"tapatan_{input_format}_{difficulty}_{mode}_{demos_tag}_{safe_model_name}_{timestamp}.json"
    output_path = os.path.join(task_output_dir, filename)

    payload = {
        "task": "tapatan",
        "difficulty": difficulty,
        "input_format": input_format,
        "mode": mode,
        "model": model_name,
        "num_test_samples": len(selected_tests),
        "num_demos": num_demos if mode in {"examples", "combined"} else 0,
        "timestamp": timestamp,
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

    print("\n=== Tapatan Experiment Summary ===")
    print(f" Model: {model_name}")
    print(f" Difficulty: {difficulty}")
    print(f" Mode: {mode}")
    print(f" Results saved to: {output_path}")
    print("==================================\n")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tapatan experiments.")
    parser.add_argument("--difficulty", type=str, default="1", choices=["1", "2", "3", "easy", "medium", "hard"])
    parser.add_argument("--mode", type=str, default="rules", choices=["rules", "examples", "combined"])
    parser.add_argument(
        "--input-format",
        type=str,
        default="move_sequence",
        choices=["move_sequence", "final_board_state", "moves", "board"],
    )
    parser.add_argument("--num-test-samples", type=int, default=50)
    parser.add_argument("--num-demos", type=int, default=None,
                        help="Number of demos (default: easy=12, medium=18, hard=24)")
    parser.add_argument("--model", type=str, default="qwen3-14b")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default=os.path.join(WORKSPACE_ROOT, "experiment_outputs"))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    asyncio.run(
        run_tapatan_experiment(
            difficulty=args.difficulty,
            mode=args.mode,
            num_test_samples=args.num_test_samples,
            num_demos=args.num_demos,
            output_dir=args.output_dir,
            model_name=args.model,
            seed=args.seed,
            max_tokens=args.max_tokens,
            input_format=args.input_format,
        )
    )
