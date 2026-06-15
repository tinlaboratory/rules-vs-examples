"""Noun class agreement task with Yes/No outputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from prompts.noun_class_agreement_prompts import (
    get_example_based_prompt,
    get_combined_prompt,
    get_rule_based_prompt,
    normalize_difficulty,
)
from utils.cluster.checkpointing import (
    clear_checkpoint,
    get_checkpoint_every,
    get_snapshot_every,
    is_resume_enabled,
    load_checkpoint,
    save_checkpoint,
    save_result_snapshot,
)
from utils.cluster.cluster_model_interface import get_vllm_batch_size, query_prompts_batched
from utils.dataset_io import get_data_root, load_records, select_latest_file

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_ROOT = get_data_root(WORKSPACE_ROOT)
NCA_DATA_DIR = os.path.join(DATA_ROOT, "noun_class_agreement")
DIFFICULTY_NAME = {1: "easy", 2: "medium", 3: "hard"}
_STRICT_LINE_RE = re.compile(r"^\s*(?:Answer\s*:?\s*)?(Yes|No)\s*[.!]?\s*$", re.IGNORECASE)
_ANSWER_PREFIX_RE = re.compile(r"(?:^|\n)\s*(?:Answer|Final answer)\s*:?\s*(Yes|No)\b", re.IGNORECASE)
_ANY_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedYesNo:
    answer: str
    method: str


def extract_yes_no(text: str) -> ParsedYesNo:
    """Parse a Yes/No answer with deterministic regex rules."""
    if not isinstance(text, str):
        return ParsedYesNo("", "non_string")

    raw = text.strip()
    if not raw:
        return ParsedYesNo("", "empty")

    for line in raw.splitlines():
        match = _STRICT_LINE_RE.match(line)
        if match:
            return ParsedYesNo("Yes" if match.group(1).lower() == "yes" else "No", "strict_line")

    match = _ANSWER_PREFIX_RE.search(raw)
    if match:
        return ParsedYesNo("Yes" if match.group(1).lower() == "yes" else "No", "answer_prefix")

    match = _ANY_YESNO_RE.search(raw)
    if match:
        return ParsedYesNo("Yes" if match.group(1).lower() == "yes" else "No", "first_yes_no")

    return ParsedYesNo("", "no_match")

# Difficulty-specific demo counts based on coverage requirements:
#   pos_min = ceil(N_nouns / 2)   — each positive sentence covers 2 nouns
#   neg_min = E_error_types       — show each error type at least once
#   k = 2 * max(pos_min, neg_min) — keep Yes/No balanced at k/2 each
#
# D1: N=8,  E=2 → pos_min=4, neg_min=2, k=8  (4Y + 4N)
# D2: N=8,  E=6 → pos_min=4, neg_min=6, k=12 (6Y + 6N)
# D3: N=12, E=8 → pos_min=6, neg_min=8, k=16 (8Y + 8N)
_DIFFICULTY_NUM_DEMOS = {1: 8, 2: 12, 3: 16}


def get_default_num_demos(difficulty: int) -> int:
    """Return the plan-based demo count for a difficulty level."""
    return _DIFFICULTY_NUM_DEMOS[difficulty]


def _extract_n(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r"_n(\d+)\.(?:jsonl|json)$", base)
    return int(m.group(1)) if m else -1


def _select_latest_file(patterns: Sequence[str]) -> Optional[str]:
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(glob(pattern))
    if not matches:
        return None
    return max(matches, key=_extract_n)


def load_nca_split(difficulty: int, split: str) -> List[Dict[str, Any]]:
    base = os.path.join(NCA_DATA_DIR, f"d{difficulty}", split)
    hf_base = os.path.join(NCA_DATA_DIR, DIFFICULTY_NAME[difficulty])
    patterns = [
        os.path.join(hf_base, f"{split}.jsonl"),
        os.path.join(base, f"noun_class_agreement_d{difficulty}_{split}_n*.jsonl"),
        os.path.join(base, f"noun_class_agreement_d{difficulty}_{split}_n*.json"),
    ]
    path = select_latest_file(patterns)
    if not path:
        raise FileNotFoundError(
            f"No noun class agreement data found for d{difficulty}/{split}. "
            f"Run data_generation/generate_noun_class_agreement.py first."
        )
    return load_records(path)


def _select_positive_demos(
    positives: Sequence[Dict[str, Any]],
    num_needed: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Select positive demos ensuring every noun appears at least once.

    Phase 1 – greedy coverage: repeatedly pick an unused example that
    covers the most still-uncovered nouns (preferring examples that cover
    2 uncovered nouns over 1).  Among ties, choose randomly.

    Phase 2 – fill: pad remaining slots with randomly chosen unused
    positives.
    """
    if num_needed <= 0 or not positives:
        return []

    # Collect all nouns present in the positive pool.
    all_nouns: set[str] = set()
    for item in positives:
        meta = item.get("meta") or {}
        for noun in {meta.get("subject_noun", ""), meta.get("object_noun", "")} - {""}:
            all_nouns.add(noun)

    pool = list(positives)
    rng.shuffle(pool)

    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    uncovered: set[str] = set(all_nouns)

    # Phase 1: greedily cover all nouns.
    while uncovered and len(selected) < num_needed:
        best_score = 0
        candidates: List[Dict[str, Any]] = []
        for item in pool:
            if str(item.get("id")) in selected_ids:
                continue
            meta = item.get("meta") or {}
            item_nouns = {meta.get("subject_noun", ""), meta.get("object_noun", "")} - {""}
            score = len(item_nouns & uncovered)
            if score > best_score:
                best_score = score
                candidates = [item]
            elif score == best_score and score > 0:
                candidates.append(item)

        if not candidates:
            break

        chosen = rng.choice(candidates)
        selected.append(chosen)
        selected_ids.add(str(chosen.get("id")))
        meta = chosen.get("meta") or {}
        for noun in {meta.get("subject_noun", ""), meta.get("object_noun", "")} - {""}:
            uncovered.discard(noun)

    # Phase 2: fill remaining slots randomly from unused positives.
    if len(selected) < num_needed:
        remaining = [item for item in pool if str(item.get("id")) not in selected_ids]
        selected.extend(remaining[: num_needed - len(selected)])

    return selected[:num_needed]


def _select_negative_demos(
    negatives: Sequence[Dict[str, Any]],
    num_needed: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if num_needed <= 0 or not negatives:
        return []

    by_error: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in negatives:
        meta = item.get("meta") or {}
        error_type = str(meta.get("error_type") or "unknown")
        by_error[error_type].append(item)

    keys = list(by_error.keys())
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(by_error[key])

    selected: List[Dict[str, Any]] = []
    while len(selected) < num_needed:
        added = False
        for key in keys:
            if len(selected) >= num_needed:
                break
            bucket = by_error[key]
            if bucket:
                selected.append(bucket.pop())
                added = True
        if not added:
            break

    if len(selected) < num_needed:
        leftovers = [item for bucket in by_error.values() for item in bucket]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: num_needed - len(selected)])

    return selected[:num_needed]


def sample_balanced_demos(
    train_examples: Sequence[Dict[str, Any]],
    num_demos: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if num_demos <= 0 or not train_examples:
        return []

    positives = [item for item in train_examples if str(item.get("label")) == "Yes"]
    negatives = [item for item in train_examples if str(item.get("label")) == "No"]

    target_pos = min(len(positives), num_demos // 2)
    target_neg = min(len(negatives), num_demos - target_pos)

    selected: List[Dict[str, Any]] = []
    if target_pos > 0:
        selected.extend(_select_positive_demos(positives, target_pos, rng))
    if target_neg > 0:
        selected.extend(_select_negative_demos(negatives, target_neg, rng))

    if len(selected) < num_demos:
        selected_ids = {str(item.get("id")) for item in selected}
        leftovers = [item for item in train_examples if str(item.get("id")) not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: num_demos - len(selected)])

    rng.shuffle(selected)
    return selected[:num_demos]


def _sample_balanced_test_cases(
    test_pool: Sequence[Dict[str, Any]],
    num_test_samples: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if num_test_samples <= 0 or not test_pool:
        return []
    if num_test_samples >= len(test_pool):
        selected = list(test_pool)
        rng.shuffle(selected)
        return selected

    positives = [item for item in test_pool if str(item.get("label")) == "Yes"]
    negatives = [item for item in test_pool if str(item.get("label")) == "No"]
    target_pos = min(len(positives), num_test_samples // 2)
    target_neg = min(len(negatives), num_test_samples - target_pos)

    selected: List[Dict[str, Any]] = []
    if target_pos > 0:
        selected.extend(rng.sample(positives, target_pos))
    if target_neg > 0:
        selected.extend(_select_negative_demos(negatives, target_neg, rng))

    if len(selected) < num_test_samples:
        selected_ids = {str(item.get("id")) for item in selected}
        leftovers = [item for item in test_pool if str(item.get("id")) not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: num_test_samples - len(selected)])

    rng.shuffle(selected)
    return selected[:num_test_samples]


async def run_noun_class_agreement_experiment(
    difficulty: int | str,
    mode: str,
    num_test_samples: int,
    num_demos: Optional[int],
    output_dir: str,
    model_name: str,
    seed: int = 123,
    max_tokens: int = 8,
    query_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    query_batch_fn: Optional[Callable[[List[str]], Awaitable[List[str]]]] = None,
    query_batch_size: Optional[int] = None,
    max_concurrent: int = 1,
) -> Dict[str, Any]:
    if mode not in {"rules", "examples", "combined"}:
        raise ValueError(f"Unsupported mode '{mode}'. Choose from rules/examples/combined.")

    level = normalize_difficulty(difficulty)

    if num_demos is None:
        num_demos = get_default_num_demos(level)

    rng = random.Random(seed)

    train_examples: List[Dict[str, Any]] = []
    if mode in {"examples", "combined"} and num_demos > 0:
        train_examples = load_nca_split(level, "train")

    test_pool = load_nca_split(level, "test")
    if not test_pool:
        raise ValueError(f"No test examples found for noun_class_agreement difficulty {level}.")

    selected_tests = _sample_balanced_test_cases(test_pool, num_test_samples, rng)

    safe_model_name = model_name.replace("/", "_")
    task_output_dir = os.path.join(output_dir, "noun_class_agreement", f"d{level}", mode, safe_model_name)
    os.makedirs(task_output_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        task_output_dir,
        f"noun_class_agreement_d{level}_{mode}_demos{num_demos}_{safe_model_name}_checkpoint.json",
    )
    snapshot_path = os.path.join(
        task_output_dir,
        f"noun_class_agreement_d{level}_{mode}_demos{num_demos}_{safe_model_name}_latest.json",
    )
    resume_enabled = is_resume_enabled()
    checkpoint_every = get_checkpoint_every()
    snapshot_every = get_snapshot_every(checkpoint_every)
    run_signature = {
        "task": "noun_class_agreement",
        "difficulty": level,
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
            f"[checkpoint] Resuming noun_class_agreement from {start_index}/{len(requests)} "
            f"using {checkpoint_path}"
        )
    else:
        for example in selected_tests:
            sentence = str(example.get("sentence") or "")
            demos: List[Dict[str, Any]] = []
            if mode in {"examples", "combined"}:
                demos = sample_balanced_demos(train_examples, num_demos, rng)

            if mode == "rules":
                prompt = get_rule_based_prompt(level, sentence)
            elif mode == "examples":
                prompt = get_example_based_prompt(level, demos, sentence)
            else:
                prompt = get_combined_prompt(level, demos, sentence)

            requests.append(
                {
                    "example": example,
                    "sentence": sentence,
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
        tqdm(total=len(requests), initial=start_index, desc=f"noun_class_agreement {mode}", unit="sample")
        if use_tqdm
        else None
    )

    def _finalize_one(idx: int, request: Dict[str, Any], raw_response: str) -> None:
        example = request["example"]
        parsed = extract_yes_no(raw_response)
        prediction = parsed.answer
        expected = str(example.get("label") or "")
        meta = example.get("meta") or {}
        result_entry = {
            "id": example.get("id"),
            "difficulty": level,
            "difficulty_name": DIFFICULTY_NAME[level],
            "sentence": request["sentence"],
            "expected_label": expected,
            "prediction": prediction,
            "parse_method": parsed.method,
            "error_type": meta.get("error_type"),
            "meta": meta,
            "prompt": request["prompt"],
            "raw_response": raw_response,
            "demos_used": request["demos"] if mode in {"examples", "combined"} else [],
        }
        results.append(result_entry)

        if use_tqdm and pbar is not None:
            pbar.update(1)

    def _persist_progress() -> None:
        processed = len(results)
        if processed % snapshot_every != 0 and processed < len(requests):
            return
        snapshot_payload = {
            "task": "noun_class_agreement",
            "difficulty": level,
            "difficulty_name": DIFFICULTY_NAME[level],
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
                _finalize_one(chunk_start + offset, request, raw_response)
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
                _finalize_one(chunk_start + offset, request, raw_response)
            _persist_progress()
    else:
        if max_concurrent > 1:
            # Fire concurrent local-model requests in chunks for throughput.
            async def _query_one(prompt: str) -> str:
                try:
                    return await query_fn(prompt)
                except Exception as exc:  # noqa: BLE001
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
                except Exception as exc:  # noqa: BLE001
                    raw_response = str(exc)
                _finalize_one(idx, request, raw_response)
                _persist_progress()

    if pbar is not None:
        pbar.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        task_output_dir,
        f"noun_class_agreement_d{level}_{mode}_demos{num_demos}_{safe_model_name}_{timestamp}.json",
    )

    payload = {
        "task": "noun_class_agreement",
        "difficulty": level,
        "difficulty_name": DIFFICULTY_NAME[level],
        "mode": mode,
        "model": model_name,
        "num_test_samples": len(selected_tests),
        "num_demos": num_demos if mode in {"examples", "combined"} else 0,
        "seed": seed,
        "timestamp": timestamp,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    if resume_enabled:
        clear_checkpoint(checkpoint_path)
    # Remove the live snapshot now that the final timestamped file exists.
    if os.path.exists(snapshot_path):
        os.remove(snapshot_path)

    print(f"\nResults saved to: {output_path}")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run noun class agreement experiments.")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="1",
        choices=["1", "2", "3", "easy", "medium", "hard"],
    )
    parser.add_argument("--mode", type=str, default="rules", choices=["rules", "examples", "combined"])
    parser.add_argument("--num-test-samples", type=int, default=50)
    parser.add_argument(
        "--num-demos",
        type=int,
        default=None,
        help="Number of demos per test example. Defaults to difficulty-specific counts (D1=8, D2=12, D3=16).",
    )
    parser.add_argument("--model", type=str, default="qwen3-14b")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default=os.path.join(WORKSPACE_ROOT, "experiment_outputs"))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    asyncio.run(
        run_noun_class_agreement_experiment(
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
