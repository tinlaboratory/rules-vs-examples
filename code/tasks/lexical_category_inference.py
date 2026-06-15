from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from glob import glob
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from prompts.lexical_category_inference_prompts import (
    get_default_num_demos,
    get_demo_schedule_for_num_demos,
    get_example_based_prompt,
    get_combined_prompt,
    get_rule_based_prompt,
    normalize_difficulty,
    normalize_logic_condition,
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
LEXICAL_CATEGORY_INFERENCE_DATA_DIR = os.path.join(DATA_ROOT, "lexical_category_inference")
BALANCED_TEST_SUBSETS_PATH = os.path.join(
    LEXICAL_CATEGORY_INFERENCE_DATA_DIR,
    "balanced_test_subsets_n600_seed123.json",
)
DIFFICULTY_NAME = {
    "d1": "easy",
    "d2": "medium",
    "d3": "hard",
}
PROMPT_DEMO_SEMANTICS = "or_bridged_cross_label_validated_v38"
OUT_OF_CATEGORY_LABEL = "OUT_OF_CATEGORY"
_SUPPORT_PATTERN_CACHE: Dict[tuple[int, int], List[tuple[int, ...]]] = {}


@dataclass(frozen=True)
class ParsedYesNo:
    answer: str
    method: str


def extract_exact_yes_no(text: str) -> ParsedYesNo:
    """Parse Yes/No only when the whole response is exactly the answer."""
    if not isinstance(text, str):
        return ParsedYesNo("", "non_string")

    raw = text.strip()
    if not raw:
        return ParsedYesNo("", "empty")
    if raw == "Yes":
        return ParsedYesNo("Yes", "exact")
    if raw == "No":
        return ParsedYesNo("No", "exact")
    if raw.lower() == "yes":
        return ParsedYesNo("Yes", "exact_casefold")
    if raw.lower() == "no":
        return ParsedYesNo("No", "exact_casefold")
    return ParsedYesNo("", "not_exact")


def extract_first_answer_token(text: str) -> ParsedYesNo:
    """Parse Yes/No only when the first generated word is the answer."""
    if not isinstance(text, str):
        return ParsedYesNo("", "non_string")

    raw = text.lstrip()
    if not raw:
        return ParsedYesNo("", "empty")

    match = re.match(r"([A-Za-z]+)", raw)
    token = match.group(1).lower() if match else ""
    if token == "yes":
        return ParsedYesNo("Yes", "first_answer_token")
    if token == "no":
        return ParsedYesNo("No", "first_answer_token")
    return ParsedYesNo("", "not_first_answer_token")

SAFE_COMPONENT_IMPLICATIONS: Dict[str, set[str]] = {
    "bird": {"animal"},
    "insect": {"animal"},
    "mammal": {"animal"},
    "fish": {"animal"},
    "farm_animal": {"animal"},
    "sea_animal": {"animal"},
    "watercraft": {"vehicle"},
    "footwear": {"clothing"},
    "headwear": {"clothing"},
    "outerwear": {"clothing"},
    "protective_clothing": {"clothing"},
    "womens_clothing": {"clothing"},
    "clothing_accessory": {"clothing"},
    "fruit": {"food"},
    "vegetable": {"food"},
    "dessert": {"food"},
    "breakfast_food": {"food"},
    "candy": {"food"},
    "condiment": {"food"},
    "seafood": {"food"},
    "kitchen_tool": {"tool"},
    "garden_tool": {"tool"},
}

COMPONENT_CONCEPT_PHRASES: tuple[tuple[str, str], ...] = (
    ("things used for transportation", "vehicle"),
    ("classified as watercraft", "watercraft"),
    ("things classified as watercraft", "watercraft"),
    ("classified as vehicle", "vehicle"),
    ("things classified as vehicle", "vehicle"),
    ("classified as bird", "bird"),
    ("things classified as bird", "bird"),
    ("things that are birds", "bird"),
    ("things classified as insect", "insect"),
    ("things that are insects", "insect"),
    ("things classified as mammal", "mammal"),
    ("things that are mammals", "mammal"),
    ("things that are fish", "fish"),
    ("classified as fish", "fish"),
    ("things classified as fish", "fish"),
    ("farm animal", "farm_animal"),
    ("sea animal", "sea_animal"),
    ("classified as animal", "animal"),
    ("things classified as animal", "animal"),
    ("things that are animals", "animal"),
    ("classified as footwear", "footwear"),
    ("things classified as footwear", "footwear"),
    ("classified as headwear", "headwear"),
    ("things classified as headwear", "headwear"),
    ("classified as outerwear", "outerwear"),
    ("things classified as outerwear", "outerwear"),
    ("protective clothing", "protective_clothing"),
    ("women's clothing", "womens_clothing"),
    ("womens clothing", "womens_clothing"),
    ("clothing accessory", "clothing_accessory"),
    ("classified as clothing", "clothing"),
    ("things classified as clothing", "clothing"),
    ("classified as fruit", "fruit"),
    ("things classified as fruit", "fruit"),
    ("things that are fruit", "fruit"),
    ("classified as vegetable", "vegetable"),
    ("things classified as vegetable", "vegetable"),
    ("things that are vegetables", "vegetable"),
    ("classified as dessert", "dessert"),
    ("things classified as dessert", "dessert"),
    ("breakfast food", "breakfast_food"),
    ("classified as candy", "candy"),
    ("things classified as candy", "candy"),
    ("classified as condiment", "condiment"),
    ("things classified as condiment", "condiment"),
    ("classified as seafood", "seafood"),
    ("things classified as seafood", "seafood"),
    ("classified as food", "food"),
    ("things classified as food", "food"),
    ("things that are food", "food"),
    ("kitchen tool", "kitchen_tool"),
    ("garden tool", "garden_tool"),
    ("classified as tool", "tool"),
    ("things classified as tool", "tool"),
    ("things that are tools", "tool"),
    ("things that are utensils", "utensil"),
    ("classified as utensil", "utensil"),
)

DESIGN_CELL_SPECS = {
    ("shared", "d1"): {"category_operator": "single", "category_arity": 1, "shared_d1": True},
    ("either", "d2"): {"category_operator": "or", "category_arity": 2, "shared_d1": False},
    ("either", "d3"): {"category_operator": "or", "category_arity": 3, "shared_d1": False},
    ("both", "d2"): {"category_operator": "and", "category_arity": 2, "shared_d1": False},
    ("both", "d3"): {"category_operator": "and", "category_arity": 3, "shared_d1": False},
}
REAL_DESIGN_CELLS = [
    ("shared", "d1"),
    ("either", "d2"),
    ("either", "d3"),
    ("both", "d2"),
    ("both", "d3"),
]


def _as_list(value: Optional[str] | Sequence[str], *, default: str) -> List[str]:
    if value is None:
        return [default]
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _normalize_logic_condition_for_plan(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "all":
        return "all"
    logic = normalize_logic_condition(normalized)
    if logic is None:
        raise ValueError("logic_condition cannot be empty in a Lexical Category Inference cell plan.")
    return logic


def _extract_n(path: str) -> int:
    base = os.path.basename(path)
    match = re.search(r"_n(\d+)\.(?:jsonl|json)$", base)
    return int(match.group(1)) if match else -1


def _hf_logic_name(logic: str) -> str:
    return {"shared": "shared", "either": "disjunctive", "both": "conjunctive"}[logic]


def _hf_level_name(level: str) -> str:
    return DIFFICULTY_NAME[level]


def resolve_design_cell(difficulty: str, logic_condition: Optional[str]) -> tuple[str, str, Dict[str, Any]]:
    level = normalize_difficulty(difficulty)
    logic = normalize_logic_condition(logic_condition)
    if level == "d1":
        if logic in {None, "shared", "either", "both"}:
            return "shared", "d1", DESIGN_CELL_SPECS[("shared", "d1")]
        raise ValueError(f"d1 is shared; unsupported logic_condition={logic_condition!r}.")
    if logic not in {"either", "both"}:
        raise ValueError(
            "lexical_category_inference d2/d3 require logic_condition='either' or 'both'. "
            "The d1 baseline is loaded from shared/d1."
        )
    return logic, level, DESIGN_CELL_SPECS[(logic, level)]


def get_design_cell_plan(
    difficulty: Optional[str] | Sequence[str] = "all",
    logic_condition: Optional[str] | Sequence[str] = "all",
) -> List[tuple[str, str, Dict[str, Any]]]:
    """Return the ordered, deduplicated lexical_category_inference cells to run.

    The full experiment grid has five real cells. `shared/d1` is emitted at most
    once even when a caller asks for both OR and AND conditions.
    """

    difficulty_values = _as_list(difficulty, default="all")
    logic_default = "all" if any(str(item).strip().lower() == "all" for item in difficulty_values) else ""
    logic_values = _as_list(logic_condition, default=logic_default)

    cells: Dict[tuple[str, str], tuple[str, str, Dict[str, Any]]] = {}
    for difficulty_value in difficulty_values:
        difficulty_text = str(difficulty_value).strip().lower()
        levels = ["d1", "d2", "d3"] if difficulty_text == "all" else [normalize_difficulty(difficulty_text)]

        for level in levels:
            for logic_value in logic_values:
                logic_text = str(logic_value).strip().lower()
                if not logic_text:
                    if level == "d1":
                        logic_text = "shared"
                    else:
                        raise ValueError(
                            "lexical_category_inference d2/d3 require logic_condition='either', "
                            "'both', or 'all'."
                        )
                logic = _normalize_logic_condition_for_plan(logic_text)

                if logic == "all":
                    if level == "d1":
                        targets = [("shared", "d1")]
                    else:
                        targets = [("either", level), ("both", level)]
                elif level == "d1":
                    if logic in {"shared", "either", "both"}:
                        targets = [("shared", "d1")]
                    else:
                        raise ValueError(f"d1 is shared; unsupported logic_condition={logic_value!r}.")
                elif logic == "shared":
                    if difficulty_text == "all":
                        targets = []
                    else:
                        raise ValueError("logic_condition='shared' is only valid for d1.")
                else:
                    targets = [(logic, level)]

                for target_logic, target_level in targets:
                    spec = DESIGN_CELL_SPECS[(target_logic, target_level)]
                    cells[(target_logic, target_level)] = (target_logic, target_level, dict(spec))

    return [cells[key] for key in REAL_DESIGN_CELLS if key in cells]


def load_lexical_category_inference_split(
    difficulty: str,
    split: str,
    logic_condition: Optional[str] = None,
) -> List[Dict[str, Any]]:
    logic, level, _ = resolve_design_cell(difficulty, logic_condition)
    base_dir = os.path.join(LEXICAL_CATEGORY_INFERENCE_DATA_DIR, logic, level, split)
    hf_path = os.path.join(
        LEXICAL_CATEGORY_INFERENCE_DATA_DIR,
        _hf_level_name(level),
        _hf_logic_name(logic),
        f"{split}.jsonl",
    )
    patterns = [
        hf_path,
        os.path.join(base_dir, f"lexical_category_inference_{logic}_{level}_{split}_n*.jsonl"),
        os.path.join(base_dir, f"lexical_category_inference_{logic}_{level}_{split}_n*.json"),
    ]
    path = select_latest_file(patterns)
    if not path:
        raise FileNotFoundError(
            f"No lexical_category_inference data found for {logic}/{level}/{split}. "
            f"Run data_generation/generate_lexical_category_inference.py first."
        )
    return load_records(path)


def load_lexical_category_inference_exact_split(
    difficulty: str,
    split: str,
    num_rows: int,
    logic_condition: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    logic, level, _ = resolve_design_cell(difficulty, logic_condition)
    path = os.path.join(
        LEXICAL_CATEGORY_INFERENCE_DATA_DIR,
        logic,
        level,
        split,
        f"lexical_category_inference_{logic}_{level}_{split}_n{int(num_rows)}.jsonl",
    )
    hf_path = os.path.join(
        LEXICAL_CATEGORY_INFERENCE_DATA_DIR,
        _hf_level_name(level),
        _hf_logic_name(logic),
        f"{split}.jsonl",
    )
    if os.path.exists(hf_path):
        return load_records(hf_path)
    if not os.path.exists(path):
        json_path = path[:-1]
        if not os.path.exists(json_path):
            return None
        path = json_path
    return load_records(path)


def _available_support_rounds(example: Dict[str, Any]) -> int:
    labels = list(example.get("labels") or [])
    support_groups = dict(example.get("support_groups_by_label") or {})
    if not labels:
        return 0

    for label in labels:
        groups = support_groups.get(label) or []
        if not groups:
            raise ValueError(
                f"Episode {example.get('episode_id') or example.get('id')} "
                f"does not have embedded support groups for {label}."
            )
    return min(len(support_groups.get(label) or []) for label in labels)


def _resolve_support_groups_shown(
    example: Dict[str, Any],
    prompt_demos: Sequence[Dict[str, Any]],
) -> Dict[str, List[List[str]]]:
    labels = list(example.get("labels") or [])
    support_groups = dict(example.get("support_groups_by_label") or {})
    per_label_support_indices: Dict[str, set[int]] = {label: set() for label in labels}
    for demo in prompt_demos:
        indices_by_label = demo.get("support_indices_by_label") or {}
        if indices_by_label:
            candidate_groups = demo.get("candidate_groups") or []
            if candidate_groups:
                visible_gold_labels = [
                    str(group.get("gold_label"))
                    for group in candidate_groups
                    if str(group.get("gold_label")) in per_label_support_indices
                ]
            else:
                visible_gold_labels = labels
            for label in visible_gold_labels:
                support_index = int(indices_by_label.get(label, -1))
                if support_index >= 0:
                    per_label_support_indices[label].add(support_index)
            continue

        support_index = int(demo.get("support_index", -1))
        if support_index < 0:
            continue
        for label in labels:
            per_label_support_indices[label].add(support_index)

    if not any(indices for indices in per_label_support_indices.values()):
        return {}

    shown: Dict[str, List[List[str]]] = {}
    for label in labels:
        all_groups = support_groups.get(label) or []
        groups: List[List[str]] = []
        for support_index in sorted(per_label_support_indices[label]):
            if support_index < 0 or support_index >= len(all_groups):
                raise ValueError(
                    f"Episode {example.get('episode_id') or example.get('id')} "
                    f"does not have support group index {support_index} for {label}."
                )
            if not all_groups[support_index]:
                raise ValueError(
                    f"Episode {example.get('episode_id') or example.get('id')} "
                    f"has an empty support group at index {support_index} for {label}."
                )
            words = [str(word).strip().lower() for word in all_groups[support_index]]
            if len(words) != 1:
                raise ValueError(
                    f"Episode {example.get('episode_id') or example.get('id')} "
                    f"support group index {support_index} for {label} has {len(words)} words; "
                    "prompt-visible support groups must be singleton words."
                )
            groups.append(words)
        shown[label] = groups
    return shown


def _select_balanced_rows_by_key(
    rows: Sequence[Dict[str, Any]],
    num_needed: int,
    key_fn: Callable[[Dict[str, Any]], str],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if num_needed <= 0 or not rows:
        return []

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[key_fn(row)].append(row)

    keys = list(buckets.keys())
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(buckets[key])

    selected: List[Dict[str, Any]] = []
    while len(selected) < num_needed:
        added = False
        for key in keys:
            if len(selected) >= num_needed:
                break
            bucket = buckets[key]
            if not bucket:
                continue
            selected.append(bucket.pop())
            added = True
        if not added:
            break

    if len(selected) < num_needed:
        leftovers = [item for bucket in buckets.values() for item in bucket]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: num_needed - len(selected)])

    return selected[:num_needed]


@lru_cache(maxsize=1)
def _load_balanced_test_subsets() -> Dict[str, Any]:
    if not os.path.exists(BALANCED_TEST_SUBSETS_PATH):
        return {}
    with open(BALANCED_TEST_SUBSETS_PATH, "r") as handle:
        data = json.load(handle)
    return dict(data.get("cells") or {})


def _balanced_subset_cell_key(test_pool: Sequence[Dict[str, Any]]) -> Optional[str]:
    if not test_pool:
        return None

    first = test_pool[0]
    difficulty_text = str(first.get("difficulty") or "")
    try:
        level = normalize_difficulty(difficulty_text)
    except ValueError:
        return None

    logic = str(first.get("logic_condition") or "").strip().lower()
    operator = str(first.get("category_operator") or "").strip().lower()
    if level == "d1":
        logic = "shared"
    elif logic in {"or", "and"}:
        logic = {"or": "either", "and": "both"}[logic]
    elif logic not in {"either", "both"}:
        logic = {"or": "either", "and": "both"}.get(operator, logic)

    if logic not in {"shared", "either", "both"}:
        return None
    return f"{logic}/{level}"


def _select_precomputed_balanced_test_subset(
    test_pool: Sequence[Dict[str, Any]],
    num_test_samples: int,
    seed: Optional[int],
) -> Optional[List[Dict[str, Any]]]:
    if seed != 123 or num_test_samples != 600:
        return None

    cell_key = _balanced_subset_cell_key(test_pool)
    if not cell_key:
        return None

    subset = _load_balanced_test_subsets().get(cell_key)
    if not subset:
        return None

    source_pool_num_rows = subset.get("source_pool_num_rows")
    if source_pool_num_rows is not None and len(test_pool) != int(source_pool_num_rows):
        return None

    selected_ids = [str(row_id) for row_id in subset.get("selected_ids") or []]
    if len(selected_ids) != num_test_samples:
        return None

    rows_by_id = {str(row.get("id")): row for row in test_pool}
    try:
        return [rows_by_id[row_id] for row_id in selected_ids]
    except KeyError as exc:
        raise ValueError(
            f"Balanced test subset {cell_key} references missing row id {exc.args[0]!r}."
        ) from exc


def sample_balanced_test_cases(
    test_pool: Sequence[Dict[str, Any]],
    num_test_samples: int,
    rng: random.Random,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if num_test_samples <= 0 or not test_pool:
        return []

    precomputed = _select_precomputed_balanced_test_subset(test_pool, num_test_samples, seed)
    if precomputed is not None:
        selected = list(precomputed)
        rng.shuffle(selected)
        return selected

    if num_test_samples >= len(test_pool):
        selected = list(test_pool)
        rng.shuffle(selected)
        return selected

    positives = [row for row in test_pool if str(row.get("answer")) == "Yes"]
    negatives = [row for row in test_pool if str(row.get("answer")) == "No"]

    target_yes = min(len(positives), num_test_samples // 2)
    target_no = min(len(negatives), num_test_samples - target_yes)

    selected: List[Dict[str, Any]] = []
    if target_yes > 0:
        selected.extend(rng.sample(positives, target_yes))
    if target_no > 0:
        selected.extend(
            _select_balanced_rows_by_key(
                negatives,
                target_no,
                key_fn=lambda row: str(row.get("corruption_type") or "unknown"),
                rng=rng,
            )
        )

    if len(selected) < num_test_samples:
        selected_ids = {str(item.get("id")) for item in selected}
        leftovers = [item for item in test_pool if str(item.get("id")) not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: num_test_samples - len(selected)])

    rng.shuffle(selected)
    return selected[:num_test_samples]


def _demo_negative_schedule(difficulty: str) -> List[str]:
    if difficulty == "d1":
        return ["two_swap", "out_of_category"]
    if difficulty == "d2":
        return ["two_swap", "three_cycle", "out_of_category"]
    return ["two_swap", "three_cycle", "four_cycle", "out_of_category"]


def _negative_demo_severity(corruption_type: str) -> tuple[int, int]:
    return {
        "out_of_category": (1, 0),
        "single_slot_intrusion": (2, 0),
        "two_swap": (2, 1),
        "three_cycle": (3, 1),
        "four_cycle": (4, 1),
    }[corruption_type]


def _ordered_support_index_patterns(
    labels: Sequence[str],
    available_support_rounds: int,
    limit: int | None = None,
) -> List[Dict[str, int]]:
    if not labels or available_support_rounds <= 0:
        return []

    if limit is not None:
        max_patterns = max(0, int(limit))
        ordered_patterns: List[tuple[int, ...]] = []
        seen_patterns: set[tuple[int, ...]] = set()
        for index in range(available_support_rounds):
            if len(ordered_patterns) >= max_patterns:
                break
            pattern = tuple(index for _ in labels)
            ordered_patterns.append(pattern)
            seen_patterns.add(pattern)
        for pattern in itertools.product(range(available_support_rounds), repeat=len(labels)):
            if len(ordered_patterns) >= max_patterns:
                break
            if pattern in seen_patterns:
                continue
            ordered_patterns.append(pattern)
            seen_patterns.add(pattern)
        return [
            {
                str(label): int(support_index)
                for label, support_index in zip(labels, pattern)
            }
            for pattern in ordered_patterns
        ]

    cache_key = (len(labels), available_support_rounds)
    ordered_patterns = _SUPPORT_PATTERN_CACHE.get(cache_key)
    if ordered_patterns is None:
        base_patterns = [
            tuple(index for _ in labels)
            for index in range(available_support_rounds)
        ]
        base_pattern_set = set(base_patterns)

        remaining_patterns = []
        ideal_count = len(labels) / float(available_support_rounds)
        for pattern in itertools.product(range(available_support_rounds), repeat=len(labels)):
            if pattern in base_pattern_set:
                continue
            counts = [pattern.count(index) for index in range(available_support_rounds)]
            distinct_rounds = len(set(pattern))
            imbalance = sum(abs(count - ideal_count) for count in counts)
            remaining_patterns.append(
                (
                    -distinct_rounds,
                    imbalance,
                    pattern,
                )
            )

        remaining_patterns.sort()
        ordered_patterns = list(base_patterns) + [pattern for _, _, pattern in remaining_patterns]
        _SUPPORT_PATTERN_CACHE[cache_key] = ordered_patterns
    if limit is not None:
        ordered_patterns = ordered_patterns[: max(0, int(limit))]
    return [
        {
            str(label): int(support_index)
            for label, support_index in zip(labels, pattern)
        }
        for pattern in ordered_patterns
    ]


def _ordered_component_balanced_support_patterns(
    example: Dict[str, Any],
    labels: Sequence[str],
    available_support_rounds: int,
    limit: int,
) -> List[Dict[str, int]]:
    if limit <= 0:
        return []
    if str(example.get("category_operator") or "") != "or" or _component_count(example) <= 1:
        return _ordered_support_index_patterns(labels, available_support_rounds, limit=limit)

    component_count = _component_count(example)
    indices_by_label_component: Dict[str, Dict[int, List[int]]] = {}
    for label in labels:
        per_component = {component_index: [] for component_index in range(component_count)}
        for support_index in range(available_support_rounds):
            components = _support_component_indices(example, str(label), support_index)
            for component_index in components:
                if int(component_index) in per_component:
                    per_component[int(component_index)].append(support_index)
        if any(not indices for indices in per_component.values()):
            return _ordered_support_index_patterns(labels, available_support_rounds, limit=limit)
        indices_by_label_component[str(label)] = per_component

    patterns: List[Dict[str, int]] = []
    seen_keys: set[tuple[int, ...]] = set()
    usage_by_label_component = {
        str(label): {component_index: 0 for component_index in range(component_count)}
        for label in labels
    }
    support_use_by_label = {str(label): Counter() for label in labels}

    attempts = 0
    while len(patterns) < limit and attempts < limit * component_count * max(4, len(labels)):
        pattern: Dict[str, int] = {}
        for label_offset, label in enumerate(labels):
            label = str(label)
            target_component = (attempts + label_offset) % component_count
            candidate_indices = indices_by_label_component[label][target_component]
            chosen_index = min(
                candidate_indices,
                key=lambda support_index: (
                    support_use_by_label[label][support_index],
                    usage_by_label_component[label][target_component],
                    support_index,
                ),
            )
            pattern[label] = chosen_index
        key = tuple(pattern[str(label)] for label in labels)
        if key not in seen_keys:
            patterns.append(pattern)
            seen_keys.add(key)
            for label in labels:
                label = str(label)
                support_index = pattern[label]
                support_use_by_label[label][support_index] += 1
                for component_index in _support_component_indices(example, label, support_index):
                    usage_by_label_component[label][int(component_index)] += 1
        attempts += 1

    if len(patterns) < limit:
        for pattern in _ordered_support_index_patterns(
            labels,
            available_support_rounds,
            limit=max(limit * 4, limit + available_support_rounds),
        ):
            if len(patterns) >= limit:
                break
            key = tuple(pattern[str(label)] for label in labels)
            if key in seen_keys:
                continue
            patterns.append(pattern)
            seen_keys.add(key)

    return patterns[:limit]


def _component_count(example: Dict[str, Any]) -> int:
    return max(1, int(example.get("category_arity") or example.get("condition_count") or 1))


def _normalize_component_gloss_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower().replace("-", " "))


def _component_glosses_for_label(example: Dict[str, Any], label: str) -> List[str]:
    rule_gloss = str((example.get("rule_glosses") or {}).get(label) or "")
    if ":" in rule_gloss:
        rule_gloss = rule_gloss.split(":", 1)[1]
    pieces = [
        piece.strip()
        for piece in rule_gloss.split(";")
        if piece.strip()
    ]
    if pieces:
        return pieces
    return [rule_gloss.strip()] if rule_gloss.strip() else []


@lru_cache(maxsize=4096)
def _component_concepts_from_gloss(gloss: str) -> frozenset[str]:
    clean = _normalize_component_gloss_text(gloss)
    concepts: set[str] = set()
    for phrase, concept in COMPONENT_CONCEPT_PHRASES:
        if phrase in clean:
            concepts.add(concept)
    return frozenset(concepts)


def _concept_closure(concepts: set[str]) -> set[str]:
    closure = set(concepts)
    changed = True
    while changed:
        changed = False
        for concept in list(closure):
            for parent in SAFE_COMPONENT_IMPLICATIONS.get(concept, set()):
                if parent not in closure:
                    closure.add(parent)
                    changed = True
    return closure


def _support_implied_target_components(
    example: Dict[str, Any],
    source_label: str,
    support_index: int,
    target_label: str,
) -> set[int]:
    source_glosses = _component_glosses_for_label(example, source_label)
    target_glosses = _component_glosses_for_label(example, target_label)
    if not source_glosses or not target_glosses:
        return set()

    source_component_indices = _support_component_indices(example, source_label, support_index)
    if str(example.get("category_operator") or "") == "and":
        source_component_indices = list(range(len(source_glosses)))
    source_concepts: set[str] = set()
    for component_index in source_component_indices:
        if 0 <= int(component_index) < len(source_glosses):
            source_concepts.update(_component_concepts_from_gloss(source_glosses[int(component_index)]))
    source_closure = _concept_closure(source_concepts)
    if not source_closure:
        return set()

    implied: set[int] = set()
    for target_component_index, target_gloss in enumerate(target_glosses):
        target_concepts = _component_concepts_from_gloss(target_gloss)
        if target_concepts and (target_concepts & source_closure):
            implied.add(target_component_index)
    return implied


def _support_group_is_semantic_member_for_label(
    example: Dict[str, Any],
    source_label: str,
    support_index: int,
    target_label: str,
) -> bool:
    if source_label == target_label:
        return True
    target_glosses = _component_glosses_for_label(example, target_label)
    implied_components = _support_implied_target_components(
        example,
        source_label,
        support_index,
        target_label,
    )
    if not implied_components:
        return False
    operator = str(example.get("category_operator") or "")
    if operator == "and":
        return len(implied_components) == len(target_glosses)
    return bool(implied_components)


def _support_component_indices(
    example: Dict[str, Any],
    label: str,
    support_index: int,
) -> List[int]:
    support_components = example.get("support_component_indices_by_label") or {}
    label_components = support_components.get(label) or []
    if 0 <= support_index < len(label_components):
        return [int(index) for index in (label_components[support_index] or [])]
    if str(example.get("category_operator") or "") == "and":
        return list(range(_component_count(example)))
    return [support_index % _component_count(example)]


def _first_support_index_for_component(
    example: Dict[str, Any],
    label: str,
    component_index: int,
    used_indices: set[int],
    available_support_rounds: int,
) -> int:
    exact_unused: List[int] = []
    any_unused: List[int] = []
    exact_used: List[int] = []
    any_used: List[int] = []
    for support_index in range(available_support_rounds):
        components = set(_support_component_indices(example, label, support_index))
        if component_index not in components:
            continue
        if components == {component_index}:
            (exact_used if support_index in used_indices else exact_unused).append(support_index)
        else:
            (any_used if support_index in used_indices else any_unused).append(support_index)
    for candidates in (exact_unused, any_unused, exact_used, any_used):
        if candidates:
            return candidates[0]
    return component_index % available_support_rounds


def _out_of_category_index_for_component(
    example: Dict[str, Any],
    label: str,
    component_index: int,
) -> int:
    groups = (example.get("out_of_category_groups_by_label") or {}).get(label) or []
    if not groups:
        return 0
    failed_components = (example.get("out_of_category_failed_component_indices_by_label") or {}).get(label) or []
    out_components = (example.get("out_of_category_component_indices_by_label") or {}).get(label) or []
    component_count = _component_count(example)
    for support_index, indices in enumerate(failed_components):
        failed = {int(index) for index in (indices or [])}
        satisfied = {
            int(index)
            for index in (
                out_components[support_index]
                if support_index < len(out_components)
                else []
            )
        }
        if (
            failed == {int(component_index)}
            and len(satisfied) == component_count - 1
            and int(component_index) not in satisfied
        ):
            return support_index
    for support_index, indices in enumerate(out_components):
        satisfied = {int(index) for index in (indices or [])}
        if satisfied and component_index not in satisfied:
            return support_index
    for support_index, indices in enumerate(out_components):
        if component_index in {int(index) for index in (indices or [])}:
            return support_index
    return component_index % len(groups)


def _out_of_category_index_for_none_of_components(
    example: Dict[str, Any],
    label: str,
    used_indices: set[int] | None = None,
) -> int:
    groups = (example.get("out_of_category_groups_by_label") or {}).get(label) or []
    if not groups:
        return 0
    used_indices = set(used_indices or set())
    failed_components = (example.get("out_of_category_failed_component_indices_by_label") or {}).get(label) or []
    satisfied_components = (example.get("out_of_category_component_indices_by_label") or {}).get(label) or []
    expected_components = set(range(_component_count(example)))

    strict_candidates: List[int] = []
    for support_index in range(len(groups)):
        satisfied = {
            int(index)
            for index in (
                satisfied_components[support_index]
                if support_index < len(satisfied_components)
                else []
            )
        }
        failed = {
            int(index)
            for index in (
                failed_components[support_index]
                if support_index < len(failed_components)
                else []
            )
        }
        if not satisfied and failed == expected_components:
            strict_candidates.append(support_index)

    for support_index in strict_candidates:
        if support_index not in used_indices:
            return support_index
    if strict_candidates:
        return strict_candidates[0]
    return 0


def _ordered_positive_demo_supports(
    example: Dict[str, Any],
    labels: Sequence[str],
    available_support_rounds: int,
    num_positive: int,
) -> List[Dict[str, int]]:
    max_support_patterns = available_support_rounds ** len(labels)
    if num_positive > max_support_patterns:
        raise ValueError(
            f"Only {max_support_patterns} distinct positive prompt demos are available without "
            "reusing episode-local support groups."
        )
    component_count = _component_count(example)
    if str(example.get("category_operator") or "") != "or" or component_count <= 1:
        support_patterns = _ordered_support_index_patterns(
            labels,
            available_support_rounds,
            limit=num_positive,
        )
        if len(support_patterns) < num_positive:
            raise ValueError(
                f"Only {len(support_patterns)} distinct positive prompt demos are available after support scheduling."
            )
        return support_patterns[:num_positive]

    selected: List[Dict[str, int]] = []
    selected_keys: set[tuple[int, ...]] = set()
    selected_visible_keys: set[tuple[tuple[str, ...], ...]] = set()
    used_indices_by_label: Dict[str, set[int]] = {str(label): set() for label in labels}
    component_usage_by_label = {
        str(label): {component_index: 0 for component_index in range(component_count)}
        for label in labels
    }
    support_use_by_label = {str(label): Counter() for label in labels}

    def add_selected_pattern(pattern: Dict[str, int]) -> None:
        selected.append(dict(pattern))
        selected_keys.add(tuple(pattern[str(label)] for label in labels))
        selected_visible_keys.add(_support_pattern_visible_key(example, labels, pattern))
        for label in labels:
            label = str(label)
            support_index = int(pattern[label])
            used_indices_by_label[label].add(support_index)
            support_use_by_label[label][support_index] += 1
            for component_index in _support_component_indices(example, label, support_index):
                if int(component_index) in component_usage_by_label[label]:
                    component_usage_by_label[label][int(component_index)] += 1

    def projected_component_balance_score(pattern: Dict[str, int]) -> tuple[int, int, int, tuple[int, ...]]:
        projected_counts = {
            label: dict(counts)
            for label, counts in component_usage_by_label.items()
        }
        support_reuse = 0
        for label in labels:
            label = str(label)
            support_index = int(pattern[label])
            support_reuse += support_use_by_label[label][support_index]
            for component_index in _support_component_indices(example, label, support_index):
                if int(component_index) in projected_counts[label]:
                    projected_counts[label][int(component_index)] += 1
        max_gap = 0
        square_sum = 0
        for counts in projected_counts.values():
            values = [int(value) for value in counts.values()]
            if values:
                max_gap = max(max_gap, max(values) - min(values))
                square_sum += sum(value * value for value in values)
        return (
            max_gap,
            square_sum,
            support_reuse,
            tuple(int(pattern[str(label)]) for label in labels),
        )

    def least_used_support_index_for_component(label: str, component_index: int, tie_offset: int) -> int:
        candidates = [
            support_index
            for support_index in range(available_support_rounds)
            if component_index in _support_component_indices(example, label, support_index)
        ]
        if not candidates:
            return component_index % available_support_rounds
        lowest_use = min(support_use_by_label[label][support_index] for support_index in candidates)
        tied_candidates = [
            support_index
            for support_index in candidates
            if support_use_by_label[label][support_index] == lowest_use
        ]
        exact_tied_candidates = [
            support_index
            for support_index in tied_candidates
            if set(_support_component_indices(example, label, support_index)) == {component_index}
        ]
        if exact_tied_candidates:
            tied_candidates = exact_tied_candidates
        return tied_candidates[tie_offset % len(tied_candidates)]

    def max_component_usage_gap() -> int:
        max_gap = 0
        for counts in component_usage_by_label.values():
            values = [int(value) for value in counts.values()]
            if values:
                max_gap = max(max_gap, max(values) - min(values))
        return max_gap

    def reset_selected_patterns() -> None:
        selected.clear()
        selected_keys.clear()
        selected_visible_keys.clear()
        for label in labels:
            label = str(label)
            used_indices_by_label[label].clear()
            support_use_by_label[label].clear()
            for component_index in component_usage_by_label[label]:
                component_usage_by_label[label][component_index] = 0

    component_coverage_repeats = {
        1: min(num_positive, available_support_rounds),
        2: (num_positive + component_count - 1) // component_count,
        3: (num_positive + component_count - 1) // component_count,
    }.get(component_count, 3)
    for repeat_index in range(component_coverage_repeats):
        for component_index in range(component_count):
            if len(selected) >= num_positive:
                break
            pattern = {
                str(label): least_used_support_index_for_component(
                    str(label),
                    (component_index + repeat_index + label_offset) % component_count,
                    repeat_index + component_index + label_offset,
                )
                for label_offset, label in enumerate(labels)
            }
            key = tuple(pattern[str(label)] for label in labels)
            visible_key = _support_pattern_visible_key(example, labels, pattern)
            if key in selected_keys or visible_key in selected_visible_keys:
                continue
            add_selected_pattern(pattern)
        if len(selected) >= num_positive:
            break

    if len(selected) >= num_positive and max_component_usage_gap() > 4:
        reset_selected_patterns()

    support_patterns: List[Dict[str, int]] = []
    while len(selected) < num_positive:
        if not support_patterns:
            support_pattern_limit = min(
                max_support_patterns,
                max(num_positive + component_count + available_support_rounds, num_positive * 12),
            )
            support_patterns = _ordered_component_balanced_support_patterns(
                example,
                labels,
                available_support_rounds,
                limit=support_pattern_limit,
            )
        remaining_patterns = [
            pattern
            for pattern in support_patterns
            if tuple(pattern[str(label)] for label in labels) not in selected_keys
            and _support_pattern_visible_key(example, labels, pattern) not in selected_visible_keys
        ]
        if not remaining_patterns:
            break
        pattern = min(remaining_patterns, key=projected_component_balance_score)
        if len(selected) >= num_positive:
            break
        add_selected_pattern(pattern)

    if len(selected) < num_positive:
        raise ValueError(
            f"Only {len(selected)} distinct positive prompt demos are available after component coverage scheduling."
        )
    return selected


def _single_slot_intrusion_budget(difficulty: str, num_negative: int) -> int:
    if num_negative < 8:
        return 0
    if difficulty == "d1":
        return min(4, num_negative)
    if difficulty == "d2":
        return min(4, num_negative)
    return min(8, num_negative)


def _support_group_words(
    example: Dict[str, Any],
    label: str,
    support_index: int,
) -> List[str]:
    groups = (example.get("support_groups_by_label") or {}).get(label) or []
    if support_index >= len(groups):
        return []
    return [str(word).strip().lower() for word in groups[support_index]]


def _support_pattern_visible_key(
    example: Dict[str, Any],
    labels: Sequence[str],
    support_indices_by_label: Dict[str, int],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(_support_group_words(example, str(label), int(support_indices_by_label[str(label)])))
        for label in labels
    )


def _ordered_single_slot_intrusion_specs(
    example: Dict[str, Any],
    labels: Sequence[str],
    support_patterns: Sequence[Dict[str, int]],
    available_support_rounds: int,
    num_intrusions: int,
    *,
    start_index: int = 0,
) -> List[Dict[str, Any]]:
    if num_intrusions <= 0 or len(labels) < 2 or available_support_rounds < 2:
        return []

    labels = [str(label) for label in labels]
    label_offset = _stable_demo_label_offset(example, labels)
    specs: List[Dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    max_attempts = max(1, num_intrusions * len(labels) * (len(labels) - 1) * available_support_rounds * 2)

    for attempt in range(max_attempts):
        if len(specs) >= num_intrusions:
            break
        base_pattern = dict(support_patterns[(start_index + attempt) % len(support_patterns)])
        target_index = (label_offset + attempt) % len(labels)
        target_label = labels[target_index]
        source_shift = 1 + ((attempt // len(labels)) % (len(labels) - 1))
        source_label = labels[(target_index + source_shift) % len(labels)]
        source_base_index = int(base_pattern[source_label])

        intruder_index: int | None = None
        for delta in range(1, available_support_rounds + 1):
            candidate_index = (source_base_index + delta + attempt // max(1, len(labels))) % available_support_rounds
            if candidate_index == source_base_index:
                continue
            intruder_words = _support_group_words(example, source_label, candidate_index)
            if len(intruder_words) != 1:
                continue
            if _support_group_is_semantic_member_for_label(
                example,
                source_label,
                candidate_index,
                target_label,
            ):
                continue
            prompt_words: List[str] = []
            duplicate = False
            for label in labels:
                if label == target_label:
                    words = intruder_words
                else:
                    words = _support_group_words(example, label, int(base_pattern[label]))
                if len(words) != 1:
                    duplicate = True
                    break
                word = words[0]
                if word in prompt_words:
                    duplicate = True
                    break
                prompt_words.append(word)
            if duplicate:
                continue
            intruder_index = candidate_index
            break

        if intruder_index is None:
            continue

        key = (
            target_label,
            source_label,
            intruder_index,
            tuple(int(base_pattern[label]) for label in labels),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        specs.append(
            {
                "support_indices_by_label": base_pattern,
                "corruption_type": "single_slot_intrusion",
                "support_replacement_labels_by_display_label": {target_label: source_label},
                "support_replacement_indices_by_display_label": {target_label: intruder_index},
                "intrusion_target_label": target_label,
                "intruder_source_label": source_label,
            }
        )

    return specs


def _ordered_or_bridge_specs(
    example: Dict[str, Any],
    labels: Sequence[str],
    support_patterns: Sequence[Dict[str, int]],
    available_support_rounds: int,
    num_bridges: int,
) -> List[Dict[str, Any]]:
    component_count = _component_count(example)
    if (
        num_bridges <= 0
        or str(example.get("category_operator") or "") != "or"
        or component_count <= 1
        or not support_patterns
    ):
        return []

    labels = [str(label) for label in labels]
    label_offset = _stable_demo_label_offset(example, labels)
    specs: List[Dict[str, Any]] = []
    seen_positive_patterns: set[tuple[int, ...]] = set()
    seen_positive_visible_keys: set[tuple[tuple[str, ...], ...]] = set()
    used_support_by_label: Dict[str, set[int]] = {label: set() for label in labels}
    used_out_by_label: Dict[str, set[int]] = {label: set() for label in labels}

    target_slots: List[tuple[str, int]] = []
    for label_index in range(len(labels)):
        label = labels[(label_offset + label_index) % len(labels)]
        for component_index in range(component_count):
            target_slots.append((label, component_index))

    attempts = 0
    max_attempts = max(num_bridges * 8, len(target_slots) * max(1, len(support_patterns)) * 2)
    slot_index = 0
    while len(specs) < num_bridges and attempts < max_attempts:
        target_label, component_index = target_slots[slot_index % len(target_slots)]
        label_base_offset = labels.index(target_label) if target_label in labels else slot_index
        base_pattern = dict(support_patterns[(label_offset + label_base_offset) % len(support_patterns)])
        for other_offset, other_label in enumerate(labels):
            if other_label == target_label:
                continue
            other_target_order = [label for label in labels if label != other_label]
            fixed_component = other_target_order.index(target_label) % component_count
            base_pattern[other_label] = _first_support_index_for_component(
                example,
                other_label,
                fixed_component,
                used_support_by_label[other_label],
                available_support_rounds,
            )
        target_support_index = _first_support_index_for_component(
            example,
            target_label,
            component_index,
            used_support_by_label[target_label],
            available_support_rounds,
        )
        bridge_pattern = dict(base_pattern)
        bridge_pattern[target_label] = target_support_index
        pattern_key = tuple(int(bridge_pattern[label]) for label in labels)
        visible_key = _support_pattern_visible_key(example, labels, bridge_pattern)
        if pattern_key in seen_positive_patterns or visible_key in seen_positive_visible_keys:
            for alternate_index in range(available_support_rounds):
                if component_index not in _support_component_indices(example, target_label, alternate_index):
                    continue
                bridge_pattern[target_label] = alternate_index
                pattern_key = tuple(int(bridge_pattern[label]) for label in labels)
                visible_key = _support_pattern_visible_key(example, labels, bridge_pattern)
                if pattern_key not in seen_positive_patterns and visible_key not in seen_positive_visible_keys:
                    target_support_index = alternate_index
                    break

        if pattern_key not in seen_positive_patterns and visible_key not in seen_positive_visible_keys:
            out_index = _out_of_category_index_for_none_of_components(
                example,
                target_label,
                used_out_by_label[target_label],
            )
            seen_positive_patterns.add(pattern_key)
            seen_positive_visible_keys.add(visible_key)
            used_support_by_label[target_label].add(target_support_index)
            used_out_by_label[target_label].add(out_index)
            specs.append(
                {
                    "support_indices_by_label": dict(bridge_pattern),
                    "positive_support_indices_by_label": dict(bridge_pattern),
                    "replacement_indices_by_label": {target_label: out_index},
                    "corruption_type": "out_of_category",
                    "out_of_category_label": target_label,
                    "or_bridge_label": target_label,
                    "or_bridge_component_index": int(component_index),
                    "curriculum_role": "or_component_bridge",
                }
            )

        attempts += 1
        slot_index += 1

    return specs


def _ordered_negative_demo_specs(
    example: Dict[str, Any],
    labels: Sequence[str],
    difficulty: str,
    available_support_rounds: int,
    num_negative: int,
    support_patterns_override: Sequence[Dict[str, int]] | None = None,
) -> List[Dict[str, Any]]:
    negative_types = _demo_negative_schedule(difficulty)
    max_negative_demos = (available_support_rounds ** len(labels)) * len(negative_types)
    if num_negative > max_negative_demos:
        raise ValueError(
            f"Only {max_negative_demos} distinct negative prompt demos are available without "
            "reusing the same support-pattern/logic combination."
        )
    if support_patterns_override is not None:
        support_patterns = [dict(pattern) for pattern in support_patterns_override[:num_negative]]
        if len(support_patterns) < num_negative:
            raise ValueError(
                f"Only {len(support_patterns)} support patterns were provided for {num_negative} negative demos."
            )
    elif str(example.get("category_operator") or "") == "or":
        support_patterns = _ordered_positive_demo_supports(
            example,
            labels,
            available_support_rounds,
            num_negative,
        )
    else:
        support_patterns = _ordered_component_balanced_support_patterns(
            example,
            labels,
            available_support_rounds,
            limit=max(1, num_negative + _single_slot_intrusion_budget(difficulty, num_negative)),
        )

    ordered_types: List[str] = []
    if negative_types:
        ordered_types.extend(negative_types)
        harder_first = sorted(
            negative_types,
            key=lambda logic: (
                -_negative_demo_severity(logic)[0],
                -_negative_demo_severity(logic)[1],
                logic,
            ),
        )
        while len(ordered_types) < num_negative:
            ordered_types.extend(harder_first)

    specs: List[Dict[str, Any]] = []
    if str(example.get("category_operator") or "") == "or":
        bridge_budget = min(num_negative, len(labels) * _component_count(example)) if num_negative >= 8 else 0
        bridge_specs = _ordered_or_bridge_specs(
            example,
            labels,
            support_patterns,
            available_support_rounds,
            bridge_budget,
        )
        specs.extend(bridge_specs[:num_negative])

    if str(example.get("category_operator") or "") == "and":
        component_count = _component_count(example)
        available_near_misses: List[tuple[str, int, int]] = []
        failed_components = example.get("out_of_category_failed_component_indices_by_label") or {}
        satisfied_components = example.get("out_of_category_component_indices_by_label") or {}
        for label in labels:
            label = str(label)
            label_failed = failed_components.get(label) or []
            label_satisfied = satisfied_components.get(label) or []
            for support_index, indices in enumerate(label_failed):
                failed = {int(index) for index in (indices or [])}
                satisfied = {
                    int(index)
                    for index in (
                        label_satisfied[support_index]
                        if support_index < len(label_satisfied)
                        else []
                    )
                }
                if len(failed) != 1 or len(satisfied) != component_count - 1:
                    continue
                failed_component = next(iter(failed))
                if failed_component in satisfied:
                    continue
                available_near_misses.append((label, failed_component, support_index))
        # Pure-IO prompts need explicit AND boundary evidence, but v36 gives
        # some No-demo budget to slot intrusions. Keep at least two failed
        # examples per component when possible while rotating labels so no
        # displayed category is starved of near-miss evidence.
        near_miss_budget = min(
            num_negative,
            8,
        )
        if not available_near_misses:
            near_miss_budget = 0
        near_miss_label_counts: Counter[str] = Counter()
        near_miss_component_counts: Counter[int] = Counter()
        near_miss_candidate_counts: Counter[tuple[str, int, int]] = Counter()

        selected_near_misses: List[tuple[str, int, int]] = []
        available_by_label: Dict[str, List[tuple[str, int, int]]] = defaultdict(list)
        for candidate in available_near_misses:
            available_by_label[str(candidate[0])].append(candidate)

        def choose_near_miss_candidate(candidates: Sequence[tuple[str, int, int]]) -> tuple[str, int, int]:
            return min(
                candidates,
                key=lambda item: (
                    near_miss_component_counts[int(item[1])],
                    near_miss_label_counts[str(item[0])],
                    near_miss_candidate_counts[(str(item[0]), int(item[1]), int(item[2]))],
                    str(item[0]),
                    int(item[1]),
                    int(item[2]),
                ),
            )

        for label in labels:
            if len(selected_near_misses) >= near_miss_budget:
                break
            candidates = available_by_label.get(str(label)) or []
            if not candidates:
                continue
            candidate = choose_near_miss_candidate(candidates)
            selected_near_misses.append(candidate)
            near_miss_label_counts[str(candidate[0])] += 1
            near_miss_component_counts[int(candidate[1])] += 1
            near_miss_candidate_counts[(str(candidate[0]), int(candidate[1]), int(candidate[2]))] += 1

        while len(selected_near_misses) < near_miss_budget:
            candidate = min(
                available_near_misses,
                key=lambda item: (
                    near_miss_component_counts[int(item[1])],
                    near_miss_label_counts[str(item[0])],
                    near_miss_candidate_counts[(str(item[0]), int(item[1]), int(item[2]))],
                    str(item[0]),
                    int(item[1]),
                    int(item[2]),
                ),
            )
            selected_near_misses.append(candidate)
            near_miss_label_counts[str(candidate[0])] += 1
            near_miss_component_counts[int(candidate[1])] += 1
            near_miss_candidate_counts[(str(candidate[0]), int(candidate[1]), int(candidate[2]))] += 1

        for index, candidate in enumerate(selected_near_misses):
            label, component_index, near_miss_support_index = candidate
            support_indices_by_label = dict(
                support_patterns[index % len(support_patterns)]
            )
            specs.append(
                {
                    "support_indices_by_label": support_indices_by_label,
                    "replacement_indices_by_label": {label: near_miss_support_index},
                    "corruption_type": "component_near_miss",
                    "out_of_category_labels": [label],
                    "near_miss_component_index": component_index,
                    "near_miss_label": label,
                }
            )

    raw_intrusion_budget = _single_slot_intrusion_budget(difficulty, num_negative)
    if str(example.get("category_operator") or "") == "or" and difficulty == "d3":
        raw_intrusion_budget = min(raw_intrusion_budget, 4)
    if str(example.get("category_operator") or "") == "and" and difficulty == "d3" and num_negative == 24:
        raw_intrusion_budget = min(raw_intrusion_budget, 4)
    intrusion_budget = min(
        raw_intrusion_budget,
        max(0, num_negative - len(specs)),
    )
    intrusion_specs = _ordered_single_slot_intrusion_specs(
        example,
        labels,
        support_patterns,
        available_support_rounds,
        intrusion_budget,
        start_index=len(specs),
    )
    specs.extend(intrusion_specs[: max(0, num_negative - len(specs))])

    general_start = len(specs)
    out_of_category_label_offset = _stable_demo_label_offset(example, labels)
    out_of_category_count = 0
    operator = str(example.get("category_operator") or "")
    fill_types = ordered_types
    if operator == "and" and difficulty == "d3":
        if num_negative == 24:
            fill_base = (
                ["two_swap"] * 4
                + ["three_cycle"] * 4
                + ["four_cycle"] * 2
                + ["out_of_category"] * 2
            )
        else:
            fill_base = (
                ["two_swap"] * 8
                + ["three_cycle"] * 4
                + ["four_cycle"] * 4
                + ["out_of_category"] * 4
            )
        fill_types = []
        while len(fill_types) < num_negative:
            fill_types.extend(fill_base)
    elif operator == "and" and difficulty == "d2":
        if num_negative == 16:
            fill_base = ["two_swap"] * 2 + ["three_cycle"] + ["out_of_category"]
        else:
            fill_base = (
                ["two_swap"] * 6
                + ["three_cycle"] * 4
                + ["out_of_category"] * 4
            )
        fill_types = []
        while len(fill_types) < num_negative:
            fill_types.extend(fill_base)
    elif operator == "or" and difficulty == "d3" and num_negative == 24:
        fill_base = ["two_swap"] * 3 + ["three_cycle"] * 3 + ["four_cycle"] * 2
        fill_types = []
        while len(fill_types) < num_negative:
            fill_types.extend(fill_base)
    elif operator == "or" and difficulty == "d2" and num_negative == 16:
        fill_base = ["two_swap"] * 2 + ["three_cycle"] * 2
        fill_types = []
        while len(fill_types) < num_negative:
            fill_types.extend(fill_base)

    for index, corruption_type in enumerate(fill_types):
        if len(specs) >= num_negative:
            break
        support_indices_by_label = dict(support_patterns[(general_start + index) % len(support_patterns)])
        spec = {
            "support_indices_by_label": support_indices_by_label,
            "corruption_type": corruption_type,
        }
        if corruption_type == "out_of_category":
            spec["out_of_category_label"] = str(
                labels[(out_of_category_label_offset + out_of_category_count) % len(labels)]
            )
            out_of_category_count += 1
        specs.append(spec)
    return specs


def _build_demo_display_mapping(
    labels: Sequence[str],
    corruption_type: str,
    offset: int,
) -> Dict[str, str]:
    ordered = list(labels)
    if not ordered:
        return {}
    rotation = offset % len(ordered)
    ordered = ordered[rotation:] + ordered[:rotation]
    display_label_by_gold = {label: label for label in labels}

    if corruption_type in {"out_of_category", "component_near_miss", "single_slot_intrusion"}:
        pass
    elif corruption_type == "two_swap":
        display_label_by_gold[ordered[0]] = ordered[1]
        display_label_by_gold[ordered[1]] = ordered[0]
    elif corruption_type == "three_cycle":
        display_label_by_gold[ordered[0]] = ordered[1]
        display_label_by_gold[ordered[1]] = ordered[2]
        display_label_by_gold[ordered[2]] = ordered[0]
    elif corruption_type == "four_cycle":
        for index, label in enumerate(ordered):
            display_label_by_gold[label] = ordered[(index + 1) % len(ordered)]
    else:
        raise ValueError(f"Unsupported demo corruption type: {corruption_type}")

    return display_label_by_gold


def _display_mapping_has_semantic_cross_fit(
    example: Dict[str, Any],
    support_indices_by_label: Dict[str, int],
    display_label_by_gold: Dict[str, str],
) -> bool:
    for gold_label, display_label in display_label_by_gold.items():
        gold_label = str(gold_label)
        display_label = str(display_label)
        if gold_label == display_label:
            continue
        support_index = int(support_indices_by_label.get(gold_label, -1))
        if support_index < 0:
            continue
        if _support_group_is_semantic_member_for_label(
            example,
            gold_label,
            support_index,
            display_label,
        ):
            return True
    return False


def _build_semantically_valid_demo_display_mapping(
    example: Dict[str, Any],
    labels: Sequence[str],
    corruption_type: str,
    offset: int,
    support_indices_by_label: Dict[str, int],
) -> Dict[str, str]:
    if corruption_type not in {"two_swap", "three_cycle", "four_cycle"}:
        return _build_demo_display_mapping(labels, corruption_type, offset)

    label_count = max(1, len(labels))
    first_mapping: Dict[str, str] | None = None
    for offset_delta in range(label_count):
        mapping = _build_demo_display_mapping(labels, corruption_type, offset + offset_delta)
        if first_mapping is None:
            first_mapping = mapping
        if not _display_mapping_has_semantic_cross_fit(example, support_indices_by_label, mapping):
            return mapping
    return first_mapping or _build_demo_display_mapping(labels, corruption_type, offset)


def _find_semantically_safe_wrong_label_demo(
    example: Dict[str, Any],
    labels: Sequence[str],
    corruption_type: str,
    offset: int,
    support_indices_by_label: Dict[str, int],
    alternate_support_patterns: Sequence[Dict[str, int]],
    component_usage_by_label: Dict[str, Counter[int]] | None = None,
    avoid_positive_visible_keys: set[tuple[tuple[str, ...], ...]] | None = None,
) -> tuple[Dict[str, int], Dict[str, str], bool]:
    if corruption_type not in {"two_swap", "three_cycle", "four_cycle"}:
        return (
            dict(support_indices_by_label),
            _build_demo_display_mapping(labels, corruption_type, offset),
            True,
        )

    labels = [str(label) for label in labels]
    seen_patterns: set[tuple[int, ...]] = set()
    candidate_patterns = [dict(support_indices_by_label)]
    candidate_patterns.extend(dict(pattern) for pattern in alternate_support_patterns)

    label_count = max(1, len(labels))
    valid_candidates: List[tuple[Dict[str, int], Dict[str, str]]] = []
    for pattern in candidate_patterns:
        pattern_key = tuple(int(pattern.get(label, -1)) for label in labels)
        if pattern_key in seen_patterns:
            continue
        seen_patterns.add(pattern_key)
        for offset_delta in range(label_count):
            mapping = _build_demo_display_mapping(labels, corruption_type, offset + offset_delta)
            if not _display_mapping_has_semantic_cross_fit(example, pattern, mapping):
                valid_candidates.append((dict(pattern), mapping))
                break

    if valid_candidates:
        if avoid_positive_visible_keys:
            visible_safe_candidates = [
                candidate
                for candidate in valid_candidates
                if _support_pattern_visible_key(example, labels, candidate[0]) not in avoid_positive_visible_keys
            ]
            if visible_safe_candidates:
                valid_candidates = visible_safe_candidates
        if str(example.get("category_operator") or "") == "or" and component_usage_by_label is not None:
            def projected_balance_score(candidate: tuple[Dict[str, int], Dict[str, str]]) -> tuple[int, int, tuple[int, ...]]:
                pattern, _ = candidate
                projected = {
                    str(label): Counter(component_usage_by_label.get(str(label), Counter()))
                    for label in labels
                }
                for label in labels:
                    label = str(label)
                    for component_index in _support_component_indices(example, label, int(pattern[label])):
                        projected[label][int(component_index)] += 1
                max_gap = 0
                square_sum = 0
                for label in labels:
                    values = [
                        int(projected[str(label)].get(component_index, 0))
                        for component_index in range(_component_count(example))
                    ]
                    max_gap = max(max_gap, max(values) - min(values))
                    square_sum += sum(value * value for value in values)
                return (
                    max_gap,
                    square_sum,
                    tuple(int(pattern[str(label)]) for label in labels),
                )

            pattern, mapping = min(valid_candidates, key=projected_balance_score)
            return dict(pattern), mapping, True
        pattern, mapping = valid_candidates[0]
        return dict(pattern), mapping, True

    fallback_mapping = _build_demo_display_mapping(labels, corruption_type, offset)
    return dict(support_indices_by_label), fallback_mapping, False


def _build_demo_candidate_groups(
    example: Dict[str, Any],
    support_indices_by_label: Dict[str, int],
    display_label_by_gold: Dict[str, str],
    out_of_category_label: str | None = None,
    out_of_category_labels: Sequence[str] | None = None,
    replacement_indices_by_label: Dict[str, int] | None = None,
    support_replacement_labels_by_display_label: Dict[str, str] | None = None,
    support_replacement_indices_by_display_label: Dict[str, int] | None = None,
) -> List[Dict[str, Any]]:
    labels = list(example.get("labels") or [])
    support_groups = dict(example.get("support_groups_by_label") or {})
    out_of_category_groups = dict(example.get("out_of_category_groups_by_label") or {})
    out_label_set = {str(label) for label in (out_of_category_labels or [])}
    replacement_indices_by_label = {
        str(label): int(index)
        for label, index in (replacement_indices_by_label or {}).items()
    }
    support_replacement_labels_by_display_label = {
        str(label): str(source_label)
        for label, source_label in (support_replacement_labels_by_display_label or {}).items()
    }
    support_replacement_indices_by_display_label = {
        str(label): int(index)
        for label, index in (support_replacement_indices_by_display_label or {}).items()
    }
    if out_of_category_label is not None:
        out_label_set.add(str(out_of_category_label))
    gold_label_by_display_label = {
        str(display_label): str(gold_label)
        for gold_label, display_label in display_label_by_gold.items()
    }
    candidate_groups: List[Dict[str, Any]] = []
    for display_label in labels:
        display_label = str(display_label)
        mapped_gold_label = gold_label_by_display_label.get(display_label, display_label)
        support_label = support_replacement_labels_by_display_label.get(display_label, mapped_gold_label)
        groups = support_groups.get(support_label) or []
        support_index = int(
            support_replacement_indices_by_display_label.get(
                display_label,
                support_indices_by_label.get(support_label, support_indices_by_label[display_label]),
            )
        )
        if support_index >= len(groups):
            raise ValueError(
                f"Episode {example.get('episode_id') or example.get('id')} "
                f"does not have support group index {support_index} for {support_label}."
            )
        gold_label = str(support_label)
        words = [str(word).strip().lower() for word in groups[support_index]]
        if str(display_label) in out_label_set:
            out_groups = out_of_category_groups.get(display_label) or []
            replacement_index = int(replacement_indices_by_label.get(str(display_label), support_index))
            if replacement_index >= len(out_groups):
                raise ValueError(
                    f"Episode {example.get('episode_id') or example.get('id')} "
                    f"does not have out-of-category replacement group index {replacement_index} for {display_label}."
                )
            gold_label = OUT_OF_CATEGORY_LABEL
            words = [str(word).strip().lower() for word in out_groups[replacement_index]]
        if len(words) != 1:
            raise ValueError(
                f"Episode {example.get('episode_id') or example.get('id')} "
                f"demo group for {display_label} has {len(words)} words; examples-mode lists require singleton words."
            )
        candidate_groups.append(
            {
                "gold_label": gold_label,
                "display_label": str(display_label),
                "words": words,
            }
        )
    return candidate_groups


def _stable_demo_label_offset(example: Dict[str, Any], labels: Sequence[str]) -> int:
    if not labels:
        return 0

    stable_key = str(example.get("episode_id") or example.get("id") or "")
    match = re.search(r"(\d+)$", stable_key)
    if match:
        return int(match.group(1)) % len(labels)

    return sum((index + 1) * ord(char) for index, char in enumerate(stable_key)) % len(labels)


def _stable_demo_hash_int(*parts: Any) -> int:
    payload = "\u241f".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def compute_prompt_demo_order_stats(demos: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    answers = [str(demo.get("answer") or "") for demo in demos]
    yes_count = sum(1 for answer in answers if answer == "Yes")
    no_count = sum(1 for answer in answers if answer == "No")
    transition_count = sum(
        1 for previous, current in zip(answers, answers[1:]) if previous != current
    )
    max_same_answer_run = 0
    current_run = 0
    current_answer: str | None = None
    for answer in answers:
        if answer == current_answer:
            current_run += 1
        else:
            current_answer = answer
            current_run = 1
        max_same_answer_run = max(max_same_answer_run, current_run)

    answer_4grams = [
        "".join("Y" if answer == "Yes" else "N" for answer in answers[index : index + 4])
        for index in range(max(0, len(answers) - 3))
    ]
    answer_4gram_counts = Counter(answer_4grams)
    dominant_answer_4gram = None
    dominant_answer_4gram_count = 0
    if answer_4gram_counts:
        dominant_answer_4gram, dominant_answer_4gram_count = max(
            answer_4gram_counts.items(),
            key=lambda item: (item[1], item[0]),
        )

    transition_denominator = max(1, len(answers) - 1)
    return {
        "demo_count": len(answers),
        "yes_count": yes_count,
        "no_count": no_count,
        "first_answer": answers[0] if answers else None,
        "last_answer": answers[-1] if answers else None,
        "first_is_yes": bool(answers and answers[0] == "Yes"),
        "last_is_yes": bool(answers and answers[-1] == "Yes"),
        "transition_count": transition_count,
        "alternation_rate": transition_count / transition_denominator if answers else 0.0,
        "perfect_alternation": bool(
            len(answers) > 2
            and all(previous != current for previous, current in zip(answers, answers[1:]))
        ),
        "max_same_answer_run": max_same_answer_run,
        "dominant_answer_4gram": dominant_answer_4gram,
        "dominant_answer_4gram_count": dominant_answer_4gram_count,
        "dominant_answer_4gram_fraction": (
            dominant_answer_4gram_count / len(answer_4grams) if answer_4grams else 0.0
        ),
    }


def validate_prompt_demo_order_balance(demos: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    stats = compute_prompt_demo_order_stats(demos)
    failures: List[Dict[str, Any]] = []
    if stats["perfect_alternation"]:
        failures.append({"reason": "perfect_yes_no_alternation"})
    if float(stats["alternation_rate"]) > 0.65:
        failures.append(
            {
                "reason": "alternation_rate_too_high",
                "alternation_rate": stats["alternation_rate"],
            }
        )
    if int(stats["max_same_answer_run"]) > 3:
        failures.append(
            {
                "reason": "same_answer_run_too_long",
                "max_same_answer_run": stats["max_same_answer_run"],
            }
        )
    if int(stats["yes_count"]) != int(stats["no_count"]):
        failures.append(
            {
                "reason": "answer_counts_not_balanced",
                "yes_count": stats["yes_count"],
                "no_count": stats["no_count"],
            }
        )
    return {"passed": not failures, "stats": stats, "failures": failures}


def _order_prompt_demo_blocks_v36(
    example: Dict[str, Any],
    demo_blocks: Sequence[Dict[str, Dict[str, Any]]],
    unpaired_positive_demos: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    episode_key = str(example.get("episode_id") or example.get("id") or "")
    ordered_blocks = [
        dict(block)
        for block in sorted(
            demo_blocks,
            key=lambda block: _stable_demo_hash_int(
                episode_key,
                "v36_block_order",
                block.get("negative", {}).get("corruption_type"),
                block.get("negative", {}).get("minimal_pair_id"),
            ),
        )
    ]
    if not ordered_blocks:
        return [dict(demo) for demo in unpaired_positive_demos]

    transition_budget = max(1, len(ordered_blocks) // 6)
    transition_boundaries = {
        boundary_index
        for _, boundary_index in sorted(
            (
                _stable_demo_hash_int(episode_key, "v36_boundary_transition", boundary_index),
                boundary_index,
            )
            for boundary_index in range(1, len(ordered_blocks))
        )[:transition_budget]
    }
    first_negative_first = (
        _stable_demo_hash_int(episode_key, "v36_answer_order_start") % 2 == 0
    )
    ordered: List[Dict[str, Any]] = []

    def block_answers(negative_first: bool) -> tuple[str, str]:
        return ("No", "Yes") if negative_first else ("Yes", "No")

    current_run_answer: str | None = None
    current_run_length = 0

    for block_index, block in enumerate(ordered_blocks):
        if block_index == 0:
            negative_first = first_negative_first
        else:
            previous_answer = ordered[-1]["answer"]
            want_transition = block_index in transition_boundaries
            desired_first_answer = (
                "Yes" if previous_answer == "No" else "No"
            ) if want_transition else previous_answer
            negative_first = desired_first_answer == "No"
            first_answer, _ = block_answers(negative_first)
            if current_run_answer == first_answer and current_run_length >= 3:
                negative_first = not negative_first

        positive_demo = dict(block["positive"])
        negative_demo = dict(block["negative"])
        pair_order = "negative_first" if negative_first else "positive_first"
        positive_demo["pair_order"] = pair_order
        negative_demo["pair_order"] = pair_order
        positive_demo["pair_block_index"] = block_index
        negative_demo["pair_block_index"] = block_index
        if negative_first:
            block_rows = [negative_demo, positive_demo]
        else:
            block_rows = [positive_demo, negative_demo]

        for row in block_rows:
            answer = str(row.get("answer") or "")
            if answer == current_run_answer:
                current_run_length += 1
            else:
                current_run_answer = answer
                current_run_length = 1
            ordered.append(row)

    ordered.extend(dict(demo) for demo in unpaired_positive_demos)
    return ordered


def _build_prompt_demos(
    example: Dict[str, Any],
    num_demos: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if num_demos < 0:
        raise ValueError("num_demos must be non-negative for lexical_category_inference.")
    if num_demos == 0:
        return []

    labels = list(example.get("labels") or [])
    available_support_rounds = _available_support_rounds(example)
    out_of_category_offset = _stable_demo_label_offset(example, labels)
    target_no = num_demos // 2
    target_yes = num_demos - target_no
    difficulty = normalize_difficulty(str(example.get("difficulty") or "d2"))
    try:
        positive_supports = _ordered_positive_demo_supports(example, labels, available_support_rounds, target_yes)
    except ValueError as exc:
        raise ValueError(
            f"Episode {example.get('episode_id') or example.get('id')} {exc}"
        ) from exc

    try:
        negative_specs = _ordered_negative_demo_specs(
            example,
            labels,
            difficulty,
            available_support_rounds,
            target_no,
            support_patterns_override=positive_supports if str(example.get("category_operator") or "") == "or" else None,
        )
    except ValueError as exc:
        raise ValueError(
            f"Episode {example.get('episode_id') or example.get('id')} {exc}"
        ) from exc

    demo_blocks: List[Dict[str, Dict[str, Any]]] = []
    unpaired_positive_demos: List[Dict[str, Any]] = []
    alternate_wrong_label_support_patterns = _ordered_component_balanced_support_patterns(
        example,
        labels,
        available_support_rounds,
        limit=max(target_no * 4, target_no + available_support_rounds),
    )

    positive_mapping = {label: label for label in labels}
    positive_index = 0
    positive_component_usage_by_label: Dict[str, Counter[int]] = {
        str(label): Counter() for label in labels
    }
    used_positive_visible_keys: set[tuple[tuple[str, ...], ...]] = set()

    def build_positive_demo(
        support_indices_by_label: Dict[str, int],
        *,
        minimal_pair_id: str | None = None,
    ) -> Dict[str, Any]:
        nonlocal positive_index
        demo: Dict[str, Any] = {
            "candidate_groups": _build_demo_candidate_groups(example, support_indices_by_label, positive_mapping),
            "answer": "Yes",
            "corruption_type": "identity",
            "support_indices_by_label": dict(support_indices_by_label),
            "curriculum_role": "positive_boundary" if minimal_pair_id else "positive_support",
        }
        if minimal_pair_id:
            demo["minimal_pair_id"] = minimal_pair_id
        positive_index += 1
        used_positive_visible_keys.add(_support_pattern_visible_key(example, labels, support_indices_by_label))
        if str(example.get("category_operator") or "") == "or":
            for label in labels:
                label = str(label)
                support_index = int(support_indices_by_label[label])
                for component_index in _support_component_indices(example, label, support_index):
                    positive_component_usage_by_label[label][int(component_index)] += 1
        return demo

    for negative_index in range(target_no):
        negative_spec = negative_specs[negative_index]
        negative_support_indices_by_label = dict(negative_spec["support_indices_by_label"])
        positive_support_indices_by_label = dict(
            negative_spec.get("positive_support_indices_by_label") or negative_support_indices_by_label
        )
        replacement_indices_by_label = dict(negative_spec.get("replacement_indices_by_label") or {})
        support_replacement_labels_by_display_label = dict(
            negative_spec.get("support_replacement_labels_by_display_label") or {}
        )
        support_replacement_indices_by_display_label = dict(
            negative_spec.get("support_replacement_indices_by_display_label") or {}
        )
        corruption_type = str(negative_spec["corruption_type"])
        label_offset = out_of_category_offset + negative_index
        if corruption_type in {"two_swap", "three_cycle", "four_cycle"}:
            (
                negative_support_indices_by_label,
                current_mapping,
                found_semantically_safe_mapping,
            ) = _find_semantically_safe_wrong_label_demo(
                example,
                labels,
                corruption_type,
                label_offset,
                negative_support_indices_by_label,
                alternate_wrong_label_support_patterns,
                positive_component_usage_by_label,
                used_positive_visible_keys,
            )
            positive_support_indices_by_label = dict(negative_support_indices_by_label)
            if not found_semantically_safe_mapping:
                original_corruption_type = corruption_type
                corruption_type = "out_of_category"
            else:
                original_corruption_type = None
        else:
            original_corruption_type = None
        out_of_category_label = (
            str(negative_spec.get("out_of_category_label") or labels[label_offset % len(labels)])
            if corruption_type == "out_of_category"
            else None
        )
        if out_of_category_label is not None:
            out_groups = (example.get("out_of_category_groups_by_label") or {}).get(out_of_category_label) or []
            if out_groups and out_of_category_label not in replacement_indices_by_label:
                replacement_indices_by_label[out_of_category_label] = _out_of_category_index_for_none_of_components(
                    example,
                    out_of_category_label,
                    {int(label_offset % len(out_groups))},
                )
        minimal_pair_id = f"pair_{negative_index:02d}"
        positive_demo = None
        if positive_index < target_yes:
            positive_demo = build_positive_demo(
                dict(positive_support_indices_by_label),
                minimal_pair_id=minimal_pair_id,
            )
            if negative_spec.get("curriculum_role") == "or_component_bridge":
                positive_demo["curriculum_role"] = "or_component_bridge_positive"
                positive_demo["or_bridge_label"] = negative_spec.get("or_bridge_label")
                positive_demo["or_bridge_component_index"] = negative_spec.get("or_bridge_component_index")

        if corruption_type in {"two_swap", "three_cycle", "four_cycle"}:
            negative_mapping = current_mapping
        else:
            negative_mapping = _build_semantically_valid_demo_display_mapping(
                example,
                labels,
                corruption_type,
                offset=label_offset,
                support_indices_by_label=negative_support_indices_by_label,
            )
        if corruption_type == "component_near_miss":
            negative_mapping = {label: label for label in labels}
        negative_demo: Dict[str, Any] = {
            "candidate_groups": _build_demo_candidate_groups(
                example,
                negative_support_indices_by_label,
                negative_mapping,
                out_of_category_label=out_of_category_label,
                out_of_category_labels=negative_spec.get("out_of_category_labels"),
                replacement_indices_by_label=replacement_indices_by_label,
                support_replacement_labels_by_display_label=support_replacement_labels_by_display_label,
                support_replacement_indices_by_display_label=support_replacement_indices_by_display_label,
            ),
            "answer": "No",
            "corruption_type": corruption_type,
            "support_indices_by_label": dict(negative_support_indices_by_label),
            "replacement_indices_by_label": dict(replacement_indices_by_label),
            "support_replacement_labels_by_display_label": dict(support_replacement_labels_by_display_label),
            "support_replacement_indices_by_display_label": dict(support_replacement_indices_by_display_label),
            "minimal_pair_id": minimal_pair_id,
            "curriculum_role": "negative_boundary",
        }
        if original_corruption_type is not None:
            negative_demo["fallback_from_corruption_type"] = original_corruption_type
        if negative_spec.get("curriculum_role") == "or_component_bridge":
            negative_demo["curriculum_role"] = "or_component_bridge_negative"
            negative_demo["or_bridge_label"] = negative_spec.get("or_bridge_label")
            negative_demo["or_bridge_component_index"] = negative_spec.get("or_bridge_component_index")
        if "near_miss_component_index" in negative_spec:
            negative_demo["near_miss_component_index"] = negative_spec["near_miss_component_index"]
        if "near_miss_label" in negative_spec:
            negative_demo["near_miss_label"] = negative_spec["near_miss_label"]
        if "intrusion_target_label" in negative_spec:
            negative_demo["intrusion_target_label"] = negative_spec["intrusion_target_label"]
        if "intruder_source_label" in negative_spec:
            negative_demo["intruder_source_label"] = negative_spec["intruder_source_label"]
        if positive_demo is None:
            raise ValueError(
                f"Episode {example.get('episode_id') or example.get('id')} cannot build "
                "a paired positive demo for every negative demo."
            )
        demo_blocks.append(
            {
                "positive": positive_demo,
                "negative": negative_demo,
            }
        )

    for support_indices_by_label in positive_supports:
        if positive_index >= target_yes:
            break
        unpaired_positive_demos.append(build_positive_demo(support_indices_by_label))

    demos = _order_prompt_demo_blocks_v36(example, demo_blocks, unpaired_positive_demos)

    for curriculum_position, demo in enumerate(demos):
        demo["curriculum_position"] = curriculum_position

    if len(demos) > num_demos:
        demos = demos[:num_demos]
    if len(demos) != num_demos:
        raise ValueError(
            f"Episode {example.get('episode_id') or example.get('id')} built {len(demos)} demos; "
            f"expected {num_demos}."
        )

    _validate_prompt_demo_coverage(example, demos, num_demos)
    return demos


def compute_prompt_demo_coverage(
    example: Dict[str, Any],
    demos: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    labels = [str(label) for label in (example.get("labels") or [])]
    component_count = _component_count(example)
    per_label_positive_component_counts = {
        label: {str(component_index): 0 for component_index in range(component_count)}
        for label in labels
    }
    per_label_negative_component_near_miss_counts = {
        label: {f"fails_{component_index}": 0 for component_index in range(component_count)}
        for label in labels
    }
    unique_positive_words_by_label: Dict[str, set[str]] = {label: set() for label in labels}
    unique_negative_words_by_label: Dict[str, set[str]] = {label: set() for label in labels}
    negative_corruption_counts: Dict[str, int] = defaultdict(int)
    minimal_pair_ids: set[str] = set()

    yes_count = 0
    no_count = 0
    for demo in demos:
        minimal_pair_id = demo.get("minimal_pair_id")
        if minimal_pair_id:
            minimal_pair_ids.add(str(minimal_pair_id))
        answer = str(demo.get("answer") or "")
        if answer == "Yes":
            yes_count += 1
            support_indices = demo.get("support_indices_by_label") or {}
            for label in labels:
                support_index = int(support_indices.get(label, 0))
                for component_index in _support_component_indices(example, label, support_index):
                    per_label_positive_component_counts[label][str(component_index)] += 1
            for group in demo.get("candidate_groups") or []:
                display_label = str(group.get("display_label") or "")
                if display_label in unique_positive_words_by_label:
                    unique_positive_words_by_label[display_label].update(
                        str(word).strip().lower()
                        for word in (group.get("words") or [])
                    )
        elif answer == "No":
            no_count += 1
            corruption_type = str(demo.get("corruption_type") or "")
            negative_corruption_counts[corruption_type] += 1
            for group in demo.get("candidate_groups") or []:
                display_label = str(group.get("display_label") or "")
                gold_label = str(group.get("gold_label") or "")
                if display_label in unique_negative_words_by_label and gold_label != display_label:
                    unique_negative_words_by_label[display_label].update(
                        str(word).strip().lower()
                        for word in (group.get("words") or [])
                    )
            if corruption_type == "component_near_miss":
                label = str(demo.get("near_miss_label") or "")
                failed_component = demo.get("near_miss_component_index")
                if label in per_label_negative_component_near_miss_counts and failed_component is not None:
                    key = f"fails_{int(failed_component)}"
                    if key in per_label_negative_component_near_miss_counts[label]:
                        per_label_negative_component_near_miss_counts[label][key] += 1

    return {
        "yes_count": yes_count,
        "no_count": no_count,
        "per_label_positive_component_counts": per_label_positive_component_counts,
        "per_label_negative_component_near_miss_counts": per_label_negative_component_near_miss_counts,
        "unique_positive_word_counts_by_label": {
            label: len(words) for label, words in sorted(unique_positive_words_by_label.items())
        },
        "unique_negative_word_counts_by_label": {
            label: len(words) for label, words in sorted(unique_negative_words_by_label.items())
        },
        "minimal_pair_count": len(minimal_pair_ids),
        "negative_corruption_counts": dict(sorted(negative_corruption_counts.items())),
        "order_stats": compute_prompt_demo_order_stats(demos),
    }


def _demo_words_by_display_label(demo: Dict[str, Any]) -> Dict[str, tuple[str, ...]]:
    return {
        str(group.get("display_label") or ""): tuple(
            str(word).strip().lower() for word in (group.get("words") or [])
        )
        for group in (demo.get("candidate_groups") or [])
    }


def compute_prompt_demo_pair_isolation_stats(
    demos: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    expected_hamming_by_corruption = {
        "out_of_category": 1,
        "component_near_miss": 1,
        "single_slot_intrusion": 1,
        "two_swap": 2,
        "three_cycle": 3,
        "four_cycle": 4,
    }
    by_pair: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for demo in demos:
        pair_id = demo.get("minimal_pair_id")
        if pair_id:
            by_pair[str(pair_id)].append(demo)

    failures: List[Dict[str, Any]] = []
    checked_pairs = 0
    hamming_counts: Dict[str, Counter[int]] = defaultdict(Counter)
    for pair_id, pair_demos in sorted(by_pair.items()):
        positive = [demo for demo in pair_demos if str(demo.get("answer") or "") == "Yes"]
        negative = [demo for demo in pair_demos if str(demo.get("answer") or "") == "No"]
        if len(positive) != 1 or len(negative) != 1:
            failures.append(
                {
                    "minimal_pair_id": pair_id,
                    "reason": "pair_does_not_have_one_yes_and_one_no",
                    "yes_count": len(positive),
                    "no_count": len(negative),
                }
            )
            continue
        checked_pairs += 1
        positive_words = _demo_words_by_display_label(positive[0])
        negative_words = _demo_words_by_display_label(negative[0])
        labels = sorted(set(positive_words) | set(negative_words))
        hamming_distance = sum(
            1 for label in labels if positive_words.get(label) != negative_words.get(label)
        )
        corruption_type = str(negative[0].get("corruption_type") or "")
        hamming_counts[corruption_type][hamming_distance] += 1
        expected = expected_hamming_by_corruption.get(corruption_type)
        if expected is not None and hamming_distance != expected:
            failures.append(
                {
                    "minimal_pair_id": pair_id,
                    "reason": "pair_hamming_distance_mismatch",
                    "corruption_type": corruption_type,
                    "hamming_distance": hamming_distance,
                    "expected_hamming_distance": expected,
                }
            )

    return {
        "passed": not failures,
        "checked_pair_count": checked_pairs,
        "failure_count": len(failures),
        "hamming_counts_by_corruption_type": {
            corruption_type: {str(distance): count for distance, count in sorted(counts.items())}
            for corruption_type, counts in sorted(hamming_counts.items())
        },
        "failures": failures,
    }


def compute_prompt_demo_or_positive_bridge_stats(
    example: Dict[str, Any],
    demos: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    labels = [str(label) for label in (example.get("labels") or [])]
    component_count = _component_count(example)
    if str(example.get("category_operator") or "") != "or" or component_count <= 1:
        return {
            "passed": True,
            "checked_label_count": 0,
            "failure_count": 0,
            "failures": [],
            "bridges_by_label": {},
        }

    expected_components = set(range(component_count))
    contexts_by_label: Dict[str, Dict[tuple[tuple[str, tuple[str, ...]], ...], Dict[str, Any]]] = {
        label: defaultdict(lambda: {"positive_components": set(), "none_negative_count": 0})
        for label in labels
    }

    for demo in demos:
        words_by_label = _demo_words_by_display_label(demo)
        answer = str(demo.get("answer") or "")
        for target_label in labels:
            if target_label not in words_by_label:
                continue
            context_key = tuple(
                (label, words_by_label.get(label, tuple()))
                for label in labels
                if label != target_label
            )
            if answer == "Yes":
                support_indices = demo.get("support_indices_by_label") or {}
                support_index = int(support_indices.get(target_label, -1))
                if support_index >= 0:
                    contexts_by_label[target_label][context_key]["positive_components"].update(
                        _support_component_indices(example, target_label, support_index)
                    )
            elif answer == "No":
                for group in demo.get("candidate_groups") or []:
                    if (
                        str(group.get("display_label") or "") == target_label
                        and str(group.get("gold_label") or "") == OUT_OF_CATEGORY_LABEL
                    ):
                        contexts_by_label[target_label][context_key]["none_negative_count"] += 1

    failures: List[Dict[str, Any]] = []
    bridges_by_label: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        best_context = None
        best_components: set[int] = set()
        best_negative_count = 0
        for context_key, stats in contexts_by_label[label].items():
            components = {int(index) for index in stats["positive_components"]}
            negative_count = int(stats["none_negative_count"])
            if (
                len(components),
                negative_count,
            ) > (
                len(best_components),
                best_negative_count,
            ):
                best_context = context_key
                best_components = components
                best_negative_count = negative_count
        bridges_by_label[label] = {
            "components_seen_in_best_bridge": sorted(best_components),
            "none_of_components_negative_count": best_negative_count,
        }
        if best_components != expected_components:
            failures.append(
                {
                    "label": label,
                    "reason": "or_label_lacks_same_slot_positive_component_bridge",
                    "component_count": component_count,
                    "components_seen_in_best_bridge": sorted(best_components),
                    "none_of_components_negative_count": best_negative_count,
                }
            )
        elif best_negative_count < 1:
            failures.append(
                {
                    "label": label,
                    "reason": "or_label_lacks_same_slot_none_of_components_bridge_negative",
                    "component_count": component_count,
                    "components_seen_in_best_bridge": sorted(best_components),
                    "none_of_components_negative_count": best_negative_count,
                }
            )

    return {
        "passed": not failures,
        "checked_label_count": len(labels),
        "failure_count": len(failures),
        "failures": failures,
        "bridges_by_label": bridges_by_label,
    }


def _find_group_index(groups: Sequence[Sequence[Any]], words: Sequence[Any]) -> int | None:
    normalized_words = [str(word).strip().lower() for word in words]
    for index, group in enumerate(groups):
        if [str(word).strip().lower() for word in group] == normalized_words:
            return index
    return None


def validate_prompt_demo_semantics(
    example: Dict[str, Any],
    demos: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    labels = [str(label) for label in (example.get("labels") or [])]
    component_count = _component_count(example)
    expected_components = set(range(component_count))
    operator = str(example.get("category_operator") or "")
    support_groups = example.get("support_groups_by_label") or {}
    support_components = example.get("support_component_indices_by_label") or {}
    out_groups = example.get("out_of_category_groups_by_label") or {}
    out_satisfied = example.get("out_of_category_component_indices_by_label") or {}
    out_failed = example.get("out_of_category_failed_component_indices_by_label") or {}
    gold_groups = example.get("gold_groups") or {}
    failures: List[Dict[str, Any]] = []
    answer_by_visible_list: Dict[tuple[str, ...], str] = {}

    def component_profile_for_out_group(label: str, words: Sequence[Any]) -> tuple[set[int], set[int], set[int], int | None]:
        group_index = _find_group_index(out_groups.get(label) or [], words)
        if group_index is None:
            return set(), set(), expected_components, None
        satisfied = {
            int(index)
            for index in (
                (out_satisfied.get(label) or [])[group_index]
                if group_index < len(out_satisfied.get(label) or [])
                else []
            )
        }
        failed = {
            int(index)
            for index in (
                (out_failed.get(label) or [])[group_index]
                if group_index < len(out_failed.get(label) or [])
                else []
            )
        }
        unknown = expected_components - satisfied - failed
        return satisfied, failed, unknown, group_index

    def support_component_profile(label: str, words: Sequence[Any]) -> tuple[set[int], int | None]:
        group_index = _find_group_index(support_groups.get(label) or [], words)
        if group_index is None:
            if [str(word).strip().lower() for word in words] == [
                str(word).strip().lower() for word in (gold_groups.get(label) or [])
            ]:
                return set(expected_components), -1
            return set(), None
        components = {
            int(index)
            for index in (
                (support_components.get(label) or [])[group_index]
                if group_index < len(support_components.get(label) or [])
                else []
            )
        }
        return components, group_index

    for demo_index, demo in enumerate(demos):
        answer = str(demo.get("answer") or "")
        corruption_type = str(demo.get("corruption_type") or "")
        near_miss_label = str(demo.get("near_miss_label") or "")
        near_miss_component = demo.get("near_miss_component_index")
        intrusion_target_label = str(demo.get("intrusion_target_label") or "")
        intruder_source_label = str(demo.get("intruder_source_label") or "")
        out_of_category_groups = [
            group
            for group in (demo.get("candidate_groups") or [])
            if str(group.get("gold_label") or "") == OUT_OF_CATEGORY_LABEL
        ]
        wrong_label_groups = [
            group
            for group in (demo.get("candidate_groups") or [])
            if str(group.get("gold_label") or "") not in {str(group.get("display_label") or ""), OUT_OF_CATEGORY_LABEL}
        ]
        visible_list = tuple(
            word
            for group in (demo.get("candidate_groups") or [])
            for word in [str((group.get("words") or [""])[0]).strip().lower()]
        )
        if visible_list:
            previous_answer = answer_by_visible_list.get(visible_list)
            if previous_answer is not None and previous_answer != answer:
                failures.append(
                    {
                        "demo_index": demo_index,
                        "reason": "visible_list_has_conflicting_answers",
                        "visible_list": list(visible_list),
                        "previous_answer": previous_answer,
                        "answer": answer,
                    }
                )
            else:
                answer_by_visible_list[visible_list] = answer
        if corruption_type == "single_slot_intrusion":
            prompt_words = [
                str(word).strip().lower()
                for group in (demo.get("candidate_groups") or [])
                for word in (group.get("words") or [])
            ]
            if len(wrong_label_groups) != 1:
                failures.append(
                    {
                        "demo_index": demo_index,
                        "reason": "single_slot_intrusion_wrong_slot_count",
                        "wrong_slot_count": len(wrong_label_groups),
                    }
                )
            if len(prompt_words) != len(set(prompt_words)):
                failures.append(
                    {
                        "demo_index": demo_index,
                        "reason": "single_slot_intrusion_duplicate_words",
                        "words": prompt_words,
                    }
                )
        for group in demo.get("candidate_groups") or []:
            display_label = str(group.get("display_label") or "")
            gold_label = str(group.get("gold_label") or "")
            words = [str(word).strip().lower() for word in (group.get("words") or [])]
            if display_label not in labels:
                failures.append(
                    {
                        "demo_index": demo_index,
                        "reason": "unknown_display_label",
                        "display_label": display_label,
                        "words": words,
                    }
                )
                continue
            if answer == "Yes":
                components, group_index = support_component_profile(display_label, words)
                if gold_label != display_label or group_index is None:
                    failures.append(
                        {
                            "demo_index": demo_index,
                            "reason": "yes_demo_item_not_display_label_support",
                            "display_label": display_label,
                            "gold_label": gold_label,
                            "words": words,
                        }
                    )
                    continue
                if operator == "and" and components != expected_components:
                    failures.append(
                        {
                            "demo_index": demo_index,
                            "reason": "yes_demo_and_item_not_full_member",
                            "display_label": display_label,
                            "words": words,
                            "components": sorted(components),
                        }
                    )
                elif operator == "or" and not components:
                    failures.append(
                        {
                            "demo_index": demo_index,
                            "reason": "yes_demo_or_item_has_no_component",
                            "display_label": display_label,
                            "words": words,
                        }
                    )
            elif gold_label == OUT_OF_CATEGORY_LABEL:
                satisfied, failed, unknown, group_index = component_profile_for_out_group(display_label, words)
                if group_index is None:
                    failures.append(
                        {
                            "demo_index": demo_index,
                            "reason": "out_of_category_demo_word_not_in_pool",
                            "display_label": display_label,
                            "words": words,
                        }
                    )
                    continue
                if corruption_type == "component_near_miss":
                    expected_failed = {int(near_miss_component)} if near_miss_component is not None else set()
                    if (
                        display_label != near_miss_label
                        or len(out_of_category_groups) != 1
                        or satisfied != expected_components - expected_failed
                        or failed != expected_failed
                        or unknown
                    ):
                        failures.append(
                            {
                                "demo_index": demo_index,
                                "reason": "invalid_component_near_miss_demo",
                                "display_label": display_label,
                                "near_miss_label": near_miss_label,
                                "words": words,
                                "satisfied_component_indices": sorted(satisfied),
                                "failed_component_indices": sorted(failed),
                                "unknown_component_indices": sorted(unknown),
                                "expected_failed_component_indices": sorted(expected_failed),
                            }
                        )
                elif corruption_type == "out_of_category" and operator == "or":
                    if satisfied or failed != expected_components or unknown:
                        failures.append(
                            {
                                "demo_index": demo_index,
                                "reason": "invalid_or_none_of_components_demo",
                                "display_label": display_label,
                                "words": words,
                                "satisfied_component_indices": sorted(satisfied),
                                "failed_component_indices": sorted(failed),
                                "unknown_component_indices": sorted(unknown),
                            }
                        )
            elif answer == "No" and gold_label != display_label:
                source_components, source_group_index = support_component_profile(gold_label, words)
                if corruption_type == "single_slot_intrusion":
                    if display_label != intrusion_target_label or gold_label != intruder_source_label:
                        failures.append(
                            {
                                "demo_index": demo_index,
                                "reason": "single_slot_intrusion_metadata_mismatch",
                                "display_label": display_label,
                                "gold_label": gold_label,
                                "intrusion_target_label": intrusion_target_label,
                                "intruder_source_label": intruder_source_label,
                                "words": words,
                            }
                        )
                    source_is_member = (
                        source_group_index is not None
                        and (
                            (operator == "and" and source_components == expected_components)
                            or (operator == "or" and bool(source_components))
                            or (operator == "single" and bool(source_components))
                        )
                    )
                    if not source_is_member:
                        failures.append(
                            {
                                "demo_index": demo_index,
                                "reason": "single_slot_intruder_not_source_member",
                                "display_label": display_label,
                                "gold_label": gold_label,
                                "words": words,
                            }
                        )
                components, group_index = support_component_profile(display_label, words)
                if group_index is not None and (
                    (operator == "and" and components == expected_components)
                    or (operator == "or" and components)
                    or (operator == "single" and components)
                ):
                    failures.append(
                        {
                            "demo_index": demo_index,
                            "reason": "wrong_label_demo_item_is_display_label_support",
                            "display_label": display_label,
                            "gold_label": gold_label,
                            "words": words,
                        }
                    )
                if (
                    source_group_index is not None
                    and int(source_group_index) >= 0
                    and _support_group_is_semantic_member_for_label(
                        example,
                        gold_label,
                        int(source_group_index),
                        display_label,
                    )
                ):
                    failures.append(
                        {
                            "demo_index": demo_index,
                            "reason": "wrong_label_demo_item_is_semantic_display_label_member",
                            "display_label": display_label,
                            "gold_label": gold_label,
                            "words": words,
                            "source_support_index": int(source_group_index),
                            "implied_target_component_indices": sorted(
                                _support_implied_target_components(
                                    example,
                                    gold_label,
                                    int(source_group_index),
                                    display_label,
                                )
                            ),
                        }
                    )

    return {
        "passed": not failures,
        "invalid_demo_count": len({failure["demo_index"] for failure in failures}),
        "failure_count": len(failures),
        "failures": failures,
    }


def _validate_prompt_demo_coverage(
    example: Dict[str, Any],
    demos: Sequence[Dict[str, Any]],
    num_demos: int,
) -> None:
    difficulty = normalize_difficulty(str(example.get("difficulty") or "d2"))
    default_num_demos = get_default_num_demos(difficulty, str(example.get("logic_condition") or ""))
    if num_demos < default_num_demos:
        return

    coverage = compute_prompt_demo_coverage(example, demos)
    semantic_report = validate_prompt_demo_semantics(example, demos)
    if not semantic_report["passed"]:
        raise ValueError(
            f"Episode {example.get('episode_id') or example.get('id')} examples demos "
            f"failed semantic validation: {semantic_report['failures'][:3]}"
        )
    order_report = validate_prompt_demo_order_balance(demos)
    if not order_report["passed"]:
        raise ValueError(
            f"Episode {example.get('episode_id') or example.get('id')} examples demos "
            f"failed answer-order validation: {order_report['failures'][:3]}"
        )
    pair_report = compute_prompt_demo_pair_isolation_stats(demos)
    if not pair_report["passed"]:
        raise ValueError(
            f"Episode {example.get('episode_id') or example.get('id')} examples demos "
            f"failed pair-isolation validation: {pair_report['failures'][:3]}"
        )
    operator = str(example.get("category_operator") or "")
    if operator == "or":
        bridge_report = compute_prompt_demo_or_positive_bridge_stats(example, demos)
        if not bridge_report["passed"]:
            raise ValueError(
                f"Episode {example.get('episode_id') or example.get('id')} examples demos "
                f"failed OR positive bridge validation: {bridge_report['failures'][:3]}"
            )
        base_minimum = 4 if difficulty == "d2" else 3 if difficulty == "d3" else 1
        positive_count = int(coverage.get("yes_count") or 0)
        custom_demo_count = num_demos > default_num_demos
        for label, counts in coverage["per_label_positive_component_counts"].items():
            component_values = [int(count) for count in counts.values()]
            if (
                not custom_demo_count
                and component_values
                and max(component_values) - min(component_values) > 6
            ):
                raise ValueError(
                    f"Episode {example.get('episode_id') or example.get('id')} examples demos "
                    f"have imbalanced positive component exposure for {label}: {counts}."
                )
            minimum = base_minimum
            if custom_demo_count and component_values:
                denominator = max(1, 4 * len(component_values))
                scaled_minimum = (positive_count + denominator - 1) // denominator
                minimum = max(base_minimum, scaled_minimum)
            for component_index, count in counts.items():
                if int(count) < minimum:
                    raise ValueError(
                        f"Episode {example.get('episode_id') or example.get('id')} examples demos "
                        f"show only {count} positives for {label} component {component_index}; "
                        f"need at least {minimum}."
                    )
    elif operator == "and" and difficulty in {"d2", "d3"}:
        component_totals = {
            f"fails_{component_index}": 0
            for component_index in range(_component_count(example))
        }
        for counts in coverage["per_label_negative_component_near_miss_counts"].values():
            for component_key, count in counts.items():
                component_totals[component_key] = component_totals.get(component_key, 0) + int(count)
        for component_key, count in component_totals.items():
            if int(count) < 1:
                raise ValueError(
                    f"Episode {example.get('episode_id') or example.get('id')} examples demos "
                    f"do not include any AND near-miss that {component_key}."
                )
        if int(coverage["negative_corruption_counts"].get("component_near_miss", 0)) > 8:
            raise ValueError(
                f"Episode {example.get('episode_id') or example.get('id')} examples demos "
                "overweight AND near-miss negatives."
            )


def _coerce_content(raw_response: Any) -> str:
    if isinstance(raw_response, dict):
        return str(raw_response.get("content") or "")
    if raw_response is None:
        return ""
    return str(raw_response)


async def run_lexical_category_inference_experiment(
    difficulty: str,
    mode: str,
    num_test_samples: int,
    num_demos: Optional[int],
    output_dir: str,
    model_name: str,
    logic_condition: Optional[str] = None,
    seed: int = 123,
    max_tokens: int = 2,
    query_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    query_batch_fn: Optional[Callable[[List[str]], Awaitable[List[str]]]] = None,
    query_batch_size: Optional[int] = None,
    max_concurrent: int = 1,
) -> Dict[str, Any]:
    if mode not in {"rules", "examples", "combined"}:
        raise ValueError(f"Unsupported mode '{mode}'. Choose from rules/examples/combined.")

    logic, level, design_cell = resolve_design_cell(difficulty, logic_condition)
    effective_num_demos = (
        num_demos
        if num_demos is not None
        else get_default_num_demos(level, logic)
    )
    output_num_demos = effective_num_demos if mode in {"examples", "combined"} else 0
    output_demo_schedule = (
        "none"
        if mode not in {"examples", "combined"}
        else get_demo_schedule_for_num_demos(level, int(effective_num_demos), logic)
    )
    filename_schedule_suffix = (
        f"_{output_demo_schedule}"
        if output_demo_schedule not in {"none", "custom"}
        else ""
    )

    rng = random.Random(seed)
    test_pool = (
        load_lexical_category_inference_exact_split(
            level,
            "test",
            num_test_samples,
            logic_condition=logic,
        )
        if seed == 123 and num_test_samples == 600
        else None
    )
    if test_pool is None:
        test_pool = load_lexical_category_inference_split(level, "test", logic_condition=logic)
    if not test_pool:
        raise ValueError(f"No test examples found for lexical_category_inference cell {logic}/{level}.")

    selected_tests = sample_balanced_test_cases(test_pool, num_test_samples, rng, seed=seed)

    safe_model_name = model_name.replace("/", "_")
    task_output_dir = os.path.join(output_dir, "lexical_category_inference", logic, level, mode, safe_model_name)
    os.makedirs(task_output_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        task_output_dir,
        f"lexical_category_inference_{logic}_{level}_{mode}_demos{output_num_demos}"
        f"{filename_schedule_suffix}_{safe_model_name}_checkpoint.json",
    )
    snapshot_path = os.path.join(
        task_output_dir,
        f"lexical_category_inference_{logic}_{level}_{mode}_demos{output_num_demos}"
        f"{filename_schedule_suffix}_{safe_model_name}_latest.json",
    )

    resume_enabled = is_resume_enabled()
    checkpoint_every = get_checkpoint_every()
    snapshot_every = get_snapshot_every(checkpoint_every)
    run_signature = {
        "task": "lexical_category_inference",
        "logic_condition": logic,
        "difficulty": level,
        "category_operator": design_cell["category_operator"],
        "category_arity": design_cell["category_arity"],
        "shared_d1": design_cell["shared_d1"],
        "mode": mode,
        "model_name": model_name,
        "prompt_demo_semantics": PROMPT_DEMO_SEMANTICS,
        "num_test_samples": len(selected_tests),
        "num_demos": output_num_demos,
        "demo_schedule": output_demo_schedule,
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
            f"[checkpoint] Resuming lexical_category_inference from {start_index}/{len(requests)} "
            f"using {checkpoint_path}"
        )
    else:
        for example in selected_tests:
            support_groups_shown: Dict[str, List[List[str]]] = {}
            prompt_demos: List[Dict[str, Any]] = []
            prompt_demo_coverage: Dict[str, Any] = {}
            if mode in {"examples", "combined"}:
                prompt_demos = _build_prompt_demos(example, effective_num_demos, rng)
                support_groups_shown = _resolve_support_groups_shown(example, prompt_demos)
                prompt_demo_coverage = compute_prompt_demo_coverage(example, prompt_demos)

            if mode == "rules":
                prompt = get_rule_based_prompt(example)
            elif mode == "examples":
                prompt = get_example_based_prompt(example, prompt_demos)
            else:
                prompt = get_combined_prompt(example, prompt_demos)

            requests.append(
                {
                    "example": example,
                    "prompt": prompt,
                    "support_groups_shown": support_groups_shown,
                    "prompt_demos": prompt_demos,
                    "prompt_demo_coverage": prompt_demo_coverage,
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
        tqdm(total=len(requests), initial=start_index, desc=f"lexical_category_inference {logic}/{level} {mode}", unit="sample")
        if use_tqdm
        else None
    )

    def _finalize_one(idx: int, request: Dict[str, Any], raw_response: Any) -> None:
        del idx
        example = request["example"]
        content = _coerce_content(raw_response)
        parsed_exact = extract_exact_yes_no(content)
        parsed = extract_first_answer_token(content)
        predicted_answer = parsed.answer
        expected_answer = str(example.get("answer") or "")
        parse_success = bool(predicted_answer)

        result_entry = {
            "id": example["id"],
            "episode_id": example["episode_id"],
            "logic_condition": logic,
            "difficulty": level,
            "difficulty_name": DIFFICULTY_NAME[level],
            "category_operator": design_cell["category_operator"],
            "category_arity": int(design_cell["category_arity"]),
            "condition_count": int(example.get("condition_count") or design_cell["category_arity"]),
            "shared_d1": bool(design_cell["shared_d1"]),
            "design_cell": f"{logic}/{level}",
            "labels": list(example.get("labels") or []),
            "bundle_ids": dict(example.get("bundle_ids") or {}),
            "candidate_groups": list(example.get("candidate_groups") or []),
            "expected_answer": expected_answer,
            "corruption_type": example.get("corruption_type"),
            "negative_semantic_subtype": example.get("negative_semantic_subtype"),
            "target_label": example.get("target_label"),
            "failed_component_indices": list(example.get("failed_component_indices") or []),
            "satisfied_component_indices": list(example.get("satisfied_component_indices") or []),
            "semantic_validation_flags": dict(example.get("semantic_validation_flags") or {}),
            "num_wrong_labels": int(example.get("num_wrong_labels", 0)),
            "uses_all_labels_once": bool(example.get("uses_all_labels_once")),
            "candidate_confusability_score": float(example.get("candidate_confusability_score", 0.0)),
            "num_out_of_category_items": int(example.get("num_out_of_category_items", 0)),
            "knowledge_types": dict(example.get("knowledge_types") or {}),
            "category_family_by_label": dict(example.get("category_family_by_label") or {}),
            "category_families_by_label": dict(example.get("category_families_by_label") or {}),
            "difficulty_features": dict(example.get("difficulty_features") or {}),
            "candidate_features": dict(example.get("candidate_features") or {}),
            "rule_glosses": dict(example.get("rule_glosses") or {}),
            "support_groups_shown": request["support_groups_shown"] if mode in {"examples", "combined"} else {},
            "out_of_category_groups_by_label": dict(example.get("out_of_category_groups_by_label") or {}) if mode in {"examples", "combined"} else {},
            "prompt_demos": request.get("prompt_demos", []) if mode in {"examples", "combined"} else [],
            "prompt_demo_coverage": request.get("prompt_demo_coverage", {}) if mode in {"examples", "combined"} else {},
            "prompt": request["prompt"],
            "raw_response": content,
            "predicted_answer": predicted_answer,
            "parse_method": parsed.method,
            "parse_success": parse_success,
            "exact_predicted_answer": parsed_exact.answer,
            "exact_parse_method": parsed_exact.method,
            "exact_parse_success": bool(parsed_exact.answer),
            "first_token_predicted_answer": predicted_answer,
            "first_token_parse_method": parsed.method,
            "first_token_parse_success": parse_success,
            "first_token_is_correct": parse_success and predicted_answer == expected_answer,
            "is_correct": parse_success and predicted_answer == expected_answer,
        }
        results.append(result_entry)

        if use_tqdm and pbar is not None:
            pbar.update(1)

    def _persist_progress() -> None:
        processed = len(results)
        if processed % snapshot_every != 0 and processed < len(requests):
            return
        snapshot_payload = {
            "task": "lexical_category_inference",
            "logic_condition": logic,
            "difficulty": level,
            "difficulty_name": DIFFICULTY_NAME[level],
            "category_operator": design_cell["category_operator"],
            "category_arity": int(design_cell["category_arity"]),
            "shared_d1": bool(design_cell["shared_d1"]),
            "mode": mode,
            "model": model_name,
            "prompt_demo_semantics": PROMPT_DEMO_SEMANTICS,
            "num_test_samples": len(requests),
            "num_demos": effective_num_demos if mode in {"examples", "combined"} else 0,
            "demo_schedule": output_demo_schedule,
            "max_tokens": max_tokens,
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
            chunk_prompts = [request["prompt"] for request in chunk_requests]
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
            chunk_prompts = [request["prompt"] for request in chunk_requests]

            try:
                chunk_responses = await query_batch_fn(chunk_prompts)
                if len(chunk_responses) != len(chunk_requests):
                    raise RuntimeError(
                        f"Batched query size mismatch (got {len(chunk_responses)} "
                        f"for {len(chunk_requests)} prompts)."
                    )
            except Exception as batch_exc:  # noqa: BLE001
                print(f"[query_batch_fn fallback] chunk {chunk_start}:{chunk_end} failed: {batch_exc}")
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

            for offset, (request, raw_response) in enumerate(zip(chunk_requests, chunk_responses), start=1):
                _finalize_one(chunk_start + offset, request, raw_response)
            _persist_progress()
    else:
        if max_concurrent > 1:
            async def _query_one(prompt: str) -> str:
                try:
                    return await query_fn(prompt)
                except Exception as exc:  # noqa: BLE001
                    return str(exc)

            for chunk_start in range(start_index, len(requests), max_concurrent):
                chunk_end = min(chunk_start + max_concurrent, len(requests))
                chunk_requests = requests[chunk_start:chunk_end]
                chunk_responses = await asyncio.gather(
                    *[_query_one(request["prompt"]) for request in chunk_requests]
                )
                for offset, (request, raw_response) in enumerate(zip(chunk_requests, chunk_responses), start=1):
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
        f"lexical_category_inference_{logic}_{level}_{mode}_demos{output_num_demos}"
        f"{filename_schedule_suffix}_{safe_model_name}_{timestamp}.json",
    )

    payload = {
        "task": "lexical_category_inference",
        "logic_condition": logic,
        "difficulty": level,
        "difficulty_name": DIFFICULTY_NAME[level],
        "category_operator": design_cell["category_operator"],
        "category_arity": int(design_cell["category_arity"]),
        "shared_d1": bool(design_cell["shared_d1"]),
        "design_cell": f"{logic}/{level}",
        "mode": mode,
        "model": model_name,
        "prompt_demo_semantics": PROMPT_DEMO_SEMANTICS,
        "num_test_samples": len(selected_tests),
        "num_demos": effective_num_demos if mode in {"examples", "combined"} else 0,
        "demo_schedule": output_demo_schedule,
        "max_tokens": max_tokens,
        "seed": seed,
        "timestamp": timestamp,
        "results": results,
    }

    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    if resume_enabled:
        clear_checkpoint(checkpoint_path)
    if os.path.exists(snapshot_path):
        os.remove(snapshot_path)

    print(f"\nResults saved to: {output_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lexical Category Inference experiments.")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "easy", "medium", "hard", "d1", "d2", "d3"],
    )
    parser.add_argument("--logic-condition", type=str, default="all", choices=["shared", "either", "both", "all"])
    parser.add_argument("--mode", type=str, default="rules", choices=["rules", "examples", "combined"])
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=600,
        help="Number of test samples to score. The seed-123 default uses the precomputed n=600 category-balanced subset.",
    )
    parser.add_argument(
        "--num-demos",
        type=int,
        default=None,
        help=(
            "Number of prompt demos to show. If omitted, examples/combined use "
            "examples_minimal counts: d1=24, d2=32, d3=48. Rules mode uses no demos."
        ),
    )
    parser.add_argument("--model", type=str, default="qwen3-14b")
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default=os.path.join(WORKSPACE_ROOT, "experiment_outputs"))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    async def _run_plan() -> None:
        cell_plan = get_design_cell_plan(args.difficulty, args.logic_condition)
        for index, (logic, level, _) in enumerate(cell_plan, start=1):
            print(f"--- [{index}/{len(cell_plan)}] lexical_category_inference cell={logic}/{level} ---")
            await run_lexical_category_inference_experiment(
                difficulty=level,
                logic_condition=logic,
                mode=args.mode,
                num_test_samples=args.num_test_samples,
                num_demos=args.num_demos,
                output_dir=args.output_dir,
                model_name=args.model,
                seed=args.seed,
                max_tokens=args.max_tokens,
            )

    asyncio.run(_run_plan())
