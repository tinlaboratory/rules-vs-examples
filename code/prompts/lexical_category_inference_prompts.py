"""Prompt builders for the Lexical Category Inference Yes/No task."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from prompts.markdown_loader import render_markdown_prompt


_DIFFICULTY_ALIASES = {
    "1": "d1",
    "easy": "d1",
    "d1": "d1",
    "2": "d2",
    "medium": "d2",
    "d2": "d2",
    "3": "d3",
    "hard": "d3",
    "d3": "d3",
}

_LOGIC_CONDITION_ALIASES = {
    "shared": "shared",
    "single": "shared",
    "either": "either",
    "or": "either",
    "any": "either",
    "both": "both",
    "and": "both",
}

# Difficulty-specific demo counts for examples/combined prompts.
#
# examples_minimal is the primary fair comparison against rules:
# num_demos = 2 * max(12, 8 * category_arity).
#
# The factor of 2 enforces a 50/50 pure-IO curriculum with a local Yes pair for
# each No row. The 12-positive floor covers the non-composite slot/order and
# boundary families; the 8 * arity term gives restrained component coverage for
# OR bridges and AND near-miss evidence.
_NEGATIVE_DEMO_TYPE_COUNTS = {
    "d1": 2,
    "d2": 3,
    "d3": 4,
}
_CATEGORY_ARITY_BY_DIFFICULTY = {
    "d1": 1,
    "d2": 2,
    "d3": 3,
}
_COMPOSITE_RULE_COMPONENT_RE = re.compile(
    r"(?:^|;\s*(?:and\s*)?)"
    r"exactly\s+(?:one|two|three|four|\d+)\s+words?\s+that\s+fits?\s+this\s+description:\s*"
    r"(.*?)(?=(?:;\s*(?:and\s*)?exactly\s+(?:one|two|three|four|\d+)\s+words?\s+that\s+fits?\s+this\s+description:)|$)",
    flags=re.IGNORECASE,
)
_RULE_GLOSS_PROMPT_REPLACEMENTS = {
    "items that fit at least one of these descriptions:": "items that fit at least one of these descriptions -",
    "items that fit all of these descriptions:": "items that fit all of these descriptions -",
    "items that fit all three of these descriptions:": "items that fit all three of these descriptions -",
}
_CLASSIFIED_CATEGORY_RE = re.compile(r"\bthings classified as ([^;,\n]+)")
_TAXONOMIC_CATEGORY_RE = re.compile(r"\bthings that are ([^;,\n]+)")
_CATEGORY_DISPLAY_LABELS = {
    "animal": "animals",
    "animals": "animals",
    "arts and crafts supply": "arts and crafts supplies",
    "bird": "birds",
    "birds": "birds",
    "body part": "body parts",
    "breakfast food": "breakfast foods",
    "candy": "candy",
    "clothing": "clothing",
    "clothing accessory": "clothing accessories",
    "condiment": "condiments",
    "construction equipment": "construction equipment",
    "container": "containers",
    "dessert": "desserts",
    "drink": "drinks",
    "electronic device": "electronic devices",
    "farm animal": "farm animals",
    "fastener": "fasteners",
    "fish": "fish",
    "food": "food",
    "footwear": "footwear",
    "fruit": "fruit",
    "fruits": "fruit",
    "furniture": "furniture",
    "game": "games",
    "garden tool": "garden tools",
    "hardware": "hardware",
    "headwear": "headwear",
    "home appliance": "home appliances",
    "home decor": "home decor",
    "insect": "insects",
    "insects": "insects",
    "jewelry": "jewelry",
    "kitchen appliance": "kitchen appliances",
    "kitchen tool": "kitchen tools",
    "lighting": "lighting",
    "mammal": "mammals",
    "mammals": "mammals",
    "medical equipment": "medical equipment",
    "musical instrument": "musical instruments",
    "office supply": "office supplies",
    "outerwear": "outerwear",
    "part of car": "car parts",
    "personal hygiene item": "personal hygiene items",
    "plant": "plants",
    "protective clothing": "protective clothing",
    "safety equipment": "safety equipment",
    "school supply": "school supplies",
    "scientific equipment": "scientific equipment",
    "sea animal": "sea animals",
    "seafood": "seafood",
    "sports equipment": "sports equipment",
    "tool": "tools",
    "tools": "tools",
    "toy": "toys",
    "utensil": "utensils",
    "utensils": "utensils",
    "vegetable": "vegetables",
    "vegetables": "vegetables",
    "vehicle": "vehicles",
    "watercraft": "watercraft",
    "weapon": "weapons",
    "weapons": "weapons",
    "women's clothing": "women's clothing",
}


_DEFAULT_NUM_DEMOS = {
    "d1": 24,
    "d2": 32,
    "d3": 48,
}
DEMO_SCHEDULE = "examples_minimal"


def get_default_num_demos(
    difficulty: str,
    logic_condition: str | None = None,
) -> int:
    del logic_condition
    return _DEFAULT_NUM_DEMOS[normalize_difficulty(difficulty)]


def demo_schedule_label() -> str:
    return DEMO_SCHEDULE


def get_demo_schedule_for_num_demos(
    difficulty: str,
    num_demos: int,
    logic_condition: str | None = None,
) -> str:
    if int(num_demos) == get_default_num_demos(difficulty, logic_condition):
        return demo_schedule_label()
    return "custom"


def normalize_difficulty(difficulty: int | str) -> str:
    value = str(difficulty).strip().lower()
    if value not in _DIFFICULTY_ALIASES:
        raise ValueError(f"Unsupported lexical_category_inference difficulty: {difficulty}")
    return _DIFFICULTY_ALIASES[value]


def normalize_logic_condition(logic_condition: str | None) -> str | None:
    if logic_condition is None:
        return None
    value = str(logic_condition).strip().lower()
    if value not in _LOGIC_CONDITION_ALIASES:
        raise ValueError(f"Unsupported lexical_category_inference logic_condition: {logic_condition}")
    return _LOGIC_CONDITION_ALIASES[value]


def _prompt_label_names(labels: Sequence[str]) -> Dict[str, str]:
    return {
        str(label): f"Category {index + 1}"
        for index, label in enumerate(labels)
    }


def _format_prompt_label_list(prompt_label_names: Dict[str, str], labels: Sequence[str]) -> str:
    names = [prompt_label_names[str(label)] for label in labels]
    if len(names) <= 1:
        return "".join(names)
    if len(names) == 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _candidate_groups_by_display_label(
    labels: Sequence[str],
    candidate_groups: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    ordered_labels = [str(label) for label in labels]
    ordered_label_set = set(ordered_labels)
    groups_by_display_label: Dict[str, Dict[str, Any]] = {}
    duplicate_labels: List[str] = []
    for group in candidate_groups:
        display_label = str(group.get("display_label"))
        if display_label in groups_by_display_label:
            duplicate_labels.append(display_label)
        groups_by_display_label[display_label] = dict(group)

    missing_labels = [
        label
        for label in ordered_labels
        if label not in groups_by_display_label
    ]
    extra_labels = [
        label
        for label in groups_by_display_label
        if label not in ordered_label_set
    ]
    if duplicate_labels or missing_labels or extra_labels:
        raise ValueError(
            "Compact lexical_category_inference prompts require each category label exactly once. "
            f"duplicate={duplicate_labels}, missing={missing_labels}, extra={extra_labels}"
        )
    return groups_by_display_label


def _representative_word(group: Dict[str, Any]) -> str:
    words = list(group.get("words") or [])
    if len(words) != 1:
        raise ValueError(
            "lexical_category_inference compact prompts require each prompt-visible item "
            f"to contain exactly one word, found {len(words)}."
        )
    return str(words[0]).strip().lower()


def _format_candidate_groups(
    labels: Sequence[str],
    candidate_groups: Sequence[Dict[str, Any]],
    prompt_label_names: Dict[str, str],
) -> str:
    del prompt_label_names
    groups_by_display_label = _candidate_groups_by_display_label(labels, candidate_groups)
    return ", ".join(
        _representative_word(groups_by_display_label[str(label)])
        for label in labels
    )


def _format_demo_blocks(
    labels: Sequence[str],
    demos: Sequence[Dict[str, Any]],
    prompt_label_names: Dict[str, str],
    include_answers: bool = True,
    item_prefix: str = "",
) -> str:
    blocks: List[str] = []
    for demo in demos:
        candidate_block = _format_candidate_groups(
            labels=labels,
            candidate_groups=demo["candidate_groups"],
            prompt_label_names=prompt_label_names,
        )
        answer_line = f'\nAnswer: {demo["answer"]}' if include_answers else "\nAnswer:"
        blocks.append(f"{item_prefix}{candidate_block}{answer_line}")
    return "\n".join(blocks)


def _format_single_item_rule_gloss(
    gloss: str,
    category_operator: str | None = None,
) -> str:
    text = str(gloss)
    for source, replacement in _RULE_GLOSS_PROMPT_REPLACEMENTS.items():
        if source in text:
            return _simplify_prompt_rule_gloss(text.replace(source, replacement))

    components = [
        match.strip()
        for match in _COMPOSITE_RULE_COMPONENT_RE.findall(text)
        if match.strip()
    ]
    if len(components) <= 1:
        return _simplify_prompt_rule_gloss(text)
    if category_operator == "and":
        prefix = "items that fit all three of these descriptions - " if len(components) == 3 else "items that fit all of these descriptions - "
    else:
        prefix = "items that fit at least one of these descriptions - "
    return _simplify_prompt_rule_gloss(prefix + "; ".join(components))


def _display_category_label(raw_label: str) -> str:
    label = " ".join(str(raw_label).strip().lower().split())
    return _CATEGORY_DISPLAY_LABELS.get(label, label)


def _simplify_prompt_rule_gloss(text: str) -> str:
    text = _CLASSIFIED_CATEGORY_RE.sub(
        lambda match: _display_category_label(match.group(1)),
        text,
    )

    def replace_taxonomic(match: re.Match[str]) -> str:
        raw_label = " ".join(match.group(1).strip().lower().split())
        if raw_label not in _CATEGORY_DISPLAY_LABELS:
            return match.group(0)
        return _display_category_label(raw_label)

    return _TAXONOMIC_CATEGORY_RE.sub(replace_taxonomic, text)


def _infer_rule_markdown_path(example: Dict[str, Any]) -> tuple[str, str, str]:
    raw_difficulty = example.get("difficulty") or example.get("difficulty_name")
    category_operator = str(example.get("category_operator") or "").strip().lower()
    raw_logic = example.get("logic_condition")

    if raw_difficulty:
        difficulty = normalize_difficulty(raw_difficulty)
    elif category_operator == "single":
        difficulty = "d1"
    else:
        raise ValueError(
            "Lexical category inference rules prompts require a difficulty "
            "when category_operator is composite."
        )

    difficulty_heading = {
        "d1": "Difficulty 1",
        "d2": "Difficulty 2",
        "d3": "Difficulty 3",
    }[difficulty]

    if difficulty == "d1":
        return ("Rules", "Shared", difficulty_heading)

    logic = normalize_logic_condition(raw_logic) if raw_logic else None
    if logic is None and category_operator:
        logic = normalize_logic_condition(category_operator)

    if logic == "either":
        return ("Rules", "Either", difficulty_heading)
    if logic == "both":
        return ("Rules", "Both", difficulty_heading)
    raise ValueError(
        "Lexical category inference d2/d3 rules prompts require "
        "logic_condition='either'/'both' or category_operator='or'/'and'."
    )


def get_rule_based_prompt(example: Dict[str, Any]) -> str:
    labels = list(example["labels"])
    prompt_label_names = _prompt_label_names(labels)
    glosses = example["rule_glosses"]
    category_operator = str(example.get("category_operator") or "")
    label_block = "\n".join(
        f'{prompt_label_names[label]}: {_format_single_item_rule_gloss(glosses[label], category_operator)}'
        for label in labels
    )
    candidate_block = _format_candidate_groups(labels, example["candidate_groups"], prompt_label_names)
    prompt_labels = _format_prompt_label_list(prompt_label_names, labels)

    return render_markdown_prompt(
        "lexical_category_inference.md",
        _infer_rule_markdown_path(example),
        prompt_labels=prompt_labels,
        label_block=label_block,
        candidate_block=candidate_block,
    )


def get_example_based_prompt(example: Dict[str, Any], demos: Sequence[Dict[str, Any]]) -> str:
    labels = list(example["labels"])
    prompt_label_names = _prompt_label_names(labels)
    examples_text = _format_demo_blocks(
        labels,
        demos,
        prompt_label_names,
        include_answers=True,
        item_prefix="List: ",
    )
    candidate_block = _format_candidate_groups(labels, example["candidate_groups"], prompt_label_names)
    if examples_text:
        examples_text += "\n\n"

    return f"""{examples_text}Your answer must be exactly one word: Yes or No.
List: {candidate_block}
Answer:"""


def get_combined_prompt(example: Dict[str, Any], demos: Sequence[Dict[str, Any]]) -> str:
    labels = list(example["labels"])
    prompt_label_names = _prompt_label_names(labels)
    glosses = example["rule_glosses"]
    category_operator = str(example.get("category_operator") or "")
    label_block = "\n".join(
        f'{prompt_label_names[label]}: {_format_single_item_rule_gloss(glosses[label], category_operator)}'
        for label in labels
    )
    examples_text = _format_demo_blocks(
        labels,
        demos,
        prompt_label_names,
        include_answers=True,
        item_prefix="List: ",
    )
    candidate_block = _format_candidate_groups(labels, example["candidate_groups"], prompt_label_names)
    prompt_labels = _format_prompt_label_list(prompt_label_names, labels)

    return f"""You will be shown a list of four words. Decide whether this list is correct based on the following rules and solved examples.

RULES:
1. There are exactly four categories: {prompt_labels}.
2. Each category has the meaning shown in the category definitions below.
3. You are also given solved examples from this same episode.
4. A correct list contains exactly four comma-separated items.
5. In a correct list, the first item is a member of Category 1, the second item is a member of Category 2, and so on.
6. If a category says "at least one", an item may satisfy any listed description. If it says "all", an item must satisfy every listed description.
7. Decide whether the list satisfies the category definitions, using the solved examples only as demonstrations.
8. Your answer must be exactly one word: Yes or No.

CATEGORY DEFINITIONS:
{label_block}

EXAMPLES:
{examples_text}

List: {candidate_block}
Answer:"""
