#!/usr/bin/env python3
"""Generate data for the noun class agreement task.

The script creates balanced Yes/No splits by generating paired examples:
1) a grammatical sentence (Yes)
2) a matched corrupted sentence (No)

Difficulty settings:
- D1 (easy): 2 classes, determiner agreement only
- D2 (medium): 4 classes, determiner + adjective suffix agreement
- D3 (hard): 6 classes, determiner + adjective suffix + verb agreement
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


DIFFICULTY_NAME = {1: "easy", 2: "medium", 3: "hard"}


@dataclass(frozen=True)
class DifficultySpec:
    classes: List[str]
    det_by_class: Dict[str, str]
    adj_suffix_by_class: Optional[Dict[str, str]]
    verb_subj_prefix_by_class: Optional[Dict[str, str]]
    verb_obj_suffix_by_class: Optional[Dict[str, str]]
    noun_to_class: Dict[str, str]
    adj_stems: List[str]
    verb_roots: List[str]
    error_types: List[str]
    checks_per_sentence: int

    @property
    def nouns_by_class(self) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {klass: [] for klass in self.classes}
        for noun, klass in self.noun_to_class.items():
            grouped.setdefault(klass, []).append(noun)
        return grouped


def build_specs() -> Dict[int, DifficultySpec]:
    shared_stems = ["glim", "prun", "fen", "zay", "nal", "tor"]
    shared_roots = ["dax", "miv", "lorp"]

    return {
        1: DifficultySpec(
            classes=["A", "B"],
            det_by_class={"A": "ka", "B": "ti"},
            adj_suffix_by_class=None,
            verb_subj_prefix_by_class=None,
            verb_obj_suffix_by_class=None,
            noun_to_class={
                "tac": "A",
                "sular": "A",
                "fep": "A",
                "wug": "A",
                "bim": "B",
                "noko": "B",
                "glarn": "B",
                "zesh": "B",
            },
            adj_stems=[],
            verb_roots=shared_roots,
            error_types=["subj_det_wrong", "obj_det_wrong"],
            checks_per_sentence=2,
        ),
        2: DifficultySpec(
            classes=["A", "B", "C", "D"],
            det_by_class={"A": "ka", "B": "ti", "C": "su", "D": "vo"},
            adj_suffix_by_class={"A": "en", "B": "os", "C": "im", "D": "at"},
            verb_subj_prefix_by_class=None,
            verb_obj_suffix_by_class=None,
            noun_to_class={
                "tav": "A",
                "fap": "A",
                "bim": "B",
                "glarn": "B",
                "sular": "C",
                "wug": "C",
                "noko": "D",
                "zesh": "D",
            },
            adj_stems=shared_stems,
            verb_roots=shared_roots,
            error_types=[
                "subj_det_wrong",
                "subj_adj_wrong",
                "subj_noun_wrong",
                "obj_det_wrong",
                "obj_adj_wrong",
                "obj_noun_wrong",
            ],
            checks_per_sentence=4,
        ),
        3: DifficultySpec(
            classes=["A", "B", "C", "D", "E", "F"],
            det_by_class={"A": "ka", "B": "ti", "C": "su", "D": "vo", "E": "ne", "F": "la"},
            adj_suffix_by_class={
                "A": "en",
                "B": "os",
                "C": "im",
                "D": "at",
                "E": "uk",
                "F": "esh",
            },
            verb_subj_prefix_by_class={
                "A": "ge",
                "B": "du",
                "C": "ri",
                "D": "zo",
                "E": "pa",
                "F": "li",
            },
            verb_obj_suffix_by_class={
                "A": "an",
                "B": "eb",
                "C": "ig",
                "D": "ot",
                "E": "ul",
                "F": "er",
            },
            noun_to_class={
                "tav": "A",
                "fap": "A",
                "bim": "B",
                "glarn": "B",
                "sular": "C",
                "wug": "C",
                "noko": "D",
                "zesh": "D",
                "tac": "E",
                "fep": "E",
                "lurn": "F",
                "prax": "F",
            },
            adj_stems=shared_stems,
            verb_roots=shared_roots,
            error_types=[
                "subj_det_wrong",
                "subj_adj_wrong",
                "subj_noun_wrong",
                "obj_det_wrong",
                "obj_adj_wrong",
                "obj_noun_wrong",
                "verb_prefix_wrong",
                "verb_suffix_wrong",
            ],
            checks_per_sentence=6,
        ),
    }


def choose_other_class(classes: List[str], klass: str, rng: random.Random) -> str:
    options = [c for c in classes if c != klass]
    if not options:
        raise ValueError("Cannot choose alternate class from a single-class inventory.")
    return rng.choice(options)


def generate_yes_structure(spec: DifficultySpec, difficulty: int, rng: random.Random) -> Dict[str, Dict[str, Optional[str]]]:
    nouns_by_class = spec.nouns_by_class

    subj_class = rng.choice(spec.classes)
    obj_class = rng.choice(spec.classes)
    subj_noun = rng.choice(nouns_by_class[subj_class])
    obj_noun = rng.choice(nouns_by_class[obj_class])
    verb_root = rng.choice(spec.verb_roots)

    structure = {
        "subj": {
            "noun": subj_noun,
            "det": spec.det_by_class[subj_class],
            "adj_stem": None,
            "adj_suffix": None,
        },
        "verb": {
            "root": verb_root,
            "prefix": None,
            "suffix": None,
        },
        "obj": {
            "noun": obj_noun,
            "det": spec.det_by_class[obj_class],
            "adj_stem": None,
            "adj_suffix": None,
        },
    }

    if difficulty >= 2:
        structure["subj"]["adj_stem"] = rng.choice(spec.adj_stems)
        structure["subj"]["adj_suffix"] = spec.adj_suffix_by_class[subj_class]
        structure["obj"]["adj_stem"] = rng.choice(spec.adj_stems)
        structure["obj"]["adj_suffix"] = spec.adj_suffix_by_class[obj_class]

    if difficulty == 3:
        structure["verb"]["prefix"] = spec.verb_subj_prefix_by_class[subj_class]
        structure["verb"]["suffix"] = spec.verb_obj_suffix_by_class[obj_class]

    return structure


def render_sentence(structure: Dict[str, Dict[str, Optional[str]]], difficulty: int) -> str:
    subj = structure["subj"]
    verb = structure["verb"]
    obj = structure["obj"]

    if difficulty == 1:
        return f"{subj['det']} {subj['noun']} {verb['root']} {obj['det']} {obj['noun']}"

    subj_adj = f"{subj['adj_stem']}-{subj['adj_suffix']}"
    obj_adj = f"{obj['adj_stem']}-{obj['adj_suffix']}"

    if difficulty == 3:
        verb_token = f"{verb['prefix']}-{verb['root']}-{verb['suffix']}"
    else:
        verb_token = str(verb["root"])

    return (
        f"{subj['det']} {subj_adj} {subj['noun']} "
        f"{verb_token} "
        f"{obj['det']} {obj_adj} {obj['noun']}"
    )


def is_grammatical(structure: Dict[str, Dict[str, Optional[str]]], spec: DifficultySpec, difficulty: int) -> bool:
    subj = structure["subj"]
    verb = structure["verb"]
    obj = structure["obj"]

    subj_noun = str(subj.get("noun") or "")
    obj_noun = str(obj.get("noun") or "")

    subj_class = spec.noun_to_class.get(subj_noun)
    obj_class = spec.noun_to_class.get(obj_noun)
    if subj_class is None or obj_class is None:
        return False

    if subj.get("det") != spec.det_by_class[subj_class]:
        return False
    if obj.get("det") != spec.det_by_class[obj_class]:
        return False

    if verb.get("root") not in spec.verb_roots:
        return False

    if difficulty >= 2:
        if subj.get("adj_suffix") != spec.adj_suffix_by_class[subj_class]:
            return False
        if obj.get("adj_suffix") != spec.adj_suffix_by_class[obj_class]:
            return False

    if difficulty == 3:
        if verb.get("prefix") != spec.verb_subj_prefix_by_class[subj_class]:
            return False
        if verb.get("suffix") != spec.verb_obj_suffix_by_class[obj_class]:
            return False

    return True


def corrupt_structure(
    structure: Dict[str, Dict[str, Optional[str]]],
    error_type: str,
    spec: DifficultySpec,
    difficulty: int,
    rng: random.Random,
) -> Dict[str, Dict[str, Optional[str]]]:
    corrupted = copy.deepcopy(structure)
    subj = corrupted["subj"]
    verb = corrupted["verb"]
    obj = corrupted["obj"]

    subj_class = spec.noun_to_class[str(subj["noun"])]
    obj_class = spec.noun_to_class[str(obj["noun"])]
    nouns_by_class = spec.nouns_by_class

    if error_type == "subj_det_wrong":
        wrong_class = choose_other_class(spec.classes, subj_class, rng)
        subj["det"] = spec.det_by_class[wrong_class]
    elif error_type == "obj_det_wrong":
        wrong_class = choose_other_class(spec.classes, obj_class, rng)
        obj["det"] = spec.det_by_class[wrong_class]
    elif error_type == "subj_adj_wrong":
        wrong_class = choose_other_class(spec.classes, subj_class, rng)
        subj["adj_suffix"] = spec.adj_suffix_by_class[wrong_class]
    elif error_type == "obj_adj_wrong":
        wrong_class = choose_other_class(spec.classes, obj_class, rng)
        obj["adj_suffix"] = spec.adj_suffix_by_class[wrong_class]
    elif error_type == "subj_noun_wrong":
        wrong_class = choose_other_class(spec.classes, subj_class, rng)
        subj["noun"] = rng.choice(nouns_by_class[wrong_class])
    elif error_type == "obj_noun_wrong":
        wrong_class = choose_other_class(spec.classes, obj_class, rng)
        obj["noun"] = rng.choice(nouns_by_class[wrong_class])
    elif error_type == "verb_prefix_wrong":
        wrong_class = choose_other_class(spec.classes, subj_class, rng)
        verb["prefix"] = spec.verb_subj_prefix_by_class[wrong_class]
    elif error_type == "verb_suffix_wrong":
        wrong_class = choose_other_class(spec.classes, obj_class, rng)
        verb["suffix"] = spec.verb_obj_suffix_by_class[wrong_class]
    else:
        raise ValueError(f"Unknown error type: {error_type}")

    if difficulty == 1 and error_type not in {"subj_det_wrong", "obj_det_wrong"}:
        raise ValueError(f"Unsupported error type for difficulty 1: {error_type}")
    if difficulty == 2 and error_type in {"verb_prefix_wrong", "verb_suffix_wrong"}:
        raise ValueError(f"Unsupported error type for difficulty 2: {error_type}")

    return corrupted


def build_error_plan(error_types: List[str], num_pairs: int, rng: random.Random) -> List[str]:
    if num_pairs <= 0:
        return []

    per_type = num_pairs // len(error_types)
    remainder = num_pairs % len(error_types)
    plan = []

    for error_type in error_types:
        plan.extend([error_type] * per_type)

    tail = list(error_types)
    rng.shuffle(tail)
    plan.extend(tail[:remainder])
    rng.shuffle(plan)
    return plan


def make_record(
    *,
    record_id: str,
    pair_id: str,
    difficulty: int,
    structure: Dict[str, Dict[str, Optional[str]]],
    label: str,
    error_type: Optional[str],
) -> Dict[str, object]:
    subject_noun = str(structure["subj"]["noun"])
    object_noun = str(structure["obj"]["noun"])
    return {
        "id": record_id,
        "pair_id": pair_id,
        "task": "noun_class_agreement",
        "difficulty": difficulty,
        "difficulty_name": DIFFICULTY_NAME[difficulty],
        "sentence": render_sentence(structure, difficulty),
        "label": label,
        "meta": {
            "subject_noun": subject_noun,
            "object_noun": object_noun,
            "error_type": error_type,
        },
    }


def generate_split_records(
    *,
    difficulty: int,
    spec: DifficultySpec,
    split: str,
    num_samples: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    if num_samples % 2 != 0:
        raise ValueError(
            f"{split} sample count must be even for exact 50/50 labels. Got: {num_samples}"
        )

    num_pairs = num_samples // 2
    error_plan = build_error_plan(spec.error_types, num_pairs, rng)
    records: List[Dict[str, object]] = []

    for pair_idx, error_type in enumerate(error_plan, start=1):
        pair_id = f"nca_d{difficulty}_{split}_pair{pair_idx:06d}"

        generated = False
        for _ in range(200):
            yes_struct = generate_yes_structure(spec, difficulty, rng)
            no_struct = corrupt_structure(yes_struct, error_type, spec, difficulty, rng)

            if not is_grammatical(yes_struct, spec, difficulty):
                continue
            if is_grammatical(no_struct, spec, difficulty):
                continue
            if render_sentence(yes_struct, difficulty) == render_sentence(no_struct, difficulty):
                continue

            yes_id = f"nca_d{difficulty}_{split}_{(pair_idx * 2 - 1):06d}"
            no_id = f"nca_d{difficulty}_{split}_{(pair_idx * 2):06d}"

            records.append(
                make_record(
                    record_id=yes_id,
                    pair_id=pair_id,
                    difficulty=difficulty,
                    structure=yes_struct,
                    label="Yes",
                    error_type=None,
                )
            )
            records.append(
                make_record(
                    record_id=no_id,
                    pair_id=pair_id,
                    difficulty=difficulty,
                    structure=no_struct,
                    label="No",
                    error_type=error_type,
                )
            )
            generated = True
            break

        if not generated:
            raise RuntimeError(
                f"Failed to generate valid pair for difficulty {difficulty}, error_type={error_type}"
            )

    rng.shuffle(records)

    # Reassign IDs after shuffling to keep split-local IDs contiguous.
    for idx, record in enumerate(records, start=1):
        record["id"] = f"nca_d{difficulty}_{split}_{idx:06d}"

    return records


def write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def summarize_split(records: List[Dict[str, object]]) -> str:
    label_counts = Counter(str(r.get("label")) for r in records)
    error_counts = Counter(
        str((r.get("meta") or {}).get("error_type"))
        for r in records
        if str(r.get("label")) == "No"
    )
    error_counts.pop("None", None)
    errors = ", ".join(f"{k}:{v}" for k, v in sorted(error_counts.items()))
    return f"labels={dict(label_counts)} | negative_error_types={{{errors}}}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate noun class agreement datasets.")
    parser.add_argument("--train_samples", type=int, default=500, help="Number of train examples per difficulty.")
    parser.add_argument("--test_samples", type=int, default=500, help="Number of test examples per difficulty.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["1", "2", "3"],
        help="Difficulties to generate (1 2 3).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output root directory. Defaults to <repo>/data/noun_class_agreement.",
    )
    args = parser.parse_args()

    specs = build_specs()
    difficulties = []
    for raw in args.difficulties:
        value = int(raw)
        if value not in specs:
            raise ValueError(f"Unsupported difficulty '{raw}'. Choose from 1, 2, 3.")
        difficulties.append(value)

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root) if args.output_root else repo_root / "data" / "noun_class_agreement"

    for difficulty in difficulties:
        spec = specs[difficulty]
        difficulty_dir = output_root / f"d{difficulty}"

        train_rng = random.Random(args.seed + difficulty * 1000 + 11)
        test_rng = random.Random(args.seed + difficulty * 1000 + 29)

        train_records = generate_split_records(
            difficulty=difficulty,
            spec=spec,
            split="train",
            num_samples=args.train_samples,
            rng=train_rng,
        )
        test_records = generate_split_records(
            difficulty=difficulty,
            spec=spec,
            split="test",
            num_samples=args.test_samples,
            rng=test_rng,
        )

        train_path = (
            difficulty_dir
            / "train"
            / f"noun_class_agreement_d{difficulty}_train_n{len(train_records)}.jsonl"
        )
        test_path = (
            difficulty_dir
            / "test"
            / f"noun_class_agreement_d{difficulty}_test_n{len(test_records)}.jsonl"
        )

        write_jsonl(train_path, train_records)
        write_jsonl(test_path, test_records)

        print(
            f"D{difficulty} ({DIFFICULTY_NAME[difficulty]}) train -> {train_path}\n"
            f"  {summarize_split(train_records)}"
        )
        print(
            f"D{difficulty} ({DIFFICULTY_NAME[difficulty]}) test  -> {test_path}\n"
            f"  {summarize_split(test_records)}"
        )


if __name__ == "__main__":
    main()
