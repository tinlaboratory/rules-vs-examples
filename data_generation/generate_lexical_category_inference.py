#!/usr/bin/env python3
"""Generate data for the Lexical Category Inference Yes/No task."""

from __future__ import annotations

import argparse
import ast
import csv
import functools
import hashlib
import heapq
import itertools
import json
import math
import os
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = WORKSPACE_ROOT / "data" / "lexical_category_inference"
DEFAULT_SOURCE_SEED_PATH = DATA_ROOT / "source_seed_categories.json"
DEFAULT_BUNDLE_BANK_V1 = DATA_ROOT / "bundle_bank_v1.json"
DEFAULT_BUNDLE_BANK_SMOKE = DATA_ROOT / "bundle_bank_smoke.json"
DEFAULT_MANUAL_ADDITIONS_PATH = DATA_ROOT / "bundle_bank_manual_additions.json"
DEFAULT_SOURCE_NORMS_ROOT = DATA_ROOT / "source_norms"
DEFAULT_SEMANTIC_WORD_OVERRIDES = DATA_ROOT / "semantic_word_overrides.json"
DEFAULT_REVIEW_ARTIFACT_ROOT = DATA_ROOT / "review_artifacts"
DEFAULT_SEMANTIC_VALIDATION_ARTIFACT = DEFAULT_REVIEW_ARTIFACT_ROOT / "semantic_validation_v34.json"
DEFAULT_BOTH_D3_NEAR_MISS_AUDIT_ARTIFACT = DEFAULT_REVIEW_ARTIFACT_ROOT / "both_d3_near_miss_semantic_audit_v34.json"
DEFAULT_SOURCE_INTERSECTION_POSITIVE_SENSE_AUDIT_ARTIFACT = DEFAULT_REVIEW_ARTIFACT_ROOT / "source_intersection_positive_sense_audit_v34.json"
DEFAULT_HIGH_RISK_RATED_PROPERTY_AUDIT_ARTIFACT = DEFAULT_REVIEW_ARTIFACT_ROOT / "high_risk_rated_property_plausibility_audit_v34.json"
DEFAULT_SOURCE_INTERSECTION_MATERIAL_AUDIT_ARTIFACT = DEFAULT_REVIEW_ARTIFACT_ROOT / "source_intersection_material_same_instance_audit_v34.json"
DEFAULT_SOURCE_INTERSECTION_MATERIAL_FALSE_EVIDENCE_AUDIT_ARTIFACT = DEFAULT_REVIEW_ARTIFACT_ROOT / "source_intersection_material_false_evidence_audit_v34.json"
DEFAULT_SOURCE_INTERSECTION_FUNCTIONAL_FALSE_EVIDENCE_AUDIT_ARTIFACT = DEFAULT_REVIEW_ARTIFACT_ROOT / "source_intersection_functional_false_evidence_audit_v34.json"
DEFAULT_REPOMIX_XML = Path(os.environ.get("LCI_REPOMIX_XML", str(DATA_ROOT / "repomix-output.xml")))
DEFAULT_LOCAL_SOURCE_REPO = Path(
    os.environ.get("LCI_SOURCE_REPO", str(WORKSPACE_ROOT.parent / "category-source-repo"))
)

LABEL_INVENTORY = [
    "DAX",
    "LOR",
    "TEP",
    "VIM",
    "KOR",
    "NUL",
    "BEX",
    "FAD",
    "SIV",
    "MOP",
    "RUK",
    "JEN",
]

DIFFICULTY_ALIASES = {
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

DIFFICULTY_NAME = {
    "d1": "easy",
    "d2": "medium",
    "d3": "hard",
}

LOGIC_CONDITION_ALIASES = {
    "shared": "shared",
    "single": "shared",
    "either": "either",
    "or": "either",
    "any": "either",
    "both": "both",
    "and": "both",
    "all": "all",
}

REAL_DESIGN_CELLS = [
    ("shared", "d1"),
    ("either", "d2"),
    ("either", "d3"),
    ("both", "d2"),
    ("both", "d3"),
]


@functools.lru_cache(maxsize=200_000)
def _normalize_word_surface_cached(word: str) -> str:
    return word.strip().lower()


def _normalize_word_surface(word: Any) -> str:
    return _normalize_word_surface_cached(str(word))


INCIDENTAL_DISJUNCTION_REPLACEMENTS = (
    ("kitchen tool or dishware names", "kitchenware names"),
    ("weather event or condition names", "weather condition names"),
    ("gemstone or mineral names", "gemstone names"),
    ("herb or spice names", "herb and spice names"),
    ("grain or cereal names", "grain and cereal names"),
    ("holiday or observance names", "holiday and observance names"),
    ("computer hardware or accessory names", "computer hardware names"),
    ("emotion or feeling names", "emotion names"),
    ("names derived from scientists or inventors", "names derived from scientists and inventors"),
    ("food or drink names", "food and drink names"),
    ("symbols used in mathematics or science", "symbols used in mathematics and science"),
    ("common editing or navigation commands", "editing and navigation commands"),
    ("military or historical group names", "military and historical group names"),
    ("office or communication terms", "office communication terms"),
    ("plant or flower names", "plant and flower names"),
    ("ordinary English words or given names", "ordinary English words and given names"),
    ("science or social science fields", "science and social science fields"),
    ("vehicle or transport terms", "vehicle and transport terms"),
    ("weather, climate, or sky terms", "weather, climate, and sky terms"),
    ("weather or sky terms", "weather and sky terms"),
    ("college or professional mascot names", "college and professional mascot names"),
    ("common color words named after foods or drinks", "food and drink color names"),
    ("color words or personal names derived from gems", "gemstone-derived color words and personal names"),
    ("letter names used in mathematical or scientific notation", "letter names used in mathematical and scientific notation"),
    ("U.S. political or historical names", "U.S. political and historical names"),
    ("technology terms that are also ordinary words or names", "technology terms that are also ordinary words and names"),
    ("ride names or ride types at an amusement park", "amusement park ride names"),
    ("things tied to treating a minor cut or scrape at home", "home first-aid items"),
    ("places where benches are a salient part of how people wait, watch, or sit", "places where benches are salient for waiting, watching, and sitting"),
    ("things whose identity is often understood through stacked or nested layers", "things understood through stacked and nested layers"),
    ("named neighborhoods, areas, or landmark districts in New York City", "named neighborhoods, areas, and landmark districts in New York City"),
    ("things that literally or metaphorically suck something in", "things that draw material inward"),
    ("words or names printed on a classic Monopoly board", "words and names printed on a classic Monopoly board"),
    ("famous video game protagonists or mascot characters", "famous video game protagonists and mascot characters"),
    ("things whose design includes a spout for pouring or spraying", "things with a spout used for pouring and spraying"),
    ("things people commonly picture spinning or rotating", "things people commonly picture spinning and rotating"),
    ("things people shake to mix, reveal, activate, or signal something", "things people shake to mix, reveal, activate, and signal something"),
    ("things a fortune teller might claim to read for insight or prediction", "things a fortune teller might claim to read for insight and prediction"),
    ("palindromic words or phrases", "palindromic words and phrases"),
)


@functools.lru_cache(maxsize=16_384)
def _category_gloss_without_incidental_disjunction_cached(gloss: str) -> str:
    text = gloss.strip()
    for old, new in INCIDENTAL_DISJUNCTION_REPLACEMENTS:
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    text = re.sub(r"\band/or\b", "and", text, flags=re.IGNORECASE)
    text = re.sub(r"\bor\b", "and", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _category_gloss_without_incidental_disjunction(gloss: Any) -> str:
    return _category_gloss_without_incidental_disjunction_cached(str(gloss))


def _category_gloss_has_incidental_disjunction(gloss: Any) -> bool:
    return bool(re.search(r"\bor\b|\band/or\b", str(gloss), flags=re.IGNORECASE))


def _normalize_word_list(words: Iterable[Any]) -> List[str]:
    return [_normalize_word_surface(word) for word in words]


def _normalize_word_blocks(blocks: Iterable[Iterable[Any]]) -> List[List[str]]:
    return [_normalize_word_list(block) for block in blocks]


def _normalize_source_word_fields(source: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(source)
    if isinstance(normalized.get("source_words"), list):
        normalized["source_words"] = _normalize_word_list(normalized["source_words"])
    if isinstance(normalized.get("component_words_by_index"), list):
        normalized["component_words_by_index"] = [
            _normalize_word_list(words) if isinstance(words, list) else words
            for words in normalized["component_words_by_index"]
        ]
    if isinstance(normalized.get("near_miss_words_by_component"), list):
        normalized["near_miss_words_by_component"] = [
            _normalize_word_list(words) if isinstance(words, list) else words
            for words in normalized["near_miss_words_by_component"]
        ]
    if isinstance(normalized.get("component_indices_by_word"), dict):
        normalized["component_indices_by_word"] = {
            _normalize_word_surface(word): [int(index) for index in indices]
            for word, indices in normalized["component_indices_by_word"].items()
            if isinstance(indices, list)
        }
    if isinstance(normalized.get("near_miss_component_indices_by_word"), dict):
        normalized["near_miss_component_indices_by_word"] = {
            _normalize_word_surface(word): [int(index) for index in indices]
            for word, indices in normalized["near_miss_component_indices_by_word"].items()
            if isinstance(indices, list)
        }
    if isinstance(normalized.get("component_rule_glosses"), list):
        normalized["component_rule_glosses"] = [
            _category_gloss_without_incidental_disjunction(gloss)
            for gloss in normalized["component_rule_glosses"]
        ]
    if isinstance(normalized.get("source_rows"), list):
        normalized["source_rows"] = [
            _normalize_source_word_fields(row) if isinstance(row, dict) else row
            for row in normalized["source_rows"]
        ]
    return normalized


def _normalize_bundle_word_fields(bundle: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(bundle)
    if isinstance(normalized.get("blocks"), list):
        normalized["blocks"] = _normalize_word_blocks(normalized["blocks"])
    if "rule_gloss" in normalized:
        normalized["rule_gloss"] = _category_gloss_without_incidental_disjunction(normalized["rule_gloss"])
    if isinstance(normalized.get("source"), dict):
        normalized["source"] = _normalize_source_word_fields(normalized["source"])
    return normalized


_SEMANTIC_WORD_OVERRIDES_CACHE: Dict[str, Dict[str, set[str]]] | None = None


def _load_semantic_word_overrides(path: str | Path | None = None) -> Dict[str, Dict[str, set[str]]]:
    global _SEMANTIC_WORD_OVERRIDES_CACHE
    if path is None and _SEMANTIC_WORD_OVERRIDES_CACHE is not None:
        return _SEMANTIC_WORD_OVERRIDES_CACHE
    override_path = Path(path) if path else DEFAULT_SEMANTIC_WORD_OVERRIDES
    if not override_path.exists():
        return {}
    with override_path.open("r") as handle:
        raw = json.load(handle)

    overrides: Dict[str, Dict[str, set[str]]] = {}
    for raw_gloss, raw_spec in raw.items():
        if not isinstance(raw_spec, dict):
            continue
        gloss = _category_gloss_without_incidental_disjunction(raw_gloss).lower()
        overrides[gloss] = {
            "positive": {_normalize_word_surface(word) for word in raw_spec.get("positive") or []},
            "negative": {_normalize_word_surface(word) for word in raw_spec.get("negative") or []},
        }
    if path is None:
        _SEMANTIC_WORD_OVERRIDES_CACHE = overrides
    return overrides


def _semantic_predicate_membership_by_gloss(
    predicates: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, set[str]]:
    if predicates is None:
        predicates = _load_source_norm_predicates(DEFAULT_SOURCE_NORMS_ROOT)
    overrides = _load_semantic_word_overrides()
    membership: Dict[str, set[str]] = {}
    for predicate in predicates:
        gloss = _category_gloss_without_incidental_disjunction(predicate.get("rule_gloss") or "").lower()
        if not gloss:
            continue
        words = {
            _normalize_word_surface(word)
            for word in (predicate.get("word_scores") or {})
        }
        override = overrides.get(gloss) or {}
        words.update(override.get("positive", set()))
        words.difference_update(override.get("negative", set()))
        membership.setdefault(gloss, set()).update(words)
    return membership


class SemanticValidator:
    """High-precision semantic guardrail for sourced norm artifacts."""

    def __init__(
        self,
        *,
        overrides: Dict[str, Dict[str, set[str]]] | None = None,
        predicates: Sequence[Dict[str, Any]] | None = None,
    ) -> None:
        self.overrides = overrides if overrides is not None else _load_semantic_word_overrides()
        self.membership_by_gloss = _semantic_predicate_membership_by_gloss(predicates)
        self._word_concept_cache: Dict[str, set[str]] = {}
        self._word_domain_cache: Dict[str, set[str]] = {}

    def _gloss_concepts(self, gloss: Any) -> set[str]:
        concepts: set[str] = set()
        for token in _source_norm_concept_tokens_from_gloss(gloss):
            alias = SEMANTIC_CONCEPT_ALIASES.get(token)
            if alias:
                concepts.add(alias)
        return concepts

    def _word_concepts(self, word: Any) -> set[str]:
        normalized_word = _normalize_word_surface(word)
        cached = self._word_concept_cache.get(normalized_word)
        if cached is not None:
            return set(cached)

        concepts: set[str] = set()
        for gloss, words in self.membership_by_gloss.items():
            if normalized_word in words:
                concepts.update(self._gloss_concepts(gloss))
        for gloss, override in self.overrides.items():
            if normalized_word in override.get("positive", set()):
                concepts.update(self._gloss_concepts(gloss))

        self._word_concept_cache[normalized_word] = set(concepts)
        return concepts

    def _gloss_domains(self, gloss: Any) -> set[str]:
        tokens = _source_norm_concept_tokens_from_gloss(gloss)
        domains: set[str] = set()
        for domain, keywords in SEMANTIC_DOMAIN_KEYWORDS.items():
            if tokens & keywords:
                domains.add(domain)
        return domains

    def _word_domains(self, word: Any) -> set[str]:
        normalized_word = _normalize_word_surface(word)
        cached = self._word_domain_cache.get(normalized_word)
        if cached is not None:
            return set(cached)

        domains: set[str] = set()
        for gloss, words in self.membership_by_gloss.items():
            if normalized_word in words:
                domains.update(self._gloss_domains(gloss))
        for gloss, override in self.overrides.items():
            if normalized_word in override.get("positive", set()):
                domains.update(self._gloss_domains(gloss))

        self._word_domain_cache[normalized_word] = set(domains)
        return domains

    def _has_artifact_evidence(self, word: Any) -> bool:
        normalized_word = _normalize_word_surface(word)
        word_concepts = self._word_concepts(normalized_word)
        word_domains = self._word_domains(normalized_word)
        return (
            normalized_word in HIGH_RISK_CURATED_ARTIFACT_WORDS
            or bool(word_concepts & HIGH_RISK_ARTIFACT_CONCEPTS)
            or bool(word_domains & HIGH_RISK_ARTIFACT_DOMAINS)
        )

    def _has_animal_evidence(self, word: Any) -> bool:
        word_concepts = self._word_concepts(word)
        word_domains = self._word_domains(word)
        return bool(
            word_domains & {"animal"}
            or word_concepts & {"animal", "bird", "fish", "mammal", "insect", "reptile"}
        )

    def _has_food_evidence(self, word: Any) -> bool:
        return bool(self._word_domains(word) & {"food"})

    def _has_bodypart_evidence(self, word: Any) -> bool:
        word_concepts = self._word_concepts(word)
        word_domains = self._word_domains(word)
        return bool(
            word_concepts & HIGH_RISK_BODYPART_CONCEPTS
            or word_domains & HIGH_RISK_BODYPART_DOMAINS
        )

    def _has_material_variable_artifact_evidence(self, word: Any) -> bool:
        word_concepts = self._word_concepts(word)
        word_domains = self._word_domains(word)
        return bool(
            word_domains & {"artifact", "clothing"}
            or word_concepts & HIGH_RISK_ARTIFACT_CONCEPTS
        )

    def _is_transportation_part_word(self, word: Any) -> bool:
        return _normalize_word_surface(word) in SOURCE_NORM_TRANSPORTATION_PART_WORDS

    def rated_property_plausibility_issues(
        self,
        word: Any,
        gloss: Any,
        value: Any,
        *,
        basis: str = "",
    ) -> List[str]:
        clean_gloss = _source_norm_clean_gloss(gloss)
        if clean_gloss not in SOURCE_NORM_HIGH_RISK_RATED_PROPERTY_GLOSSES:
            return []

        normalized_word = _normalize_word_surface(word)
        artifact_evidence = self._has_artifact_evidence(normalized_word)
        animal_evidence = self._has_animal_evidence(normalized_word)
        food_evidence = self._has_food_evidence(normalized_word)
        bodypart_evidence = self._has_bodypart_evidence(normalized_word)
        issues: List[str] = []

        if value is True:
            if clean_gloss == "stationary things":
                if artifact_evidence:
                    issues.append("artifact_positive_for_stationary_things")
                if animal_evidence:
                    issues.append("animal_positive_for_stationary_things")
                if food_evidence:
                    issues.append("food_positive_for_stationary_things")
            if artifact_evidence and clean_gloss in {
                "living things",
                "natural things",
                "non-manmade things",
            }:
                issues.append(f"artifact_positive_for_{_source_norm_slug(clean_gloss)}")
            if bodypart_evidence and clean_gloss == "living things":
                issues.append("bodypart_positive_for_living_things")
            if animal_evidence and clean_gloss in {
                "manmade things",
                "non-natural things",
                "nonliving things",
            }:
                issues.append(f"animal_positive_for_{_source_norm_slug(clean_gloss)}")
        elif value is False:
            if artifact_evidence and clean_gloss in {
                "manmade things",
                "non-natural things",
            }:
                issues.append(f"artifact_failed_{_source_norm_slug(clean_gloss)}")
            if animal_evidence and clean_gloss in {
                "living things",
                "movable things",
                "natural things",
                "non-manmade things",
            }:
                issues.append(f"animal_failed_{_source_norm_slug(clean_gloss)}")
        return sorted(set(issues))

    def _apply_rated_property_plausibility_gate(
        self,
        word: Any,
        gloss: Any,
        entry: Dict[str, Any],
    ) -> Dict[str, Any]:
        if entry.get("basis") in {"override_negative", "override_positive"}:
            return entry
        issues = self.rated_property_plausibility_issues(
            word,
            gloss,
            entry.get("value"),
            basis=str(entry.get("basis") or ""),
        )
        if not issues:
            return entry
        return {
            "value": None,
            "basis": "rated_property_plausibility_conflict",
            "plausibility_issues": issues,
            "rejected_evidence": dict(entry),
        }

    def _conservative_negative_basis(self, word: Any, gloss: Any) -> str | None:
        word_concepts = self._word_concepts(word)
        word_domains = self._word_domains(word)
        if not word_concepts and not word_domains:
            return None
        target_concepts = self._gloss_concepts(gloss)
        target_domains = self._gloss_domains(gloss)
        clean_gloss = _source_norm_clean_gloss(gloss)
        if target_concepts & word_concepts:
            return None
        animal_concepts = {"bird", "fish", "mammal", "insect", "reptile"}
        if "4 legs" in clean_gloss and word_concepts & {"mammal"}:
            return None
        if ("fur" in clean_gloss or "furry" in clean_gloss) and word_concepts & {"mammal"}:
            return None
        if "movable" in clean_gloss and (
            word_domains & {"animal", "artifact"}
            or word_concepts & animal_concepts
        ):
            return None
        if "nonliving" in clean_gloss and (
            word_concepts & (animal_concepts | {"living"})
            or word_domains & {"animal"}
        ):
            return None
        if "manmade" in clean_gloss and not clean_gloss.startswith("non-manmade") and (
            word_concepts & animal_concepts
            or word_domains & {"animal"}
        ):
            return None
        if "non-natural" in clean_gloss and (
            word_concepts & animal_concepts
            or word_domains & {"animal"}
        ):
            return None
        if "4 legs" in clean_gloss and word_concepts & {
            "bird",
            "fish",
            "insect",
            "vehicle",
            "clothing",
            "furniture",
            "bodypart",
            "musicalinstrument",
            "tool",
            "weapon",
            "utensil",
        }:
            return "conservative_taxonomy_property_exclusion"
        if ("fur" in clean_gloss or "furry" in clean_gloss) and word_concepts & {
            "bird",
            "fish",
            "insect",
            "reptile",
            "vehicle",
            "clothing",
            "furniture",
            "bodypart",
            "musicalinstrument",
            "tool",
            "weapon",
            "utensil",
        }:
            return "conservative_taxonomy_property_exclusion"
        if ("feathers" in clean_gloss or "beak" in clean_gloss) and word_concepts & {
            "fish",
            "mammal",
            "insect",
            "reptile",
            "vehicle",
            "clothing",
            "furniture",
            "bodypart",
            "musicalinstrument",
            "tool",
            "weapon",
            "utensil",
        }:
            return "conservative_taxonomy_property_exclusion"
        if "fly" in clean_gloss and (
            word_domains & {"bodypart", "clothing", "food", "plant"}
            or (word_domains & {"artifact"} and not (word_concepts & {"vehicle"}))
        ):
            return "conservative_taxonomy_property_exclusion"
        if "lay eggs" in clean_gloss and (
            word_concepts & {"mammal"}
            or word_domains & {"artifact", "bodypart", "clothing", "food", "plant"}
        ):
            return "conservative_taxonomy_property_exclusion"
        if any(
            fragment in clean_gloss
            for fragment in (
                "breakfast food",
                "candy",
                "condiment",
                "dessert",
                "drink",
                "seafood",
            )
        ) and word_domains & {"animal", "artifact", "bodypart", "clothing", "plant"}:
            return "conservative_domain_exclusion"
        if any(fragment in clean_gloss for fragment in ("fruit", "vegetable")) and word_domains & {
            "animal",
            "artifact",
            "bodypart",
            "clothing",
        }:
            return "conservative_domain_exclusion"
        if "edible" in clean_gloss and word_domains & {
            "artifact",
            "bodypart",
            "clothing",
        }:
            return "conservative_domain_exclusion"
        if ("juicy" in clean_gloss or "taste" in clean_gloss) and word_domains & {
            "artifact",
            "bodypart",
            "clothing",
        }:
            return "conservative_domain_exclusion"
        if clean_gloss in SOURCE_NORM_MATERIAL_COMPONENT_GLOSSES:
            if self._has_material_variable_artifact_evidence(word):
                return None
            if word_domains & {"animal", "bodypart", "food", "plant"}:
                return "conservative_domain_exclusion"
        if clean_gloss in SOURCE_NORM_TRANSPORTATION_FUNCTION_GLOSSES:
            if self._is_transportation_part_word(word):
                return None
        if not target_concepts:
            target_concepts = set()
        if target_domains and word_domains and not (target_domains & word_domains):
            for left in target_domains:
                for right in word_domains:
                    if frozenset({left, right}) in SEMANTIC_DOMAIN_EXCLUSIONS:
                        return "conservative_domain_exclusion"
        if not target_concepts:
            return None
        for exclusion_group in SEMANTIC_EXCLUSION_GROUPS:
            target_hits = target_concepts & exclusion_group
            if not target_hits:
                continue
            incompatible_hits = (word_concepts & exclusion_group) - target_hits
            if incompatible_hits:
                return "conservative_mutual_exclusion"
        return None

    def component_evidence_profile(
        self,
        word: Any,
        component_glosses: Sequence[Any],
        component_words_by_index: Sequence[Sequence[Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        normalized_word = _normalize_word_surface(word)
        component_words_by_index = component_words_by_index or []
        profile: List[Dict[str, Any]] = []
        for index, raw_gloss in enumerate(component_glosses):
            gloss = _category_gloss_without_incidental_disjunction(raw_gloss).lower()
            override = self.overrides.get(gloss) or {}
            if normalized_word in override.get("negative", set()):
                profile.append({"value": False, "basis": "override_negative"})
                continue
            if normalized_word in override.get("positive", set()):
                profile.append({"value": True, "basis": "override_positive"})
                continue
            if index < len(component_words_by_index) and any(
                normalized_word == _normalize_word_surface(item)
                for item in component_words_by_index[index]
            ):
                profile.append(
                    self._apply_rated_property_plausibility_gate(
                        normalized_word,
                        gloss,
                        {"value": True, "basis": "bundle_component_words"},
                    )
                )
                continue
            if normalized_word in self.membership_by_gloss.get(gloss, set()):
                profile.append(
                    self._apply_rated_property_plausibility_gate(
                        normalized_word,
                        gloss,
                        {"value": True, "basis": "source_positive"},
                    )
                )
                continue
            negative_basis = self._conservative_negative_basis(normalized_word, gloss)
            if negative_basis:
                profile.append(
                    self._apply_rated_property_plausibility_gate(
                        normalized_word,
                        gloss,
                        {"value": False, "basis": negative_basis},
                    )
                )
                continue
            profile.append({"value": None, "basis": "unknown"})
        return profile

    def component_profile(
        self,
        word: Any,
        component_glosses: Sequence[Any],
        component_words_by_index: Sequence[Sequence[Any]] | None = None,
    ) -> List[bool]:
        return [
            evidence["value"] is True
            for evidence in self.component_evidence_profile(
                word,
                component_glosses,
                component_words_by_index=component_words_by_index,
            )
        ]


_SEMANTIC_VALIDATOR_CACHE: SemanticValidator | None = None
_SEMANTIC_BUNDLE_WORD_PROFILE_CACHE: Dict[tuple[str, str, tuple[str, ...]], List[bool]] = {}
_SEMANTIC_BUNDLE_WORD_EVIDENCE_CACHE: Dict[tuple[str, str, tuple[str, ...]], List[Dict[str, Any]]] = {}
_BUNDLE_STRICT_NEAR_MISS_RECORD_CACHE: Dict[str, List[tuple[int, str]]] = {}


def get_default_semantic_validator() -> SemanticValidator:
    global _SEMANTIC_VALIDATOR_CACHE
    if _SEMANTIC_VALIDATOR_CACHE is None:
        _SEMANTIC_VALIDATOR_CACHE = SemanticValidator()
    return _SEMANTIC_VALIDATOR_CACHE


def validated_component_profile(
    word: Any,
    component_glosses: Sequence[Any],
    overrides: Dict[str, Dict[str, set[str]]] | None = None,
) -> List[bool]:
    validator = SemanticValidator(overrides=overrides) if overrides is not None else get_default_semantic_validator()
    return validator.component_profile(word, component_glosses)


def _normalize_row_word_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    if isinstance(normalized.get("query_words"), list):
        normalized["query_words"] = _normalize_word_list(normalized["query_words"])
    if isinstance(normalized.get("support_groups_by_label"), dict):
        normalized["support_groups_by_label"] = {
            label: _normalize_word_blocks(groups)
            for label, groups in normalized["support_groups_by_label"].items()
        }
    if isinstance(normalized.get("gold_groups"), dict):
        normalized["gold_groups"] = {
            label: _normalize_word_list(words)
            for label, words in normalized["gold_groups"].items()
        }
    if isinstance(normalized.get("rule_glosses"), dict):
        normalized["rule_glosses"] = {
            label: _category_gloss_without_incidental_disjunction(gloss)
            for label, gloss in normalized["rule_glosses"].items()
        }
    if isinstance(normalized.get("out_of_category_groups_by_label"), dict):
        normalized["out_of_category_groups_by_label"] = {
            label: _normalize_word_blocks(groups)
            for label, groups in normalized["out_of_category_groups_by_label"].items()
        }
    if isinstance(normalized.get("candidate_groups"), list):
        normalized["candidate_groups"] = [
            {
                **group,
                "words": _normalize_word_list(group.get("words") or []),
            }
            if isinstance(group, dict)
            else group
            for group in normalized["candidate_groups"]
        ]
    return normalized


DESIGN_CELL_SPECS = {
    ("shared", "d1"): {
        "logic_condition": "shared",
        "difficulty": "d1",
        "difficulty_name": "easy",
        "category_operator": "single",
        "category_arity": 1,
        "shared_d1": True,
        "condition_count": 1,
    },
    ("either", "d2"): {
        "logic_condition": "either",
        "difficulty": "d2",
        "difficulty_name": "medium",
        "category_operator": "or",
        "category_arity": 2,
        "shared_d1": False,
        "condition_count": 2,
    },
    ("either", "d3"): {
        "logic_condition": "either",
        "difficulty": "d3",
        "difficulty_name": "hard",
        "category_operator": "or",
        "category_arity": 3,
        "shared_d1": False,
        "condition_count": 3,
    },
    ("both", "d2"): {
        "logic_condition": "both",
        "difficulty": "d2",
        "difficulty_name": "medium",
        "category_operator": "and",
        "category_arity": 2,
        "shared_d1": False,
        "condition_count": 2,
    },
    ("both", "d3"): {
        "logic_condition": "both",
        "difficulty": "d3",
        "difficulty_name": "hard",
        "category_operator": "and",
        "category_arity": 3,
        "shared_d1": False,
        "condition_count": 3,
    },
}

ATOMIC_HARD_KNOWLEDGE_TYPES = {
    "associative_relations",
    "encyclopedic",
    "multiword_expression",
    "word_meaning_plus_word_form",
}

COMPOSITE_KNOWLEDGE_TYPES = {
    "two_parts_composite",
    "two_way_intersection",
    "three_parts_composite",
    "three_way_intersection",
}

HARD_KNOWLEDGE_TYPES = ATOMIC_HARD_KNOWLEDGE_TYPES | COMPOSITE_KNOWLEDGE_TYPES

SURFACE_FORM_META_GROUPS = {
    "meta::homophones",
    "meta::starts_with_hidden_class",
    "meta::anagrams",
    "meta::palindromes",
    "meta::ends_with_hidden_class",
}

KNOWLEDGE_TYPE_DIFFICULTY = {
    "semantic": 1.0,
    "taxonomic": 1.0,
    "associative_relations": 2.0,
    "encyclopedic": 2.0,
    "multiword_expression": 3.0,
    "word_meaning_plus_word_form": 3.0,
    "two_parts_composite": 3.5,
    "two_way_intersection": 5.0,
    "three_parts_composite": 4.5,
    "three_way_intersection": 6.0,
}

COMPOSITE_PART_COUNTS = {
    "two_parts_composite": 2,
    "two_way_intersection": 2,
    "three_parts_composite": 3,
    "three_way_intersection": 3,
}

COMPOSITE_RECIPE_COUNTS = {
    "d2": (2, 2),
    "d3": (2, 1, 1),
}

COUNT_WORD = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
}

COMPLEXITY_SCORE = {
    "simple": 1.0,
    "medium": 2.0,
    "complex": 3.0,
}

D1_NEGATIVE_SCHEDULE = {
    "two_label_swap": 0.5,
    "out_of_category": 0.5,
}

D2_NEGATIVE_SCHEDULE = {
    "two_label_swap": 1 / 3,
    "three_cycle": 1 / 3,
    "out_of_category": 1 / 3,
}

D3_NEGATIVE_SCHEDULE = {
    "two_label_swap": 0.25,
    "three_cycle": 0.25,
    "four_cycle": 0.25,
    "out_of_category": 0.25,
}

EITHER_D2_NEGATIVE_SCHEDULE = {
    "two_label_swap": 1 / 3,
    "three_cycle": 1 / 3,
    "none_of_components_out_of_category": 1 / 3,
}

EITHER_D3_NEGATIVE_SCHEDULE = {
    "two_label_swap": 0.25,
    "three_cycle": 0.25,
    "four_cycle": 0.25,
    "none_of_components_out_of_category": 0.25,
}

BOTH_D2_NEGATIVE_SCHEDULE = {
    "two_label_swap": 0.25,
    "three_cycle": 0.25,
    "component_near_miss": 0.25,
    "general_out_of_category": 0.25,
}

BOTH_D3_NEGATIVE_SCHEDULE = {
    "two_label_swap": 0.20,
    "three_cycle": 0.20,
    "four_cycle": 0.20,
    "component_near_miss": 0.20,
    "general_out_of_category": 0.20,
}

OUT_OF_CATEGORY_LABEL = "OUT_OF_CATEGORY"
OUT_OF_CATEGORY_GROUPS_PER_LABEL = 6

DESIGN_CELL_CONFIG = {
    ("shared", "d1"): {
        "min_red_herring_collisions": 0,
        "preferred_types": {"semantic", "associative_relations", "encyclopedic"},
        "min_preferred_types": 4,
        "negative_schedule": D1_NEGATIVE_SCHEDULE,
    },
    ("either", "d2"): {
        "min_red_herring_collisions": 0,
        "preferred_types": {"two_parts_composite"},
        "min_preferred_types": 4,
        "required_composite_parts": 2,
        "negative_schedule": EITHER_D2_NEGATIVE_SCHEDULE,
    },
    ("either", "d3"): {
        "min_red_herring_collisions": 0,
        "preferred_types": {"three_parts_composite"},
        "min_preferred_types": 4,
        "required_composite_parts": 3,
        "negative_schedule": EITHER_D3_NEGATIVE_SCHEDULE,
    },
    ("both", "d2"): {
        "min_red_herring_collisions": 0,
        "preferred_types": {"two_way_intersection"},
        "min_preferred_types": 4,
        "required_composite_parts": 2,
        "negative_schedule": BOTH_D2_NEGATIVE_SCHEDULE,
    },
    ("both", "d3"): {
        "min_red_herring_collisions": 0,
        "preferred_types": {"three_way_intersection"},
        "min_preferred_types": 4,
        "required_composite_parts": 3,
        "negative_schedule": BOTH_D3_NEGATIVE_SCHEDULE,
    },
}

TITLE_STOPWORDS = {
    "A",
    "AN",
    "THE",
    "OF",
    "FOR",
    "TO",
    "IN",
    "ON",
    "AT",
    "THAT",
    "THATS",
    "AREN",
    "ARENT",
    "WITH",
    "WORD",
    "WORDS",
    "THING",
    "THINGS",
    "KIND",
    "KINDS",
    "TYPE",
    "TYPES",
    "PART",
    "PARTS",
    "NAME",
    "NAMES",
    "MAKE",
    "UP",
    "SAME",
}

AUTO_BUNDLE_BUILDER = "paper_source_auto_v1"
SOURCE_NORM_TARGET_BUNDLES_PER_CELL = 60
TARGET_BUNDLES_PER_DESIGN_CELL = {
    ("shared", "d1"): SOURCE_NORM_TARGET_BUNDLES_PER_CELL,
    ("either", "d2"): SOURCE_NORM_TARGET_BUNDLES_PER_CELL,
    ("either", "d3"): SOURCE_NORM_TARGET_BUNDLES_PER_CELL,
    ("both", "d2"): SOURCE_NORM_TARGET_BUNDLES_PER_CELL,
    ("both", "d3"): SOURCE_NORM_TARGET_BUNDLES_PER_CELL,
}

TARGET_BUNDLE_COUNTS_BY_TYPE = {
    ("shared", "d1"): {
        "semantic": SOURCE_NORM_TARGET_BUNDLES_PER_CELL,
    },
    ("either", "d2"): {
        "two_parts_composite": TARGET_BUNDLES_PER_DESIGN_CELL[("either", "d2")],
    },
    ("either", "d3"): {
        "three_parts_composite": TARGET_BUNDLES_PER_DESIGN_CELL[("either", "d3")],
    },
    ("both", "d2"): {
        "two_way_intersection": TARGET_BUNDLES_PER_DESIGN_CELL[("both", "d2")],
    },
    ("both", "d3"): {
        "three_way_intersection": TARGET_BUNDLES_PER_DESIGN_CELL[("both", "d3")],
    },
}

CATEGORY_FAMILY_TAG_MAP = {
    "abstract": "abstract_and_social",
    "actions": "actions_and_events",
    "animals": "life_science",
    "astronomy": "science_and_measurement",
    "award": "arts_and_media",
    "birds": "life_science",
    "books": "arts_and_media",
    "body": "body_and_health",
    "browse": "arts_and_media",
    "calendar": "time_and_calendar",
    "cities": "places",
    "clothing": "objects_and_tools",
    "color": "visual_and_material",
    "computer": "technology",
    "countries": "places",
    "dance": "arts_and_media",
    "education": "education",
    "food": "food_and_drink",
    "fish": "life_science",
    "games": "sports_and_games",
    "geography": "places",
    "health": "body_and_health",
    "history": "history_and_culture",
    "household": "objects_and_tools",
    "kitchen": "objects_and_tools",
    "kit": "objects_and_tools",
    "language": "language_and_symbols",
    "legal": "abstract_and_social",
    "literature": "arts_and_media",
    "materials": "visual_and_material",
    "measurement": "science_and_measurement",
    "medical": "body_and_health",
    "music": "arts_and_media",
    "mythology": "history_and_culture",
    "mystic": "abstract_and_social",
    "names": "people_and_names",
    "nature": "nature_and_weather",
    "object": "objects_and_tools",
    "office": "objects_and_tools",
    "people": "people_and_names",
    "plant": "life_science",
    "plants": "life_science",
    "roles": "people_and_names",
    "reptiles": "life_science",
    "science": "science_and_measurement",
    "school": "education",
    "shape": "visual_and_material",
    "sign": "language_and_symbols",
    "sky": "nature_and_weather",
    "space": "science_and_measurement",
    "sports": "sports_and_games",
    "technology": "technology",
    "theater": "arts_and_media",
    "tool": "objects_and_tools",
    "transport": "transportation",
    "weather": "nature_and_weather",
    "work": "people_and_names",
}

CATEGORY_FAMILY_TEXT_PATTERNS = [
    (re.compile(r"\b(animal|animals|bird|birds|fish|insect|mammal|mammals|flower|tree|plant|zodiac|wings|beak|feathers|fur|tail|lays eggs|live in water|swims|flies)\b", re.I), "life_science"),
    (re.compile(r"\b(food|foods|fruit|fruits|vegetable|vegetables|drink|kitchen|cooking|edible|taste|tastes|sweet|juicy)\b", re.I), "food_and_drink"),
    (re.compile(r"\b(city|state|country|place|geography|river|airport)\b", re.I), "places"),
    (re.compile(r"\b(music|dance|opera|theater|literature|shakespeare|book|movie|art)\b", re.I), "arts_and_media"),
    (re.compile(r"\b(science|chemical|element|unit|measurement|astronomy|planet|moon|constellation)\b", re.I), "science_and_measurement"),
    (re.compile(r"\b(name|given|surname|people|person|occupation|role)\b", re.I), "people_and_names"),
    (re.compile(r"\b(sport|game|team|mascot)\b", re.I), "sports_and_games"),
    (re.compile(r"\b(weather|climate|sky|storm|sound)\b", re.I), "nature_and_weather"),
    (re.compile(r"\b(color|colour|colours|shape|material|gem|mineral|green|brown|round|long|metal|plastic|wood)\b", re.I), "visual_and_material"),
    (re.compile(r"\b(tool|tools|object|furniture|clothing|household|office|handle|wheels|appliance|hardware|graspable|holdable|manmade|nonliving)\b", re.I), "objects_and_tools"),
    (re.compile(r"\b(computer|programming|keyboard|technology|interface)\b", re.I), "technology"),
    (re.compile(r"\b(myth|deity|historical|president|biblical|classical|weapon|weapons|killing)\b", re.I), "history_and_culture"),
    (re.compile(r"\b(legal|virtue|abstract|social|pleasant|unpleasant|precious|dangerous)\b", re.I), "abstract_and_social"),
    (re.compile(r"\b(action|verb|event|movable|stationary|fast|loud)\b", re.I), "actions_and_events"),
    (re.compile(r"\b(letter|symbol|word|language)\b", re.I), "language_and_symbols"),
    (re.compile(r"\b(vehicle|vehicles|transport|transportation|watercraft)\b", re.I), "transportation"),
    (re.compile(r"\b(body|health|anatomy)\b", re.I), "body_and_health"),
    (re.compile(r"\b(month|calendar|time)\b", re.I), "time_and_calendar"),
]

ATOMIC_INGREDIENT_TARGET_BUNDLE_COUNTS_BY_TYPE = {
    "d1": {
        "semantic": 17,
        "encyclopedic": 8,
        "associative_relations": 5,
    },
    "d2": {
        "associative_relations": 8,
        "encyclopedic": 11,
        "multiword_expression": 11,
    },
    "d3": {
        "associative_relations": 5,
        "encyclopedic": 5,
        "multiword_expression": 20,
    },
}

_VALID_QUARTET_CACHE: Dict[tuple[str, str, tuple[str, ...]], List[tuple[str, str, str, str]]] = {}
_SCORED_QUARTET_CACHE: Dict[tuple[str, str, tuple[str, ...]], List[Dict[str, Any]]] = {}
_SELECTION_VALIDITY_CACHE: Dict[tuple[str, str, tuple[str, ...]], bool] = {}
_QUARTET_SELECTION_POOL_CACHE: Dict[tuple[str, int], List[Dict[str, Any]]] = {}
_SELECTION_POOL_INDEX_CACHE: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
_SELECTION_POOL_DEGREE_CACHE: Dict[int, Dict[str, int]] = {}
_OUT_OF_CATEGORY_BLOCK_CACHE: Dict[int, List[Dict[str, Any]]] = {}
_GENERAL_OUT_OF_CATEGORY_CANDIDATE_CACHE: Dict[tuple[int, str], List[Dict[str, Any]]] = {}
_EXACT_CATEGORY_BALANCE_SCHEDULE_CACHE: Dict[
    tuple[str, str, int, tuple[str, ...]],
    List[tuple[str, str, str, str]],
] = {}

QUARTET_SELECTION_QUANTILES = {
    "d1": (0.0, 0.5),
    "d2": (0.0, 1.0),
    "d3": (0.0, 1.0),
}

MEAN_GAP_THRESHOLDS = {
    "negative_avg_num_wrong_labels": {"d1_to_d2": 0.25, "d2_to_d3": 0.25},
    "negative_avg_confusability_score": {"d1_to_d2": 0.05, "d2_to_d3": 0.25},
    "episode_rule_lossiness_score": {"d1_to_d2": 0.25, "d2_to_d3": 0.15},
    "episode_boundary_fuzziness_score": {"d1_to_d2": 1.0, "d2_to_d3": 1.0},
    "episode_examples_advantage_proxy": {"d1_to_d2": 1.5, "d2_to_d3": 1.0},
}

QUANTILE_SEPARATION_FEATURES = [
    "episode_rule_lossiness_score",
    "episode_boundary_fuzziness_score",
    "episode_examples_advantage_proxy",
]

NEGATIVE_STRUCTURE_TARGETS = {
    "d1": {
        "negative_all_labels_once_rate": {"min": 1.0},
        "negative_avg_num_wrong_labels": {"min": 1.25, "max": 1.75},
        "negative_out_of_category_rate": {"min": 0.4, "max": 0.6},
    },
    "d2": {
        "negative_all_labels_once_rate": {"min": 1.0},
        "negative_avg_num_wrong_labels": {"min": 1.75, "max": 2.25},
        "negative_out_of_category_rate": {"min": 0.2, "max": 0.55},
    },
    "d3": {
        "negative_all_labels_once_rate": {"min": 1.0},
        "negative_avg_num_wrong_labels": {"min": 2.15, "max": 2.75},
        "negative_out_of_category_rate": {"min": 0.2, "max": 0.45},
    },
}

META_GROUP_CONFIG = {
    "meta::words_before_hidden_term": {
        "knowledge_type": "multiword_expression",
        "rule_gloss": "words that can appear immediately before the same hidden term in familiar expressions",
        "tags": ["phrase", "before_hidden", "compound"],
        "allowed_knowledge_types": {"multiword_expression"},
    },
    "meta::words_after_hidden_term": {
        "knowledge_type": "multiword_expression",
        "rule_gloss": "words that can appear immediately after the same hidden term in familiar expressions",
        "tags": ["phrase", "after_hidden", "compound"],
        "allowed_knowledge_types": {"multiword_expression"},
    },
    "meta::homophones": {
        "knowledge_type": "word_meaning_plus_word_form",
        "rule_gloss": "words that sound like members of the same hidden category",
        "tags": ["surface_form", "homophone", "sound"],
        "allowed_knowledge_types": {"word_meaning_plus_word_form"},
    },
    "meta::starts_with_hidden_class": {
        "knowledge_type": "word_meaning_plus_word_form",
        "rule_gloss": "words that begin with items from the same hidden semantic class",
        "tags": ["surface_form", "starts_with", "hidden_class"],
        "allowed_knowledge_types": {"word_meaning_plus_word_form"},
    },
    "meta::anagrams": {
        "knowledge_type": "word_meaning_plus_word_form",
        "rule_gloss": "words whose letters can be rearranged into members of the same hidden category",
        "tags": ["surface_form", "anagram", "letters"],
        "allowed_knowledge_types": {"word_meaning_plus_word_form"},
    },
    "meta::palindromes": {
        "knowledge_type": "word_meaning_plus_word_form",
        "rule_gloss": "palindromic words or phrases",
        "tags": ["surface_form", "palindrome", "letters"],
        "allowed_knowledge_types": {"word_meaning_plus_word_form", "semantic"},
    },
    "meta::ends_with_hidden_class": {
        "knowledge_type": "word_meaning_plus_word_form",
        "rule_gloss": "words that end with items from the same hidden semantic class",
        "tags": ["surface_form", "ends_with", "hidden_class"],
        "allowed_knowledge_types": {"word_meaning_plus_word_form"},
    },
}

D1_SEMANTIC_BUNDLE_SPECS = [
    {
        "id": "common_fruits",
        "gloss": "common fruit names",
        "tags": ["semantic", "food", "fruit"],
        "blocks": [
            ["APPLE", "BANANA", "PEAR", "PEACH"],
            ["PLUM", "MANGO", "PAPAYA", "GRAPE"],
            ["LEMON", "LIME", "CHERRY", "APRICOT"],
        ],
    },
    {
        "id": "common_vegetables",
        "gloss": "common vegetable names",
        "tags": ["semantic", "food", "vegetable"],
        "blocks": [
            ["CARROT", "POTATO", "ONION", "CELERY"],
            ["BROCCOLI", "CABBAGE", "SPINACH", "LETTUCE"],
            ["PEA", "CORN", "BEET", "RADISH"],
        ],
    },
    {
        "id": "farm_animals",
        "gloss": "animals commonly kept on farms",
        "tags": ["semantic", "animals", "farm"],
        "blocks": [
            ["COW", "HORSE", "PIG", "SHEEP"],
            ["GOAT", "CHICKEN", "TURKEY", "DONKEY"],
            ["YAK", "ALPACA", "OX", "MULE"],
        ],
    },
    {
        "id": "wild_mammals",
        "gloss": "wild mammal names",
        "tags": ["semantic", "animals", "wild"],
        "blocks": [
            ["LION", "TIGER", "BEAR", "WOLF"],
            ["FOX", "DEER", "MOOSE", "ZEBRA"],
            ["GIRAFFE", "RHINO", "HIPPO", "ELEPHANT"],
        ],
    },
    {
        "id": "bird_names",
        "gloss": "bird names",
        "tags": ["semantic", "animals", "birds"],
        "blocks": [
            ["EAGLE", "FALCON", "HAWK", "ROBIN"],
            ["SPARROW", "CROW", "RAVEN", "SWAN"],
            ["HERON", "OWL", "PELICAN", "PENGUIN"],
        ],
    },
    {
        "id": "insect_names",
        "gloss": "insect names",
        "tags": ["semantic", "animals", "insects"],
        "blocks": [
            ["ANT", "BEE", "FLY", "MOTH"],
            ["WASP", "BEETLE", "CRICKET", "CICADA"],
            ["LOCUST", "TERMITE", "DRAGONFLY", "MOSQUITO"],
        ],
    },
    {
        "id": "basic_colors",
        "gloss": "color names",
        "tags": ["semantic", "color", "visual"],
        "blocks": [
            ["RED", "BLUE", "GREEN", "YELLOW"],
            ["PURPLE", "ORANGE", "BLACK", "WHITE"],
            ["PINK", "BROWN", "GRAY", "VIOLET"],
        ],
    },
    {
        "id": "geometric_shapes",
        "gloss": "geometric shape names",
        "tags": ["semantic", "shape", "geometry"],
        "blocks": [
            ["CIRCLE", "SQUARE", "TRIANGLE", "RECTANGLE"],
            ["OVAL", "CUBE", "SPHERE", "CONE"],
            ["CYLINDER", "DIAMOND", "PENTAGON", "HEXAGON"],
        ],
    },
    {
        "id": "weather_words",
        "gloss": "weather event or condition names",
        "tags": ["semantic", "weather", "nature"],
        "blocks": [
            ["RAIN", "SNOW", "HAIL", "SLEET"],
            ["FOG", "WIND", "THUNDER", "LIGHTNING"],
            ["DRIZZLE", "BLIZZARD", "TORNADO", "HURRICANE"],
        ],
    },
    {
        "id": "musical_instruments",
        "gloss": "musical instrument names",
        "tags": ["semantic", "music", "instrument"],
        "blocks": [
            ["PIANO", "GUITAR", "VIOLIN", "DRUMS"],
            ["FLUTE", "TRUMPET", "CELLO", "HARP"],
            ["OBOE", "CLARINET", "TUBA", "SAXOPHONE"],
        ],
    },
    {
        "id": "sports_equipment",
        "gloss": "sports equipment names",
        "tags": ["semantic", "sports", "equipment"],
        "blocks": [
            ["BAT", "BALL", "GLOVE", "HELMET"],
            ["RACKET", "NET", "SKATES", "PADDLE"],
            ["CLUB", "GOAL", "CLEATS", "WHISTLE"],
        ],
    },
    {
        "id": "kitchen_tools",
        "gloss": "kitchen tool or dishware names",
        "tags": ["semantic", "kitchen", "object"],
        "blocks": [
            ["SPOON", "FORK", "KNIFE", "PLATE"],
            ["BOWL", "PAN", "POT", "CUP"],
            ["WHISK", "TONGS", "LADLE", "STRAINER"],
        ],
    },
    {
        "id": "school_supplies",
        "gloss": "school supply names",
        "tags": ["semantic", "school", "object"],
        "blocks": [
            ["PENCIL", "PEN", "ERASER", "RULER"],
            ["NOTEBOOK", "BINDER", "MARKER", "CRAYON"],
            ["FOLDER", "GLUE", "SCISSORS", "CALCULATOR"],
        ],
    },
    {
        "id": "clothing_items",
        "gloss": "clothing item names",
        "tags": ["semantic", "clothing", "object"],
        "blocks": [
            ["SHIRT", "PANTS", "COAT", "DRESS"],
            ["SKIRT", "SOCKS", "SHOES", "HAT"],
            ["SCARF", "GLOVES", "BELT", "SWEATER"],
        ],
    },
    {
        "id": "furniture_items",
        "gloss": "furniture item names",
        "tags": ["semantic", "furniture", "object"],
        "blocks": [
            ["CHAIR", "TABLE", "SOFA", "BED"],
            ["DESK", "DRESSER", "CABINET", "STOOL"],
            ["BENCH", "SHELF", "COUCH", "FUTON"],
        ],
    },
    {
        "id": "vehicle_names",
        "gloss": "vehicle names",
        "tags": ["semantic", "transport", "object"],
        "blocks": [
            ["CAR", "TRUCK", "BUS", "TRAIN"],
            ["PLANE", "BOAT", "BICYCLE", "SCOOTER"],
            ["MOTORCYCLE", "SUBWAY", "TRAM", "FERRY"],
        ],
    },
    {
        "id": "hand_tools",
        "gloss": "hand tool names",
        "tags": ["semantic", "tool", "object"],
        "blocks": [
            ["HAMMER", "SAW", "DRILL", "WRENCH"],
            ["PLIERS", "CHISEL", "LEVEL", "CLAMP"],
            ["SANDER", "FILE", "AXE", "TROWEL"],
        ],
    },
    {
        "id": "body_parts",
        "gloss": "body part names",
        "tags": ["semantic", "body", "anatomy"],
        "blocks": [
            ["HEAD", "HAND", "ARM", "LEG"],
            ["EYE", "EAR", "NOSE", "MOUTH"],
            ["ELBOW", "KNEE", "SHOULDER", "FOOT"],
        ],
    },
    {
        "id": "occupations",
        "gloss": "occupation names",
        "tags": ["semantic", "roles", "work"],
        "blocks": [
            ["BAKER", "BARBER", "DOCTOR", "FARMER"],
            ["LAWYER", "MASON", "NURSE", "TEACHER"],
            ["ACTOR", "CARPENTER", "CHEF", "PILOT"],
        ],
    },
    {
        "id": "common_given_names",
        "gloss": "common given names",
        "tags": ["semantic", "people", "names"],
        "blocks": [
            ["ALEX", "DAVID", "EMMA", "GRACE"],
            ["JAMES", "LAURA", "MARIA", "PETER"],
            ["HENRY", "JULIA", "OSCAR", "SARAH"],
        ],
    },
    {
        "id": "common_surnames",
        "gloss": "common surnames",
        "tags": ["semantic", "people", "names"],
        "blocks": [
            ["BAKER", "BROWN", "CLARK", "SMITH"],
            ["DAVIS", "JONES", "MILLER", "WILSON"],
            ["COOPER", "FISHER", "MARTIN", "TAYLOR"],
        ],
    },
    {
        "id": "music_genres",
        "gloss": "music genre names",
        "tags": ["semantic", "music", "genre"],
        "blocks": [
            ["BLUES", "COUNTRY", "JAZZ", "ROCK"],
            ["DISCO", "FUNK", "GOSPEL", "REGGAE"],
            ["CLASSICAL", "HIP-HOP", "METAL", "TECHNO"],
        ],
    },
    {
        "id": "flower_names",
        "gloss": "flower names",
        "tags": ["semantic", "plants", "flowers"],
        "blocks": [
            ["DAISY", "IRIS", "LILY", "ROSE"],
            ["JASMINE", "LAVENDER", "ORCHID", "TULIP"],
            ["CAMELLIA", "DAHLIA", "POPPY", "VIOLET"],
        ],
    },
    {
        "id": "tree_names",
        "gloss": "tree names",
        "tags": ["semantic", "plants", "trees"],
        "blocks": [
            ["ASH", "BIRCH", "CEDAR", "MAPLE"],
            ["ELM", "OAK", "PINE", "WILLOW"],
            ["ASPEN", "CYPRESS", "SPRUCE", "SYCAMORE"],
        ],
    },
    {
        "id": "gemstone_names",
        "gloss": "gemstone or mineral names",
        "tags": ["semantic", "gems", "minerals"],
        "blocks": [
            ["AMBER", "JADE", "OPAL", "RUBY"],
            ["BERYL", "CORAL", "GARNET", "PEARL"],
            ["ONYX", "SAPPHIRE", "TOPAZ", "TURQUOISE"],
        ],
    },
    {
        "id": "country_names",
        "gloss": "country names",
        "tags": ["semantic", "geography", "countries"],
        "blocks": [
            ["BRAZIL", "CANADA", "FRANCE", "JAPAN"],
            ["CHILE", "EGYPT", "GREECE", "INDIA"],
            ["KENYA", "MEXICO", "NORWAY", "SPAIN"],
        ],
    },
    {
        "id": "us_state_names",
        "gloss": "U.S. state names",
        "tags": ["semantic", "geography", "states"],
        "blocks": [
            ["ALASKA", "FLORIDA", "GEORGIA", "TEXAS"],
            ["ARIZONA", "MONTANA", "NEVADA", "VIRGINIA"],
            ["INDIANA", "MAINE", "OHIO", "WASHINGTON"],
        ],
    },
    {
        "id": "city_names",
        "gloss": "city names",
        "tags": ["semantic", "geography", "cities"],
        "blocks": [
            ["AUSTIN", "BOSTON", "CHICAGO", "DENVER"],
            ["DALLAS", "HELENA", "MADISON", "PHOENIX"],
            ["ATLANTA", "CLEVELAND", "SEATTLE", "TUCSON"],
        ],
    },
    {
        "id": "celestial_body_names",
        "gloss": "celestial body names",
        "tags": ["semantic", "space", "astronomy"],
        "blocks": [
            ["EARTH", "JUPITER", "MARS", "VENUS"],
            ["EUROPA", "MERCURY", "NEPTUNE", "SATURN"],
            ["CERES", "PLUTO", "TITAN", "URANUS"],
        ],
    },
    {
        "id": "school_subjects",
        "gloss": "school subject names",
        "tags": ["semantic", "education", "subjects"],
        "blocks": [
            ["ART", "HISTORY", "MATH", "SCIENCE"],
            ["BIOLOGY", "CHEMISTRY", "ECONOMICS", "PHYSICS"],
            ["GEOGRAPHY", "LITERATURE", "MUSIC", "PSYCHOLOGY"],
        ],
    },
    {
        "id": "fish_names",
        "gloss": "fish names",
        "tags": ["semantic", "animals", "fish"],
        "blocks": [
            ["SALMON", "TROUT", "BASS", "COD"],
            ["CARP", "PIKE", "PERCH", "TUNA"],
            ["HADDOCK", "HALIBUT", "SARDINE", "ANCHOVY"],
        ],
    },
    {
        "id": "reptile_names",
        "gloss": "reptile names",
        "tags": ["semantic", "animals", "reptiles"],
        "blocks": [
            ["SNAKE", "LIZARD", "TURTLE", "GECKO"],
            ["IGUANA", "COBRA", "PYTHON", "VIPER"],
            ["ALLIGATOR", "CROCODILE", "CHAMELEON", "TORTOISE"],
        ],
    },
    {
        "id": "dog_breeds",
        "gloss": "dog breed names",
        "tags": ["semantic", "animals", "breeds"],
        "blocks": [
            ["BEAGLE", "BOXER", "COLLIE", "POODLE"],
            ["BULLDOG", "DACHSHUND", "HUSKY", "MASTIFF"],
            ["DALMATIAN", "GREYHOUND", "RETRIEVER", "TERRIER"],
        ],
    },
    {
        "id": "dessert_names",
        "gloss": "dessert names",
        "tags": ["semantic", "food", "dessert"],
        "blocks": [
            ["CAKE", "PIE", "TART", "BROWNIE"],
            ["COOKIE", "PUDDING", "CUSTARD", "GELATO"],
            ["SORBET", "CHEESECAKE", "ECLAIR", "TIRAMISU"],
        ],
    },
    {
        "id": "beverage_names",
        "gloss": "beverage names",
        "tags": ["semantic", "food", "drink"],
        "blocks": [
            ["WATER", "TEA", "COFFEE", "JUICE"],
            ["MILK", "SODA", "CIDER", "LEMONADE"],
            ["COCOA", "SMOOTHIE", "ESPRESSO", "SELTZER"],
        ],
    },
    {
        "id": "herbs_and_spices",
        "gloss": "herb or spice names",
        "tags": ["semantic", "food", "plants"],
        "blocks": [
            ["BASIL", "THYME", "CUMIN", "DILL"],
            ["MINT", "OREGANO", "PARSLEY", "ROSEMARY"],
            ["CINNAMON", "PAPRIKA", "SAFFRON", "TURMERIC"],
        ],
    },
    {
        "id": "grains_and_cereals",
        "gloss": "grain or cereal names",
        "tags": ["semantic", "food", "plants"],
        "blocks": [
            ["RICE", "WHEAT", "OATS", "BARLEY"],
            ["CORN", "RYE", "MILLET", "QUINOA"],
            ["SORGHUM", "FARRO", "BULGUR", "AMARANTH"],
        ],
    },
    {
        "id": "sport_names",
        "gloss": "sport names",
        "tags": ["semantic", "sports", "games"],
        "blocks": [
            ["SOCCER", "TENNIS", "BASEBALL", "BASKETBALL"],
            ["HOCKEY", "CRICKET", "RUGBY", "GOLF"],
            ["VOLLEYBALL", "LACROSSE", "BOXING", "SKIING"],
        ],
    },
    {
        "id": "board_game_names",
        "gloss": "board game names",
        "tags": ["semantic", "games", "object"],
        "blocks": [
            ["CHESS", "CHECKERS", "MONOPOLY", "SCRABBLE"],
            ["CLUE", "RISK", "SORRY", "TROUBLE"],
            ["BACKGAMMON", "PARCHEESI", "OTHELLO", "CATAN"],
        ],
    },
    {
        "id": "holiday_names",
        "gloss": "holiday or observance names",
        "tags": ["semantic", "calendar", "culture"],
        "blocks": [
            ["CHRISTMAS", "EASTER", "HALLOWEEN", "THANKSGIVING"],
            ["PASSOVER", "RAMADAN", "HANUKKAH", "DIWALI"],
            ["HOLI", "KWANZAA", "SOLSTICE", "MAY DAY"],
        ],
    },
    {
        "id": "month_names",
        "gloss": "month names",
        "tags": ["semantic", "calendar", "time"],
        "blocks": [
            ["JANUARY", "FEBRUARY", "MARCH", "APRIL"],
            ["MAY", "JUNE", "JULY", "AUGUST"],
            ["SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"],
        ],
    },
    {
        "id": "metal_names",
        "gloss": "metal names",
        "tags": ["semantic", "materials", "science"],
        "blocks": [
            ["IRON", "COPPER", "GOLD", "SILVER"],
            ["TIN", "ZINC", "NICKEL", "LEAD"],
            ["ALUMINUM", "TITANIUM", "PLATINUM", "MERCURY"],
        ],
    },
    {
        "id": "fabric_names",
        "gloss": "fabric names",
        "tags": ["semantic", "materials", "clothing"],
        "blocks": [
            ["COTTON", "SILK", "WOOL", "LINEN"],
            ["DENIM", "VELVET", "SATIN", "TWEED"],
            ["FLANNEL", "FLEECE", "MUSLIN", "NYLON"],
        ],
    },
    {
        "id": "computer_parts",
        "gloss": "computer hardware or accessory names",
        "tags": ["semantic", "computer", "object"],
        "blocks": [
            ["MOUSE", "KEYBOARD", "MONITOR", "PRINTER"],
            ["ROUTER", "MODEM", "SCANNER", "SPEAKER"],
            ["TRACKPAD", "WEBCAM", "PROCESSOR", "BATTERY"],
        ],
    },
    {
        "id": "office_supplies",
        "gloss": "office supply names",
        "tags": ["semantic", "office", "object"],
        "blocks": [
            ["ENVELOPE", "STAPLER", "CLIPBOARD", "PAPERCLIP"],
            ["LABEL", "TONER", "PLANNER", "CALENDAR"],
            ["NOTEPAD", "HIGHLIGHTER", "TAPE", "STAPLE"],
        ],
    },
    {
        "id": "medical_items",
        "gloss": "medical item names",
        "tags": ["semantic", "health", "object"],
        "blocks": [
            ["BANDAGE", "SYRINGE", "SCALPEL", "STETHOSCOPE"],
            ["THERMOMETER", "CRUTCH", "SPLINT", "GAUZE"],
            ["MASK", "GLOVE", "PILL", "TABLET"],
        ],
    },
    {
        "id": "emotion_words",
        "gloss": "emotion or feeling names",
        "tags": ["semantic", "abstract", "social"],
        "blocks": [
            ["JOY", "ANGER", "FEAR", "SADNESS"],
            ["PRIDE", "SHAME", "HOPE", "ENVY"],
            ["GUILT", "GRIEF", "LOVE", "SURPRISE"],
        ],
    },
    {
        "id": "natural_landforms",
        "gloss": "natural landform names",
        "tags": ["semantic", "geography", "nature"],
        "blocks": [
            ["MOUNTAIN", "VALLEY", "CANYON", "MESA"],
            ["ISLAND", "BEACH", "CLIFF", "DUNE"],
            ["PLATEAU", "VOLCANO", "GLACIER", "REEF"],
        ],
    },
]

INTERSECTION_BUNDLE_SPECS = [
    {
        "id": "animal_sports_team_names",
        "components": ["animal names", "North American sports team names"],
        "component_knowledge_types": ["semantic", "encyclopedic"],
        "tags": ["semantic", "animals", "sports"],
        "blocks": [
            ["BEARS", "BULLS", "CUBS", "EAGLES"],
            ["FALCONS", "HAWKS", "JAGUARS", "LIONS"],
            ["PANTHERS", "RAMS", "RAVENS", "WOLVES"],
        ],
    },
    {
        "id": "animal_zodiac_names",
        "components": ["animal names", "Chinese zodiac signs"],
        "component_knowledge_types": ["semantic", "encyclopedic"],
        "tags": ["semantic", "animals", "calendar"],
        "blocks": [
            ["RAT", "OX", "TIGER", "RABBIT"],
            ["DRAGON", "SNAKE", "HORSE", "GOAT"],
            ["MONKEY", "ROOSTER", "DOG", "PIG"],
        ],
    },
    {
        "id": "animal_action_verbs",
        "components": ["animal names", "action verbs"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["semantic", "animals", "actions"],
        "blocks": [
            ["BUFFALO", "PARROT", "BEETLE", "WEASEL"],
            ["HOUND", "QUAIL", "APE", "CHICKEN"],
            ["CROW", "FERRET", "GOOSE", "BADGER"],
        ],
    },
    {
        "id": "bird_surnames",
        "components": ["bird names", "common surnames"],
        "component_knowledge_types": ["semantic", "encyclopedic"],
        "tags": ["semantic", "birds", "names"],
        "blocks": [
            ["FINCH", "ROBIN", "DOVE", "CRANE"],
            ["SWIFT", "SPARROW", "HERON", "LARK"],
            ["WREN", "DRAKE", "MARTIN", "KITE"],
        ],
    },
    {
        "id": "body_part_action_verbs",
        "components": ["body part names", "action verbs"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["semantic", "body", "actions"],
        "blocks": [
            ["HEAD", "HAND", "ARM", "BACK"],
            ["FACE", "EYE", "NOSE", "MOUTH"],
            ["SHOULDER", "ELBOW", "KNEE", "FOOT"],
        ],
    },
    {
        "id": "deities_celestial_bodies",
        "components": ["mythological deity names", "celestial body names"],
        "component_knowledge_types": ["encyclopedic", "encyclopedic"],
        "tags": ["mythology", "space", "encyclopedic"],
        "blocks": [
            ["MERCURY", "VENUS", "MARS", "JUPITER"],
            ["SATURN", "URANUS", "NEPTUNE", "PLUTO"],
            ["CERES", "ERIS", "HAUMEA", "MAKEMAKE"],
        ],
    },
    {
        "id": "chemical_elements_mythological_names",
        "components": ["chemical element names", "names derived from mythological figures"],
        "component_knowledge_types": ["encyclopedic", "encyclopedic"],
        "tags": ["science", "mythology", "encyclopedic"],
        "blocks": [
            ["HELIUM", "SELENIUM", "TELLURIUM", "CERIUM"],
            ["PALLADIUM", "THORIUM", "TITANIUM", "PROMETHIUM"],
            ["NIOBIUM", "TANTALUM", "VANADIUM", "IRIDIUM"],
        ],
    },
    {
        "id": "chemical_elements_person_names",
        "components": ["chemical element names", "names derived from scientists or inventors"],
        "component_knowledge_types": ["encyclopedic", "encyclopedic"],
        "tags": ["science", "people", "encyclopedic"],
        "blocks": [
            ["EINSTEINIUM", "FERMIUM", "MENDELEVIUM", "NOBELIUM"],
            ["CURIUM", "RUTHERFORDIUM", "SEABORGIUM", "BOHRIUM"],
            ["MEITNERIUM", "ROENTGENIUM", "COPERNICIUM", "LAWRENCIUM"],
        ],
    },
    {
        "id": "chemical_elements_place_names",
        "components": ["chemical element names", "names derived from places"],
        "component_knowledge_types": ["encyclopedic", "encyclopedic"],
        "tags": ["science", "geography", "encyclopedic"],
        "blocks": [
            ["CALIFORNIUM", "BERKELIUM", "DUBNIUM", "DARMSTADTIUM"],
            ["HAFNIUM", "HOLMIUM", "LUTETIUM", "MOSCOVIUM"],
            ["POLONIUM", "GERMANIUM", "AMERICIUM", "EUROPIUM"],
        ],
    },
    {
        "id": "city_given_names",
        "components": ["city names", "common given names"],
        "component_knowledge_types": ["encyclopedic", "semantic"],
        "tags": ["geography", "names", "semantic"],
        "blocks": [
            ["AUSTIN", "CHARLOTTE", "EUGENE", "HELENA"],
            ["JACKSON", "VICTORIA", "FLORENCE", "ALEXANDRIA"],
            ["MADISON", "SYDNEY", "ADELAIDE", "SAVANNAH"],
        ],
    },
    {
        "id": "clothing_item_action_verbs",
        "components": ["clothing item names", "action verbs"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["semantic", "clothing", "actions"],
        "blocks": [
            ["DRESS", "SUIT", "SOCK", "SHOE"],
            ["CAP", "BELT", "BOOT", "TIE"],
            ["BUTTON", "ZIP", "LACE", "HEM"],
        ],
    },
    {
        "id": "dance_styles_music_genres",
        "components": ["dance styles", "music genres"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["music", "dance", "performance"],
        "blocks": [
            ["SALSA", "TANGO", "SWING", "DISCO"],
            ["WALTZ", "FOXTROT", "JIVE", "SAMBA"],
            ["RUMBA", "MAMBO", "MERENGUE", "BACHATA"],
        ],
    },
    {
        "id": "food_drink_color_names",
        "components": ["food or drink names", "color names"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["semantic", "food", "color"],
        "blocks": [
            ["CHOCOLATE", "CREAM", "COFFEE", "HONEY"],
            ["MUSTARD", "SAFFRON", "SALMON", "CINNAMON"],
            ["GINGER", "VANILLA", "CARAMEL", "OYSTER"],
        ],
    },
    {
        "id": "fruit_color_names",
        "components": ["fruit names", "color names"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["semantic", "food", "color"],
        "blocks": [
            ["ORANGE", "APRICOT", "PEACH", "PLUM"],
            ["CHERRY", "LEMON", "LIME", "OLIVE"],
            ["GRAPE", "MELON", "PAPAYA", "TANGERINE"],
        ],
    },
    {
        "id": "gemstone_given_names",
        "components": ["gemstone or mineral names", "common given names"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["semantic", "gems", "names"],
        "blocks": [
            ["RUBY", "AMBER", "JADE", "PEARL"],
            ["OPAL", "BERYL", "GARNET", "CRYSTAL"],
            ["CORAL", "ONYX", "SAPPHIRE", "TOPAZ"],
        ],
    },
    {
        "id": "keyboard_keys_editing_commands",
        "components": ["keyboard key names", "common editing or navigation commands"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["object", "computer", "action"],
        "blocks": [
            ["ENTER", "RETURN", "DELETE", "TAB"],
            ["SHIFT", "CONTROL", "OPTION", "COMMAND"],
            ["ESCAPE", "HOME", "END", "INSERT"],
        ],
    },
    {
        "id": "material_color_names",
        "components": ["material names", "color names"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["semantic", "materials", "color"],
        "blocks": [
            ["GOLD", "SILVER", "COPPER", "BRONZE"],
            ["BRASS", "STEEL", "IRON", "PEWTER"],
            ["PLATINUM", "LEAD", "NICKEL", "SLATE"],
        ],
    },
    {
        "id": "moons_mythological_figures",
        "components": ["moon names", "mythological figure names"],
        "component_knowledge_types": ["encyclopedic", "encyclopedic"],
        "tags": ["space", "mythology", "encyclopedic"],
        "blocks": [
            ["IO", "EUROPA", "GANYMEDE", "CALLISTO"],
            ["TITAN", "ENCELADUS", "MIMAS", "RHEA"],
            ["IAPETUS", "TRITON", "NEREID", "CHARON"],
        ],
    },
    {
        "id": "mythological_figures_constellations",
        "components": ["mythological figure names", "constellation names"],
        "component_knowledge_types": ["encyclopedic", "encyclopedic"],
        "tags": ["mythology", "space", "encyclopedic"],
        "blocks": [
            ["ANDROMEDA", "ORION", "PERSEUS", "CASSIOPEIA"],
            ["CEPHEUS", "HERCULES", "PEGASUS", "CENTAURUS"],
            ["DRACO", "HYDRA", "AURIGA", "BOOTES"],
        ],
    },
    {
        "id": "opera_titles_character_names",
        "components": ["opera titles", "character names"],
        "component_knowledge_types": ["encyclopedic", "semantic"],
        "tags": ["music", "theater", "names"],
        "blocks": [
            ["CARMEN", "AIDA", "NORMA", "TOSCA"],
            ["OTELLO", "FALSTAFF", "TURANDOT", "RIGOLETTO"],
            ["LOHENGRIN", "PARSIFAL", "MANON", "NABUCCO"],
        ],
    },
    {
        "id": "occupation_surnames",
        "components": ["occupation names", "common surnames"],
        "component_knowledge_types": ["semantic", "encyclopedic"],
        "tags": ["semantic", "roles", "names"],
        "blocks": [
            ["BAKER", "BARBER", "BUTCHER", "CARPENTER"],
            ["FARMER", "MASON", "TAYLOR", "SMITH"],
            ["COOPER", "CARTER", "FISHER", "FLETCHER"],
        ],
    },
    {
        "id": "shakespeare_characters_given_names",
        "components": ["Shakespeare character names", "common given names"],
        "component_knowledge_types": ["encyclopedic", "semantic"],
        "tags": ["literature", "names", "encyclopedic"],
        "blocks": [
            ["HAMLET", "JULIET", "ROMEO", "OTHELLO"],
            ["OPHELIA", "CORDELIA", "PORTIA", "IAGO"],
            ["VIOLA", "ROSALIND", "TITANIA", "ARIEL"],
        ],
    },
    {
        "id": "si_units_scientist_names",
        "components": ["measurement unit names", "scientist names"],
        "component_knowledge_types": ["encyclopedic", "encyclopedic"],
        "tags": ["science", "people", "measurement"],
        "blocks": [
            ["NEWTON", "PASCAL", "JOULE", "TESLA"],
            ["WATT", "AMPERE", "OHM", "WEBER"],
            ["HENRY", "BECQUEREL", "SIEVERT", "KELVIN"],
        ],
    },
    {
        "id": "virtue_given_names",
        "components": ["virtue words", "common given names"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["semantic", "abstract", "names"],
        "blocks": [
            ["GRACE", "HOPE", "FAITH", "CHARITY"],
            ["JOY", "PATIENCE", "PRUDENCE", "MERCY"],
            ["HONOR", "JUSTICE", "CONSTANCE", "FELICITY"],
        ],
    },
    {
        "id": "weather_sports_team_names",
        "components": ["weather, climate, or sky terms", "North American sports team names"],
        "component_knowledge_types": ["semantic", "encyclopedic"],
        "tags": ["weather", "sports", "sky"],
        "blocks": [
            ["THUNDER", "LIGHTNING", "STORM", "HURRICANES"],
            ["CYCLONES", "TORNADOES", "BLIZZARD", "HEAT"],
            ["SUNS", "SKY", "GALAXY", "AVALANCHE"],
        ],
    },
    {
        "id": "school_subjects_sciences",
        "components": ["school subject names", "science or social science fields"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["education", "science", "semantic"],
        "blocks": [
            ["BIOLOGY", "CHEMISTRY", "PHYSICS", "GEOLOGY"],
            ["ASTRONOMY", "ECOLOGY", "BOTANY", "ZOOLOGY"],
            ["GENETICS", "STATISTICS", "ECONOMICS", "PSYCHOLOGY"],
        ],
    },
    {
        "id": "sports_terms_action_verbs",
        "components": ["sports terms", "action verbs"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["sports", "actions", "semantic"],
        "blocks": [
            ["BAT", "BOWL", "PITCH", "CATCH"],
            ["THROW", "KICK", "PUNT", "PASS"],
            ["SCORE", "TACKLE", "SERVE", "VOLLEY"],
        ],
    },
    {
        "id": "flower_given_names",
        "components": ["flower names", "common given names"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["plant", "names", "semantic"],
        "blocks": [
            ["ROSE", "LILY", "DAISY", "IRIS"],
            ["JASMINE", "HEATHER", "HOLLY", "LAUREL"],
            ["POPPY", "DAHLIA", "CAMELLIA", "ZINNIA"],
        ],
    },
    {
        "id": "animal_constellation_names",
        "components": ["animal names", "constellation names"],
        "component_knowledge_types": ["semantic", "encyclopedic"],
        "tags": ["animals", "space", "encyclopedic"],
        "blocks": [
            ["CHAMELEON", "DOLPHIN", "GIRAFFE", "LIZARD"],
            ["LYNX", "PEACOCK", "PHOENIX", "TOUCAN"],
            ["WHALE", "CRAB", "SWAN", "FOX"],
        ],
    },
    {
        "id": "military_group_sports_team_names",
        "components": ["military or historical group names", "sports team names"],
        "component_knowledge_types": ["semantic", "encyclopedic"],
        "tags": ["history", "sports", "groups"],
        "blocks": [
            ["COMMANDERS", "RANGERS", "RAIDERS", "WARRIORS"],
            ["KNIGHTS", "CAVALIERS", "SENTINELS", "TROJANS"],
            ["SPARTANS", "VIKINGS", "PATRIOTS", "MINUTEMEN"],
        ],
    },
    {
        "id": "plant_given_names",
        "components": ["plant or flower names", "common given names"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["plant", "names", "semantic"],
        "blocks": [
            ["BASIL", "DAHLIA", "HAZEL", "HOLLY"],
            ["IRIS", "IVY", "JASMINE", "LAUREL"],
            ["LILY", "ROSE", "SAGE", "VIOLET"],
        ],
    },
    {
        "id": "constellation_zodiac_names",
        "components": ["constellation names", "zodiac signs"],
        "component_knowledge_types": ["encyclopedic", "encyclopedic"],
        "tags": ["space", "calendar", "symbols"],
        "blocks": [
            ["ARIES", "TAURUS", "GEMINI", "CANCER"],
            ["LEO", "VIRGO", "LIBRA", "SCORPIO"],
            ["SAGITTARIUS", "CAPRICORN", "AQUARIUS", "PISCES"],
        ],
    },
    {
        "id": "classical_deities_celestial_bodies",
        "components": ["classical deity names", "celestial body names"],
        "component_knowledge_types": ["encyclopedic", "encyclopedic"],
        "tags": ["mythology", "space", "encyclopedic"],
        "blocks": [
            ["MERCURY", "VENUS", "MARS", "JUPITER"],
            ["SATURN", "URANUS", "NEPTUNE", "PLUTO"],
            ["CERES", "JUNO", "VESTA", "PALLAS"],
        ],
    },
    {
        "id": "greek_letters_science_symbols",
        "components": ["Greek letter names", "symbols used in mathematics or science"],
        "component_knowledge_types": ["semantic", "encyclopedic"],
        "tags": ["language", "science", "symbols"],
        "blocks": [
            ["ALPHA", "BETA", "GAMMA", "DELTA"],
            ["EPSILON", "THETA", "LAMBDA", "SIGMA"],
            ["OMEGA", "PI", "PHI", "TAU"],
        ],
    },
    {
        "id": "kitchen_objects_action_verbs",
        "components": ["kitchen object names", "action verbs"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["kitchen", "object", "actions"],
        "blocks": [
            ["BOWL", "CUP", "DISH", "PLATE"],
            ["FORK", "KNIFE", "SPOON", "WHISK"],
            ["LADLE", "SKEWER", "GRILL", "POT"],
        ],
    },
    {
        "id": "tool_names_action_verbs",
        "components": ["tool names", "action verbs"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["tool", "object", "actions"],
        "blocks": [
            ["HAMMER", "SAW", "DRILL", "FILE"],
            ["RAKE", "HOE", "PLANE", "SAND"],
            ["WRENCH", "SCREW", "NAIL", "CLAMP"],
        ],
    },
    {
        "id": "weather_terms_action_verbs",
        "components": ["weather or sky terms", "action verbs"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["weather", "sky", "actions"],
        "blocks": [
            ["RAIN", "SNOW", "HAIL", "SLEET"],
            ["STORM", "THUNDER", "DRIZZLE", "SHOWER"],
            ["FLOOD", "CLOUD", "MIST", "WIND"],
        ],
    },
    {
        "id": "vehicle_terms_action_verbs",
        "components": ["vehicle or transport terms", "action verbs"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["transport", "object", "actions"],
        "blocks": [
            ["TRAIN", "BUS", "BIKE", "SHIP"],
            ["TRUCK", "FERRY", "PLANE", "BOAT"],
            ["CYCLE", "SKATE", "SLED", "TAXI"],
        ],
    },
    {
        "id": "dance_styles_action_verbs",
        "components": ["dance style names", "action verbs"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["dance", "music", "actions"],
        "blocks": [
            ["WALTZ", "TANGO", "SWING", "TAP"],
            ["JIVE", "SALSA", "SAMBA", "RUMBA"],
            ["FOXTROT", "HUSTLE", "MAMBO", "CHA-CHA"],
        ],
    },
    {
        "id": "biblical_books_given_names",
        "components": ["Bible book names", "common given names"],
        "component_knowledge_types": ["encyclopedic", "semantic"],
        "tags": ["history", "literature", "names"],
        "blocks": [
            ["RUTH", "ESTHER", "MARK", "LUKE"],
            ["JOHN", "JAMES", "JUDE", "JOEL"],
            ["DANIEL", "SAMUEL", "JEREMIAH", "ISAIAH"],
        ],
    },
    {
        "id": "us_presidents_surnames",
        "components": ["U.S. president surnames", "common surnames"],
        "component_knowledge_types": ["encyclopedic", "semantic"],
        "tags": ["history", "people", "names"],
        "blocks": [
            ["ADAMS", "JACKSON", "JOHNSON", "LINCOLN"],
            ["GRANT", "HAYES", "GARFIELD", "CLEVELAND"],
            ["HARRISON", "MCKINLEY", "WILSON", "TRUMAN"],
        ],
    },
    {
        "id": "mythological_figures_given_names",
        "components": ["mythological figure names", "common given names"],
        "component_knowledge_types": ["encyclopedic", "semantic"],
        "tags": ["mythology", "names", "people"],
        "blocks": [
            ["DIANA", "HECTOR", "HELEN", "JASON"],
            ["CASSANDRA", "PENELOPE", "PHOEBE", "DAPHNE"],
            ["ARIADNE", "IRIS", "HERMES", "MORGAN"],
        ],
    },
    {
        "id": "programming_languages_common_words",
        "components": ["programming language names", "ordinary English words or given names"],
        "component_knowledge_types": ["encyclopedic", "semantic"],
        "tags": ["technology", "language", "semantic"],
        "blocks": [
            ["PYTHON", "RUBY", "SWIFT", "RUST"],
            ["GO", "SCRATCH", "PROCESSING", "BASIC"],
            ["ADA", "JULIA", "DYLAN", "CRYSTAL"],
        ],
    },
    {
        "id": "household_objects_action_verbs",
        "components": ["household object names", "action verbs"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["object", "household", "actions"],
        "blocks": [
            ["BRUSH", "COMB", "MOP", "VACUUM"],
            ["DUST", "IRON", "LOCK", "KEY"],
            ["TAPE", "GLUE", "POLISH", "SOAP"],
        ],
    },
    {
        "id": "office_terms_action_verbs",
        "components": ["office or communication terms", "action verbs"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["object", "office", "actions"],
        "blocks": [
            ["STAMP", "STAPLE", "CLIP", "FILE"],
            ["LABEL", "PRINT", "COPY", "FAX"],
            ["MAIL", "PHONE", "TEXT", "POST"],
        ],
    },
    {
        "id": "food_terms_action_verbs",
        "components": ["food or seasoning names", "action verbs"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["food", "actions", "kitchen"],
        "blocks": [
            ["TOAST", "GRILL", "ROAST", "STEW"],
            ["PICKLE", "JAM", "BUTTER", "OIL"],
            ["SALT", "PEPPER", "SPICE", "MARINADE"],
        ],
    },
    {
        "id": "sound_words_action_verbs",
        "components": ["sound words", "action verbs"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["sound", "actions", "events"],
        "blocks": [
            ["BARK", "BUZZ", "CHIRP", "CLICK"],
            ["CLANG", "CRASH", "HISS", "HUM"],
            ["RING", "ROAR", "RUMBLE", "WHISTLE"],
        ],
    },
    {
        "id": "legal_terms_action_verbs",
        "components": ["legal terms", "action verbs"],
        "component_knowledge_types": ["semantic", "semantic"],
        "tags": ["legal", "abstract", "actions"],
        "blocks": [
            ["APPEAL", "CLAIM", "CHARGE", "CONVICT"],
            ["SENTENCE", "ACQUIT", "SUE", "FILE"],
            ["BRIEF", "MOTION", "PARDON", "REMAND"],
        ],
    },
]

THREE_WAY_COMPONENT_BY_INTERSECTION_ID = {
    "animal_sports_team_names": ("college or professional mascot names", "encyclopedic"),
    "animal_zodiac_names": ("traditional zodiac cycle signs", "encyclopedic"),
    "animal_action_verbs": ("common words with both noun and verb uses", "semantic"),
    "bird_surnames": ("names commonly used for people", "semantic"),
    "body_part_action_verbs": ("common words with both noun and verb uses", "semantic"),
    "deities_celestial_bodies": ("names from ancient mythology", "encyclopedic"),
    "chemical_elements_mythological_names": ("scientific names with classical mythological origins", "encyclopedic"),
    "chemical_elements_person_names": ("scientific eponyms named for people", "encyclopedic"),
    "chemical_elements_place_names": ("scientific eponyms named for places", "encyclopedic"),
    "city_given_names": ("place names commonly used as personal names", "semantic"),
    "clothing_item_action_verbs": ("common words with both noun and verb uses", "semantic"),
    "dance_styles_music_genres": ("performing-arts style names", "semantic"),
    "food_drink_color_names": ("common color words named after foods or drinks", "semantic"),
    "fruit_color_names": ("common color words named after fruits", "semantic"),
    "gemstone_given_names": ("color words or personal names derived from gems", "semantic"),
    "keyboard_keys_editing_commands": ("computer interface terms", "semantic"),
    "material_color_names": ("common color words named after materials", "semantic"),
    "moons_mythological_figures": ("astronomical names from mythology", "encyclopedic"),
    "mythological_figures_constellations": ("astronomical names from mythology", "encyclopedic"),
    "opera_titles_character_names": ("proper names used as performing-arts titles", "encyclopedic"),
    "occupation_surnames": ("common words used for both roles and group names", "semantic"),
    "shakespeare_characters_given_names": ("literary names commonly used for people", "encyclopedic"),
    "si_units_scientist_names": ("science eponyms used as measurement terms", "encyclopedic"),
    "virtue_given_names": ("abstract virtue words used as personal names", "semantic"),
    "weather_sports_team_names": ("college or professional mascot names", "encyclopedic"),
    "school_subjects_sciences": ("academic fields taught as school subjects", "semantic"),
    "sports_terms_action_verbs": ("common words used both as sports terms and actions", "semantic"),
    "flower_given_names": ("plant names commonly used as personal names", "semantic"),
    "animal_constellation_names": ("astronomical names that are also animal names", "encyclopedic"),
    "military_group_sports_team_names": ("college or professional mascot names", "encyclopedic"),
    "plant_given_names": ("plant names commonly used as personal names", "semantic"),
    "constellation_zodiac_names": ("traditional zodiac signs", "encyclopedic"),
    "classical_deities_celestial_bodies": ("names from classical mythology", "encyclopedic"),
    "greek_letters_science_symbols": ("letter names used in mathematical or scientific notation", "encyclopedic"),
    "kitchen_objects_action_verbs": ("common words with both noun and verb uses", "semantic"),
    "tool_names_action_verbs": ("common words with both noun and verb uses", "semantic"),
    "weather_terms_action_verbs": ("common words with both noun and verb uses", "semantic"),
    "vehicle_terms_action_verbs": ("transport words used as verbs", "semantic"),
    "dance_styles_action_verbs": ("performing-arts action words", "semantic"),
    "biblical_books_given_names": ("biblical names commonly used for people", "encyclopedic"),
    "us_presidents_surnames": ("U.S. political or historical names", "encyclopedic"),
    "mythological_figures_given_names": ("mythological names commonly used for people", "encyclopedic"),
    "programming_languages_common_words": ("technology terms that are also ordinary words or names", "encyclopedic"),
    "household_objects_action_verbs": ("common words with both noun and verb uses", "semantic"),
    "office_terms_action_verbs": ("common words with both noun and verb uses", "semantic"),
    "food_terms_action_verbs": ("common words with both noun and verb uses", "semantic"),
    "sound_words_action_verbs": ("common words with both noun and verb uses", "semantic"),
    "legal_terms_action_verbs": ("legal vocabulary with both noun and verb uses", "semantic"),
}


def normalize_difficulty(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in DIFFICULTY_ALIASES:
        raise ValueError(f"Unsupported lexical_category_inference difficulty: {value}")
    return DIFFICULTY_ALIASES[normalized]


def normalize_logic_condition(value: str | None, *, allow_all: bool = False) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in LOGIC_CONDITION_ALIASES:
        raise ValueError(f"Unsupported lexical_category_inference logic_condition: {value}")
    group = LOGIC_CONDITION_ALIASES[normalized]
    if group == "all" and not allow_all:
        raise ValueError("logic_condition='all' is only valid when selecting multiple design cells.")
    return group


def design_cell_key(logic_condition: str, difficulty: str) -> str:
    return f"{logic_condition}/{difficulty}"


def resolve_design_cell(difficulty: str, logic_condition: str | None) -> Dict[str, Any]:
    level = normalize_difficulty(difficulty)
    group = normalize_logic_condition(logic_condition)
    if level == "d1":
        if group in {None, "shared", "either", "both"}:
            return dict(DESIGN_CELL_SPECS[("shared", "d1")])
        raise ValueError(f"d1 is shared; unsupported logic_condition={logic_condition!r}.")
    if group not in {"either", "both"}:
        raise ValueError(
            "lexical_category_inference d2/d3 require --logic-condition either or both. "
            "The d1 baseline is the only shared cell."
        )
    return dict(DESIGN_CELL_SPECS[(group, level)])


def iter_requested_design_cells(difficulty: str, logic_condition: str | None) -> List[Dict[str, Any]]:
    group = normalize_logic_condition(logic_condition, allow_all=True)
    difficulty_value = str(difficulty).strip().lower()
    difficulties = ["d1", "d2", "d3"] if difficulty_value == "all" else [normalize_difficulty(difficulty)]

    cells: List[Dict[str, Any]] = []
    for level in difficulties:
        if group in {None, "all"}:
            if level == "d1":
                cells.append(dict(DESIGN_CELL_SPECS[("shared", "d1")]))
            else:
                cells.append(dict(DESIGN_CELL_SPECS[("either", level)]))
                cells.append(dict(DESIGN_CELL_SPECS[("both", level)]))
            continue

        if level == "d1":
            if group in {"shared", "either", "both"}:
                cells.append(dict(DESIGN_CELL_SPECS[("shared", "d1")]))
            continue

        if group == "shared":
            continue
        cells.append(dict(DESIGN_CELL_SPECS[(group, level)]))

    deduped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for cell in cells:
        deduped[(cell["logic_condition"], cell["difficulty"])] = cell
    return [deduped[key] for key in REAL_DESIGN_CELLS if key in deduped]


def _safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    index = (len(sorted_values) - 1) * probability
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return sorted_values[lower]

    fraction = index - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _value_within_bounds(value: float, bounds: Dict[str, float]) -> bool:
    if "min" in bounds and value < float(bounds["min"]):
        return False
    if "max" in bounds and value > float(bounds["max"]):
        return False
    return True


def _value_matches_exact_target(value: float, target: float, tolerance: float = 1e-9) -> bool:
    return abs(float(value) - float(target)) <= tolerance


def _tokenize_surface(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).upper())


def _flatten_words(bundle: Dict[str, Any]) -> List[str]:
    return [_normalize_word_surface(word) for block in bundle["blocks"] for word in block]


def _tag_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = {str(item) for item in left}
    right_set = {str(item) for item in right}
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _gloss_token_jaccard(left: str, right: str) -> float:
    left_tokens = set(_tokenize_surface(left))
    right_tokens = set(_tokenize_surface(right))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def load_bundle_bank(bundle_bank_path: str | None = None) -> List[Dict[str, Any]]:
    if bundle_bank_path:
        path = Path(bundle_bank_path)
    elif DEFAULT_BUNDLE_BANK_V1.exists():
        path = DEFAULT_BUNDLE_BANK_V1
    else:
        path = DEFAULT_BUNDLE_BANK_SMOKE

    with path.open("r") as handle:
        bank = json.load(handle)
    if not isinstance(bank, list):
        raise ValueError(f"Expected bundle bank to be a list, found {type(bank).__name__}")
    return [
        _with_category_family_metadata(_normalize_bundle_word_fields(bundle))
        for bundle in bank
        if isinstance(bundle, dict)
    ]


def load_source_seed_categories(source_seed_path: str | Path | None = None) -> List[Dict[str, Any]]:
    path = Path(source_seed_path) if source_seed_path else DEFAULT_SOURCE_SEED_PATH
    with path.open("r") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        categories = payload.get("categories", [])
    elif isinstance(payload, list):
        categories = payload
    else:
        raise ValueError(f"Unsupported source seed payload in {path}")
    if not isinstance(categories, list):
        raise ValueError(f"Expected list of categories in {path}")
    return [
        _normalize_source_word_fields(row)
        for row in categories
        if isinstance(row, dict)
    ]


def _normalize_title_for_group(title: str) -> str:
    tokens = re.findall(r"[A-Z0-9]+", str(title).upper())
    return " ".join(tokens)


def _title_content_tokens(title: str, limit: int = 2) -> List[str]:
    tokens = [
        token
        for token in re.findall(r"[A-Z0-9]+", str(title).upper())
        if token not in TITLE_STOPWORDS and len(token) > 2
    ]
    deduped = list(dict.fromkeys(tokens))
    return deduped[:limit]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "bundle"


def _seed_group_key(row: Dict[str, Any]) -> str:
    name = str(row.get("seed_category_name") or "").upper().strip()
    if "HOMOPHONE" in name:
        return "meta::homophones"
    if "ANAGRAM" in name:
        return "meta::anagrams"
    if "PALINDROME" in name:
        return "meta::palindromes"
    if "STARTING WITH" in name or "STARTS WITH" in name:
        return "meta::starts_with_hidden_class"
    if "ENDING WITH" in name or "ENDS WITH" in name:
        return "meta::ends_with_hidden_class"
    if "WORDS BEFORE" in name or name.startswith("_"):
        return "meta::words_before_hidden_term"
    if "WORDS AFTER" in name or re.search(r"_\s*$", name):
        return "meta::words_after_hidden_term"
    return f"exact::{_normalize_title_for_group(name)}"


def _broad_red_herring_tag(knowledge_type: str) -> str:
    return {
        "semantic": "lexical_semantics",
        "associative_relations": "association",
        "encyclopedic": "world_knowledge",
        "multiword_expression": "phrase",
        "word_meaning_plus_word_form": "surface_form",
    }.get(str(knowledge_type), "semantic")


def _bundle_red_herring_tags(
    group_key: str,
    knowledge_type: str,
    seed_names: Sequence[str],
) -> List[str]:
    tags: List[str] = [_broad_red_herring_tag(knowledge_type)]
    if group_key in META_GROUP_CONFIG:
        tags.extend(META_GROUP_CONFIG[group_key]["tags"])
    elif seed_names:
        tags.extend(_title_content_tokens(seed_names[0], limit=2))

    deduped = list(dict.fromkeys(str(tag) for tag in tags if str(tag).strip()))
    return deduped[:3]


def _category_family_evidence(bundle: Dict[str, Any]) -> List[str]:
    source = bundle.get("source") or {}
    evidence: List[str] = []
    evidence.extend(str(tag) for tag in bundle.get("red_herring_tags", []) if str(tag).strip())
    for component_tags in source.get("component_red_herring_tags") or []:
        if isinstance(component_tags, list):
            evidence.extend(str(tag) for tag in component_tags if str(tag).strip())
    evidence.extend(str(gloss) for gloss in source.get("component_rule_glosses") or [] if str(gloss).strip())
    evidence.append(str(bundle.get("rule_gloss") or ""))
    return evidence


def _category_families_for_bundle(bundle: Dict[str, Any]) -> List[str]:
    families: List[str] = []
    for item in _category_family_evidence(bundle):
        normalized = str(item).strip().lower()
        if not normalized or normalized in {"semantic", "intersection", "all_conditions", "three_way"}:
            continue
        if normalized in CATEGORY_FAMILY_TAG_MAP:
            families.append(CATEGORY_FAMILY_TAG_MAP[normalized])
        for pattern, family in CATEGORY_FAMILY_TEXT_PATTERNS:
            if pattern.search(normalized):
                families.append(family)
    families = list(dict.fromkeys(families))
    return families or ["general_semantic"]


def _with_category_family_metadata(bundle: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(bundle)
    families = _category_families_for_bundle(copied)
    copied["category_family"] = families[0]
    copied["category_families"] = families
    source = dict(copied.get("source") or {})
    source["category_family"] = copied["category_family"]
    source["category_families"] = list(families)
    copied["source"] = source
    return copied


def _canonical_seed_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "seed_category_name": str(row.get("seed_category_name") or ""),
        "source_words": _normalize_word_list(row.get("source_words") or []),
        "knowledge_type": str(row.get("knowledge_type") or ""),
        "complexity": str(row.get("complexity") or ""),
        "based_on": str(row.get("based_on") or ""),
        "original_game_index": row.get("original_game_index"),
        "original_board_id": row.get("original_board_id"),
        "original_nyt_difficulty": row.get("original_nyt_difficulty"),
        "source_repo": str(row.get("source_repo") or "lexical_category_inference_source"),
    }


def _chunked(values: Sequence[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    return [list(values[i:i + size]) for i in range(0, len(values), size) if len(values[i:i + size]) == size]


def _exact_group_chunks(rows: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    rows = list(rows)
    if len(rows) < 3:
        return []
    if len(rows) == 3:
        return [rows]
    if len(rows) == 4:
        return [
            [rows[index] for index in range(4) if index != drop_index]
            for drop_index in range(4)
        ]
    combos = list(itertools.combinations(rows, 3))
    return [list(combo) for combo in combos[:4]]


def _difficulty_for_exact_bundle(knowledge_type: str, complexity: str) -> str:
    if knowledge_type in {"multiword_expression", "word_meaning_plus_word_form"} or complexity == "complex":
        return "d3"
    return "d1"


def _meta_bundle_complexity_and_difficulty(
    group_key: str,
    source_rows: Sequence[Dict[str, Any]],
    bundle_index: int,
) -> tuple[str, str]:
    complex_count = sum(1 for row in source_rows if str(row.get("complexity")) == "complex")
    complexity = "complex" if complex_count * 2 >= len(source_rows) and complex_count > 0 else "simple"

    if group_key in {"meta::words_before_hidden_term", "meta::words_after_hidden_term"}:
        difficulty = "d3" if complexity == "complex" or bundle_index % 2 == 0 else "d2"
    elif group_key == "meta::homophones":
        difficulty = "d3" if complex_count >= 1 or bundle_index % 2 == 0 else "d2"
    elif group_key in {"meta::anagrams", "meta::palindromes"}:
        difficulty = "d3" if complex_count >= 1 else "d2"
    elif group_key in {"meta::starts_with_hidden_class", "meta::ends_with_hidden_class"}:
        difficulty = "d3"
    else:
        difficulty = "d3"

    return complexity, difficulty


def _sort_seed_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("seed_category_name") or ""),
            int(row.get("original_game_index") or -1),
            tuple(str(word) for word in row.get("source_words") or []),
        ),
    )


def _dedupe_seed_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in _sort_seed_rows(rows):
        words = tuple(str(word) for word in row.get("source_words") or [])
        if len(words) != 4 or words in seen:
            continue
        seen.add(words)
        deduped.append(row)
    return deduped


def _build_auto_bundle(
    group_key: str,
    source_rows: Sequence[Dict[str, Any]],
    bundle_index: int,
) -> Dict[str, Any]:
    source_rows = [_canonical_seed_row(row) for row in source_rows]
    seed_names = [row["seed_category_name"] for row in source_rows]
    blocks = [_normalize_word_list(row["source_words"]) for row in source_rows]

    if group_key in META_GROUP_CONFIG:
        config = META_GROUP_CONFIG[group_key]
        knowledge_type = str(config["knowledge_type"])
        complexity, difficulty = _meta_bundle_complexity_and_difficulty(
            group_key=group_key,
            source_rows=source_rows,
            bundle_index=bundle_index,
        )
        rule_gloss = _category_gloss_without_incidental_disjunction(config["rule_gloss"])
        group_slug = group_key.split("::", 1)[1]
    else:
        group_title = group_key.split("::", 1)[1]
        knowledge_type_counts = Counter(row["knowledge_type"] for row in source_rows if row["knowledge_type"])
        complexity_counts = Counter(row["complexity"] for row in source_rows if row["complexity"])
        knowledge_type = knowledge_type_counts.most_common(1)[0][0] if knowledge_type_counts else "semantic"
        complexity = complexity_counts.most_common(1)[0][0] if complexity_counts else "simple"
        difficulty = _difficulty_for_exact_bundle(knowledge_type, complexity)
        rule_gloss = _category_gloss_without_incidental_disjunction(group_title.lower())
        group_slug = _slugify(group_title)

    red_herring_tags = _bundle_red_herring_tags(group_key, knowledge_type, seed_names)
    source_summary = {
        "seed_category_name": seed_names[0] if seed_names else group_slug,
        "source_words": list(blocks[0]) if blocks else [],
        "seed_category_names": seed_names,
        "source_repo": "lexical_category_inference_source",
        "bundle_builder": AUTO_BUNDLE_BUILDER,
        "group_key": group_key,
        "source_rows": source_rows,
    }

    return _with_category_family_metadata({
        "bundle_id": f"paper_{group_slug}_{bundle_index:03d}",
        "difficulty": difficulty,
        "knowledge_type": knowledge_type,
        "complexity": complexity,
        "examples_favored": True,
        "rule_gloss": rule_gloss,
        "red_herring_tags": red_herring_tags,
        "source": source_summary,
        "blocks": blocks,
    })


def _bundle_intrinsic_difficulty_score(bundle: Dict[str, Any]) -> float:
    knowledge_score = KNOWLEDGE_TYPE_DIFFICULTY.get(str(bundle.get("knowledge_type")), 1.0)
    complexity_score = COMPLEXITY_SCORE.get(str(bundle.get("complexity")), 1.0)
    source = bundle.get("source") or {}
    group_key = str(source.get("group_key") or "")
    meta_bonus = 0.75 if group_key.startswith("meta::") else 0.0
    auto_bonus = 0.25 if source.get("bundle_builder") == AUTO_BUNDLE_BUILDER else 0.0
    composite_bonus = 0.6 * max(0, _bundle_composite_num_parts(bundle) - 1)
    return knowledge_score + complexity_score + meta_bonus + auto_bonus + composite_bonus


def _balanced_sort_key(bundle: Dict[str, Any], difficulty: str) -> tuple[Any, ...]:
    source = bundle.get("source") or {}
    group_key = str(source.get("group_key") or "")
    manual_rank = 0 if source.get("bundle_builder") != AUTO_BUNDLE_BUILDER else 1
    score = _bundle_intrinsic_difficulty_score(bundle)

    if difficulty == "d1":
        return (manual_rank, score, str(bundle.get("bundle_id") or ""))
    if difficulty == "d2":
        return (manual_rank, abs(score - 4.5), -score, str(bundle.get("bundle_id") or ""))
    return (manual_rank, -score, group_key, str(bundle.get("bundle_id") or ""))


def _select_easy_atomic_bundles(bundles: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chosen: List[Dict[str, Any]] = []
    used_ids: set[str] = set()
    for knowledge_type, knowledge_target in TARGET_BUNDLE_COUNTS_BY_TYPE[("shared", "d1")].items():
        candidates = [
            bundle
            for bundle in bundles
            if bundle.get("knowledge_type") == knowledge_type and str(bundle.get("bundle_id") or "") not in used_ids
            and len(_flatten_words(bundle)) == len(set(_flatten_words(bundle)))
        ]
        if len(candidates) < knowledge_target:
            raise ValueError(
                f"Need at least {knowledge_target} {knowledge_type} atomic bundles for d1, found {len(candidates)}."
        )
        picked = sorted(candidates, key=lambda bundle: _balanced_sort_key(bundle, "d1"))[:knowledge_target]
        for bundle in picked:
            chosen.append(_apply_design_metadata_to_bundle(bundle, "shared", "d1"))
        used_ids.update(str(bundle.get("bundle_id") or "") for bundle in picked)
    return sorted(chosen, key=lambda bundle: str(bundle.get("bundle_id") or ""))


def _apply_design_metadata_to_bundle(
    bundle: Dict[str, Any],
    logic_condition: str,
    difficulty: str,
) -> Dict[str, Any]:
    spec = DESIGN_CELL_SPECS[(logic_condition, difficulty)]
    copied = dict(bundle)
    copied.update(
        {
            "logic_condition": spec["logic_condition"],
            "category_operator": spec["category_operator"],
            "category_arity": spec["category_arity"],
            "condition_count": spec["condition_count"],
            "shared_d1": spec["shared_d1"],
            "difficulty": spec["difficulty"],
            "difficulty_name": spec["difficulty_name"],
            "design_cell": design_cell_key(spec["logic_condition"], spec["difficulty"]),
        }
    )
    source = dict(copied.get("source") or {})
    source.update(
        {
            "logic_condition": spec["logic_condition"],
            "category_operator": spec["category_operator"],
            "category_arity": spec["category_arity"],
            "condition_count": spec["condition_count"],
            "shared_d1": spec["shared_d1"],
        }
    )
    copied["source"] = source
    return _normalize_bundle_word_fields(_with_category_family_metadata(copied))


def _bundle_composite_num_parts(bundle: Dict[str, Any]) -> int:
    if bundle.get("category_arity") is not None:
        return int(bundle["category_arity"])
    source = bundle.get("source") or {}
    if source.get("category_arity") is not None:
        return int(source["category_arity"])
    composite_recipe = source.get("composite_recipe")
    if isinstance(composite_recipe, list) and composite_recipe:
        return len(composite_recipe)
    return COMPOSITE_PART_COUNTS.get(str(bundle.get("knowledge_type") or ""), 1)


def _bundle_component_types(bundle: Dict[str, Any]) -> List[str]:
    source = bundle.get("source") or {}
    component_types = source.get("component_knowledge_types")
    if isinstance(component_types, list) and component_types:
        return [str(value) for value in component_types]
    knowledge_type = str(bundle.get("knowledge_type") or "")
    return [knowledge_type] if knowledge_type else []


def _is_surface_form_bundle(bundle: Dict[str, Any]) -> bool:
    source = bundle.get("source") or {}
    group_key = str(source.get("group_key") or "")
    tags = {str(tag) for tag in bundle.get("red_herring_tags", [])}
    tags.update(str(tag) for tag in bundle.get("tags", []) if str(tag))
    return (
        group_key in SURFACE_FORM_META_GROUPS
        or str(bundle.get("knowledge_type") or "") == "word_meaning_plus_word_form"
        or "surface_form" in tags
    )


def _component_pairwise_gloss_overlap(components: Sequence[Dict[str, Any]]) -> float:
    scores: List[float] = []
    for left, right in itertools.combinations(components, 2):
        scores.append(_gloss_token_jaccard(left.get("rule_gloss", ""), right.get("rule_gloss", "")))
    return _safe_mean(scores)


def _component_pairwise_tag_overlap(components: Sequence[Dict[str, Any]]) -> float:
    scores: List[float] = []
    for left, right in itertools.combinations(components, 2):
        scores.append(
            _tag_jaccard(
                left.get("red_herring_tags", []),
                right.get("red_herring_tags", []),
            )
        )
    return _safe_mean(scores)


def _component_combo_has_unique_words(components: Sequence[Dict[str, Any]]) -> bool:
    words = list(itertools.chain.from_iterable(_flatten_words(bundle) for bundle in components))
    return len(words) == len(set(words))


def _component_clause(rule_gloss: str, count: int) -> str:
    count_word = COUNT_WORD.get(int(count), str(count))
    noun = "word" if int(count) == 1 else "words"
    verb = "fits" if int(count) == 1 else "fit"
    return f"exactly {count_word} {noun} that {verb} this description: {str(rule_gloss).strip()}"


def _clean_seed_category_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip())


def _component_display_rule_gloss(component: Dict[str, Any]) -> str:
    gloss = _category_gloss_without_incidental_disjunction(component.get("rule_gloss") or "")
    source = component.get("source") or {}
    group_key = str(source.get("group_key") or "")
    seed_names = [
        _clean_seed_category_name(name)
        for name in source.get("seed_category_names", [])
        if _clean_seed_category_name(name)
    ]
    if group_key in META_GROUP_CONFIG and seed_names:
        return f"{gloss} ({'; '.join(seed_names[:3])})"
    return gloss


def _composite_rule_gloss(component_glosses: Sequence[str], recipe_counts: Sequence[int]) -> str:
    del recipe_counts
    return "items that fit at least one of these descriptions: " + "; ".join(
        _category_gloss_without_incidental_disjunction(gloss)
        for gloss in component_glosses
        if str(gloss).strip()
    )


def _intersection_rule_gloss(component_glosses: Sequence[str]) -> str:
    arity = len([gloss for gloss in component_glosses if str(gloss).strip()])
    if arity == 3:
        prefix = "items that fit all three of these descriptions: "
    else:
        prefix = "items that fit all of these descriptions: "
    return prefix + "; ".join(
        _category_gloss_without_incidental_disjunction(gloss)
        for gloss in component_glosses
        if str(gloss).strip()
    )


def _deterministic_combination(words: Sequence[str], subset_size: int, key: str) -> List[str]:
    candidates = sorted(itertools.combinations(_normalize_word_list(words), subset_size))
    if not candidates:
        raise ValueError(f"Cannot build a size-{subset_size} subset from {words}.")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(candidates)
    return list(candidates[index])


def _deterministic_permutation(words: Sequence[str], key: str) -> List[str]:
    decorated = []
    for index, word in enumerate(words):
        normalized_word = _normalize_word_surface(word)
        digest = hashlib.sha256(f"{key}:{index}:{normalized_word}".encode("utf-8")).hexdigest()
        decorated.append((digest, normalized_word))
    decorated.sort()
    return [word for _, word in decorated]


def _composite_bundle_id(difficulty: str, component_ids: Sequence[str]) -> str:
    digest = hashlib.sha1(f"either:{difficulty}:{'|'.join(component_ids)}".encode("utf-8")).hexdigest()[:12]
    prefix = "two_parts_composite" if difficulty == "d2" else "three_parts_composite"
    return f"{prefix}_{digest}"


def _merged_component_tags(components: Sequence[Dict[str, Any]]) -> List[str]:
    counts: Counter[str] = Counter()
    order: Dict[str, int] = {}
    for component_index, component in enumerate(components):
        for tag_index, tag in enumerate(component.get("red_herring_tags", [])):
            tag = str(tag)
            counts[tag] += 1
            order.setdefault(tag, component_index * 10 + tag_index)
    ranked = sorted(counts, key=lambda tag: (-counts[tag], order[tag], tag))
    return ranked[:3]


def _composite_candidate_is_valid(
    difficulty: str,
    components: Sequence[Dict[str, Any]],
) -> bool:
    if difficulty not in COMPOSITE_RECIPE_COUNTS:
        return False
    if len(components) != len(COMPOSITE_RECIPE_COUNTS[difficulty]):
        return False
    if not components or not _component_combo_has_unique_words(components):
        return False
    if any(_is_surface_form_bundle(component) for component in components):
        return False
    if len({str(component.get("rule_gloss") or "") for component in components}) != len(components):
        return False
    if not _source_norm_component_glosses_are_clean(components):
        return False

    component_types = [str(component.get("knowledge_type") or "") for component in components]
    return all(kind in {"semantic", "encyclopedic", "associative_relations", "taxonomic"} for kind in component_types)


def _composite_candidate_score(
    difficulty: str,
    components: Sequence[Dict[str, Any]],
) -> float:
    intrinsic = _safe_mean([_bundle_intrinsic_difficulty_score(component) for component in components])
    tag_overlap = _component_pairwise_tag_overlap(components)
    gloss_overlap = _component_pairwise_gloss_overlap(components)
    hard_atomic_fraction = (
        sum(1 for component in components if component.get("knowledge_type") in ATOMIC_HARD_KNOWLEDGE_TYPES)
        / len(components)
    )
    recipe_bonus = 1.0
    return intrinsic + recipe_bonus + hard_atomic_fraction + (1.5 * tag_overlap) + gloss_overlap


def _composite_bundle_from_components(
    difficulty: str,
    components: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if difficulty not in {"d2", "d3"}:
        raise ValueError(f"OR composites are only used for d2/d3, got {difficulty}.")
    ordered_components = sorted(components, key=lambda bundle: str(bundle.get("bundle_id") or ""))

    recipe_counts = COMPOSITE_RECIPE_COUNTS[difficulty]
    component_ids = [str(bundle.get("bundle_id") or "") for bundle in ordered_components]
    bundle_id = _composite_bundle_id(difficulty, component_ids)
    knowledge_type = "two_parts_composite" if difficulty == "d2" else "three_parts_composite"
    complexity = "medium" if difficulty == "d2" else "complex"

    blocks: List[List[str]] = []
    component_indices_by_word: Dict[str, List[int]] = {}
    component_words_by_index: List[List[str]] = [[] for _ in ordered_components]
    for block_index in range(3):
        block_words: List[str] = []
        for component_index, (component, subset_size) in enumerate(zip(ordered_components, recipe_counts)):
            source_block = list(component["blocks"][block_index])
            subset_key = (
                f"{difficulty}:{bundle_id}:{component.get('bundle_id')}:{block_index}:{component_index}:{subset_size}"
            )
            selected_words = _deterministic_combination(source_block, subset_size, subset_key)
            for word in selected_words:
                normalized_word = _normalize_word_surface(word)
                component_indices_by_word.setdefault(normalized_word, []).append(component_index)
                component_words_by_index[component_index].append(normalized_word)
            block_words.extend(selected_words)
        if len(block_words) != len(set(block_words)):
            raise ValueError(
                f"Composite bundle {bundle_id} produced duplicate words within block {block_index}: {block_words}"
            )
        blocks.append(_deterministic_permutation(block_words, f"{bundle_id}:block:{block_index}"))

    component_glosses = [_component_display_rule_gloss(bundle) for bundle in ordered_components]
    component_types = [str(bundle.get("knowledge_type") or "") for bundle in ordered_components]
    red_herring_tags = list(dict.fromkeys(["two_part_union", *_merged_component_tags(ordered_components)]))[:3]
    source_summary = {
        "bundle_builder": "two_parts_composite_v2",
        "source_repo": "composite_from_atomic_bundle_pool",
        "component_bundle_ids": component_ids,
        "component_rule_glosses": component_glosses,
        "component_knowledge_types": component_types,
        "component_predicate_types": [
            (bundle.get("source") or {}).get("predicate_type")
            for bundle in ordered_components
        ],
        "component_red_herring_tags": [list(bundle.get("red_herring_tags", [])) for bundle in ordered_components],
        "component_indices_by_word": {
            word: sorted(set(indices))
            for word, indices in component_indices_by_word.items()
        },
        "component_words_by_index": [
            list(dict.fromkeys(words))
            for words in component_words_by_index
        ],
        "composite_operator": "or",
        "composite_recipe": list(recipe_counts),
        "composite_type_name": f"{len(recipe_counts)}-condition union",
    }

    return _apply_design_metadata_to_bundle({
        "bundle_id": bundle_id,
        "difficulty": difficulty,
        "knowledge_type": knowledge_type,
        "complexity": complexity,
        "examples_favored": True,
        "rule_gloss": _composite_rule_gloss(component_glosses, recipe_counts),
        "red_herring_tags": red_herring_tags,
        "source": source_summary,
        "blocks": blocks,
    }, "either", difficulty)


def _select_composite_bundles(
    difficulty: str,
    atomic_pool: Sequence[Dict[str, Any]],
    target_count: int,
) -> List[Dict[str, Any]]:
    if difficulty not in COMPOSITE_RECIPE_COUNTS:
        raise ValueError(f"No union composite recipe is defined for {difficulty}.")
    component_count = len(COMPOSITE_RECIPE_COUNTS[difficulty])
    scored_candidates: List[tuple[float, tuple[str, ...], tuple[Dict[str, Any], ...]]] = []

    for components in itertools.combinations(list(atomic_pool), component_count):
        if not _composite_candidate_is_valid(difficulty, components):
            continue
        score = _composite_candidate_score(difficulty, components)
        scored_candidates.append(
            (
                score,
                tuple(
                    str(component.get("bundle_id") or "")
                    for component in components
                ),
                tuple(components),
            )
        )

    if len(scored_candidates) < target_count:
        raise ValueError(
            f"Need at least {target_count} valid {difficulty} composite candidates, found {len(scored_candidates)}."
        )

    scored_candidates.sort(key=lambda item: (item[0], item[1]))
    if difficulty == "d2":
        low_quantile, high_quantile = 0.0, 1.0
    else:
        low_quantile, high_quantile = 0.5, 1.0
    start_index = max(0, min(len(scored_candidates) - 1, int(len(scored_candidates) * low_quantile)))
    end_index = max(start_index + target_count, min(len(scored_candidates), int(len(scored_candidates) * high_quantile)))
    selection_pool = scored_candidates[start_index:end_index]
    if len(selection_pool) < target_count:
        selection_pool = scored_candidates

    selected: List[Dict[str, Any]] = []
    selected_component_sets: set[tuple[str, ...]] = set()
    if difficulty == "d2":
        used_words: set[str] = set()
        used_components: set[str] = set()
        disjoint_first_pass = sorted(
            selection_pool,
            key=lambda item: (-item[0], item[1]),
        )
        for _, component_ids, components in disjoint_first_pass:
            component_set = set(component_ids)
            bundle = _composite_bundle_from_components(difficulty, components)
            bundle_words = set(_flatten_words(bundle))
            if component_set & used_components or bundle_words & used_words:
                continue
            selected.append(bundle)
            selected_component_sets.add(tuple(component_ids))
            used_components.update(component_set)
            used_words.update(bundle_words)
            if len(selected) >= target_count:
                break

    component_usage: Counter[str] = Counter()
    for component_ids in selected_component_sets:
        component_usage.update(component_ids)
    remaining = list(selection_pool)
    while len(selected) < target_count and remaining:
        best_index = min(
            range(len(remaining)),
            key=lambda index: (
                sum(component_usage[component_id] for component_id in remaining[index][1]),
                max(component_usage[component_id] for component_id in remaining[index][1]),
                -remaining[index][0],
                remaining[index][1],
            ),
        )
        _, component_ids, components = remaining.pop(best_index)
        if tuple(component_ids) in selected_component_sets:
            continue
        bundle = _composite_bundle_from_components(difficulty, components)
        selected.append(bundle)
        selected_component_sets.add(tuple(component_ids))
        component_usage.update(component_ids)

    if len(selected) < target_count:
        raise ValueError(
            f"Could only select {len(selected)} {difficulty} composite bundles from {len(selection_pool)} candidates."
        )

    return sorted(selected, key=lambda bundle: str(bundle.get("bundle_id") or ""))


def _semantic_bundle_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    slug = _slugify(str(spec["id"]))
    blocks = _normalize_word_blocks(spec["blocks"])
    if len(blocks) != 3 or any(len(block) != 4 for block in blocks):
        raise ValueError(f"Semantic bundle {slug} must have exactly three four-word blocks.")
    flattened = list(itertools.chain.from_iterable(blocks))
    if len(flattened) != len(set(flattened)):
        raise ValueError(f"Semantic bundle {slug} repeats words within the bundle.")
    rule_gloss = _category_gloss_without_incidental_disjunction(spec["gloss"])
    source_summary = {
        "bundle_builder": "manual_semantic_v1",
        "source_repo": "manual_curated_semantic_bundle_pool",
        "seed_category_name": slug,
        "source_words": list(blocks[0]),
    }
    return _with_category_family_metadata({
        "bundle_id": f"semantic_{slug}",
        "difficulty": "d1",
        "knowledge_type": "semantic",
        "complexity": "simple",
        "examples_favored": True,
        "rule_gloss": rule_gloss,
        "red_herring_tags": list(dict.fromkeys(str(tag) for tag in spec.get("tags", [])))[:3],
        "source": source_summary,
        "blocks": blocks,
    })


def _manual_d1_semantic_bundles() -> List[Dict[str, Any]]:
    bundles = [_semantic_bundle_from_spec(spec) for spec in D1_SEMANTIC_BUNDLE_SPECS]
    ids = [str(bundle["bundle_id"]) for bundle in bundles]
    if len(ids) != len(set(ids)):
        raise ValueError("Manual d1 semantic bundle ids must be unique.")
    return sorted(bundles, key=lambda bundle: str(bundle.get("bundle_id") or ""))


def _intersection_bundle_from_spec(spec: Dict[str, Any], difficulty: str = "d2") -> Dict[str, Any]:
    slug = _slugify(str(spec["id"]))
    component_glosses = [
        _category_gloss_without_incidental_disjunction(component)
        for component in spec["components"]
        if str(component).strip()
    ]
    if len(component_glosses) != 2:
        raise ValueError(f"Intersection bundle {slug} must have exactly two component glosses.")
    component_types = [
        str(component_type).strip()
        for component_type in (spec.get("component_knowledge_types") or ["semantic"] * len(component_glosses))
        if str(component_type).strip()
    ]
    if len(component_types) != len(component_glosses):
        raise ValueError(f"Intersection bundle {slug} must provide one component type per component.")

    blocks = _normalize_word_blocks(spec["blocks"])
    if len(blocks) != 3 or any(len(block) != 4 for block in blocks):
        raise ValueError(f"Intersection bundle {slug} must have exactly three four-word blocks.")
    flattened = list(itertools.chain.from_iterable(blocks))
    if len(flattened) != len(set(flattened)):
        raise ValueError(f"Intersection bundle {slug} repeats words within the bundle.")

    source_summary = {
        "bundle_builder": "manual_intersection_v1",
        "source_repo": "manual_curated_intersection_bundle_pool",
        "seed_category_name": slug,
        "source_words": list(blocks[0]),
        "component_rule_glosses": component_glosses,
        "component_knowledge_types": component_types,
        "component_red_herring_tags": [
            list(spec.get("tags") or []),
            ["intersection", "all_conditions"],
        ],
        "composite_operator": "and",
        "composite_recipe": [1, 1],
        "composite_type_name": "two-way intersection",
    }

    return _apply_design_metadata_to_bundle({
        "bundle_id": f"intersection_{slug}",
        "difficulty": difficulty,
        "knowledge_type": "two_way_intersection",
        "complexity": "medium",
        "examples_favored": True,
        "rule_gloss": _intersection_rule_gloss(component_glosses),
        "red_herring_tags": list(dict.fromkeys(["intersection", "all_conditions", *list(spec.get("tags") or [])]))[:3],
        "source": source_summary,
        "blocks": blocks,
    }, "both", difficulty)


def _select_intersection_bundles(target_count: int, difficulty: str = "d2") -> List[Dict[str, Any]]:
    bundles = [_intersection_bundle_from_spec(spec, difficulty=difficulty) for spec in INTERSECTION_BUNDLE_SPECS]
    if len(bundles) < target_count:
        raise ValueError(
            f"Need at least {target_count} two-way intersection bundles for both/{difficulty}, found {len(bundles)}."
        )
    ids = [str(bundle["bundle_id"]) for bundle in bundles]
    if len(ids) != len(set(ids)):
        raise ValueError("Two-way intersection bundle ids must be unique.")
    return sorted(bundles, key=lambda bundle: str(bundle.get("bundle_id") or ""))[:target_count]


def _three_way_intersection_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for spec in INTERSECTION_BUNDLE_SPECS:
        spec_id = str(spec["id"])
        if spec_id not in THREE_WAY_COMPONENT_BY_INTERSECTION_ID:
            raise ValueError(f"Missing curated third component for {spec_id}.")
        third_component, third_type = THREE_WAY_COMPONENT_BY_INTERSECTION_ID[spec_id]
        specs.append(
            {
                "id": f"{spec_id}_three_way",
                "components": [*list(spec["components"]), third_component],
                "component_knowledge_types": [
                    *list(spec.get("component_knowledge_types") or ["semantic", "semantic"]),
                    third_type,
                ],
                "tags": list(spec.get("tags") or []),
                "blocks": _normalize_word_blocks(spec["blocks"]),
            }
        )
    return specs


def _three_way_intersection_bundle_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    slug = _slugify(str(spec["id"]))
    component_glosses = [
        _category_gloss_without_incidental_disjunction(component)
        for component in spec["components"]
        if str(component).strip()
    ]
    if len(component_glosses) != 3:
        raise ValueError(f"Three-way intersection bundle {slug} must have exactly three component glosses.")
    component_types = [
        str(component_type).strip()
        for component_type in (spec.get("component_knowledge_types") or ["semantic"] * len(component_glosses))
        if str(component_type).strip()
    ]
    if len(component_types) != len(component_glosses):
        raise ValueError(f"Three-way intersection bundle {slug} must provide one component type per component.")

    blocks = _normalize_word_blocks(spec["blocks"])
    if len(blocks) != 3 or any(len(block) != 4 for block in blocks):
        raise ValueError(f"Three-way intersection bundle {slug} must have exactly three four-word blocks.")
    flattened = list(itertools.chain.from_iterable(blocks))
    if len(flattened) != len(set(flattened)):
        raise ValueError(f"Three-way intersection bundle {slug} repeats words within the bundle.")

    source_summary = {
        "bundle_builder": "manual_three_way_intersection_v1",
        "source_repo": "manual_curated_three_way_intersection_bundle_pool",
        "seed_category_name": slug,
        "source_words": list(blocks[0]),
        "component_rule_glosses": component_glosses,
        "component_knowledge_types": component_types,
        "component_red_herring_tags": [
            list(spec.get("tags") or []),
            ["intersection", "all_conditions", "three_way"],
        ],
        "composite_operator": "and",
        "composite_recipe": [1, 1, 1],
        "composite_type_name": "three-way intersection",
    }

    return _apply_design_metadata_to_bundle({
        "bundle_id": f"three_way_intersection_{slug}",
        "difficulty": "d3",
        "knowledge_type": "three_way_intersection",
        "complexity": "complex",
        "examples_favored": True,
        "rule_gloss": _intersection_rule_gloss(component_glosses),
        "red_herring_tags": list(dict.fromkeys(["intersection", "all_conditions", "three_way", *list(spec.get("tags") or [])]))[:3],
        "source": source_summary,
        "blocks": blocks,
    }, "both", "d3")


def _select_three_way_intersection_bundles(target_count: int) -> List[Dict[str, Any]]:
    bundles = [_three_way_intersection_bundle_from_spec(spec) for spec in _three_way_intersection_specs()]
    if len(bundles) < target_count:
        raise ValueError(
            f"Need at least {target_count} curated three-way intersection bundles for both/d3, found {len(bundles)}."
        )
    ids = [str(bundle["bundle_id"]) for bundle in bundles]
    if len(ids) != len(set(ids)):
        raise ValueError("Three-way intersection bundle ids must be unique.")
    return sorted(bundles, key=lambda bundle: str(bundle.get("bundle_id") or ""))[:target_count]


SOURCE_NORM_MIN_MEMBERS = 12
SOURCE_NORM_TARGET_VISIBLE_MEMBERS = 24
SOURCE_NORM_MAX_ATOMIC_PREDICATES = SOURCE_NORM_TARGET_BUNDLES_PER_CELL
SOURCE_NORM_PROPERTY_SPECS = {
    "property_manmade_mean": [
        ("high", "manmade things", 5.4, True),
        ("low", "non-manmade things", 2.6, False),
    ],
    "property_precious_mean": [
        ("high", "precious things", 4.8, True),
        ("low", "non-precious things", 2.4, False),
    ],
    "property_lives_mean": [
        ("high", "living things", 5.4, True),
        ("low", "nonliving things", 2.6, False),
    ],
    "property_heavy_mean": [
        ("high", "heavy things", 5.2, True),
        ("low", "lightweight things", 2.8, False),
    ],
    "property_natural_mean": [
        ("high", "natural things", 5.4, True),
        ("low", "non-natural things", 2.6, False),
    ],
    "property_moves_mean": [
        ("high", "movable things", 5.2, True),
        ("low", "stationary things", 2.8, False),
    ],
    "property_grasp_mean": [
        ("high", "graspable things", 5.2, True),
        ("low", "hard-to-grasp things", 2.8, False),
    ],
    "property_hold_mean": [
        ("high", "holdable things", 5.2, True),
        ("low", "hard-to-hold things", 2.8, False),
    ],
    "property_be-moved_mean": [
        ("high", "easily moved things", 5.2, True),
        ("low", "hard-to-move things", 2.8, False),
    ],
    "property_pleasant_mean": [
        ("high", "pleasant things", 5.2, True),
        ("low", "unpleasant things", 2.8, False),
    ],
}

SOURCE_NORM_REFERENCES = {
    "thingsplus": {
        "source_name": "THINGSplus",
        "source_url": "https://osf.io/jum2f/",
        "paper_url": "https://doi.org/10.3758/s13428-023-02110-8",
        "citation": "Stoinski, Perkuhn, and Hebart (2024)",
    },
    "mcrae": {
        "source_name": "McRae semantic feature production norms",
        "source_url": "https://static-content.springer.com/esm/art%3A10.3758%2FBF03192726/MediaObjects/McRae-BRM-2005.zip",
        "paper_url": "https://doi.org/10.3758/BF03192726",
        "citation": "McRae, Cree, Seidenberg, and McNorgan (2005)",
    },
    "cslb": {
        "source_name": "CSLB concept property norms",
        "source_url": "https://cslb.psychol.cam.ac.uk/propnorms",
        "paper_url": "https://doi.org/10.3758/s13428-013-0420-4",
        "citation": "Devereux, Tyler, Geertzen, and Randall (2014)",
    },
}

SOURCE_NORM_EXPERIMENTAL_PREDICATE_TYPES = {"category", "produced_feature"}

SOURCE_NORM_INTERSECTION_SAFE_RATED_PROPERTY_GLOSSES = {
    "heavy things",
    "lightweight things",
    "living things",
    "manmade things",
    "movable things",
    "natural things",
    "non-manmade things",
    "non-natural things",
    "nonliving things",
}

SOURCE_NORM_STRICT_NEAR_MISS_RATED_FAILURE_GLOSSES = {
    "living things",
    "movable things",
    "nonliving things",
}

SOURCE_NORM_HIGH_RISK_RATED_PROPERTY_GLOSSES = {
    "living things",
    "manmade things",
    "movable things",
    "natural things",
    "non-manmade things",
    "non-natural things",
    "nonliving things",
    "stationary things",
}

SOURCE_NORM_MATERIAL_COMPONENT_GLOSSES = {
    "things made of cloth",
    "things made of fabric",
    "things made of glass",
    "things made of metal",
    "things made of paper",
    "things made of plastic",
    "things made of rubber",
    "things made of steel",
    "things made of stone",
    "things made of wood",
}

SOURCE_NORM_TRANSPORTATION_FUNCTION_GLOSSES = {
    "things used for transportation",
}

SOURCE_NORM_TRANSPORTATION_PART_WORDS = {
    "airbag",
    "brake",
    "bumper",
    "car door",
    "exhaust pipe",
    "gearshift",
    "headlight",
    "headrest",
    "hubcap",
    "rearview mirror",
    "seatbelt",
    "spark plug",
    "steering wheel",
    "sunroof",
    "taillight",
    "wheel",
}

HIGH_RISK_ARTIFACT_CONCEPTS = {
    "appliance",
    "clothing",
    "container",
    "device",
    "equipment",
    "fastener",
    "furniture",
    "hardware",
    "musicalinstrument",
    "office",
    "supply",
    "tool",
    "utensil",
    "vehicle",
    "weapon",
}

HIGH_RISK_BODYPART_CONCEPTS = {
    "bodypart",
}

HIGH_RISK_BODYPART_DOMAINS = {
    "bodypart",
}

HIGH_RISK_ARTIFACT_DOMAINS = {
    "artifact",
    "clothing",
}

HIGH_RISK_CURATED_ARTIFACT_WORDS = {
    "chicken wire",
    "crystal ball",
    "pepper mill",
}

SOURCE_NORM_CONCEPT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "classified",
    "fit",
    "for",
    "in",
    "is",
    "item",
    "items",
    "of",
    "that",
    "thing",
    "things",
    "to",
    "used",
    "with",
}

SOURCE_NORM_OPPOSITION_AXES = [
    ({"living", "live", "lives"}, {"nonliving"}),
    ({"manmade"}, {"nonmanmade"}),
    ({"natural"}, {"nonnatural"}),
    ({"precious"}, {"nonprecious"}),
    ({"pleasant"}, {"unpleasant"}),
    ({"movable", "moves", "move"}, {"stationary"}),
    ({"heavy"}, {"lightweight"}),
    ({"graspable"}, {"hardtograsp"}),
    ({"holdable"}, {"hardtohold"}),
    ({"easily", "moved"}, {"hardtomove"}),
]

SOURCE_NORM_HIERARCHY_AXES = [
    ("animal", {"bird", "mammal", "fish", "insect", "reptile", "seafood"}),
    ("food", {"breakfast", "condiment", "fruit", "seafood", "vegetable"}),
    ("plant", {"flower", "fruit", "tree", "vegetable"}),
    ("vehicle", {"transportation"}),
]

SOURCE_NORM_ENTAILMENT_REDUNDANCY_AXES = [
    (
        {"natural"},
        (
            {"animal"},
            {"beak"},
            {"bird"},
            {"body", "part"},
            {"egg"},
            {"feather"},
            {"fish"},
            {"flower"},
            {"fruit"},
            {"fur"},
            {"furry"},
            {"insect"},
            {"lay", "egg"},
            {"mammal"},
            {"nonmanmade"},
            {"plant"},
            {"reptile"},
            {"sea", "animal"},
            {"seafood"},
            {"tree"},
            {"vegetable"},
            {"wing"},
        ),
    ),
    (
        {"movable"},
        (
            {"animal"},
            {"bird"},
            {"fish"},
            {"fly"},
            {"insect"},
            {"mammal"},
            {"reptile"},
            {"sea", "animal"},
            {"seafood"},
            {"transportation"},
            {"vehicle"},
            {"watercraft"},
        ),
    ),
    (
        {"living"},
        (
            {"animal"},
            {"beak"},
            {"bird"},
            {"eat"},
            {"egg"},
            {"feather"},
            {"fish"},
            {"flower"},
            {"food"},
            {"fruit"},
            {"fur"},
            {"furry"},
            {"insect"},
            {"lay", "egg"},
            {"mammal"},
            {"plant"},
            {"reptile"},
            {"sea", "animal"},
            {"seafood"},
            {"tree"},
            {"vegetable"},
            {"flower"},
            {"wing"},
        ),
    ),
    (
        {"furry"},
        (
            {"mammal"},
        ),
    ),
    (
        {"nonmanmade"},
        (
            {"animal"},
            {"beak"},
            {"bird"},
            {"body", "part"},
            {"egg"},
            {"feather"},
            {"fish"},
            {"flower"},
            {"fruit"},
            {"fur"},
            {"furry"},
            {"insect"},
            {"lay", "egg"},
            {"mammal"},
            {"natural"},
            {"plant"},
            {"reptile"},
            {"sea", "animal"},
            {"seafood"},
            {"tree"},
            {"vegetable"},
            {"flower"},
            {"wing"},
        ),
    ),
    (
        {"animal"},
        (
            {"beak"},
            {"bird"},
            {"egg"},
            {"feather"},
            {"fish"},
            {"fur"},
            {"furry"},
            {"insect"},
            {"lay", "egg"},
            {"mammal"},
            {"reptile"},
            {"sea", "animal"},
            {"wing"},
        ),
    ),
    (
        {"eat"},
        (
            {"animal"},
            {"bird"},
            {"fish"},
            {"insect"},
            {"lay", "egg"},
            {"mammal"},
            {"reptile"},
            {"sea", "animal"},
        ),
    ),
    (
        {"swim"},
        (
            {"fish"},
            {"sea", "animal"},
            {"seafood"},
        ),
    ),
    (
        {"water", "live"},
        (
            {"fish"},
            {"sea", "animal"},
            {"seafood"},
        ),
    ),
    (
        {"wing"},
        (
            {"bird"},
        ),
    ),
    (
        {"lay", "egg"},
        (
            {"bird"},
        ),
    ),
    (
        {"lightweight"},
        (
            {"insect"},
        ),
    ),
    (
        {"sea", "animal"},
        (
            {"seafood"},
        ),
    ),
    (
        {"plant"},
        (
            {"flower"},
            {"fruit"},
            {"tree"},
            {"vegetable"},
        ),
    ),
    (
        {"food"},
        (
            {"breakfast", "food"},
            {"candy"},
            {"condiment"},
            {"dessert"},
            {"drink"},
            {"edible"},
            {"fruit"},
            {"seafood"},
            {"taste", "good"},
            {"taste", "sweet"},
            {"vegetable"},
        ),
    ),
    (
        {"edible"},
        (
            {"breakfast", "food"},
            {"candy"},
            {"condiment"},
            {"dessert"},
            {"food"},
            {"fruit"},
            {"seafood"},
            {"taste", "good"},
            {"taste", "sweet"},
            {"vegetable"},
        ),
    ),
    (
        {"clothing"},
        (
            {"clothing", "accessory"},
            {"footwear"},
            {"headwear"},
            {"outerwear"},
            {"protective", "clothing"},
            {"women", "clothing"},
        ),
    ),
    (
        {"manmade"},
        (
            {"appliance"},
            {"clothing"},
            {"container"},
            {"device"},
            {"equipment"},
            {"fastener"},
            {"footwear"},
            {"furniture"},
            {"game"},
            {"hardware"},
            {"headwear"},
            {"home", "decor"},
            {"instrument"},
            {"jewelry"},
            {"lighting"},
            {"made"},
            {"musical", "instrument"},
            {"nonnatural"},
            {"nonliving"},
            {"outerwear"},
            {"supply"},
            {"tool"},
            {"toy"},
            {"transportation"},
            {"utensil"},
            {"vehicle"},
            {"weapon"},
        ),
    ),
    (
        {"nonnatural"},
        (
            {"appliance"},
            {"clothing"},
            {"container"},
            {"device"},
            {"equipment"},
            {"fastener"},
            {"footwear"},
            {"furniture"},
            {"game"},
            {"hardware"},
            {"headwear"},
            {"home", "decor"},
            {"instrument"},
            {"jewelry"},
            {"lighting"},
            {"manmade"},
            {"made"},
            {"musical", "instrument"},
            {"nonliving"},
            {"outerwear"},
            {"supply"},
            {"tool"},
            {"toy"},
            {"transportation"},
            {"utensil"},
            {"vehicle"},
            {"weapon"},
        ),
    ),
    (
        {"nonliving"},
        (
            {"appliance"},
            {"clothing"},
            {"container"},
            {"device"},
            {"equipment"},
            {"fastener"},
            {"footwear"},
            {"furniture"},
            {"game"},
            {"hardware"},
            {"headwear"},
            {"home", "decor"},
            {"instrument"},
            {"jewelry"},
            {"lighting"},
            {"manmade"},
            {"made"},
            {"musical", "instrument"},
            {"nonnatural"},
            {"outerwear"},
            {"supply"},
            {"tool"},
            {"toy"},
            {"transportation"},
            {"utensil"},
            {"vehicle"},
            {"weapon"},
        ),
    ),
]

SOURCE_NORM_LINT_BANNED_CONCEPT_FRAGMENTS = (
    "precious",
    "non-precious",
    "hard-to-grasp",
    "hard-to-hold",
    "hard-to-move",
    "unpleasant",
)

SOURCE_NORM_INTERSECTION_BANNED_COMPONENT_FRAGMENTS = (
    "colourful",
    "different colours",
    "things that are big",
    "things that are colorful",
    "things that are large",
    "things that are long",
    "things that are loud",
    "things that are small",
)

# Predicates in this list are allowed as positive components, but they are too
# subjective or under-specified to use as failed components in sourced AND
# near-misses unless an explicit override supplies negative evidence.
SOURCE_NORM_NEAR_MISS_HIGH_RISK_FRAGMENTS = (
    "4 legs",
    "beak",
    "dangerous",
    "edible",
    "eat",
    "eaten",
    "feathers",
    "fly",
    "fur",
    "furry",
    "grow",
    "hunted",
    "juicy",
    "legs",
    "live in water",
    "lays eggs",
    "swim",
    "tail",
    "taste",
    "tool",
    "wings",
)

# Conservative mutual exclusions. These are intentionally narrow: they only
# create negative evidence when a word is positively attested under an
# incompatible high-level concept. This is independent of absence from the
# target source-norm table and catches cases such as fish vs bird without
# turning fuzzy categories like food into closed-world labels.
SEMANTIC_EXCLUSION_GROUPS = (
    frozenset({"bird", "fish", "mammal", "insect", "reptile"}),
    frozenset({"feathers", "fur", "scales"}),
    frozenset({"vehicle", "clothing", "furniture", "bodypart"}),
    frozenset({"heavy", "lightweight"}),
    frozenset({"living", "nonliving"}),
    frozenset({"manmade", "nonmanmade"}),
    frozenset({"natural", "nonnatural"}),
    frozenset({"movable", "stationary"}),
)

SEMANTIC_CONCEPT_ALIASES = {
    "birds": "bird",
    "bird": "bird",
    "fish": "fish",
    "fishes": "fish",
    "mammal": "mammal",
    "mammals": "mammal",
    "feather": "feathers",
    "feathers": "feathers",
    "fur": "fur",
    "furry": "fur",
    "scale": "scales",
    "scales": "scales",
    "insect": "insect",
    "insects": "insect",
    "reptile": "reptile",
    "reptiles": "reptile",
    "vehicle": "vehicle",
    "vehicles": "vehicle",
    "transportation": "vehicle",
    "clothing": "clothing",
    "furniture": "furniture",
    "body": "bodypart",
    "part": "bodypart",
    "musical": "musicalinstrument",
    "instrument": "musicalinstrument",
    "instruments": "musicalinstrument",
    "appliance": "appliance",
    "appliances": "appliance",
    "container": "container",
    "containers": "container",
    "device": "device",
    "devices": "device",
    "equipment": "equipment",
    "fastener": "fastener",
    "fasteners": "fastener",
    "hardware": "hardware",
    "office": "office",
    "supply": "supply",
    "supplies": "supply",
    "tool": "tool",
    "tools": "tool",
    "weapon": "weapon",
    "weapons": "weapon",
    "utensil": "utensil",
    "utensils": "utensil",
    "heavy": "heavy",
    "lightweight": "lightweight",
    "living": "living",
    "nonliving": "nonliving",
    "manmade": "manmade",
    "nonmanmade": "nonmanmade",
    "natural": "natural",
    "nonnatural": "nonnatural",
    "movable": "movable",
    "stationary": "stationary",
}

SEMANTIC_DOMAIN_KEYWORDS = {
    "animal": {
        "animal",
        "bird",
        "fish",
        "mammal",
        "insect",
        "reptile",
    },
    "food": {
        "breakfast",
        "candy",
        "condiment",
        "dessert",
        "drink",
        "food",
        "fruit",
        "seafood",
        "vegetable",
    },
    "artifact": {
        "appliance",
        "art",
        "craft",
        "container",
        "device",
        "equipment",
        "fastener",
        "furniture",
        "game",
        "hardware",
        "instrument",
        "jewelry",
        "lighting",
        "supply",
        "tool",
        "toy",
        "utensil",
        "vehicle",
        "watercraft",
        "weapon",
    },
    "clothing": {
        "clothing",
        "footwear",
        "headwear",
        "outerwear",
    },
    "plant": {
        "plant",
    },
    "bodypart": {
        "body",
        "bodypart",
    },
}

SEMANTIC_DOMAIN_EXCLUSIONS = {
    frozenset({"animal", "artifact"}),
    frozenset({"animal", "bodypart"}),
    frozenset({"animal", "clothing"}),
    frozenset({"animal", "plant"}),
    frozenset({"artifact", "bodypart"}),
    frozenset({"artifact", "food"}),
    frozenset({"artifact", "plant"}),
    frozenset({"bodypart", "clothing"}),
    frozenset({"bodypart", "food"}),
    frozenset({"bodypart", "plant"}),
    frozenset({"clothing", "food"}),
    frozenset({"clothing", "plant"}),
}


def _source_norm_word_is_usable(word: Any) -> bool:
    text = _normalize_word_surface(word)
    if not text or text in {"nan", "none", "null"}:
        return False
    if len(text) > 36 or any(character.isdigit() for character in text):
        return False
    if any(character in text for character in ["/", "\\", "(", ")", "[", "]", "{", "}", ",", ";", ":"]):
        return False
    return bool(re.search(r"[a-z]", text))


def _source_norm_slug(text: str) -> str:
    return _slugify(re.sub(r"\s+", " ", str(text).strip().lower()))


def _source_norm_clean_gloss(gloss: Any) -> str:
    text = _category_gloss_without_incidental_disjunction(gloss)
    text = text.replace("people's", "people")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _source_norm_predicate_is_experimental_clean(predicate: Dict[str, Any]) -> bool:
    return str(predicate.get("predicate_type") or "") in SOURCE_NORM_EXPERIMENTAL_PREDICATE_TYPES


def _source_norm_predicate_is_allowed_for_intersection_pool(predicate: Dict[str, Any]) -> bool:
    predicate_type = str(predicate.get("predicate_type") or "")
    if predicate_type in SOURCE_NORM_EXPERIMENTAL_PREDICATE_TYPES:
        return True
    if predicate_type != "rated_property":
        return False
    return _source_norm_clean_gloss(predicate.get("rule_gloss") or "") in SOURCE_NORM_INTERSECTION_SAFE_RATED_PROPERTY_GLOSSES


def _source_norm_predicate_has_semantic_domain(predicate: Dict[str, Any]) -> bool:
    tokens = _source_norm_concept_tokens_from_gloss(predicate.get("rule_gloss") or "")
    return any(tokens & keywords for keywords in SEMANTIC_DOMAIN_KEYWORDS.values())


def _source_norm_concept_token(token: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(token).strip().lower())
    if not text:
        return ""
    irregular = {
        "children": "child",
        "feet": "foot",
        "geese": "goose",
        "mice": "mouse",
        "people": "person",
        "teeth": "tooth",
    }
    if text in irregular:
        return irregular[text]
    if text.endswith("ies") and len(text) > 4:
        return text[:-3] + "y"
    if text.endswith("es") and len(text) > 4:
        candidate = text[:-2]
        if candidate.endswith(("s", "x", "ch", "sh")):
            return candidate
    if text.endswith("s") and len(text) > 3:
        return text[:-1]
    return text


def _source_norm_concept_tokens_from_gloss(gloss: Any) -> set[str]:
    clean = _source_norm_clean_gloss(gloss)
    clean = clean.replace("non-natural", "nonnatural")
    clean = clean.replace("non-manmade", "nonmanmade")
    clean = clean.replace("non-precious", "nonprecious")
    tokens = {
        _source_norm_concept_token(token)
        for token in re.split(r"\s+", clean.replace("-", " "))
    }
    return {
        token
        for token in tokens
        if token and token not in SOURCE_NORM_CONCEPT_STOPWORDS
    }


def _source_norm_tokens_are_opposed(left: set[str], right: set[str]) -> bool:
    for positive, negative in SOURCE_NORM_OPPOSITION_AXES:
        if (left & positive and right & negative) or (right & positive and left & negative):
            return True
    return False


def _source_norm_tokens_are_hierarchically_redundant(left: set[str], right: set[str]) -> bool:
    for generic, specifics in SOURCE_NORM_HIERARCHY_AXES:
        generic_tokens = {generic}
        if (left == generic_tokens and right & specifics) or (right == generic_tokens and left & specifics):
            return True
    return False


def _source_norm_tokens_are_entailment_redundant(left: set[str], right: set[str]) -> bool:
    for generic_tokens, specific_token_sets in SOURCE_NORM_ENTAILMENT_REDUNDANCY_AXES:
        if left == generic_tokens and any(specific_tokens <= right for specific_tokens in specific_token_sets):
            return True
        if right == generic_tokens and any(specific_tokens <= left for specific_tokens in specific_token_sets):
            return True
    return False


def _source_norm_component_glosses_are_clean(components: Sequence[Dict[str, Any]]) -> bool:
    token_sets = [
        _source_norm_concept_tokens_from_gloss(component.get("rule_gloss") or "")
        for component in components
    ]
    if any(not tokens for tokens in token_sets):
        return False
    for left, right in itertools.combinations(token_sets, 2):
        if _source_norm_tokens_are_opposed(left, right):
            return False
        if _source_norm_tokens_are_hierarchically_redundant(left, right):
            return False
        if _source_norm_tokens_are_entailment_redundant(left, right):
            return False
        overlap = len(left & right)
        if overlap and overlap / max(1, min(len(left), len(right))) >= 0.8:
            return False
    return True


def _source_norm_predicate_is_clean_for_intersection(predicate: Dict[str, Any]) -> bool:
    gloss = _source_norm_clean_gloss(predicate.get("rule_gloss") or "")
    return not any(
        fragment in gloss
        for fragment in SOURCE_NORM_INTERSECTION_BANNED_COMPONENT_FRAGMENTS
    )


def _rule_gloss_components_for_lint(gloss: Any) -> List[str]:
    text = str(gloss).strip().lower()
    if text.startswith("items that fit ") and ":" in text:
        text = text.split(":", 1)[1]
    return [
        _source_norm_clean_gloss(part)
        for part in text.split(";")
        if str(part).strip()
    ]


def _source_norm_gloss_is_usable(gloss: str) -> bool:
    text = _source_norm_clean_gloss(gloss)
    if not text or _category_gloss_has_incidental_disjunction(text):
        return False
    banned_fragments = (
        "letter",
        "word length",
        "rhym",
        "pronounced",
        "spelled",
        "syllable",
        "homophone",
        "anagram",
    )
    if any(fragment in text for fragment in banned_fragments):
        return False
    return len(_tokenize_surface(text)) <= 10


def _source_norm_category_gloss(category: str) -> str:
    return f"things classified as {str(category).strip().lower()}"


def _source_norm_pluralize_phrase(phrase: str) -> str:
    words = str(phrase).strip().lower().split()
    if not words:
        return ""
    irregular = {
        "child": "children",
        "fish": "fish",
        "foot": "feet",
        "goose": "geese",
        "mouse": "mice",
        "person": "people",
        "sheep": "sheep",
        "tooth": "teeth",
    }
    last = words[-1]
    if last in irregular:
        words[-1] = irregular[last]
    elif last.endswith("y") and (len(last) < 2 or last[-2] not in "aeiou"):
        words[-1] = last[:-1] + "ies"
    elif last.endswith(("s", "x", "ch", "sh")):
        words[-1] = last + "es"
    else:
        words[-1] = last + "s"
    return " ".join(words)


def _source_norm_taxonomic_gloss(raw_feature_tail: str) -> str:
    phrase = _source_norm_pluralize_phrase(str(raw_feature_tail).replace("_", " "))
    return _source_norm_clean_gloss(f"things that are {phrase}")


def _source_norm_base_verb_phrase(raw_feature_tail: str) -> str:
    words = str(raw_feature_tail).replace("_", " ").strip().lower().split()
    if not words:
        return ""
    first = words[0]
    irregular = {"has": "have", "does": "do", "goes": "go"}
    if first in irregular:
        words[0] = irregular[first]
    elif first.endswith("ies") and len(first) > 3:
        words[0] = first[:-3] + "y"
    elif first.endswith("es") and first[:-2].endswith(("s", "x", "ch", "sh")):
        words[0] = first[:-2]
    elif first.endswith("s") and len(first) > 3:
        words[0] = first[:-1]
    return " ".join(words)


def _mcrae_feature_gloss(feature: str) -> str:
    raw = str(feature).strip().lower()
    raw = raw.replace("inbeh_-_", "does_").replace("inbeh_", "does_")
    raw = raw.replace("beh_-_", "does_").replace("beh_", "does_")
    text = raw.replace("_", " ")
    if raw.startswith("does_"):
        return _source_norm_clean_gloss("things that " + _source_norm_base_verb_phrase(raw[5:]))
    prefixes = [
        ("made_of_", "things made of "),
        ("used_for_", "things used for "),
        ("used_by_", "things used by "),
        ("used_in_", "things used in "),
        ("lives_in_", "things that live in "),
        ("found_in_", "things found in "),
        ("grows_on_", "things that grow on "),
        ("grows_in_", "things that grow in "),
        ("comes_from_", "things that come from "),
        ("eaten_by_", "things eaten by "),
        ("eaten_at_", "things eaten at "),
        ("eaten_for_", "things eaten for "),
        ("worn_by_", "things worn by "),
        ("has_", "things with "),
        ("is_", "things that are "),
        ("can_", "things that can "),
        ("requires_", "things that require "),
        ("causes_", "things that cause "),
        ("tastes_", "things that taste "),
        ("smells_", "things that smell "),
        ("sounds_", "things that sound "),
    ]
    for prefix, replacement in prefixes:
        if raw.startswith(prefix):
            return _source_norm_clean_gloss(replacement + raw[len(prefix):].replace("_", " "))
    if raw.startswith("a_"):
        return _source_norm_taxonomic_gloss(raw[2:])
    if raw.startswith("an_"):
        return _source_norm_taxonomic_gloss(raw[3:])
    return _source_norm_clean_gloss(text)


def _source_norm_family_tags(gloss: str, dataset: str, predicate_type: str) -> List[str]:
    tokens = _tokenize_surface(gloss)
    tags = [str(dataset), str(predicate_type)]
    stopwords = {"things", "classified", "items", "names", "that", "with", "used"}
    tags.extend(token for token in tokens if token not in stopwords)
    deduped = list(dict.fromkeys(tag for tag in tags if tag))
    return deduped[:3]


def _source_norm_reference(dataset: str) -> Dict[str, str]:
    return dict(SOURCE_NORM_REFERENCES.get(dataset, {"source_name": dataset, "source_url": "", "paper_url": "", "citation": ""}))


def _source_norm_predicate(
    *,
    predicate_id: str,
    dataset: str,
    predicate_type: str,
    gloss: str,
    word_scores: Dict[str, float],
    entity_scores_by_word: Dict[str, Dict[str, float]] | None = None,
    source_file: str,
    source_field: str,
    threshold: str,
) -> Dict[str, Any] | None:
    clean_scores = {
        _normalize_word_surface(word): float(score)
        for word, score in word_scores.items()
        if _source_norm_word_is_usable(word)
    }
    clean_entity_scores_by_word: Dict[str, Dict[str, float]] = {}
    for word, entity_scores in (entity_scores_by_word or {}).items():
        normalized_word = _normalize_word_surface(word)
        if normalized_word not in clean_scores:
            continue
        clean_entities = {
            str(entity_id): float(score)
            for entity_id, score in (entity_scores or {}).items()
            if str(entity_id)
        }
        if clean_entities:
            clean_entity_scores_by_word[normalized_word] = clean_entities
    if len(clean_scores) < SOURCE_NORM_MIN_MEMBERS:
        return None
    clean_gloss = _source_norm_clean_gloss(gloss)
    if not _source_norm_gloss_is_usable(clean_gloss):
        return None
    override = _load_semantic_word_overrides().get(clean_gloss) or {}
    for word in override.get("negative", set()):
        clean_scores.pop(word, None)
        clean_entity_scores_by_word.pop(word, None)
    for word in override.get("positive", set()):
        if _source_norm_word_is_usable(word):
            score = max(clean_scores.values(), default=1.0)
            clean_scores.setdefault(word, score)
            clean_entity_scores_by_word.setdefault(word, {f"override:{word}": score})
    if len(clean_scores) < SOURCE_NORM_MIN_MEMBERS:
        return None
    reference = _source_norm_reference(dataset)
    return {
        "predicate_id": predicate_id,
        "source_dataset": dataset,
        "source_name": reference.get("source_name", dataset),
        "source_url": reference.get("source_url", ""),
        "paper_url": reference.get("paper_url", ""),
        "citation": reference.get("citation", ""),
        "predicate_type": predicate_type,
        "rule_gloss": clean_gloss,
        "word_scores": clean_scores,
        "entity_scores_by_word": clean_entity_scores_by_word,
        "source_file": source_file,
        "source_field": source_field,
        "threshold": threshold,
    }


def _source_norm_ranked_words(predicate: Dict[str, Any], limit: int | None = None) -> List[str]:
    scores = predicate.get("word_scores") or {}
    gloss = _source_norm_clean_gloss(predicate.get("rule_gloss") or "")
    override = _load_semantic_word_overrides().get(gloss) or {}
    excluded_words = set(override.get("negative", set()))
    ranked = [
        word
        for word, _ in sorted(
            ((str(word), float(score)) for word, score in scores.items()),
            key=lambda item: (-item[1], item[0]),
        )
        if _normalize_word_surface(word) not in excluded_words
    ]
    return ranked[:limit] if limit is not None else ranked


def _source_norm_blocks_from_words(words: Sequence[str], key: str) -> List[List[str]]:
    ranked = _normalize_word_list(words)
    if len(ranked) < SOURCE_NORM_MIN_MEMBERS:
        raise ValueError(f"Need at least {SOURCE_NORM_MIN_MEMBERS} source words for {key}.")
    selected_count = min(
        len(ranked),
        max(SOURCE_NORM_MIN_MEMBERS, SOURCE_NORM_TARGET_VISIBLE_MEMBERS),
    )
    selected = ranked[:selected_count]
    if len(selected) != len(set(selected)):
        raise ValueError(f"Source predicate {key} produced duplicate words.")
    return [selected[index:index + 4] for index in range(0, len(selected), 4)]


def _source_norm_bundle_from_predicate(predicate: Dict[str, Any]) -> Dict[str, Any]:
    predicate_id = str(predicate["predicate_id"])
    dataset = str(predicate["source_dataset"])
    blocks = _source_norm_blocks_from_words(_source_norm_ranked_words(predicate), predicate_id)
    block_words = list(itertools.chain.from_iterable(blocks))
    source_summary = {
        "bundle_builder": "source_norms_atomic_v1",
        "source_repo": "concept_property_norms",
        "source_dataset": dataset,
        "source_name": predicate.get("source_name"),
        "source_url": predicate.get("source_url"),
        "paper_url": predicate.get("paper_url"),
        "citation": predicate.get("citation"),
        "seed_category_name": predicate_id,
        "source_words": list(blocks[0]),
        "source_file": predicate.get("source_file"),
        "source_field": predicate.get("source_field"),
        "predicate_type": predicate.get("predicate_type"),
        "threshold": predicate.get("threshold"),
        "membership_count": len(predicate.get("word_scores") or {}),
        "membership_basis": "sourced_concept_property_membership",
        "component_indices_by_word": {
            word: [0]
            for word in block_words
        },
        "component_words_by_index": [block_words],
    }
    return _with_category_family_metadata({
        "bundle_id": f"source_{predicate_id}",
        "difficulty": "d1",
        "knowledge_type": "semantic",
        "complexity": "simple",
        "examples_favored": True,
        "rule_gloss": predicate["rule_gloss"],
        "red_herring_tags": _source_norm_family_tags(predicate["rule_gloss"], dataset, predicate.get("predicate_type", "")),
        "source": source_summary,
        "blocks": blocks,
    })


def _read_tsv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", errors="replace") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _read_csv_rows(path: Path) -> List[List[str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", errors="replace") as handle:
        return list(csv.reader(handle))


def _load_thingsplus_predicates(source_norms_root: Path) -> List[Dict[str, Any]]:
    things_root = source_norms_root / "thingsplus"
    category_path = things_root / "category53_wide-format.tsv"
    typicality_path = things_root / "typicality53_mean-ratings.tsv"
    property_path = things_root / "property-ratings.tsv"
    typicality_scores: Dict[tuple[str, str], float] = {}
    for row in _read_tsv_dicts(typicality_path):
        word = _normalize_word_surface(row.get("Word") or row.get("member") or "")
        unique_id = _normalize_word_surface(row.get("uniqueID") or "")
        category = _normalize_word_surface(row.get("category") or "")
        try:
            score = float(row.get("typicality_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if word and category:
            typicality_scores[(category, word)] = score
        if unique_id and category:
            typicality_scores[(category, unique_id)] = score

    predicates: List[Dict[str, Any]] = []
    category_rows = _read_tsv_dicts(category_path)
    if category_rows:
        columns = [
            column
            for column in category_rows[0].keys()
            if column not in {"uniqueID", "Word"}
        ]
        for column in columns:
            words: Dict[str, float] = {}
            entity_scores_by_word: Dict[str, Dict[str, float]] = {}
            normalized_column = _normalize_word_surface(column)
            for row in category_rows:
                if str(row.get(column) or "").strip() != "1":
                    continue
                word = _normalize_word_surface(row.get("Word") or "")
                unique_id = _normalize_word_surface(row.get("uniqueID") or word)
                if not word:
                    continue
                score = typicality_scores.get(
                    (normalized_column, unique_id),
                    typicality_scores.get((normalized_column, word), 1.0),
                )
                words[word] = max(float(score), words.get(word, float("-inf")))
                entity_scores_by_word.setdefault(word, {})[unique_id] = float(score)
            predicate = _source_norm_predicate(
                predicate_id=f"thingsplus_category_{_source_norm_slug(column)}",
                dataset="thingsplus",
                predicate_type="category",
                gloss=_source_norm_category_gloss(column),
                word_scores=words,
                entity_scores_by_word=entity_scores_by_word,
                source_file=str(category_path.relative_to(DATA_ROOT)),
                source_field=column,
                threshold="category53_wide_format == 1",
            )
            if predicate:
                predicates.append(predicate)

    property_rows = _read_tsv_dicts(property_path)
    for column, specs in SOURCE_NORM_PROPERTY_SPECS.items():
        if not property_rows or column not in property_rows[0]:
            continue
        for suffix, gloss, threshold_value, high_is_positive in specs:
            words: Dict[str, float] = {}
            entity_scores_by_word: Dict[str, Dict[str, float]] = {}
            for row in property_rows:
                word = _normalize_word_surface(row.get("Word") or "")
                unique_id = _normalize_word_surface(row.get("uniqueID") or word)
                try:
                    value = float(row.get(column) or "nan")
                except (TypeError, ValueError):
                    continue
                if high_is_positive and value >= threshold_value:
                    score = value
                elif not high_is_positive and value <= threshold_value:
                    score = 7.0 - value
                else:
                    continue
                if not word:
                    continue
                words[word] = max(float(score), words.get(word, float("-inf")))
                entity_scores_by_word.setdefault(word, {})[unique_id] = float(score)
            predicate = _source_norm_predicate(
                predicate_id=f"thingsplus_property_{suffix}_{_source_norm_slug(column.replace('property_', '').replace('_mean', ''))}",
                dataset="thingsplus",
                predicate_type="rated_property",
                gloss=gloss,
                word_scores=words,
                entity_scores_by_word=entity_scores_by_word,
                source_file=str(property_path.relative_to(DATA_ROOT)),
                source_field=column,
                threshold=(f"{column} >= {threshold_value}" if high_is_positive else f"{column} <= {threshold_value}"),
            )
            if predicate:
                predicates.append(predicate)
    return predicates


def _load_mcrae_predicates(source_norms_root: Path) -> List[Dict[str, Any]]:
    mcrae_path = source_norms_root / "mcrae" / "McRae-BRM-InPress" / "CONCS_FEATS_concstats_brm.txt"
    feature_scores: Dict[str, Dict[str, float]] = {}
    if not mcrae_path.exists():
        return []
    for row in _read_tsv_dicts(mcrae_path):
        feature = str(row.get("Feature") or "").strip()
        word = _normalize_word_surface(row.get("Concept") or "")
        try:
            production_frequency = float(row.get("Prod_Freq") or 0.0)
        except (TypeError, ValueError):
            production_frequency = 0.0
        if production_frequency < 5 or not feature:
            continue
        feature_scores.setdefault(feature, {})[word] = production_frequency

    predicates: List[Dict[str, Any]] = []
    for feature, scores in sorted(feature_scores.items()):
        predicate = _source_norm_predicate(
            predicate_id=f"mcrae_feature_{_source_norm_slug(feature)}",
            dataset="mcrae",
            predicate_type="produced_feature",
            gloss=_mcrae_feature_gloss(feature),
            word_scores=scores,
            entity_scores_by_word={
                word: {f"surface:{word}": score}
                for word, score in scores.items()
            },
            source_file=str(mcrae_path.relative_to(DATA_ROOT)),
            source_field=feature,
            threshold="Prod_Freq >= 5",
        )
        if predicate:
            predicates.append(predicate)
    return predicates


def _cslb_column_value(row: Dict[str, str], names: Sequence[str]) -> str:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        if name in lowered:
            return lowered[name]
    return ""


def _load_cslb_predicates(source_norms_root: Path) -> List[Dict[str, Any]]:
    cslb_root = source_norms_root / "cslb"
    if not cslb_root.exists():
        return []
    candidate_files = [
        path
        for path in cslb_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tsv", ".txt", ".csv"}
    ]
    feature_scores: Dict[str, Dict[str, float]] = {}
    source_file_by_feature: Dict[str, str] = {}
    for path in candidate_files:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        with path.open("r", newline="", errors="replace") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if not reader.fieldnames:
                continue
            lower_fields = {field.lower() for field in reader.fieldnames}
            if not {"concept", "feature"} <= lower_fields:
                continue
            for row in reader:
                concept = _normalize_word_surface(_cslb_column_value(row, ["concept", "word", "item"]))
                feature = _cslb_column_value(row, ["feature", "property", "normalized_feature", "normed_feature"])
                frequency_text = _cslb_column_value(row, ["prod_freq", "production_frequency", "frequency", "freq"])
                try:
                    frequency = float(frequency_text or 1.0)
                except (TypeError, ValueError):
                    frequency = 1.0
                if frequency < 2:
                    continue
                feature_scores.setdefault(feature, {})[concept] = frequency
                source_file_by_feature[feature] = str(path.relative_to(DATA_ROOT))

    predicates: List[Dict[str, Any]] = []
    for feature, scores in sorted(feature_scores.items()):
        predicate = _source_norm_predicate(
            predicate_id=f"cslb_feature_{_source_norm_slug(feature)}",
            dataset="cslb",
            predicate_type="produced_feature",
            gloss=_mcrae_feature_gloss(feature),
            word_scores=scores,
            entity_scores_by_word={
                word: {f"surface:{word}": score}
                for word, score in scores.items()
            },
            source_file=source_file_by_feature.get(feature, ""),
            source_field=feature,
            threshold="production frequency >= 2",
        )
        if predicate:
            predicates.append(predicate)
    return predicates


def _load_source_norm_predicates(source_norms_root: str | Path | None = None) -> List[Dict[str, Any]]:
    root = Path(source_norms_root) if source_norms_root else DEFAULT_SOURCE_NORMS_ROOT
    predicates = (
        _load_thingsplus_predicates(root)
        + _load_mcrae_predicates(root)
        + _load_cslb_predicates(root)
    )
    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for predicate in sorted(predicates, key=lambda item: str(item.get("predicate_id") or "")):
        words = tuple(_source_norm_ranked_words(predicate))
        key = (str(predicate.get("rule_gloss") or ""), words)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(predicate)
    if not deduped:
        raise FileNotFoundError(
            f"No source-norm predicates could be loaded from {root}. "
            "Expected THINGSplus/McRae files under data/lexical_category_inference/source_norms."
        )
    return deduped


def _source_norm_predicate_priority(predicate: Dict[str, Any]) -> tuple[Any, ...]:
    dataset_rank = {"thingsplus": 0, "mcrae": 1, "cslb": 2}.get(str(predicate.get("source_dataset")), 3)
    type_rank = {"category": 0, "produced_feature": 1, "rated_property": 2}.get(str(predicate.get("predicate_type")), 3)
    member_count = len(predicate.get("word_scores") or {})
    return (type_rank, dataset_rank, abs(member_count - 32), str(predicate.get("predicate_id") or ""))


def _select_source_predicates_for_atomic_pool(
    predicates: Sequence[Dict[str, Any]],
    target_count: int = SOURCE_NORM_MAX_ATOMIC_PREDICATES,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for predicate in sorted(predicates, key=_source_norm_predicate_priority):
        dataset = str(predicate.get("source_dataset") or "unknown")
        grouped.setdefault(dataset, []).append(predicate)

    selected: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    while len(selected) < target_count:
        made_progress = False
        for dataset in sorted(grouped):
            while grouped[dataset]:
                candidate = grouped[dataset].pop(0)
                candidate_id = str(candidate.get("predicate_id") or "")
                if candidate_id in seen_ids:
                    continue
                selected.append(candidate)
                seen_ids.add(candidate_id)
                made_progress = True
                break
            if len(selected) >= target_count:
                break
        if not made_progress:
            break
    return selected


def _source_norm_predicate_token_set(predicate: Dict[str, Any]) -> set[str]:
    return set(_tokenize_surface(str(predicate.get("rule_gloss") or "")))


def _source_norm_components_are_redundant(components: Sequence[Dict[str, Any]], intersection_size: int) -> bool:
    if len({str(component.get("rule_gloss") or "") for component in components}) != len(components):
        return True
    if not _source_norm_component_glosses_are_clean(components):
        return True
    component_sets = [set(component.get("word_scores") or {}) for component in components]
    for left, right in itertools.combinations(component_sets, 2):
        if not left or not right:
            return True
        pair_intersection = len(left & right)
        if pair_intersection / min(len(left), len(right)) >= 0.95:
            return True
    if any(intersection_size / len(component_set) >= 0.95 for component_set in component_sets if component_set):
        return True
    token_sets = [_source_norm_predicate_token_set(component) for component in components]
    for left, right in itertools.combinations(token_sets, 2):
        if left and right and len(left & right) / len(left | right) >= 0.75:
            return True
    return False


def _source_norm_entity_ids_for_word(
    component: Dict[str, Any],
    word: Any,
) -> set[str]:
    entity_scores = component.get("entity_scores_by_word") or {}
    return {
        str(entity_id)
        for entity_id in (entity_scores.get(_normalize_word_surface(word)) or {})
        if str(entity_id)
    }


def _source_norm_common_thingsplus_entity_ids(
    word: Any,
    components: Sequence[Dict[str, Any]],
) -> List[str]:
    thingsplus_entity_sets = [
        _source_norm_entity_ids_for_word(component, word)
        for component in components
        if str(component.get("source_dataset") or "") == "thingsplus"
        and _source_norm_entity_ids_for_word(component, word)
    ]
    if len(thingsplus_entity_sets) < 2:
        return []
    common = set(thingsplus_entity_sets[0])
    for entity_ids in thingsplus_entity_sets[1:]:
        common &= entity_ids
    return sorted(common)


def _source_norm_has_same_sense_for_intersection_word(
    word: Any,
    components: Sequence[Dict[str, Any]],
) -> bool:
    thingsplus_entity_sets = [
        _source_norm_entity_ids_for_word(component, word)
        for component in components
        if str(component.get("source_dataset") or "") == "thingsplus"
        and _source_norm_entity_ids_for_word(component, word)
    ]
    if len(thingsplus_entity_sets) < 2:
        return True
    common = set(thingsplus_entity_sets[0])
    for entity_ids in thingsplus_entity_sets[1:]:
        common &= entity_ids
    return bool(common)


@functools.lru_cache(maxsize=200_000)
def _source_norm_positive_plausibility_issue_tuple(
    word: str,
    gloss: str,
) -> tuple[str, ...]:
    validator = get_default_semantic_validator()
    return tuple(
        validator.rated_property_plausibility_issues(
            word,
            gloss,
            True,
            basis="source_positive",
        )
    )


def _source_norm_positive_plausibility_issues(
    word: Any,
    component: Dict[str, Any],
    validator: SemanticValidator | None = None,
) -> List[str]:
    normalized_word = _normalize_word_surface(word)
    clean_gloss = _source_norm_clean_gloss(component.get("rule_gloss") or "")
    if clean_gloss not in SOURCE_NORM_HIGH_RISK_RATED_PROPERTY_GLOSSES:
        return []
    if validator is None:
        return list(_source_norm_positive_plausibility_issue_tuple(normalized_word, clean_gloss))
    return validator.rated_property_plausibility_issues(
        normalized_word,
        clean_gloss,
        True,
        basis="source_positive",
    )


def _source_norm_word_has_high_risk_positive_conflict(
    word: Any,
    components: Sequence[Dict[str, Any]],
    validator: SemanticValidator | None = None,
) -> bool:
    validator = validator or get_default_semantic_validator()
    return any(
        _source_norm_positive_plausibility_issues(word, component, validator)
        for component in components
    )


def _source_norm_material_component_indices(components: Sequence[Dict[str, Any]]) -> List[int]:
    return [
        index
        for index, component in enumerate(components)
        if _source_norm_clean_gloss(component.get("rule_gloss") or "")
        in SOURCE_NORM_MATERIAL_COMPONENT_GLOSSES
    ]


def _source_norm_has_uncurated_multi_material_components(components: Sequence[Dict[str, Any]]) -> bool:
    # McRae produced material features are attached to surface nouns, so a
    # generic word like "fork" can inherit evidence for metal and plastic from
    # different common variants. Strict AND benchmarks need same-instance
    # composition; without curated evidence, multi-material intersections are
    # excluded from the source-derived bundle bank.
    return len(_source_norm_material_component_indices(components)) >= 2


def _source_norm_intersection_words(components: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not components:
        return {}
    if _source_norm_has_uncurated_multi_material_components(components):
        return {}
    common = set(components[0].get("word_scores") or {})
    for component in components[1:]:
        common &= set(component.get("word_scores") or {})
    overrides = _load_semantic_word_overrides()
    for component in components:
        gloss = _category_gloss_without_incidental_disjunction(component.get("rule_gloss") or "").lower()
        common -= overrides.get(gloss, {}).get("negative", set())
    scores: Dict[str, float] = {}
    high_risk_components = [
        component
        for component in components
        if _source_norm_clean_gloss(component.get("rule_gloss") or "")
        in SOURCE_NORM_HIGH_RISK_RATED_PROPERTY_GLOSSES
    ]
    for word in common:
        if not _source_norm_has_same_sense_for_intersection_word(word, components):
            continue
        if high_risk_components and _source_norm_word_has_high_risk_positive_conflict(
            word,
            high_risk_components,
            None,
        ):
            continue
        scores[word] = sum(float((component.get("word_scores") or {}).get(word, 0.0)) for component in components)
    return scores


def _source_norm_intersection_score(components: Sequence[Dict[str, Any]], intersection_size: int, target_size: int) -> tuple[Any, ...]:
    sources = [str(component.get("source_dataset") or "") for component in components]
    types = [str(component.get("predicate_type") or "") for component in components]
    member_counts = [len(component.get("word_scores") or {}) for component in components]
    source_diversity_bonus = len(set(sources))
    type_diversity_bonus = len(set(types))
    generic_penalty = _safe_mean([math.log(max(2, count)) for count in member_counts])
    size_penalty = abs(intersection_size - target_size) / max(1, target_size)
    ids = tuple(str(component.get("predicate_id") or "") for component in components)
    return (size_penalty, -source_diversity_bonus, -type_diversity_bonus, generic_penalty, ids)


def _source_norm_near_misses_by_component(
    components: Sequence[Dict[str, Any]],
    intersection_words: set[str],
    *,
    limit_per_component: int = 64,
) -> tuple[List[List[str]], Dict[str, List[int]]]:
    validator = get_default_semantic_validator()
    component_glosses = [
        _category_gloss_without_incidental_disjunction(component.get("rule_gloss") or "").lower()
        for component in components
    ]
    component_word_sets: List[set[str]] = []
    for component in components:
        gloss = _category_gloss_without_incidental_disjunction(component.get("rule_gloss") or "").lower()
        words = {
            _normalize_word_surface(word)
            for word in (component.get("word_scores") or {})
        }
        override = validator.overrides.get(gloss) or {}
        words.update(override.get("positive", set()))
        words.difference_update(override.get("negative", set()))
        component_word_sets.append(words)

    near_misses_by_component: List[List[str]] = [
        []
        for _ in components
    ]
    component_indices_by_word: Dict[str, List[int]] = {}
    globally_used: set[str] = set()
    rank_maps: List[Dict[str, int]] = []
    for component in components:
        rank_maps.append({
            _normalize_word_surface(word): rank
            for rank, word in enumerate(_source_norm_ranked_words(component))
        })

    for failed_index in range(len(components)):
        satisfied_component_indices = [
            index
            for index in range(len(components))
            if index != failed_index
        ]
        if not satisfied_component_indices:
            continue
        candidate_words = set(component_word_sets[satisfied_component_indices[0]])
        for component_index in satisfied_component_indices[1:]:
            candidate_words &= component_word_sets[component_index]
        candidate_words -= component_word_sets[failed_index]
        candidate_words -= set(intersection_words)
        ranked_candidates = sorted(
            candidate_words,
            key=lambda word: (
                sum(rank_maps[index].get(word, 10_000_000) for index in satisfied_component_indices),
                rank_maps[failed_index].get(word, 10_000_000),
                word,
            ),
        )
        for word in ranked_candidates:
            if len(near_misses_by_component[failed_index]) >= limit_per_component:
                break
            if word in globally_used:
                continue
            evidence = validator.component_evidence_profile(
                word,
                component_glosses,
                component_words_by_index=component_word_sets,
            )
            values = [entry.get("value") for entry in evidence]
            if not values:
                continue
            satisfied_indices = [
                index
                for index, value in enumerate(values)
                if value is True
            ]
            failed_indices = [
                index
                for index, value in enumerate(values)
                if value is False
            ]
            unknown_indices = [
                index
                for index, value in enumerate(values)
                if value is None
            ]
            if unknown_indices:
                continue
            if satisfied_indices != satisfied_component_indices or failed_indices != [failed_index]:
                continue
            if not _evidence_has_independent_negative_for_high_risk_failure(
                evidence[failed_index],
                component_glosses[failed_index],
            ):
                continue
            if _strict_near_miss_semantic_disqualification_reasons(
                word,
                component_glosses,
                evidence,
                validator,
            ):
                continue
            near_misses_by_component[failed_index].append(word)
            component_indices_by_word[word] = satisfied_indices
            globally_used.add(word)

    return near_misses_by_component, component_indices_by_word


def _source_norm_intersection_candidate_bundle(
    components: Sequence[Dict[str, Any]],
    difficulty: str,
    intersection_scores: Dict[str, float],
) -> Dict[str, Any]:
    canonical_components = sorted(components, key=lambda component: str(component.get("predicate_id") or ""))
    canonical_component_ids = [str(component["predicate_id"]) for component in canonical_components]
    digest = hashlib.sha1(f"both:{difficulty}:{'|'.join(canonical_component_ids)}".encode("utf-8")).hexdigest()[:12]
    if canonical_components:
        rotation = int(digest[:2], 16) % len(canonical_components)
        ordered_components = canonical_components[rotation:] + canonical_components[:rotation]
    else:
        ordered_components = []
    component_ids = [str(component["predicate_id"]) for component in ordered_components]
    knowledge_type = "two_way_intersection" if difficulty == "d2" else "three_way_intersection"
    bundle_id = f"{knowledge_type}_{digest}"
    ranked_words = [
        word
        for word, _ in sorted(intersection_scores.items(), key=lambda item: (-float(item[1]), item[0]))
    ]
    blocks = _source_norm_blocks_from_words(ranked_words, bundle_id)
    block_words = list(itertools.chain.from_iterable(blocks))
    all_component_indices = list(range(len(ordered_components)))
    near_miss_words_by_component, near_miss_component_indices_by_word = _source_norm_near_misses_by_component(
        ordered_components,
        set(intersection_scores),
    )
    component_glosses = [str(component["rule_gloss"]) for component in ordered_components]
    component_types = ["semantic" for _ in ordered_components]
    component_sources = [str(component.get("source_dataset") or "") for component in ordered_components]
    source_entity_ids_by_word = {
        word: _source_norm_common_thingsplus_entity_ids(word, ordered_components)
        for word in block_words
    }
    component_entity_ids_by_word_by_index = [
        {
            word: sorted(_source_norm_entity_ids_for_word(component, word))
            for word in block_words
            if _source_norm_entity_ids_for_word(component, word)
        }
        for component in ordered_components
    ]
    source_summary = {
        "bundle_builder": "source_norms_intersection_v1",
        "source_repo": "concept_property_norms",
        "source_datasets": component_sources,
        "source_names": [component.get("source_name") for component in ordered_components],
        "source_urls": [component.get("source_url") for component in ordered_components],
        "paper_urls": [component.get("paper_url") for component in ordered_components],
        "citations": [component.get("citation") for component in ordered_components],
        "seed_category_name": bundle_id,
        "source_words": list(blocks[0]),
        "component_predicate_ids": component_ids,
        "component_rule_glosses": component_glosses,
        "component_knowledge_types": component_types,
        "component_predicate_types": [component.get("predicate_type") for component in ordered_components],
        "component_source_files": [component.get("source_file") for component in ordered_components],
        "component_source_fields": [component.get("source_field") for component in ordered_components],
        "component_thresholds": [component.get("threshold") for component in ordered_components],
        "component_membership_counts": [len(component.get("word_scores") or {}) for component in ordered_components],
        "component_indices_by_word": {
            word: all_component_indices
            for word in block_words
        },
        "source_entity_ids_by_word": source_entity_ids_by_word,
        "component_entity_ids_by_word_by_index": component_entity_ids_by_word_by_index,
        "component_words_by_index": [
            _source_norm_ranked_words(component)
            for component in ordered_components
        ],
        "near_miss_words_by_component": near_miss_words_by_component,
        "near_miss_component_indices_by_word": near_miss_component_indices_by_word,
        "intersection_membership_count": len(intersection_scores),
        "membership_basis": "sourced_concept_property_membership",
        "intersection_basis": "same_source_entity_membership",
        "component_red_herring_tags": [
            _source_norm_family_tags(component["rule_gloss"], str(component.get("source_dataset") or ""), str(component.get("predicate_type") or ""))
            for component in ordered_components
        ],
        "composite_operator": "and",
        "composite_recipe": [1 for _ in ordered_components],
        "composite_type_name": f"{len(ordered_components)}-way sourced conceptual intersection",
    }
    return _apply_design_metadata_to_bundle({
        "bundle_id": bundle_id,
        "difficulty": difficulty,
        "knowledge_type": knowledge_type,
        "complexity": "medium" if difficulty == "d2" else "complex",
        "examples_favored": True,
        "rule_gloss": _intersection_rule_gloss(component_glosses),
        "red_herring_tags": list(dict.fromkeys(["intersection", "all_conditions", *component_sources]))[:3],
        "source": source_summary,
        "blocks": blocks,
    }, "both", difficulty)


def _select_source_intersection_bundles(
    predicates: Sequence[Dict[str, Any]],
    *,
    difficulty: str,
    target_count: int,
) -> List[Dict[str, Any]]:
    component_count = {"d2": 2, "d3": 3}[difficulty]
    target_size = 28 if difficulty == "d2" else 18
    scored: List[tuple[tuple[Any, ...], tuple[str, ...], Dict[str, float], tuple[Dict[str, Any], ...]]] = []
    predicate_list = sorted(predicates, key=lambda predicate: str(predicate.get("predicate_id") or ""))
    overrides = _load_semantic_word_overrides()
    cleaned_word_scores: List[Dict[str, float]] = []
    cleaned_word_sets: List[set[str]] = []
    for predicate in predicate_list:
        gloss = _category_gloss_without_incidental_disjunction(predicate.get("rule_gloss") or "").lower()
        excluded = overrides.get(gloss, {}).get("negative", set())
        scores = {
            _normalize_word_surface(word): float(score)
            for word, score in (predicate.get("word_scores") or {}).items()
            if _normalize_word_surface(word) not in excluded
        }
        cleaned_word_scores.append(scores)
        cleaned_word_sets.append(set(scores))

    for component_indices in itertools.combinations(range(len(predicate_list)), component_count):
        components = tuple(predicate_list[index] for index in component_indices)
        intersection_scores = _source_norm_intersection_words(components)
        common_words = set(intersection_scores)
        intersection_size = len(intersection_scores)
        if intersection_size < SOURCE_NORM_MIN_MEMBERS:
            continue
        source_near_miss_covered = False
        for failed_index in component_indices:
            satisfied_indices = [index for index in component_indices if index != failed_index]
            if not satisfied_indices:
                break
            near_miss_words = set(cleaned_word_sets[satisfied_indices[0]])
            for satisfied_index in satisfied_indices[1:]:
                near_miss_words &= cleaned_word_sets[satisfied_index]
            near_miss_words -= cleaned_word_sets[failed_index]
            near_miss_words -= common_words
            if near_miss_words:
                source_near_miss_covered = True
                break
        if not source_near_miss_covered:
            continue
        if _source_norm_components_are_redundant(components, intersection_size):
            continue
        ids = tuple(str(component.get("predicate_id") or "") for component in components)
        score = _source_norm_intersection_score(components, intersection_size, target_size)
        scored.append((score, ids, intersection_scores, tuple(components)))

    if len(scored) < target_count:
        raise ValueError(
            f"Need at least {target_count} sourced {difficulty} intersection candidates, found {len(scored)}."
        )

    selected: List[Dict[str, Any]] = []
    selected_component_sets: set[tuple[str, ...]] = set()
    component_usage: Counter[str] = Counter()
    source_usage: Counter[str] = Counter()
    selected_word_usage: Counter[str] = Counter()
    family_usage: Counter[str] = Counter()
    near_miss_coverage_cache: Dict[tuple[str, ...], bool] = {}
    scored.sort(key=lambda item: item[0])
    hard_component_usage_cap = (
        max(9, math.ceil(target_count * 0.12))
        if difficulty == "d3"
        else None
    )

    def has_any_strict_near_miss(
        ids: tuple[str, ...],
        components: tuple[Dict[str, Any], ...],
        intersection_scores: Dict[str, float],
    ) -> bool:
        cached = near_miss_coverage_cache.get(ids)
        if cached is not None:
            return cached
        near_misses_by_component, _ = _source_norm_near_misses_by_component(
            sorted(components, key=lambda component: str(component.get("predicate_id") or "")),
            set(intersection_scores),
            limit_per_component=1,
        )
        covered = len(near_misses_by_component) == component_count and any(
            bool(words) for words in near_misses_by_component
        )
        near_miss_coverage_cache[ids] = covered
        return covered

    for max_component_usage in (1, 2, 3, 5, 8, 999):
        for _, ids, intersection_scores, components in scored:
            if len(selected) >= target_count:
                break
            if ids in selected_component_sets:
                continue
            if hard_component_usage_cap is not None and any(
                component_usage[component_id] >= hard_component_usage_cap
                for component_id in ids
            ):
                continue
            if any(component_usage[component_id] >= max_component_usage for component_id in ids):
                continue
            candidate_sources = [
                str(component.get("source_dataset") or "")
                for component in components
            ]
            if max_component_usage <= 2 and len(set(candidate_sources)) == 1 and source_usage[candidate_sources[0]] > len(selected) // 2:
                continue
            candidate_words = {
                word
                for word, _ in sorted(intersection_scores.items(), key=lambda item: (-float(item[1]), item[0]))[:SOURCE_NORM_MIN_MEMBERS]
            }
            if difficulty in {"d2", "d3"}:
                overlap = sum(1 for word in candidate_words if selected_word_usage[word] > 0)
                allowed_overlap = {1: 0, 2: 1, 3: 2, 5: 4, 8: 6}.get(max_component_usage, SOURCE_NORM_MIN_MEMBERS)
                if overlap > allowed_overlap:
                    continue
            if not has_any_strict_near_miss(ids, components, intersection_scores):
                continue
            bundle = _source_norm_intersection_candidate_bundle(components, difficulty, intersection_scores)
            primary_family = _primary_category_family(bundle)
            if difficulty == "d3" and max_component_usage < 999:
                family_cap = max(1, math.ceil(target_count * 0.35))
                if primary_family and family_usage[primary_family] >= family_cap:
                    continue
            selected.append(bundle)
            selected_component_sets.add(ids)
            component_usage.update(ids)
            source_usage.update(candidate_sources)
            selected_word_usage.update(candidate_words)
            if primary_family:
                family_usage[primary_family] += 1
        if len(selected) >= target_count:
            break

    if len(selected) < target_count:
        raise ValueError(f"Could only select {len(selected)} sourced {difficulty} intersection bundles.")
    return sorted(selected[:target_count], key=lambda bundle: str(bundle.get("bundle_id") or ""))


def _source_intersection_candidate_bundles(
    predicates: Sequence[Dict[str, Any]],
    *,
    difficulty: str,
) -> List[Dict[str, Any]]:
    component_count = {"d2": 2, "d3": 3}[difficulty]
    target_size = 28 if difficulty == "d2" else 18
    scored: List[tuple[tuple[Any, ...], Dict[str, float], tuple[Dict[str, Any], ...]]] = []
    predicate_list = sorted(predicates, key=lambda predicate: str(predicate.get("predicate_id") or ""))

    for component_indices in itertools.combinations(range(len(predicate_list)), component_count):
        components = tuple(predicate_list[index] for index in component_indices)
        intersection_scores = _source_norm_intersection_words(components)
        intersection_size = len(intersection_scores)
        if intersection_size < SOURCE_NORM_MIN_MEMBERS:
            continue
        if _source_norm_components_are_redundant(components, intersection_size):
            continue
        near_misses_by_component, _ = _source_norm_near_misses_by_component(
            sorted(components, key=lambda component: str(component.get("predicate_id") or "")),
            set(intersection_scores),
            limit_per_component=1,
        )
        if not any(bool(words) for words in near_misses_by_component):
            continue
        score = _source_norm_intersection_score(components, intersection_size, target_size)
        scored.append((score, intersection_scores, tuple(components)))

    scored.sort(key=lambda item: item[0])
    candidates: List[Dict[str, Any]] = []
    seen_bundle_ids: set[str] = set()
    for _, intersection_scores, components in scored:
        bundle = _source_norm_intersection_candidate_bundle(components, difficulty, intersection_scores)
        bundle_id = str(bundle.get("bundle_id") or "")
        if bundle_id in seen_bundle_ids:
            continue
        seen_bundle_ids.add(bundle_id)
        candidates.append(bundle)
    return candidates


def _category_balance_target_usage(num_rows: int, bundle_count: int) -> int:
    total_category_slots = int(num_rows) * 4
    if bundle_count <= 0 or total_category_slots % bundle_count != 0:
        raise ValueError(
            f"Cannot exactly balance {num_rows} rows across {bundle_count} category definitions."
        )
    return total_category_slots // bundle_count


def _exact_balance_valid_edges(
    logic_condition: str,
    difficulty: str,
    bundles: Sequence[Dict[str, Any]],
) -> List[tuple[str, str, str, str]]:
    bundle_by_id = {str(bundle["bundle_id"]): bundle for bundle in bundles}
    valid_edges: List[tuple[str, str, str, str]] = []
    for bundle_ids in itertools.combinations(sorted(bundle_by_id), 4):
        if _selection_is_valid(
            logic_condition,
            difficulty,
            [bundle_by_id[bundle_id] for bundle_id in bundle_ids],
        ):
            valid_edges.append(tuple(bundle_ids))
    return valid_edges


def _find_exact_category_balance_quartet_schedule(
    logic_condition: str,
    difficulty: str,
    bundles: Sequence[Dict[str, Any]],
    *,
    num_rows: int,
    rng: random.Random,
    max_restarts: int = 250,
) -> List[tuple[str, str, str, str]]:
    bundle_ids = sorted(str(bundle["bundle_id"]) for bundle in bundles)
    cache_key = (logic_condition, difficulty, int(num_rows), tuple(bundle_ids))
    cached = _EXACT_CATEGORY_BALANCE_SCHEDULE_CACHE.get(cache_key)
    if cached is not None:
        scheduled_edges = list(cached)
        rng.shuffle(scheduled_edges)
        return scheduled_edges
    target_usage = _category_balance_target_usage(num_rows, len(bundle_ids))
    edges = _exact_balance_valid_edges(logic_condition, difficulty, bundles)
    if not edges:
        raise RuntimeError(f"No valid quartet edges for exact category balance in {logic_condition}/{difficulty}.")

    edges_by_bundle: Dict[str, List[tuple[str, str, str, str]]] = defaultdict(list)
    for edge in edges:
        for bundle_id in edge:
            edges_by_bundle[bundle_id].append(edge)

    for bundle_id in bundle_ids:
        if not edges_by_bundle[bundle_id]:
            raise RuntimeError(
                f"Category definition {bundle_id} cannot appear in any valid {logic_condition}/{difficulty} quartet."
            )

    try:
        import numpy as np
        from scipy import sparse
        from scipy.optimize import Bounds, LinearConstraint, milp

        bundle_index = {bundle_id: index for index, bundle_id in enumerate(bundle_ids)}
        row_indices: List[int] = []
        col_indices: List[int] = []
        values: List[float] = []
        for edge_index, edge in enumerate(edges):
            for bundle_id in edge:
                row_indices.append(bundle_index[bundle_id])
                col_indices.append(edge_index)
                values.append(1.0)
        incidence = sparse.coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(bundle_ids), len(edges)),
        ).tocsr()
        objective_rng = np.random.default_rng(rng.randrange(2**31))
        objective = objective_rng.random(len(edges)) * 1e-6
        target = np.full(len(bundle_ids), float(target_usage))
        result = milp(
            c=objective,
            integrality=np.ones(len(edges)),
            bounds=Bounds(np.zeros(len(edges)), np.full(len(edges), float(target_usage))),
            constraints=LinearConstraint(incidence, target, target),
            options={"time_limit": 180, "mip_rel_gap": 0.0},
        )
        if result.success and result.x is not None:
            edge_counts = np.rint(result.x).astype(int)
            usage = incidence @ edge_counts
            if (
                int(edge_counts.sum()) == int(num_rows)
                and int(usage.min()) == int(target_usage)
                and int(usage.max()) == int(target_usage)
            ):
                scheduled_edges = [
                    edge
                    for edge, count in zip(edges, edge_counts)
                    for _ in range(int(count))
                ]
                rng.shuffle(scheduled_edges)
                _EXACT_CATEGORY_BALANCE_SCHEDULE_CACHE[cache_key] = list(scheduled_edges)
                return scheduled_edges
    except Exception:
        pass

    for restart in range(max_restarts):
        remaining = {bundle_id: target_usage for bundle_id in bundle_ids}
        scheduled_edges: List[tuple[str, str, str, str]] = []
        local_rng = random.Random(rng.randrange(2**31) + restart)
        for _ in range(num_rows):
            anchor = max(
                bundle_ids,
                key=lambda bundle_id: (
                    remaining[bundle_id],
                    -len(edges_by_bundle[bundle_id]),
                    local_rng.random(),
                ),
            )
            if remaining[anchor] <= 0:
                break
            anchor_edges = edges_by_bundle[anchor]
            edge_window = (
                local_rng.sample(anchor_edges, 5000)
                if len(anchor_edges) > 5000
                else list(anchor_edges)
            )
            candidates = [
                edge
                for edge in edge_window
                if all(remaining[bundle_id] > 0 for bundle_id in edge)
            ]
            if not candidates and len(anchor_edges) > len(edge_window):
                candidates = [
                    edge
                    for edge in anchor_edges
                    if all(remaining[bundle_id] > 0 for bundle_id in edge)
                ]
            if not candidates:
                break
            choice_window = heapq.nlargest(
                min(300, len(candidates)),
                candidates,
                key=lambda edge: (
                    sum(remaining[bundle_id] for bundle_id in edge),
                    min(remaining[bundle_id] for bundle_id in edge),
                    local_rng.random(),
                ),
            )
            chosen = local_rng.choice(choice_window)
            scheduled_edges.append(chosen)
            for bundle_id in chosen:
                remaining[bundle_id] -= 1
        if len(scheduled_edges) == num_rows and all(remaining[bundle_id] == 0 for bundle_id in bundle_ids):
            _EXACT_CATEGORY_BALANCE_SCHEDULE_CACHE[cache_key] = list(scheduled_edges)
            return scheduled_edges

    raise RuntimeError(
        f"Could not find exact {target_usage}-per-category quartet schedule for "
        f"{logic_condition}/{difficulty} after {max_restarts} restarts."
    )


def _select_exact_balance_feasible_source_intersection_bundles(
    predicates: Sequence[Dict[str, Any]],
    *,
    difficulty: str,
    target_count: int,
    num_rows: int = 600,
) -> List[Dict[str, Any]]:
    candidates = _source_intersection_candidate_bundles(predicates, difficulty=difficulty)
    if len(candidates) < target_count:
        raise ValueError(
            f"Need at least {target_count} sourced {difficulty} intersection candidates, found {len(candidates)}."
        )

    family_caps = [10, 12, 14, 16, 18, 20, target_count]
    for family_cap in family_caps:
        selected: List[Dict[str, Any]] = []
        used_words: Counter[str] = Counter()
        family_usage: Counter[str] = Counter()
        remaining = list(candidates)
        while len(selected) < target_count and remaining:
            ranked = sorted(
                remaining,
                key=lambda bundle: (
                    sum(used_words[word] for word in set(_flatten_words(bundle))),
                    family_usage[_primary_category_family(bundle)],
                    str(bundle.get("bundle_id") or ""),
                ),
            )
            bundle = ranked[0]
            remaining.remove(bundle)
            family = _primary_category_family(bundle)
            if family_usage[family] >= family_cap and len(remaining) > target_count - len(selected):
                continue
            selected.append(bundle)
            used_words.update(set(_flatten_words(bundle)))
            family_usage[family] += 1

        if len(selected) != target_count:
            continue
        try:
            _find_exact_category_balance_quartet_schedule(
                "both",
                difficulty,
                selected,
                num_rows=num_rows,
                rng=random.Random(910_037 + family_cap),
                max_restarts=80,
            )
        except RuntimeError:
            continue
        return sorted(selected, key=lambda bundle: str(bundle.get("bundle_id") or ""))

    raise RuntimeError(
        f"Could not select {target_count} sourced {difficulty} intersection bundles "
        f"with exact {num_rows}-row category-balance feasibility."
    )


def build_bundle_bank_from_source_norms(source_norms_root: str | Path | None = None) -> List[Dict[str, Any]]:
    predicates = _load_source_norm_predicates(source_norms_root)
    experimental_predicates = [
        predicate
        for predicate in predicates
        if _source_norm_predicate_is_experimental_clean(predicate)
    ]
    domain_predicates = [
        predicate
        for predicate in experimental_predicates
        if _source_norm_predicate_has_semantic_domain(predicate)
    ]
    selected_predicates = _select_source_predicates_for_atomic_pool(domain_predicates)
    intersection_predicates = _select_source_predicates_for_atomic_pool(
        [
            predicate
            for predicate in predicates
            if _source_norm_predicate_is_allowed_for_intersection_pool(predicate)
            and _source_norm_predicate_is_clean_for_intersection(predicate)
        ],
        target_count=max(SOURCE_NORM_MAX_ATOMIC_PREDICATES, 240),
    )
    atomic_bundles = [_source_norm_bundle_from_predicate(predicate) for predicate in selected_predicates]
    shared_d1_bundles = _select_easy_atomic_bundles(atomic_bundles)
    either_d2_bundles = _select_composite_bundles(
        difficulty="d2",
        atomic_pool=atomic_bundles,
        target_count=TARGET_BUNDLES_PER_DESIGN_CELL[("either", "d2")],
    )
    either_d3_bundles = _select_composite_bundles(
        difficulty="d3",
        atomic_pool=atomic_bundles,
        target_count=TARGET_BUNDLES_PER_DESIGN_CELL[("either", "d3")],
    )
    both_d2_bundles = _select_exact_balance_feasible_source_intersection_bundles(
        intersection_predicates,
        difficulty="d2",
        target_count=TARGET_BUNDLES_PER_DESIGN_CELL[("both", "d2")],
    )
    both_d3_bundles = _select_exact_balance_feasible_source_intersection_bundles(
        intersection_predicates,
        difficulty="d3",
        target_count=TARGET_BUNDLES_PER_DESIGN_CELL[("both", "d3")],
    )
    return sorted(
        shared_d1_bundles + either_d2_bundles + either_d3_bundles + both_d2_bundles + both_d3_bundles,
        key=lambda bundle: (
            str(bundle.get("logic_condition") or ""),
            str(bundle.get("difficulty") or ""),
            str(bundle.get("bundle_id") or ""),
        ),
    )


def build_bundle_bank_from_source_seeds(
    source_seed_path: str | Path | None = None,
    existing_bundle_bank_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    del source_seed_path, existing_bundle_bank_path
    return build_bundle_bank_from_source_norms(DEFAULT_SOURCE_NORMS_ROOT)


def build_bundle_bank_from_taxonomy_source_seeds(
    source_seed_path: str | Path | None = None,
    existing_bundle_bank_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    source_categories = load_source_seed_categories(source_seed_path)
    group_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in source_categories:
        group_rows.setdefault(_seed_group_key(row), []).append(row)

    auto_bundles: List[Dict[str, Any]] = []

    for group_key, rows in sorted(group_rows.items()):
        deduped_rows = _dedupe_seed_rows(rows)
        if group_key in META_GROUP_CONFIG:
            allowed = set(META_GROUP_CONFIG[group_key]["allowed_knowledge_types"])
            deduped_rows = [row for row in deduped_rows if row.get("knowledge_type") in allowed]
            chunks = _chunked(deduped_rows, 3)
            for bundle_index, chunk in enumerate(chunks, start=1):
                auto_bundles.append(_build_auto_bundle(group_key, chunk, bundle_index))
        else:
            chunks = _exact_group_chunks(deduped_rows)
            for bundle_index, chunk in enumerate(chunks, start=1):
                auto_bundles.append(_build_auto_bundle(group_key, chunk, bundle_index))

    existing_path = Path(existing_bundle_bank_path) if existing_bundle_bank_path else DEFAULT_BUNDLE_BANK_V1
    manual_bundles: List[Dict[str, Any]] = []
    if existing_path.exists():
        with existing_path.open("r") as handle:
            existing_bundles = json.load(handle)
        for bundle in existing_bundles:
            source = bundle.get("source") or {}
            if source.get("bundle_builder") in {
                AUTO_BUNDLE_BUILDER,
                "manual_curated_v1",
                "manual_semantic_v1",
                "manual_intersection_v1",
                "manual_three_way_intersection_v1",
            }:
                continue
            manual_bundles.append(_with_category_family_metadata(_normalize_bundle_word_fields(bundle)))

    manual_additions_path = DEFAULT_MANUAL_ADDITIONS_PATH
    if manual_additions_path.exists():
        with manual_additions_path.open("r") as handle:
            additions = json.load(handle)
        if not isinstance(additions, list):
            raise ValueError(f"Expected list of manual additions in {manual_additions_path}")
        manual_bundles.extend(
                _with_category_family_metadata(_normalize_bundle_word_fields(bundle))
                for bundle in additions
                if isinstance(bundle, dict)
            )

    manual_semantic_bundles = _manual_d1_semantic_bundles()
    combined_atomic = [
        bundle
        for bundle in (manual_semantic_bundles + manual_bundles + auto_bundles)
        if str(bundle.get("knowledge_type") or "") not in COMPOSITE_KNOWLEDGE_TYPES
    ]
    shared_d1_bundles = _select_easy_atomic_bundles(combined_atomic)
    or_atomic_pool = [
        bundle
        for bundle in combined_atomic
        if not _is_surface_form_bundle(bundle)
        and str(bundle.get("knowledge_type") or "") in {"semantic", "encyclopedic", "associative_relations", "taxonomic"}
    ]
    either_d2_bundles = _select_composite_bundles(
        difficulty="d2",
        atomic_pool=or_atomic_pool,
        target_count=TARGET_BUNDLES_PER_DESIGN_CELL[("either", "d2")],
    )
    either_d3_bundles = _select_composite_bundles(
        difficulty="d3",
        atomic_pool=or_atomic_pool,
        target_count=TARGET_BUNDLES_PER_DESIGN_CELL[("either", "d3")],
    )
    both_d2_bundles = _select_intersection_bundles(
        TARGET_BUNDLES_PER_DESIGN_CELL[("both", "d2")],
        difficulty="d2",
    )
    both_d3_bundles = _select_three_way_intersection_bundles(
        TARGET_BUNDLES_PER_DESIGN_CELL[("both", "d3")]
    )
    return sorted(
        shared_d1_bundles + either_d2_bundles + either_d3_bundles + both_d2_bundles + both_d3_bundles,
        key=lambda bundle: (
            str(bundle.get("logic_condition") or ""),
            str(bundle.get("difficulty") or ""),
            str(bundle.get("bundle_id") or ""),
        ),
    )


def write_bundle_bank(path: Path, bundles: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(
            [
                _with_category_family_metadata(_normalize_bundle_word_fields(bundle))
                for bundle in bundles
                if isinstance(bundle, dict)
            ],
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def count_red_herring_collisions(bundles: Sequence[Dict[str, Any]]) -> int:
    counts = Counter(
        tag
        for bundle in bundles
        for tag in bundle.get("red_herring_tags", [])
    )
    return sum(max(0, count - 1) for count in counts.values())


def _has_unique_words(bundles: Sequence[Dict[str, Any]]) -> bool:
    words = list(itertools.chain.from_iterable(_flatten_words(bundle) for bundle in bundles))
    return len(words) == len(set(words))


def _preferred_type_count(bundles: Sequence[Dict[str, Any]], preferred_types: set[str]) -> int:
    return sum(1 for bundle in bundles if bundle.get("knowledge_type") in preferred_types)


def _bundle_rule_compression_ratio(bundle: Dict[str, Any]) -> float:
    member_tokens = sum(len(_tokenize_surface(word)) for word in _flatten_words(bundle))
    gloss_tokens = len(_tokenize_surface(bundle.get("rule_gloss", "")))
    return float(member_tokens) / float(gloss_tokens) if gloss_tokens else float(member_tokens)


def _pairwise_tag_overlap_jaccard(bundles: Sequence[Dict[str, Any]]) -> float:
    pair_scores: List[float] = []
    for left, right in itertools.combinations(bundles, 2):
        pair_scores.append(
            _tag_jaccard(
                left.get("red_herring_tags", []),
                right.get("red_herring_tags", []),
            )
        )
    return _safe_mean(pair_scores)


def compute_episode_difficulty_features(bundles: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if len(bundles) != 4:
        raise ValueError(f"Expected 4 bundles per episode, found {len(bundles)}")

    red_herring_collisions = float(count_red_herring_collisions(bundles))
    tag_overlap_jaccard = _pairwise_tag_overlap_jaccard(bundles)
    hard_knowledge_fraction = (
        sum(1 for bundle in bundles if bundle.get("knowledge_type") in HARD_KNOWLEDGE_TYPES) / len(bundles)
    )
    composite_bundle_fraction = (
        sum(1 for bundle in bundles if bundle.get("knowledge_type") in COMPOSITE_KNOWLEDGE_TYPES) / len(bundles)
    )
    intersection_bundle_fraction = (
        sum(1 for bundle in bundles if bundle.get("knowledge_type") in {"two_way_intersection", "three_way_intersection"}) / len(bundles)
    )
    three_way_intersection_bundle_fraction = (
        sum(1 for bundle in bundles if bundle.get("knowledge_type") == "three_way_intersection") / len(bundles)
    )
    composite_part_count_mean = _safe_mean([_bundle_composite_num_parts(bundle) for bundle in bundles])
    category_arity_mean = _safe_mean([float(bundle.get("category_arity") or _bundle_composite_num_parts(bundle)) for bundle in bundles])
    operator_or_fraction = _safe_mean([1.0 if bundle.get("category_operator") == "or" else 0.0 for bundle in bundles])
    operator_and_fraction = _safe_mean([1.0 if bundle.get("category_operator") == "and" else 0.0 for bundle in bundles])
    component_type_difficulty_mean = _safe_mean(
        [
            _safe_mean(
                [
                    KNOWLEDGE_TYPE_DIFFICULTY.get(component_type, 1.0)
                    for component_type in _bundle_component_types(bundle)
                ]
            )
            for bundle in bundles
        ]
    )
    avg_knowledge_type_difficulty = _safe_mean(
        [
            KNOWLEDGE_TYPE_DIFFICULTY.get(str(bundle.get("knowledge_type")), 1.0)
            for bundle in bundles
        ]
    )
    avg_bundle_complexity = _safe_mean(
        [
            COMPLEXITY_SCORE.get(str(bundle.get("complexity")), 1.0)
            for bundle in bundles
        ]
    )
    rule_compression_ratio = _safe_mean(
        [_bundle_rule_compression_ratio(bundle) for bundle in bundles]
    )
    rule_lossiness_score = (
        rule_compression_ratio
        + component_type_difficulty_mean
        + (1.5 * composite_bundle_fraction)
        + (0.75 * max(0.0, composite_part_count_mean - 1.0))
        + (2.25 * intersection_bundle_fraction)
    )
    boundary_fuzziness_score = (
        red_herring_collisions
        + tag_overlap_jaccard
        + avg_knowledge_type_difficulty
        + avg_bundle_complexity
        + composite_bundle_fraction
        + (composite_part_count_mean - 1.0)
        + (1.5 * intersection_bundle_fraction)
    )
    examples_advantage_proxy = (
        boundary_fuzziness_score
        + rule_lossiness_score
        + hard_knowledge_fraction
        + composite_bundle_fraction
        + (0.5 * max(0.0, composite_part_count_mean - 1.0))
        + (1.5 * intersection_bundle_fraction)
    )
    return {
        "red_herring_collisions": red_herring_collisions,
        "tag_overlap_jaccard": tag_overlap_jaccard,
        "hard_knowledge_fraction": hard_knowledge_fraction,
        "composite_bundle_fraction": composite_bundle_fraction,
        "intersection_bundle_fraction": intersection_bundle_fraction,
        "three_way_intersection_bundle_fraction": three_way_intersection_bundle_fraction,
        "composite_part_count_mean": composite_part_count_mean,
        "category_arity_mean": category_arity_mean,
        "operator_or_fraction": operator_or_fraction,
        "operator_and_fraction": operator_and_fraction,
        "component_type_difficulty_mean": component_type_difficulty_mean,
        "avg_knowledge_type_difficulty": avg_knowledge_type_difficulty,
        "avg_bundle_complexity": avg_bundle_complexity,
        "rule_compression_ratio": rule_compression_ratio,
        "rule_lossiness_score": rule_lossiness_score,
        "boundary_fuzziness_score": boundary_fuzziness_score,
        "examples_advantage_proxy": examples_advantage_proxy,
    }


def _quartet_selection_score(features: Dict[str, float]) -> float:
    # Sample from low / middle / high score bands so realized episodes preserve the intended
    # difficulty ladder without forcing a bank that is dominated by one knowledge group.
    return (
        float(features["examples_advantage_proxy"])
        + (1.5 * float(features["rule_lossiness_score"]))
        + (0.5 * float(features["avg_bundle_complexity"]))
    )


def _selection_is_valid(logic_condition: str, difficulty: str, bundles: Sequence[Dict[str, Any]]) -> bool:
    if len(bundles) != 4:
        return False
    bundle_ids = tuple(sorted(str(bundle.get("bundle_id") or id(bundle)) for bundle in bundles))
    cache_key = (logic_condition, difficulty, bundle_ids)
    cached = _SELECTION_VALIDITY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    def remember(value: bool) -> bool:
        _SELECTION_VALIDITY_CACHE[cache_key] = bool(value)
        return bool(value)

    if not _has_unique_words(bundles):
        return remember(False)
    if len({str(bundle.get("rule_gloss") or "") for bundle in bundles}) != len(bundles):
        return remember(False)

    config = DESIGN_CELL_CONFIG[(logic_condition, difficulty)]
    if count_red_herring_collisions(bundles) < config["min_red_herring_collisions"]:
        return remember(False)

    if any(str(bundle.get("logic_condition") or "") != logic_condition for bundle in bundles):
        return remember(False)
    if any(str(bundle.get("difficulty") or "") != difficulty for bundle in bundles):
        return remember(False)

    preferred_count = _preferred_type_count(bundles, set(config["preferred_types"]))
    if preferred_count < int(config["min_preferred_types"]):
        return remember(False)

    required_composite_parts = config.get("required_composite_parts")
    if required_composite_parts is not None:
        if any(_bundle_composite_num_parts(bundle) != int(required_composite_parts) for bundle in bundles):
            return remember(False)

    if logic_condition == "both":
        component_count = int(config.get("required_composite_parts") or bundles[0].get("category_arity") or 1)
        validator = get_default_semantic_validator()
        episode_words = set(
            itertools.chain.from_iterable(_flatten_words(bundle) for bundle in bundles)
        )
        strict_failed_components = {
            component_index
            for bundle in bundles
            for component_index, _word in _strict_near_miss_records_for_bundle(bundle, validator)
        }
        if not set(range(component_count)) <= strict_failed_components:
            return remember(False)
        cross_fit_safe_failed_components: set[int] = set()
        for bundle in bundles:
            for component_index, word in _strict_near_miss_records_for_bundle(bundle, validator):
                if word in episode_words:
                    continue
                if any(
                    other is not bundle and _word_fits_bundle_semantically(word, other, validator)
                    for other in bundles
                ):
                    continue
                cross_fit_safe_failed_components.add(component_index)
                if set(range(component_count)) <= cross_fit_safe_failed_components:
                    break
        if not set(range(component_count)) <= cross_fit_safe_failed_components:
            return remember(False)

    return remember(True)


def _primary_category_family(bundle: Dict[str, Any]) -> str:
    families = bundle.get("category_families") or []
    if families:
        return str(families[0])
    return str(bundle.get("category_family") or "unknown")


def _quartet_max_family_count(bundles: Sequence[Dict[str, Any]]) -> int:
    counts = Counter(_primary_category_family(bundle) for bundle in bundles)
    return max(counts.values(), default=0)


def _valid_quartets_for_design_cell(
    logic_condition: str,
    difficulty: str,
    eligible_bundles: Sequence[Dict[str, Any]],
) -> List[tuple[str, str, str, str]]:
    cache_key = (
        logic_condition,
        difficulty,
        tuple(sorted(str(bundle["bundle_id"]) for bundle in eligible_bundles)),
    )
    if cache_key in _VALID_QUARTET_CACHE:
        return _VALID_QUARTET_CACHE[cache_key]

    valid_quartets: List[tuple[str, str, str, str]] = []
    for candidate in itertools.combinations(eligible_bundles, 4):
        if _selection_is_valid(logic_condition, difficulty, candidate):
            valid_quartets.append(tuple(str(bundle["bundle_id"]) for bundle in candidate))
    _VALID_QUARTET_CACHE[cache_key] = valid_quartets
    return valid_quartets


def _scored_valid_quartets_for_design_cell(
    logic_condition: str,
    difficulty: str,
    eligible_bundles: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    cache_key = (
        logic_condition,
        difficulty,
        tuple(sorted(str(bundle["bundle_id"]) for bundle in eligible_bundles)),
    )
    if cache_key in _SCORED_QUARTET_CACHE:
        return _SCORED_QUARTET_CACHE[cache_key]

    scored_quartets: List[Dict[str, Any]] = []
    for candidate in itertools.combinations(eligible_bundles, 4):
        if not _selection_is_valid(logic_condition, difficulty, candidate):
            continue
        bundle_ids = tuple(str(bundle["bundle_id"]) for bundle in candidate)
        features = compute_episode_difficulty_features(candidate)
        scored_quartets.append(
            {
                "bundle_ids": bundle_ids,
                "selection_score": _quartet_selection_score(features),
            }
        )
    scored_quartets.sort(
        key=lambda item: (
            float(item["selection_score"]),
            tuple(item["bundle_ids"]),
        )
    )
    _SCORED_QUARTET_CACHE[cache_key] = scored_quartets
    return scored_quartets


def _quartet_selection_pool(
    difficulty: str,
    scored_quartets: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not scored_quartets:
        return []
    cache_key = (difficulty, id(scored_quartets))
    if cache_key in _QUARTET_SELECTION_POOL_CACHE:
        return _QUARTET_SELECTION_POOL_CACHE[cache_key]

    low_quantile, high_quantile = QUARTET_SELECTION_QUANTILES.get(difficulty, (0.0, 1.0))
    total = len(scored_quartets)
    start_index = max(0, min(total - 1, int(total * low_quantile)))
    end_index = max(start_index + 1, min(total, int(total * high_quantile)))
    pool = list(scored_quartets[start_index:end_index])
    pool_keys = {tuple(item["bundle_ids"]) for item in pool}

    # The quantile band controls difficulty, but every bundle that participates
    # in at least one valid quartet should remain reachable for split-level
    # balancing.
    covered_bundle_ids = {bundle_id for item in pool for bundle_id in item["bundle_ids"]}
    all_bundle_ids = {bundle_id for item in scored_quartets for bundle_id in item["bundle_ids"]}
    for missing_bundle_id in sorted(all_bundle_ids - covered_bundle_ids):
        candidates = [
            item
            for item in scored_quartets
            if missing_bundle_id in item["bundle_ids"]
        ]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda item: (float(item["selection_score"]), tuple(item["bundle_ids"])),
        )
        key = tuple(best["bundle_ids"])
        if key not in pool_keys:
            pool.append(best)
            pool_keys.add(key)
    _QUARTET_SELECTION_POOL_CACHE[cache_key] = pool
    return pool


def _choose_quartet_from_pool(
    selection_pool: Sequence[Dict[str, Any]],
    rng: random.Random,
    bundle_usage: Counter[str] | None = None,
    bundle_usage_cap: int | None = None,
    bundle_usage_hard_cap: int | None = None,
) -> tuple[str, str, str, str]:
    if not selection_pool:
        raise RuntimeError("Cannot choose an episode quartet from an empty selection pool.")
    if bundle_usage is None:
        return tuple(rng.choice(selection_pool)["bundle_ids"])

    index_key = id(selection_pool)
    bundle_to_items = _SELECTION_POOL_INDEX_CACHE.get(index_key)
    if bundle_to_items is None:
        bundle_to_items = {}
        for item in selection_pool:
            for bundle_id in item["bundle_ids"]:
                bundle_to_items.setdefault(str(bundle_id), []).append(item)
        for items in bundle_to_items.values():
            items.sort(
                key=lambda item: (
                    float(item["selection_score"]),
                    tuple(item["bundle_ids"]),
                )
            )
        _SELECTION_POOL_INDEX_CACHE[index_key] = bundle_to_items
    bundle_degrees = _SELECTION_POOL_DEGREE_CACHE.get(index_key)
    if bundle_degrees is None:
        bundle_degrees = {bundle_id: len(items) for bundle_id, items in bundle_to_items.items()}
        _SELECTION_POOL_DEGREE_CACHE[index_key] = bundle_degrees
    resolved_hard_cap = int(bundle_usage_hard_cap) if bundle_usage_hard_cap is not None else None

    reachable_bundle_ids = set(bundle_to_items)
    minimum_usage = min(bundle_usage[bundle_id] for bundle_id in reachable_bundle_ids)

    if bundle_usage_cap is not None:
        for target_bundle_id in sorted(
            reachable_bundle_ids,
            key=lambda bundle_id: (
                bundle_usage[bundle_id],
                bundle_degrees.get(bundle_id, 0),
                bundle_id,
            ),
        ):
            feasible_items = []
            for item in bundle_to_items[target_bundle_id]:
                ids = tuple(str(bundle_id) for bundle_id in item["bundle_ids"])
                if resolved_hard_cap is not None and any(
                    bundle_usage[bundle_id] + 1 > resolved_hard_cap
                    for bundle_id in ids
                ):
                    continue
                feasible_items.append(item)
            if not feasible_items:
                continue
            def cap_candidate_key(item: Dict[str, Any]) -> tuple[Any, ...]:
                return (
                    -sum(
                        max(0, int(bundle_usage_cap) - bundle_usage[str(bundle_id)])
                        for bundle_id in item["bundle_ids"]
                    ),
                    max(bundle_usage[str(bundle_id)] + 1 for bundle_id in item["bundle_ids"]),
                    sum(
                        max(0, bundle_usage[str(bundle_id)] + 1 - int(bundle_usage_cap))
                        for bundle_id in item["bundle_ids"]
                    ),
                    sum(bundle_usage[str(bundle_id)] + 1 for bundle_id in item["bundle_ids"]),
                    sum(math.log1p(bundle_degrees.get(str(bundle_id), 0)) for bundle_id in item["bundle_ids"]),
                    float(item["selection_score"]),
                    tuple(item["bundle_ids"]),
                )

            ranked_feasible = sorted(feasible_items, key=cap_candidate_key)
            choice_window = ranked_feasible[: min(12, len(ranked_feasible))]
            chosen = rng.choice(choice_window)
            chosen_ids = tuple(str(bundle_id) for bundle_id in chosen["bundle_ids"])
            bundle_usage.update(chosen_ids)
            return chosen_ids
        raise RuntimeError(
            "Cannot choose a quartet without exceeding the configured bundle usage hard cap."
        )

    underused_candidates: List[Dict[str, Any]] = []
    for target_window in (8, 16, 32, 64, len(reachable_bundle_ids)):
        target_bundle_ids = sorted(
            reachable_bundle_ids,
            key=lambda bundle_id: (
                bundle_usage[bundle_id],
                bundle_degrees.get(bundle_id, 0),
                rng.random(),
                bundle_id,
            ),
        )[: min(target_window, len(reachable_bundle_ids))]
        candidate_by_key: Dict[tuple[str, ...], Dict[str, Any]] = {}
        per_target_limit = 5000 if bundle_usage_cap is not None else 200
        for target_bundle_id in target_bundle_ids:
            items = bundle_to_items[target_bundle_id]
            if len(items) > per_target_limit:
                start = rng.randrange(max(1, len(items) - per_target_limit + 1))
                items = items[start:start + per_target_limit]
            for item in items[:per_target_limit]:
                ids = tuple(str(bundle_id) for bundle_id in item["bundle_ids"])
                if resolved_hard_cap is not None and any(
                    bundle_usage[bundle_id] + 1 > resolved_hard_cap
                    for bundle_id in ids
                ):
                    continue
                candidate_by_key[ids] = item
        underused_candidates = list(candidate_by_key.values())
        if underused_candidates:
            break

    if not underused_candidates:
        if bundle_usage_cap is not None:
            raise RuntimeError(
                "Cannot choose a quartet without exceeding the configured bundle usage hard cap."
            )
        target_bundle_ids = [
            bundle_id
            for bundle_id in reachable_bundle_ids
            if bundle_usage[bundle_id] == minimum_usage
        ]
        target_bundle_id = rng.choice(target_bundle_ids)
        underused_candidates = list(bundle_to_items[target_bundle_id])
        rng.shuffle(underused_candidates)

    chosen = min(
        underused_candidates,
        key=lambda item: (
            -sum(
                max(0, int(bundle_usage_cap) - bundle_usage[str(bundle_id)])
                for bundle_id in item["bundle_ids"]
            ) if bundle_usage_cap is not None else 0,
            max(bundle_usage[str(bundle_id)] + 1 for bundle_id in item["bundle_ids"]),
            sum(
                max(0, bundle_usage[str(bundle_id)] + 1 - int(bundle_usage_cap))
                for bundle_id in item["bundle_ids"]
            ) if bundle_usage_cap is not None else 0,
            sum(bundle_usage[str(bundle_id)] + 1 for bundle_id in item["bundle_ids"]),
            -sum(
                1
                for bundle_id in item["bundle_ids"]
                if bundle_usage[str(bundle_id)] == minimum_usage
            ),
            sum(math.log1p(bundle_degrees.get(str(bundle_id), 0)) for bundle_id in item["bundle_ids"]),
            float(item["selection_score"]),
            tuple(item["bundle_ids"]),
        ),
    )
    chosen_ids = tuple(str(bundle_id) for bundle_id in chosen["bundle_ids"])
    bundle_usage.update(chosen_ids)
    return chosen_ids


def _choose_unconstrained_balanced_quartet(
    bundle_ids: Sequence[str],
    rng: random.Random,
    bundle_usage: Counter[str],
) -> tuple[str, str, str, str]:
    ranked = [
        (bundle_usage[str(bundle_id)], rng.random(), str(bundle_id))
        for bundle_id in bundle_ids
    ]
    ranked.sort()
    chosen_ids = tuple(bundle_id for _, _, bundle_id in ranked[:4])
    bundle_usage.update(chosen_ids)
    return chosen_ids


def _sample_valid_quartet_without_exhaustive_scoring(
    logic_condition: str,
    difficulty: str,
    eligible_bundles: Sequence[Dict[str, Any]],
    rng: random.Random,
    bundle_usage: Counter[str] | None = None,
    bundle_usage_cap: int | None = None,
    bundle_usage_hard_cap: int | None = None,
    max_attempts: int = 4000,
) -> tuple[str, str, str, str] | None:
    bundle_by_id = {str(bundle["bundle_id"]): bundle for bundle in eligible_bundles}
    bundle_ids = sorted(bundle_by_id)
    if len(bundle_ids) < 4:
        return None

    if bundle_usage is None:
        for _ in range(max_attempts):
            chosen_ids = tuple(rng.sample(bundle_ids, 4))
            if _selection_is_valid(
                logic_condition,
                difficulty,
                [bundle_by_id[bundle_id] for bundle_id in chosen_ids],
            ):
                return chosen_ids
        return None

    if bundle_usage_cap is not None:
        scored_quartets = _scored_valid_quartets_for_design_cell(
            logic_condition,
            difficulty,
            eligible_bundles,
        )
        if scored_quartets:
            return _choose_quartet_from_pool(
                scored_quartets,
                rng,
                bundle_usage=bundle_usage,
                bundle_usage_cap=bundle_usage_cap,
                bundle_usage_hard_cap=bundle_usage_hard_cap,
            )

    window_sizes = list(dict.fromkeys([12, 18, 24, 32, 48]))
    for window_size in window_sizes:
        candidate_ids = sorted(
            bundle_ids,
            key=lambda bundle_id: (bundle_usage[bundle_id], rng.random(), bundle_id),
        )[:window_size]
        if len(candidate_ids) < 4:
            continue
        combinations = list(itertools.combinations(candidate_ids, 4))
        if len(combinations) > 750:
            combinations = rng.sample(combinations, 750)
        valid_candidates: List[tuple[str, str, str, str]] = []
        for chosen_ids in combinations:
            if bundle_usage_hard_cap is not None and any(
                bundle_usage[bundle_id] + 1 > int(bundle_usage_hard_cap)
                for bundle_id in chosen_ids
            ):
                continue
            if not _selection_is_valid(
                logic_condition,
                difficulty,
                [bundle_by_id[bundle_id] for bundle_id in chosen_ids],
            ):
                continue
            valid_candidates.append(tuple(chosen_ids))
        if valid_candidates:
            chosen_ids = min(
                valid_candidates,
                key=lambda chosen: (
                    _quartet_max_family_count([bundle_by_id[bundle_id] for bundle_id in chosen]),
                    max(bundle_usage[bundle_id] for bundle_id in chosen),
                    sum(bundle_usage[bundle_id] for bundle_id in chosen),
                    rng.random(),
                    chosen,
                ),
            )
            bundle_usage.update(chosen_ids)
            return tuple(chosen_ids)

    target_bundle_ids = sorted(
        bundle_ids,
        key=lambda bundle_id: (bundle_usage[bundle_id], rng.random(), bundle_id),
    )[: min(16, len(bundle_ids))]
    targeted_candidates: List[tuple[str, str, str, str]] = []
    for target_bundle_id in target_bundle_ids:
        other_ids = [bundle_id for bundle_id in bundle_ids if bundle_id != target_bundle_id]
        for _ in range(max_attempts // max(1, len(target_bundle_ids))):
            chosen_ids = tuple(sorted([target_bundle_id, *rng.sample(other_ids, 3)]))
            if bundle_usage_hard_cap is not None and any(
                bundle_usage[bundle_id] + 1 > int(bundle_usage_hard_cap)
                for bundle_id in chosen_ids
            ):
                continue
            if not _selection_is_valid(
                logic_condition,
                difficulty,
                [bundle_by_id[bundle_id] for bundle_id in chosen_ids],
            ):
                continue
            targeted_candidates.append(chosen_ids)
    if targeted_candidates:
        chosen_ids = min(
            targeted_candidates,
            key=lambda chosen: (
                max(bundle_usage[bundle_id] for bundle_id in chosen),
                sum(bundle_usage[bundle_id] for bundle_id in chosen),
                _quartet_max_family_count([bundle_by_id[bundle_id] for bundle_id in chosen]),
                rng.random(),
                chosen,
            ),
        )
        bundle_usage.update(chosen_ids)
        return tuple(chosen_ids)

    for _ in range(max_attempts):
        chosen_ids = tuple(rng.sample(bundle_ids, 4))
        if bundle_usage_hard_cap is not None and any(
            bundle_usage[bundle_id] + 1 > int(bundle_usage_hard_cap)
            for bundle_id in chosen_ids
        ):
            continue
        if _selection_is_valid(
            logic_condition,
            difficulty,
            [bundle_by_id[bundle_id] for bundle_id in chosen_ids],
        ):
            bundle_usage.update(chosen_ids)
            return tuple(chosen_ids)
    return None


def _out_of_category_candidate_score(
    candidate_bundle: Dict[str, Any],
    assigned_bundles: Sequence[Dict[str, Any]],
) -> tuple[float, float, str]:
    tag_overlap = max(
        (
            _tag_jaccard(
                candidate_bundle.get("red_herring_tags", []),
                bundle.get("red_herring_tags", []),
            )
            for bundle in assigned_bundles
        ),
        default=0.0,
    )
    gloss_overlap = max(
        (
            _gloss_token_jaccard(
                candidate_bundle.get("rule_gloss", ""),
                bundle.get("rule_gloss", ""),
            )
            for bundle in assigned_bundles
        ),
        default=0.0,
    )
    type_match = max(
        (
            1.0 if candidate_bundle.get("knowledge_type") == bundle.get("knowledge_type") else 0.0
            for bundle in assigned_bundles
        ),
        default=0.0,
    )
    return (
        tag_overlap + gloss_overlap + (0.25 * type_match),
        gloss_overlap,
        str(candidate_bundle.get("bundle_id") or ""),
    )


def _bundle_component_index_map(bundle: Dict[str, Any]) -> Dict[str, List[int]]:
    bundle = _normalize_bundle_word_fields(bundle)
    source = bundle.get("source") or {}
    raw_map = source.get("component_indices_by_word")
    if isinstance(raw_map, dict) and raw_map:
        return {
            _normalize_word_surface(word): [int(index) for index in indices]
            for word, indices in raw_map.items()
            if isinstance(indices, list)
        }

    arity = int(bundle.get("category_arity") or _bundle_composite_num_parts(bundle) or 1)
    operator = str(bundle.get("category_operator") or (source.get("composite_operator") or ""))
    if operator == "and":
        default_indices = list(range(max(1, arity)))
    else:
        default_indices = [0]
    return {
        word: list(default_indices)
        for word in _flatten_words(bundle)
    }


def _component_indices_for_word(bundle: Dict[str, Any], word: str) -> List[int]:
    component_map = _bundle_component_index_map(bundle)
    normalized_word = _normalize_word_surface(word)
    indices = component_map.get(normalized_word)
    if indices:
        return [int(index) for index in indices]
    arity = int(bundle.get("category_arity") or _bundle_composite_num_parts(bundle) or 1)
    if str(bundle.get("category_operator") or "") == "and":
        return list(range(max(1, arity)))
    return [0]


def _bundle_component_glosses(bundle: Dict[str, Any]) -> List[str]:
    source = bundle.get("source") or {}
    glosses = source.get("component_rule_glosses")
    if isinstance(glosses, list) and glosses:
        return [str(gloss) for gloss in glosses]
    return [str(bundle.get("rule_gloss") or "")]


def _semantic_component_profile_for_bundle(
    word: Any,
    bundle: Dict[str, Any],
    validator: SemanticValidator | None = None,
) -> List[bool]:
    validator = validator or get_default_semantic_validator()
    source = bundle.get("source") or {}
    normalized_word = _normalize_word_surface(word)
    component_glosses = _bundle_component_glosses(bundle)
    cache_key = (
        str(bundle.get("bundle_id") or source.get("seed_category_name") or id(bundle)),
        normalized_word,
        tuple(component_glosses),
    )
    cached = _SEMANTIC_BUNDLE_WORD_PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    component_words_by_index = source.get("component_words_by_index")
    if not isinstance(component_words_by_index, list):
        component_words_by_index = None
    profile = validator.component_profile(
        normalized_word,
        component_glosses,
        component_words_by_index=component_words_by_index,
    )
    _SEMANTIC_BUNDLE_WORD_PROFILE_CACHE[cache_key] = list(profile)
    return profile


def _semantic_component_evidence_profile_for_bundle(
    word: Any,
    bundle: Dict[str, Any],
    validator: SemanticValidator | None = None,
) -> List[Dict[str, Any]]:
    validator = validator or get_default_semantic_validator()
    source = bundle.get("source") or {}
    normalized_word = _normalize_word_surface(word)
    component_glosses = _bundle_component_glosses(bundle)
    cache_key = (
        str(bundle.get("bundle_id") or source.get("seed_category_name") or id(bundle)),
        normalized_word,
        tuple(component_glosses),
    )
    cached = _SEMANTIC_BUNDLE_WORD_EVIDENCE_CACHE.get(cache_key)
    if cached is not None:
        return [dict(entry) for entry in cached]
    component_words_by_index = source.get("component_words_by_index")
    if not isinstance(component_words_by_index, list):
        component_words_by_index = None
    evidence = validator.component_evidence_profile(
        normalized_word,
        component_glosses,
        component_words_by_index=component_words_by_index,
    )
    _SEMANTIC_BUNDLE_WORD_EVIDENCE_CACHE[cache_key] = [dict(entry) for entry in evidence]
    return evidence


def _semantic_satisfied_component_indices(
    word: Any,
    bundle: Dict[str, Any],
    validator: SemanticValidator | None = None,
) -> List[int]:
    profile = _semantic_component_profile_for_bundle(word, bundle, validator)
    return [index for index, fits in enumerate(profile) if fits]


def _semantic_failed_component_indices_with_evidence(
    word: Any,
    bundle: Dict[str, Any],
    validator: SemanticValidator | None = None,
) -> List[int]:
    evidence = _semantic_component_evidence_profile_for_bundle(word, bundle, validator)
    return [index for index, entry in enumerate(evidence) if entry.get("value") is False]


def _semantic_unknown_component_indices(
    word: Any,
    bundle: Dict[str, Any],
    validator: SemanticValidator | None = None,
) -> List[int]:
    evidence = _semantic_component_evidence_profile_for_bundle(word, bundle, validator)
    return [index for index, entry in enumerate(evidence) if entry.get("value") is None]


def _word_fits_bundle_semantically(
    word: Any,
    bundle: Dict[str, Any],
    validator: SemanticValidator | None = None,
) -> bool:
    profile = _semantic_component_profile_for_bundle(word, bundle, validator)
    if not profile:
        return False
    operator = str(bundle.get("category_operator") or (bundle.get("source") or {}).get("category_operator") or "")
    if operator == "and":
        return all(profile)
    return any(profile)


def _word_is_strict_semantic_negative_for_bundle(
    word: Any,
    bundle: Dict[str, Any],
    validator: SemanticValidator | None = None,
    *,
    mode: str = "any_failure",
) -> bool:
    evidence = _semantic_component_evidence_profile_for_bundle(word, bundle, validator)
    if not evidence:
        return False
    values = [entry.get("value") for entry in evidence]
    operator = str(bundle.get("category_operator") or (bundle.get("source") or {}).get("category_operator") or "")
    if mode in {"all_components_false", "none_of_components"}:
        return bool(values) and all(value is False for value in values)
    if mode == "component_near_miss":
        if operator != "and":
            return False
        component_glosses = _bundle_component_glosses(bundle)
        return (
            values.count(True) == len(values) - 1
            and values.count(False) == 1
            and values.count(None) == 0
            and not _strict_near_miss_semantic_disqualification_reasons(
                word,
                component_glosses,
                evidence,
                validator,
            )
        )
    if operator == "and":
        return any(value is False for value in values)
    return all(value is False for value in values)


def _strict_near_miss_records_for_bundle(
    bundle: Dict[str, Any],
    validator: SemanticValidator | None = None,
) -> List[tuple[int, str]]:
    bundle_id = str(bundle.get("bundle_id") or id(bundle))
    cached = _BUNDLE_STRICT_NEAR_MISS_RECORD_CACHE.get(bundle_id)
    if cached is not None:
        return list(cached)
    validator = validator or get_default_semantic_validator()
    records: List[tuple[int, str]] = []
    source = bundle.get("source") or {}
    near_misses = source.get("near_miss_words_by_component")
    if isinstance(near_misses, list):
        for component_index, words in enumerate(near_misses):
            for raw_word in words:
                word = _normalize_word_surface(raw_word)
                if not word:
                    continue
                if _word_is_strict_semantic_negative_for_bundle(
                    word,
                    bundle,
                    validator,
                    mode="component_near_miss",
                ):
                    records.append((int(component_index), word))
    _BUNDLE_STRICT_NEAR_MISS_RECORD_CACHE[bundle_id] = list(records)
    return records


def _source_norm_gloss_is_high_risk_near_miss_failure(gloss: Any) -> bool:
    text = _source_norm_clean_gloss(gloss)
    return any(fragment in text for fragment in SOURCE_NORM_NEAR_MISS_HIGH_RISK_FRAGMENTS)


def _source_norm_gloss_is_rated_property_near_miss_failure(gloss: Any) -> bool:
    return _source_norm_clean_gloss(gloss) in SOURCE_NORM_STRICT_NEAR_MISS_RATED_FAILURE_GLOSSES


def _source_norm_gloss_is_material_component(gloss: Any) -> bool:
    return _source_norm_clean_gloss(gloss) in SOURCE_NORM_MATERIAL_COMPONENT_GLOSSES


def _material_false_evidence_disqualification_reasons(
    word: Any,
    gloss: Any,
    evidence_entry: Dict[str, Any],
    validator: SemanticValidator | None = None,
) -> List[str]:
    if evidence_entry.get("value") is not False:
        return []
    clean_gloss = _source_norm_clean_gloss(gloss)
    if clean_gloss not in SOURCE_NORM_MATERIAL_COMPONENT_GLOSSES:
        return []
    basis = str(evidence_entry.get("basis") or "")
    if basis in {"override_negative"}:
        return []
    validator = validator or get_default_semantic_validator()
    normalized_word = _normalize_word_surface(word)
    if basis == "conservative_domain_exclusion" and validator._has_material_variable_artifact_evidence(normalized_word):
        return [f"material_failure_artifact_like_domain_exclusion:{clean_gloss}"]
    if normalized_word in {"crown", "button", "punch", "bracelet"} and basis != "override_negative":
        return [f"material_failure_known_ambiguous_word:{clean_gloss}"]
    return []


def _source_norm_gloss_is_transportation_function(gloss: Any) -> bool:
    return _source_norm_clean_gloss(gloss) in SOURCE_NORM_TRANSPORTATION_FUNCTION_GLOSSES


def _functional_false_evidence_disqualification_reasons(
    word: Any,
    gloss: Any,
    evidence_entry: Dict[str, Any],
    validator: SemanticValidator | None = None,
) -> List[str]:
    if evidence_entry.get("value") is not False:
        return []
    clean_gloss = _source_norm_clean_gloss(gloss)
    if clean_gloss not in SOURCE_NORM_TRANSPORTATION_FUNCTION_GLOSSES:
        return []
    basis = str(evidence_entry.get("basis") or "")
    if basis in {"override_negative"}:
        return []
    validator = validator or get_default_semantic_validator()
    normalized_word = _normalize_word_surface(word)
    if basis in {"conservative_mutual_exclusion", "conservative_domain_exclusion"} and validator._is_transportation_part_word(normalized_word):
        return [f"transportation_failure_vehicle_part_{basis}:{clean_gloss}"]
    return []


def _evidence_has_independent_negative_for_high_risk_failure(
    evidence_entry: Dict[str, Any],
    gloss: Any,
) -> bool:
    if evidence_entry.get("value") is not False:
        return False
    if _source_norm_gloss_is_material_component(gloss):
        return str(evidence_entry.get("basis") or "") != "unknown"
    basis = str(evidence_entry.get("basis") or "")
    if _source_norm_gloss_is_rated_property_near_miss_failure(gloss):
        return basis == "override_negative"
    if not _source_norm_gloss_is_high_risk_near_miss_failure(gloss):
        return basis != "unknown"
    return basis in {
        "override_negative",
        "conservative_domain_exclusion",
        "conservative_mutual_exclusion",
        "conservative_taxonomy_property_exclusion",
    }


def _strict_near_miss_semantic_disqualification_reasons(
    word: Any,
    component_glosses: Sequence[Any],
    evidence: Sequence[Dict[str, Any]],
    validator: SemanticValidator | None = None,
) -> List[str]:
    validator = validator or get_default_semantic_validator()
    normalized_word = _normalize_word_surface(word)
    values = [entry.get("value") for entry in evidence]
    reasons: List[str] = []
    if not values:
        return ["empty_evidence_profile"]
    if values.count(True) != len(values) - 1 or values.count(False) != 1 or values.count(None) != 0:
        reasons.append("not_structural_one_component_boundary")

    word_concepts = validator._word_concepts(normalized_word)
    word_domains = validator._word_domains(normalized_word)
    animal_concepts = {"bird", "fish", "mammal", "insect", "reptile", "feathers", "fur", "scales"}
    artifact_concepts = {
        "bodypart",
        "clothing",
        "furniture",
        "musicalinstrument",
        "tool",
        "utensil",
        "vehicle",
        "weapon",
    }
    if {"living", "nonliving"} <= word_concepts:
        reasons.append("cross_sense_living_and_nonliving")
    if {"movable", "stationary"} <= word_concepts:
        reasons.append("cross_sense_movable_and_stationary")
    if word_concepts & animal_concepts and word_concepts & artifact_concepts:
        reasons.append("cross_sense_animal_and_artifact_concepts")

    for index, entry in enumerate(evidence):
        if entry.get("plausibility_issues"):
            reasons.extend(
                f"rated_property_plausibility_conflict:{index}:{issue}"
                for issue in entry.get("plausibility_issues", [])
            )
        if entry.get("value") is True and index < len(component_glosses):
            clean_gloss = _source_norm_clean_gloss(component_glosses[index])
            issues = validator.rated_property_plausibility_issues(
                normalized_word,
                clean_gloss,
                True,
                basis=str(entry.get("basis") or ""),
            )
            reasons.extend(
                f"rated_property_positive_plausibility_conflict:{index}:{issue}"
                for issue in issues
            )
        if entry.get("value") is not False:
            continue
        gloss = component_glosses[index] if index < len(component_glosses) else ""
        clean_gloss = _source_norm_clean_gloss(gloss)
        basis = str(entry.get("basis") or "")
        material_reasons = _material_false_evidence_disqualification_reasons(
            normalized_word,
            clean_gloss,
            entry,
            validator,
        )
        reasons.extend(
            f"material_failed_component_plausibility_conflict:{index}:{reason}"
            for reason in material_reasons
        )
        functional_reasons = _functional_false_evidence_disqualification_reasons(
            normalized_word,
            clean_gloss,
            entry,
            validator,
        )
        reasons.extend(
            f"functional_failed_component_plausibility_conflict:{index}:{reason}"
            for reason in functional_reasons
        )
        if _source_norm_gloss_is_rated_property_near_miss_failure(clean_gloss) and basis != "override_negative":
            reasons.append(f"rated_property_failed_without_override:{index}:{clean_gloss}")
        if basis == "conservative_taxonomy_property_exclusion":
            if "4 legs" in clean_gloss and word_concepts & {"mammal"}:
                reasons.append(f"compatible_mammal_four_legs_failure:{index}")
            if ("fur" in clean_gloss or "furry" in clean_gloss) and word_concepts & {"mammal"}:
                reasons.append(f"compatible_mammal_fur_failure:{index}")
        if basis == "conservative_mutual_exclusion":
            if "movable" in clean_gloss and (
                word_domains & {"animal", "artifact"}
                or word_concepts & (animal_concepts | artifact_concepts)
            ):
                reasons.append(f"compatible_movable_failure:{index}")
            if "nonliving" in clean_gloss and (
                word_domains & {"animal"}
                or word_concepts & (animal_concepts | {"living"})
            ):
                reasons.append(f"compatible_nonliving_failure:{index}")
    return sorted(set(reasons))


def _semantic_group_metadata(
    words: Sequence[Any],
    bundle: Dict[str, Any],
    validator: SemanticValidator | None = None,
) -> Dict[str, Any]:
    component_count = max(1, int(bundle.get("category_arity") or _bundle_composite_num_parts(bundle) or 1))
    word = _normalize_word_surface(words[0] if words else "")
    satisfied = _semantic_satisfied_component_indices(word, bundle, validator)
    failed = _semantic_failed_component_indices_with_evidence(word, bundle, validator)
    unknown = _semantic_unknown_component_indices(word, bundle, validator)
    evidence = _semantic_component_evidence_profile_for_bundle(word, bundle, validator)
    return {
        "satisfied_component_indices": satisfied,
        "failed_component_indices": failed,
        "unknown_component_indices": unknown,
        "semantic_component_evidence": evidence[:component_count],
        "semantic_full_member_for_display_label": _word_fits_bundle_semantically(word, bundle, validator),
        "semantic_strict_negative_for_display_label": _word_is_strict_semantic_negative_for_bundle(word, bundle, validator),
        "semantic_component_near_miss_for_display_label": _word_is_strict_semantic_negative_for_bundle(
            word,
            bundle,
            validator,
            mode="component_near_miss",
        ),
    }


def _word_fits_assigned_label(
    word: Any,
    label: str,
    episode: Dict[str, Any],
    validator: SemanticValidator | None = None,
) -> bool:
    bundle = (episode.get("_bundle_metadata_by_label") or {}).get(label)
    if not bundle:
        return False
    return _word_fits_bundle_semantically(word, bundle, validator)


def _ordered_singleton_support_words(
    bundle: Dict[str, Any],
    query_word: str,
) -> List[str]:
    words = [
        word
        for word in _flatten_words(bundle)
        if word != query_word
    ]
    seen: set[str] = set()
    unique_words = [
        word
        for word in words
        if not (word in seen or seen.add(word))
    ]
    arity = max(1, int(bundle.get("category_arity") or _bundle_composite_num_parts(bundle) or 1))
    component_map = _bundle_component_index_map(bundle)

    ordered: List[str] = []
    used: set[str] = set()
    for component_index in range(arity):
        for word in unique_words:
            if word in used:
                continue
            if component_index in component_map.get(word, []):
                ordered.append(word)
                used.add(word)
                break
    ordered.extend(word for word in unique_words if word not in used)
    return ordered


def _out_of_category_word_candidates(bundle_bank: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cache_key = id(bundle_bank)
    cached = _OUT_OF_CATEGORY_BLOCK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    candidates: List[Dict[str, Any]] = []
    for bundle in bundle_bank:
        bundle = _normalize_bundle_word_fields(bundle)
        bundle_id = str(bundle.get("bundle_id") or "")
        if not bundle_id:
            continue
        component_map = _bundle_component_index_map(bundle)
        for block_index, block in enumerate(bundle.get("blocks") or []):
            for word_index, raw_word in enumerate(block):
                word = _normalize_word_surface(raw_word)
                if not word:
                    continue
                candidates.append(
                    {
                        "bundle_id": bundle_id,
                        "block_index": block_index,
                        "word_index": word_index,
                        "words": [word],
                        "component_indices": list(component_map.get(word, [])),
                        "bundle": bundle,
                    }
                )
    _OUT_OF_CATEGORY_BLOCK_CACHE[cache_key] = candidates
    return candidates


def _general_out_of_category_candidates_for_bundle(
    target_bundle: Dict[str, Any],
    bundle_bank: Sequence[Dict[str, Any]],
    validator: SemanticValidator,
) -> List[Dict[str, Any]]:
    cache_key = (id(bundle_bank), str(target_bundle.get("bundle_id") or ""))
    cached = _GENERAL_OUT_OF_CATEGORY_CANDIDATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    operator = str(target_bundle.get("category_operator") or (target_bundle.get("source") or {}).get("category_operator") or "")
    if operator == "and":
        candidates = [
            candidate
            for candidate in _out_of_category_word_candidates(bundle_bank)
            if not _semantic_satisfied_component_indices(candidate["words"][0], target_bundle, validator)
            and _word_is_strict_semantic_negative_for_bundle(
                candidate["words"][0],
                target_bundle,
                validator,
                mode="any_failure",
            )
        ]
    else:
        candidates = [
            candidate
            for candidate in _out_of_category_word_candidates(bundle_bank)
            if _word_is_strict_semantic_negative_for_bundle(
                candidate["words"][0],
                target_bundle,
                validator,
                mode="all_components_false",
            )
        ]
    _GENERAL_OUT_OF_CATEGORY_CANDIDATE_CACHE[cache_key] = candidates
    return candidates


def _build_out_of_category_groups_by_label(
    labels: Sequence[str],
    assigned_bundles_by_label: Dict[str, Dict[str, Any]],
    bundle_bank: Sequence[Dict[str, Any]],
    rng: random.Random,
) -> tuple[Dict[str, List[List[str]]], Dict[str, List[List[int]]], Dict[str, List[List[int]]]]:
    validator = get_default_semantic_validator()
    assigned_bundles = list(assigned_bundles_by_label.values())
    assigned_bundle_ids = {str(bundle.get("bundle_id") or "") for bundle in assigned_bundles}
    episode_words = set(
        itertools.chain.from_iterable(_flatten_words(bundle) for bundle in assigned_bundles)
    )
    used_words: set[str] = set()
    groups_by_label: Dict[str, List[List[str]]] = {}
    component_indices_by_label: Dict[str, List[List[int]]] = {}
    failed_component_indices_by_label: Dict[str, List[List[int]]] = {}

    def add_group(
        groups: List[List[str]],
        component_groups: List[List[int]],
        failed_groups: List[List[int]],
        word: str,
        component_indices: Sequence[int],
        failed_component_indices: Sequence[int],
        *,
        allow_reuse: bool = False,
        allow_episode_word: bool = False,
    ) -> bool:
        normalized_word = _normalize_word_surface(word)
        if not normalized_word:
            return False
        if normalized_word in episode_words and not allow_episode_word:
            return False
        if normalized_word in used_words and not allow_reuse:
            return False
        groups.append([normalized_word])
        component_groups.append([int(index) for index in component_indices])
        failed_groups.append([int(index) for index in failed_component_indices])
        used_words.add(normalized_word)
        return True

    def add_near_misses(
        label: str,
        groups: List[List[str]],
        component_groups: List[List[int]],
        failed_groups: List[List[int]],
    ) -> None:
        bundle = assigned_bundles_by_label[label]
        if str(bundle.get("category_operator") or "") != "and":
            return
        source = bundle.get("source") or {}
        near_misses = source.get("near_miss_words_by_component")
        if not isinstance(near_misses, list):
            return
        component_count = max(1, int(bundle.get("category_arity") or len(near_misses) or 1))
        candidates_by_failed_component: Dict[int, List[tuple[str, List[int]]]] = {
            component_index: []
            for component_index in range(component_count)
        }
        seen_near_miss_words: set[str] = set()
        for words in near_misses:
            for raw_word in words:
                word = _normalize_word_surface(raw_word)
                if not word or word in seen_near_miss_words:
                    continue
                seen_near_miss_words.add(word)
                if not _word_is_strict_semantic_negative_for_bundle(
                    word,
                    bundle,
                    validator,
                    mode="component_near_miss",
                ):
                    continue
                indices = _semantic_satisfied_component_indices(word, bundle, validator)
                failed_indices = _semantic_failed_component_indices_with_evidence(word, bundle, validator)
                unknown_indices = _semantic_unknown_component_indices(word, bundle, validator)
                if (
                    len(indices) != component_count - 1
                    or len(failed_indices) != 1
                    or unknown_indices
                ):
                    continue
                for failed_component in failed_indices:
                    candidates_by_failed_component[failed_component].append((word, indices))

        for failed_component in range(component_count):
            added = False
            for allow_reuse, allow_episode_word in ((False, False), (True, False)):
                for word, indices in candidates_by_failed_component.get(failed_component, []):
                    if add_group(
                        groups,
                        component_groups,
                        failed_groups,
                        word,
                        indices,
                        [failed_component],
                        allow_reuse=allow_reuse,
                        allow_episode_word=allow_episode_word,
                    ):
                        added = True
                        break
                if added:
                    break

    for label in labels:
        groups: List[List[str]] = []
        component_groups: List[List[int]] = []
        failed_groups: List[List[int]] = []
        add_near_misses(str(label), groups, component_groups, failed_groups)
        target_bundle = assigned_bundles_by_label[str(label)]
        component_count = max(1, int(target_bundle.get("category_arity") or _bundle_composite_num_parts(target_bundle) or 1))
        target_candidates = list(
            _general_out_of_category_candidates_for_bundle(target_bundle, bundle_bank, validator)
        )
        rng.shuffle(target_candidates)
        for allow_reuse in (False, True):
            for maximum_score in (0.75, float("inf")):
                for candidate in target_candidates:
                    if len(groups) >= OUT_OF_CATEGORY_GROUPS_PER_LABEL:
                        break
                    if candidate["bundle_id"] in assigned_bundle_ids:
                        continue
                    words = list(candidate["words"])
                    word = _normalize_word_surface(words[0] if words else "")
                    word_set = set(words)
                    if word_set & episode_words:
                        continue
                    if word_set & used_words and not allow_reuse:
                        continue
                    score = _out_of_category_candidate_score(candidate["bundle"], assigned_bundles)[0]
                    if score > maximum_score:
                        continue
                    failed_indices = _semantic_failed_component_indices_with_evidence(
                        word,
                        target_bundle,
                        validator,
                    )
                    if not failed_indices:
                        continue
                    if add_group(
                        groups,
                        component_groups,
                        failed_groups,
                        word,
                        [],
                        failed_indices,
                        allow_reuse=allow_reuse,
                    ):
                        continue
                if len(groups) >= OUT_OF_CATEGORY_GROUPS_PER_LABEL:
                    break
            if len(groups) >= OUT_OF_CATEGORY_GROUPS_PER_LABEL:
                break
        if not groups:
            raise ValueError(
                f"Could not assign any validated out-of-category groups for {label} "
                f"({target_bundle.get('bundle_id')}: {target_bundle.get('rule_gloss')})."
            )
        groups_by_label[str(label)] = groups
        component_indices_by_label[str(label)] = component_groups
        failed_component_indices_by_label[str(label)] = failed_groups

    if assigned_bundles and str(assigned_bundles[0].get("category_operator") or "") == "and":
        component_count = max(
            1,
            int(assigned_bundles[0].get("category_arity") or _bundle_composite_num_parts(assigned_bundles[0]) or 1),
        )
        strict_failed_components = {
            failed_indices[0]
            for label in labels
            for component_indices, failed_indices in zip(
                component_indices_by_label.get(str(label), []),
                failed_component_indices_by_label.get(str(label), []),
            )
            if len(component_indices) == component_count - 1 and len(failed_indices) == 1
        }
        expected_components = set(range(component_count))
        if not expected_components <= strict_failed_components:
            missing = sorted(expected_components - strict_failed_components)
            raise ValueError(
                "Could not assign strict AND component near-miss groups covering "
                f"all failed components; missing {missing}."
            )

    return groups_by_label, component_indices_by_label, failed_component_indices_by_label


def sample_episode_core(
    difficulty: str,
    bundle_bank: Sequence[Dict[str, Any]],
    episode_index: int,
    split: str,
    rng: random.Random,
    bundle_usage: Counter[str] | None = None,
    logic_condition: str | None = None,
    bundle_usage_cap: int | None = None,
    bundle_usage_hard_cap: int | None = None,
    forced_bundle_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    design_cell = resolve_design_cell(difficulty, logic_condition)
    level = str(design_cell["difficulty"])
    group = str(design_cell["logic_condition"])
    eligible = [
        dict(bundle)
        for bundle in bundle_bank
        if bundle.get("difficulty") == level
        and bundle.get("logic_condition") == group
        and bool(bundle.get("examples_favored"))
    ]
    if len(eligible) < 4:
        raise ValueError(f"Need at least 4 eligible bundles for {group}/{level}, found {len(eligible)}.")

    selected: List[Dict[str, Any]] = []
    bundle_by_id = {str(bundle["bundle_id"]): bundle for bundle in eligible}
    if forced_bundle_ids is not None:
        chosen_ids = tuple(str(bundle_id) for bundle_id in forced_bundle_ids)
        if len(chosen_ids) != 4 or len(set(chosen_ids)) != 4:
            raise ValueError("forced_bundle_ids must contain exactly four distinct bundle ids.")
        missing_ids = [bundle_id for bundle_id in chosen_ids if bundle_id not in bundle_by_id]
        if missing_ids:
            raise ValueError(f"forced_bundle_ids contains ineligible bundle ids: {missing_ids}")
        selected = [dict(bundle_by_id[bundle_id]) for bundle_id in chosen_ids]
        if not _selection_is_valid(group, level, selected):
            raise ValueError(f"Forced {group}/{level} bundle quartet does not satisfy generation constraints.")
        if bundle_usage is not None:
            bundle_usage.update(chosen_ids)
    elif len(eligible) == 4:
        if not _selection_is_valid(group, level, eligible):
            raise ValueError(f"The available {group}/{level} bundle pool does not satisfy generation constraints.")
        selected = list(eligible)
    else:
        chosen_ids = None
        if len(eligible) > 120:
            chosen_ids = _sample_valid_quartet_without_exhaustive_scoring(
                logic_condition=group,
                difficulty=level,
                eligible_bundles=eligible,
                rng=rng,
                bundle_usage=bundle_usage,
                bundle_usage_cap=bundle_usage_cap,
                bundle_usage_hard_cap=bundle_usage_hard_cap,
            )
        elif 4 < len(eligible) <= 36:
            chosen_ids = _sample_valid_quartet_without_exhaustive_scoring(
                logic_condition=group,
                difficulty=level,
                eligible_bundles=eligible,
                rng=rng,
                bundle_usage=bundle_usage,
                bundle_usage_cap=bundle_usage_cap,
                bundle_usage_hard_cap=bundle_usage_hard_cap,
            )
        if chosen_ids is not None:
            selected = [dict(bundle_by_id[bundle_id]) for bundle_id in chosen_ids]
        else:
            scored_quartets = _scored_valid_quartets_for_design_cell(group, level, eligible)
            all_quartets_are_valid = len(scored_quartets) == math.comb(len(eligible), 4)
            if bundle_usage is not None and all_quartets_are_valid:
                chosen_ids = _choose_unconstrained_balanced_quartet(
                    bundle_ids=sorted(bundle_by_id),
                    rng=rng,
                    bundle_usage=bundle_usage,
                )
                selected = [dict(bundle_by_id[bundle_id]) for bundle_id in chosen_ids]
            else:
                selection_pool = _quartet_selection_pool(level, scored_quartets)
                if group == "both" and level == "d3":
                    selection_pool = scored_quartets
                if selection_pool:
                    chosen_ids = _choose_quartet_from_pool(
                        selection_pool,
                        rng,
                        bundle_usage=bundle_usage,
                        bundle_usage_cap=bundle_usage_cap,
                        bundle_usage_hard_cap=bundle_usage_hard_cap,
                    )
                    selected = [dict(bundle_by_id[bundle_id]) for bundle_id in chosen_ids]
        if not selected:
            raise RuntimeError(f"Could not sample a valid bundle quartet for {group}/{level}.")

    labels = rng.sample(LABEL_INVENTORY, 4)
    assigned_bundles = rng.sample(selected, len(selected))

    bundle_ids: Dict[str, str] = {}
    knowledge_types: Dict[str, str] = {}
    category_family_by_label: Dict[str, str] = {}
    category_families_by_label: Dict[str, List[str]] = {}
    rule_glosses: Dict[str, str] = {}
    support_groups_by_label: Dict[str, List[List[str]]] = {}
    support_component_indices_by_label: Dict[str, List[List[int]]] = {}
    gold_groups: Dict[str, List[str]] = {}
    gold_component_indices_by_label: Dict[str, List[int]] = {}
    bundle_metadata_by_label: Dict[str, Dict[str, Any]] = {}
    assigned_bundles_by_label: Dict[str, Dict[str, Any]] = {}

    for label, bundle in zip(labels, assigned_bundles):
        bundle_words = _flatten_words(bundle)
        if len(bundle_words) < 2:
            raise ValueError(f"Bundle {bundle.get('bundle_id')} does not have enough words for a singleton episode.")
        query_word = _normalize_word_surface(rng.choice(bundle_words))
        support_words = _ordered_singleton_support_words(bundle, query_word)
        support_groups = [[word] for word in support_words]
        bundle_ids[label] = str(bundle["bundle_id"])
        knowledge_types[label] = str(bundle["knowledge_type"])
        bundle_families = list(bundle.get("category_families") or _category_families_for_bundle(bundle))
        category_family_by_label[label] = str(bundle.get("category_family") or bundle_families[0])
        category_families_by_label[label] = [str(family) for family in bundle_families]
        rule_glosses[label] = str(bundle["rule_gloss"])
        support_groups_by_label[label] = support_groups
        support_component_indices_by_label[label] = [
            _component_indices_for_word(bundle, word)
            for word in support_words
        ]
        gold_groups[label] = [query_word]
        gold_component_indices_by_label[label] = _component_indices_for_word(bundle, query_word)
        assigned_bundles_by_label[label] = bundle
        bundle_metadata_by_label[label] = {
            "bundle_id": bundle["bundle_id"],
            "knowledge_type": bundle["knowledge_type"],
            "complexity": bundle["complexity"],
            "composite_num_parts": _bundle_composite_num_parts(bundle),
            "logic_condition": bundle.get("logic_condition"),
            "category_operator": bundle.get("category_operator"),
            "category_arity": int(bundle.get("category_arity") or _bundle_composite_num_parts(bundle)),
            "condition_count": int(bundle.get("condition_count") or _bundle_composite_num_parts(bundle)),
            "shared_d1": bool(bundle.get("shared_d1")),
            "component_knowledge_types": list(_bundle_component_types(bundle)),
            "category_family": category_family_by_label[label],
            "category_families": list(category_families_by_label[label]),
            "rule_gloss": bundle["rule_gloss"],
            "red_herring_tags": list(bundle.get("red_herring_tags", [])),
            "source": dict(bundle.get("source", {})),
        }

    difficulty_features = compute_episode_difficulty_features(assigned_bundles)
    query_words = [
        _normalize_word_surface(word)
        for label in labels
        for word in gold_groups[label]
    ]
    rng.shuffle(query_words)
    (
        out_of_category_groups_by_label,
        out_of_category_component_indices_by_label,
        out_of_category_failed_component_indices_by_label,
    ) = _build_out_of_category_groups_by_label(
        labels=labels,
        assigned_bundles_by_label=assigned_bundles_by_label,
        bundle_bank=bundle_bank,
        rng=rng,
    )

    return {
        "episode_id": f"lexical_category_inference_episode_{group}_{level}_{episode_index:06d}",
        "task": "lexical_category_inference",
        "format": "yes_no",
        "split": split,
        "logic_condition": group,
        "category_operator": design_cell["category_operator"],
        "category_arity": int(design_cell["category_arity"]),
        "condition_count": int(design_cell["condition_count"]),
        "shared_d1": bool(design_cell["shared_d1"]),
        "design_cell": design_cell_key(group, level),
        "difficulty": level,
        "difficulty_name": design_cell["difficulty_name"],
        "labels": labels,
        "bundle_ids": bundle_ids,
        "knowledge_types": knowledge_types,
        "category_family_by_label": category_family_by_label,
        "category_families_by_label": category_families_by_label,
        "rule_glosses": rule_glosses,
        "support_groups_by_label": support_groups_by_label,
        "support_component_indices_by_label": support_component_indices_by_label,
        "gold_groups": gold_groups,
        "gold_component_indices_by_label": gold_component_indices_by_label,
        "out_of_category_groups_by_label": out_of_category_groups_by_label,
        "out_of_category_component_indices_by_label": out_of_category_component_indices_by_label,
        "out_of_category_failed_component_indices_by_label": out_of_category_failed_component_indices_by_label,
        "query_words": query_words,
        "difficulty_features": difficulty_features,
        "_bundle_metadata_by_label": bundle_metadata_by_label,
    }


def sample_episode(
    difficulty: str,
    bundle_bank: Sequence[Dict[str, Any]],
    episode_index: int,
    split: str,
    rng: random.Random,
    id_offset: int = 0,
    logic_condition: str | None = None,
) -> Dict[str, Any]:
    del id_offset
    return sample_episode_core(difficulty, bundle_bank, episode_index, split, rng, logic_condition=logic_condition)


def compute_candidate_confusability_score(
    episode: Dict[str, Any],
    label_pairs: Sequence[tuple[str, str]],
) -> float:
    metadata = episode["_bundle_metadata_by_label"]
    scores: List[float] = []
    for source_label, target_label in label_pairs:
        source = metadata[source_label]
        target = metadata[target_label]
        source_operator = str((source.get("source") or {}).get("composite_operator") or "")
        target_operator = str((target.get("source") or {}).get("composite_operator") or "")
        intersection_operator_bonus = 0.5 if source_operator == target_operator == "both" else 0.0
        scores.append(
            _tag_jaccard(source.get("red_herring_tags", []), target.get("red_herring_tags", []))
            + (0.5 * int(source.get("knowledge_type") == target.get("knowledge_type")))
            + (0.25 * int(source.get("composite_num_parts") == target.get("composite_num_parts")))
            + (0.5 * int(source.get("complexity") == target.get("complexity")))
            + _gloss_token_jaccard(source.get("rule_gloss", ""), target.get("rule_gloss", ""))
            + intersection_operator_bonus
        )
    return _safe_mean(scores)


def _format_candidate_groups(
    episode: Dict[str, Any],
    display_label_by_gold: Dict[str, str],
    rng: random.Random,
    out_of_category_groups_by_display_label: Dict[str, Sequence[str]] | None = None,
) -> List[Dict[str, Any]]:
    out_of_category_groups_by_display_label = out_of_category_groups_by_display_label or {}
    validator = get_default_semantic_validator()
    metadata = episode.get("_bundle_metadata_by_label") or {}
    candidate_groups: List[Dict[str, Any]] = []
    for gold_label in episode["labels"]:
        display_label = display_label_by_gold[gold_label]
        words = _normalize_word_list(
            out_of_category_groups_by_display_label.get(
                display_label,
                episode["gold_groups"][gold_label],
            )
        )
        group = {
            "gold_label": OUT_OF_CATEGORY_LABEL if display_label in out_of_category_groups_by_display_label else gold_label,
            "display_label": display_label,
            "words": words,
        }
        display_bundle = metadata.get(display_label)
        if display_bundle and words:
            group.update(_semantic_group_metadata(words, display_bundle, validator))
        candidate_groups.append(group)
    rng.shuffle(candidate_groups)
    return candidate_groups


def _select_scored_candidates(
    candidates: Sequence[tuple[Any, ...]],
    mode: str,
    score_index: int = -1,
) -> List[tuple[Any, ...]]:
    if not candidates:
        return []
    unique_scores = sorted({float(candidate[score_index]) for candidate in candidates})
    if mode == "lowest":
        target_score = unique_scores[0]
    elif mode == "highest":
        target_score = unique_scores[-1]
    elif mode == "high_not_max":
        target_score = unique_scores[-2] if len(unique_scores) >= 2 else unique_scores[-1]
    else:
        raise ValueError(f"Unsupported score selection mode: {mode}")
    return [candidate for candidate in candidates if float(candidate[score_index]) == target_score]


def _best_replacement_pair(
    episode: Dict[str, Any],
    mode: str,
    rng: random.Random,
) -> tuple[str, str, float]:
    pairs: List[tuple[str, str, float]] = []
    for gold_label in episode["labels"]:
        for other_label in episode["labels"]:
            if gold_label == other_label:
                continue
            score = compute_candidate_confusability_score(episode, [(gold_label, other_label)])
            pairs.append((gold_label, other_label, score))

    chosen = _select_scored_candidates(pairs, mode=mode)
    return rng.choice(chosen)


def _display_mapping_is_valid_negative(
    episode: Dict[str, Any],
    display_label_by_gold: Dict[str, str],
    validator: SemanticValidator | None = None,
) -> bool:
    validator = validator or get_default_semantic_validator()
    metadata = episode.get("_bundle_metadata_by_label") or {}
    for gold_label, display_label in display_label_by_gold.items():
        if gold_label == display_label:
            continue
        display_bundle = metadata.get(display_label)
        if not display_bundle:
            return False
        for word in (episode.get("gold_groups") or {}).get(gold_label, []):
            if _word_fits_bundle_semantically(word, display_bundle, validator):
                return False
    return True


def _highest_confusable_swap_pair(
    episode: Dict[str, Any],
    mode: str,
    rng: random.Random,
) -> tuple[str, str, float]:
    candidates: List[tuple[str, str, float]] = []
    for left, right in itertools.combinations(episode["labels"], 2):
        display_label_by_gold = {label: label for label in episode["labels"]}
        display_label_by_gold[left] = right
        display_label_by_gold[right] = left
        if not _display_mapping_is_valid_negative(episode, display_label_by_gold):
            continue
        score = compute_candidate_confusability_score(episode, [(left, right), (right, left)])
        candidates.append((left, right, score))
    if not candidates:
        raise ValueError(f"Episode {episode.get('episode_id')} has no semantically valid two-label swap.")
    chosen = _select_scored_candidates(candidates, mode=mode)
    return rng.choice(chosen)


def _is_single_cycle(ordered_labels: Sequence[str], mapping: Dict[str, str]) -> bool:
    start = ordered_labels[0]
    seen = set()
    current = start
    for _ in ordered_labels:
        if current in seen:
            return False
        seen.add(current)
        current = mapping[current]
    return current == start and len(seen) == len(ordered_labels)


def _best_cycle_mapping(
    episode: Dict[str, Any],
    cycle_size: int,
    rng: random.Random,
) -> tuple[Dict[str, str], float]:
    candidate_mappings: List[tuple[Dict[str, str], float]] = []
    for labels_subset in itertools.combinations(episode["labels"], cycle_size):
        ordered_subset = list(labels_subset)
        for perm in itertools.permutations(ordered_subset):
            mapping = {source: target for source, target in zip(ordered_subset, perm)}
            if any(source == target for source, target in mapping.items()):
                continue
            if not _is_single_cycle(ordered_subset, mapping):
                continue
            display_label_by_gold = {label: label for label in episode["labels"]}
            display_label_by_gold.update(mapping)
            if not _display_mapping_is_valid_negative(episode, display_label_by_gold):
                continue
            score = compute_candidate_confusability_score(
                episode,
                [(source, target) for source, target in mapping.items()],
            )
            candidate_mappings.append((mapping, score))

    if not candidate_mappings:
        raise ValueError(f"Episode {episode.get('episode_id')} has no semantically valid {cycle_size}-cycle.")
    best_score = max(score for _, score in candidate_mappings)
    best = [(mapping, score) for mapping, score in candidate_mappings if score == best_score]
    return rng.choice(best)


def build_positive_candidate(episode: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    display_label_by_gold = {label: label for label in episode["labels"]}
    return {
        "candidate_groups": _format_candidate_groups(episode, display_label_by_gold, rng),
        "answer": "Yes",
        "corruption_type": "identity",
        "num_wrong_labels": 0,
        "uses_all_labels_once": True,
        "corrupted_gold_labels": [],
        "candidate_confusability_score": 0.0,
    }


def build_two_label_swap_negative(
    episode: Dict[str, Any],
    rng: random.Random,
    mode: str = "highest",
) -> Dict[str, Any]:
    left, right, score = _highest_confusable_swap_pair(episode, mode=mode, rng=rng)
    display_label_by_gold = {label: label for label in episode["labels"]}
    display_label_by_gold[left] = right
    display_label_by_gold[right] = left
    return {
        "candidate_groups": _format_candidate_groups(episode, display_label_by_gold, rng),
        "answer": "No",
        "corruption_type": "two_label_swap",
        "num_wrong_labels": 2,
        "uses_all_labels_once": True,
        "corrupted_gold_labels": sorted([left, right]),
        "candidate_confusability_score": score,
    }


def build_three_cycle_negative(episode: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    mapping, score = _best_cycle_mapping(episode, 3, rng)
    display_label_by_gold = {label: label for label in episode["labels"]}
    display_label_by_gold.update(mapping)
    return {
        "candidate_groups": _format_candidate_groups(episode, display_label_by_gold, rng),
        "answer": "No",
        "corruption_type": "three_cycle",
        "num_wrong_labels": 3,
        "uses_all_labels_once": True,
        "corrupted_gold_labels": sorted(mapping.keys()),
        "candidate_confusability_score": score,
    }


def build_four_cycle_negative(episode: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    mapping, score = _best_cycle_mapping(episode, 4, rng)
    display_label_by_gold = {label: label for label in episode["labels"]}
    display_label_by_gold.update(mapping)
    return {
        "candidate_groups": _format_candidate_groups(episode, display_label_by_gold, rng),
        "answer": "No",
        "corruption_type": "four_cycle",
        "num_wrong_labels": 4,
        "uses_all_labels_once": True,
        "corrupted_gold_labels": sorted(mapping.keys()),
        "candidate_confusability_score": score,
    }


def _out_of_category_group_records(
    episode: Dict[str, Any],
    label: str,
) -> List[Dict[str, Any]]:
    groups = (episode.get("out_of_category_groups_by_label") or {}).get(label) or []
    component_groups = (episode.get("out_of_category_component_indices_by_label") or {}).get(label) or []
    failed_component_groups = (episode.get("out_of_category_failed_component_indices_by_label") or {}).get(label) or []
    records: List[Dict[str, Any]] = []
    component_count = max(1, int(episode.get("category_arity") or episode.get("condition_count") or 1))
    for index, group in enumerate(groups):
        satisfied = [
            int(component_index)
            for component_index in (component_groups[index] if index < len(component_groups) else [])
        ]
        failed = [
            int(component_index)
            for component_index in (
                failed_component_groups[index]
                if index < len(failed_component_groups)
                else [component_index for component_index in range(component_count) if component_index not in set(satisfied)]
            )
        ]
        records.append(
            {
                "words": _normalize_word_list(group),
                "satisfied_component_indices": satisfied,
                "failed_component_indices": failed,
            }
        )
    return records


def build_out_of_category_negative(
    episode: Dict[str, Any],
    rng: random.Random,
    *,
    corruption_type: str = "out_of_category",
    semantic_mode: str | None = None,
) -> Dict[str, Any]:
    labels = list(episode["labels"])
    component_count = max(1, int(episode.get("category_arity") or episode.get("condition_count") or 1))
    semantic_mode = semantic_mode or (
        "component_near_miss"
        if corruption_type == "component_near_miss"
        else "none_of_components_out_of_category"
    )

    label_records: Dict[str, List[Dict[str, Any]]] = {}
    for label in labels:
        records = _out_of_category_group_records(episode, str(label))
        if semantic_mode == "component_near_miss":
            records = [
                record for record in records
                if len(record["satisfied_component_indices"]) == component_count - 1
                and len(record["failed_component_indices"]) == 1
            ]
        elif semantic_mode in {"general_out_of_category", "none_of_components_out_of_category"}:
            records = [
                record for record in records
                if not record["satisfied_component_indices"]
                and record["failed_component_indices"]
            ]
        label_records[str(label)] = records

    eligible_labels = [label for label, records in label_records.items() if records]
    if not eligible_labels:
        raise ValueError(
            f"Episode {episode.get('episode_id')} has no {semantic_mode} groups."
        )
    target_label = str(rng.choice(eligible_labels))
    replacement_record = rng.choice(label_records[target_label])
    replacement_group = _normalize_word_list(replacement_record["words"])
    satisfied = list(replacement_record["satisfied_component_indices"])
    failed = list(replacement_record["failed_component_indices"])
    display_label_by_gold = {label: label for label in labels}
    return {
        "candidate_groups": _format_candidate_groups(
            episode,
            display_label_by_gold,
            rng,
            out_of_category_groups_by_display_label={target_label: replacement_group},
        ),
        "answer": "No",
        "corruption_type": corruption_type,
        "negative_semantic_subtype": (
            f"and_fails_component_{failed[0]}"
            if semantic_mode == "component_near_miss" and len(failed) == 1
            else (
                "and_fails_multiple_components"
                if semantic_mode == "component_near_miss"
                else "or_fails_all_components"
                if semantic_mode == "none_of_components_out_of_category"
                else "and_no_verified_components_with_explicit_failure"
            )
        ),
        "target_label": target_label,
        "satisfied_component_indices": satisfied,
        "failed_component_indices": failed,
        "num_wrong_labels": 1,
        "uses_all_labels_once": True,
        "corrupted_gold_labels": [target_label],
        "candidate_confusability_score": 0.0,
        "num_out_of_category_items": 1,
        "semantic_validation_flags": {
            "corrupted_word_fails_displayed_label": True,
            "semantic_mode": semantic_mode,
        },
    }


def build_yes_no_row_from_episode(
    episode: Dict[str, Any],
    row_index: int,
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "id": f"lexical_category_inference_{episode['logic_condition']}_{episode['difficulty']}_{row_index:06d}",
        "task": "lexical_category_inference",
        "format": "yes_no",
        "split": episode["split"],
        "logic_condition": episode["logic_condition"],
        "category_operator": episode["category_operator"],
        "category_arity": int(episode["category_arity"]),
        "condition_count": int(episode["condition_count"]),
        "shared_d1": bool(episode["shared_d1"]),
        "design_cell": episode["design_cell"],
        "difficulty": episode["difficulty"],
        "difficulty_name": episode["difficulty_name"],
        "episode_id": episode["episode_id"],
        "labels": list(episode["labels"]),
        "bundle_ids": dict(episode["bundle_ids"]),
        "knowledge_types": dict(episode["knowledge_types"]),
        "category_family_by_label": dict(episode["category_family_by_label"]),
        "category_families_by_label": {
            label: list(families)
            for label, families in episode["category_families_by_label"].items()
        },
        "composite_num_parts_by_label": {
            label: int((episode["_bundle_metadata_by_label"][label]).get("composite_num_parts", 1))
            for label in episode["labels"]
        },
        "rule_glosses": dict(episode["rule_glosses"]),
        "support_groups_by_label": {
            label: [_normalize_word_list(group) for group in groups]
            for label, groups in episode["support_groups_by_label"].items()
        },
        "support_component_indices_by_label": {
            label: [[int(index) for index in indices] for indices in groups]
            for label, groups in (episode.get("support_component_indices_by_label") or {}).items()
        },
        "gold_groups": {
            label: _normalize_word_list(words)
            for label, words in episode["gold_groups"].items()
        },
        "gold_component_indices_by_label": {
            label: [int(index) for index in indices]
            for label, indices in (episode.get("gold_component_indices_by_label") or {}).items()
        },
        "out_of_category_groups_by_label": {
            label: [_normalize_word_list(group) for group in groups]
            for label, groups in (episode.get("out_of_category_groups_by_label") or {}).items()
        },
        "out_of_category_component_indices_by_label": {
            label: [[int(index) for index in indices] for indices in groups]
            for label, groups in (episode.get("out_of_category_component_indices_by_label") or {}).items()
        },
        "out_of_category_failed_component_indices_by_label": {
            label: [[int(index) for index in indices] for indices in groups]
            for label, groups in (episode.get("out_of_category_failed_component_indices_by_label") or {}).items()
        },
        "candidate_groups": [
            {
                "gold_label": str(group["gold_label"]),
                "display_label": str(group["display_label"]),
                "words": _normalize_word_list(group["words"]),
                "satisfied_component_indices": [
                    int(index) for index in group.get("satisfied_component_indices", [])
                ],
                "failed_component_indices": [
                    int(index) for index in group.get("failed_component_indices", [])
                ],
                "unknown_component_indices": [
                    int(index) for index in group.get("unknown_component_indices", [])
                ],
                "semantic_component_evidence": [
                    dict(entry) for entry in group.get("semantic_component_evidence", [])
                ],
                "semantic_full_member_for_display_label": bool(
                    group.get("semantic_full_member_for_display_label", False)
                ),
                "semantic_strict_negative_for_display_label": bool(
                    group.get("semantic_strict_negative_for_display_label", False)
                ),
                "semantic_component_near_miss_for_display_label": bool(
                    group.get("semantic_component_near_miss_for_display_label", False)
                ),
            }
            for group in candidate["candidate_groups"]
        ],
        "answer": candidate["answer"],
        "corruption_type": candidate["corruption_type"],
        "num_wrong_labels": int(candidate["num_wrong_labels"]),
        "uses_all_labels_once": bool(candidate["uses_all_labels_once"]),
        "corrupted_gold_labels": list(candidate["corrupted_gold_labels"]),
        "candidate_confusability_score": float(candidate["candidate_confusability_score"]),
        "num_out_of_category_items": int(candidate.get("num_out_of_category_items", 0)),
        "negative_semantic_subtype": candidate.get("negative_semantic_subtype"),
        "target_label": candidate.get("target_label"),
        "failed_component_indices": [
            int(index) for index in candidate.get("failed_component_indices", [])
        ],
        "satisfied_component_indices": [
            int(index) for index in candidate.get("satisfied_component_indices", [])
        ],
        "semantic_validation_flags": dict(candidate.get("semantic_validation_flags") or {}),
        "difficulty_features": dict(episode["difficulty_features"]),
        "candidate_features": {
            "num_wrong_labels": int(candidate["num_wrong_labels"]),
            "uses_all_labels_once": bool(candidate["uses_all_labels_once"]),
            "candidate_confusability_score": float(candidate["candidate_confusability_score"]),
            "num_out_of_category_items": int(candidate.get("num_out_of_category_items", 0)),
        },
    }
    return row


def _schedule_counts(total: int, schedule: Dict[str, float]) -> Dict[str, int]:
    raw = {key: total * weight for key, weight in schedule.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    if remainder > 0:
        ranked = sorted(
            raw.items(),
            key=lambda item: (item[1] - math.floor(item[1]), item[0]),
            reverse=True,
        )
        for key, _ in ranked[:remainder]:
            counts[key] += 1
    return counts


def _build_negative_candidate_for_type(
    negative_type: str,
    episode: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    if negative_type == "two_label_swap":
        swap_mode = "highest" if episode.get("difficulty") == "d3" else "high_not_max"
        return build_two_label_swap_negative(episode, rng, mode=swap_mode)
    if negative_type == "three_cycle":
        return build_three_cycle_negative(episode, rng)
    if negative_type == "four_cycle":
        return build_four_cycle_negative(episode, rng)
    if negative_type == "out_of_category":
        return build_out_of_category_negative(episode, rng)
    if negative_type == "general_out_of_category":
        return build_out_of_category_negative(
            episode,
            rng,
            corruption_type="general_out_of_category",
            semantic_mode="general_out_of_category",
        )
    if negative_type == "none_of_components_out_of_category":
        return build_out_of_category_negative(
            episode,
            rng,
            corruption_type="none_of_components_out_of_category",
            semantic_mode="none_of_components_out_of_category",
        )
    if negative_type == "component_near_miss":
        return build_out_of_category_negative(
            episode,
            rng,
            corruption_type="component_near_miss",
            semantic_mode="component_near_miss",
        )
    raise ValueError(f"Unknown negative candidate type: {negative_type}")


def _bundle_usage_caps_for_split(
    logic_condition: str,
    difficulty: str,
    num_rows: int,
    bundle_bank: Sequence[Dict[str, Any]],
) -> tuple[int | None, int | None]:
    if logic_condition != "both" or difficulty != "d3":
        return None, None
    eligible_count = sum(
        1
        for bundle in bundle_bank
        if bundle.get("difficulty") == difficulty
        and bundle.get("logic_condition") == logic_condition
        and bool(bundle.get("examples_favored"))
    )
    if eligible_count < 4:
        return None, None
    target_usage = (int(num_rows) * 4) / eligible_count
    return math.ceil(target_usage * 1.08), math.ceil(target_usage * 1.25)


def _eligible_generation_bundles(
    logic_condition: str,
    difficulty: str,
    bundle_bank: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        dict(bundle)
        for bundle in bundle_bank
        if bundle.get("difficulty") == difficulty
        and bundle.get("logic_condition") == logic_condition
        and bool(bundle.get("examples_favored"))
    ]


def _generate_exact_category_balanced_split_rows(
    difficulty: str,
    logic_condition: str,
    split: str,
    num_rows: int,
    rng: random.Random,
    bundle_bank: Sequence[Dict[str, Any]],
    row_index_offset: int = 0,
    episode_index_offset: int = 0,
) -> List[Dict[str, Any]]:
    if num_rows % 2 != 0:
        raise ValueError(
            f"lexical_category_inference requires an even number of rows per split for exact 50/50 balance, got {num_rows}."
        )

    eligible = _eligible_generation_bundles(logic_condition, difficulty, bundle_bank)
    _category_balance_target_usage(num_rows, len(eligible))
    scheduled_edges = _find_exact_category_balance_quartet_schedule(
        logic_condition,
        difficulty,
        eligible,
        num_rows=num_rows,
        rng=rng,
    )

    negative_schedule = DESIGN_CELL_CONFIG[(logic_condition, difficulty)]["negative_schedule"]
    negative_type_counts = _schedule_counts(num_rows // 2, negative_schedule)
    row_types = ["identity"] * (num_rows // 2)
    for negative_type, count in negative_type_counts.items():
        row_types.extend([negative_type] * count)
    if len(row_types) != num_rows:
        raise RuntimeError(f"Exact split row-type schedule has {len(row_types)} entries, expected {num_rows}.")
    rng.shuffle(row_types)

    rows: List[Dict[str, Any]] = []
    next_row_index = row_index_offset + 1
    next_episode_index = episode_index_offset + 1
    for forced_bundle_ids, row_type in zip(scheduled_edges, row_types):
        episode = None
        candidate = None
        for attempt in range(80):
            try:
                episode = sample_episode_core(
                    difficulty,
                    bundle_bank,
                    next_episode_index,
                    split,
                    rng,
                    logic_condition=logic_condition,
                    forced_bundle_ids=forced_bundle_ids,
                )
                candidate = (
                    build_positive_candidate(episode, rng)
                    if row_type == "identity"
                    else _build_negative_candidate_for_type(row_type, episode, rng)
                )
            except ValueError:
                if attempt >= 79:
                    raise
                continue
            break
        if episode is None or candidate is None:
            raise RuntimeError(f"Could not build exact-balanced {row_type} row for {logic_condition}/{difficulty}.")
        rows.append(build_yes_no_row_from_episode(episode, next_row_index, candidate))
        next_row_index += 1
        next_episode_index += 1

    rng.shuffle(rows)
    return rows


def _generate_split_rows(
    difficulty: str,
    logic_condition: str,
    split: str,
    num_rows: int,
    rng: random.Random,
    bundle_bank: Sequence[Dict[str, Any]],
    row_index_offset: int = 0,
    episode_index_offset: int = 0,
) -> List[Dict[str, Any]]:
    if num_rows % 2 != 0:
        raise ValueError(
            f"lexical_category_inference requires an even number of rows per split for exact 50/50 balance, got {num_rows}."
        )

    eligible_count = len(_eligible_generation_bundles(logic_condition, difficulty, bundle_bank))
    if split == "test" and eligible_count == SOURCE_NORM_TARGET_BUNDLES_PER_CELL:
        try:
            _category_balance_target_usage(num_rows, eligible_count)
        except ValueError:
            pass
        else:
            return _generate_exact_category_balanced_split_rows(
                difficulty=difficulty,
                logic_condition=logic_condition,
                split=split,
                num_rows=num_rows,
                rng=rng,
                bundle_bank=bundle_bank,
                row_index_offset=row_index_offset,
                episode_index_offset=episode_index_offset,
            )

    positive_target = num_rows // 2
    negative_target = num_rows // 2
    negative_schedule = DESIGN_CELL_CONFIG[(logic_condition, difficulty)]["negative_schedule"]
    negative_type_counts = _schedule_counts(negative_target, negative_schedule)

    rows: List[Dict[str, Any]] = []
    next_row_index = row_index_offset + 1
    next_episode_index = episode_index_offset + 1
    bundle_usage: Counter[str] = Counter()
    bundle_usage_cap, bundle_usage_hard_cap = _bundle_usage_caps_for_split(
        logic_condition,
        difficulty,
        num_rows,
        bundle_bank,
    )

    for _ in range(positive_target):
        episode = None
        for attempt in range(1000):
            trial_usage = Counter(bundle_usage)
            try:
                episode = sample_episode_core(
                    difficulty,
                    bundle_bank,
                    next_episode_index,
                    split,
                    rng,
                    bundle_usage=trial_usage,
                    logic_condition=logic_condition,
                    bundle_usage_cap=bundle_usage_cap,
                    bundle_usage_hard_cap=bundle_usage_hard_cap,
                )
            except ValueError:
                if attempt >= 999:
                    raise
                continue
            bundle_usage.clear()
            bundle_usage.update(trial_usage)
            break
        if episode is None:
            raise RuntimeError(f"Could not sample a valid positive episode for {logic_condition}/{difficulty}.")
        candidate = build_positive_candidate(episode, rng)
        rows.append(build_yes_no_row_from_episode(episode, next_row_index, candidate))
        next_row_index += 1
        next_episode_index += 1

    for negative_type, count in negative_type_counts.items():
        for _ in range(count):
            episode = None
            candidate = None
            for attempt in range(1000):
                trial_usage = Counter(bundle_usage)
                try:
                    episode = sample_episode_core(
                        difficulty,
                        bundle_bank,
                        next_episode_index,
                        split,
                        rng,
                        bundle_usage=trial_usage,
                        logic_condition=logic_condition,
                        bundle_usage_cap=bundle_usage_cap,
                        bundle_usage_hard_cap=bundle_usage_hard_cap,
                    )
                    candidate = _build_negative_candidate_for_type(negative_type, episode, rng)
                except ValueError:
                    if attempt >= 999:
                        raise
                    continue
                bundle_usage.clear()
                bundle_usage.update(trial_usage)
                break
            if episode is None or candidate is None:
                raise RuntimeError(
                    f"Could not sample a valid {negative_type} episode for {logic_condition}/{difficulty}."
                )
            rows.append(build_yes_no_row_from_episode(episode, next_row_index, candidate))
            next_row_index += 1
            next_episode_index += 1

    rng.shuffle(rows)
    return rows


def generate_dataset_for_difficulty(
    difficulty: str,
    train_samples: int,
    test_samples: int,
    seed: int,
    bundle_bank_path: str | None = None,
    logic_condition: str | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    design_cell = resolve_design_cell(difficulty, logic_condition)
    level = str(design_cell["difficulty"])
    group = str(design_cell["logic_condition"])
    bundle_bank = load_bundle_bank(bundle_bank_path)
    train_rng = random.Random(seed)
    test_rng = random.Random(seed + 100_000)

    train_rows = _generate_split_rows(
        difficulty=level,
        logic_condition=group,
        split="train",
        num_rows=train_samples,
        rng=train_rng,
        bundle_bank=bundle_bank,
        row_index_offset=0,
        episode_index_offset=0,
    )
    test_rows = _generate_split_rows(
        difficulty=level,
        logic_condition=group,
        split="test",
        num_rows=test_samples,
        rng=test_rng,
        bundle_bank=bundle_bank,
        row_index_offset=train_samples,
        episode_index_offset=train_samples,
    )
    return train_rows, test_rows


def generate_dataset_for_design_cell(
    logic_condition: str,
    difficulty: str,
    train_samples: int,
    test_samples: int,
    seed: int,
    bundle_bank_path: str | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return generate_dataset_for_difficulty(
        difficulty=difficulty,
        logic_condition=logic_condition,
        train_samples=train_samples,
        test_samples=test_samples,
        seed=seed,
        bundle_bank_path=bundle_bank_path,
    )


def build_dataset_lint_report(
    rows_by_design_cell: Dict[str, Sequence[Dict[str, Any]]],
    output_root: str | Path | None = None,
) -> Dict[str, Any]:
    expected_keys = [design_cell_key(group, difficulty) for group, difficulty in REAL_DESIGN_CELLS]
    surface_form_phrases = [
        "words starting",
        "words ending",
        "four-letter",
        "five-letter",
        "anagram",
        "homophone",
        "hidden word",
        "prefix",
        "suffix",
        "contains letter",
        "starting letter",
        "ending letter",
        "word length",
    ]

    normalized_rows: Dict[str, List[Dict[str, Any]]] = {}
    for raw_key, rows in rows_by_design_cell.items():
        if isinstance(raw_key, tuple):
            key = design_cell_key(str(raw_key[0]), str(raw_key[1]))
        else:
            key = str(raw_key)
        normalized_rows[key] = [dict(row) for row in rows]

    passed = True
    missing_cells = [key for key in expected_keys if key not in normalized_rows]
    extra_cells = [key for key in normalized_rows if key not in expected_keys]
    design_cell_presence = {
        "passed": not missing_cells and not extra_cells,
        "expected_cells": expected_keys,
        "missing_cells": missing_cells,
        "extra_cells": extra_cells,
    }
    passed = passed and bool(design_cell_presence["passed"])

    per_design_cell: Dict[str, Dict[str, Any]] = {}
    design_cell_checks: Dict[str, Dict[str, Any]] = {}
    candidate_structure_checks: Dict[str, Dict[str, Any]] = {}
    category_diversity_checks: Dict[str, Dict[str, Any]] = {}
    category_definition_text_checks: Dict[str, Dict[str, Any]] = {}
    concept_quality_checks: Dict[str, Dict[str, Any]] = {}
    positive_coverage_checks: Dict[str, Dict[str, Any]] = {}
    component_coverage_checks: Dict[str, Dict[str, Any]] = {}
    semantic_negative_validity_checks: Dict[str, Dict[str, Any]] = {}
    surface_form_checks: Dict[str, Dict[str, Any]] = {}

    for group, difficulty in REAL_DESIGN_CELLS:
        key = design_cell_key(group, difficulty)
        spec = DESIGN_CELL_SPECS[(group, difficulty)]
        rows = normalized_rows.get(key, [])
        negative_rows = [row for row in rows if row.get("answer") == "No"]
        answer_counts = Counter(str(row.get("answer")) for row in rows)
        arity_values = [int(row.get("category_arity") or 0) for row in rows]
        condition_values = [int(row.get("condition_count") or 0) for row in rows]
        operator_counts = Counter(str(row.get("category_operator")) for row in rows)
        logic_counts = Counter(str(row.get("logic_condition")) for row in rows)
        bundle_quartets = Counter(
            tuple(str((row.get("bundle_ids") or {}).get(label) or "") for label in row.get("labels") or [])
            for row in rows
        )
        gloss_quartets = Counter(
            tuple(str((row.get("rule_glosses") or {}).get(label) or "") for label in row.get("labels") or [])
            for row in rows
        )
        unique_bundle_ids = {
            str(bundle_id)
            for row in rows
            for bundle_id in (row.get("bundle_ids") or {}).values()
            if str(bundle_id)
        }
        bundle_use_counts = Counter(
            str(bundle_id)
            for row in rows
            for bundle_id in (row.get("bundle_ids") or {}).values()
            if str(bundle_id)
        )
        bundle_use_values = sorted(bundle_use_counts.values())
        bundle_use_gini = 0.0
        if bundle_use_values:
            n_values = len(bundle_use_values)
            total_usage = sum(bundle_use_values)
            if total_usage:
                bundle_use_gini = (
                    2 * sum((index + 1) * value for index, value in enumerate(bundle_use_values))
                    / (n_values * total_usage)
                    - (n_values + 1) / n_values
                )
        unique_rule_glosses = {
            str(gloss)
            for row in rows
            for gloss in (row.get("rule_glosses") or {}).values()
            if str(gloss)
        }
        category_family_counts = Counter(
            str(family)
            for row in rows
            for families in (row.get("category_families_by_label") or {}).values()
            for family in (families if isinstance(families, list) else [families])
            if str(family)
        )
        if not category_family_counts:
            category_family_counts = Counter(
                str(family)
                for row in rows
                for family in (row.get("category_family_by_label") or {}).values()
                if str(family)
            )
        total_family_slots = sum(category_family_counts.values())
        family_fractions = {
            family: count / total_family_slots
            for family, count in category_family_counts.items()
        } if total_family_slots else {}
        category_family_entropy = -sum(
            fraction * math.log(fraction, 2)
            for fraction in family_fractions.values()
            if fraction > 0
        )
        family_fraction_thresholds = {
            "shared/d1": 0.50,
            "either/d2": 0.40,
            "either/d3": 0.40,
            "both/d2": 0.35,
            "both/d3": 0.35,
        }
        max_category_family_fraction = max(family_fractions.values(), default=0.0)

        per_design_cell[key] = {
            "num_rows": len(rows),
            "answer_counts": dict(answer_counts),
            "negative_rows": len(negative_rows),
            "negative_avg_num_wrong_labels": _safe_mean(
                [float(row.get("num_wrong_labels", 0)) for row in negative_rows]
            ),
            "negative_all_labels_once_rate": _safe_mean(
                [1.0 if row.get("uses_all_labels_once") else 0.0 for row in negative_rows]
            ),
            "negative_out_of_category_rate": _safe_mean(
                [
                    1.0 if row.get("corruption_type") in {
                        "out_of_category",
                        "general_out_of_category",
                        "none_of_components_out_of_category",
                        "component_near_miss",
                    } else 0.0
                    for row in negative_rows
                ]
            ),
            "negative_avg_confusability_score": _safe_mean(
                [float(row.get("candidate_confusability_score", 0.0)) for row in negative_rows]
            ),
            "category_arity_mean": _safe_mean([float(value) for value in arity_values]),
            "condition_count_mean": _safe_mean([float(value) for value in condition_values]),
            "operator_counts": dict(operator_counts),
            "logic_condition_counts": dict(logic_counts),
            "corruption_counts": dict(Counter(str(row.get("corruption_type")) for row in negative_rows)),
            "unique_bundle_ids": len(unique_bundle_ids),
            "bundle_use_min": min(bundle_use_values, default=0),
            "bundle_use_max": max(bundle_use_values, default=0),
            "bundle_use_spread": (
                max(bundle_use_values) - min(bundle_use_values)
                if bundle_use_values
                else 0
            ),
            "bundle_use_gini": bundle_use_gini,
            "unique_rule_glosses": len(unique_rule_glosses),
            "unique_bundle_quartets": len(bundle_quartets),
            "unique_rule_gloss_quartets": len(gloss_quartets),
            "unique_category_families": len(category_family_counts),
            "category_family_counts": dict(sorted(category_family_counts.items())),
            "category_family_entropy": category_family_entropy,
            "max_category_family_fraction": max_category_family_fraction,
            "family_quota_threshold": family_fraction_thresholds.get(key, 1.0),
            "family_quota_passed": max_category_family_fraction <= family_fraction_thresholds.get(key, 1.0),
            "max_repeated_bundle_quartet_count": max(bundle_quartets.values(), default=0),
            "max_repeated_rule_gloss_quartet_count": max(gloss_quartets.values(), default=0),
        }

        metadata_failures = []
        for row in rows:
            if row.get("logic_condition") != spec["logic_condition"]:
                metadata_failures.append((row.get("id"), "logic_condition"))
            if row.get("category_operator") != spec["category_operator"]:
                metadata_failures.append((row.get("id"), "category_operator"))
            if int(row.get("category_arity") or 0) != int(spec["category_arity"]):
                metadata_failures.append((row.get("id"), "category_arity"))
            if int(row.get("condition_count") or 0) != int(spec["condition_count"]):
                metadata_failures.append((row.get("id"), "condition_count"))
            if bool(row.get("shared_d1")) != bool(spec["shared_d1"]):
                metadata_failures.append((row.get("id"), "shared_d1"))
            if row.get("difficulty") != spec["difficulty"]:
                metadata_failures.append((row.get("id"), "difficulty"))

        metadata_passed = not metadata_failures and bool(rows)
        design_cell_checks[key] = {
            "passed": metadata_passed,
            "expected": {
                "logic_condition": spec["logic_condition"],
                "category_operator": spec["category_operator"],
                "category_arity": spec["category_arity"],
                "shared_d1": spec["shared_d1"],
            },
            "failure_count": len(metadata_failures),
            "sample_failures": metadata_failures[:10],
        }
        passed = passed and metadata_passed

        balance_passed = answer_counts.get("Yes", 0) == answer_counts.get("No", 0) and len(rows) > 0
        candidate_schema_passed = all(
            len(row.get("candidate_groups") or []) == 4
            and all(len(group.get("words") or []) == 1 for group in row.get("candidate_groups") or [])
            for row in rows
        )
        difficulty_targets = NEGATIVE_STRUCTURE_TARGETS[difficulty]
        negative_results: Dict[str, Any] = {}
        negative_passed = True
        for feature, bounds in difficulty_targets.items():
            value = float(per_design_cell[key][feature])
            feature_passed = _value_within_bounds(value, bounds)
            negative_results[feature] = {
                "passed": feature_passed,
                "value": value,
                "bounds": dict(bounds),
            }
            negative_passed = negative_passed and feature_passed
        candidate_structure_checks[key] = {
            "passed": balance_passed and candidate_schema_passed and negative_passed,
            "answer_balance": {"passed": balance_passed, "counts": dict(answer_counts)},
            "candidate_schema": {"passed": candidate_schema_passed},
            "negative_structure": negative_results,
        }
        passed = passed and bool(candidate_structure_checks[key]["passed"])

        diversity_thresholds = {
            "unique_bundle_ids": min(20, max(1, len(rows) // 8)),
            "unique_rule_glosses": min(20, max(1, len(rows) // 8)),
            "unique_bundle_quartets": min(100, max(1, len(rows) // 4)),
            "unique_rule_gloss_quartets": min(100, max(1, len(rows) // 4)),
            "unique_category_families": min(8, max(1, len(rows) // 256)),
        }
        diversity_results: Dict[str, Any] = {}
        diversity_passed = bool(rows)
        for feature, minimum in diversity_thresholds.items():
            value = int(per_design_cell[key][feature])
            feature_passed = value >= minimum
            diversity_results[feature] = {
                "passed": feature_passed,
                "value": value,
                "min": minimum,
            }
            diversity_passed = diversity_passed and feature_passed
        category_diversity_checks[key] = {
            "passed": diversity_passed,
            "features": diversity_results,
            "category_family_counts": per_design_cell[key]["category_family_counts"],
            "bundle_use_min": per_design_cell[key]["bundle_use_min"],
            "bundle_use_max": per_design_cell[key]["bundle_use_max"],
            "bundle_use_spread": per_design_cell[key]["bundle_use_spread"],
            "bundle_use_gini": per_design_cell[key]["bundle_use_gini"],
            "category_family_entropy": per_design_cell[key]["category_family_entropy"],
            "max_category_family_fraction": per_design_cell[key]["max_category_family_fraction"],
            "family_quota_threshold": per_design_cell[key]["family_quota_threshold"],
            "family_quota_passed": per_design_cell[key]["family_quota_passed"],
            "max_repeated_bundle_quartet_count": per_design_cell[key]["max_repeated_bundle_quartet_count"],
            "max_repeated_rule_gloss_quartet_count": per_design_cell[key]["max_repeated_rule_gloss_quartet_count"],
        }
        if key == "both/d3" and len(rows) >= 500:
            target_usage = (len(rows) * 4 / max(1, len(unique_bundle_ids))) if rows else 0.0
            max_min_ratio = (
                per_design_cell[key]["bundle_use_max"] / max(1, per_design_cell[key]["bundle_use_min"])
                if per_design_cell[key]["bundle_use_min"]
                else float("inf")
            )
            max_min_ratio_threshold = 1.38 if len(unique_bundle_ids) <= 70 else 1.35
            bundle_balance_passed = (
                per_design_cell[key]["bundle_use_gini"] <= 0.08
                and per_design_cell[key]["bundle_use_spread"] <= 40
                and max_min_ratio <= max_min_ratio_threshold
                and per_design_cell[key]["bundle_use_max"] <= math.ceil(target_usage * 1.25)
            )
            category_diversity_checks[key]["bundle_balance"] = {
                "passed": bundle_balance_passed,
                "target_usage": target_usage,
                "max_min_ratio": max_min_ratio,
                "max_allowed_usage": math.ceil(target_usage * 1.25) if target_usage else 0,
                "gini_max": 0.08,
                "spread_max": 40,
                "max_min_ratio_max": max_min_ratio_threshold,
                "max_min_ratio_note": (
                    "Relaxed from 1.35 to 1.38 for v34 strict both/d3 banks "
                    "with 70 or fewer bundles; Gini, spread, and 125% max-usage caps remain enforced."
                ) if max_min_ratio_threshold > 1.35 else "",
            }
            category_diversity_checks[key]["passed"] = (
                bool(category_diversity_checks[key]["passed"]) and bundle_balance_passed
            )
            diversity_passed = diversity_passed and bundle_balance_passed
        elif key == "both/d3":
            category_diversity_checks[key]["bundle_balance"] = {
                "passed": True,
                "skipped": True,
                "reason": "strict both/d3 bundle-balance thresholds apply only to full-size splits",
            }
        passed = passed and diversity_passed

        incidental_or_hits = []
        malformed_gloss_hits = []
        for row in rows:
            for label, gloss in (row.get("rule_glosses") or {}).items():
                if _category_gloss_has_incidental_disjunction(gloss):
                    incidental_or_hits.append((row.get("id"), label, gloss))
                normalized_gloss = str(gloss).strip().lower()
                if "indoes" in normalized_gloss or "does produces" in normalized_gloss:
                    malformed_gloss_hits.append((row.get("id"), label, gloss))
        category_text_passed = not incidental_or_hits and not malformed_gloss_hits and bool(rows)
        category_definition_text_checks[key] = {
            "passed": category_text_passed,
            "banned_tokens": ["or", "and/or", "indoes", "does produces"],
            "sample_hits": (incidental_or_hits + malformed_gloss_hits)[:10],
        }
        passed = passed and category_text_passed

        concept_quality_failures = []
        for row in rows:
            for label, gloss in (row.get("rule_glosses") or {}).items():
                components = _rule_gloss_components_for_lint(gloss)
                banned_hits = [
                    fragment
                    for fragment in SOURCE_NORM_LINT_BANNED_CONCEPT_FRAGMENTS
                    if any(fragment in component for component in components)
                ]
                if banned_hits:
                    concept_quality_failures.append((row.get("id"), label, "banned_fragment", banned_hits, gloss))
                    continue
                if len(components) > 1 and not _source_norm_component_glosses_are_clean(
                    [{"rule_gloss": component} for component in components]
                ):
                    concept_quality_failures.append((row.get("id"), label, "unclean_component_combo", components, gloss))
        concept_quality_passed = not concept_quality_failures and bool(rows)
        concept_quality_checks[key] = {
            "passed": concept_quality_passed,
            "banned_fragments": list(SOURCE_NORM_LINT_BANNED_CONCEPT_FRAGMENTS),
            "sample_failures": concept_quality_failures[:10],
        }
        passed = passed and concept_quality_passed

        min_positive_items = None
        coverage_failures = []
        for row in rows:
            labels = list(row.get("labels") or [])
            support = row.get("support_groups_by_label") or {}
            gold = row.get("gold_groups") or {}
            for label in labels:
                positives = set(str(word) for word in gold.get(label, []))
                for group in support.get(label, []):
                    positives.update(str(word) for word in group)
                count = len(positives)
                min_positive_items = count if min_positive_items is None else min(min_positive_items, count)
                if count < 12:
                    coverage_failures.append((row.get("id"), label, count))
        coverage_passed = not coverage_failures and bool(rows)
        positive_coverage_checks[key] = {
            "passed": coverage_passed,
            "minimum_verified_positive_items_per_category": min_positive_items or 0,
            "sample_failures": coverage_failures[:10],
        }
        passed = passed and coverage_passed

        component_failures = []
        for row in rows:
            labels = list(row.get("labels") or [])
            component_count = int(row.get("category_arity") or 1)
            expected_components = set(range(max(1, component_count)))
            support_components = row.get("support_component_indices_by_label") or {}
            out_failed_components = row.get("out_of_category_failed_component_indices_by_label") or {}
            for label in labels:
                visible_support_components = {
                    int(index)
                    for indices in support_components.get(label, [])
                    for index in indices
                }
                if not expected_components <= visible_support_components:
                    component_failures.append(
                        (row.get("id"), label, "support", sorted(visible_support_components), sorted(expected_components))
                    )
                if row.get("category_operator") == "and" and component_count > 1:
                    visible_failed_components = {
                        int(component_index)
                        for indices in out_failed_components.get(label, [])
                        for component_index in indices
                    }
                    if not (visible_failed_components & expected_components):
                        component_failures.append(
                            (row.get("id"), label, "near_miss_failed_components", sorted(visible_failed_components), "need at least one explicit failed component")
                        )
        component_passed = not component_failures and bool(rows)
        component_coverage_checks[key] = {
            "passed": component_passed,
            "sample_failures": component_failures[:10],
        }
        passed = passed and component_passed

        invalid_swap_cross_fit = []
        invalid_out_of_category = []
        invalid_positive = []
        invalid_component_near_miss = []
        for row in rows:
            for group in row.get("candidate_groups") or []:
                full_member = bool(group.get("semantic_full_member_for_display_label"))
                gold_label = str(group.get("gold_label") or "")
                display_label = str(group.get("display_label") or "")
                corruption_type = str(row.get("corruption_type") or "")
                if row.get("answer") == "Yes" and not full_member:
                    invalid_positive.append((row.get("id"), display_label, group.get("words")))
                if row.get("answer") != "No":
                    continue
                is_corrupted = gold_label == OUT_OF_CATEGORY_LABEL or gold_label != display_label
                strict_negative = bool(group.get("semantic_strict_negative_for_display_label"))
                if not is_corrupted:
                    continue
                if corruption_type == "component_near_miss" and not bool(
                    group.get("semantic_component_near_miss_for_display_label")
                ):
                    invalid_component_near_miss.append(
                        (
                            row.get("id"),
                            display_label,
                            group.get("words"),
                            group.get("satisfied_component_indices"),
                            group.get("failed_component_indices"),
                            group.get("unknown_component_indices"),
                        )
                    )
                    continue
                if full_member or (
                    corruption_type in {
                        "out_of_category",
                        "general_out_of_category",
                        "none_of_components_out_of_category",
                        "component_near_miss",
                    }
                    and not strict_negative
                ):
                    failure = (row.get("id"), corruption_type, gold_label, display_label, group.get("words"))
                    if corruption_type in {"two_label_swap", "three_cycle", "four_cycle"}:
                        invalid_swap_cross_fit.append(failure)
                    if corruption_type in {
                        "out_of_category",
                        "general_out_of_category",
                        "none_of_components_out_of_category",
                        "component_near_miss",
                    }:
                        invalid_out_of_category.append(failure)
                    continue
        semantic_validity_passed = (
            not invalid_positive
            and not invalid_swap_cross_fit
            and not invalid_out_of_category
            and not invalid_component_near_miss
        )
        semantic_negative_validity_checks[key] = {
            "passed": semantic_validity_passed,
            "invalid_positive_count": len(invalid_positive),
            "invalid_swap_cross_fit_count": len(invalid_swap_cross_fit),
            "invalid_out_of_category_count": len(invalid_out_of_category),
            "invalid_component_near_miss_count": len(invalid_component_near_miss),
            "sample_failures": (
                invalid_positive
                + invalid_swap_cross_fit
                + invalid_out_of_category
                + invalid_component_near_miss
            )[:10],
        }
        passed = passed and semantic_validity_passed

        phrase_hits = []
        for row in rows:
            gloss_blob = " ".join(str(value).lower() for value in (row.get("rule_glosses") or {}).values())
            for phrase in surface_form_phrases:
                if phrase in gloss_blob:
                    phrase_hits.append((row.get("id"), phrase))
        surface_passed = not phrase_hits
        surface_form_checks[key] = {
            "passed": surface_passed,
            "banned_phrases": surface_form_phrases,
            "sample_hits": phrase_hits[:10],
        }
        passed = passed and surface_passed

    monotonicity_checks: Dict[str, Dict[str, Any]] = {}
    for condition, keys in {
        "either": ["shared/d1", "either/d2", "either/d3"],
        "both": ["shared/d1", "both/d2", "both/d3"],
    }.items():
        condition_passed = all(key in per_design_cell for key in keys)
        values = {
            key: float(per_design_cell.get(key, {}).get("category_arity_mean", 0.0))
            for key in keys
        }
        expected = {keys[0]: 1.0, keys[1]: 2.0, keys[2]: 3.0}
        if condition_passed:
            condition_passed = values == expected
        monotonicity_checks[condition] = {
            "passed": condition_passed,
            "path": keys,
            "feature": "category_arity_mean",
            "values": values,
            "expected_values": expected,
        }
        passed = passed and condition_passed

    file_checks: Dict[str, Any] = {"passed": True, "skipped": output_root is None}
    if output_root is not None:
        root = Path(output_root)
        cell_dirs = []
        if root.exists():
            for logic_dir in root.iterdir():
                if not logic_dir.is_dir():
                    continue
                if logic_dir.name not in {"shared", "either", "both"}:
                    continue
                for difficulty_dir in logic_dir.iterdir():
                    if difficulty_dir.is_dir():
                        cell_dirs.append(f"{logic_dir.name}/{difficulty_dir.name}")
        unexpected_dirs = sorted(set(cell_dirs) - set(expected_keys))
        missing_dirs = [key for key in expected_keys if key not in cell_dirs]
        missing_files = []
        extra_d1_dirs = [key for key in cell_dirs if key in {"either/d1", "both/d1"}]
        for key in expected_keys:
            group, difficulty = key.split("/")
            for split in ("train", "test"):
                split_dir = root / group / difficulty / split
                pattern = f"lexical_category_inference_{group}_{difficulty}_{split}_n*.jsonl"
                if not list(split_dir.glob(pattern)):
                    missing_files.append(f"{key}/{split}/{pattern}")
        file_passed = not unexpected_dirs and not missing_dirs and not missing_files and not extra_d1_dirs
        file_checks = {
            "passed": file_passed,
            "expected_cells": expected_keys,
            "discovered_cell_dirs": sorted(cell_dirs),
            "missing_cell_dirs": missing_dirs,
            "unexpected_cell_dirs": unexpected_dirs,
            "missing_files": missing_files,
            "shared_d1_exists_once": cell_dirs.count("shared/d1") == 1,
            "no_separate_either_or_both_d1": not extra_d1_dirs,
        }
        passed = passed and file_passed

    return {
        "design": "logic_condition_v2",
        "real_design_cells": expected_keys,
        "design_cell_presence": design_cell_presence,
        "per_design_cell": per_design_cell,
        "design_cell_checks": design_cell_checks,
        "candidate_structure_checks": candidate_structure_checks,
        "category_diversity_checks": category_diversity_checks,
        "category_definition_text_checks": category_definition_text_checks,
        "concept_quality_checks": concept_quality_checks,
        "positive_coverage_checks": positive_coverage_checks,
        "component_coverage_checks": component_coverage_checks,
        "semantic_negative_validity_checks": semantic_negative_validity_checks,
        "surface_form_checks": surface_form_checks,
        "monotonicity_checks": monotonicity_checks,
        "monotonicity_note": (
            "Monotonicity is an intended structural proxy based on the number of component descriptions; "
            "it is not a guaranteed empirical accuracy ordering."
        ),
        "file_checks": file_checks,
        "passed": passed,
    }


def write_dataset_lint_report(output_path: Path, report: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_semantic_validation_report(
    bundle_bank: Sequence[Dict[str, Any]],
    validator: SemanticValidator | None = None,
) -> Dict[str, Any]:
    validator = validator or get_default_semantic_validator()
    records: List[Dict[str, Any]] = []

    for bundle in bundle_bank:
        bundle = _normalize_bundle_word_fields(bundle)
        source = bundle.get("source") or {}
        component_glosses = _bundle_component_glosses(bundle)
        if not component_glosses:
            continue
        operator = str(bundle.get("category_operator") or source.get("category_operator") or "")

        for word in _flatten_words(bundle):
            evidence = _semantic_component_evidence_profile_for_bundle(word, bundle, validator)
            profile = [entry.get("value") is True for entry in evidence]
            full_member = all(profile) if operator == "and" else any(profile)
            if not full_member:
                records.append(
                    {
                        "bundle_id": bundle.get("bundle_id"),
                        "logic_condition": bundle.get("logic_condition"),
                        "difficulty": bundle.get("difficulty"),
                        "rule_gloss": bundle.get("rule_gloss"),
                        "component_glosses": component_glosses,
                        "word": word,
                        "role": "support_or_query",
                        "claimed_component_indices": _component_indices_for_word(bundle, word),
                        "validated_component_membership": profile,
                        "validated_component_evidence": evidence,
                        "validated_full_category_member": full_member,
                        "valid_for_role": False,
                        "reason": "positive_word_fails_component_validation",
                    }
                )

        if operator != "and":
            continue
        near_misses = source.get("near_miss_words_by_component")
        near_miss_components = source.get("near_miss_component_indices_by_word") or {}
        if not isinstance(near_misses, list):
            continue
        for component_index, words in enumerate(near_misses):
            for word in words:
                normalized_word = _normalize_word_surface(word)
                evidence = _semantic_component_evidence_profile_for_bundle(normalized_word, bundle, validator)
                values = [entry.get("value") for entry in evidence]
                profile = [value is True for value in values]
                full_member = all(profile)
                satisfied = [index for index, value in enumerate(values) if value is True]
                failed = [index for index, value in enumerate(values) if value is False]
                unknown = [index for index, value in enumerate(values) if value is None]
                failed_component_glosses = [
                    component_glosses[index]
                    for index in failed
                    if index < len(component_glosses)
                ]
                high_risk_failure_without_override = [
                    index
                    for index in failed
                    if index < len(component_glosses)
                    and not _evidence_has_independent_negative_for_high_risk_failure(evidence[index], component_glosses[index])
                ]
                semantic_disqualification_reasons = _strict_near_miss_semantic_disqualification_reasons(
                    normalized_word,
                    component_glosses,
                    evidence,
                    validator,
                )
                valid_for_role = (
                    len(satisfied) == len(values) - 1
                    and len(failed) == 1
                    and not unknown
                    and not full_member
                    and not high_risk_failure_without_override
                    and not semantic_disqualification_reasons
                )
                if valid_for_role:
                    continue
                if full_member:
                    reason = "near_miss_satisfies_all_components"
                elif semantic_disqualification_reasons:
                    reason = "near_miss_semantic_disqualification"
                elif not satisfied:
                    reason = "near_miss_fails_all_components"
                elif high_risk_failure_without_override:
                    reason = "near_miss_high_risk_failure_without_override_negative"
                elif unknown:
                    reason = "near_miss_has_unknown_component"
                elif len(failed) != 1:
                    reason = "near_miss_not_single_failed_component"
                elif len(satisfied) != len(values) - 1:
                    reason = "near_miss_not_boundary_case"
                else:
                    reason = "near_miss_invalid_component_profile"
                records.append(
                    {
                        "bundle_id": bundle.get("bundle_id"),
                        "logic_condition": bundle.get("logic_condition"),
                        "difficulty": bundle.get("difficulty"),
                        "rule_gloss": bundle.get("rule_gloss"),
                        "component_glosses": component_glosses,
                        "word": normalized_word,
                        "role": "near_miss",
                        "near_miss_component_index": component_index,
                        "claimed_component_indices": [
                            int(index) for index in near_miss_components.get(normalized_word, [])
                        ],
                        "validated_component_membership": profile,
                        "validated_component_evidence": evidence,
                        "validated_full_category_member": full_member,
                        "satisfied_component_indices": satisfied,
                        "failed_component_indices": failed,
                        "unknown_component_indices": unknown,
                        "failed_component_glosses": failed_component_glosses,
                        "semantic_disqualification_reasons": semantic_disqualification_reasons,
                        "valid_for_role": False,
                        "reason": reason,
                    }
                )

    return {
        "version": "v34",
        "validation_mode": "generator_invariants_plus_human_override_adjudication",
        "independent_audit_summary": {
            "human_override_glosses": len(_load_semantic_word_overrides()),
            "strict_and_near_miss_requirement": "exactly k-1 true components, exactly one false component, zero unknown components, no cross-sense or high-risk rated-property plausibility conflicts",
            "source_intersection_positive_requirement": "THINGSplus-derived AND positives must have at least one same uniqueID sense across all THINGSplus components.",
            "high_risk_rated_property_requirement": "High-risk rated-property positives, portable-artifact stationary cases, body-part living cases, animal artificiality cases, and near-miss satisfied components must not contradict artifact or animal domain evidence.",
            "positive_words_checked": sum(len(_flatten_words(bundle)) for bundle in bundle_bank),
            "invalid_records_are_failures": True,
        },
        "passed": not records,
        "invalid_record_count": len(records),
        "records": records,
    }


def write_semantic_validation_report(output_path: Path, report: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _source_intersection_positive_occurrences(
    rows: Sequence[Dict[str, Any]],
) -> Dict[tuple[str, str], List[Dict[str, Any]]]:
    occurrences: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        labels = [str(label) for label in (row.get("labels") or [])]
        bundle_ids = {
            str(label): str(bundle_id)
            for label, bundle_id in (row.get("bundle_ids") or {}).items()
        }
        for label in labels:
            bundle_id = bundle_ids.get(label)
            if not bundle_id:
                continue
            for role, groups_by_label in (
                ("support", row.get("support_groups_by_label") or {}),
                ("gold", row.get("gold_groups") or {}),
            ):
                groups = groups_by_label.get(label) or []
                if role == "gold" and groups and isinstance(groups[0], str):
                    groups = [groups]
                for group_index, group_words in enumerate(groups):
                    for word in group_words:
                        occurrences[(bundle_id, _normalize_word_surface(word))].append(
                            {
                                "row_id": row.get("id"),
                                "split": row.get("split"),
                                "label": label,
                                "role": role,
                                "group_index": group_index,
                            }
                        )
            for group_index, group in enumerate(row.get("candidate_groups") or []):
                if str(group.get("display_label") or "") != label:
                    continue
                if str(group.get("gold_label") or "") != label:
                    continue
                for word in group.get("words") or []:
                    occurrences[(bundle_id, _normalize_word_surface(word))].append(
                        {
                            "row_id": row.get("id"),
                            "split": row.get("split"),
                            "label": label,
                            "role": "candidate",
                            "answer": row.get("answer"),
                            "candidate_group_index": group_index,
                        }
                    )
    return occurrences


def build_source_intersection_positive_sense_audit(
    rows: Sequence[Dict[str, Any]],
    bundle_bank: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    occurrences = _source_intersection_positive_occurrences(rows)
    records: List[Dict[str, Any]] = []
    invalid_records: List[Dict[str, Any]] = []
    checked_words = 0

    for bundle in bundle_bank:
        bundle = _normalize_bundle_word_fields(bundle)
        source = bundle.get("source") or {}
        if str(bundle.get("category_operator") or "") != "and":
            continue
        if str(source.get("bundle_builder") or "") != "source_norms_intersection_v1":
            continue
        source_datasets = [str(dataset) for dataset in (source.get("source_datasets") or [])]
        thingsplus_indices = [
            index for index, dataset in enumerate(source_datasets)
            if dataset == "thingsplus"
        ]
        if len(thingsplus_indices) < 2:
            continue
        component_entity_ids_by_word_by_index = source.get("component_entity_ids_by_word_by_index") or []
        same_source_entity_ids_by_word = source.get("source_entity_ids_by_word") or {}
        positive_words = sorted(set(_flatten_words(bundle)))
        for word in positive_words:
            normalized_word = _normalize_word_surface(word)
            checked_words += 1
            per_component_entity_ids: Dict[str, List[str]] = {}
            for component_index in thingsplus_indices:
                component_map = (
                    component_entity_ids_by_word_by_index[component_index]
                    if component_index < len(component_entity_ids_by_word_by_index)
                    and isinstance(component_entity_ids_by_word_by_index[component_index], dict)
                    else {}
                )
                per_component_entity_ids[str(component_index)] = sorted(
                    str(entity_id)
                    for entity_id in (component_map.get(normalized_word) or [])
                )
            common_entity_ids = sorted(
                str(entity_id)
                for entity_id in (same_source_entity_ids_by_word.get(normalized_word) or [])
            )
            valid = bool(common_entity_ids)
            record = {
                "bundle_id": bundle.get("bundle_id"),
                "logic_condition": bundle.get("logic_condition"),
                "difficulty": bundle.get("difficulty"),
                "rule_gloss": bundle.get("rule_gloss"),
                "component_glosses": source.get("component_rule_glosses") or [],
                "component_predicate_ids": source.get("component_predicate_ids") or [],
                "thingsplus_component_indices": thingsplus_indices,
                "word": normalized_word,
                "component_entity_ids": per_component_entity_ids,
                "same_source_entity_ids": common_entity_ids,
                "valid_same_source_positive": valid,
                "occurrence_count": len(occurrences.get((str(bundle.get("bundle_id")), normalized_word), [])),
                "sample_occurrences": occurrences.get((str(bundle.get("bundle_id")), normalized_word), [])[:20],
            }
            if valid:
                records.append(record)
            else:
                invalid_records.append(record)

    return {
        "version": "v34",
        "validation_mode": "thingsplus_unique_id_same_sense_positive_audit",
        "checked_words": checked_words,
        "checked_records": len(records) + len(invalid_records),
        "invalid_record_count": len(invalid_records),
        "passed": not invalid_records,
        "records": records,
        "invalid_records": invalid_records,
    }


def write_source_intersection_positive_sense_audit(output_path: Path, report: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_source_intersection_material_same_instance_audit(
    rows: Sequence[Dict[str, Any]],
    bundle_bank: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    occurrences = _source_intersection_positive_occurrences(rows)
    records: List[Dict[str, Any]] = []
    invalid_records: List[Dict[str, Any]] = []

    for bundle in bundle_bank:
        bundle = _normalize_bundle_word_fields(bundle)
        source = bundle.get("source") or {}
        if str(bundle.get("category_operator") or "") != "and":
            continue
        if str(source.get("bundle_builder") or "") != "source_norms_intersection_v1":
            continue
        component_glosses = [
            _source_norm_clean_gloss(gloss)
            for gloss in (source.get("component_rule_glosses") or [])
        ]
        material_component_indices = [
            index
            for index, gloss in enumerate(component_glosses)
            if gloss in SOURCE_NORM_MATERIAL_COMPONENT_GLOSSES
        ]
        if len(material_component_indices) < 2:
            continue

        positive_words = sorted(set(_flatten_words(bundle)))
        record = {
            "bundle_id": bundle.get("bundle_id"),
            "logic_condition": bundle.get("logic_condition"),
            "difficulty": bundle.get("difficulty"),
            "rule_gloss": bundle.get("rule_gloss"),
            "component_glosses": component_glosses,
            "material_component_indices": material_component_indices,
            "material_component_glosses": [
                component_glosses[index]
                for index in material_component_indices
            ],
            "positive_words": positive_words,
            "occurrence_count": sum(
                len(occurrences.get((str(bundle.get("bundle_id")), _normalize_word_surface(word)), []))
                for word in positive_words
            ),
            "sample_occurrences_by_word": {
                _normalize_word_surface(word): occurrences.get(
                    (str(bundle.get("bundle_id")), _normalize_word_surface(word)),
                    [],
                )[:10]
                for word in positive_words[:20]
            },
            "valid_for_strict_and_benchmark": False,
            "reason": "uncurated_multi_material_surface_feature_overlap",
        }
        invalid_records.append(record)

    return {
        "version": "v34",
        "validation_mode": "source_intersection_material_same_instance_audit",
        "material_glosses": sorted(SOURCE_NORM_MATERIAL_COMPONENT_GLOSSES),
        "checked_bundle_count": sum(
            1
            for bundle in bundle_bank
            if str(bundle.get("category_operator") or "") == "and"
            and str((bundle.get("source") or {}).get("bundle_builder") or "") == "source_norms_intersection_v1"
        ),
        "invalid_record_count": len(invalid_records),
        "passed": not invalid_records,
        "records": records,
        "invalid_records": invalid_records,
    }


def write_source_intersection_material_same_instance_audit(output_path: Path, report: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_source_intersection_material_false_evidence_audit(
    rows: Sequence[Dict[str, Any]],
    bundle_bank: Sequence[Dict[str, Any]],
    validator: SemanticValidator | None = None,
) -> Dict[str, Any]:
    validator = validator or get_default_semantic_validator()
    occurrences = _near_miss_occurrences(rows)
    records: List[Dict[str, Any]] = []
    invalid_records: List[Dict[str, Any]] = []
    checked_near_miss_words = 0

    for bundle in bundle_bank:
        bundle = _normalize_bundle_word_fields(bundle)
        source = bundle.get("source") or {}
        if str(bundle.get("category_operator") or "") != "and":
            continue
        if str(source.get("bundle_builder") or "") != "source_norms_intersection_v1":
            continue
        component_glosses = _bundle_component_glosses(bundle)
        near_misses = source.get("near_miss_words_by_component")
        if not isinstance(near_misses, list):
            continue
        bundle_id = str(bundle.get("bundle_id") or "")
        for failed_component_index, words in enumerate(near_misses):
            if failed_component_index >= len(component_glosses):
                continue
            failed_gloss = component_glosses[failed_component_index]
            if not _source_norm_gloss_is_material_component(failed_gloss):
                continue
            for word in sorted(set(_normalize_word_list(words))):
                checked_near_miss_words += 1
                evidence = _semantic_component_evidence_profile_for_bundle(word, bundle, validator)
                entry = (
                    evidence[failed_component_index]
                    if failed_component_index < len(evidence)
                    else {"value": None, "basis": "missing"}
                )
                reasons = _material_false_evidence_disqualification_reasons(
                    word,
                    failed_gloss,
                    entry,
                    validator,
                )
                record = {
                    "bundle_id": bundle_id,
                    "logic_condition": bundle.get("logic_condition"),
                    "difficulty": bundle.get("difficulty"),
                    "rule_gloss": bundle.get("rule_gloss"),
                    "component_glosses": component_glosses,
                    "word": _normalize_word_surface(word),
                    "role": "source_derived_and_component_near_miss",
                    "failed_component_index": failed_component_index,
                    "failed_component_gloss": _source_norm_clean_gloss(failed_gloss),
                    "failed_component_evidence": entry,
                    "word_concepts": sorted(validator._word_concepts(word)),
                    "word_domains": sorted(validator._word_domains(word)),
                    "occurrence_count": len(occurrences.get((bundle_id, _normalize_word_surface(word)), [])),
                    "sample_occurrences": occurrences.get((bundle_id, _normalize_word_surface(word)), [])[:20],
                    "valid_material_false_evidence": not reasons,
                    "reasons": reasons,
                }
                if reasons:
                    invalid_records.append(record)
                else:
                    records.append(record)

    return {
        "version": "v34",
        "validation_mode": "source_intersection_material_false_evidence_audit",
        "material_glosses": sorted(SOURCE_NORM_MATERIAL_COMPONENT_GLOSSES),
        "checked_near_miss_words": checked_near_miss_words,
        "invalid_record_count": len(invalid_records),
        "passed": not invalid_records,
        "records": records,
        "invalid_records": invalid_records,
    }


def write_source_intersection_material_false_evidence_audit(output_path: Path, report: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_source_intersection_functional_false_evidence_audit(
    rows: Sequence[Dict[str, Any]],
    bundle_bank: Sequence[Dict[str, Any]],
    validator: SemanticValidator | None = None,
) -> Dict[str, Any]:
    validator = validator or get_default_semantic_validator()
    occurrences = _near_miss_occurrences(rows)
    records: List[Dict[str, Any]] = []
    invalid_records: List[Dict[str, Any]] = []
    checked_near_miss_words = 0

    for bundle in bundle_bank:
        bundle = _normalize_bundle_word_fields(bundle)
        source = bundle.get("source") or {}
        if str(bundle.get("category_operator") or "") != "and":
            continue
        if str(source.get("bundle_builder") or "") != "source_norms_intersection_v1":
            continue
        component_glosses = _bundle_component_glosses(bundle)
        near_misses = source.get("near_miss_words_by_component")
        if not isinstance(near_misses, list):
            continue
        bundle_id = str(bundle.get("bundle_id") or "")
        for failed_component_index, words in enumerate(near_misses):
            if failed_component_index >= len(component_glosses):
                continue
            failed_gloss = component_glosses[failed_component_index]
            if not _source_norm_gloss_is_transportation_function(failed_gloss):
                continue
            for word in sorted(set(_normalize_word_list(words))):
                checked_near_miss_words += 1
                evidence = _semantic_component_evidence_profile_for_bundle(word, bundle, validator)
                entry = (
                    evidence[failed_component_index]
                    if failed_component_index < len(evidence)
                    else {"value": None, "basis": "missing"}
                )
                if entry.get("value") is not False:
                    reasons = [f"transportation_failure_not_false:{entry.get('basis') or 'missing'}"]
                else:
                    reasons = _functional_false_evidence_disqualification_reasons(
                        word,
                        failed_gloss,
                        entry,
                        validator,
                    )
                record = {
                    "bundle_id": bundle_id,
                    "logic_condition": bundle.get("logic_condition"),
                    "difficulty": bundle.get("difficulty"),
                    "rule_gloss": bundle.get("rule_gloss"),
                    "component_glosses": component_glosses,
                    "word": _normalize_word_surface(word),
                    "role": "source_derived_and_component_near_miss",
                    "failed_component_index": failed_component_index,
                    "failed_component_gloss": _source_norm_clean_gloss(failed_gloss),
                    "failed_component_evidence": entry,
                    "word_concepts": sorted(validator._word_concepts(word)),
                    "word_domains": sorted(validator._word_domains(word)),
                    "occurrence_count": len(occurrences.get((bundle_id, _normalize_word_surface(word)), [])),
                    "sample_occurrences": occurrences.get((bundle_id, _normalize_word_surface(word)), [])[:20],
                    "valid_functional_false_evidence": not reasons,
                    "reasons": reasons,
                }
                if reasons:
                    invalid_records.append(record)
                else:
                    records.append(record)

    return {
        "version": "v34",
        "validation_mode": "source_intersection_functional_false_evidence_audit",
        "functional_glosses": sorted(SOURCE_NORM_TRANSPORTATION_FUNCTION_GLOSSES),
        "transportation_part_words": sorted(SOURCE_NORM_TRANSPORTATION_PART_WORDS),
        "checked_near_miss_words": checked_near_miss_words,
        "invalid_record_count": len(invalid_records),
        "passed": not invalid_records,
        "records": records,
        "invalid_records": invalid_records,
    }


def write_source_intersection_functional_false_evidence_audit(output_path: Path, report: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _rated_property_plausibility_issue_records(
    word: Any,
    component_glosses: Sequence[Any],
    evidence: Sequence[Dict[str, Any]],
    validator: SemanticValidator,
) -> List[Dict[str, Any]]:
    normalized_word = _normalize_word_surface(word)
    records: List[Dict[str, Any]] = []
    for index, entry in enumerate(evidence):
        if index >= len(component_glosses):
            continue
        clean_gloss = _source_norm_clean_gloss(component_glosses[index])
        if clean_gloss not in SOURCE_NORM_HIGH_RISK_RATED_PROPERTY_GLOSSES:
            continue
        issues = list(entry.get("plausibility_issues") or [])
        issues.extend(
            validator.rated_property_plausibility_issues(
                normalized_word,
                clean_gloss,
                entry.get("value"),
                basis=str(entry.get("basis") or ""),
            )
        )
        issues = sorted(set(issues))
        if not issues:
            continue
        records.append(
            {
                "component_index": index,
                "component_gloss": clean_gloss,
                "value": entry.get("value"),
                "basis": entry.get("basis"),
                "plausibility_issues": issues,
                "evidence": entry,
            }
        )
    return records


def _near_miss_occurrences(
    rows: Sequence[Dict[str, Any]],
) -> Dict[tuple[str, str], List[Dict[str, Any]]]:
    occurrences: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("corruption_type") or "") != "component_near_miss":
            continue
        target_label = str(row.get("target_label") or "")
        bundle_id = str((row.get("bundle_ids") or {}).get(target_label) or "")
        if not bundle_id:
            continue
        for group_index, group in enumerate(row.get("candidate_groups") or []):
            if str(group.get("display_label") or "") != target_label:
                continue
            if str(group.get("gold_label") or "") != OUT_OF_CATEGORY_LABEL:
                continue
            words = _normalize_word_list(group.get("words") or [])
            if not words:
                continue
            occurrences[(bundle_id, words[0])].append(
                {
                    "row_id": row.get("id"),
                    "split": row.get("split"),
                    "label": target_label,
                    "role": "candidate_component_near_miss",
                    "candidate_group_index": group_index,
                    "satisfied_component_indices": group.get("satisfied_component_indices") or [],
                    "failed_component_indices": group.get("failed_component_indices") or [],
                    "unknown_component_indices": group.get("unknown_component_indices") or [],
                }
            )
    return occurrences


def build_high_risk_rated_property_plausibility_audit(
    rows: Sequence[Dict[str, Any]],
    bundle_bank: Sequence[Dict[str, Any]],
    validator: SemanticValidator | None = None,
) -> Dict[str, Any]:
    validator = validator or get_default_semantic_validator()
    positive_occurrences = _source_intersection_positive_occurrences(rows)
    near_miss_occurrences = _near_miss_occurrences(rows)
    invalid_records: List[Dict[str, Any]] = []
    accepted_records_sample: List[Dict[str, Any]] = []
    accepted_component_counts: Counter[str] = Counter()
    checked_positive_words = 0
    checked_near_miss_words = 0

    def record_accepted_high_risk_components(
        *,
        bundle: Dict[str, Any],
        bundle_id: str,
        word: str,
        role: str,
        component_glosses: Sequence[Any],
        evidence: Sequence[Dict[str, Any]],
        occurrences: List[Dict[str, Any]],
        near_miss_component_index: int | None = None,
    ) -> None:
        normalized_word = _normalize_word_surface(word)
        for index, entry in enumerate(evidence):
            if index >= len(component_glosses) or entry.get("value") is not True:
                continue
            clean_gloss = _source_norm_clean_gloss(component_glosses[index])
            if clean_gloss not in SOURCE_NORM_HIGH_RISK_RATED_PROPERTY_GLOSSES:
                continue
            accepted_component_counts[f"{role}:{clean_gloss}"] += 1
            if len(accepted_records_sample) >= 500:
                continue
            record = {
                "bundle_id": bundle_id,
                "logic_condition": bundle.get("logic_condition"),
                "difficulty": bundle.get("difficulty"),
                "rule_gloss": bundle.get("rule_gloss"),
                "component_glosses": list(component_glosses),
                "word": normalized_word,
                "role": role,
                "component_index": index,
                "component_gloss": clean_gloss,
                "basis": entry.get("basis"),
                "word_concepts": sorted(validator._word_concepts(normalized_word)),
                "word_domains": sorted(validator._word_domains(normalized_word)),
                "accepted": True,
                "occurrence_count": len(occurrences),
                "sample_occurrences": occurrences[:10],
            }
            if near_miss_component_index is not None:
                record["near_miss_component_index"] = near_miss_component_index
            accepted_records_sample.append(record)

    for bundle in bundle_bank:
        bundle = _normalize_bundle_word_fields(bundle)
        source = bundle.get("source") or {}
        if str(bundle.get("category_operator") or "") != "and":
            continue
        if str(source.get("bundle_builder") or "") != "source_norms_intersection_v1":
            continue
        bundle_id = str(bundle.get("bundle_id") or "")
        component_glosses = _bundle_component_glosses(bundle)

        for word in sorted(set(_flatten_words(bundle))):
            checked_positive_words += 1
            evidence = _semantic_component_evidence_profile_for_bundle(word, bundle, validator)
            issues = _rated_property_plausibility_issue_records(word, component_glosses, evidence, validator)
            occurrences = positive_occurrences.get((bundle_id, _normalize_word_surface(word)), [])
            record_accepted_high_risk_components(
                bundle=bundle,
                bundle_id=bundle_id,
                word=word,
                role="source_derived_and_positive",
                component_glosses=component_glosses,
                evidence=evidence,
                occurrences=occurrences,
            )
            if issues:
                invalid_records.append(
                    {
                        "bundle_id": bundle_id,
                        "logic_condition": bundle.get("logic_condition"),
                        "difficulty": bundle.get("difficulty"),
                        "rule_gloss": bundle.get("rule_gloss"),
                        "component_glosses": component_glosses,
                        "word": _normalize_word_surface(word),
                        "role": "source_derived_and_positive",
                        "issues": issues,
                        "word_concepts": sorted(validator._word_concepts(word)),
                        "word_domains": sorted(validator._word_domains(word)),
                        "occurrence_count": len(occurrences),
                        "sample_occurrences": occurrences[:20],
                    }
                )

        near_misses = source.get("near_miss_words_by_component")
        if not isinstance(near_misses, list):
            continue
        for failed_component_index, words in enumerate(near_misses):
            for word in sorted(set(_normalize_word_list(words))):
                checked_near_miss_words += 1
                evidence = _semantic_component_evidence_profile_for_bundle(word, bundle, validator)
                issues = _rated_property_plausibility_issue_records(word, component_glosses, evidence, validator)
                occurrences = near_miss_occurrences.get((bundle_id, _normalize_word_surface(word)), [])
                record_accepted_high_risk_components(
                    bundle=bundle,
                    bundle_id=bundle_id,
                    word=word,
                    role="source_derived_and_component_near_miss",
                    component_glosses=component_glosses,
                    evidence=evidence,
                    occurrences=occurrences,
                    near_miss_component_index=failed_component_index,
                )
                if issues:
                    invalid_records.append(
                        {
                            "bundle_id": bundle_id,
                            "logic_condition": bundle.get("logic_condition"),
                            "difficulty": bundle.get("difficulty"),
                            "rule_gloss": bundle.get("rule_gloss"),
                            "component_glosses": component_glosses,
                            "word": _normalize_word_surface(word),
                            "role": "source_derived_and_component_near_miss",
                            "near_miss_component_index": failed_component_index,
                            "issues": issues,
                            "word_concepts": sorted(validator._word_concepts(word)),
                            "word_domains": sorted(validator._word_domains(word)),
                            "occurrence_count": len(occurrences),
                            "sample_occurrences": occurrences[:20],
                        }
                    )

    reason_counts = Counter(
        issue
        for record in invalid_records
        for issue_blob in record.get("issues", [])
        for issue in issue_blob.get("plausibility_issues", [])
    )
    return {
        "version": "v34",
        "validation_mode": "high_risk_rated_property_plausibility_audit",
        "checked_positive_words": checked_positive_words,
        "checked_near_miss_words": checked_near_miss_words,
        "invalid_record_count": len(invalid_records),
        "passed": not invalid_records,
        "reason_counts": dict(sorted(reason_counts.items())),
        "accepted_component_counts": dict(sorted(accepted_component_counts.items())),
        "accepted_records_sample": accepted_records_sample,
        "invalid_records": invalid_records,
    }


def write_high_risk_rated_property_plausibility_audit(output_path: Path, report: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_both_d3_near_miss_semantic_audit(
    rows: Sequence[Dict[str, Any]],
    bundle_bank: Sequence[Dict[str, Any]],
    validator: SemanticValidator | None = None,
) -> Dict[str, Any]:
    validator = validator or get_default_semantic_validator()
    bundle_by_id = {
        str(bundle.get("bundle_id") or ""): _normalize_bundle_word_fields(bundle)
        for bundle in bundle_bank
        if str(bundle.get("bundle_id") or "")
    }
    records_by_key: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        if row.get("design_cell") != "both/d3" or row.get("corruption_type") != "component_near_miss":
            continue
        target_label = str(row.get("target_label") or "")
        bundle_id = str((row.get("bundle_ids") or {}).get(target_label) or "")
        bundle = bundle_by_id.get(bundle_id)
        if not bundle:
            continue
        target_groups = [
            group
            for group in row.get("candidate_groups") or []
            if str(group.get("display_label") or "") == target_label
            and str(group.get("gold_label") or "") == OUT_OF_CATEGORY_LABEL
        ]
        for group in target_groups:
            words = _normalize_word_list(group.get("words") or [])
            if not words:
                continue
            word = words[0]
            component_glosses = _bundle_component_glosses(bundle)
            evidence = _semantic_component_evidence_profile_for_bundle(word, bundle, validator)
            values = [entry.get("value") for entry in evidence]
            satisfied = [index for index, value in enumerate(values) if value is True]
            failed = [index for index, value in enumerate(values) if value is False]
            unknown = [index for index, value in enumerate(values) if value is None]
            reasons = _strict_near_miss_semantic_disqualification_reasons(
                word,
                component_glosses,
                evidence,
                validator,
            )
            key = (bundle_id, word, tuple(satisfied), tuple(failed), tuple(unknown))
            record = records_by_key.get(key)
            if record is None:
                record = {
                    "bundle_id": bundle_id,
                    "rule_gloss": bundle.get("rule_gloss"),
                    "component_glosses": component_glosses,
                    "word": word,
                    "row_ids": [],
                    "target_labels": [],
                    "satisfied_component_indices": satisfied,
                    "failed_component_indices": failed,
                    "unknown_component_indices": unknown,
                    "semantic_component_evidence": evidence,
                    "word_concepts": sorted(validator._word_concepts(word)),
                    "word_domains": sorted(validator._word_domains(word)),
                    "valid_for_strict_near_miss": (
                        values.count(True) == len(values) - 1
                        and values.count(False) == 1
                        and values.count(None) == 0
                        and not reasons
                    ),
                    "semantic_disqualification_reasons": reasons,
                }
                records_by_key[key] = record
            record["row_ids"].append(row.get("id"))
            record["target_labels"].append(target_label)

    records = list(records_by_key.values())
    invalid_records = [
        record for record in records if not bool(record.get("valid_for_strict_near_miss"))
    ]
    reason_counts = Counter(
        reason
        for record in invalid_records
        for reason in record.get("semantic_disqualification_reasons", [])
    )
    return {
        "version": "v34",
        "design_cell": "both/d3",
        "checked_rows": sum(
            1
            for row in rows
            if row.get("design_cell") == "both/d3" and row.get("corruption_type") == "component_near_miss"
        ),
        "checked_unique_patterns": len(records),
        "invalid_pattern_count": len(invalid_records),
        "passed": not invalid_records,
        "reason_counts": dict(sorted(reason_counts.items())),
        "records": records,
    }


def write_both_d3_near_miss_semantic_audit(output_path: Path, report: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_normalize_row_word_fields(row)))
            handle.write("\n")


def _extract_taxonomy_payload_from_xml(xml_path: Path) -> Any:
    raw = xml_path.read_text()
    target_name = "game_with_knowledge_taxonomy.json"

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        root = None

    if root is not None:
        for elem in root.iter():
            attr_blob = " ".join(str(value) for value in elem.attrib.values())
            text_blob = elem.text or ""
            if target_name not in attr_blob and target_name not in text_blob:
                continue

            for candidate in elem.iter():
                candidate_text = (candidate.text or "").strip()
                if not candidate_text or candidate_text == target_name:
                    continue
                try:
                    return json.loads(candidate_text)
                except json.JSONDecodeError:
                    continue

    regexes = [
        re.compile(
            rf"{re.escape(target_name)}.*?<!\[CDATA\[(.*?)\]\]>",
            re.DOTALL,
        ),
        re.compile(
            rf"{re.escape(target_name)}.*?<content>(.*?)</content>",
            re.DOTALL,
        ),
    ]
    for pattern in regexes:
        match = pattern.search(raw)
        if not match:
            continue
        candidate_text = match.group(1).strip()
        try:
            return json.loads(candidate_text)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not locate valid JSON for {target_name} inside {xml_path}.")


def _normalize_source_seed_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "reasoning_annotation" in payload[0]:
        categories = []
        for game_index, row in enumerate(payload):
            if not isinstance(row, dict):
                continue
            annotations = row.get("reasoning_annotation") or []
            if not isinstance(annotations, list):
                continue
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue

                seed_category_name = annotation.get("Categories")
                raw_words = annotation.get("Words in Category")
                based_on = str(annotation.get("Based On") or "").strip().lower()
                raw_knowledge_type = str(annotation.get("Knowledge Type") or "").strip().lower()
                complexity = annotation.get("Complexity")
                if not seed_category_name or raw_words is None or not complexity:
                    continue

                if isinstance(raw_words, str):
                    try:
                        parsed_words = ast.literal_eval(raw_words)
                    except (SyntaxError, ValueError):
                        parsed_words = []
                else:
                    parsed_words = raw_words

                if not isinstance(parsed_words, list):
                    continue

                knowledge_type = "semantic"
                if "multiword" in raw_knowledge_type:
                    knowledge_type = "multiword_expression"
                elif based_on == "word form" or "orthograph" in raw_knowledge_type or "phonolog" in raw_knowledge_type:
                    knowledge_type = "word_meaning_plus_word_form"
                elif based_on == "word meaning + word form":
                    knowledge_type = "word_meaning_plus_word_form"
                elif "associative relations" in raw_knowledge_type:
                    knowledge_type = "associative_relations"
                elif "encyclopedic" in raw_knowledge_type:
                    knowledge_type = "encyclopedic"
                elif "semantic relations" in raw_knowledge_type:
                    knowledge_type = "semantic"

                categories.append(
                    {
                        "seed_category_name": str(seed_category_name),
                        "source_words": _normalize_word_list(parsed_words),
                        "knowledge_type": knowledge_type,
                        "complexity": str(complexity).strip().lower(),
                        "based_on": based_on,
                        "knowledge_type_raw": raw_knowledge_type,
                        "original_game_index": game_index,
                        "original_board_id": row.get("game_id") or row.get("board_id"),
                        "original_nyt_difficulty": row.get("nyt_difficulty") or row.get("color"),
                        "source_repo": "lexical_category_inference_source",
                    }
                )
        return {
            "source_repo": "lexical_category_inference_source",
            "notes": "Simplified source extraction for lexical_category_inference from game_with_knowledge_taxonomy.json.",
            "categories": categories,
        }

    if isinstance(payload, dict) and isinstance(payload.get("categories"), list):
        categories = []
        for row in payload["categories"]:
            if not isinstance(row, dict):
                continue
            categories.append(
                {
                    "seed_category_name": row.get("seed_category_name") or row.get("name"),
                    "source_words": _normalize_word_list(row.get("source_words") or row.get("words") or []),
                    "knowledge_type": row.get("knowledge_type"),
                    "complexity": row.get("complexity"),
                    "original_game_index": row.get("original_game_index"),
                    "original_board_id": row.get("original_board_id"),
                    "original_nyt_difficulty": row.get("original_nyt_difficulty"),
                    "source_repo": "lexical_category_inference_source",
                }
            )
        return {
            "source_repo": "lexical_category_inference_source",
            "notes": payload.get("notes") or "Simplified source extraction for lexical_category_inference.",
            "categories": categories,
        }

    if isinstance(payload, list):
        categories = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            if not (row.get("knowledge_type") and row.get("complexity")):
                continue
            name = row.get("seed_category_name") or row.get("category_name") or row.get("name")
            words = row.get("source_words") or row.get("words") or row.get("members") or row.get("group")
            if not name or not isinstance(words, list):
                continue
            categories.append(
                {
                    "seed_category_name": name,
                    "source_words": _normalize_word_list(words),
                    "knowledge_type": row.get("knowledge_type"),
                    "complexity": row.get("complexity"),
                    "original_game_index": row.get("game_index") or row.get("original_game_index"),
                    "original_board_id": row.get("board_id") or row.get("original_board_id"),
                    "original_nyt_difficulty": row.get("nyt_difficulty") or row.get("original_nyt_difficulty"),
                    "source_repo": "lexical_category_inference_source",
                }
            )
        return {
            "source_repo": "lexical_category_inference_source",
            "notes": "Simplified source extraction for lexical_category_inference.",
            "categories": categories,
        }

    raise ValueError("Unsupported taxonomy payload format for source seed extraction.")


def write_source_seed_categories_from_xml(
    xml_path: str | Path,
    output_path: str | Path = DEFAULT_SOURCE_SEED_PATH,
) -> Path:
    source_path = Path(xml_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source taxonomy file not found: {source_path}")

    if source_path.suffix.lower() == ".json":
        with source_path.open("r") as handle:
            payload = json.load(handle)
    else:
        payload = _extract_taxonomy_payload_from_xml(source_path)
    normalized = _normalize_source_seed_payload(payload)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        json.dump(normalized, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination


def _resolve_default_source_taxonomy_path() -> Path | None:
    candidates = [
        DEFAULT_LOCAL_SOURCE_REPO / "game_with_knowledge_taxonomy.json",
        DEFAULT_LOCAL_SOURCE_REPO / "repomix-output.xml",
        DEFAULT_REPOMIX_XML,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Lexical Category Inference datasets.")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "easy", "medium", "hard", "d1", "d2", "d3"],
    )
    parser.add_argument(
        "--logic-condition",
        type=str,
        default="all",
        choices=["shared", "either", "both", "all"],
        help="Logic condition to generate. d1 always resolves to shared/d1.",
    )
    parser.add_argument("--train_samples", type=int, default=6000)
    parser.add_argument("--test_samples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DATA_ROOT),
        help="Output root directory. Defaults to <repo>/data/lexical_category_inference.",
    )
    parser.add_argument(
        "--bundle_bank",
        type=str,
        default=None,
        help="Optional bundle bank path. Defaults to bundle_bank_v1.json, falling back to bundle_bank_smoke.json.",
    )
    parser.add_argument(
        "--extract_source_xml",
        type=str,
        default=None,
        help="Optional repomix XML path or direct taxonomy JSON path for one-time source extraction.",
    )
    parser.add_argument(
        "--extract_source_only",
        action="store_true",
        help="Only extract source_seed_categories.json from --extract_source_xml and exit.",
    )
    parser.add_argument(
        "--rebuild_bundle_bank",
        action="store_true",
        help="Rebuild bundle_bank_v1.json from sourced concept-property norm datasets.",
    )
    parser.add_argument(
        "--rebuild_bundle_bank_only",
        action="store_true",
        help="Rebuild sourced bundle_bank_v1.json and exit without generating dataset splits.",
    )
    parser.add_argument(
        "--skip_difficulty_lint",
        action="store_true",
        help="Skip the cross-cell dataset lint step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    extract_xml = args.extract_source_xml
    if extract_xml:
        destination = write_source_seed_categories_from_xml(extract_xml, output_root / "source_seed_categories.json")
        print(f"Wrote {destination}")
        if args.extract_source_only:
            return
    elif args.extract_source_only:
        default_source_path = _resolve_default_source_taxonomy_path()
        if default_source_path is not None:
            destination = write_source_seed_categories_from_xml(default_source_path, output_root / "source_seed_categories.json")
            print(f"Wrote {destination}")
            return
        raise FileNotFoundError(
            "--extract_source_only was requested, but no XML path was provided and "
            "no default local taxonomy source could be found."
        )

    if args.rebuild_bundle_bank or args.rebuild_bundle_bank_only:
        bundle_bank_path = output_root / "bundle_bank_v1.json"
        source_seed_path = output_root / "source_seed_categories.json"
        rebuilt_bank = build_bundle_bank_from_source_seeds(
            source_seed_path=source_seed_path,
            existing_bundle_bank_path=bundle_bank_path,
        )
        write_bundle_bank(bundle_bank_path, rebuilt_bank)
        print(f"Wrote {bundle_bank_path}")
        semantic_report = build_semantic_validation_report(rebuilt_bank)
        write_semantic_validation_report(DEFAULT_SEMANTIC_VALIDATION_ARTIFACT, semantic_report)
        print(f"Wrote {DEFAULT_SEMANTIC_VALIDATION_ARTIFACT}")
        if not semantic_report.get("passed"):
            raise RuntimeError(
                "lexical_category_inference semantic bundle validation failed. "
                f"Invalid records: {semantic_report.get('invalid_record_count')}"
            )
        if args.rebuild_bundle_bank_only:
            return
    else:
        semantic_report = build_semantic_validation_report(load_bundle_bank(args.bundle_bank))
        write_semantic_validation_report(DEFAULT_SEMANTIC_VALIDATION_ARTIFACT, semantic_report)
        print(f"Wrote {DEFAULT_SEMANTIC_VALIDATION_ARTIFACT}")
        if not semantic_report.get("passed"):
            raise RuntimeError(
                "lexical_category_inference semantic bundle validation failed. "
                f"Invalid records: {semantic_report.get('invalid_record_count')}"
            )

    design_cells = iter_requested_design_cells(args.difficulty, args.logic_condition)
    generated_test_rows: Dict[str, List[Dict[str, Any]]] = {}
    generated_all_rows: List[Dict[str, Any]] = []

    for cell in design_cells:
        group = str(cell["logic_condition"])
        difficulty = str(cell["difficulty"])
        seed_offset = 1000 * (REAL_DESIGN_CELLS.index((group, difficulty)) + 1)
        train_rows, test_rows = generate_dataset_for_difficulty(
            difficulty=difficulty,
            logic_condition=group,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            seed=args.seed + seed_offset,
            bundle_bank_path=args.bundle_bank,
        )
        difficulty_dir = output_root / group / difficulty
        train_path = difficulty_dir / "train" / f"lexical_category_inference_{group}_{difficulty}_train_n{len(train_rows)}.jsonl"
        test_path = difficulty_dir / "test" / f"lexical_category_inference_{group}_{difficulty}_test_n{len(test_rows)}.jsonl"
        write_jsonl(train_path, train_rows)
        write_jsonl(test_path, test_rows)
        generated_test_rows[design_cell_key(group, difficulty)] = test_rows
        generated_all_rows.extend(train_rows)
        generated_all_rows.extend(test_rows)
        print(f"Wrote {train_path}")
        print(f"Wrote {test_path}")

    if any(row.get("design_cell") == "both/d3" for row in generated_all_rows):
        near_miss_audit = build_both_d3_near_miss_semantic_audit(
            generated_all_rows,
            load_bundle_bank(args.bundle_bank),
        )
        write_both_d3_near_miss_semantic_audit(
            DEFAULT_BOTH_D3_NEAR_MISS_AUDIT_ARTIFACT,
            near_miss_audit,
        )
        print(f"Wrote {DEFAULT_BOTH_D3_NEAR_MISS_AUDIT_ARTIFACT}")
        if not near_miss_audit.get("passed"):
            raise RuntimeError(
                "lexical_category_inference both/d3 strict near-miss semantic audit failed. "
                f"Invalid patterns: {near_miss_audit.get('invalid_pattern_count')}"
            )

    if any(row.get("logic_condition") == "both" for row in generated_all_rows):
        positive_sense_audit = build_source_intersection_positive_sense_audit(
            generated_all_rows,
            load_bundle_bank(args.bundle_bank),
        )
        write_source_intersection_positive_sense_audit(
            DEFAULT_SOURCE_INTERSECTION_POSITIVE_SENSE_AUDIT_ARTIFACT,
            positive_sense_audit,
        )
        print(f"Wrote {DEFAULT_SOURCE_INTERSECTION_POSITIVE_SENSE_AUDIT_ARTIFACT}")
        if not positive_sense_audit.get("passed"):
            raise RuntimeError(
                "lexical_category_inference source-derived AND positive sense audit failed. "
                f"Invalid records: {positive_sense_audit.get('invalid_record_count')}"
            )
        material_same_instance_audit = build_source_intersection_material_same_instance_audit(
            generated_all_rows,
            load_bundle_bank(args.bundle_bank),
        )
        write_source_intersection_material_same_instance_audit(
            DEFAULT_SOURCE_INTERSECTION_MATERIAL_AUDIT_ARTIFACT,
            material_same_instance_audit,
        )
        print(f"Wrote {DEFAULT_SOURCE_INTERSECTION_MATERIAL_AUDIT_ARTIFACT}")
        if not material_same_instance_audit.get("passed"):
            raise RuntimeError(
                "lexical_category_inference source-derived AND material same-instance audit failed. "
                f"Invalid records: {material_same_instance_audit.get('invalid_record_count')}"
            )
        material_false_evidence_audit = build_source_intersection_material_false_evidence_audit(
            generated_all_rows,
            load_bundle_bank(args.bundle_bank),
        )
        write_source_intersection_material_false_evidence_audit(
            DEFAULT_SOURCE_INTERSECTION_MATERIAL_FALSE_EVIDENCE_AUDIT_ARTIFACT,
            material_false_evidence_audit,
        )
        print(f"Wrote {DEFAULT_SOURCE_INTERSECTION_MATERIAL_FALSE_EVIDENCE_AUDIT_ARTIFACT}")
        if not material_false_evidence_audit.get("passed"):
            raise RuntimeError(
                "lexical_category_inference source-derived AND material false-evidence audit failed. "
                f"Invalid records: {material_false_evidence_audit.get('invalid_record_count')}"
            )
        functional_false_evidence_audit = build_source_intersection_functional_false_evidence_audit(
            generated_all_rows,
            load_bundle_bank(args.bundle_bank),
        )
        write_source_intersection_functional_false_evidence_audit(
            DEFAULT_SOURCE_INTERSECTION_FUNCTIONAL_FALSE_EVIDENCE_AUDIT_ARTIFACT,
            functional_false_evidence_audit,
        )
        print(f"Wrote {DEFAULT_SOURCE_INTERSECTION_FUNCTIONAL_FALSE_EVIDENCE_AUDIT_ARTIFACT}")
        if not functional_false_evidence_audit.get("passed"):
            raise RuntimeError(
                "lexical_category_inference source-derived AND functional false-evidence audit failed. "
                f"Invalid records: {functional_false_evidence_audit.get('invalid_record_count')}"
            )
        high_risk_property_audit = build_high_risk_rated_property_plausibility_audit(
            generated_all_rows,
            load_bundle_bank(args.bundle_bank),
        )
        write_high_risk_rated_property_plausibility_audit(
            DEFAULT_HIGH_RISK_RATED_PROPERTY_AUDIT_ARTIFACT,
            high_risk_property_audit,
        )
        print(f"Wrote {DEFAULT_HIGH_RISK_RATED_PROPERTY_AUDIT_ARTIFACT}")
        if not high_risk_property_audit.get("passed"):
            raise RuntimeError(
                "lexical_category_inference high-risk rated-property plausibility audit failed. "
                f"Invalid records: {high_risk_property_audit.get('invalid_record_count')}"
            )

    if not args.skip_difficulty_lint and set(generated_test_rows) == {
        design_cell_key(group, difficulty)
        for group, difficulty in REAL_DESIGN_CELLS
    }:
        lint_report = build_dataset_lint_report(generated_test_rows, output_root=output_root)
        lint_path = output_root / "lexical_category_inference_dataset_lint.json"
        write_dataset_lint_report(lint_path, lint_report)
        print(f"Wrote {lint_path}")
        if not lint_report.get("passed"):
            failed_features = []
            for section_name in (
                "design_cell_checks",
                "candidate_structure_checks",
                "category_diversity_checks",
                "category_definition_text_checks",
                "concept_quality_checks",
                "positive_coverage_checks",
                "component_coverage_checks",
                "semantic_negative_validity_checks",
                "surface_form_checks",
                "monotonicity_checks",
            ):
                failed_features.extend(
                    f"{section_name}:{feature}"
                    for feature, check in lint_report.get(section_name, {}).items()
                    if not check.get("passed")
                )
            if not lint_report.get("file_checks", {}).get("passed"):
                failed_features.append("file_checks")
            raise RuntimeError(
                "lexical_category_inference logic-condition dataset lint failed. "
                f"Failed checks: {', '.join(failed_features)}"
            )


if __name__ == "__main__":
    main()
