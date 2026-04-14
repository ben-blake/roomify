"""Tests for src/roomify/promptBuilder.py — Phase 2.

TDD: written before implementation. Run with:  pytest tests/testPromptBuilder.py -v
"""

from __future__ import annotations

import pytest

from roomify.promptBuilder import RoomSpec, buildPrompt

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

FULL_SPEC = RoomSpec(
    id="test_bedroom",
    roomType="bedroom",
    size="10x12 ft",
    style="scandinavian",
    furniture=["queen bed", "nightstand", "reading chair"],
    lighting="natural light from east window",
    mood="cozy, airy",
)

MINIMAL_SPEC = RoomSpec(
    id="test_minimal",
    roomType="kitchen",
    # all optional fields left as defaults
)


# ---------------------------------------------------------------------------
# Return type and basic contract
# ---------------------------------------------------------------------------


def test_returns_tuple_of_two_strings():
    positive, negative = buildPrompt(FULL_SPEC, "minimal")
    assert isinstance(positive, str)
    assert isinstance(negative, str)


def test_positive_prompt_nonempty_for_all_strategies():
    for strategy in ("minimal", "descriptive", "styleAnchored"):
        positive, _ = buildPrompt(FULL_SPEC, strategy)
        assert positive.strip(), f"positive prompt empty for strategy '{strategy}'"


def test_negative_prompt_nonempty_for_all_strategies():
    for strategy in ("minimal", "descriptive", "styleAnchored"):
        _, negative = buildPrompt(FULL_SPEC, strategy)
        assert negative.strip(), f"negative prompt empty for strategy '{strategy}'"


# ---------------------------------------------------------------------------
# Strategies produce distinct outputs
# ---------------------------------------------------------------------------


def test_strategies_produce_distinct_positive_prompts():
    results = {s: buildPrompt(FULL_SPEC, s)[0] for s in ("minimal", "descriptive", "styleAnchored")}
    # all three must be different from each other
    prompts = list(results.values())
    assert prompts[0] != prompts[1], "minimal == descriptive"
    assert prompts[1] != prompts[2], "descriptive == styleAnchored"
    assert prompts[0] != prompts[2], "minimal == styleAnchored"


def test_all_strategies_share_same_negative_prompt():
    negatives = [buildPrompt(FULL_SPEC, s)[1] for s in ("minimal", "descriptive", "styleAnchored")]
    assert negatives[0] == negatives[1] == negatives[2], (
        "negative prompt should be the same across all strategies"
    )


# ---------------------------------------------------------------------------
# Template interpolation — roomType always appears
# ---------------------------------------------------------------------------


def test_minimal_contains_roomtype():
    positive, _ = buildPrompt(FULL_SPEC, "minimal")
    assert "bedroom" in positive.lower()


def test_minimal_contains_style():
    positive, _ = buildPrompt(FULL_SPEC, "minimal")
    assert "scandinavian" in positive.lower()


def test_descriptive_contains_roomtype():
    positive, _ = buildPrompt(FULL_SPEC, "descriptive")
    assert "bedroom" in positive.lower()


def test_descriptive_contains_furniture():
    positive, _ = buildPrompt(FULL_SPEC, "descriptive")
    assert "queen bed" in positive.lower()


def test_styleanchored_contains_style_keywords():
    positive, _ = buildPrompt(FULL_SPEC, "styleAnchored")
    # styleAnchored template injects extra editorial keywords
    assert any(kw in positive.lower() for kw in ("catalog", "editorial", "architectural"))


# ---------------------------------------------------------------------------
# Optional fields gracefully missing
# ---------------------------------------------------------------------------


def test_minimal_spec_no_exception():
    """buildPrompt must not raise when optional fields are empty."""
    positive, negative = buildPrompt(MINIMAL_SPEC, "minimal")
    assert positive.strip()
    assert negative.strip()


def test_descriptive_spec_no_exception_with_empty_furniture():
    positive, _ = buildPrompt(MINIMAL_SPEC, "descriptive")
    assert positive.strip()


def test_styleanchored_spec_no_exception_with_empty_fields():
    positive, _ = buildPrompt(MINIMAL_SPEC, "styleAnchored")
    assert positive.strip()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unknown_strategy_raises_valueerror():
    with pytest.raises(ValueError, match="unknown strategy"):
        buildPrompt(FULL_SPEC, "nonexistent")


def test_error_message_includes_strategy_name():
    with pytest.raises(ValueError, match="nonexistent"):
        buildPrompt(FULL_SPEC, "nonexistent")


# ---------------------------------------------------------------------------
# RoomSpec validation
# ---------------------------------------------------------------------------


def test_roomspec_requires_id():
    with pytest.raises(TypeError):
        RoomSpec()  # id is required positional arg


def test_roomspec_requires_roomtype():
    with pytest.raises(TypeError):
        RoomSpec(id="x")  # roomType is required


def test_roomspec_furniture_defaults_to_empty_list():
    spec = RoomSpec(id="x", roomType="office")
    assert spec.furniture == []


def test_roomspec_furniture_list_is_independent():
    """Two RoomSpec instances must not share the same default list."""
    spec1 = RoomSpec(id="a", roomType="office")
    spec2 = RoomSpec(id="b", roomType="kitchen")
    spec1.furniture.append("desk")
    assert spec2.furniture == [], "furniture default list is shared — use field(default_factory=list)"
