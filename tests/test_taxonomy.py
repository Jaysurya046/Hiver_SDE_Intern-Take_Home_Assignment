"""Tests for the intent taxonomy: completeness and consistency."""
from agent import TAXONOMY, INTENTS, KEYWORD_RULES, HIGH_RISK_INTENTS


EXPECTED_INTENTS = [
    "ORDER_STATUS", "DAMAGED_OR_WRONG", "REFUND", "RETURN_CANCEL",
    "PAYMENT", "ACCOUNT", "DELIVERY_ADDRESS", "APP_WEBSITE",
    "SERVICE_COMPLAINT", "REVIEW_CONTENT", "OTHER",
]


def test_taxonomy_has_11_intents():
    assert len(TAXONOMY) == 11


def test_taxonomy_keys_match_intents():
    assert list(TAXONOMY.keys()) == INTENTS


def test_all_expected_intents_present():
    assert set(INTENTS) == set(EXPECTED_INTENTS)


def test_keyword_rules_cover_non_other_intents():
    """Every intent except OTHER should have a keyword rule."""
    for intent in INTENTS:
        if intent == "OTHER":
            assert intent not in KEYWORD_RULES
        else:
            assert intent in KEYWORD_RULES, f"Missing keyword rule for {intent}"


def test_high_risk_intents_are_valid():
    assert HIGH_RISK_INTENTS.issubset(set(INTENTS)), \
        f"Invalid high-risk intents: {HIGH_RISK_INTENTS - set(INTENTS)}"


def test_high_risk_expected_set():
    assert HIGH_RISK_INTENTS == {"SERVICE_COMPLAINT", "ACCOUNT", "REFUND", "PAYMENT", "REVIEW_CONTENT"}


def test_taxonomy_descriptions_nonempty():
    for intent, desc in TAXONOMY.items():
        assert len(desc) > 10, f"Description too short for {intent}"
