"""Shared fixtures for the Hiver AmazonHelp test suite.

All fixtures are tiny deterministic data — no Ollama, no network, no full dataset.
"""
import sys
import os

import pytest
import pandas as pd

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.fixture
def repo_root():
    return os.path.abspath(REPO_ROOT)


@pytest.fixture
def golden_df(repo_root):
    return pd.read_csv(os.path.join(repo_root, "golden_set.csv"))


@pytest.fixture
def threads_df(repo_root):
    return pd.read_csv(os.path.join(repo_root, "data", "amazonhelp_threads.csv"), nrows=500)


@pytest.fixture
def intent_preds(repo_root):
    return pd.read_csv(os.path.join(repo_root, "results", "intent_preds.csv"))


@pytest.fixture
def route_preds(repo_root):
    return pd.read_csv(os.path.join(repo_root, "results", "route_preds.csv"))


@pytest.fixture
def drafts_df(repo_root):
    return pd.read_csv(os.path.join(repo_root, "results", "drafts_50.csv"))


@pytest.fixture
def judge_scores(repo_root):
    return pd.read_csv(os.path.join(repo_root, "results", "judge_scores.csv"))


@pytest.fixture
def summary(repo_root):
    import json
    with open(os.path.join(repo_root, "results", "summary.json")) as f:
        return json.load(f)


@pytest.fixture
def tiny_texts():
    """Small fixed texts for deterministic classification tests."""
    return [
        "Where is my order? It's been 5 days and hasn't arrived.",
        "I got a refund but it was the wrong amount, charged twice.",
        "The item arrived broken and the box was crushed.",
        "How do I return this product?",
        "I can't log into my account, I forgot the password.",
        "Your support is pathetic, worst experience ever.",
        "The delivery driver left the package in the rain.",
        "The app keeps crashing when I try to checkout.",
        "This review was removed unfairly from the listing.",
        "Thanks for the quick response!",
    ]


@pytest.fixture
def tiny_expected_intents():
    """Expected keyword-rule outputs for tiny_texts."""
    return [
        "ORDER_STATUS",
        "REFUND",
        "DAMAGED_OR_WRONG",
        "RETURN_CANCEL",
        "ACCOUNT",
        "SERVICE_COMPLAINT",
        "DELIVERY_ADDRESS",
        "APP_WEBSITE",
        "REVIEW_CONTENT",
        "OTHER",
    ]
