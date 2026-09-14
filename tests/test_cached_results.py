"""Tests for cached result file integrity.

Validates that committed result files exist, have expected schemas,
and that summary.json metrics can be recomputed from the CSVs.
"""
import json
import os

import pandas as pd
import pytest

from evaluate import route_metrics


def test_result_files_exist(repo_root):
    expected = [
        "results/intent_preds.csv",
        "results/route_preds.csv",
        "results/drafts_50.csv",
        "results/judge_scores.csv",
        "results/hand_ratings.csv",
        "results/judge_human_agreement.csv",
        "results/summary.json",
    ]
    for f in expected:
        path = os.path.join(repo_root, f)
        assert os.path.exists(path), f"Missing: {f}"


def test_intent_preds_schema(intent_preds):
    expected = {"ex_id", "gold", "route_gold", "kw", "ml", "ml_conf", "llm", "llm_conf"}
    assert expected == set(intent_preds.columns)
    assert len(intent_preds) == 200


def test_route_preds_schema(route_preds):
    expected = {"ex_id", "system", "intent", "route", "reasons"}
    assert expected == set(route_preds.columns)
    assert len(route_preds) == 600
    assert set(route_preds.system.unique()) == {"keyword", "ml", "llm"}


def test_drafts_schema(drafts_df):
    expected = {"ex_id", "conv_id", "text", "brand_reply", "intent", "route",
                "reply_canned", "reply_retrieval", "reply_llm_rag"}
    assert expected == set(drafts_df.columns)
    assert len(drafts_df) == 50


def test_judge_scores_schema(judge_scores):
    expected = {"ex_id", "case", "system", "groundedness", "helpfulness", "safety_tone", "brand_voice"}
    assert expected == set(judge_scores.columns)
    assert len(judge_scores) == 150
    assert set(judge_scores.system.unique()) == {"canned", "retrieval", "llm_rag"}


def test_summary_intent_accuracy_recomputable(intent_preds, summary):
    """Recompute headline intent metrics from the committed CSV and compare to summary.json."""
    gold = intent_preds.gold
    # LLM accuracy
    llm_acc = float((gold == intent_preds.llm).mean())
    summary_llm = next(s for s in summary["intent"] if "LLM" in s["system"])
    assert abs(llm_acc - summary_llm["accuracy"]) < 0.001, \
        f"Recomputed LLM acc={llm_acc}, summary says {summary_llm['accuracy']}"


def test_summary_route_metrics_recomputable(intent_preds, route_preds, summary):
    """Recompute LLM route metrics and compare to summary."""
    llm_route = route_preds[route_preds.system == "llm"]
    m = route_metrics(intent_preds.route_gold, llm_route.route)
    s = summary["route"]["llm"]
    assert abs(m["route_accuracy"] - s["route_accuracy"]) < 0.001
    assert m["missed_escalations"] == s["missed_escalations"]
    assert m["false_alarms"] == s["false_alarms"]


def test_report_files_exist(repo_root):
    for f in ["report/report.md", "report/failures.md",
              "report/golden_protocol.md", "report/brand_selection.md"]:
        assert os.path.exists(os.path.join(repo_root, f)), f"Missing: {f}"
