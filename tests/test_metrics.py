"""Tests for metric functions on tiny known examples."""
from evaluate import intent_report, route_metrics


def test_intent_report_perfect():
    gold = ["A", "B", "C", "A"]
    pred = ["A", "B", "C", "A"]
    r = intent_report(gold, pred, "test")
    assert r["accuracy"] == 1.0
    assert r["macro_f1"] == 1.0
    assert r["system"] == "test"


def test_intent_report_all_wrong():
    gold = ["A", "B", "C"]
    pred = ["B", "C", "A"]
    r = intent_report(gold, pred, "bad")
    assert r["accuracy"] == 0.0
    assert r["macro_f1"] == 0.0


def test_intent_report_partial():
    gold = ["A", "A", "B", "B"]
    pred = ["A", "B", "B", "A"]
    r = intent_report(gold, pred, "half")
    assert r["accuracy"] == 0.5


def test_route_metrics_perfect():
    gold = ["human", "auto", "human", "auto"]
    pred = ["human", "auto", "human", "auto"]
    m = route_metrics(gold, pred)
    assert m["route_accuracy"] == 1.0
    assert m["missed_escalations"] == 0
    assert m["false_alarms"] == 0


def test_route_metrics_all_missed():
    gold = ["human", "human", "human"]
    pred = ["auto", "auto", "auto"]
    m = route_metrics(gold, pred)
    assert m["missed_escalations"] == 3
    assert m["false_alarms"] == 0
    assert m["human_recall"] == 0.0


def test_route_metrics_all_escalated():
    gold = ["auto", "auto", "auto"]
    pred = ["human", "human", "human"]
    m = route_metrics(gold, pred)
    assert m["false_alarms"] == 3
    assert m["missed_escalations"] == 0
    assert m["escalation_rate"] == 1.0


def test_route_metrics_mixed():
    gold = ["human", "human", "auto", "auto", "human"]
    pred = ["human", "auto", "human", "auto", "human"]
    m = route_metrics(gold, pred)
    assert m["missed_escalations"] == 1  # gold=human, pred=auto
    assert m["false_alarms"] == 1        # gold=auto, pred=human
    assert m["route_accuracy"] == 0.6
