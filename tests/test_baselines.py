"""Tests for deterministic baselines (keyword classifier)."""
import numpy as np
from agent import KeywordClassifier


def test_keyword_classifier_known_texts(tiny_texts, tiny_expected_intents):
    clf = KeywordClassifier()
    preds = clf.predict(tiny_texts)
    assert len(preds) == len(tiny_texts)
    for text, pred, expected in zip(tiny_texts, preds, tiny_expected_intents):
        assert pred == expected, f"Text: {text[:50]}... Expected {expected}, got {pred}"


def test_keyword_classifier_returns_numpy_array():
    clf = KeywordClassifier()
    preds = clf.predict(["where is my order"])
    assert isinstance(preds, np.ndarray)


def test_keyword_classifier_predict_proba_shape():
    clf = KeywordClassifier()
    texts = ["where is my order", "refund please", "hello"]
    P = clf.predict_proba(texts)
    assert P.shape == (3, 11)  # 11 intents
    assert (P.sum(axis=1) == 1.0).all()


def test_keyword_other_fallback():
    clf = KeywordClassifier()
    preds = clf.predict(["xyzzy foobar quux"])
    assert preds[0] == "OTHER"


def test_keyword_case_insensitive():
    clf = KeywordClassifier()
    p1 = clf.predict(["WHERE IS MY ORDER"])
    p2 = clf.predict(["where is my order"])
    assert p1[0] == p2[0] == "ORDER_STATUS"
