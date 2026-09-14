"""Evaluation harness: metrics, LLM-as-judge, and judge-human agreement."""
import json
import re
import time

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from agent import OLLAMA_URL, JUDGE_MODEL, clean_text

RUBRIC = {
    "groundedness": "Does the reply address the customer's actual issue? "
                    "5 = directly addresses the specific problem; 3 = relevant but generic; 1 = ignores the issue.",
    "helpfulness": "Does the reply give a clear path forward? "
                   "5 = specific actionable next step; 3 = polite but vague; 1 = dead end.",
    "safety_tone": "Is the reply safe and empathetic? "
                   "5 = warm, professional, makes no unsafe promises; 3 = neutral; "
                   "1 = promises money/action publicly, rude, or dismissive.",
    "brand_voice": "Does the reply match AmazonHelp's Twitter style? "
                   "5 = brief, warm, ends with agent initials like ^XX; "
                   "3 = close but wordy or missing the signature; 1 = wrong tone entirely.",
}
RUBRIC_COLS = list(RUBRIC)


def majority_baseline(y):
    """Trivial baseline: always predict the most frequent training-pool label."""
    return pd.Series(y).mode().iat[0]


def intent_report(gold, pred, name):
    """Accuracy + macro-F1 for one system's predictions."""
    return {
        "system": name,
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, average="macro", labels=np.unique(gold))),
    }


def route_metrics(gold_route, pred_route):
    """Route decision quality. 'missed_escalation' = true human, predicted auto (worst error)."""
    gold = np.array(gold_route) == "human"
    pred = np.array(pred_route) == "human"
    tp = int((gold & pred).sum())
    fp = int((~gold & pred).sum())
    fn = int((gold & ~pred).sum())
    tn = int((~gold & ~pred).sum())
    return {
        "route_accuracy": float((gold == pred).mean()),
        "escalation_rate": float(pred.mean()),
        "human_recall": tp / (tp + fn) if tp + fn else float("nan"),
        "auto_precision": tn / (tn + fn) if tn + fn else float("nan"),
        "missed_escalations": fn,   # true human routed to auto
        "false_alarms": fp,        # true auto sent to human
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


class LLMJudge:
    """Judge reply quality with a local LLM, batched, on the 4-dimension rubric."""

    def __init__(self, model=JUDGE_MODEL, url=OLLAMA_URL, batch_size=5):
        self.model, self.url, self.batch_size = model, url, batch_size
        self.sys = (
            "You are a strict evaluator of customer-support replies written by an AI agent "
            "for the AmazonHelp Twitter account. Score each reply on four dimensions, each 1-5:\n"
            + "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
            + '\n\nRespond with ONLY a JSON array, one object per case: '
              '[{"id": 1, "groundedness": n, "helpfulness": n, "safety_tone": n, "brand_voice": n}]. No other text.'
        )

    def _chat(self, user, num_predict=500):
        payload = {
            "model": self.model, "stream": False, "think": False,
            "messages": [{"role": "system", "content": self.sys},
                         {"role": "user", "content": user}],
            "options": {"temperature": 0.0, "num_predict": num_predict, "num_ctx": 8192},
        }
        r = requests.post(self.url, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def score(self, messages, replies):
        """Score parallel lists; returns DataFrame indexed like inputs."""
        rows = []
        for s in range(0, len(messages), self.batch_size):
            batch_msg, batch_rep = messages[s:s + self.batch_size], replies[s:s + self.batch_size]
            cases = "\n\n".join(
                f"Case {j+1}:\nCustomer: {clean_text(m)}\nReply: {clean_text(r)}"
                for j, (m, r) in enumerate(zip(batch_msg, batch_rep))
            )
            for attempt in range(3):
                try:
                    raw = self._chat(cases)
                    arr = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
                    assert len(arr) == len(batch_msg)
                    break
                except Exception:
                    if attempt == 2:
                        arr = [{c: 3 for c in RUBRIC_COLS} for _ in batch_msg]
                    time.sleep(1)
            for j, item in enumerate(arr):
                item = {c: item.get(c, 3) for c in RUBRIC_COLS}
                rows.append({"case": s + j, **{c: float(item[c]) for c in RUBRIC_COLS}})
        return pd.DataFrame(rows).set_index("case")


def judge_human_agreement(judge_df, human_df):
    """Agreement between judge scores and my hand ratings, per rubric dimension."""
    out = []
    for c in RUBRIC_COLS:
        j, h = judge_df[c].values, human_df[c].values
        out.append({
            "dimension": c,
            "spearman_rho": float(pd.Series(j).corr(pd.Series(h), method="spearman")),
            "cohen_kappa_linear": float(cohen_kappa_score(j, h, weights="linear")),
            "mean_abs_diff": float(np.abs(j - h).mean()),
            "exact_match_rate": float((j == h).mean()),
        })
    return pd.DataFrame(out)
