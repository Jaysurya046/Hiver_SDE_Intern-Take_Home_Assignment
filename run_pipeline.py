"""Staged evaluation pipeline with caching.

Stages:
  0  prep       - load corpus, build train pool, train ML intent model, build retriever
  1  intent     - classify the 200 golden messages: majority / keyword / ML / LLM
  2  route      - escalation policy applied per system; route metrics
  3  drafts     - replies for the 50-message judge sample: canned / retrieval / llm_rag
  4  judge      - LLM-as-judge scores the 3x50 replies on the rubric
  5  agreement  - judge vs hand ratings (Cohen's kappa, Spearman)
  6  summarize  - headline tables + failure analysis; writes results/summary.json

Usage: python run_pipeline.py [--fresh] [--sample K]
Cached LLM outputs live in results/; --fresh regenerates them.
"""
import argparse
import json
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, "src")
from agent import (KeywordClassifier, MLIntentClassifier, LLMIntentClassifier, Retriever,
                   CANNED_REPLY, KEYWORD_RULES, clean_text, looks_english,
                   escalation_reasons, retrieval_reply, llm_reply)
from evaluate import (LLMJudge, intent_report, route_metrics, judge_human_agreement,
                      RUBRIC_COLS)

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


def cache_path(name):
    return os.path.join(RESULTS, name)


def prep():
    """Load corpus, build the (message -> reply) pool excluding golden convs, train models."""
    golden = pd.read_csv("golden_set.csv")
    threads = pd.read_csv("data/amazonhelp_threads.csv")
    firsts = (threads.sort_values("depth").groupby("conv_id").first()
              .query("inbound == True").reset_index())
    replies = threads.query("inbound == False and depth == 1")[["conv_id", "text"]]
    pairs = firsts[["conv_id", "text"]].merge(replies, on="conv_id", suffixes=("_msg", "_reply"))
    pairs.columns = ["conv_id", "msg", "reply"]
    pool = pairs[~pairs.conv_id.isin(set(golden.conv_id))].copy()
    pool = pool[pool.msg.map(looks_english)].reset_index(drop=True)
    pool["weak"] = pool.msg.map(lambda t: next(
        (i for i, p in KEYWORD_RULES.items() if re.search(p, clean_text(t), re.I)), None))
    labeled = pool.dropna(subset=["weak"]).reset_index(drop=True)
    ml = MLIntentClassifier().fit(labeled.msg.map(clean_text).tolist(), labeled.weak)
    ret = Retriever(pool.msg.tolist(), pool.reply.tolist())
    return golden, pool, labeled, ml, ret


def stage_intent(golden, ml, ret, fresh=False):
    f = cache_path("intent_preds.csv")
    if os.path.exists(f) and not fresh:
        return pd.read_csv(f)
    texts = golden.text.tolist()
    maj = intent_report(golden.intent, ["ORDER_STATUS"] * len(golden), "majority")
    kw_pred = KeywordClassifier().predict(texts)
    ml_pred = ml.predict(texts)
    ml_conf = ml.predict_proba(texts).max(axis=1)
    t0 = time.time()
    llm = LLMIntentClassifier(batch_size=10)
    llm_pred, llm_conf = llm.predict(texts)
    print(f"  LLM intent on {len(texts)} msgs: {time.time()-t0:.0f}s")
    df = pd.DataFrame({
        "ex_id": golden.ex_id, "gold": golden.intent, "route_gold": golden.route,
        "kw": kw_pred, "ml": ml_pred, "ml_conf": ml_conf,
        "llm": llm_pred, "llm_conf": llm_conf,
    })
    df.to_csv(f, index=False)
    return df


def stage_route(preds, golden, ret, fresh=False):
    f = cache_path("route_preds.csv")
    if os.path.exists(f) and not fresh:
        return pd.read_csv(f)
    texts = golden.set_index("ex_id").text
    max_sims = {ex: ret.max_sim(texts[ex]) for ex in preds.ex_id}
    rows = []
    for sysname, icol, ccol in [("keyword", "kw", None), ("ml", "ml", "ml_conf"), ("llm", "llm", "llm_conf")]:
        for _, r in preds.iterrows():
            conf = float(r[ccol]) if ccol else 1.0
            reasons = escalation_reasons(texts[r.ex_id], r[icol], conf, max_sims[r.ex_id])
            rows.append({"ex_id": r.ex_id, "system": sysname, "intent": r[icol],
                         "route": "human" if reasons else "auto",
                         "reasons": "; ".join(reasons) if reasons else "standard flow"})
    df = pd.DataFrame(rows)
    df.to_csv(f, index=False)
    return df


def stage_drafts(golden, ret, n=50, seed=42, fresh=False):
    f = cache_path("drafts_50.csv")
    if os.path.exists(f) and not fresh:
        return pd.read_csv(f)
    sample = golden.sample(n=n, random_state=seed).reset_index(drop=True)
    drafts = {"canned": [], "retrieval": [], "llm_rag": []}
    t0 = time.time()
    for i, msg in enumerate(sample.text):
        drafts["canned"].append(CANNED_REPLY)
        rr, _ = retrieval_reply(msg, ret)
        drafts["retrieval"].append(rr)
        drafts["llm_rag"].append(llm_reply(msg, ret))
        if (i + 1) % 10 == 0:
            print(f"  drafted {i+1}/{n} ({time.time()-t0:.0f}s)")
    df = sample[["ex_id", "conv_id", "text", "brand_reply", "intent", "route"]].copy()
    for k, v in drafts.items():
        df[f"reply_{k}"] = v
    df.to_csv(f, index=False)
    return df


def stage_judge(drafts, fresh=False):
    f = cache_path("judge_scores.csv")
    if os.path.exists(f) and not fresh:
        return pd.read_csv(f)
    judge = LLMJudge(batch_size=5)
    msgs = drafts.text.tolist()
    parts = {}
    for system in ["canned", "retrieval", "llm_rag"]:
        t0 = time.time()
        parts[system] = judge.score(msgs, drafts[f"reply_{system}"].tolist())
        print(f"  judged {system}: {time.time()-t0:.0f}s")
    rows = []
    for system, sc in parts.items():
        d = sc.reset_index()  # keep the "case" column (0..49, aligned with drafts order)
        d.insert(0, "system", system)
        d.insert(0, "ex_id", drafts.ex_id.values)
        rows.append(d)
    df = pd.concat(rows).reset_index(drop=True)
    df.to_csv(f, index=False)
    return df


def stage_agreement(drafts, judge_scores, fresh=False):
    """Judge vs my hand ratings on the agent (llm_rag) replies."""
    hand_f = cache_path("hand_ratings.csv")
    if not os.path.exists(hand_f):
        raise SystemExit(f"Missing {hand_f}: hand-rate the llm_rag drafts first.")
    hand = pd.read_csv(hand_f).set_index("case")
    agent_scores = judge_scores[judge_scores.system == "llm_rag"].set_index("case")
    agree = judge_human_agreement(agent_scores[RUBRIC_COLS], hand[RUBRIC_COLS])
    agree.to_csv(cache_path("judge_human_agreement.csv"), index=False)
    return agree


def stage_summarize(preds, route_df, judge_scores, agree):
    gold = preds.gold
    reports = [
        intent_report(gold, ["ORDER_STATUS"] * len(preds), "majority (trivial)"),
        intent_report(gold, preds.kw, "keyword rules (trivial)"),
        intent_report(gold, preds.ml, "tfidf+logreg (simple)"),
        intent_report(gold, preds.llm, "LLM gemma4:e4b (agent)"),
    ]
    intent_tbl = pd.DataFrame(reports)

    routes = {}
    for system in ["keyword", "ml", "llm"]:
        sub = route_df[route_df.system == system]
        routes[system] = route_metrics(preds.route_gold, sub.route)

    js = judge_scores.groupby("system")[RUBRIC_COLS].mean()
    js["overall"] = judge_scores.groupby("system")[RUBRIC_COLS].mean().mean(axis=1)

    summary = {
        "intent": intent_tbl.round(4).to_dict("records"),
        "route": {k: {kk: vv for kk, vv in v.items() if kk != "confusion"}
                  for k, v in routes.items()},
        "judge_means": js.round(3).reset_index().to_dict("records"),
        "judge_human_agreement": agree.round(3).to_dict("records"),
    }
    with open(cache_path("summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="regenerate cached LLM outputs")
    ap.add_argument("--sample", type=int, default=50, help="judge sample size")
    args = ap.parse_args()

    t0 = time.time()
    print("[0] prep (corpus, ML training, retriever)")
    golden, pool, labeled, ml, ret = prep()
    print(f"    pool={len(pool)} labeled={len(labeled)} ({time.time()-t0:.0f}s)")

    print("[1] intent evaluation on golden set")
    preds = stage_intent(golden, ml, ret, fresh=args.fresh)
    print(preds.filter(["gold", "kw", "ml", "llm"]).head(3).to_string())

    print("[2] route evaluation")
    route_df = stage_route(preds, golden, ret, fresh=args.fresh)

    print("[3] reply drafting for judge sample")
    drafts = stage_drafts(golden, ret, n=args.sample, fresh=args.fresh)

    print("[4] LLM-as-judge")
    judge_scores = stage_judge(drafts, fresh=args.fresh)

    print("[5] judge-human agreement")
    try:
        agree = stage_agreement(drafts, judge_scores)
        print(agree.to_string())
    except SystemExit as e:
        print(f"  {e}")
        agree = pd.DataFrame()

    print("[6] summary")
    summary = stage_summarize(preds, route_df, judge_scores, agree)
    print(json.dumps(summary["intent"], indent=2))
    print(f"\nTOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
