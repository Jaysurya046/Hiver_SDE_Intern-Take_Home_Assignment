"""Generate report/failures.md — real examples for every failure mode.

Run after run_pipeline.py has produced results/*.csv.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, "src")
from agent import clean_text

OUT = []


def emit(s=""):
    OUT.append(s)


def main():
    golden = pd.read_csv("golden_set.csv").set_index("ex_id")
    preds = pd.read_csv("results/intent_preds.csv")
    route = pd.read_csv("results/route_preds.csv")
    drafts = pd.read_csv("results/drafts_50.csv")
    judge = pd.read_csv("results/judge_scores.csv")

    emit("# Failure analysis: raw evidence\n")

    # ---- 1. intent confusions with texts
    wrong = preds[preds.llm != preds.gold].copy()
    wrong["text"] = wrong.ex_id.map(golden.text)
    emit("## Intent confusions (LLM), top pairs with examples\n")
    for (g, p), n in wrong.groupby(["gold", "llm"]).size().sort_values(ascending=False).head(8).items():
        emit(f"### {n}x gold={g} -> pred={p}\n")
        for _, r in wrong[(wrong.gold == g) & (wrong.llm == p)].head(2).iterrows():
            emit(f"- ex {r.ex_id} (conf {r.llm_conf}): {clean_text(r.text)[:220]}")
        emit()

    # ---- 2. missed escalations (worst route error)
    llm_route = route[route.system == "llm"].set_index("ex_id")
    merged = preds.join(llm_route, rsuffix="_r")
    missed = merged[(merged.route_gold == "human") & (merged.route == "auto")].copy()
    missed["text"] = missed.ex_id.map(golden.text)
    emit(f"## Missed escalations (gold=human, agent=auto): {len(missed)}\n")
    for _, r in missed.head(12).iterrows():
        emit(f"- ex {r.ex_id} [{r.gold}, conf {r.llm_conf:.2f}]: {clean_text(r.text)[:200]}")
        emit(f"  - reasons: {r.reasons}")
    emit()

    # ---- 3. false alarms
    fa = merged[(merged.route_gold == "auto") & (merged.route == "human")].copy()
    fa["text"] = fa.ex_id.map(golden.text)
    emit(f"## False alarms (gold=auto, agent=human): {len(fa)}\n")
    for _, r in fa.head(8).iterrows():
        emit(f"- ex {r.ex_id} [{r.gold}]: {clean_text(r.text)[:180]}")
        emit(f"  - reasons: {r.reasons}")
    emit()

    # ---- 4. worst agent drafts by judge score
    agent = judge[judge.system == "llm_rag"].copy()
    agent["overall"] = agent[["groundedness", "helpfulness", "safety_tone", "brand_voice"]].mean(axis=1)
    dmap = drafts.set_index("ex_id")
    emit("## Worst 8 agent drafts by judge score\n")
    for _, r in agent.nsmallest(8, "overall").iterrows():
        d = dmap.loc[r.ex_id]
        emit(f"- ex {int(r.ex_id)} (score {r.overall:.1f}): {clean_text(d.text)[:160]}")
        emit(f"  - draft: {clean_text(d.reply_llm_rag)[:220]}")
    emit()

    # ---- 5. best agent drafts (for contrast)
    emit("## Best 3 agent drafts by judge score\n")
    for _, r in agent.nlargest(3, "overall").iterrows():
        d = dmap.loc[r.ex_id]
        emit(f"- ex {int(r.ex_id)} (score {r.overall:.1f}): {clean_text(d.text)[:140]}")
        emit(f"  - draft: {clean_text(d.reply_llm_rag)[:220]}")
    emit()

    # ---- 6. safety heuristics on all agent drafts
    import re
    bad = {"asks for order number publicly": r"order ?number|order ?id|tracking ?number",
           "asks for personal data publicly": r"email|phone number|password|card number|address\?",
           "promises money": r"refund (you|your) \$|we will refund|we'll refund"}
    emit("## Safety scan of agent drafts (regex heuristics)\n")
    flagged = 0
    for name, pat in bad.items():
        hits = [i for i, t in enumerate(drafts.reply_llm_rag) if re.search(pat, str(t), re.I)]
        if hits:
            flagged += len(hits)
            emit(f"- {name}: {len(hits)} hit(s), ex_ids {drafts.ex_id[hits].tolist()}")
    if not flagged:
        emit("- none flagged")
    emit()

    os.makedirs("report", exist_ok=True)
    with open("report/failures.md", "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))
    print(f"report/failures.md written ({len(OUT)} lines)")


if __name__ == "__main__":
    main()
